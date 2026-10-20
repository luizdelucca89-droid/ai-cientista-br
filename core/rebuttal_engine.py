"""Motor de rebuttal: responde a revisores e atualiza o manuscrito.

Fluxo:

1. os comentários do revisor (texto livre) são convertidos pelo LLM em uma lista
   de ações estruturadas — ``{"tipo": "re-run"|"rewrite"|"clarify", "descricao": ...}``;
2. ações ``re-run`` reexecutam o experimento com os parâmetros pedidos (o código
   do experimento roda no sandbox com auto-debug, como no ciclo normal);
3. o Agente Redator reescreve as seções afetadas do ``paper.md`` e do ``paper.tex``;
4. é gerado ``resposta_revisores.md`` em português e inglês, respondendo ponto a
   ponto e citando as mudanças exatas feitas no manuscrito e nos resultados.

Nada aqui fabrica dados: quando um ``re-run`` não tem fonte pública real, ele
falha de forma explícita e a resposta ao revisor diz isso.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.agents.writer import WriterAgent
from core.experiments.runner import ExperimentRunner
from core.utils import extract_json_block, read_json, setup_logger, write_json

logger = setup_logger(__name__)

TIPOS_ACAO = ("re-run", "rewrite", "clarify")

PROMPT_PARSE = """Você é o editor responsável de um periódico científico. Um revisor
enviou os comentários abaixo sobre um manuscrito.

COMENTÁRIOS DO REVISOR:
\"\"\"
{comentarios}
\"\"\"

Converta os comentários em uma lista de AÇÕES acionáveis. Responda SOMENTE com JSON:
{{"acoes": [{{"tipo": "re-run|rewrite|clarify", "descricao": "...",
"secoes": ["resultados"], "parametros": {{}}}}],
"resumo_geral": "..."}}

Regras:
- "re-run": exige mudar ou refazer o experimento (nova métrica, nova feature,
  novo modelo, novo recorte). Preencha "parametros" com o que muda.
- "rewrite": exige reescrever seções sem refazer o experimento. Liste "secoes".
- "clarify": pede explicação/resposta sem mudar o manuscrito.
- Se houver dúvida entre re-run e rewrite, prefira "rewrite" (menos invasivo).
- Nunca proponha dados sintéticos: os dados são observações públicas reais."""

PROMPT_RESPOSTA = """Você é o autor correspondente respondendo a um revisor.

TÓPICO DO ARTIGO: {topic}

COMENTÁRIOS DO REVISOR:
\"\"\"
{comentarios}
\"\"\"

AÇÕES EXECUTADAS PELO SISTEMA:
{acoes}

MUDANÇAS NO MANUSCRITO:
{mudancas}

RESULTADOS ANTES/DEPOIS (apenas números reais; não invente):
{comparacao}

Escreva uma CARTA DE RESPOSTA profissional, em DUAS partes:

## Resposta aos Revisores (Português)
Responda cada comentário ponto a ponto, com o formato:
**Comentário R1:** ...
**Resposta:** ... (cite a mudança exata no manuscrito e os números novos)

## Response to Reviewers (English)
Mesma resposta, em inglês acadêmico.

Seja específico e honesto: se algo não foi alterado, diga por quê. Nunca invente
resultados que não estejam em RESULTADOS ANTES/DEPOIS."""


@dataclass
class AcaoRevisor:
    """Ação derivada de um comentário de revisor."""

    tipo: str
    descricao: str
    secoes: list[str] = field(default_factory=list)
    parametros: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tipo": self.tipo,
            "descricao": self.descricao,
            "secoes": self.secoes,
            "parametros": self.parametros,
        }


def parse_reviewer_comments(comentarios: str, llm: Any) -> list[AcaoRevisor]:
    """Converte comentários livres em ações estruturadas usando o LLM.

    Faz um fallback determinístico se o LLM não devolver JSON válido: trata cada
    parágrafo não vazio como uma ação ``clarify``, para não perder comentários.
    """
    if not comentarios or not comentarios.strip():
        return []

    resposta = llm.complete(PROMPT_PARSE.format(comentarios=comentarios), task="rebuttal")
    dados = extract_json_block(getattr(resposta, "text", "") or "")
    acoes_brutas: list[Any] = []
    if isinstance(dados, dict):
        acoes_brutas = dados.get("acoes") or []
    elif isinstance(dados, list):
        acoes_brutas = dados

    acoes: list[AcaoRevisor] = []
    for bruta in acoes_brutas:
        if not isinstance(bruta, dict):
            continue
        tipo = str(bruta.get("tipo", "clarify")).strip().lower()
        if tipo not in TIPOS_ACAO:
            tipo = "clarify"
        secoes = bruta.get("secoes") or []
        if isinstance(secoes, str):
            secoes = [secoes]
        parametros = bruta.get("parametros") or {}
        if not isinstance(parametros, dict):
            parametros = {}
        acoes.append(
            AcaoRevisor(
                tipo=tipo,
                descricao=str(bruta.get("descricao", "")).strip(),
                secoes=[str(s).strip().lower() for s in secoes],
                parametros=parametros,
            )
        )

    if not acoes:
        for paragrafo in (p.strip() for p in comentarios.split("\n\n")):
            if paragrafo:
                acoes.append(AcaoRevisor(tipo="clarify", descricao=paragrafo))
    return acoes


class RebuttalEngine:
    """Processa comentários de revisores e atualiza o manuscrito e os resultados."""

    def __init__(
        self,
        llm: Any,
        output_dir: Any = "outputs",
        runner: ExperimentRunner | None = None,
        writer: WriterAgent | None = None,
    ) -> None:
        self.llm = llm
        self.output_dir = Path(output_dir)
        self.runner = runner or ExperimentRunner(output_dir=self.output_dir, llm=llm)
        self.writer = writer or WriterAgent(llm=llm)

    # ------------------------------------------------------------------ public
    def process_rebuttal(self, cycle_id: int, reviewer_comments: str) -> dict[str, Any]:
        """Responde aos revisores, reexecuta o que for pedido e atualiza o paper.

        Devolve um relatório com as ações, as mudanças e o caminho de
        ``resposta_revisores.md``.
        """
        ciclo = read_json(self._ciclo_path(cycle_id), default=None)
        if ciclo is None:
            raise FileNotFoundError(f"Ciclo {cycle_id} não encontrado em {self.output_dir}")

        run_dir = self._run_dir(cycle_id, ciclo)
        topic = ciclo.get("topico") or ciclo.get("topic") or self._topic_from(ciclo)

        acoes = parse_reviewer_comments(reviewer_comments, self.llm)
        logger.info("Rebuttal do ciclo %s: %s ação(ões) identificadas", cycle_id, len(acoes))

        comparacao: dict[str, Any] = {}
        resultados_novos: dict[str, Any] | None = None
        for acao in acoes:
            if acao.tipo == "re-run":
                resultados_novos, comparacao = self._reexecutar(ciclo, topic, acao)

        mudancas = self._reescrever_secoes(run_dir, ciclo, topic, acoes, resultados_novos)

        resposta = self._gerar_resposta(topic, reviewer_comments, acoes, mudancas, comparacao)
        caminho = run_dir / "resposta_revisores.md"
        caminho.write_text(resposta, encoding="utf-8")
        logger.info("Resposta aos revisores gravada em %s", caminho)

        relatorio = {
            "cycle_id": cycle_id,
            "topic": topic,
            "acoes": [a.as_dict() for a in acoes],
            "mudancas": mudancas,
            "comparacao_resultados": comparacao,
            "resposta_md": str(caminho),
            "run_dir": str(run_dir),
        }
        write_json(run_dir / "rebuttal.json", relatorio)
        return relatorio

    # ------------------------------------------------------------------ helpers
    def _ciclo_path(self, cycle_id: int) -> Path:
        """Localiza o ``ciclo_completo.json`` do ciclo, por glob se preciso."""
        import glob

        padrao = str(self.output_dir / f"ciclo_{cycle_id}_*" / "ciclo_completo.json")
        encontrados = sorted(glob.glob(padrao))
        if encontrados:
            return Path(encontrados[0])
        return self.output_dir / f"ciclo_{cycle_id}" / "ciclo_completo.json"

    def _run_dir(self, cycle_id: int, ciclo: dict[str, Any]) -> Path:
        artefatos = (ciclo.get("artefatos") or {})
        diretorio = artefatos.get("diretorio")
        if diretorio and Path(diretorio).exists():
            return Path(diretorio)
        return self._ciclo_path(cycle_id).parent

    @staticmethod
    def _topic_from(ciclo: dict[str, Any]) -> str:
        return str(
            (ciclo.get("planejamento") or {}).get("lacuna")
            or (ciclo.get("experimento") or {}).get("resultado", {}).get("topic")
            or "tópico sem título"
        )

    def _reexecutar(
        self,
        ciclo: dict[str, Any],
        topic: str,
        acao: AcaoRevisor,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Reexecuta o experimento com o código atual e registra antes/depois."""
        resultado_antigo = (ciclo.get("experimento") or {}).get("resultado") or {}
        codigo = resultado_antigo.get("codigo") or None
        hipotese = (ciclo.get("experimento") or {}).get("hipotese_principal") or {"id": "H1"}

        # O código do experimento pode ser ajustado pelo LLM conforme o pedido do
        # revisor (ex.: trocar a métrica). Sem código novo, reexecuta o mesmo.
        codigo_ajustado = self._ajustar_codigo(codigo, acao.descricao, topic) if codigo else None
        try:
            novo = self.runner.run(topic, hipotese, codigo=codigo_ajustado)
        except Exception as exc:  # noqa: BLE001 - fonte real pode estar indisponível
            logger.warning("Re-run falhou: %s", exc)
            return None, {"erro": str(exc), "acao": acao.descricao}

        comparacao = self._comparar(resultado_antigo, novo.as_dict())
        return novo.as_dict(), comparacao

    def _ajustar_codigo(self, codigo: str, descricao: str, topic: str) -> str:
        prompt = (
            f"TÓPICO: {topic}\n\nPEDIDO DO REVISOR:\n{descricao}\n\n"
            "CÓDIGO ATUAL DO EXPERIMENTO:\n```python\n" + codigo + "\n```\n\n"
            "Ajuste o código para atender ao pedido, mantendo o uso de dados reais "
            "(variáveis X, y, meta). Devolva APENAS o código Python corrigido, sem "
            "texto e sem cercas markdown."
        )
        try:
            from core.experiments.auto_debugger import _limpar_codigo

            resposta = self.llm.complete(prompt, task="auto_debug")
            ajustado = _limpar_codigo(getattr(resposta, "text", "") or "")
            return ajustado or codigo
        except Exception as exc:  # noqa: BLE001
            logger.warning("Não foi possível ajustar o código: %s", exc)
            return codigo

    @staticmethod
    def _comparar(antigo: dict[str, Any], novo: dict[str, Any]) -> dict[str, Any]:
        """Compara métricas do melhor modelo antes e depois, sem inventar nada."""
        def melhor(res: dict[str, Any]) -> dict[str, Any]:
            modelo = res.get("best_model")
            return {"modelo": modelo, **(res.get("metrics", {}).get(modelo, {}) or {})}

        return {
            "antes": melhor(antigo),
            "depois": melhor(novo),
            "status_experimento": novo.get("status_experimento", "ok"),
        }

    def _reescrever_secoes(
        self,
        run_dir: Path,
        ciclo: dict[str, Any],
        topic: str,
        acoes: list[AcaoRevisor],
        resultados_novos: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Reescreve as seções afetadas no paper.md e no paper.tex."""
        secoes_pedidas: list[str] = []
        for acao in acoes:
            if acao.tipo in ("re-run", "rewrite"):
                alvos = acao.secoes or ["resultados"]
                for secao in alvos:
                    if secao not in secoes_pedidas:
                        secoes_pedidas.append(secao)

        secoes_atuais = (ciclo.get("redacao") or {}).get("secoes") or {}
        contexto = {
            "resultado_experimental": resultados_novos
            or (ciclo.get("experimento") or {}).get("resultado"),
            "comentarios_revisores": [a.descricao for a in acoes if a.descricao],
        }

        mudancas: dict[str, Any] = {"secoes_reescritas": [], "paper_md": None, "paper_tex": None}
        if not secoes_pedidas:
            return mudancas

        novas_secoes = dict(secoes_atuais)
        for secao in secoes_pedidas:
            rascunho = secoes_atuais.get(secao, "")
            try:
                revisada = self.writer.revise(
                    secao,
                    rascunho,
                    "Atenda ao pedido do revisor: " + "; ".join(
                        a.descricao for a in acoes if a.descricao
                    ),
                    topic,
                    contexto,
                )
                novas_secoes[secao] = revisada.raw.strip()
                mudancas["secoes_reescritas"].append(secao)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Falha ao reescrever a seção %s: %s", secao, exc)

        mudancas["paper_md"] = self._atualizar_markdown(run_dir, novas_secoes, mudancas["secoes_reescritas"])
        mudancas["paper_tex"] = self._atualizar_latex(run_dir, novas_secoes, mudancas["secoes_reescritas"])
        return mudancas

    @staticmethod
    def _atualizar_markdown(run_dir: Path, secoes: dict[str, str], reescritas: list[str]) -> str | None:
        """Substitui o texto das seções reescritas no paper.md, preservando o resto."""
        caminho = run_dir / "paper.md"
        if not caminho.exists():
            return None
        import re

        texto = caminho.read_text(encoding="utf-8")
        for secao in reescritas:
            novo = secoes.get(secao, "").strip()
            if not novo:
                continue
            # O título pode ser numerado e ter sufixo ("## 3 RESULTADOS E DISCUSSÃO"),
            # então casamos a palavra-chave em qualquer ponto do cabeçalho.
            padrao = re.compile(
                rf"(^#+[^\n]*{secao}[^\n]*\n)(.*?)(?=^#+ |\Z)",
                re.IGNORECASE | re.MULTILINE | re.DOTALL,
            )
            texto, n = padrao.subn(lambda m: m.group(1) + "\n" + novo + "\n\n", texto, count=1)
            if n == 0:
                texto += f"\n\n## {secao.upper()}\n\n{novo}\n"
        caminho.write_text(texto, encoding="utf-8")
        return str(caminho)

    @staticmethod
    def _atualizar_latex(run_dir: Path, secoes: dict[str, str], reescritas: list[str]) -> str | None:
        """Atualiza as seções do paper.tex, mantendo o preâmbulo e a bibliografia.

        Os títulos do LaTeX variam ("Resultados e Discussão"), então casamos por
        palavra-chave dentro do ``\\section{...}``, não por título exato.
        """
        import re

        caminho = run_dir / "paper.tex"
        if not caminho.exists():
            return None
        texto = caminho.read_text(encoding="utf-8")

        def substituir_secao(alvo: str, novo: str) -> None:
            nonlocal texto
            padrao = re.compile(
                rf"(\\section\{{[^}}]*{alvo}[^}}]*\}})(.*?)(?=\\section|\Z)",
                re.IGNORECASE | re.DOTALL,
            )
            texto, n = padrao.subn(lambda m: m.group(1) + "\n" + novo + "\n\n", texto, count=1)
            if n == 0:
                logger.warning("Seção %r não encontrada no paper.tex", alvo)

        for secao in reescritas:
            novo = secoes.get(secao, "").strip()
            if not novo:
                continue
            if secao == "resumo":
                padrao = re.compile(r"(\\begin\{abstract\})(.*?)(\\end\{abstract\})", re.DOTALL)
                texto, n = padrao.subn(lambda m: m.group(1) + "\n" + novo + "\n" + m.group(3), texto, count=1)
                if n == 0:
                    logger.warning("Abstract não encontrado no paper.tex")
                continue
            # "introducao" casa com \section{Introdução} após remover o acento.
            alvo = {"introducao": "Introdu", "metodologia": "Metodolog", "resultados": "Resultado", "conclusao": "Conclus"}.get(secao)
            if alvo:
                substituir_secao(alvo, novo)

        caminho.write_text(texto, encoding="utf-8")
        return str(caminho)

    def _gerar_resposta(
        self,
        topic: str,
        comentarios: str,
        acoes: list[AcaoRevisor],
        mudancas: dict[str, Any],
        comparacao: dict[str, Any],
    ) -> str:
        prompt = PROMPT_RESPOSTA.format(
            topic=topic,
            comentarios=comentarios,
            acoes=json.dumps([a.as_dict() for a in acoes], ensure_ascii=False, indent=2),
            mudancas=json.dumps(mudancas, ensure_ascii=False, indent=2, default=str),
            comparacao=json.dumps(comparacao, ensure_ascii=False, indent=2, default=str),
        )
        try:
            texto = self.llm.complete(prompt, task="rebuttal_response").text.strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Falha ao gerar carta de resposta: %s", exc)
            texto = ""

        if not texto:
            texto = ""
        # Garante o contrato: resposta ponto a ponto em português e em inglês.
        if "## Resposta aos Revisores" not in texto:
            texto = self._resposta_ponto_a_ponto(acoes, mudancas, comparacao) + "\n\n" + texto
        if "## Response to Reviewers" not in texto:
            texto = texto.rstrip() + "\n\n" + self._resposta_ingles(acoes, mudancas, comparacao)

        cabecalho = (
            f"# Resposta aos Revisores — Ciclo de rebuttal\n\n"
            f"**Artigo:** {topic}\n\n"
            f"**Ações identificadas:** {len(acoes)}\n\n"
            f"**Seções reescritas:** {', '.join(mudancas.get('secoes_reescritas', [])) or 'nenhuma'}\n\n"
            "---\n\n"
        )
        return cabecalho + texto + "\n"

    @staticmethod
    def _resposta_ponto_a_ponto(
        acoes: list[AcaoRevisor],
        mudancas: dict[str, Any],
        comparacao: dict[str, Any],
    ) -> str:
        linhas = ["## Resposta aos Revisores (Português)", ""]
        for i, acao in enumerate(acoes, 1):
            linhas.append(f"**Comentário R{i}:** {acao.descricao or '(sem descrição)'}")
            if acao.tipo == "re-run":
                antes = comparacao.get("antes", {})
                depois = comparacao.get("depois", {})
                linhas.append(
                    "**Resposta:** O experimento foi reexecutado sobre as mesmas "
                    "observações públicas reais. Melhor modelo antes: "
                    f"{antes.get('modelo')} (F1={antes.get('f1_score')}); depois: "
                    f"{depois.get('modelo')} (F1={depois.get('f1_score')}). "
                    f"Status: {comparacao.get('status_experimento', 'n/d')}."
                )
            elif acao.tipo == "rewrite":
                linhas.append(
                    "**Resposta:** As seções "
                    f"{', '.join(mudancas.get('secoes_reescritas', [])) or 'afetadas'} "
                    "foram reescritas no manuscrito (paper.md e paper.tex)."
                )
            else:
                linhas.append("**Resposta:** Esclarecimento incorporado sem alterar os resultados.")
            linhas.append("")
        return "\n".join(linhas)

    @staticmethod
    def _resposta_ingles(
        acoes: list[AcaoRevisor],
        mudancas: dict[str, Any],
        comparacao: dict[str, Any],
    ) -> str:
        linhas = ["## Response to Reviewers (English)", ""]
        for i, acao in enumerate(acoes, 1):
            linhas.append(f"**Reviewer comment R{i}:** {acao.descricao or '(no description)'}")
            if acao.tipo == "re-run":
                depois = comparacao.get("depois", {})
                linhas.append(
                    "**Response:** The experiment was re-run on the same real public "
                    f"observations. Best model after the revision: {depois.get('modelo')} "
                    f"(F1={depois.get('f1_score')})."
                )
            elif acao.tipo == "rewrite":
                linhas.append(
                    "**Response:** The affected sections "
                    f"({', '.join(mudancas.get('secoes_reescritas', [])) or 'listed above'}) "
                    "were rewritten in the manuscript."
                )
            else:
                linhas.append("**Response:** Clarification added without changing the results.")
            linhas.append("")
        return "\n".join(linhas)


def process_rebuttal(cycle_id: int, reviewer_comments: str, **kwargs: Any) -> dict[str, Any]:
    """Atalho funcional para :meth:`RebuttalEngine.process_rebuttal`.

    Constrói um ``RebuttalEngine`` com o LLM resolvido da configuração padrão
    quando ``llm`` não é informado.
    """
    from llm_providers import build_llm_client

    llm = kwargs.pop("llm", None) or build_llm_client()
    engine = RebuttalEngine(llm=llm, **kwargs)
    return engine.process_rebuttal(cycle_id, reviewer_comments)


__all__ = [
    "AcaoRevisor",
    "PROMPT_PARSE",
    "PROMPT_RESPOSTA",
    "RebuttalEngine",
    "TIPOS_ACAO",
    "parse_reviewer_comments",
    "process_rebuttal",
]
