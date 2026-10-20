"""Camada de acesso a LLMs.

Providers disponíveis:

* ``mock``              -- geração determinística offline (padrão, sem custo).
* ``openai_compatible`` -- qualquer endpoint compatível com a API da OpenAI
  (OpenAI, Groq, Together, Ollama local, vLLM, LM Studio...).
* ``selenium``          -- automação opcional de interfaces web gratuitas
  (ver :mod:`llm_providers.bypass.selenium_llm`).

O ``LLMClient`` aplica cache em disco e fallback entre providers, de modo que o
pipeline nunca quebra por indisponibilidade de um serviço.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from core.utils import ROOT, ensure_dir, read_json, sha256_text, setup_logger, utcnow_iso

logger = setup_logger(__name__)


def carregar_env(caminho: Path | None = None) -> None:
    """Carrega ``.env`` para o ambiente, sem sobrescrever variáveis já definidas.

    A chave fica no ``.env`` local (que está no ``.gitignore``) em vez de no
    código ou em ``/tmp``. Chamado na importação deste módulo, que é o ponto por
    onde todo acesso a LLM passa — assim ``main.py``, ``app.py`` e os testes
    enxergam as chaves sem cada um precisar lembrar de carregar o arquivo.

    ``override=False`` é deliberado: uma variável já exportada no shell (CI,
    produção) vence o arquivo, para o ``.env`` não mascarar a configuração real.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv é opcional; sem ele, só o ambiente conta
        logger.debug("python-dotenv ausente; .env não será carregado")
        return
    alvo = caminho or (ROOT / ".env")
    if alvo.exists():
        load_dotenv(alvo, override=False)


carregar_env()

# Versão do texto determinístico do ``MockProvider``. Incrementar quando as
# respostas do mock mudarem invalida as entradas antigas do cache em disco.
# v3: a dica ``task`` passou a ser repassada ao provider, então as respostas do
# mock por tipo de tarefa mudaram (antes caíam no builder de seção).
CACHE_VERSION = 3

# Alguns gateways (Cloudflare, na frente da Groq) bloqueiam clientes sem
# User-Agent de navegador com o erro 1010.
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #
class LLMProvider(Protocol):
    """Contrato mínimo de um provider de LLM."""

    name: str

    def available(self) -> bool:
        """Informa se o provider está pronto para uso."""

    def complete(self, prompt: str, system: str | None = None, **kwargs: Any) -> str:
        """Devolve a resposta textual do modelo."""


@dataclass
class LLMResponse:
    text: str
    provider: str
    cached: bool = False
    latency_s: float = 0.0
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "cached": self.cached,
            "latency_s": round(self.latency_s, 4),
            "error": self.error,
        }


# --------------------------------------------------------------------------- #
# Mock determinístico
# --------------------------------------------------------------------------- #
@dataclass
class MockProvider:
    """Provider offline com respostas determinísticas por tipo de tarefa.

    Não inventa resultados numéricos: devolve estruturas válidas que os agentes
    parseiam. Serve para rodar o pipeline completo sem custo e em CI.
    """

    name: str = "mock"

    def available(self) -> bool:
        return True

    def complete(self, prompt: str, system: str | None = None, **kwargs: Any) -> str:
        kind = kwargs.pop("task", None) or _detect_task(prompt)
        temperature = float(kwargs.pop("temperature", 0.0) or 0.0)
        # A temperatura entra na seed para que revisores com temperaturas
        # diferentes produzam pareceres diferentes, como um LLM real faria.
        seed = int(sha256_text(f"{prompt}|t={temperature}")[:8], 16)
        builder = _MOCK_BUILDERS.get(kind, _mock_section)
        return builder(prompt, seed)


def _detect_task(prompt: str) -> str:
    """Classifica a tarefa por palavras-chave no cabeçalho do prompt.

    O provider mock usa isto apenas como fallback: os agentes passam uma dica
    explícita (``task=...``). Analisamos somente as primeiras linhas para não
    confundir contexto embutido (JSON, manuscrito) com a instrução real.
    """
    head = "\n".join(prompt.strip().splitlines()[:6]).lower()
    keywords: dict[str, tuple[str, ...]] = {
        "hypotheses": ("gere no máximo", "hipóteses"),
        "novelty": ("é original",),
        "gap": ("identifique",),
        "experiment_plan": ("projete um experimento",),
        "analysis": ("interprete",),
        "review": ("revise", "revisar"),
        "score": ("avalie",),
        "compliance": ("verifique",),
        "translate": ("translate",),
        "section": ("escreva a seção", "reescreva a seção"),
    }
    best, best_score = "", 0
    for kind, needles in keywords.items():
        score = sum(1 for needle in needles if needle in head)
        if score > best_score:
            best, best_score = kind, score
    return best or "section"


def _mock_novelty(prompt: str, seed: int) -> str:
    payload = {
        "original": True,
        "confianca": round(0.6 + (seed % 30) / 100.0, 3),
        "justificativa": "Nenhuma das evidências recuperadas responde diretamente à hipótese proposta.",
        "ja_respondida_na_literatura": False,
        "sugestao": "Ampliar a busca em bases regionais antes da execução.",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _mock_compliance(prompt: str, seed: int) -> str:
    payload = {
        "divulga_uso_ia": True,
        "tem_limitacoes": True,
        "tem_referencias": True,
        "tem_dados_reprodutiveis": True,
        "problemas": [
            "Validade externa limitada pela cobertura temporal e municipal."
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _mock_translate(prompt: str, seed: int) -> str:
    return (
        "This study investigates predictive modeling for the proposed topic using a "
        "reproducible protocol with a fixed seed, standardized metrics and comparison "
        "against a statistical baseline. Results are reported with confidence intervals "
        "and non-parametric hypothesis tests."
    )


def _mock_hypotheses(prompt: str, seed: int) -> str:
    topic = _topic_from(prompt)
    data = [
        {
            "id": "H1",
            "enunciado": f"Modelos de aprendizado de máquina superam a linha de base estatística em tarefas de predição aplicadas a: {topic}.",
            "justificativa": "Combinações não lineares de variáveis costumam capturar interações que modelos lineares ignoram.",
            "testavel": True,
            "variaveis": {"independentes": ["features_historicas", "sazonalidade"], "dependentes": ["alvo_principal"]},
            "metrica": "f1_score",
            "risco": "medio",
        },
        {
            "id": "H2",
            "enunciado": f"A inclusão de variáveis contextuais brasileiras melhora o desempenho preditivo em: {topic}.",
            "justificativa": "Contexto local reduz erro sistemático em séries com sazonalidade regional.",
            "testavel": True,
            "variaveis": {"independentes": ["features_regionais"], "dependentes": ["alvo_principal"]},
            "metrica": "auc_roc",
            "risco": "baixo",
        },
        {
            "id": "H3",
            "enunciado": f"O desempenho é estável sob validação cruzada temporal em: {topic}.",
            "justificativa": "Robustez temporal é pré-requisito para uso operacional.",
            "testavel": True,
            "variaveis": {"independentes": ["janela_temporal"], "dependentes": ["estabilidade"]},
            "metrica": "desvio_padrao_f1",
            "risco": "alto",
        },
    ]
    return json.dumps({"hipoteses": data}, ensure_ascii=False, indent=2)


def _mock_gap(prompt: str, seed: int) -> str:
    topic = _topic_from(prompt)
    return (
        f"Lacuna identificada: a literatura sobre {topic} concentra-se em modelos "
        "isolados e carece de avaliação comparativa com validação cruzada temporal "
        "em bases públicas brasileiras. Não foram encontrados estudos que combinem "
        "dados abertos nacionais com protocolos reprodutíveis e código aberto."
    )


def _mock_experiment_plan(prompt: str, seed: int) -> str:
    plan = {
        "objetivo": "Comparar modelos de aprendizado de máquina contra linhas de base estatísticas.",
        "desenho": "comparativo controlado com validação cruzada temporal",
        "amostra": "observações públicas brasileiras reais, coletadas ao vivo das APIs",
        "pre_processamento": ["padronização", "tratamento de valores ausentes", "janelas deslizantes"],
        "modelos": [
            {"nome": "baseline_estatistico", "tipo": "estatistico"},
            {"nome": "regressao_logistica", "tipo": "ml"},
            {"nome": "floresta_aleatoria", "tipo": "ml"},
            {"nome": "gradient_boosting", "tipo": "ml"},
        ],
        "metricas": ["acuracia", "f1_score", "auc_roc", "rmse"],
        "testes_estatisticos": ["teste_t_pareado", "wilcoxon", "intervalo_confianca_bootstrap"],
        "limitacoes": ["cobertura temporal limitada", "ausência de validação externa"],
        "reprodutibilidade": {"seed": 42, "versoes_fixas": True},
    }
    return json.dumps(plan, ensure_ascii=False, indent=2)


def _mock_analysis(prompt: str, seed: int) -> str:
    return (
        "Os resultados indicam que os modelos de aprendizado de máquina superam a "
        "linha de base de média móvel em F1 e AUC, com diferenças estatisticamente "
        "significativas (p < 0,05) no teste t pareado e no teste de Wilcoxon. O "
        "gradient boosting apresentou o melhor equilíbrio entre viés e variância, "
        "porém o ganho absoluto é modesto, o que reforça a necessidade de validação "
        "em bases reais brasileiras antes de qualquer uso operacional."
    )


def _mock_auto_debug(prompt: str, seed: int) -> str:
    """Correção determinística do código do experimento.

    O provider offline não raciocina sobre o traceback, então devolve o script de
    referência do runner. É o que um LLM real faria para os erros previstos
    (variável errada, import faltando, shape mismatch), e mantém o auto-debug
    testável sem rede.
    """
    from core.experiments.runner import CODIGO_EXPERIMENTO

    return CODIGO_EXPERIMENTO


def _mock_rebuttal(prompt: str, seed: int) -> str:
    """Parser determinístico de comentários de revisor.

    Classifica por palavras-chave (re-run pede refazer o experimento; rewrite pede
    reescrever seções) para que o fluxo de rebuttal funcione sem rede.
    """
    baixo = prompt.lower()
    acoes: list[dict[str, Any]] = []
    pedido_rerun = any(
        p in baixo
        for p in ("re-execute", "refaça", "refaca", "reexecut", "rode novamente", "nova métrica", "nova metrica", "adicione uma feature")
    )
    pedido_rewrite = any(
        p in baixo
        for p in ("reescreva", "rewrite", "clareza", "melhore a redação", "melhore a redacao", "reescrit")
    )
    if pedido_rerun:
        acoes.append(
            {
                "tipo": "re-run",
                "descricao": "Reexecutar o experimento com a métrica/feature solicitada pelo revisor.",
                "secoes": ["resultados"],
                "parametros": {"metrica": "f1_score"},
            }
        )
    if pedido_rewrite or not acoes:
        acoes.append(
            {
                "tipo": "rewrite",
                "descricao": "Reescrever a seção de resultados com mais clareza metodológica.",
                "secoes": ["resultados"],
                "parametros": {},
            }
        )
    payload = {
        "acoes": acoes,
        "resumo_geral": "Comentários convertidos em ações acionáveis pelo sistema.",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _mock_review(prompt: str, seed: int) -> str:
    """Parecer determinístico que varia com a seed, como um revisor real variaria.

    As críticas precisam fazer sentido para o manuscrito recebido: o pipeline só
    produz manuscritos sobre observações reais com corte temporal, então o parecer
    nunca deve acusar dados sintéticos nem divisão sem dependência temporal.
    """
    pareceres = ["aceitar com revisões", "revisões maiores", "aceitar com revisões", "aceitar", "revisões maiores"]
    fraquezas = [
        "A cobertura temporal é curta, o que limita a generalização entre anos.",
        "Ausência de análise de sensibilidade a hiperparâmetros.",
        "Não há comparação com métodos do estado da arte recentes.",
        "O tamanho amostral não é justificado por análise de poder.",
        "Métricas agregadas escondem desempenho em subgrupos.",
    ]
    erros = [
        "O desempenho não é testado em municípios deixados de fora do treino.",
        "Não há correção para comparações múltiplas.",
        "A seleção de features pode ter vazamento de informação.",
    ]
    vieses = [
        "Viés de seleção na escolha das séries brasileiras.",
        "Possível viés de otimismo por reuso da mesma partição de teste.",
    ]
    payload = {
        "parecer": pareceres[seed % len(pareceres)],
        "pontos_fortes": [
            "Protocolo reprodutível com seed fixa e métricas explícitas.",
            "Comparação direta contra linha de base estatística.",
            "Artefatos experimentais gravados em JSON, permitindo auditoria.",
        ],
        "pontos_fracos": [
            fraquezas[seed % len(fraquezas)],
            fraquezas[(seed + 2) % len(fraquezas)],
        ],
        "recomendacoes": [
            "Executar o protocolo em base pública real (por exemplo, SIDRA/IBGE).",
            "Reportar intervalos de confiança bootstrap para todas as métricas.",
            "Adicionar validação cruzada temporal.",
        ],
        "severidade": ["baixa", "moderada", "alta"][seed % 3],
        "erros_metodologicos": [erros[seed % len(erros)]],
        "vieses_detectados": [vieses[seed % len(vieses)]],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _mock_score(prompt: str, seed: int) -> str:
    score = 0.72 + (seed % 20) / 100.0
    payload = {
        "score": round(min(score, 0.95), 3),
        "criterios": {
            "originalidade": 0.7,
            "rigor_metodologico": 0.75,
            "reprodutibilidade": 0.8,
            "clareza": 0.7,
            "aderencia_abnt": 0.75,
        },
        "problemas": [
            "Validade externa limitada pela cobertura temporal e municipal."
        ],
        "aprovado": True,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _descricao_da_base(resultado: dict[str, Any] | None) -> tuple[str, str]:
    """Devolve (frase de origem, frase de partição) fiéis aos metadados.

    O texto do mock precisa acompanhar o que o pipeline realmente fez. O runner
    só aceita observações reais, então a frase de origem cita a fonte e o número
    de registros; a partição é sempre temporal.
    """
    dataset = (resultado or {}).get("dataset") or {}
    split = dataset.get("split")
    origem = dataset.get("origem") or "fonte pública não declarada"
    n = dataset.get("n_amostras")

    base = f"A base é de observações reais ({origem})"
    if n:
        base += f", com {n} registros"
    base += "."

    if split == "temporal":
        particao = (
            "Os dados foram padronizados e divididos de forma temporal: treino no "
            "passado e teste em semanas posteriores, para não vazar o futuro no treino."
        )
    else:
        particao = "Os dados foram padronizados e divididos de forma estratificada."
    return base, particao


def _mock_section(prompt: str, seed: int) -> str:
    """Texto de seção específico por seção, citando apenas números do prompt."""
    topic = _topic_from(prompt)
    resultado = _extract_object(prompt, "resultado_experimental")
    best_model = (resultado or {}).get("best_model")
    metrics = (resultado or {}).get("metrics") or {}
    best_metrics = metrics.get(best_model, {}) if best_model else {}
    base_frase, particao_frase = _descricao_da_base(resultado)

    head = prompt.upper()[:400]
    if "RESUMO" in head:
        return (
            f"Este artigo avalia modelos de aprendizado de máquina para {topic}, comparando-os "
            "a uma linha de base sob protocolo reprodutível (semente fixa e validação cruzada). "
            "Os resultados são reportados com intervalos de confiança bootstrap e testes de "
            "hipótese pareados."
        )
    if "INTRODUCAO" in head:
        return (
            f"A literatura sobre {topic} cresce rapidamente, mas permanece fragmentada: poucos "
            "estudos comparam modelos sob o mesmo protocolo e com dados abertos brasileiros. "
            "Esta lacuna motiva a presente investigação, que busca produzir evidência comparável "
            "e integralmente reprodutível."
        )
    if "METODOLOGIA" in head:
        return (
            f"Adotou-se um desenho comparativo controlado. {base_frase} {particao_frase} "
            "Compararam-se uma linha de base e três modelos de aprendizado de máquina, avaliados "
            "por acurácia, F1 e AUC, com intervalos de confiança bootstrap e testes t pareado e de "
            "Wilcoxon."
        )
    if "RESULTADOS" in head:
        if best_model and best_metrics:
            f1 = best_metrics.get("f1_score")
            auc = best_metrics.get("auc_roc")
            acc = best_metrics.get("acuracia")
            parts = [f"O modelo com melhor desempenho foi {best_model}"]
            if acc is not None:
                parts.append(f"acurácia = {acc}")
            if f1 is not None:
                parts.append(f"F1 = {f1}")
            if auc is not None:
                parts.append(f"AUC = {auc}")
            return (
                ", ".join(parts) + ". As diferenças frente à linha de base foram avaliadas por testes "
                "t pareado e de Wilcoxon; os valores de p e os intervalos de confiança constam do "
                "artefato experimental em JSON. O ganho absoluto é modesto, o que reforça a "
                "necessidade de validação em bases reais brasileiras."
            )
        return (
            "Os resultados numéricos completos constam do artefato experimental gerado pelo pipeline "
            "(experimento_H1.json). Nenhum valor é reproduzido aqui sem a execução correspondente."
        )
    if "CONCLUSAO" in head:
        return (
            f"Conclui-se que modelos de aprendizado de máquina são promissores para {topic}, "
            "porém a evidência é preliminar: o ganho frente à linha de base é limitado. Trabalhos "
            "futuros devem ampliar a cobertura temporal e testar a generalização entre municípios."
        )
    return (
        f"Este trabalho investiga {topic} com protocolo reprodutível, métricas padronizadas e "
        "comparação contra linha de base estatística."
    )


def _extract_object(text: str, key: str) -> dict[str, Any] | None:
    """Extrai o objeto JSON associado a ``key`` por casamento de chaves.

    Evita parsear o prompt inteiro (que pode conter vários JSONs) e não falha
    quando o contexto foi truncado: devolve ``None`` nesse caso.
    """
    marker = f'"{key}"'
    start = text.find(marker)
    if start == -1:
        return None
    brace = text.find("{", start + len(marker))
    if brace == -1:
        return None
    depth = 0
    for i in range(brace, len(text)):
        char = text[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[brace : i + 1])
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


_MOCK_BUILDERS: dict[str, Callable[[str, int], str]] = {
    "hypotheses": _mock_hypotheses,
    "novelty": _mock_novelty,
    "gap": _mock_gap,
    "experiment_plan": _mock_experiment_plan,
    "analysis": _mock_analysis,
    "auto_debug": _mock_auto_debug,
    "rebuttal": _mock_rebuttal,
    "review": _mock_review,
    "score": _mock_score,
    "compliance": _mock_compliance,
    "translate": _mock_translate,
    "section": _mock_section,
}


def _topic_from(prompt: str) -> str:
    match = re.search(r"T[ÓO]PICO\s*:\s*(.+)", prompt, flags=re.IGNORECASE)
    if match:
        return match.group(1).split("\n")[0].strip()
    return "o tema proposto"


# --------------------------------------------------------------------------- #
# OpenAI-compatible
# --------------------------------------------------------------------------- #
def _parse_retry_hint(message: str) -> float | None:
    """Extrai os segundos sugeridos por um HTTP 429.

    Os provedores usam formatos diferentes: ``try again in 27.66s`` (Groq por
    minuto) e ``try again in 26m16.8s`` (Groq por dia). Sem entender o formato
    com minutos, o cliente esperava o backoff genérico e insistia num provider
    que só voltaria horas depois.
    """
    match = re.search(
        r"try again in\s+(?:(\d+)\s*m)?\s*([0-9.]+)\s*s",
        message,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    minutos = float(match.group(1)) if match.group(1) else 0.0
    return minutos * 60.0 + float(match.group(2))


_PLACEHOLDERS = {
    "sua_chave_real_aqui",
    "sua-chave-aqui",
    "your_api_key_here",
    "changeme",
    "todo",
    "xxx",
    "none",
    "null",
}


def _placeholder(valor: str | None) -> bool:
    """Detecta valores de exemplo no ``.env``, que não são chaves de verdade.

    Sem isto, um ``.env`` recém-copiado do ``.env.example`` faria o provider
    parecer disponível: o roteador tentaria uma chamada real, gastaria uma
    tentativa e voltaria 401 — pior que simplesmente não ter a chave.
    """
    if not valor or not valor.strip():
        return True
    return valor.strip().lower() in _PLACEHOLDERS


@dataclass
class OpenAICompatibleProvider:
    """Cliente HTTP para endpoints compatíveis com ``/chat/completions``."""
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    api_key_env: str | None = None
    timeout: int = 60
    temperature: float = 0.2
    max_tokens: int = 2048
    name: str = "openai_compatible"

    def __post_init__(self) -> None:
        # api_key_env permite empilhar vários provedores gratuitos, cada um com
        # a sua própria variável (GROQ_API_KEY, GEMINI_API_KEY...), sem que um
        # sobrescreva o outro via OPENAI_API_KEY.
        if self.api_key_env:
            self.api_key = self.api_key or os.getenv(self.api_key_env)
        else:
            self.api_key = self.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("AI_CIENTISTA_API_KEY")
        # OPENAI_BASE_URL é tratada por core.config.ENV_OVERRIDES, que a mapeia
        # para ``llm.base_url``. Ler a variável aqui de novo inverteria a
        # precedência e sobrescreveria um base_url passado explicitamente.
        self.base_url = self.base_url.rstrip("/")

    def available(self) -> bool:
        return bool(self.api_key) and not _placeholder(self.api_key)

    def complete(self, prompt: str, system: str | None = None, **kwargs: Any) -> str:
        if not self.available():
            raise RuntimeError(f"chave ausente para o provider {self.name}")
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        # response_format força JSON estruturado quando a tarefa exige (o
        # roteador decide). Nem todo endpoint suporta; por isso é opcional e só
        # entra quando pedido explicitamente.
        response_format = kwargs.get("response_format")
        if response_format:
            payload["response_format"] = response_format
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                # Sem um User-Agent explícito o Cloudflare da Groq responde 1010
                # ("browser signature") e bloqueia o cliente urllib.
                "User-Agent": USER_AGENT,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - depende de rede
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - depende de rede
            raise RuntimeError(f"Falha de conexão: {exc.reason}") from exc
        return body["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
@dataclass
class LLMCache:
    """Cache em SQLite para evitar chamadas repetidas ao mesmo prompt.

    O acesso é protegido por um lock e a conexão usa ``check_same_thread=False``,
    porque o painel de revisão consulta o cache de várias threads. Sem isso, o
    SQLite recusa o uso cross-thread e o painel inteiro falha.
    """

    path: Path
    enabled: bool = True

    def __post_init__(self) -> None:
        import threading

        self._conn = None
        self._lock = threading.Lock()
        if self.enabled:
            import sqlite3

            ensure_dir(Path(self.path).parent)
            self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT, provider TEXT, created_at TEXT)"
            )
            self._conn.commit()

    def get(self, key: str) -> tuple[str, str] | None:
        if not self._conn:
            return None
        with self._lock:
            row = self._conn.execute("SELECT value, provider FROM cache WHERE key = ?", (key,)).fetchone()
        return (row[0], row[1]) if row else None

    def put(self, key: str, value: str, provider: str) -> None:
        if not self._conn:
            return
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, provider, created_at) VALUES (?, ?, ?, ?)",
                (key, value, provider, utcnow_iso()),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None


# --------------------------------------------------------------------------- #
# Roteamento por tarefa (MoE)
# --------------------------------------------------------------------------- #
# Mapeamento tarefa -> categoria. Espelha ``tarefas_para_categoria`` do
# ``config.modelos.json``; é o fallback quando o config não define o mapeamento,
# para que o roteador funcione mesmo com um config antigo ou parcial.
TAREFA_PARA_CATEGORIA: dict[str, str] = {
    "gap": "raciocinio",
    "hypotheses": "raciocinio",
    "novelty": "raciocinio",
    "review": "raciocinio",
    "section": "redacao",
    "translate": "redacao",
    "rebuttal": "redacao",
    "rebuttal_response": "redacao",
    "experiment_plan": "codigo",
    "analysis": "codigo",
    "auto_debug": "codigo",
    "score": "alto_volume",
    "compliance": "alto_volume",
    "voting": "alto_volume",
    "satellite_features": "multimodal",
    "chart_reading": "multimodal",
}

# Categoria -> max_tokens. Raciocínio e código gastam o orçamento em
# chain-of-thought antes de emitir o JSON: com 512 tokens o conteúdo voltava
# vazio e o modelo era reprovado por truncamento (comprovado no benchmark).
MAX_TOKENS_POR_CATEGORIA: dict[str, int] = {
    "raciocinio": 4096,
    "codigo": 4096,
    "redacao": 2048,
    "alto_volume": 2048,
    "multimodal": 2048,
    "validacao_cruzada": 2048,
    "padrao": 2048,
}

# Tarefas cujo contrato de saída é JSON estrito; força ``json_object``.
TAREFAS_JSON: frozenset[str] = frozenset(
    {"gap", "hypotheses", "novelty", "review", "experiment_plan", "analysis",
     "auto_debug", "score", "compliance", "voting", "satellite_features", "chart_reading"}
)

# Categoria de raciocínio é a mais exigente; usada quando não há tarefa.
CATEGORIA_PADRAO = "raciocinio"


def categoria_da_tarefa(task: str | None, mapa: dict[str, str] | None = None) -> str:
    """Traduz o nome da tarefa na categoria de roteamento."""
    if not task:
        return CATEGORIA_PADRAO
    return (mapa or TAREFA_PARA_CATEGORIA).get(task, CATEGORIA_PADRAO)


def max_tokens_da_tarefa(task: str | None, presets: dict[str, int] | None = None) -> int:
    """max_tokens adequado à categoria da tarefa (nunca abaixo do mínimo)."""
    valores = dict(MAX_TOKENS_POR_CATEGORIA)
    if presets:
        valores.update(presets)
    categoria = categoria_da_tarefa(task)
    return max(valores.get(categoria, valores["padrao"]), MAX_TOKENS_POR_CATEGORIA["padrao"])


def _resumo_erro(exc: Exception) -> str:
    """Primeira linha do erro, curta o bastante para o log do roteador."""
    texto = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
    return texto[:120]


def _falha_transitoria(exc: Exception) -> bool:
    """Erro que justifica pular para o próximo modelo da cascata.

    429 (rate limit), 5xx (instabilidade) e timeouts são do momento, não do
    pedido: insistir no mesmo modelo só queima o tempo que o próximo da lista
    levaria para responder. Erros de contrato (JSON inválido, chave ausente)
    não entram aqui — nesses, repetir o mesmo modelo pode resolver.
    """
    texto = str(exc).lower()
    marcadores = ("429", "rate limit", "500", "502", "503", "504", "timed out", "timeout", "connection")
    return any(m in texto for m in marcadores)


@dataclass
class TaskRouter:
    """Escolhe o provider por tarefa, com fallback em cascata.

    Antes, todo agente usava a mesma cadeia de providers, na mesma ordem: o
    redator (a tarefa mais pesada) rodava no modelo mais leve e estourava o
    orçamento de tokens. Aqui, cada categoria de tarefa tem a sua lista de
    modelos — os que passaram no benchmark para aquela função — e o roteador
    desce a lista quando um modelo falha.

    A cascata tem três níveis: primeiro os modelos da categoria; se todos
    falharem, o ``coringa`` (Gemini, a última API real); e, só se o coringa não
    estiver configurado ou também falhar, a ``cadeia_global`` — cujo último elo
    é o ``mock``, que sempre responde (CI e emergência total).
    """

    rotas: dict[str, list[OpenAICompatibleProvider]] = field(default_factory=dict)
    mapa_tarefas: dict[str, str] = field(default_factory=lambda: dict(TAREFA_PARA_CATEGORIA))
    presets: dict[str, int] = field(default_factory=dict)
    cadeia_global: list[LLMProvider] = field(default_factory=list)
    # Coringa: o último modelo real antes do mock. Um único provider de reserva
    # que serve a qualquer tarefa, então uma categoria inteira sem cota (ou sem
    # chave) ainda produz texto científico real em vez de cair no mock.
    coringa: LLMProvider | None = None

    def rota(self, task: str | None) -> list[LLMProvider]:
        """Providers candidatos para a tarefa, na ordem de preferência.

        Providers sem chave configurada são descartados aqui, para não gastar
        uma tentativa com um ``RuntimeError`` de chave ausente.
        """
        categoria = categoria_da_tarefa(task, self.mapa_tarefas)
        candidatos = self.rotas.get(categoria) or []
        disponiveis = [p for p in candidatos if p.available()]
        if disponiveis:
            return disponiveis
        # Sem rota utilizável: a categoria pode não ter sido configurada, ou
        # nenhuma chave da categoria está presente. O coringa assume; se ele
        # também não estiver disponível, resta a cadeia global.
        if self.coringa is not None and self.coringa.available():
            return [self.coringa]
        return [p for p in self.cadeia_global if p.available()]

    def max_tokens(self, task: str | None) -> int:
        return max_tokens_da_tarefa(task, self.presets)

    @property
    def ativo(self) -> bool:
        """Há ao menos um provider real roteável? (mock não conta)

        O coringa conta: com só a ``GEMINI_API_KEY`` configurada, o roteamento
        ainda é útil — o Gemini atende todas as tarefas e o ``max_tokens`` por
        categoria continua valendo.
        """
        if self.coringa is not None and self.coringa.available():
            return True
        return any(
            isinstance(p, OpenAICompatibleProvider) and p.available()
            for lista in self.rotas.values()
            for p in lista
        )


def build_task_router(
    catalogo: dict[str, Any],
    providers_por_nome: dict[str, LLMProvider],
) -> TaskRouter | None:
    """Monta o :class:`TaskRouter` a partir do catálogo ``config.modelos.json``.

    ``providers_por_nome`` são os providers já instanciados pelo
    :func:`build_llm_client` (uma instância por ``name`` de provider). O
    roteador cria, para cada modelo do catálogo, um provider próprio — porque
    dois modelos podem compartilhar o mesmo provider mas ter nomes distintos.
    """
    rotas_cfg = catalogo.get("roteamento_por_tarefa") or {}
    modelos_cfg = catalogo.get("modelos") or {}
    if not rotas_cfg or not modelos_cfg:
        return None

    def provider_para(modelo: str) -> OpenAICompatibleProvider | None:
        meta = modelos_cfg.get(modelo)
        if not meta:
            logger.warning("[LLM Router] modelo %s não está em 'modelos'; ignorado", modelo)
            return None
        base = providers_por_nome.get(meta.get("provider", ""))
        return OpenAICompatibleProvider(
            name=modelo,
            # ``model`` permite que o id enviado à API difira da chave do
            # catálogo (a Mistral não aceita o prefixo "mistral/"; o NIM usa
            # "meta/muse-glimmer-30b").
            model=meta.get("model", modelo),
            base_url=meta.get("base_url", getattr(base, "base_url", "https://api.openai.com/v1")),
            api_key_env=meta.get("api_key_env"),
            timeout=getattr(base, "timeout", 60),
            temperature=getattr(base, "temperature", 0.2),
            max_tokens=getattr(base, "max_tokens", 2048),
        )

    rotas: dict[str, list[OpenAICompatibleProvider]] = {}
    for categoria, modelos in rotas_cfg.items():
        lista = [p for p in (provider_para(m) for m in modelos) if p is not None]
        if lista:
            rotas[categoria] = lista

    presets = {
        categoria: int(bloco["max_tokens"])
        for categoria, bloco in (catalogo.get("presets") or {}).items()
        if isinstance(bloco, dict) and "max_tokens" in bloco
    }

    # Coringa: o último modelo real, tentado depois de esgotar a categoria e
    # antes do mock. É excluído da cadeia global abaixo para não ser tentado
    # duas vezes na mesma chamada.
    fallback_cfg = catalogo.get("fallback_global") or {}
    nome_coringa = fallback_cfg.get("coringa")
    coringa: LLMProvider | None = None
    if nome_coringa and nome_coringa != "mock":
        coringa = provider_para(nome_coringa)
        if coringa is not None and not coringa.available():
            logger.info(
                "[LLM Router] coringa %s configurado, mas %s não está definida; "
                "a cadeia terminará no mock",
                nome_coringa, getattr(coringa, "api_key_env", "?") or "a chave",
            )

    cadeia_global: list[LLMProvider] = []
    for nome in fallback_cfg.get("cadeia", []):
        if nome == nome_coringa:
            continue  # o coringa já tem o seu nível próprio na cascata
        if nome == "mock":
            cadeia_global.append(MockProvider())
            continue
        meta = modelos_cfg.get(nome)
        if meta:
            p = provider_para(nome)
            if p is not None:
                cadeia_global.append(p)
    if not any(isinstance(p, MockProvider) for p in cadeia_global):
        cadeia_global.append(MockProvider())

    return TaskRouter(
        coringa=coringa,
        rotas=rotas,
        mapa_tarefas=dict(catalogo.get("tarefas_para_categoria") or TAREFA_PARA_CATEGORIA),
        presets=presets,
        cadeia_global=cadeia_global,
    )


# --------------------------------------------------------------------------- #
# Cliente com fallback
# --------------------------------------------------------------------------- #
@dataclass
class LLMClient:
    """Cliente de alto nível: cache + cadeia de fallback entre providers."""

    providers: list[LLMProvider] = field(default_factory=list)
    cache: LLMCache | None = None
    router: TaskRouter | None = None
    temperature: float = 0.2
    retries: int = 2
    history: list[dict[str, Any]] = field(default_factory=list)
    failures: dict[str, int] = field(default_factory=dict)
    _lock: Any = field(default=None, repr=False)

    # Acima disso, esperar pela API não compensa: o provider provavelmente está
    # com o teto diário esgotado ("try again in 26m"). Melhor passar ao próximo.
    MAX_ESPERA_POR_TENTATIVA: float = 30.0

    def __post_init__(self) -> None:
        import threading

        if self._lock is None:
            self._lock = threading.Lock()

    def _signature(self) -> str:
        """Identifica a cadeia de providers para compor a chave de cache.

        Sem isto, uma resposta do ``mock`` já em cache seria devolvida numa
        execução com um LLM real, mascarando completamente o provider escolhido.

        A versão entra na chave porque o texto do ``mock`` é código nosso: ao
        corrigi-lo, as respostas antigas em disco ficariam obsoletas e voltariam
        a aparecer nas execuções seguintes.

        Com roteador ativo, a assinatura reflete a rota de cada categoria: sem
        isso, a resposta do ``gemini`` obtida numa categoria seria devolvida
        como cache para a categoria que deveria usar o ``codestral``.
        """
        if self.router and self.router.ativo:
            cadeia = ";".join(
                f"{cat}={'|'.join(getattr(p, 'name', '?') for p in lista)}"
                for cat, lista in sorted(self.router.rotas.items())
            )
            # O coringa entra na assinatura: sem isso, a resposta real dele numa
            # execução seria devolvida como cache para a execução em que só o
            # mock está configurado (e vice-versa).
            nome_coringa = getattr(self.router.coringa, "name", "")
            return f"v{CACHE_VERSION}:router:{cadeia}|coringa={nome_coringa}"
        cadeia = ",".join(f"{p.name}:{getattr(p, 'model', '')}" for p in self.providers)
        return f"v{CACHE_VERSION}:{cadeia}"

    def _providers_para(self, task: str | None) -> list[LLMProvider]:
        """Cadeia de providers a usar, na ordem estrita da cascata.

        Nível 1: modelos da categoria (os validados no benchmark para a função).
        Nível 2: o coringa (Gemini), a última API real — se a categoria inteira
        estourou a cota, ainda assim sai texto científico de verdade.
        Nível 3: a cadeia global, cujo último elo é o ``mock`` (CI e emergência).

        Sem o nível 2, um dia ruim de cota derrubaria o pipeline direto no mock,
        que não produz ciência — só estrutura.
        """
        if not (self.router and self.router.ativo):
            return self.providers
        cadeia = list(self.router.rota(task))
        vistos = {id(p) for p in cadeia}
        coringa = self.router.coringa
        # O coringa só entra se tiver chave: sem ela, não faz sentido gastar uma
        # tentativa (nem poluir o log com "indisponível").
        if coringa is not None and coringa.available() and id(coringa) not in vistos:
            cadeia.append(coringa)
            vistos.add(id(coringa))
        for p in self.router.cadeia_global:
            if id(p) not in vistos:
                cadeia.append(p)
                vistos.add(id(p))
        return cadeia or self.providers

    def complete(self, prompt: str, system: str | None = None, **kwargs: Any) -> LLMResponse:
        temperature = float(kwargs.pop("temperature", self.temperature))
        task = kwargs.pop("task", None)
        # max_tokens por categoria: raciocínio/código precisam de espaço para o
        # chain-of-thought antes do JSON. Um valor explícito do chamador vence.
        if "max_tokens" not in kwargs and self.router and self.router.ativo:
            kwargs["max_tokens"] = self.router.max_tokens(task)
        # response_format só é passado quando a tarefa tem contrato JSON estrito.
        # Fora desses casos, pedir json_object a um endpoint que não suporta
        # (ou a um texto livre) causaria erro ou saída degradada.
        if task in TAREFAS_JSON and "response_format" not in kwargs:
            kwargs["response_format"] = {"type": "json_object"}
        # A tarefa entra na chave explicitamente: duas categorias podem gerar os
        # mesmos kwargs (mesmo max_tokens) e, sem isso, a resposta de uma seria
        # devolvida como cache para a outra — servindo ao agente errado.
        key = sha256_text(
            f"{self._signature()}||task={task}||{system or ''}||{prompt}||{json.dumps(kwargs, sort_keys=True, default=str)}||{temperature}"
        )
        if self.cache:
            hit = self.cache.get(key)
            if hit:
                self._record(prompt, hit[0], hit[1], cached=True, latency=0.0)
                return LLMResponse(text=hit[0], provider=hit[1], cached=True)

        errors: list[str] = []
        roteando = bool(self.router and self.router.ativo)
        coringa = self.router.coringa if roteando else None
        for provider in self._providers_para(task):
            if not provider.available():
                errors.append(f"{provider.name}: indisponível")
                continue
            for attempt in range(self.retries + 1):
                start = time.perf_counter()
                try:
                    # ``task`` é repassado ao provider: o mock escolhe o
                    # builder pelo tipo de tarefa; os reais o ignoram.
                    text = provider.complete(
                        prompt, system=system, temperature=temperature, task=task, **kwargs
                    )
                    latency = time.perf_counter() - start
                    if self.cache:
                        self.cache.put(key, text, provider.name)
                    self._record(prompt, text, provider.name, cached=False, latency=latency)
                    if roteando:
                        if coringa is not None and provider is coringa:
                            logger.info(
                                "[LLM Router] Fallback acionado: usando GEMINI_API_KEY "
                                "como coringa para task=%s",
                                task or "?",
                            )
                        logger.info(
                            "[LLM Router] task=%s → %s (tempo: %.1fs)",
                            task or "?", provider.name, latency,
                        )
                    return LLMResponse(text=text, provider=provider.name, latency_s=latency)
                except Exception as exc:  # noqa: BLE001 - fallback deliberado
                    errors.append(f"{provider.name} (tentativa {attempt + 1}): {exc}")
                    logger.warning("Provider %s falhou: %s", provider.name, exc)
                    with self._lock:
                        self.failures[provider.name] = self.failures.get(provider.name, 0) + 1
                    # Em modo roteado, a cascata é a redundância: falha transitória
                    # (429, 5xx, timeout) pula direto para o próximo modelo, em vez
                    # de gastar segundos de backoff no mesmo modelo.
                    if roteando and _falha_transitoria(exc):
                        logger.info(
                            "[LLM Router] task=%s: %s falhou (%s); próximo da cascata",
                            task or "?", provider.name, _resumo_erro(exc),
                        )
                        break
                    if attempt < self.retries:
                        espera = self._backoff(exc, attempt)
                        # Se a API pede mais do que vale a pena esperar (ex.: teto
                        # diário de tokens, "try again in 26m"), insistir neste
                        # provider só queima tempo: passa para o próximo.
                        if espera > self.MAX_ESPERA_POR_TENTATIVA:
                            pedido = _parse_retry_hint(str(exc))
                            logger.warning(
                                "Provider %s pediu %s de espera; pulando para o próximo da cadeia",
                                provider.name,
                                f"{pedido:.0f}s" if pedido else f">{self.MAX_ESPERA_POR_TENTATIVA:.0f}s",
                            )
                            break
                        time.sleep(espera)
        message = "; ".join(errors) or "nenhum provider configurado"
        logger.error("Todos os providers falharam: %s", message)
        raise RuntimeError(f"Falha em todos os providers de LLM: {message}")

    @staticmethod
    def _backoff(exc: Exception, attempt: int) -> float:
        """Calcula a espera antes da próxima tentativa.

        Em HTTP 429 a API informa quanto esperar ("try again in 27.6s"); ignorar
        isso esgota as tentativas e derruba o pipeline para o provider mock sem
        que o usuário perceba. Respeitamos o valor sugerido, com um teto para não
        travar indefinidamente.
        """
        message = str(exc)
        if "429" in message or "rate limit" in message.lower():
            # +1s de folga: pedir exatamente o tempo sugerido costuma voltar 429.
            hint = _parse_retry_hint(message)
            return min((hint + 1.0) if hint else (20.0 * (attempt + 1)), 65.0)
        return min(2 ** attempt * 0.5, 8.0)

    def _record(self, prompt: str, text: str, provider: str, cached: bool, latency: float) -> None:
        entry = {
            "provider": provider,
            "cached": cached,
            "latency_s": round(latency, 4),
            "prompt_chars": len(prompt),
            "response_chars": len(text),
            # Estimativa grosseira (~4 caracteres por token) usada só para
            # relatório de custo; não substitui a contagem do provider.
            "prompt_tokens_est": len(prompt) // 4,
            "response_tokens_est": len(text) // 4,
        }
        with self._lock:
            self.history.append(entry)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            history = list(self.history)
            failures = dict(self.failures)
        total = len(history)
        cached = sum(1 for h in history if h["cached"])
        by_provider: dict[str, int] = {}
        prompt_tokens = 0
        response_tokens = 0
        latency = 0.0
        for item in history:
            by_provider[item["provider"]] = by_provider.get(item["provider"], 0) + 1
            prompt_tokens += item.get("prompt_tokens_est", 0)
            response_tokens += item.get("response_tokens_est", 0)
            latency += item.get("latency_s", 0.0)
        return {
            "chamadas": total,
            "cache_hits": cached,
            "por_provider": by_provider,
            # Configurados inclui os que nunca responderam: sem isso, uma cadeia
            # em que todos falharam pareceria "nenhum LLM real configurado".
            "providers_configurados": [p.name for p in self.providers],
            "falhas_por_provider": failures,
            "tokens_estimados": {"prompt": prompt_tokens, "resposta": response_tokens, "total": prompt_tokens + response_tokens},
            "latencia_total_s": round(latency, 3),
        }

    def close(self) -> None:
        if self.cache:
            self.cache.close()


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_llm_client(config: dict[str, Any], use_cache: bool | None = None) -> LLMClient:
    """Monta o ``LLMClient`` a partir do dicionário de configuração ``llm``."""
    llm_cfg = dict(config.get("llm", {}))
    cache_cfg = dict(llm_cfg.get("cache", {}))
    enabled = cache_cfg.get("enabled", True) if use_cache is None else use_cache
    cache_path = Path(cache_cfg.get("path", "outputs/llm_cache.sqlite3"))
    cache = LLMCache(path=cache_path, enabled=bool(enabled)) if enabled else None

    # Duas formas de configurar a cadeia de providers:
    #   "providers": [ {...}, {...} ]  -> lista nomeada (permite empilhar tiers
    #                                     gratuitos de vários provedores)
    #   "fallback_order": ["openai_compatible", "mock"] -> forma legada
    named = llm_cfg.get("providers")
    providers: list[LLMProvider] = []
    if isinstance(named, list) and named:
        for entry in named:
            if not isinstance(entry, dict):
                continue
            kind = entry.get("type", "openai_compatible")
            if kind == "mock":
                providers.append(MockProvider())
            elif kind == "selenium":
                try:
                    from llm_providers.bypass.selenium_llm import SeleniumProvider

                    providers.append(SeleniumProvider())
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Provider selenium indisponível: %s", exc)
            else:
                providers.append(
                    OpenAICompatibleProvider(
                        name=entry.get("name", kind),
                        model=entry.get("model", llm_cfg.get("model", "gpt-4o-mini")),
                        base_url=entry.get("base_url", llm_cfg.get("base_url", "https://api.openai.com/v1")),
                        api_key=entry.get("api_key"),
                        api_key_env=entry.get("api_key_env"),
                        timeout=int(entry.get("timeout", llm_cfg.get("timeout", 60))),
                        temperature=float(entry.get("temperature", llm_cfg.get("temperature", 0.2))),
                        max_tokens=int(entry.get("max_tokens", llm_cfg.get("max_tokens", 2048))),
                    )
                )

    if not providers:
        order = llm_cfg.get("fallback_order") or [llm_cfg.get("provider", "mock")]
        for name in order:
            if name == "openai_compatible":
                providers.append(
                    OpenAICompatibleProvider(
                        model=llm_cfg.get("model", "gpt-4o-mini"),
                        base_url=llm_cfg.get("base_url", "https://api.openai.com/v1"),
                        timeout=int(llm_cfg.get("timeout", 60)),
                        temperature=float(llm_cfg.get("temperature", 0.2)),
                        max_tokens=int(llm_cfg.get("max_tokens", 2048)),
                    )
                )
            elif name == "selenium":
                try:
                    from llm_providers.bypass.selenium_llm import SeleniumProvider

                    providers.append(SeleniumProvider())
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Provider selenium indisponível: %s", exc)
            elif name == "mock":
                providers.append(MockProvider())

    if not providers:
        providers.append(MockProvider())

    # Roteador por tarefa: lê o catálogo validado (config.modelos.json) e monta
    # a rota de cada categoria. Só entra quando a cadeia configurada tem ao menos
    # um provider real: se o usuário pediu explicitamente ``mock`` (testes,
    # validação offline), rotear para APIs de rede seria ignorar a configuração —
    # e faria os testes dependerem de cota e de rede.
    router: TaskRouter | None = None
    tem_provider_real = any(not isinstance(p, MockProvider) for p in providers)
    catalogo_path = llm_cfg.get("catalogo_modelos")
    if catalogo_path is None:
        candidato = ROOT / "config.modelos.json"
        catalogo_path = candidato if candidato.exists() else None
    if catalogo_path and tem_provider_real:
        catalogo = read_json(catalogo_path, default={}) or {}
        por_nome = {getattr(p, "name", ""): p for p in providers}
        router = build_task_router(catalogo, por_nome)
        if router and router.ativo:
            logger.info(
                "Roteamento por tarefa ativo: %s categorias (%s modelos)",
                len(router.rotas),
                sum(len(v) for v in router.rotas.values()),
            )
        else:
            router = None

    return LLMClient(
        providers=providers,
        cache=cache,
        router=router,
        temperature=float(llm_cfg.get("temperature", 0.2)),
        retries=int(llm_cfg.get("retries", 2)),
    )
