"""Pacote de submissão: reúne artigo, código, dados e instruções.

O ZIP de submissão é a prova de reprodutibilidade do ciclo: além do PDF em
ABNT, ele carrega o código exato que produziu os números, os metadados de
procedência dos dados e um guia com as normas de SciELO Preprints, arXiv e SBC.

Regra de integridade: nenhum pacote é gerado a partir de dados que não sejam
100% reais. Se o ``ciclo_completo.json`` mencionar dados sintéticos, a geração
levanta :class:`IntegrityError`.
"""

from __future__ import annotations

import json
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.utils import ensure_dir, setup_logger, write_json

logger = setup_logger(__name__)

# Palavras que denunciam base sintética nos metadados do ciclo.
# Sem acento de propósito: a comparação roda sobre texto normalizado.
TERMOS_SINTETICOS = ("sintetic", "synthetic", "simulad", "fake", "mock data")

INSTRUCOES = """# Instruções de submissão

Este pacote reúne o artigo, o código e a procedência dos dados do ciclo
`{cycle_id}` (tópico: _{topic}_). Antes de submeter, revise o texto e confirme
a afiliação dos autores.

## Conteúdo do pacote

| Arquivo | Descrição |
|---|---|
| `paper.pdf` | Artigo formatado em ABNT. |
| `paper.tex` | Fonte LaTeX do artigo. |
| `bibliografia.bib` | Referências em BibTeX, para editores LaTeX. |
| `codigo_experimento.py` | Script que reproduz os números do artigo. |
| `metadata_dados.json` | Procedência dos dados: fonte, URL e data de coleta. |
| `INSTRUCOES_SUBMISSAO.md` | Este guia. |

## SciELO Preprints

- Portal: https://preprints.scielo.org
- Aceita português, inglês e espanhol; revise o resumo nos dois idiomas.
- Exige declaração de conflito de interesses e de financiamento.
- A licença padrão é CC BY 4.0.

## arXiv

- Portal: https://arxiv.org/submit
- Exige o fonte LaTeX (`paper.tex` + `bibliografia.bib`) e a categoria (ex.: `cs.LG`).
- Precisa de endosso para o primeiro envio em algumas categorias.
- O idioma de submissão é o inglês; traduza o manuscrito antes.

## SBC (BRACIS, SBBD, STIL)

- Portal: https://submissoes.sbc.org.br
- Formato SBC: duas colunas, `sbc-template.sty`, 10 páginas em geral.
- Aceita português e inglês; revise as regras de anonimização (revisão dupla-cega).
- Inclua a seção de aspectos éticos e a declaração de uso de IA.

## Declaração de uso de IA

Este artigo foi produzido com apoio de um sistema autônomo multi-agente. Divulgue
essa informação na seção de metodologia ou em nota, conforme as normas do veículo.
Todos os números derivam da execução do código incluído; os dados vêm das fontes
públicas listadas em `metadata_dados.json`.
"""


class IntegrityError(RuntimeError):
    """Levantado quando o ciclo não pode ser empacotado por falta de integridade."""


@dataclass
class SubmissionPackage:
    """Resultado da montagem do pacote de submissão."""

    path: str
    cycle_id: str
    arquivos: list[str]
    tamanho_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "caminho": self.path,
            "cycle_id": self.cycle_id,
            "arquivos": self.arquivos,
            "tamanho_bytes": self.tamanho_bytes,
        }


def _carregar_ciclo(cycle_dir: Path) -> dict[str, Any]:
    caminho = cycle_dir / "ciclo_completo.json"
    if not caminho.exists():
        raise IntegrityError(f"ciclo_completo.json ausente em {cycle_dir}")
    try:
        return json.loads(caminho.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise IntegrityError(f"ciclo_completo.json ilegível: {exc}") from exc


def _detectar_sinteticos(payload: Any, trilha: str = "") -> list[str]:
    """Varre os metadados do experimento em busca de termos que denunciem dados
    sintéticos. Devolve as trilhas onde a palavra apareceu (ex.: ``dataset.tipo``).

    A varredura cobre apenas o bloco ``experimento`` (desenho e resultado), que é
    o metadado do que foi executado. A prosa das seções do artigo não entra: um
    trabalho pode citar "dados sintéticos" legitimamente ao discutir a literatura.
    """
    achados: list[str] = []
    if isinstance(payload, dict):
        for chave, valor in payload.items():
            achados.extend(_detectar_sinteticos(valor, f"{trilha}.{chave}" if trilha else str(chave)))
    elif isinstance(payload, list):
        for i, valor in enumerate(payload):
            achados.extend(_detectar_sinteticos(valor, f"{trilha}[{i}]"))
    elif isinstance(payload, str):
        texto = unicodedata.normalize("NFKD", payload.lower())
        texto = "".join(c for c in texto if not unicodedata.combining(c))
        if any(termo in texto for termo in TERMOS_SINTETICOS):
            achados.append(trilha)
    return achados


def _metadata_dados(ciclo: dict[str, Any]) -> dict[str, Any]:
    """Extrai a procedência dos dados a partir do ciclo completo."""
    resultado = ((ciclo.get("experimento") or {}).get("resultado") or {})
    dataset = resultado.get("dataset") or {}
    return {
        "topico": resultado.get("topic"),
        "tipo": dataset.get("tipo"),
        "origem": dataset.get("origem"),
        "fonte_url": dataset.get("fonte_url"),
        "coletado_em": dataset.get("coletado_em"),
        "n_amostras": dataset.get("n_amostras"),
        "n_variaveis": dataset.get("n_variaveis"),
        "variaveis": dataset.get("variaveis"),
        "split": dataset.get("split"),
        "balanceamento": dataset.get("balanceamento"),
        "best_model": resultado.get("best_model"),
        "metrics": resultado.get("metrics"),
    }


def _script_reproducao(ciclo: dict[str, Any]) -> str:
    """Gera um script autônomo que repete o experimento do ciclo."""
    resultado = ((ciclo.get("experimento") or {}).get("resultado") or {})
    topico = resultado.get("topic") or ""
    seed = resultado.get("seed", 42)
    return f'''#!/usr/bin/env python3
"""Reprodução do experimento do ciclo {ciclo.get("cycle_id", "?")}.

Executa o mesmo runner, sobre os mesmos dados públicos, com a mesma seed.
Requer as dependências de requirements.txt e acesso à internet (ou o cache de
observações reais em data/dataset_cache).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.experiments import ExperimentRunner

TOPICO = {topico!r}
SEED = {seed!r}


def main() -> int:
    runner = ExperimentRunner(seed=SEED)
    result = runner.run(TOPICO, {{"id": "H1", "enunciado": "Reprodução do ciclo"}})
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    print("\\nMelhor modelo:", result.best_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _bibtex(referencias: list[dict[str, Any]]) -> str:
    entradas: list[str] = []
    for i, ref in enumerate(referencias, start=1):
        titulo = str(ref.get("titulo") or ref.get("title") or "Sem título").replace("{", "").replace("}", "")
        autores = ref.get("autores") or ref.get("authors") or ""
        if isinstance(autores, list):
            autores = " and ".join(str(a) for a in autores)
        ano = ref.get("ano") or ref.get("year") or ""
        doi = ref.get("doi") or ""
        url = ref.get("url") or (f"https://doi.org/{doi}" if doi else "")
        entradas.append(
            "@article{ref%d,\n  title = {%s},\n  author = {%s},\n  year = {%s},\n  url = {%s}\n}"
            % (i, titulo, autores or "Desconhecido", ano, url)
        )
    return "\n\n".join(entradas) + ("\n" if entradas else "")


def generate_submission_package(
    cycle_dir: str | Path,
    output_path: str | Path | None = None,
) -> SubmissionPackage:
    """Monta o ZIP de submissão de um ciclo concluído.

    Levanta :class:`IntegrityError` se os metadados não atestarem dados 100%
    reais, ou se faltarem os artefatos obrigatórios (`paper.pdf`, `paper.tex`).
    """
    ciclo_dir = Path(cycle_dir)
    if not ciclo_dir.is_dir():
        raise IntegrityError(f"diretório do ciclo inexistente: {ciclo_dir}")

    ciclo = _carregar_ciclo(ciclo_dir)

    dataset = (((ciclo.get("experimento") or {}).get("resultado") or {}).get("dataset") or {})
    if dataset.get("tipo") != "dados_reais":
        raise IntegrityError(
            "Pacote de submissão rejeitado: dados não são 100% reais "
            f"(tipo={dataset.get('tipo')!r})."
        )
    suspeitos = _detectar_sinteticos(ciclo.get("experimento") or {})
    if suspeitos:
        raise IntegrityError(
            "Pacote de submissão rejeitado: dados não são 100% reais. "
            f"Termos suspeitos em: {', '.join(sorted(set(suspeitos))[:5])}."
        )

    paper_pdf = ciclo_dir / "paper.pdf"
    paper_tex = ciclo_dir / "paper.tex"
    if not paper_pdf.exists() or not paper_tex.exists():
        raise IntegrityError("paper.pdf e paper.tex são obrigatórios para o pacote de submissão.")

    cycle_id = str(ciclo.get("cycle_id") or ciclo_dir.name)
    topic = (((ciclo.get("experimento") or {}).get("resultado") or {}).get("topic")) or ciclo_dir.name
    destino = Path(output_path) if output_path else ciclo_dir / f"submissao_{cycle_id}.zip"
    ensure_dir(destino.parent)

    referencias = ((ciclo.get("evidencias") or {}).get("artigos")) or []
    metadata = _metadata_dados(ciclo)

    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(paper_pdf, "paper.pdf")
        zf.write(paper_tex, "paper.tex")
        zf.writestr("bibliografia.bib", _bibtex(referencias))
        zf.writestr("codigo_experimento.py", _script_reproducao(ciclo))
        zf.writestr("metadata_dados.json", json.dumps(metadata, ensure_ascii=False, indent=2))
        zf.writestr("INSTRUCOES_SUBMISSAO.md", INSTRUCOES.format(cycle_id=cycle_id, topic=topic))

    with zipfile.ZipFile(destino) as zf:
        arquivos = zf.namelist()

    pacote = SubmissionPackage(
        path=str(destino),
        cycle_id=cycle_id,
        arquivos=arquivos,
        tamanho_bytes=destino.stat().st_size,
    )
    logger.info("Pacote de submissão gerado: %s (%s arquivos)", destino, len(arquivos))
    write_json(ciclo_dir / "submissao_metadata.json", pacote.as_dict())
    return pacote
