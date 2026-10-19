"""Formatação de referências e do manuscrito segundo a ABNT.

Implementa as regras mais usadas de NBR 6023 (referências) e NBR 6022/6028
(artigo científico), produzindo um documento em Markdown e, opcionalmente, PDF.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.utils import ensure_dir, setup_logger

logger = setup_logger(__name__)


@dataclass
class ABNTFormatter:
    """Gera referências e estrutura o artigo conforme a ABNT."""

    max_references: int = 12
    reference_style: str = "ABNT NBR 6023:2018"
    access_date: str | None = None

    def __post_init__(self) -> None:
        if not self.access_date:
            from datetime import date

            self.access_date = date.today().strftime("%d %b. %Y")

    # ----------------------------------------------------------- references
    def format_reference(self, article: dict[str, Any], index: int | None = None) -> str:
        """Formata uma referência no padrão autor-data da ABNT."""
        authors = self._format_authors(article.get("autores") or [])
        title = (article.get("titulo") or "Título não informado").strip().rstrip(".")
        year = article.get("ano") or "s.d."
        journal = (article.get("periodico") or "").strip()
        doi = article.get("doi")
        url = article.get("url")
        parts = [f"{authors}. {title}." if authors else f"{title}."]
        if journal:
            parts.append(f" {journal},")
        parts.append(f" {year}.")
        if doi:
            parts.append(f" DOI: {doi}.")
        elif url:
            parts.append(f" Disponível em: {url}.")
        parts.append(f" Acesso em: {self.access_date}.")
        ref = "".join(parts)
        ref = re.sub(r"\s+", " ", ref).strip()
        if index is not None:
            return f"[{index}] {ref}"
        return ref

    @staticmethod
    def _format_authors(authors: list[str]) -> str:
        """Sobrenome em caixa alta seguido do prenome, separados por ';'."""
        formatted = []
        for full in authors[:6]:
            full = full.strip()
            if not full:
                continue
            pieces = full.split()
            if len(pieces) == 1:
                formatted.append(pieces[0].upper())
                continue
            last = pieces[-1].upper()
            first = " ".join(pieces[:-1])
            formatted.append(f"{last}, {first}")
        if len(authors) > 6:
            formatted.append("et al.")
        return "; ".join(formatted)

    def format_reference_list(self, articles: list[dict[str, Any]]) -> list[str]:
        return [self.format_reference(a, i) for i, a in enumerate(articles[: self.max_references], start=1)]

    # ------------------------------------------------------------- document
    def render_markdown(
        self,
        topic: str,
        sections: dict[str, str],
        references: list[str],
        metadata: dict[str, Any],
        figures: list[str] | None = None,
    ) -> str:
        """Monta o artigo completo em Markdown com estrutura ABNT."""
        title = metadata.get("titulo") or topic
        authors = metadata.get("autores") or ["AI Cientista Brasileiro (sistema autônomo)"]
        lines: list[str] = []
        lines.append(f"# {title}")
        lines.append("")
        lines.append(f"**Autoria:** {'; '.join(authors)}")
        lines.append(f"**Data:** {metadata.get('data', '')}")
        lines.append(f"**Norma:** {self.reference_style}")
        lines.append("")
        lines.append("## RESUMO")
        lines.append(sections.get("resumo", "").strip())
        lines.append("")
        lines.append("**Palavras-chave:** " + "; ".join(metadata.get("palavras_chave", [])))
        lines.append("")
        lines.append("## ABSTRACT")
        lines.append(sections.get("abstract", "").strip())
        lines.append("")
        lines.append("**Keywords:** " + "; ".join(metadata.get("keywords", [])))
        lines.append("")
        for key, heading in (
            ("introducao", "1 INTRODUÇÃO"),
            ("metodologia", "2 METODOLOGIA"),
            ("resultados", "3 RESULTADOS E DISCUSSÃO"),
            ("conclusao", "4 CONCLUSÃO"),
        ):
            lines.append(f"## {heading}")
            lines.append(sections.get(key, "").strip())
            lines.append("")
        if figures:
            lines.append("## FIGURAS")
            for i, fig in enumerate(figures, start=1):
                lines.append(f"**Figura {i}** — {fig}")
            lines.append("")
        lines.append("## DECLARAÇÃO DE USO DE IA")
        lines.append(
            "Este manuscrito foi produzido com auxílio de um sistema multi-agente de IA "
            "(AI Cientista Brasileiro). Todos os resultados numéricos são provenientes de "
            "execução real de código, e o texto deve ser revisado por pesquisadores humanos "
            "antes de qualquer submissão ou publicação."
        )
        lines.append("")
        lines.append("## REFERÊNCIAS")
        for ref in references:
            lines.append(ref)
            lines.append("")
        return "\n".join(lines).strip() + "\n"

    # ------------------------------------------------------------------ pdf
    def render_pdf(self, markdown_text: str, output_path: Any) -> str | None:
        """Gera um PDF a partir do Markdown (requer ``fpdf2``).

        Usa as fontes core Helvetica/Times, restritas a Latin-1; caracteres fora
        desse conjunto são transliterados para manter o PDF legível.
        """
        try:
            from fpdf import FPDF
        except ImportError:
            logger.warning("fpdf2 indisponível; PDF não será gerado")
            return None

        pdf = FPDF()
        pdf.set_margins(20, 15, 20)
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Times", size=11)

        for raw_line in markdown_text.splitlines():
            line = _to_latin1(raw_line.rstrip())
            if not line.strip():
                pdf.ln(4)
                continue
            stripped = line.strip()
            try:
                if stripped.startswith("# "):
                    pdf.set_font("Times", "B", 16)
                    pdf.multi_cell(0, 8, stripped[2:], new_x="LMARGIN", new_y="NEXT")
                    pdf.set_font("Times", size=11)
                elif stripped.startswith("## "):
                    pdf.set_font("Times", "B", 13)
                    pdf.multi_cell(0, 7, stripped[3:], new_x="LMARGIN", new_y="NEXT")
                    pdf.set_font("Times", size=11)
                elif stripped.startswith("**") and stripped.endswith("**") and len(stripped) > 4:
                    pdf.set_font("Times", "B", 11)
                    pdf.multi_cell(0, 6, stripped.strip("*"), new_x="LMARGIN", new_y="NEXT")
                    pdf.set_font("Times", size=11)
                else:
                    pdf.multi_cell(0, 6, stripped, new_x="LMARGIN", new_y="NEXT")
            except Exception as exc:  # noqa: BLE001 - PDF não deve derrubar o ciclo
                logger.warning("Falha ao renderizar linha no PDF: %s", exc)
                pdf.set_font("Times", size=11)
                try:
                    pdf.set_x(pdf.l_margin)
                    pdf.multi_cell(0, 6, stripped[:180], new_x="LMARGIN", new_y="NEXT")
                except Exception:  # noqa: BLE001
                    pdf.ln(6)

        out = Path(output_path)
        ensure_dir(out.parent)
        pdf.output(str(out))
        return str(out)


_TRANSLITERATIONS = {
    "—": "-", "–": "-", "“": '"', "”": '"', "‘": "'", "’": "'", "…": "...",
    "→": "->", "≤": "<=", "≥": ">=", "×": "x", "•": "-", "º": "o", "ª": "a",
}


def _to_latin1(text: str) -> str:
    """Converte texto para Latin-1 preservando acentos portugueses."""
    for src, dst in _TRANSLITERATIONS.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", "replace").decode("latin-1")
