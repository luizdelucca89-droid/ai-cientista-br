"""Geração de artigo em LaTeX, com compilação opcional para PDF.

Portado do AI Scientist v1 (``--writeup latex``), que usa um template ``.tex``
com ``references.bib`` e compila com ``pdflatex``. Aqui o template segue o
layout de artigo científico brasileiro (fonte Times, coluna única, resumo e
abstract), mantendo o mesmo princípio: o manuscrito é um artefato LaTeX
editável, não apenas um PDF final.

A compilação só é tentada se ``pdflatex`` estiver no ``PATH``; caso contrário, o
``.tex`` é gravado e o pipeline segue normalmente.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.paper.headings import (
    renumber_subsections,
    strip_ai_declaration_blocks,
    strip_duplicate_heading,
)
from core.utils import ensure_dir, setup_logger

logger = setup_logger(__name__)

TEMPLATE = r"""\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[brazil]{babel}
\usepackage{times}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\geometry{left=3cm,right=2cm,top=3cm,bottom=2cm}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}

\title{%(title)s}
\author{%(authors)s}
\date{%(date)s}

\begin{document}
\maketitle

\begin{abstract}
\noindent %(resumo)s

\vspace{0.5em}
\noindent\textbf{Palavras-chave:} %(palavras_chave)s
\end{abstract}

\begin{center}
\textbf{Abstract}
\end{center}
\noindent %(abstract)s

\vspace{0.5em}
\noindent\textbf{Keywords:} %(keywords)s

%(body)s

\section*{Declaração de uso de IA}
%(declaracao_ia)s

\begin{thebibliography}{99}
%(referencias)s
\end{thebibliography}

\end{document}
"""


@dataclass
class LaTeXFormatter:
    """Converte as seções do manuscrito em um documento LaTeX compilável."""

    document_class: str = "article"
    timeout_seconds: int = 120

    def available(self) -> bool:
        """Informa se ``pdflatex`` está disponível no ambiente."""
        return shutil.which("pdflatex") is not None

    def render_tex(
        self,
        topic: str,
        sections: dict[str, str],
        references: list[dict[str, Any]],
        metadata: dict[str, Any],
        figures: list[str] | None = None,
    ) -> str:
        """Monta o documento LaTeX completo."""
        body_parts: list[str] = []
        for numero, key, heading in (
            (1, "introducao", "Introdução"),
            (2, "metodologia", "Metodologia"),
            (3, "resultados", "Resultados e Discussão"),
            (4, "conclusao", "Conclusão"),
        ):
            # Remover o título do LLM antes de escapar: depois do escape os
            # acentos viram comandos e a comparação com o título esperado falha.
            body = strip_duplicate_heading(sections.get(key, "").strip(), heading)
            body = strip_ai_declaration_blocks(body)
            body = renumber_subsections(body, numero)
            content = self._escape(body)
            if content:
                body_parts.append(f"\\section{{{heading}}}\n{content}\n")
        if figures:
            figure_lines = ["\\section*{Figuras}", "\\begin{itemize}"]
            for i, fig in enumerate(figures, start=1):
                figure_lines.append(f"  \\item Figura {i}: {self._escape(fig)}")
            figure_lines.extend(["\\end{itemize}"])
            body_parts.append("\n".join(figure_lines))

        return TEMPLATE % {
            "title": self._escape(metadata.get("titulo") or topic),
            "authors": self._escape("; ".join(metadata.get("autores") or ["AI Cientista Brasileiro"])),
            "date": self._escape(str(metadata.get("data", ""))),
            "resumo": self._escape(
                strip_duplicate_heading(sections.get("resumo", "").strip(), "Resumo")
            ),
            "abstract": self._escape(
                strip_duplicate_heading(sections.get("abstract", "").strip(), "Abstract")
            ),
            "palavras_chave": self._escape("; ".join(metadata.get("palavras_chave", []))),
            "keywords": self._escape("; ".join(metadata.get("keywords", []))),
            "body": "\n".join(body_parts),
            "declaracao_ia": self._escape(
                "Este manuscrito foi produzido com auxílio de um sistema multi-agente de IA "
                "(AI Cientista Brasileiro). Os resultados numéricos provêm de execução real de código "
                "e o texto deve ser revisado por pesquisadores humanos antes de qualquer publicação."
            ),
            "referencias": self._render_bibliography(references),
        }

    def compile_pdf(self, tex_path: Path, output_dir: Path | None = None) -> str | None:
        """Compila o ``.tex`` com ``pdflatex``; devolve o caminho do PDF ou ``None``.

        A compilação ocorre em um diretório temporário para não sobrescrever o
        ``paper.pdf`` produzido pelo formatador ABNT; o resultado é copiado para
        ``paper_latex.pdf``. Nunca levanta exceção: qualquer falha (binário
        ausente, erro de LaTeX, timeout, saída não-UTF-8) apenas registra aviso e
        devolve ``None``, para que o ciclo continue com os demais artefatos.
        """
        if not self.available():
            logger.info("pdflatex não encontrado; mantendo apenas o .tex")
            return None
        workdir = Path(output_dir or tex_path.parent)
        ensure_dir(workdir)
        try:
            with tempfile.TemporaryDirectory() as build_dir:
                # pdflatex escreve mensagens em Latin-1 sob o babel brasileiro;
                # sem errors="replace" a decodificação UTF-8 padrão quebra tudo.
                process = subprocess.run(
                    [
                        "pdflatex",
                        "-interaction=nonstopmode",
                        "-halt-on-error",
                        f"-output-directory={build_dir}",
                        str(tex_path),
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout_seconds,
                    check=False,
                )
                built = Path(build_dir) / (tex_path.stem + ".pdf")
                if process.returncode == 0 and built.exists():
                    target = workdir / (tex_path.stem + "_latex.pdf")
                    shutil.copyfile(built, target)
                    return str(target)
                tail = (process.stdout or process.stderr or "").strip().splitlines()[-6:]
                logger.warning("pdflatex falhou: %s", " | ".join(tail))
                return None
        except subprocess.TimeoutExpired:
            logger.warning("pdflatex excedeu o tempo limite de %ss", self.timeout_seconds)
            return None
        except Exception as exc:  # noqa: BLE001 - artefato opcional nunca derruba o ciclo
            logger.warning("Erro ao compilar LaTeX: %s: %s", type(exc).__name__, exc)
            return None

    def write(
        self,
        topic: str,
        sections: dict[str, str],
        references: list[dict[str, Any]],
        metadata: dict[str, Any],
        output_dir: Path,
        figures: list[str] | None = None,
    ) -> dict[str, str | None]:
        """Grava ``paper.tex`` e, quando possível, compila ``paper_latex.pdf``.

        O PDF do LaTeX usa nome distinto de ``paper.pdf`` (gerado pelo formatador
        ABNT) para que os dois artefatos coexistam em vez de se sobrescreverem.
        """
        ensure_dir(output_dir)
        tex_path = output_dir / "paper.tex"
        tex_path.write_text(self.render_tex(topic, sections, references, metadata, figures), encoding="utf-8")
        pdf_path = self.compile_pdf(tex_path, output_dir)
        return {"tex": str(tex_path), "pdf": pdf_path}

    # ------------------------------------------------------------- internals
    def _render_bibliography(self, references: list[dict[str, Any]]) -> str:
        """Converte referências estruturadas em entradas ``\\bibitem``."""
        if not references:
            return "\\bibitem{sem-referencias} Nenhuma referência recuperada nas fontes consultadas."
        items: list[str] = []
        for i, ref in enumerate(references, start=1):
            if isinstance(ref, str):
                items.append(f"\\bibitem{{ref{i}}} {self._escape(ref)}")
                continue
            authors = "; ".join(ref.get("autores") or []) or "Autoria não informada"
            title = ref.get("titulo") or "Título não informado"
            year = ref.get("ano") or "s.d."
            journal = ref.get("periodico") or ""
            doi = ref.get("doi")
            entry = f"{authors}. \\textbf{{{self._escape(title)}}}. {self._escape(journal)} {year}."
            if doi:
                entry += f" DOI: \\url{{https://doi.org/{self._escape(str(doi))}}}."
            items.append(f"\\bibitem{{ref{i}}} {entry}")
        return "\n".join(items)

    @staticmethod
    def _escape(text: str) -> str:
        """Escapa caracteres reservados do LaTeX preservando acentuação UTF-8.

        LLMs reais emitem espaços e hífens "tipográficos" (U+202F, U+00A0,
        U+2011) e letras gregas que o ``inputenc``/``fontenc`` não mapeia, e que
        abortam o ``pdflatex``. Esses casos viram equivalentes ASCII ou comandos
        em modo matemático. Os comandos gerados aqui são inseridos por
        placeholder para não serem re-escapados na etapa seguinte.
        """
        if not text:
            return ""
        placeholders: dict[str, str] = {}

        def stash(latex: str) -> str:
            key = f"\x00{len(placeholders)}\x00"
            placeholders[key] = latex
            return key

        typography = {
            "\u00a0": " ",   # NBSP
            "\u202f": " ",   # narrow no-break space
            "\u2009": " ",   # thin space
            "\u200b": "",    # zero-width space
            "\u2011": "-",   # non-breaking hyphen
            "\u2012": "-",
            "\u2013": "--",
            "\u2014": "---",
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": "``",
            "\u201d": "''",
            "\u2026": stash(r"\ldots{}"),
        }
        greek = {
            "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
            "θ": "theta", "λ": "lambda", "μ": "mu", "π": "pi", "ρ": "rho",
            "σ": "sigma", "τ": "tau", "φ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega",
            "Δ": "Delta", "Σ": "Sigma", "Ω": "Omega", "Γ": "Gamma", "Λ": "Lambda", "Π": "Pi",
        }
        for char, name in greek.items():
            typography[char] = stash(f"${'\\' + name}$")

        # Sub/sobrescritos Unicode (ex.: CO₂, km²) usados pelo LLM em unidades.
        subscript = "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
        superscript = "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
        for index, char in enumerate(subscript):
            typography[char] = stash(f"$_{'0123456789+-=()'[index]}$")
        for index, char in enumerate(superscript):
            typography[char] = stash(f"$^{'0123456789+-=()'[index]}$")

        text = "".join(typography.get(ch, ch) for ch in text)

        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        text = "".join(replacements.get(ch, ch) for ch in text)

        # Rede de seguranca: o conjunto inputenc+fontenc so cobre ASCII,
        # Latin-1 e Latin Extended-A. Qualquer outro simbolo (emoji, setas,
        # caracteres de controle vindos do LLM) abortaria o pdflatex, entao e
        # descartado. Os placeholders usam \x00 e passam intactos.
        text = "".join(
            ch for ch in text
            if ch == "\x00" or ord(ch) < 0x0180
        )

        for key, latex in placeholders.items():
            text = text.replace(key, latex)
        text = re.sub(r"\n{2,}", "\n\n", text)
        return text
