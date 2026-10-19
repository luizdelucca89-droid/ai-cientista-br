"""Geração de paper (formatação ABNT e PDF)."""

from core.paper.abnt import ABNTFormatter
from core.paper.headings import strip_ai_declaration_blocks, strip_duplicate_heading
from core.paper.latex import LaTeXFormatter

__all__ = [
    "ABNTFormatter",
    "LaTeXFormatter",
    "strip_ai_declaration_blocks",
    "strip_duplicate_heading",
]
