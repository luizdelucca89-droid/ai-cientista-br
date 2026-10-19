"""Geração de paper (formatação ABNT e PDF)."""

from core.paper.abnt import ABNTFormatter
from core.paper.latex import LaTeXFormatter

__all__ = ["ABNTFormatter", "LaTeXFormatter"]
