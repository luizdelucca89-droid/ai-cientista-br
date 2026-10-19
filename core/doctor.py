"""Diagnóstico do ambiente, no espírito do ``agsci doctor`` do AI Scientist v3.

Verifica, sem executar o pipeline: versão do Python, dependências opcionais,
gravabilidade do diretório de saída, conectividade com as fontes abertas e
quais providers de LLM estão realmente disponíveis. Serve para diagnosticar
falhas antes de uma execução longa.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from core.utils import ensure_dir, setup_logger

logger = setup_logger(__name__)

REQUIRED = ["requests", "fpdf"]
OPTIONAL = ["numpy", "sklearn", "scipy", "pandas", "selenium", "matplotlib"]

# (nome, URL) das fontes abertas consultadas pelo pipeline.
SOURCES = [
    ("Crossref", "https://api.crossref.org/works?rows=1"),
    ("OpenAlex", "https://api.openalex.org/works?per-page=1"),
    ("SciELO ArticleMeta", "https://articlemeta.scielo.org/api/v1/info/"),
    ("IBGE", "https://servicodados.ibge.gov.br/api/v1/localidades/estados"),
    ("Banco Central", "https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados/ultimos/1?formato=json"),
]


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    critical: bool = True


@dataclass
class Doctor:
    """Executa uma bateria de verificações e resume o resultado."""

    checks: list[Check] = field(default_factory=list)

    def run(self, output_dir: str = "outputs", check_network: bool = True, timeout: float = 8.0) -> dict[str, Any]:
        self._check_python()
        self._check_modules()
        self._check_output_dir(output_dir)
        self._check_providers()
        if check_network:
            self._check_sources(timeout)

        critical_failures = [c for c in self.checks if c.critical and not c.ok]
        return {
            "checks": [c.__dict__ for c in self.checks],
            "ok": not critical_failures,
            "falhas_criticas": [c.name for c in critical_failures],
        }

    # ------------------------------------------------------------- checks
    def _check_python(self) -> None:
        ok = sys.version_info >= (3, 11)
        self.checks.append(
            Check("python>=3.11", ok, f"versão atual {sys.version.split()[0]}")
        )

    def _check_modules(self) -> None:
        for module in REQUIRED:
            found = importlib.util.find_spec(module) is not None
            self.checks.append(Check(f"dependência obrigatória: {module}", found))
        for module in OPTIONAL:
            found = importlib.util.find_spec(module) is not None
            self.checks.append(
                Check(
                    f"dependência opcional: {module}",
                    found,
                    "ausente: alguns recursos ficam limitados" if not found else "",
                    critical=False,
                )
            )

    def _check_output_dir(self, output_dir: str) -> None:
        try:
            path = ensure_dir(output_dir)
            probe = path / ".doctor_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            self.checks.append(Check(f"diretório de saída gravável: {output_dir}", True))
        except OSError as exc:
            self.checks.append(Check(f"diretório de saída gravável: {output_dir}", False, str(exc)))

        if shutil.which("pdflatex"):
            self.checks.append(Check("pdflatex (artigo LaTeX)", True, "paper_latex.pdf será compilado"))
        else:
            self.checks.append(
                Check(
                    "pdflatex (artigo LaTeX)",
                    False,
                    "ausente: gera apenas paper.tex (apt-get install texlive-latex-base texlive-lang-portuguese)",
                    critical=False,
                )
            )

    def _check_providers(self) -> None:
        try:
            from llm_providers import build_llm_client

            client = build_llm_client({})
            available = [p.name for p in client.providers if p.available()]
            self.checks.append(
                Check(
                    "providers de LLM disponíveis",
                    bool(available),
                    ", ".join(available) or "nenhum",
                )
            )
            client.close()
        except Exception as exc:  # noqa: BLE001 - diagnóstico não deve quebrar
            self.checks.append(Check("providers de LLM disponíveis", False, str(exc)))

    def _check_sources(self, timeout: float) -> None:
        for name, url in SOURCES:
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "ai-cientista-br/1.0"})
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    ok = 200 <= response.status < 400
                self.checks.append(Check(f"fonte aberta: {name}", ok, f"HTTP {response.status}"))
            except Exception as exc:  # noqa: BLE001 - rede é instável por natureza
                self.checks.append(
                    Check(f"fonte aberta: {name}", False, f"{type(exc).__name__}", critical=False)
                )


def render(result: dict[str, Any]) -> str:
    """Formata o resultado do diagnóstico para o terminal."""
    lines = ["DIAGNÓSTICO DO AMBIENTE", "=" * 60]
    for check in result["checks"]:
        mark = "OK  " if check["ok"] else ("FALHA" if check["critical"] else "AVISO")
        detail = f" ({check['detail']})" if check["detail"] else ""
        lines.append(f"[{mark}] {check['name']}{detail}")
    lines.append("=" * 60)
    if result["ok"]:
        lines.append("Ambiente pronto para executar o pipeline.")
    else:
        lines.append("Falhas críticas: " + ", ".join(result["falhas_criticas"]))
    return "\n".join(lines)
