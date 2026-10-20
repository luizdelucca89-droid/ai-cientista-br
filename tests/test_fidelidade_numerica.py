"""Testes do reparo de fidelidade numérica e do portão de submissão.

O LLM real inventou um número (``824`` amostras iniciais = 817 + 7) que não
existe em nenhum dado observado. Antes, isso apenas gerava um aviso e o número
ia para o PDF publicado. Estes testes garantem que o manuscrito é reparado antes
da formatação e que o pacote de submissão é rejeitado se ainda houver número sem
lastro.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.orchestrator.orchestrator import Orchestrator  # noqa: E402
from integrations.publishing import IntegrityError, generate_submission_package  # noqa: E402


class _RedatorFalso:
    """Redator que, ao ser chamado para reparo, devolve texto sem o número falso."""

    def __init__(self, texto_reparado: str) -> None:
        self.texto_reparado = texto_reparado
        self.chamadas = 0

    def run(self, prompt: str, task: str | None = None, **kwargs: object):
        self.chamadas += 1
        from core.agents.base import AgentResult

        return AgentResult(agent="redator", content=self.texto_reparado, raw=self.texto_reparado)


def _orquestrador_minimo() -> Orchestrator:
    orq = Orchestrator.__new__(Orchestrator)
    orq.agents = {}
    from core.config import Config

    orq.config = Config({"experiments": {"alpha": 0.05, "test_size": 0.3}, "seed": 42})
    return orq


def test_reparo_reescreve_secao_com_numero_sem_lastro():
    orq = _orquestrador_minimo()
    redator = _RedatorFalso("A base final reúne n = 817 amostras válidas, sem descartes.")
    orq.agents["redator"] = redator

    secoes = {
        "resumo": "Resumo sem números problemáticos.",
        "introducao": "Intro.",
        "metodologia": "A base original tinha 824 amostras, das quais 7 foram descartadas.",
        "resultados": "F1 = 0,8989.",
        "conclusao": "Conclusão.",
    }
    experiment = {
        "resultado": {"n_amostras": 817, "best_model": "floresta_aleatoria",
                      "metrics": {"floresta_aleatoria": {"f1_score": 0.8989}}},
        "hipotese_principal": "Hipótese.",
    }
    plan = {"lacuna": "Lacuna.", "hipoteses": [{"enunciado": "H1"}]}

    secoes_reparadas, fidelidade = orq._repair_numeric_fidelity(
        "dengue", secoes, plan, experiment, gathered=None
    )

    assert redator.chamadas >= 1, "o reparo deveria ter chamado o redator"
    assert fidelidade["numeros_sem_suporte"] == []
    assert fidelidade["ok"] is True
    assert "824" not in secoes_reparadas["metodologia"]


def test_reparo_nao_toca_secoes_fieis():
    orq = _orquestrador_minimo()
    redator = _RedatorFalso("nada")
    orq.agents["redator"] = redator

    secoes = {s: "texto sem números" for s in ("resumo", "introducao", "metodologia", "resultados", "conclusao")}
    experiment = {"resultado": {"n_amostras": 817}, "hipotese_principal": "H"}
    plan = {"lacuna": "L", "hipoteses": []}

    _, fidelidade = orq._repair_numeric_fidelity("dengue", secoes, plan, experiment, gathered=None)
    assert redator.chamadas == 0
    assert fidelidade["rodadas_reparo"] == 0


def _ciclo_com_numero_sem_lastro(tmp_path: Path, numeros: list[float]) -> Path:
    ciclo_dir = tmp_path / "ciclo_1_teste"
    ciclo_dir.mkdir(parents=True)
    (ciclo_dir / "paper.pdf").write_bytes(b"%PDF-1.4 fake")
    (ciclo_dir / "paper.tex").write_text("\\documentclass{article}", encoding="utf-8")
    ciclo = {
        "cycle_id": 1,
        "experimento": {"resultado": {"topic": "dengue", "dataset": {"tipo": "dados_reais"},
                                       "n_amostras": 817}},
        "evidencias": {"artigos": []},
        "fidelidade_pre_formatacao": {"numeros_sem_suporte": numeros},
        "validacao": {"fidelidade_numerica": {"numeros_sem_suporte": numeros}},
    }
    (ciclo_dir / "ciclo_completo.json").write_text(json.dumps(ciclo), encoding="utf-8")
    return ciclo_dir


def test_portao_rejeita_submissao_com_numero_sem_lastro(tmp_path):
    ciclo_dir = _ciclo_com_numero_sem_lastro(tmp_path, [824.0])
    with pytest.raises(IntegrityError, match="sem lastro"):
        generate_submission_package(ciclo_dir)


def test_portao_aceita_submissao_fiel(tmp_path):
    ciclo_dir = _ciclo_com_numero_sem_lastro(tmp_path, [])
    pacote = generate_submission_package(ciclo_dir)
    assert Path(pacote.path).exists()
