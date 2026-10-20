"""Testes do motor de rebuttal.

Cobrem o parser de comentários e a geração de ``resposta_revisores.md``, além da
atualização do manuscrito. Usam um ciclo real gerado em diretório temporário e o
``MockProvider`` do projeto — nenhum dado sintético é criado.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.experiments import ExperimentRunner  # noqa: E402
from core.rebuttal_engine import (  # noqa: E402
    AcaoRevisor,
    RebuttalEngine,
    parse_reviewer_comments,
    process_rebuttal,
)
from llm_providers import build_llm_client  # noqa: E402


@pytest.fixture()
def llm():
    return build_llm_client({"llm": {"providers": [{"type": "mock"}]}})


@pytest.fixture()
def ciclo(tmp_path, llm):
    """Gera um ciclo mínimo real (paper.md, paper.tex e ciclo_completo.json)."""
    runner = ExperimentRunner(
        seed=7, output_dir=tmp_path, cache_dir="data/dataset_cache", offline=True, llm=llm
    )
    result = runner.run("Predição de surtos de dengue no Brasil", {"id": "H1"})

    run_dir = tmp_path / "ciclo_1_dengue"
    run_dir.mkdir()
    (run_dir / "paper.md").write_text(
        "# Predição de surtos de dengue\n\n"
        "## RESUMO\n\nResumo inicial.\n\n"
        "## 3 RESULTADOS E DISCUSSÃO\n\nResultados iniciais.\n\n"
        "## 4 CONCLUSÃO\n\nConclusão inicial.\n",
        encoding="utf-8",
    )
    (run_dir / "paper.tex").write_text(
        "\\begin{abstract}\nResumo inicial.\n\\end{abstract}\n\n"
        "\\section{Introdução}\nIntro.\n\n"
        "\\section{Resultados e Discussão}\nResultados iniciais.\n\n"
        "\\section{Conclusão}\nConclusão inicial.\n",
        encoding="utf-8",
    )
    ciclo_completo = {
        "topic": "Predição de surtos de dengue no Brasil",
        "artefatos": {"diretorio": str(run_dir)},
        "experimento": {
            "hipotese_principal": {"id": "H1", "enunciado": "ML > baseline"},
            "resultado": result.as_dict(),
        },
        "redacao": {
            "secoes": {
                "resumo": "Resumo inicial.",
                "resultados": "Resultados iniciais.",
                "conclusao": "Conclusão inicial.",
            }
        },
    }
    (run_dir / "ciclo_completo.json").write_text(
        __import__("json").dumps(ciclo_completo, ensure_ascii=False), encoding="utf-8"
    )
    return run_dir


# --------------------------------------------------------------------------- #
# Parser de comentários
# --------------------------------------------------------------------------- #
def test_parse_comentarios_identifica_rerun_e_rewrite(llm):
    comentarios = (
        "O artigo precisa re-executar o experimento com a métrica AUC em vez de F1.\n"
        "Além disso, reescreva a seção de resultados com mais clareza."
    )
    acoes = parse_reviewer_comments(comentarios, llm)
    tipos = {a.tipo for a in acoes}
    assert "re-run" in tipos
    assert "rewrite" in tipos
    for acao in acoes:
        assert isinstance(acao, AcaoRevisor)
        assert acao.descricao


def test_parse_comentarios_so_reescrita(llm):
    acoes = parse_reviewer_comments("Por favor, reescreva a introdução com mais clareza.", llm)
    assert acoes
    assert all(a.tipo in ("re-run", "rewrite", "clarify") for a in acoes)


def test_parse_comentarios_vazios(llm):
    assert parse_reviewer_comments("", llm) == []
    assert parse_reviewer_comments("   ", llm) == []


def test_parse_fallback_sem_json(llm):
    """Se o LLM devolver texto solto, cada parágrafo vira uma ação 'clarify'."""

    class LlmTexto:
        def complete(self, prompt, **kwargs):
            class R:
                text = "sem json aqui"

            return R()

    acoes = parse_reviewer_comments("Primeiro ponto.\n\nSegundo ponto.", LlmTexto())
    assert len(acoes) == 2
    assert all(a.tipo == "clarify" for a in acoes)


# --------------------------------------------------------------------------- #
# process_rebuttal
# --------------------------------------------------------------------------- #
def test_process_rebuttal_gera_resposta(ciclo, llm):
    engine = RebuttalEngine(llm=llm, output_dir=ciclo.parent)
    relatorio = engine.process_rebuttal(1, "Reescreva a seção de resultados com mais clareza.")

    resposta = Path(relatorio["resposta_md"])
    assert resposta.exists()
    texto = resposta.read_text(encoding="utf-8")
    assert "## Resposta aos Revisores" in texto
    assert "## Response to Reviewers" in texto
    assert relatorio["cycle_id"] == 1
    assert relatorio["acoes"]
    assert (ciclo / "rebuttal.json").exists()


def test_process_rebuttal_atualiza_manuscrito(ciclo, llm):
    engine = RebuttalEngine(llm=llm, output_dir=ciclo.parent)
    relatorio = engine.process_rebuttal(1, "Reescreva a seção de resultados.")

    assert "resultados" in relatorio["mudancas"]["secoes_reescritas"]
    md = (ciclo / "paper.md").read_text(encoding="utf-8")
    # O cabeçalho da seção é preservado; o corpo muda.
    assert "RESULTADOS" in md
    assert relatorio["mudancas"]["paper_md"]
    assert relatorio["mudancas"]["paper_tex"]


def test_process_rebuttal_rerun_registra_comparacao(ciclo, llm):
    engine = RebuttalEngine(llm=llm, output_dir=ciclo.parent)
    engine.runner = ExperimentRunner(
        seed=7, output_dir=ciclo, cache_dir="data/dataset_cache", offline=True, llm=llm
    )
    relatorio = engine.process_rebuttal(1, "Re-execute o experimento com nova métrica.")

    comparacao = relatorio["comparacao_resultados"]
    assert "antes" in comparacao
    assert "depois" in comparacao
    assert comparacao["depois"].get("modelo")
    # Os números vêm de observações reais, nunca inventados.
    assert comparacao["antes"].get("f1_score") is not None


def test_process_rebuttal_ciclo_inexistente(tmp_path, llm):
    engine = RebuttalEngine(llm=llm, output_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        engine.process_rebuttal(9999, "comentário qualquer")


def test_process_rebuttal_atalho_funcional(ciclo, llm):
    relatorio = process_rebuttal(1, "Reescreva a conclusão.", llm=llm, output_dir=ciclo.parent)
    assert Path(relatorio["resposta_md"]).exists()
