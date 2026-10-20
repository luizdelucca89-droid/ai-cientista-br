"""Testes do agente de auto-debug e do sandbox de execução.

Estratégia: código real, sem mocks de lógica interna. O "LLM" usado é o
``MockProvider`` do projeto, que devolve o script de referência do runner — o
mesmo comportamento que um LLM real teria para os erros previstos. Nenhum dado
sintético é criado: o sandbox só manipula as observações reais fornecidas.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.experiments import (  # noqa: E402
    ExperimentRunner,
    executar_com_auto_debug,
    sandbox_namespace,
)
from llm_providers import MockProvider, build_llm_client  # noqa: E402


@pytest.fixture()
def llm():
    return build_llm_client({"llm": {"providers": [{"type": "mock"}]}})


@pytest.fixture()
def base_numpy():
    return {"X": np.arange(20, dtype=float).reshape(10, 2), "y": np.array([0, 1] * 5)}


@pytest.fixture()
def base_runner():
    """Base com as mesmas chaves que o runner injeta no sandbox."""
    rng = np.random.default_rng(0)
    return {
        "X": rng.normal(size=(40, 3)),
        "y": rng.integers(0, 2, size=40),
        "seed": 42,
        "test_size": 0.3,
    }


# --------------------------------------------------------------------------- #
# Sandbox
# --------------------------------------------------------------------------- #
def test_sandbox_bloqueia_import_de_so():
    """O código do experimento não pode importar módulos de sistema."""
    res = executar_com_auto_debug("import os\nos.system('echo oi')\n", llm=None, base={})
    assert not res.sucesso
    assert "Import bloqueado" in res.erro


def test_sandbox_bloqueia_open_e_exec():
    for codigo in ("open('/etc/passwd').read()\n", "exec('x = 1')\n", "eval('1+1')\n"):
        res = executar_com_auto_debug(codigo, llm=None, base={})
        assert not res.sucesso, codigo


def test_sandbox_permite_matematica_e_numpy(base_numpy):
    res = executar_com_auto_debug(
        "import numpy as np\nresultado = float(np.mean(X))\n",
        llm=None,
        base=base_numpy,
    )
    assert res.sucesso
    assert res.namespace["resultado"] == pytest.approx(float(np.mean(base_numpy["X"])))


def test_sandbox_namespace_expoe_import_seguro():
    ns = sandbox_namespace()
    assert ns["__builtins__"]["__import__"].__name__ == "_import_seguro"
    assert "open" not in ns["__builtins__"]


# --------------------------------------------------------------------------- #
# Correção automática
# --------------------------------------------------------------------------- #
def test_auto_debug_corrige_codigo_com_erro(llm, base_runner):
    """Código com bug proposital é corrigido pelo LLM e roda com sucesso."""
    quebrado = (
        "import numpy as np\n"
        "modelos = {'x': 1/0}\n"  # ZeroDivisionError proposital
    )
    res = executar_com_auto_debug(
        quebrado,
        llm=llm,
        objetivo="treinar modelos sobre X e y",
        base=base_runner,
        max_retries=3,
    )
    assert res.sucesso, res.erro
    assert res.tentativas >= 2
    assert res.correcoes
    assert "modelos" in res.namespace


def test_auto_debug_sem_llm_falha_sem_inventar_sucesso(base_numpy):
    res = executar_com_auto_debug("x = 1/0\n", llm=None, base=base_numpy)
    assert not res.sucesso
    assert res.status_experimento == "falha_apos_debug"
    assert res.tentativas == 1
    assert "ZeroDivisionError" in res.erro


def test_auto_debug_respeita_max_retries(llm, base_runner):
    """O limite de tentativas é respeitado e a correção roda dentro dele."""
    res = executar_com_auto_debug(
        "x = 1/0\n",
        llm=llm,
        base=base_runner,
        max_retries=2,
    )
    assert res.sucesso
    assert res.tentativas <= 2


def test_auto_debug_codigo_vazio():
    res = executar_com_auto_debug("", llm=None, base={})
    assert not res.sucesso
    assert res.status_experimento == "falha_apos_debug"


def test_limpa_cercas_markdown(llm):
    """O código devolvido pelo LLM pode vir com cercas markdown; devem ser removidas."""
    from core.experiments.auto_debugger import _limpar_codigo

    assert _limpar_codigo("```python\nx = 1\n```") == "x = 1"
    assert _limpar_codigo("x = 1") == "x = 1"


# --------------------------------------------------------------------------- #
# Integração com o runner
# --------------------------------------------------------------------------- #
def test_runner_executa_codigo_de_experimento(tmp_path):
    """O runner roda o código de verdade e registra o status e o script."""
    runner = ExperimentRunner(
        seed=7,
        output_dir=tmp_path,
        cache_dir="data/dataset_cache",
        offline=True,
        llm=build_llm_client({"llm": {"providers": [{"type": "mock"}]}}),
    )
    result = runner.run("Predição de surtos de dengue no Brasil", {"id": "H1"})

    assert result.status_experimento == "ok"
    assert result.best_model
    assert "modelos" in result.codigo
    assert result.auto_debug["tentativas"] >= 1
    assert result.as_dict()["status_experimento"] == "ok"


def test_runner_auto_debug_recupera_codigo_quebrado(tmp_path):
    """Código com NameError é corrigido pelo auto-debug e o experimento roda."""
    runner = ExperimentRunner(
        seed=7,
        output_dir=tmp_path,
        cache_dir="data/dataset_cache",
        offline=True,
        llm=build_llm_client({"llm": {"providers": [{"type": "mock"}]}}),
    )
    quebrado = (
        "from sklearn.linear_model import LogisticRegression\n"
        "modelo = LogisticRegression(max_iter=1000, random_state=seed)\n"
        "modelo.fit(X_treino, y)\n"  # NameError: X_treino não existe
        "modelos = {'regressao_logistica': modelo}\n"
    )
    result = runner.run("Predição de surtos de dengue no Brasil", {"id": "H1"}, codigo=quebrado)

    assert result.status_experimento == "ok"
    assert result.auto_debug["correcoes_aplicadas"] >= 1
    assert "X_treino" not in result.codigo


def test_runner_falha_apos_debug_nao_levanta(tmp_path):
    """Esgotado o auto-debug, o runner devolve falha registrada e não levanta.

    O ciclo precisa continuar para que o artigo reporte a dificuldade
    metodológica; por isso o runner não propaga a exceção.
    """
    runner = ExperimentRunner(
        seed=7,
        output_dir=tmp_path,
        cache_dir="data/dataset_cache",
        offline=True,
        llm=None,  # sem LLM não há correção
    )
    result = runner.run("Predição de surtos de dengue no Brasil", {"id": "H1"}, codigo="x = 1/0\n")

    assert result.status_experimento == "falha_apos_debug"
    assert result.metrics == {}
    assert result.best_model == ""
    assert "ZeroDivisionError" in result.auto_debug["erro_final"]
    # Procedência dos dados reais é preservada mesmo na falha.
    assert result.dataset["tipo"] == "dados_reais"


def test_mock_provider_tem_tarefa_auto_debug():
    resposta = MockProvider().complete("corrija", task="auto_debug")
    assert "modelos" in resposta
    assert "StandardScaler" in resposta
