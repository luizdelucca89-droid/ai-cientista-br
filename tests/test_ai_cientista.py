"""Testes automatizados do AI Cientista Brasileiro.

Estratégia: nenhum mock de lógica interna. Testamos código real:

* memória em SQLite (arquivo temporário real);
* runner experimental (treina modelos de verdade e roda testes estatísticos);
* formatador ABNT (gera Markdown e PDF reais);
* cliente LLM (cache em SQLite + fallback reais);
* ciclo completo com coleta **offline** (sem rede, determinístico);
* integrações reais, marcadas com ``@pytest.mark.network``.

Rodar tudo::

    pytest -q

Somente offline (rápido, sem rede)::

    pytest -q -m "not network"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.config import Config  # noqa: E402
from core.experiments import (  # noqa: E402
    ExperimentRunner,
    RealDataUnavailableError,
    build_agriculture_dataset,
    build_dengue_outbreak_dataset,
    build_deforestation_dataset,
    build_education_dataset,
    build_real_dataset,
)
from core.memory import MemorySystem  # noqa: E402
from core.orchestrator import Orchestrator  # noqa: E402
from core.paper import ABNTFormatter  # noqa: E402
from core.utils import extract_json_block, slugify, truncate  # noqa: E402
from integrations import BrazilianDataAggregator, CrossrefAPI  # noqa: E402
from llm_providers import MockProvider, build_llm_client  # noqa: E402


# --------------------------------------------------------------------------- #
# Utilitários
# --------------------------------------------------------------------------- #
def test_slugify_removes_accents_and_punctuation():
    assert slugify("Predição de surtos de dengue!") == "predicao-de-surtos-de-dengue"
    assert slugify("") == "sem-titulo"


def test_truncate_preserves_prefix():
    assert truncate("abcdefghij", 5) == "ab..."
    assert truncate("abc", 10) == "abc"


def test_extract_json_block_handles_fenced_and_bare():
    assert extract_json_block('texto ```json\n{"a": 1}\n``` fim') == {"a": 1}
    assert extract_json_block('antes {"b": [1, 2]} depois') == {"b": [1, 2]}
    assert extract_json_block("sem json aqui") is None


# --------------------------------------------------------------------------- #
# Memória (SQLite real)
# --------------------------------------------------------------------------- #
def test_memory_persists_cycle_and_knowledge(tmp_path):
    memory = MemorySystem(path=tmp_path / "mem.sqlite3")
    cycle_id = memory.start_cycle("dengue")
    memory.remember("dengue", "hipotese", "H1: modelos ML superam baseline")
    memory.finish_cycle(cycle_id, "concluido", {"ok": True}, score=0.8)

    cycle = memory.get_cycle(cycle_id)
    assert cycle["status"] == "concluido"
    assert cycle["score"] == 0.8
    assert cycle["payload"] == {"ok": True}
    assert memory.is_duplicate_topic("dengue") is True
    assert memory.is_duplicate_topic("outro") is False
    assert memory.recall("dengue", "hipotese")[0]["content"].startswith("H1")
    assert memory.stats() == {"ciclos": 1, "concluidos": 1, "conhecimento": 1}
    memory.close()

    reopened = MemorySystem(path=tmp_path / "mem.sqlite3")
    assert reopened.get_cycle(cycle_id) is not None
    reopened.close()


# --------------------------------------------------------------------------- #
# Experimento (treina modelos e roda testes estatísticos de verdade)
# --------------------------------------------------------------------------- #
def test_synthetic_generator_was_removed():
    """O sistema não fabrica dados: o gerador sintético não existe mais."""
    import core.experiments as experiments

    assert not hasattr(experiments, "generate_synthetic_dataset")
    assert not hasattr(experiments.runner, "generate_synthetic_dataset")


def test_experiment_runner_raises_without_real_source(tmp_path):
    """Sem fonte pública de observações, o runner falha explicitamente.

    O runner não cai para uma base sintética; ele levanta
    ``RealDataUnavailableError`` para que o ciclo registre a ausência de dados em
    vez de publicar números inventados.
    """
    runner = ExperimentRunner(seed=7, output_dir=tmp_path)
    with pytest.raises(RealDataUnavailableError):
        runner.run("previsao de churn em telecomunicacoes", {"id": "H1", "enunciado": "ML > baseline"})


def test_experiment_runner_runs_on_real_dataset(tmp_path):
    """Com fonte pública real, o runner treina e produz métricas verificáveis.

    Usa o cache versionado de observações reais (``data/dataset_cache``) em modo
    offline, então o teste é determinístico e não depende de rede.
    """
    runner = ExperimentRunner(seed=7, output_dir=tmp_path, cache_dir="data/dataset_cache", offline=True)
    result = runner.run("Predição de surtos de dengue no Brasil", {"id": "H1", "enunciado": "ML > baseline"})

    assert result.dataset["tipo"] == "dados_reais"
    assert set(result.metrics) == {
        "baseline_estatistico",
        "regressao_logistica",
        "floresta_aleatoria",
        "gradient_boosting",
    }
    for name, metrics in result.metrics.items():
        assert 0.0 <= metrics["f1_score"] <= 1.0, name
        assert metrics["acuracia_ic95_inf"] <= metrics["acuracia_ic95_sup"]
    assert result.best_model in result.metrics
    assert any("t_pareado" in key for key in result.statistical_tests)
    assert any("wilcoxon" in key for key in result.statistical_tests)
    for test in result.statistical_tests.values():
        assert 0.0 <= test["p_valor"] <= 1.0
    assert result.duration_s > 0
    assert (tmp_path / "fig_f1_modelos.png").exists()


def test_experiment_runner_can_be_disabled(tmp_path):
    runner = ExperimentRunner(allow_code_execution=False, output_dir=tmp_path)
    with pytest.raises(RuntimeError, match="desabilitada"):
        runner.run("x", {"id": "H1"})


# --------------------------------------------------------------------------- #
# ABNT
# --------------------------------------------------------------------------- #
def test_abnt_reference_formatting():
    formatter = ABNTFormatter(access_date="19 out. 2026")
    article = {
        "autores": ["Maria Silva Santos", "João Pereira"],
        "titulo": "Modelos preditivos para dengue",
        "ano": 2024,
        "periodico": "Cadernos de Saúde Pública",
        "doi": "10.1590/xyz",
    }
    ref = formatter.format_reference(article, index=1)
    assert ref.startswith("[1] SANTOS, Maria Silva; PEREIRA, João.")
    assert "Cadernos de Saúde Pública, 2024." in ref
    assert "DOI: 10.1590/xyz." in ref
    assert "Acesso em: 19 out. 2026." in ref


def test_abnt_markdown_and_pdf(tmp_path):
    formatter = ABNTFormatter(access_date="19 out. 2026")
    sections = {
        "resumo": "Resumo do trabalho.",
        "abstract": "Abstract of the work.",
        "introducao": "Introdução.",
        "metodologia": "Metodologia.",
        "resultados": "Resultados com acentuação: ção, ã, é.",
        "conclusao": "Conclusão.",
    }
    markdown = formatter.render_markdown(
        topic="Dengue e clima",
        sections=sections,
        references=["[1] SILVA, M. Título. Periódico, 2024."],
        metadata={"titulo": "Dengue e clima", "data": "2026-10-19", "palavras_chave": ["dengue"], "keywords": ["dengue"]},
    )
    assert "# Dengue e clima" in markdown
    assert "## 1 INTRODUÇÃO" in markdown
    assert "DECLARAÇÃO DE USO DE IA" in markdown
    assert "## REFERÊNCIAS" in markdown

    pdf_path = formatter.render_pdf(markdown, tmp_path / "paper.pdf")
    assert pdf_path is not None
    assert Path(pdf_path).exists()
    assert Path(pdf_path).stat().st_size > 500


# --------------------------------------------------------------------------- #
# LLM: cache e fallback reais
# --------------------------------------------------------------------------- #
def test_mock_provider_returns_valid_json_for_hypotheses():
    payload = json.loads(MockProvider().complete("TÓPICO: x\n\nGere no máximo 3 hipóteses testáveis."))
    assert len(payload["hipoteses"]) == 3
    assert payload["hipoteses"][0]["testavel"] is True


def test_llm_cache_avoids_second_call(tmp_path):
    config = {"llm": {"fallback_order": ["mock"], "cache": {"enabled": True, "path": str(tmp_path / "c.sqlite3")}}}
    client = build_llm_client(config)
    first = client.complete("TÓPICO: dengue\n\nGere no máximo 2 hipóteses testáveis.", task="hypotheses")
    second = client.complete("TÓPICO: dengue\n\nGere no máximo 2 hipóteses testáveis.", task="hypotheses")
    assert first.cached is False
    assert second.cached is True
    assert first.text == second.text
    assert client.stats()["cache_hits"] == 1
    client.close()


def test_llm_client_falls_back_when_provider_fails(tmp_path):
    class BrokenProvider:
        name = "broken"

        def available(self) -> bool:
            return True

        def complete(self, prompt: str, system: str | None = None, **kwargs: object) -> str:
            raise RuntimeError("serviço fora do ar")

    config = {"llm": {"fallback_order": [], "cache": {"enabled": False}, "retries": 0}}
    client = build_llm_client(config)
    client.providers = [BrokenProvider(), MockProvider()]
    response = client.complete("TÓPICO: dengue\n\nGere no máximo 2 hipóteses testáveis.", task="hypotheses")
    assert response.provider == "mock"
    assert "hipoteses" in response.text


def test_openai_compatible_provider_against_real_http_server(tmp_path):
    """Exercita o cliente HTTP real contra um servidor local que fala o protocolo."""
    import json as _json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    received: dict[str, object] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - assinatura da stdlib
            length = int(self.headers["Content-Length"])
            received["payload"] = _json.loads(self.rfile.read(length))
            received["auth"] = self.headers.get("Authorization")
            body = _json.dumps({"choices": [{"message": {"content": '{"ok": true}'}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        config = {
            "llm": {
                "fallback_order": ["openai_compatible"],
                "model": "test-model",
                "base_url": f"http://127.0.0.1:{port}/v1",
                "cache": {"enabled": False},
                "retries": 0,
            }
        }
        import os

        os.environ["OPENAI_API_KEY"] = "test-key"
        client = build_llm_client(config)
        response = client.complete("pergunta de teste", system="seja breve")
        assert response.text == '{"ok": true}'
        assert response.provider == "openai_compatible"
        payload = received["payload"]
        assert payload["model"] == "test-model"
        assert payload["messages"][0] == {"role": "system", "content": "seja breve"}
        assert payload["messages"][1]["content"] == "pergunta de teste"
        assert received["auth"] == "Bearer test-key"
        client.close()
    finally:
        server.shutdown()
        server.server_close()


# --------------------------------------------------------------------------- #
# Ciclo completo, offline e determinístico
# --------------------------------------------------------------------------- #
def test_full_cycle_offline(tmp_path):
    config = Config.load()
    config.data["llm"] = {"fallback_order": ["mock"], "cache": {"enabled": False}}
    config.data["integrations"] = {"offline": True}

    orchestrator = Orchestrator(config=config, output_dir=tmp_path)
    orchestrator.initialize_agents()
    assert len(orchestrator.agents) == 6
    try:
        result = orchestrator.run_research_cycle("Predição de dengue no Brasil", force=True)
    finally:
        orchestrator.close()

    assert result["status"] == "concluido"
    phases = result["phases"]
    assert len(phases["planejamento"]["hipoteses"]) >= 1
    assert phases["experimento"]["resultado"]["best_model"]
    assert phases["validacao"]["score"] > 0

    artifacts = phases["artefatos"]
    paper_md = Path(artifacts["paper_markdown"])
    assert paper_md.exists()
    content = paper_md.read_text(encoding="utf-8")
    assert "DECLARAÇÃO DE USO DE IA" in content
    assert Path(artifacts["experimento_json"]).exists()
    assert Path(artifacts["diretorio"], "ciclo_completo.json").exists()


def test_cycle_failure_is_recorded_in_memory(tmp_path):
    config = Config.load()
    config.data["llm"] = {"fallback_order": ["mock"], "cache": {"enabled": False}}
    config.data["integrations"] = {"offline": True}
    config.data["experiments"] = {"seed": 42, "allow_code_execution": False}

    orchestrator = Orchestrator(config=config, output_dir=tmp_path)
    orchestrator.initialize_agents()
    try:
        with pytest.raises(RuntimeError):
            orchestrator.run_research_cycle("Tópico que falha", force=True)
        cycles = orchestrator.memory.list_cycles(limit=5)
        assert cycles[0]["status"] == "falhou"
    finally:
        orchestrator.close()


# --------------------------------------------------------------------------- #
# Experimentos sensíveis ao desenho (sem rede)
# --------------------------------------------------------------------------- #
def test_runner_honra_modelos_do_desenho(tmp_path):
    """O runner treina os modelos pedidos no desenho, não sempre a mesma tríade.

    É o que faz um tópico diferir do outro: sem isso, todo ciclo treinaria os
    mesmos três modelos e o painel do enxame seria uma comparação vazia.
    """
    from core.experiments.runner import _modelos_do_desenho

    assert _modelos_do_desenho({"modelos": [{"nome": "RegressaoLogistica_Baseline"}]}) == ["regressao_logistica"]
    assert _modelos_do_desenho({"modelos": [{"nome": "Random Forest com Lag de 4 semanas"}]}) == ["floresta_aleatoria"]
    assert _modelos_do_desenho({"modelos": [{"nome": "Gradient Boosting"}, {"nome": "SVM"}]}) == ["gradient_boosting"]
    assert _modelos_do_desenho(None) == ["regressao_logistica", "floresta_aleatoria", "gradient_boosting"]


def test_runner_detecta_defasagem_no_desenho():
    from core.experiments.runner import _defasagem_do_desenho

    assert _defasagem_do_desenho({"pre_processamento": ["feature de defasagem temporal (lag de 4 semanas)"]}) == 4
    assert _defasagem_do_desenho({"desenho": "usa defasagem de 2 semanas"}) == 2
    assert _defasagem_do_desenho({}) == 0


def test_runner_executa_apenas_modelos_pedidos(tmp_path):
    """Dois desenhos distintos produzem métricas distintas sobre dados reais."""
    runner = ExperimentRunner(seed=42, output_dir=tmp_path, cache_dir="data/dataset_cache", offline=True)

    base = runner.run(
        "Predição de surtos de dengue no Brasil",
        {"id": "H1"},
        plan={"modelos": [{"nome": "RegressaoLogistica_Baseline"}]},
    )
    lag = runner.run(
        "Predição de surtos de dengue no Brasil",
        {"id": "H1"},
        plan={
            "modelos": [{"nome": "Random Forest com Lag de 4 semanas"}],
            "pre_processamento": ["lag de 4 semanas"],
        },
    )

    assert set(base.metrics) == {"baseline_estatistico", "regressao_logistica"}
    assert set(lag.metrics) == {"baseline_estatistico", "floresta_aleatoria"}
    # A defasagem adiciona uma cópia defasada das features reais.
    assert lag.dataset["defasagem_semanas"] == 4
    assert lag.dataset["n_variaveis"] == 2 * base.dataset["n_variaveis"]
    assert all(campo.endswith("_lag4") for campo in lag.dataset["variaveis"][len(lag.dataset["variaveis"]) // 2 :])
    assert base.dataset["tipo"] == lag.dataset["tipo"] == "dados_reais"


# --------------------------------------------------------------------------- #
# Integrações reais (rede)
# --------------------------------------------------------------------------- #
@pytest.mark.network
def test_crossref_returns_real_articles():
    api = CrossrefAPI(rate_limit_seconds=0.5, timeout=25)
    articles = api.search_scielo("dengue", rows=3)
    assert len(articles) >= 1
    assert articles[0]["titulo"]
    assert articles[0]["fonte"] == "crossref"


@pytest.mark.network
def test_build_dengue_dataset_uses_real_observations():
    """O dataset de dengue vem do InfoDengue, com observações reais.

    Verifica a proveniência (não sintético), o formato da matriz e o alvo
    binário, além do baseline de persistência declarado nos metadados.
    """
    dataset = build_dengue_outbreak_dataset(
        cidades={3304557: "Rio de Janeiro", 3106200: "Belo Horizonte"},
        anos=(2023,),
    )
    if dataset is None:
        pytest.skip("InfoDengue indisponível neste ambiente")

    assert dataset.meta["tipo"] == "dados_reais"
    assert "InfoDengue" in dataset.meta["origem"]
    assert dataset.X.shape[0] == dataset.y.shape[0] >= 50
    assert dataset.X.shape[1] == len(dataset.meta["variaveis"])
    assert set(np.unique(dataset.y)).issubset({0, 1})
    assert dataset.meta["balanceamento"]["positivos"] == int(dataset.y.sum())
    assert dataset.meta["baseline"]["tipo"] == "persistencia"
    assert dataset.meta["split"] == "temporal"


@pytest.mark.network
def test_runner_uses_real_dataset_and_temporal_split(tmp_path):
    """O runner, com tópico de dengue, treina sobre dados observados.

    O corte é temporal (sem futuro no treino) e o baseline é a persistência do
    nível de alerta — não a regra sintética.
    """
    runner = ExperimentRunner(seed=7, output_dir=tmp_path)
    result = runner.run("Predição de surtos de dengue usando dados climáticos", {"id": "H1"})

    assert result.dataset["tipo"] == "dados_reais"
    assert result.dataset["split"] == "temporal"
    assert result.dataset["baseline"]["tipo"] == "persistencia"
    for name, metrics in result.metrics.items():
        assert 0.0 <= metrics["f1_score"] <= 1.0, name
    assert result.best_model in result.metrics


def test_real_dataset_selection_by_topic():
    """O roteamento escolhe a fonte real certa e recusa temas sem fonte pública.

    Usa o cache versionado (offline) para não depender de rede.
    """
    cache = "data/dataset_cache"
    # Temas sem fonte pública implementada: nada é fabricado.
    assert build_real_dataset("previsao de churn em telecom", cache_dir=cache, offline=True) is None
    assert build_real_dataset("", cache_dir=cache, offline=True) is None
    # Temas com fonte real: cada um resolve para a origem correta.
    dengue = build_real_dataset("dengue e clima no Brasil", cache_dir=cache, offline=True)
    assert dengue is not None and "InfoDengue" in dengue.meta["origem"]
    desmat = build_real_dataset("desmatamento na Amazonia", cache_dir=cache, offline=True)
    assert desmat is not None and "PRODES" in desmat.meta["origem"]
    agro = build_real_dataset("irrigacao no Cerrado", cache_dir=cache, offline=True)
    assert agro is not None and "SIDRA" in agro.meta["origem"]
    edu = build_real_dataset("gamificacao no ensino", cache_dir=cache, offline=True)
    assert edu is not None and "SIDRA" in edu.meta["origem"]


def test_real_datasets_track_provenance():
    """Todo dataset real carrega procedência explícita nos metadados.

    Exigência do projeto: o manuscrito precisa citar de onde veio cada número.
    """
    cache = "data/dataset_cache"
    for builder, esperado in (
        (build_dengue_outbreak_dataset, "InfoDengue"),
        (build_deforestation_dataset, "PRODES"),
        (build_agriculture_dataset, "SIDRA"),
        (build_education_dataset, "SIDRA"),
    ):
        dataset = builder(cache_dir=cache, offline=True)
        assert dataset is not None, builder.__name__
        meta = dataset.meta
        assert meta["tipo"] == "dados_reais"
        assert esperado in meta["origem"]
        assert meta["fonte_url"].startswith("http")
        assert meta["n_amostras"] == dataset.X.shape[0]
        assert meta["n_variaveis"] == dataset.X.shape[1]
        assert meta["split"] == "temporal"


def test_detector_pega_numero_sem_lastro_nos_dados():
    """Um dígito trocado numa afirmação sobre dados reais é detectado.

    Caso real: o manuscrito afirmou intervalo de semanas "200201 a 202351",
    quando o InfoDengue devolveu 202201. O validador por LLM não percebeu.
    """
    from core.review.numeros import collect_numbers, find_unsupported_numbers

    resultado = {
        "dataset": {"intervalo_semanas": {"inicio": 202201, "fim": 202351}, "n_amostras": 817},
        "metrics": {"floresta_aleatoria": {"f1_score": 0.8989}},
    }
    fontes = collect_numbers(resultado)

    manuscrito = "As semanas foram de 200201 a 202351, com N = 817 amostras e F1 = 0,8989."
    suspeitos = find_unsupported_numbers(manuscrito, fontes)
    assert 200201.0 in suspeitos
    assert 202351.0 not in suspeitos
    assert 817.0 not in suspeitos
    assert 0.8989 not in suspeitos


def test_detector_ignora_numeracao_estrutural_e_valores_arredondados():
    """Numeração de seção e arredondamento legítimo não geram falso positivo."""
    from core.review.numeros import collect_numbers, find_unsupported_numbers

    fontes = collect_numbers({"metricas": {"f1_score": 0.89891, "alpha": 0.05}})
    # "2." e "3." são numeração de seção; "0,899" é arredondamento de 0,89891.
    manuscrito = "2. METODOLOGIA\n3. RESULTADOS\nF1 = 0,899 e alpha = 0,05."
    assert find_unsupported_numbers(manuscrito, fontes) == []


def test_detector_aceita_limite_por_truncamento_mas_pega_limite_falso():
    """Um limite conservador (p > 0,20 para p = 0,2065) não é número sem lastro.

    Caso real: o manuscrito escreveu "p > 0,20" e "p > 0,24" a partir de
    p = 0,2065 e p = 0,2490. Acusar isso era falso positivo. Já um limite que
    nenhum dado sustenta ("p > 0,30") continua sendo apontado.
    """
    from core.review.numeros import find_unsupported_numbers

    fontes = [0.206553, 0.249020, 0.655645]
    legitimo = "Floresta com p > 0,20 e Gradient Boosting com p > 0,24."
    assert find_unsupported_numbers(legitimo, fontes) == []

    falso = "Todos os modelos com p > 0,30."
    assert 0.3 in find_unsupported_numbers(falso, fontes)


def test_detector_ignora_numeracao_de_subsecao_e_versoes():
    """Títulos numerados e versões de software não são afirmações sobre dados.

    O LLM escreve "### 3.1. Amostragem" e "Python 3.13.15"; sem podar isso, o
    conferidor acusava 3.1-3.4 e 3.13.15 como números sem lastro.
    """
    from core.review.numeros import collect_numbers, find_unsupported_numbers

    fontes = collect_numbers({"n_amostras": 817})
    manuscrito = (
        "## 3 RESULTADOS E DISCUSSÃO\n"
        "### 3.1. Desempenho dos Modelos\n"
        "### 3.2. Análise Estatística\n"
        "### 3.3. Comparação com a Literatura\n"
        "### 3.4. Limitações\n"
        "Executado em Python 3.13.15, com N = 817 amostras. Ver Seção 3.2 e Tabela 2.1.\n"
    )
    assert find_unsupported_numbers(manuscrito, fontes) == []


def test_detector_extrai_numeros_de_estrutura_aninhada():
    from core.review.numeros import collect_numbers

    numeros = collect_numbers({"a": [1, "2.5", {"b": "x=3,7"}]})
    assert {1.0, 2.5, 3.7}.issubset(numeros)


def test_temporal_split_keeps_order():
    """O corte temporal preserva a ordem cronológica das amostras."""
    X = np.arange(20).reshape(10, 2).astype(float)
    y = np.arange(10)
    runner = ExperimentRunner(test_size=0.3)
    X_train, X_test, y_train, y_test = runner._split_temporal(X, y)
    assert len(X_train) == 7 and len(X_test) == 3
    assert X_train[-1, 0] < X_test[0, 0]
    assert list(y_train) == [0, 1, 2, 3, 4, 5, 6]
    assert list(y_test) == [7, 8, 9]


def test_persistence_baseline_flags_current_alert_level():
    """O baseline de persistência usa o nível atual da semana t."""
    meta = {"baseline": {"tipo": "persistencia", "coluna": "nivel", "limiar": 3}, "variaveis": ["temp", "nivel"]}
    X_test = np.array([[25.0, 1.0], [26.0, 3.0], [27.0, 4.0]])
    runner = ExperimentRunner()
    y_pred, score = runner._baseline_prediction(X_test, meta)
    assert list(y_pred) == [0, 1, 1]
    assert list(score) == [1.0, 3.0, 4.0]


@pytest.mark.network
def test_aggregator_collects_multiple_sources():
    aggregator = BrazilianDataAggregator.from_config(Config.load().as_dict())
    gathered = aggregator.gather("dengue e clima no Brasil")
    assert len(gathered["artigos"]) >= 1
    assert gathered["modo"] == "online"
    assert "EVIDÊNCIAS COLETADAS" in aggregator.reference_context(gathered)


# --------------------------------------------------------------------------- #
# Recursos portados do AI Scientist (Sakana AI)
# --------------------------------------------------------------------------- #
def test_json_retry_repairs_invalid_response():
    """O agente deve repetir com correção quando o provider devolve texto solto."""

    class BrokenThenGood:
        name = "broken"

        def __init__(self):
            self.calls = 0

        def available(self):
            return True

        def complete(self, prompt, system=None, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return "Desculpe, não consigo responder em JSON."
            assert "CORREÇÃO OBRIGATÓRIA" in prompt
            return '{"parecer": "aceitar"}'

    from core.agents.base import BaseAgent
    from llm_providers import LLMClient

    client = LLMClient(providers=[BrokenThenGood()])
    agent = BaseAgent(name="teste", llm=client)
    result = agent.run("Responda em JSON", expect_json=True)
    assert result.json_valid is True
    assert result.content == {"parecer": "aceitar"}
    assert client.providers[0].calls == 2


def test_json_retry_exhaustion_is_flagged():
    """Sem JSON válido após as tentativas, o resultado é marcado como inválido."""

    class AlwaysBroken:
        name = "broken"

        def available(self):
            return True

        def complete(self, prompt, system=None, **kwargs):
            return "texto sem json"

    from core.agents.base import BaseAgent
    from llm_providers import LLMClient

    agent = BaseAgent(name="teste", llm=LLMClient(providers=[AlwaysBroken()]), json_retries=1)
    result = agent.run("Responda em JSON", expect_json=True)
    assert result.json_valid is False
    assert "texto_bruto" in result.content


def test_reviewer_panel_aggregates_five_reviewers():
    """O painel roda N revisores, gera meta-parecer e voto majoritário."""
    from core.review import ReviewerPanel

    llm = build_llm_client({"llm": {"provider": "mock", "cache": {"enabled": False}}})
    panel = ReviewerPanel(llm=llm, reviewer_count=5)
    result = panel.review("dengue e clima", "Manuscrito de teste com resultados.", "evidências")

    assert result["n_validos"] == 5
    assert len(result["revisores"]) == 5
    assert result["decisao_final"] in ["aceitar", "aceitar com revisões", "revisões maiores", "rejeitar"]
    assert result["meta_parecer"].get("parecer")
    assert sum(result["distribuicao_decisoes"].values()) == 5


def test_reviewer_panel_survives_failing_reviewer():
    """Um revisor que falha não derruba o painel inteiro."""

    class Flaky:
        name = "flaky"

        def __init__(self):
            self.calls = 0

        def available(self):
            return True

        def complete(self, prompt, system=None, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("falha simulada de rede")
            return json.dumps({"parecer": "aceitar com revisões", "severidade": "moderada"})

    from core.review import ReviewerPanel
    from llm_providers import LLMClient

    panel = ReviewerPanel(llm=LLMClient(providers=[Flaky()]), reviewer_count=3)
    result = panel.review("tópico", "manuscrito")
    assert result["n_validos"] >= 1
    assert result["decisao_final"]


def test_majority_decision_is_ordinal():
    """O voto majoritário usa ordem, não média aritmética de rótulos."""
    from core.review.panel import ReviewerPanel

    reviews = [{"parecer": "aceitar"}, {"parecer": "rejeitar"}, {"parecer": "aceitar"}]
    outcome = ReviewerPanel._majority_decision(reviews)
    assert outcome["decisao"] == "aceitar"
    assert outcome["distribuicao"] == {"aceitar": 2, "rejeitar": 1}


def test_dedupe_merges_doi_preserving_richest_fields():
    """Registros com o mesmo DOI são mesclados preservando resumo e citações."""
    articles = [
        {"doi": "10.1/x", "titulo": "A", "fonte": "crossref", "resumo": "resumo real", "citacoes": None},
        {"doi": "10.1/x", "titulo": "A", "fonte": "openalex", "resumo": None, "citacoes": 42},
    ]
    merged = BrazilianDataAggregator._dedupe(articles)
    assert len(merged) == 1
    assert merged[0]["resumo"] == "resumo real"
    assert merged[0]["citacoes"] == 42
    assert set(merged[0]["fonte"].split("+")) == {"crossref", "openalex"}


def test_latex_formatter_escapes_and_structures():
    """O LaTeX gerado é compilável em estrutura e escapa caracteres reservados."""
    from core.paper import LaTeXFormatter

    formatter = LaTeXFormatter()
    tex = formatter.render_tex(
        topic="Teste de 100% de acurácia & cia",
        sections={"resumo": "Resumo com 50% de dados_e_mais.", "introducao": "Texto com #hashtag."},
        references=[{"autores": ["Silva, J."], "titulo": "Artigo", "ano": 2024, "periodico": "Rev", "doi": "10.1/x"}],
        metadata={"titulo": "Teste", "autores": ["IA"], "data": "2024-01-01", "palavras_chave": ["a"], "keywords": ["b"]},
    )
    assert "\\documentclass" in tex
    assert "\\begin{document}" in tex and "\\end{document}" in tex
    assert "\\bibitem" in tex
    assert "50\\%" in tex and "\\_mais" in tex and "\\#hashtag" in tex
    assert "\\section{Introdução}" in tex


def test_latex_escapes_unicode_from_real_llms():
    """LLMs reais emitem espaços/hífens tipográficos, gregas e subscritos.

    Esses caracteres não são mapeados por inputenc+fontenc e abortavam o
    pdflatex, fazendo o ciclo terminar sem paper_latex.pdf.
    """
    from core.paper import LaTeXFormatter

    escapar = LaTeXFormatter._escape
    # Espaços e hífens tipográficos viram equivalentes ASCII.
    assert escapar("seed\u202f=\u202f42") == "seed = 42"
    assert escapar("dados\xa0abertos") == "dados abertos"
    assert escapar("co\u2011ocorrência") == "co-ocorrência"
    assert escapar("aspas \u201cx\u201d e\u2026") == "aspas ``x'' e\\ldots{}"
    # Gregas e sub/sobrescritos viram comandos matemáticos.
    assert escapar("\u03b1") == "$\\alpha$"
    assert escapar("\u0394SOC") == "$\\Delta$SOC"
    assert escapar("N\u2082O e CO\u2082") == "N$_2$O e CO$_2$"
    assert escapar("km\u00b2") == "km$^2$"
    # Símbolos fora do alcance do LaTeX são descartados, sem deixar lixo.
    assert escapar("emoji \U0001f680 fim") == "emoji  fim"
    # Acentuação normal é preservada.
    assert escapar("acentuação ç ã é") == "acentuação ç ã é"


def test_llm_cache_key_depends_on_provider():
    """A chave de cache precisa incluir o provider.

    Sem isso, uma resposta do mock já em cache era devolvida numa execução com
    LLM real, mascarando o provider escolhido pelo usuário.
    """
    from llm_providers import LLMClient, MockProvider, OpenAICompatibleProvider

    mock = LLMClient(providers=[MockProvider()])
    real = LLMClient(providers=[OpenAICompatibleProvider(model="openai/gpt-oss-120b", api_key="x")])
    assert mock._signature() != real._signature()
    assert "mock" in mock._signature()
    assert "openai/gpt-oss-120b" in real._signature()


def test_llm_backoff_honors_rate_limit_hint():
    """Em HTTP 429, espera o tempo sugerido pela API em vez de um backoff curto.

    Ignorar o "try again in Ns" esgotava as tentativas e derrubava o pipeline
    para o provider mock sem aviso.
    """
    from llm_providers import LLMClient

    erro = Exception('HTTP 429: {"error":{"message":"Rate limit reached ... try again in 27.66s."}}')
    espera = LLMClient._backoff(erro, attempt=0)
    assert 28.0 <= espera <= 29.0, espera
    # Sem a dica, usa um backoff progressivo, mas limitado.
    assert LLMClient._backoff(Exception("HTTP 500"), attempt=0) == 0.5
    assert LLMClient._backoff(Exception("HTTP 500"), attempt=5) <= 8.0


def test_free_stack_providers_use_isolated_keys(monkeypatch):
    """Cada provider da pilha lê a sua própria variável de chave.

    Sem isso, uma unica OPENAI_API_KEY deixaria todos os providers "disponiveis"
    e o cliente tentaria provedores sem chave, gastando tentativas em 401.
    """
    import llm_providers
    from core.config import Config

    monkeypatch.setenv("GROQ_API_KEY", "chave-groq")
    monkeypatch.setenv("OPENAI_API_KEY", "chave-generica")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    config = Config.load("config.free-stack.json")
    client = llm_providers.build_llm_client(config.data, use_cache=False)
    disponiveis = {p.name for p in client.providers if p.available()}
    assert disponiveis == {"groq", "mock"}, disponiveis
    groq = next(p for p in client.providers if p.name == "groq")
    assert groq.api_key == "chave-groq"
    assert groq.base_url == "https://api.groq.com/openai/v1"


def test_config_merge_preserves_untouched_sections(tmp_path):
    """Um --config parcial nao pode desativar integracoes nem caminhos.

    Antes, carregar um arquivo com so a secao ``llm`` substituia a
    configuracao inteira e as APIs brasileiras ficavam silenciosamente vazias.
    """
    import json as _json
    from core.config import Config

    parcial = tmp_path / "parcial.json"
    parcial.write_text(_json.dumps({"llm": {"temperature": 0.9}}), encoding="utf-8")
    config = Config.load(parcial)
    assert config.data["llm"]["temperature"] == 0.9
    assert "integrations" in config.data and config.data["integrations"]
    assert "paths" in config.data
    assert config.data["integrations"]["crossref"]["base_url"].startswith("https://")


def test_long_rate_limit_skips_provider_instead_of_waiting(monkeypatch):
    """Um 429 que pede espera longa nao deve prender o ciclo.

    A Groq, ao esgotar o teto diario, responde "try again in 26m". Esperar esse
    tempo (ou repetir o provider com backoff curto) travava o ciclo por horas.
    O cliente deve desistir deste provider e passar ao proximo da cadeia.
    """
    import llm_providers
    from llm_providers import LLMClient

    esperas: list[float] = []

    class ProviderQuebrado:
        name = "quebrado"
        model = "x"

        def available(self) -> bool:
            return True

        def complete(self, prompt, system=None, **kwargs):
            raise Exception("HTTP 429: try again in 26m16s")

    class ProviderBom:
        name = "bom"
        model = "y"

        def available(self) -> bool:
            return True

        def complete(self, prompt, system=None, **kwargs):
            return "resposta boa"

    monkeypatch.setattr(llm_providers.time, "sleep", lambda s: esperas.append(s))
    client = LLMClient(providers=[ProviderQuebrado(), ProviderBom()], retries=3)
    resposta = client.complete("pergunta")
    assert resposta.text == "resposta boa"
    assert resposta.provider == "bom"
    # Nenhuma espera longa foi executada: pulou direto para o proximo provider.
    assert esperas == [], esperas


def test_stats_report_configured_providers_and_failures(monkeypatch):
    """O relatorio precisa distinguir "nao configurado" de "configurado e falhou".

    Com --config, uma cadeia em que todos os providers reais falharam aparecia
    apenas como mock, e o aviso dizia "nenhum LLM real foi configurado" — o que
    escondia justamente a informacao util: quem falhou e quantas vezes.
    """
    import llm_providers
    from llm_providers import LLMClient

    class P:
        name = "groq"
        model = "m"

        def available(self) -> bool:
            return True

        def complete(self, *a, **k):
            raise Exception("HTTP 429: try again in 9m34s")

    monkeypatch.setattr(llm_providers.time, "sleep", lambda s: None)
    client = LLMClient(providers=[P()], retries=1)
    try:
        client.complete("x")
    except RuntimeError:
        pass
    stats = client.stats()
    assert stats["providers_configurados"] == ["groq"]
    # Uma única falha: a espera pedida (9m34s) excede o teto, então o cliente
    # pula o provider na primeira tentativa em vez de insistir.
    assert stats["falhas_por_provider"]["groq"] == 1


def test_render_markdown_nao_duplica_titulos_de_secao():
    """O formatador nao pode repetir o titulo que o LLM ja escreveu.

    O redator devolve "## 1. INTRODUCAO\n\ntexto" e o formatador adiciona
    "## 1 INTRODUCAO". Sem deduplicar, toda secao de todo paper saia com dois
    titulos seguidos.
    """
    from core.paper.abnt import ABNTFormatter

    sections = {
        "resumo": "**RESUMO**\n\nEste estudo investigou o deficit.",
        "abstract": "**ABSTRACT**\n\nThis study investigated.",
        "introducao": "## 1. INTRODUCAO\n\nO agronegocio brasileiro cresce.",
        "metodologia": "## METODOLOGIA\n\nUsou-se um dataset sintetico.",
        "resultados": "## 3. RESULTADOS\n\nA acuracia foi de 0,85.",
        "conclusao": "## 4. CONCLUSAO\n\nConclui-se que ha relacao.",
    }
    md = ABNTFormatter().render_markdown(
        topic="Armazenagem de graos",
        sections=sections,
        references=["[1] REFERENCIA. Titulo. 2024."],
        metadata={"titulo": "Armazenagem de graos", "data": "2026-01-01"},
    )
    for titulo in ("## 1 INTRODUÇÃO", "## 2 METODOLOGIA", "## 3 RESULTADOS E DISCUSSÃO", "## 4 CONCLUSÃO"):
        assert md.count(titulo) == 1, (titulo, md.count(titulo))
    assert "## 1. INTRODUÇÃO" not in md
    assert md.count("## RESUMO") == 1
    # O corpo nao pode ser engolido junto com o titulo.
    assert "O agronegocio brasileiro cresce." in md
    assert "A acuracia foi de 0,85." in md


def test_strip_duplicate_heading_preserva_texto_do_corpo():
    """Nao confundir titulo com uma frase que comeca com a mesma palavra."""
    from core.paper.headings import strip_duplicate_heading as s

    # Sem marcador de titulo: e texto, nao titulo.
    assert s("Introducao e o tema central", "Introducao") == "Introducao e o tema central"
    assert s("texto sem titulo", "1 INTRODUCAO") == "texto sem titulo"
    # Com marcador: e titulo, remove.
    assert s("## 1. INTRODUCAO\n\ncorpo", "1 INTRODUCAO") == "corpo"
    assert s("**RESUMO**\n\ncorpo", "RESUMO") == "corpo"
    # O LLM abrevia o titulo esperado.
    assert s("## 3. RESULTADOS\n\ncorpo", "3 RESULTADOS E DISCUSSAO") == "corpo"
    # Titulo de outra secao nao e removido.
    assert s("## 2. MATERIAIS E METODOS\n\ncorpo", "2 METODOLOGIA").startswith("## 2.")


def test_remove_declaracao_de_ia_repetida_nas_secoes():
    """O redator fecha cada secao com uma declaracao de IA; so uma deve sobrar.

    O formatador emite a declaracao canonica no fim do documento. As copias que
    o LLM escreve dentro das secoes faziam o mesmo paragrafo aparecer quatro
    vezes no paper.
    """
    from core.paper.abnt import ABNTFormatter

    corpo = (
        "## METODOLOGIA\n\nO metodo foi quantitativo.\n\n---\n\n"
        "### Declara\u00e7\u00e3o de Uso de Intelig\u00eancia Artificial\n"
        "Declara-se que ferramentas de IA foram utilizadas."
    )
    md = ABNTFormatter().render_markdown(
        topic="Teste",
        sections={"metodologia": corpo},
        references=["[1] REF. Titulo. 2024."],
        metadata={"titulo": "Teste", "data": "2026-01-01"},
    )
    assert "O metodo foi quantitativo." in md
    # A declaracao interna saiu; a canonica do formatador permanece.
    assert md.count("Declara\u00e7\u00e3o de Uso de Intelig\u00eancia Artificial") == 0
    assert "Declara-se que ferramentas de IA foram utilizadas." not in md
    assert md.count("## DECLARA\u00c7\u00c3O DE USO DE IA") == 1

    # Variante "Declaracao sobre o Uso de Inteligencia Artificial" tambem sai.
    corpo2 = (
        "## METODOLOGIA\n\nO metodo.\n\n"
        "### Declara\u00e7\u00e3o sobre o Uso de Intelig\u00eancia Artificial\n"
        "Declara-se que ferramentas de IA."
    )
    md2 = ABNTFormatter().render_markdown(
        topic="Teste",
        sections={"metodologia": corpo2},
        references=["[1] REF. 2024."],
        metadata={"titulo": "Teste", "data": "2026-01-01"},
    )
    assert "Declara\u00e7\u00e3o sobre o Uso" not in md2
    assert "O metodo." in md2


def test_renumera_subsecoes_para_o_numero_da_secao():
    """O LLM numera subsecoes por conta propria e erra o prefixo.

    Aparecia "### 3.1" dentro da secao 2 (METODOLOGIA) e "### 4.1" dentro da
    secao 3 (RESULTADOS). O prefixo deve seguir a secao que contem a subsecao.
    """
    from core.paper.headings import renumber_subsections

    corpo = "### 3.1. Amostragem\n\ntexto\n\n### 3.2. Modelos\n\nmais"
    out = renumber_subsections(corpo, 2)
    assert "### 2.1. Amostragem" in out
    assert "### 2.2. Modelos" in out
    assert "### 3.1" not in out
    # Variante sem ponto final depois do numero.
    assert "### 3.1. Desempenho" in renumber_subsections("### 4.1 Desempenho", 3)
    # Subsecao sem numero nao e tocada.
    assert renumber_subsections("### Sem numero", 2) == "### Sem numero"


def test_latex_compile_produces_distinct_pdf(tmp_path):
    """Com pdflatex presente, o PDF do LaTeX é separado e não sobrescreve o ABNT."""
    from core.paper import LaTeXFormatter

    formatter = LaTeXFormatter()
    if not formatter.available():
        pytest.skip("pdflatex indisponível neste ambiente")

    abnt_pdf = tmp_path / "paper.pdf"
    abnt_pdf.write_bytes(b"%PDF-1.4 abnt-marker")

    result = formatter.write(
        topic="Predição de surtos de dengue no Brasil",
        sections={
            "resumo": "Resumo com acentuação: predição, irrigação, saúde.",
            "introducao": "Introdução com 50% de dados_e_mais.",
            "metodologia": "Metodologia.",
            "resultados": "Resultados.",
            "conclusao": "Conclusão.",
        },
        references=[{"autores": ["Silva, J."], "titulo": "Artigo", "ano": 2024, "periodico": "Rev", "doi": "10.1/x"}],
        metadata={"titulo": "Teste", "autores": ["IA"], "data": "2024-01-01", "palavras_chave": ["a"], "keywords": ["b"]},
        output_dir=tmp_path,
    )
    assert result["tex"] and Path(result["tex"]).exists()
    assert result["pdf"] and Path(result["pdf"]).name == "paper_latex.pdf"
    assert Path(result["pdf"]).read_bytes().startswith(b"%PDF")
    assert abnt_pdf.read_bytes() == b"%PDF-1.4 abnt-marker"


def test_latex_compile_never_raises_on_failure(tmp_path):
    """LaTeX inválido não deve propagar exceção; o ciclo segue com o .tex."""
    from core.paper import LaTeXFormatter

    formatter = LaTeXFormatter()
    if not formatter.available():
        pytest.skip("pdflatex indisponível neste ambiente")

    broken = tmp_path / "quebrado.tex"
    broken.write_text("\\documentclass{article}\\begin{document}\\naoexiste\\end{document}", encoding="utf-8")
    assert formatter.compile_pdf(broken, tmp_path) is None


def test_doctor_reports_environment(tmp_path):
    """O diagnóstico detecta dependências e gravabilidade sem rede."""
    from core.doctor import Doctor, render

    result = Doctor().run(output_dir=str(tmp_path), check_network=False)
    assert result["ok"] is True
    names = [c["name"] for c in result["checks"]]
    assert any("python>=3.11" in n for n in names)
    assert any("diret" in n for n in names)
    assert "DIAGNÓSTICO" in render(result)


def test_batch_runner_processes_multiple_topics(tmp_path):
    """O modo lote executa vários tópicos e consolida um relatório."""
    from core.batch import BatchRunner

    report = BatchRunner(base_dir=tmp_path / "lote", max_workers=2).run(
        ["dengue e clima no Brasil", "irrigação com IoT no Cerrado"]
    )
    assert report["total"] == 2
    assert report["concluidos"] == 2
    assert report["erros"] == 0
    assert (tmp_path / "lote" / "lote_resumo.json").exists()
    assert all(item.get("artefato") for item in report["resultados"])

