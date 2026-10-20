"""Testes do roteador de LLM por tarefa (MoE).

Não gastam cota: usam um provider falso que registra as chamadas e pode ser
programado para falhar, mais um catálogo embutido no próprio teste.

O ``TaskRouter`` é montado com os providers falsos diretamente (não via
``build_task_router``, que instancia providers reais) para exercitar só a lógica
de roteamento. A construção a partir do catálogo é testada em separado, apenas
inspecionando os providers criados — sem rede.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_providers import (
    LLMCache,
    LLMClient,
    MockProvider,
    OpenAICompatibleProvider,
    TaskRouter,
    build_task_router,
    categoria_da_tarefa,
    max_tokens_da_tarefa,
)

CATALOGO = {
    "roteamento_por_tarefa": {
        "raciocinio": ["modelo-raciocinio-1", "modelo-raciocinio-2"],
        "codigo": ["modelo-codigo-1", "modelo-codigo-2"],
        "redacao": ["modelo-redacao-1"],
    },
    "tarefas_para_categoria": {
        "experiment_plan": "codigo",
        "analysis": "codigo",
        "hypotheses": "raciocinio",
        "section": "redacao",
    },
    "presets": {
        "codigo": {"max_tokens": 4096},
        "raciocinio": {"max_tokens": 4096},
        "redacao": {"max_tokens": 2048},
        "padrao": {"max_tokens": 2048},
    },
    "modelos": {
        "modelo-raciocinio-1": {"provider": "fake", "base_url": "https://fake.invalid/v1", "api_key_env": "FAKE_KEY"},
        "modelo-raciocinio-2": {"provider": "fake", "base_url": "https://fake.invalid/v1", "api_key_env": "FAKE_KEY"},
        "modelo-codigo-1": {"provider": "fake", "base_url": "https://fake.invalid/v1", "api_key_env": "FAKE_KEY"},
        "modelo-codigo-2": {"provider": "fake", "base_url": "https://fake.invalid/v1", "api_key_env": "FAKE_KEY"},
        "modelo-redacao-1": {"provider": "fake", "base_url": "https://fake.invalid/v1", "api_key_env": "FAKE_KEY"},
    },
    "fallback_global": {"cadeia": ["modelo-redacao-1", "mock"]},
}


class ProviderFalso(OpenAICompatibleProvider):
    """Provider que não toca a rede: registra a chamada e devolve/levanta."""

    def __init__(self, nome: str, erro: Exception | None = None):
        super().__init__(name=nome, model=nome, api_key="fake")
        self.erro = erro
        self.chamadas: list[dict] = []

    def available(self) -> bool:  # sempre disponível, sem depender de env
        return True

    def complete(self, prompt: str, system: str | None = None, **kwargs) -> str:
        self.chamadas.append({"prompt": prompt, "system": system, **kwargs})
        if self.erro is not None:
            raise self.erro
        return json.dumps({"ok": True, "modelo": self.name})


def _fakes(*nomes: str, erros: dict[str, Exception] | None = None) -> dict[str, ProviderFalso]:
    erros = erros or {}
    return {n: ProviderFalso(n, erro=erros.get(n)) for n in nomes}


def _router(fakes: dict[str, ProviderFalso], **kwargs) -> TaskRouter:
    """TaskRouter apontando para os fakes, na ordem do catálogo embutido."""
    rotas = {
        cat: [fakes[m] for m in modelos if m in fakes]
        for cat, modelos in CATALOGO["roteamento_por_tarefa"].items()
    }
    padrao = {
        "rotas": rotas,
        "mapa_tarefas": dict(CATALOGO["tarefas_para_categoria"]),
        "presets": {c: b["max_tokens"] for c, b in CATALOGO["presets"].items()},
        "cadeia_global": [fakes["modelo-redacao-1"]] if "modelo-redacao-1" in fakes else [MockProvider()],
    }
    padrao.update(kwargs)  # o teste pode sobrescrever a cadeia global
    return TaskRouter(**padrao)


def _client(fakes: dict[str, ProviderFalso], cache: LLMCache | None = None) -> LLMClient:
    return LLMClient(providers=list(fakes.values()), router=_router(fakes), cache=cache, retries=0)


# --------------------------------------------------------------------------- #
# Tarefa 1: mapeamento
# --------------------------------------------------------------------------- #
def test_mapeamento_tarefa_para_categoria() -> None:
    assert categoria_da_tarefa("experiment_plan") == "codigo"
    assert categoria_da_tarefa("hypotheses") == "raciocinio"
    assert categoria_da_tarefa("section") == "redacao"
    # Tarefa desconhecida cai na categoria mais exigente, nunca em texto livre.
    assert categoria_da_tarefa("inexistente") == "raciocinio"
    assert categoria_da_tarefa(None) == "raciocinio"


def test_config_real_roteia_todas_as_tarefas_dos_agentes() -> None:
    """O config.modelos.json real precisa cobrir as tarefas que os agentes emitem."""
    raiz = Path(__file__).resolve().parent.parent
    catalogo = json.loads((raiz / "config.modelos.json").read_text(encoding="utf-8"))
    mapa = catalogo["tarefas_para_categoria"]
    for tarefa in ("gap", "hypotheses", "novelty", "review", "section", "translate",
                   "rebuttal", "rebuttal_response", "experiment_plan", "analysis",
                   "auto_debug", "score", "compliance", "voting",
                   "satellite_features", "chart_reading"):
        assert tarefa in mapa, f"tarefa {tarefa} sem rota"
        assert mapa[tarefa] in catalogo["roteamento_por_tarefa"], f"categoria de {tarefa} sem modelos"


# --------------------------------------------------------------------------- #
# Teste 1 do enunciado: task="codigo" escolhe o primeiro modelo de código
# --------------------------------------------------------------------------- #
def test_task_codigo_usa_primeiro_modelo_da_rota() -> None:
    fakes = _fakes("modelo-codigo-1", "modelo-codigo-2")
    client = _client(fakes)

    resp = client.complete("planeje o experimento", task="experiment_plan")

    assert resp.provider == "modelo-codigo-1"
    assert len(fakes["modelo-codigo-1"].chamadas) == 1
    assert fakes["modelo-codigo-2"].chamadas == []  # o segundo não é tocado


def test_tarefa_sem_rota_configurada_cai_para_a_cadeia_global() -> None:
    p1 = ProviderFalso("modelo-codigo-1")
    global_ = ProviderFalso("modelo-redacao-1")
    router = TaskRouter(
        rotas={"codigo": [p1]},
        mapa_tarefas={"experiment_plan": "codigo"},
        cadeia_global=[global_],
    )
    client = LLMClient(providers=[p1, global_], router=router, retries=0)

    resp = client.complete("escreva a introdução", task="section")  # sem rota

    assert resp.provider == "modelo-redacao-1"
    assert p1.chamadas == []


# --------------------------------------------------------------------------- #
# Teste 2 do enunciado: 429 no primeiro -> tenta o segundo (cascata)
# --------------------------------------------------------------------------- #
def test_fallback_em_cascata_apos_429() -> None:
    fakes = _fakes("modelo-codigo-1", "modelo-codigo-2",
                   erros={"modelo-codigo-1": RuntimeError("HTTP 429: rate limit exceeded")})
    client = _client(fakes)

    resp = client.complete("planeje o experimento", task="experiment_plan")

    assert resp.provider == "modelo-codigo-2"
    assert len(fakes["modelo-codigo-1"].chamadas) == 1  # tentou uma vez, sem backoff
    assert len(fakes["modelo-codigo-2"].chamadas) == 1


@pytest.mark.parametrize("erro", ["HTTP 503: service unavailable", "HTTP 500: internal error", "timed out"])
def test_cascata_em_falhas_transitorias(erro: str) -> None:
    fakes = _fakes("modelo-codigo-1", "modelo-codigo-2",
                   erros={"modelo-codigo-1": RuntimeError(erro)})
    client = _client(fakes)

    resp = client.complete("p", task="experiment_plan")

    assert resp.provider == "modelo-codigo-2"


def test_cascata_termina_no_mock_quando_todos_falham() -> None:
    """Todos os modelos da categoria falham -> a cadeia global (mock) responde."""
    fakes = _fakes("modelo-codigo-1", "modelo-codigo-2", erros={
        "modelo-codigo-1": RuntimeError("HTTP 429"),
        "modelo-codigo-2": RuntimeError("HTTP 503"),
    })
    mock = MockProvider()
    router = _router(fakes, cadeia_global=[mock])
    client = LLMClient(providers=list(fakes.values()), router=router, retries=0)

    resp = client.complete("p", task="experiment_plan")

    assert resp.provider == "mock"
    assert resp.text  # o mock sempre devolve algo utilizável


def test_sem_cadeia_global_propaga_o_erro() -> None:
    fakes = _fakes("modelo-codigo-1", erros={"modelo-codigo-1": RuntimeError("HTTP 429")})
    router = _router(fakes, cadeia_global=[])
    client = LLMClient(providers=list(fakes.values()), router=router, retries=0)

    with pytest.raises(RuntimeError, match="Falha em todos os providers"):
        client.complete("p", task="experiment_plan")


# --------------------------------------------------------------------------- #
# Teste 3 do enunciado: max_tokens por tarefa
# --------------------------------------------------------------------------- #
def test_max_tokens_por_tarefa() -> None:
    assert max_tokens_da_tarefa("experiment_plan") == 4096  # codigo
    assert max_tokens_da_tarefa("hypotheses") == 4096  # raciocinio
    assert max_tokens_da_tarefa("section") == 2048  # redacao
    assert max_tokens_da_tarefa("score") == 2048  # alto_volume


def test_max_tokens_chega_ao_provider() -> None:
    fakes = _fakes("modelo-codigo-1")
    client = _client(fakes)

    client.complete("p", task="experiment_plan")

    enviado = fakes["modelo-codigo-1"].chamadas[0]["max_tokens"]
    assert enviado >= 2048, f"max_tokens abaixo do minimo: {enviado}"
    assert enviado == 4096


def test_max_tokens_explicito_do_chamador_vence() -> None:
    fakes = _fakes("modelo-codigo-1")
    client = _client(fakes)

    client.complete("p", task="experiment_plan", max_tokens=8192)

    assert fakes["modelo-codigo-1"].chamadas[0]["max_tokens"] == 8192


# --------------------------------------------------------------------------- #
# response_format
# --------------------------------------------------------------------------- #
def test_response_format_json_apenas_em_tarefas_json() -> None:
    fakes = _fakes("modelo-codigo-1")
    client = _client(fakes)

    client.complete("p", task="experiment_plan")
    assert fakes["modelo-codigo-1"].chamadas[0]["response_format"] == {"type": "json_object"}


def test_response_format_ausente_em_texto_livre() -> None:
    fakes = _fakes("modelo-redacao-1")
    client = _client(fakes)

    client.complete("p", task="section")
    assert "response_format" not in fakes["modelo-redacao-1"].chamadas[0]


# --------------------------------------------------------------------------- #
# Cache: a rota entra na chave, senão a resposta de uma categoria vaza na outra
# --------------------------------------------------------------------------- #
def test_cache_nao_mistura_categorias(tmp_path: Path) -> None:
    cache = LLMCache(path=tmp_path / "c.sqlite3", enabled=True)
    fakes = _fakes("modelo-codigo-1", "modelo-raciocinio-1")
    client = LLMClient(providers=list(fakes.values()), router=_router(fakes), cache=cache, retries=0)

    client.complete("mesmo prompt", task="experiment_plan")
    client.complete("mesmo prompt", task="hypotheses")

    # Prompts idênticos, tarefas diferentes: cada rota chamou o seu modelo.
    assert len(fakes["modelo-codigo-1"].chamadas) == 1
    assert len(fakes["modelo-raciocinio-1"].chamadas) == 1
    cache.close()


# --------------------------------------------------------------------------- #
# Tarefa 4: base_url sem barra duplicada
# --------------------------------------------------------------------------- #
def test_base_url_sem_barra_final_duplicada() -> None:
    p = OpenAICompatibleProvider(name="x", model="x", base_url="https://exemplo.invalid/v1/", api_key="k")
    assert p.base_url == "https://exemplo.invalid/v1"
    assert not p.base_url.endswith("//")


def test_router_aplica_rstrip_na_base_url_do_catalogo() -> None:
    catalogo = json.loads(json.dumps(CATALOGO))
    catalogo["modelos"]["modelo-codigo-1"]["base_url"] = "https://fake.invalid/v1///"
    router = build_task_router(catalogo, {"fake": ProviderFalso("fake")})
    assert router is not None
    assert router.rotas["codigo"][0].base_url == "https://fake.invalid/v1"


def test_router_usa_model_override_do_catalogo() -> None:
    """O id enviado à API pode diferir da chave do catálogo (Mistral, NIM)."""
    catalogo = json.loads(json.dumps(CATALOGO))
    catalogo["modelos"]["modelo-codigo-1"]["model"] = "codestral-latest"
    router = build_task_router(catalogo, {"fake": ProviderFalso("fake")})
    assert router is not None
    p = router.rotas["codigo"][0]
    assert p.name == "modelo-codigo-1"  # chave do catálogo (para logs)
    assert p.model == "codestral-latest"  # id enviado à API


# --------------------------------------------------------------------------- #
# Roteador inativo quando não há chave: não deve fingir que está roteando
# --------------------------------------------------------------------------- #
def test_router_inativo_sem_chaves(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FAKE_KEY", raising=False)
    router = build_task_router(CATALOGO, {})
    assert router is not None
    assert router.ativo is False
    # Sem chave, a rota cai para a cadeia global, que aqui só tem o mock.
    assert all(getattr(p, "name", "") != "modelo-codigo-1" for p in router.rota("experiment_plan"))


# --------------------------------------------------------------------------- #
# Coringa Gemini: ultimo recurso REAL antes do mock
# --------------------------------------------------------------------------- #
def test_coringa_entra_entre_a_categoria_e_o_mock() -> None:
    """Ordem estrita: categoria -> coringa -> mock."""
    fakes = _fakes("modelo-codigo-1", "modelo-codigo-2")
    coringa = ProviderFalso("gemini-flash-latest")
    mock = MockProvider()
    router = _router(fakes, coringa=coringa, cadeia_global=[mock])
    client = LLMClient(providers=list(fakes.values()), router=router, retries=0)

    cadeia = client._providers_para("experiment_plan")

    nomes = [p.name for p in cadeia]
    assert nomes == ["modelo-codigo-1", "modelo-codigo-2", "gemini-flash-latest", "mock"]


def test_coringa_assume_quando_a_categoria_toda_falha() -> None:
    fakes = _fakes("modelo-codigo-1", "modelo-codigo-2", erros={
        "modelo-codigo-1": RuntimeError("HTTP 429: rate limit"),
        "modelo-codigo-2": RuntimeError("HTTP 503: unavailable"),
    })
    coringa = ProviderFalso("gemini-flash-latest")
    mock = MockProvider()
    router = _router(fakes, coringa=coringa, cadeia_global=[mock])
    client = LLMClient(providers=list(fakes.values()), router=router, retries=0)

    resp = client.complete("p", task="experiment_plan")

    assert resp.provider == "gemini-flash-latest"  # nunca o mock
    assert len(coringa.chamadas) == 1
    assert coringa.chamadas[0]["max_tokens"] == 4096  # preset da categoria vale para o coringa


def test_coringa_so_cai_no_mock_se_ele_tambem_falhar() -> None:
    fakes = _fakes("modelo-codigo-1", erros={"modelo-codigo-1": RuntimeError("HTTP 429")})
    coringa = ProviderFalso("gemini-flash-latest", erro=RuntimeError("HTTP 500"))
    router = _router(fakes, coringa=coringa)
    client = LLMClient(providers=list(fakes.values()), router=router, retries=0)

    resp = client.complete("p", task="experiment_plan")

    assert resp.provider == "mock"


def test_sem_coringa_a_categoria_esgotada_vai_para_o_mock() -> None:
    """Sem GEMINI_API_KEY o coringa não existe; a cadeia global responde."""
    fakes = _fakes("modelo-codigo-1", erros={"modelo-codigo-1": RuntimeError("HTTP 429")})
    router = _router(fakes, coringa=None, cadeia_global=[MockProvider()])
    client = LLMClient(providers=list(fakes.values()), router=router, retries=0)

    resp = client.complete("p", task="experiment_plan")

    assert resp.provider == "mock"


def test_coringa_nao_e_tentado_duas_vezes() -> None:
    """Se o coringa já está na rota da categoria, não duplica na cascata."""
    fakes = _fakes("modelo-codigo-1")
    coringa = fakes["modelo-codigo-1"]
    router = _router(fakes, coringa=coringa)
    client = LLMClient(providers=list(fakes.values()), router=router, retries=0)

    cadeia = client._providers_para("experiment_plan")

    assert [p.name for p in cadeia].count("modelo-codigo-1") == 1


def test_coringa_torna_o_router_ativo_sozinho() -> None:
    """Só a GEMINI_API_KEY: ainda roteia (o coringa atende todas as tarefas)."""
    coringa = ProviderFalso("gemini-flash-latest")
    router = TaskRouter(
        rotas={},
        mapa_tarefas={"experiment_plan": "codigo"},
        cadeia_global=[MockProvider()],
        coringa=coringa,
    )
    assert router.ativo is True
    assert [p.name for p in router.rota("experiment_plan")] == ["gemini-flash-latest"]


def test_router_inativo_sem_coringa_nem_rota() -> None:
    router = TaskRouter(rotas={}, mapa_tarefas={}, cadeia_global=[MockProvider()], coringa=None)
    assert router.ativo is False


def test_coringa_entra_na_assinatura_de_cache() -> None:
    """A resposta do coringa não pode ser servida como cache do mock."""
    fakes = _fakes("modelo-codigo-1")
    com_coringa = LLMClient(
        providers=list(fakes.values()),
        router=_router(fakes, coringa=ProviderFalso("gemini-flash-latest")),
        retries=0,
    )
    sem_coringa = LLMClient(
        providers=list(fakes.values()),
        router=_router(fakes, coringa=None),
        retries=0,
    )
    assert com_coringa._signature() != sem_coringa._signature()


def test_build_task_router_le_o_coringa_do_catalogo() -> None:
    catalogo = json.loads(json.dumps(CATALOGO))
    catalogo["fallback_global"] = {"coringa": "modelo-redacao-1", "cadeia": ["mock"]}
    router = build_task_router(catalogo, {"fake": ProviderFalso("fake")})
    assert router is not None
    assert router.coringa is not None
    assert router.coringa.name == "modelo-redacao-1"
    # O coringa sai da cadeia global para não ser tentado duas vezes.
    assert [p.name for p in router.cadeia_global] == ["mock"]


def test_placeholder_de_env_nao_conta_como_chave(monkeypatch: pytest.MonkeyPatch) -> None:
    """Um .env recém-copiado não deve fingir que há uma chave.

    As variáveis de ambiente são limpas porque o provider cai para
    ``OPENAI_API_KEY`` quando a chave explícita é vazia — sem isso, outro teste
    que exporta essa variável mudaria o resultado conforme a ordem.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AI_CIENTISTA_API_KEY", raising=False)
    for valor in ("sua_chave_real_aqui", "CHANGEME", "  ", "", None):
        p = OpenAICompatibleProvider(name="x", model="x", api_key=valor)
        assert p.available() is False, f"{valor!r} não deveria contar como chave"
    assert OpenAICompatibleProvider(name="x", model="x", api_key="AIzaSyReal123").available() is True


def test_cliente_sem_router_mantem_comportamento_antigo() -> None:
    """Sem roteador, o cliente usa a cadeia de providers como antes."""
    p = ProviderFalso("modelo-codigo-1")
    client = LLMClient(providers=[p], router=None, retries=0)

    resp = client.complete("p", task="experiment_plan")

    assert resp.provider == "modelo-codigo-1"
    assert "max_tokens" not in p.chamadas[0]  # sem roteador, sem preset
