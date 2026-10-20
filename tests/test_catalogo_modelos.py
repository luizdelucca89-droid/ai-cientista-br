"""Guarda-corpo do catálogo de modelos: config válido e sem segredos.

O ``config.modelos.json`` é a referência de quais modelos gratuitos passaram no
benchmark. Se ele for editado à mão com um erro de sintaxe, ou se alguém colar
uma chave de API nele, estes testes falham antes do pipeline quebrar.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
CONFIG = RAIZ / "config.modelos.json"
HARNESS = RAIZ / "benchmark_models.py"
DOC = RAIZ / "MODELOS_VALIDADOS.md"

# Padrões das chaves dos provedores usados no benchmark.
PADROES_SEGREDO = re.compile(
    r"(sk-or-v1-[A-Za-z0-9]{20,}|mstrl_[A-Za-z0-9_]{20,}|nvapi-[A-Za-z0-9_\-]{20,}|AIza[A-Za-z0-9_\-]{30,})"
)


def test_config_modelos_e_json_valido() -> None:
    dados = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert "roteamento_por_tarefa" in dados
    assert "providers" in dados


def test_roteamento_cobre_as_tarefas_dos_agentes() -> None:
    """As tarefas roteadas devem cobrir as que os agentes realmente emitem."""
    dados = json.loads(CONFIG.read_text(encoding="utf-8"))
    mapa = dados["tarefas_para_categoria"]
    roteadas = set(mapa)
    emitidas = set()
    for arquivo in list((RAIZ / "core" / "agents").glob("*.py")) + [RAIZ / "core" / "rebuttal_engine.py"]:
        emitidas.update(re.findall(r'task="([a-z_]+)"', arquivo.read_text(encoding="utf-8")))
    assert emitidas, "nenhuma tarefa encontrada nos agentes"
    faltando = emitidas - roteadas
    assert not faltando, f"tarefas sem rota no config.modelos.json: {sorted(faltando)}"
    # Toda categoria citada precisa existir em roteamento_por_tarefa.
    for tarefa, categoria in mapa.items():
        assert categoria in dados["roteamento_por_tarefa"], f"{tarefa} aponta para categoria inexistente: {categoria}"


def test_providers_do_config_usam_variavel_de_ambiente() -> None:
    """Nenhuma chave embutida: cada provider lê a chave de uma variável."""
    dados = json.loads(CONFIG.read_text(encoding="utf-8"))
    for provider in dados["providers"]:
        if provider["type"] == "mock":
            continue
        assert provider.get("api_key_env"), f"{provider['name']} sem api_key_env"
        assert provider.get("base_url"), f"{provider['name']} sem base_url"
        assert provider["base_url"].startswith("https://"), f"{provider['name']} não usa HTTPS"


def test_nenhum_segredo_no_repositorio() -> None:
    """Config, harness e doc não podem conter chave real."""
    for caminho in (CONFIG, HARNESS, DOC):
        texto = caminho.read_text(encoding="utf-8")
        achado = PADROES_SEGREDO.search(texto)
        assert achado is None, f"possível chave de API em {caminho.name}: {achado.group()[:12]}…"


def test_harness_nao_rebaixa_base_url_com_barra_final() -> None:
    """base_url com barra final gerava '//chat/completions' e 404 no Gemini."""
    texto = HARNESS.read_text(encoding="utf-8")
    assert "base_url.rstrip('/')" in texto
