"""Testes herméticos do pipeline dengue + NDVI (sem rede).

Os dados de pesquisa são 100% reais; aqui, nos testes, usamos matrizes pequenas
apenas para exercitar a lógica estatística e de agregação, nunca como dado de
pesquisa.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.experiments.dengue_experiment import executar_experimento
from core.experiments.dengue_satellite import _inicio_semana_epi
from integrations.satellite_ndvi import (
    EPOCA_COMPOSTO,
    _lonlat_para_tile,
    _tile_para_lonlat,
    data_composto_disponivel,
)


def test_inicio_semana_epi_retorna_segunda_feira() -> None:
    d = _inicio_semana_epi(202301)
    assert d.weekday() == 0  # segunda-feira
    assert d.year == 2023


def test_data_composto_disponivel_nunca_usa_futuro() -> None:
    from datetime import date

    # O composto devolvido deve começar antes (ou no) dia pedido.
    for dia in (date(2023, 1, 15), date(2023, 7, 1), date(2020, 1, 1)):
        composto = data_composto_disponivel(dia)
        assert composto <= dia
        assert (composto - EPOCA_COMPOSTO).days % 16 == 0


def test_tile_roundtrip_aproximado() -> None:
    """O ponto cai dentro do tile calculado para ele."""
    lat, lon = -22.9, -43.2
    x, y = _lonlat_para_tile(lon, lat, 6)
    lon_min, lon_max, lat_min, lat_max = _tile_para_lonlat(x, y, 6)
    assert lon_min <= lon <= lon_max
    assert lat_min <= lat <= lat_max


def test_experimento_reporta_teste_isolado_do_satelite() -> None:
    """O experimento deve expor B vs C (isola o satélite) e a decisão de H0/H1."""
    rng = np.random.default_rng(0)
    n, n_tab, n_sat = 200, 4, 2
    X = rng.normal(size=(n, n_tab + n_sat))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    meta = {
        "variaveis_tabulares": [f"t{i}" for i in range(n_tab)],
        "variaveis_satelite": [f"s{i}" for i in range(n_sat)],
        "origem": "teste", "fonte_url": "teste", "tipo": "dados_reais",
    }
    res = executar_experimento(X, y, [f"t{i}" for i in range(n_tab)] + [f"s{i}" for i in range(n_sat)], meta)
    assert "delta_f1_isolado_satelite" in res["hipotese"]
    assert "B vs C" in res["testes"]
    assert res["hipotese"]["decisao"] in (
        "rejeitar H0 (favorece H1)", "não rejeitar H0 (resultado nulo)"
    )
    assert res["seed"] == 42
    assert "ic95_delta" in res["hipotese"]
    # cenários A e C usam apenas tabulares; B usa tudo.
    assert res["metricas"]["A"]["n_features"] == n_tab
    assert res["metricas"]["B"]["n_features"] == n_tab + n_sat
    assert res["metricas"]["C"]["n_features"] == n_tab


def test_hipotese_marca_ic_que_inclui_zero() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 4))
    y = rng.integers(0, 2, size=120)
    meta = {"variaveis_tabulares": ["a", "b", "c", "d"], "origem": "t", "fonte_url": "t", "tipo": "dados_reais"}
    res = executar_experimento(X, y, ["a", "b", "c", "d"], meta)
    h = res["hipotese"]
    lo, hi = h["ic95_delta"]
    assert h["ic95_inclui_zero"] == (lo <= 0 <= hi)
    assert "observacao_estatistica" in h
