"""Dataset de surto de dengue enriquecido com features reais de satélite.

Une duas fontes reais, sem nenhum dado sintético:

* **InfoDengue** (Fiocruz/FGV) — casos, clima e nível de alerta por semana
  epidemiológica e município.
* **NASA GIBS / MODIS NDVI** — índice de vegetação por pixel, agregado ao
  município (malha real do IBGE) e à semana (ver ``integrations.satellite_ndvi``).

Cuidados metodológicos que evitam vazamento temporal:

* a linha da semana ``t`` usa somente observações publicadas até ``t``;
* o alvo é o ``nivel`` da semana ``t + 1``;
* o NDVI usa o composto cujo período **terminou** antes de ``t`` (ver
  ``DEFASAGEM_COMPOSTO_DIAS``), porque o composto do MODIS é rotulado pelo
  primeiro dia do período de 16 dias e, sem a defasagem, incluiria dias futuros;
* a divisão treino/teste é cronológica, nunca aleatória.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np

from core.experiments.datasets import (
    FEATURES_DENGUE,
    LIMIAR_SURTO,
    RealDataset,
    RealDataUnavailableError,
    _fetch_with_cache,
    _proveniencia,
)
from integrations.brazilian_apis import InfoDengueAPI
from integrations.satellite_ndvi import (
    ExtratorNDVI,
    SateliteIndisponivelError,
    data_composto_disponivel,
)

logger = logging.getLogger(__name__)

# Features de satélite acrescentadas por município/semana.
FEATURES_SATELITE: tuple[str, ...] = (
    "ndvi_medio",
    "ndvi_std",
    "ndvi_min",
    "ndvi_max",
)

# O composto do MODIS é rotulado pelo primeiro dia do período de 16 dias; usar
# o rótulo mais próximo da semana incluiria dias posteriores a ela. Recuamos um
# período inteiro para garantir que o composto terminou antes da semana alvo.
DEFASAGEM_COMPOSTO_DIAS = 16


def _inicio_semana_epi(se: int) -> date:
    """Segunda-feira da semana epidemiológica ``AAAASS``."""
    ano, semana = divmod(int(se), 100)
    try:
        return date.fromisocalendar(ano, semana, 1)
    except ValueError:
        return date(ano, 1, 1) + timedelta(weeks=max(0, semana - 1))


@dataclass
class _CacheNDVI:
    """Cache em disco de NDVI por (município, data do composto)."""

    diretorio: Any
    extrator: ExtratorNDVI
    offline: bool = False

    def obter(self, geocode: int, dia: date) -> dict[str, Any]:
        dia = data_composto_disponivel(dia - timedelta(days=DEFASAGEM_COMPOSTO_DIAS))
        caminho = self.diretorio / f"ndvi_{geocode}_{dia.isoformat()}.json"
        if caminho.exists():
            return json.loads(caminho.read_text(encoding="utf-8"))
        if self.offline:
            raise SateliteIndisponivelError(
                f"sem NDVI em cache para {geocode} em {dia} (modo offline)"
            )
        dados = self.extrator.ndvi_municipio(geocode, dia)
        self.diretorio.mkdir(parents=True, exist_ok=True)
        caminho.write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")
        return dados


def build_dengue_satellite_dataset(
    municipios: dict[int, str] | None = None,
    anos: tuple[int, ...] = (2022, 2023, 2024),
    limiar_surto: int = LIMIAR_SURTO,
    cache_dir: Any = None,
    offline: bool = False,
    max_workers: int = 8,
) -> RealDataset | None:
    """Monta o dataset dengue + NDVI. Devolve ``None`` se faltar dado real.

    ``max_workers`` paraleliza apenas os downloads de tiles do GIBS; a agregação
    é determinística e a ordem das linhas é cronológica, então o resultado não
    depende do número de threads.
    """
    from pathlib import Path

    base = Path(cache_dir) if cache_dir else Path("data/dataset_cache")
    cliente = InfoDengueAPI(rate_limit_seconds=0.3)
    registros = _fetch_with_cache(
        "infodengue_satelite",
        base,
        offline,
        lambda: cliente.coletar(municipios=municipios, anos=anos),
    )
    if not registros:
        logger.warning("InfoDengue não devolveu semanas; dataset indisponível.")
        return None

    por_cidade: dict[int, list[dict[str, Any]]] = {}
    for registro in registros:
        por_cidade.setdefault(int(registro["geocode"]), []).append(registro)

    # Pares (semana t, semana t+1) com todos os atributos tabulares presentes.
    pares: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for geocode, serie in por_cidade.items():
        serie.sort(key=lambda x: x.get("SE", 0))
        for i in range(len(serie) - 1):
            atual, proximo = serie[i], serie[i + 1]
            if any(atual.get(campo) is None for campo in FEATURES_DENGUE):
                continue
            if proximo.get("nivel") is None:
                continue
            pares.append((geocode, atual, proximo))

    if len(pares) < 50:
        logger.warning("Apenas %s semanas completas; insuficiente.", len(pares))
        return None

    pares.sort(key=lambda p: (p[1].get("SE", 0), p[0]))

    # Todas as (município, semana) distintas, para baixar cada composto uma vez.
    alvos = sorted({(g, int(atual["SE"])) for g, atual, _ in pares})
    extrator = ExtratorNDVI(zoom=9)
    cache = _CacheNDVI(base, extrator, offline=offline)

    def _buscar(item: tuple[int, int]) -> tuple[tuple[int, int], dict[str, Any]]:
        geocode, se = item
        try:
            return item, cache.obter(geocode, _inicio_semana_epi(se))
        except Exception as exc:  # noqa: BLE001 - rede/cobertura
            logger.warning("NDVI indisponível para %s/%s: %s", geocode, se, exc)
            return item, {}

    mapa_ndvi: dict[tuple[int, int], dict[str, Any]] = {}
    if offline:
        for item in alvos:
            mapa_ndvi.update([_buscar(item)])
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for item, dados in pool.map(_buscar, alvos):
                mapa_ndvi[item] = dados

    linhas: list[list[float]] = []
    rotulos: list[int] = []
    satelite: list[list[float]] = []
    geocodes: list[int] = []
    semanas: list[int] = []
    descartadas_sem_ndvi = 0

    for geocode, atual, proximo in pares:
        ndvi = mapa_ndvi.get((geocode, int(atual["SE"]))) or {}
        if not ndvi or any(ndvi.get(c) is None for c in FEATURES_SATELITE):
            descartadas_sem_ndvi += 1
            continue
        linhas.append([float(atual[campo]) for campo in FEATURES_DENGUE])
        satelite.append([float(ndvi[c]) for c in FEATURES_SATELITE])
        rotulos.append(int(float(proximo["nivel"]) >= limiar_surto))
        geocodes.append(geocode)
        semanas.append(int(atual["SE"]))

    if len(linhas) < 50:
        logger.warning("Apenas %s linhas com NDVI; insuficiente.", len(linhas))
        return None

    X_tab = np.asarray(linhas, dtype=float)
    X_sat = np.asarray(satelite, dtype=float)
    y = np.asarray(rotulos, dtype=int)

    # Divisão temporal: 80% treino (semanas mais antigas), 20% teste (recentes).
    corte = int(len(y) * 0.8)
    nomes = [m for m in FEATURES_DENGUE] + list(FEATURES_SATELITE)
    X = np.hstack([X_tab, X_sat])

    meta = _proveniencia(
        "InfoDengue (Fiocruz/FGV) + NASA GIBS/MODIS NDVI (agregado por município do IBGE)",
        "https://info.dengue.mat.br/api/alertcity | https://gibs.earthdata.nasa.gov",
        n_amostras=int(len(y)),
        n_variaveis=int(X.shape[1]),
        variaveis=nomes,
        variaveis_tabulares=list(FEATURES_DENGUE),
        variaveis_satelite=list(FEATURES_SATELITE),
        alvo=f"nivel da proxima semana >= {limiar_surto} (surto)",
        balanceamento={"positivos": int(y.sum()), "negativos": int((1 - y).sum())},
        municipios=sorted({m for m in [por_cidade[g][0]["municipio"] for g in geocodes]}),
        n_municipios=len(set(geocodes)),
        anos=list(anos),
        intervalo_semanas={"inicio": min(semanas), "fim": max(semanas)},
        descartadas_sem_ndvi=descartadas_sem_ndvi,
        split="temporal",
        split_index=corte,
        defasagem_composto_dias=DEFASAGEM_COMPOSTO_DIAS,
        fonte_satelite={
            "camada": "MODIS_Terra_L3_NDVI_16Day",
            "servico": "NASA GIBS WMTS (publico, sem chave)",
            "malha_municipal": "IBGE API v3 /malhas/municipios (GeoJSON)",
            "colormap": "https://gibs.earthdata.nasa.gov/colormaps/v1.0/MODIS_L3_NDVI.xml",
            "agregacao_espacial": "pixels do tile filtrados pela fronteira real do municipio",
            "agregacao_temporal": "composto de 16 dias encerrado antes da semana epidemiologica",
        },
        coletado_em=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    logger.info(
        "Dataset dengue+NDVI: %s amostras, %s municipios, %.1f%% surtos, corte temporal em %s",
        len(y), len(set(geocodes)), 100 * float(y.mean()), corte,
    )
    return RealDataset(X=X, y=y, meta=meta)
