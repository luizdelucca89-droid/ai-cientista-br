"""Features reais de NDVI (satélite MODIS) agregadas por município e por semana.

Zero dado sintético: cada valor vem de um PNG real do NASA GIBS (WMTS público,
sem chave) e de uma malha municipal real do IBGE. As duas agregações exigidas
são explícitas:

* **espacial (pixel -> município):** o polígono real do IBGE é rasterizado sobre
  o tile e só os pixels dentro da fronteira entram na média. Sem isso, a média
  misturaria o município com o entorno.
* **temporal (16 dias -> semana epidemiológica):** o NDVI do MODIS é um
  composto de 16 dias; cada semana epidemiológica usa o composto de data igual
  ou imediatamente anterior, que é a informação disponível até aquela semana.
  Assim não há vazamento do futuro.

Os índices de paleta do PNG são convertidos para NDVI pelo colormap oficial do
GIBS (``MODIS_L3_NDVI.xml``), então o número reportado é o valor físico de NDVI,
não o índice de cor.
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import math
import urllib.request
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
from PIL import Image
from shapely.geometry import LineString, shape

logger = logging.getLogger(__name__)

USER_AGENT = "AI-Cientista-BR/1.0 (pesquisa academica; contato: pesquisa@exemplo.br)"
GIBS_WMTS = "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best"
CAMADA_NDVI = "MODIS_Terra_L3_NDVI_16Day"
MATRIX_SET = "GoogleMapsCompatible_Level9"
COLORMAP_URL = "https://gibs.earthdata.nasa.gov/colormaps/v1.0/MODIS_L3_NDVI.xml"
IBGE_MALHA = "https://servicodados.ibge.gov.br/api/v3/malhas/municipios/{cod}"
NDVI_VALIDO = (-0.2, 1.0)

# O composto de 16 dias é publicado a cada 16 dias a partir de 2000-02-18.
EPOCA_COMPOSTO = date(2000, 2, 18)
PERIODO_COMPOSTO = 16


class SateliteIndisponivelError(RuntimeError):
    """A fonte real de satélite falhou; nada é fabricado em seu lugar."""


def _get(url: str, timeout: int = 90) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resposta:
        corpo = resposta.read()
        if resposta.headers.get("Content-Encoding", "").lower() == "gzip":
            # A API de malhas do IBGE responde gzip; urllib não descomprime.
            return gzip.decompress(corpo)
        return corpo


# --------------------------------------------------------------------------- #
# Colormap oficial: índice de paleta -> valor físico de NDVI
# --------------------------------------------------------------------------- #
def carregar_colormap(timeout: int = 60) -> list[tuple[tuple[int, int, int], float]]:
    """Lê o colormap oficial do GIBS e devolve (RGB -> NDVI) por entrada."""
    import re

    xml = _get(COLORMAP_URL, timeout=timeout).decode("utf-8", "replace")
    entradas: list[tuple[tuple[int, int, int], float]] = []
    for bruto in re.findall(r"<ColorMapEntry[^>]*/>", xml):
        if 'label="Fill"' in bruto:
            continue
        rgb_m = re.search(r'rgb="(\d+),(\d+),(\d+)"', bruto)
        val_m = re.search(r'value="\[([-0-9.]+),([-0-9.]+)\)"', bruto)
        if not (rgb_m and val_m):
            continue
        rgb = tuple(int(g) for g in rgb_m.groups())
        lo, hi = float(val_m.group(1)), float(val_m.group(2))
        entradas.append((rgb, (lo + hi) / 2.0))
    if not entradas:
        raise SateliteIndisponivelError("colormap do GIBS sem entradas utilizáveis")
    return entradas


def _tabela_rgb_para_ndvi(
    colormap: list[tuple[tuple[int, int, int], float]],
) -> dict[tuple[int, int, int], float]:
    return {rgb: ndvi for rgb, ndvi in colormap}


# --------------------------------------------------------------------------- #
# Geometria
# --------------------------------------------------------------------------- #
def malha_municipio(codigo_ibge: int, timeout: int = 60) -> dict[str, Any]:
    """Baixa a fronteira real do município no IBGE (GeoJSON)."""
    url = IBGE_MALHA.format(cod=codigo_ibge) + "?formato=application/vnd.geo+json"
    dados = json.loads(_get(url, timeout=timeout).decode("utf-8", "replace"))
    return dados


def _lonlat_para_tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    n = 2**z
    x = int((lon + 180.0) / 360.0 * n)
    lat_r = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n)
    return max(0, min(n - 1, x)), max(0, min(n - 1, y))


def _tile_para_lonlat(x: int, y: int, z: int) -> tuple[float, float, float, float]:
    """Devolve (lon_min, lon_max, lat_min, lat_max) do tile em EPSG:3857->4326."""
    n = 2**z

    def lon(xt: float) -> float:
        return xt / n * 360.0 - 180.0

    def lat(yt: float) -> float:
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yt / n))))

    return lon(x), lon(x + 1), lat(y + 1), lat(y)


def _composto_para_data(indice: int) -> date:
    return EPOCA_COMPOSTO + timedelta(days=PERIODO_COMPOSTO * indice)


def data_composto_disponivel(dia: date) -> date:
    """Composto de 16 dias mais recente publicado até ``dia`` (sem futuro)."""
    k = (dia - EPOCA_COMPOSTO).days // PERIODO_COMPOSTO
    return _composto_para_data(max(0, k))


# --------------------------------------------------------------------------- #
# Extração
# --------------------------------------------------------------------------- #
@dataclass
class ExtratorNDVI:
    """Baixa tiles reais do GIBS e agrega NDVI por município."""

    zoom: int = 9
    timeout: int = 90
    _colormap: dict[tuple[int, int, int], float] | None = field(default=None, repr=False)
    _cache_poligono: dict[int, Any] = field(default_factory=dict, repr=False)

    def _tabela(self) -> dict[tuple[int, int, int], float]:
        if self._colormap is None:
            self._colormap = _tabela_rgb_para_ndvi(carregar_colormap(self.timeout))
        return self._colormap

    def poligono(self, codigo_ibge: int) -> Any:
        if codigo_ibge not in self._cache_poligono:
            geo = malha_municipio(codigo_ibge, self.timeout)
            self._cache_poligono[codigo_ibge] = shape(geo["features"][0]["geometry"])
        return self._cache_poligono[codigo_ibge]

    def _tiles_da_geometria(self, geom: Any) -> list[tuple[int, int]]:
        lon_min, lat_min, lon_max, lat_max = geom.bounds
        x0, y0 = _lonlat_para_tile(lon_min, lat_max, self.zoom)
        x1, y1 = _lonlat_para_tile(lon_max, lat_min, self.zoom)
        return [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]

    def _baixar_tile(self, x: int, y: int, dia: date) -> Image.Image:
        url = f"{GIBS_WMTS}/{CAMADA_NDVI}/default/{dia.isoformat()}/{MATRIX_SET}/{self.zoom}/{y}/{x}.png"
        return Image.open(io.BytesIO(_get(url, self.timeout)))

    def ndvi_municipio(self, codigo_ibge: int, dia: date) -> dict[str, Any]:
        """NDVI médio real dentro da fronteira do município, para uma data.

        Agregação espacial: os pixels do(s) tile(s) são filtrados pelo polígono
        real do IBGE antes da média. Pixels de "Fill" (fora de cobertura) e
        valores fora de ``NDVI_VALIDO`` são descartados.
        """
        dia = data_composto_disponivel(dia)
        geom = self.poligono(codigo_ibge)
        tabela = self._tabela()
        valores: list[float] = []
        n_pixels_poligono = 0
        tiles_usados: list[str] = []

        for tx, ty in self._tiles_da_geometria(geom):
            try:
                img = self._baixar_tile(tx, ty, dia)
            except Exception as exc:  # noqa: BLE001 - rede
                logger.warning("tile %s/%s indisponível: %s", tx, ty, exc)
                continue
            tiles_usados.append(f"{self.zoom}/{ty}/{tx}")
            rgb = img.convert("RGB")
            arr = np.asarray(rgb, dtype=np.int16)
            h, w = arr.shape[:2]
            lon_min, lon_max, lat_min, lat_max = _tile_para_lonlat(tx, ty, self.zoom)

            # Máscara vetorizada: para cada linha (latitude) resolve a fronteira.
            lats = np.linspace(lat_max, lat_min, h, endpoint=False) + (lat_max - lat_min) / (2 * h)
            lons = np.linspace(lon_min, lon_max, w, endpoint=False) + (lon_max - lon_min) / (2 * w)
            for i, la in enumerate(lats):
                xs = geom.intersection(LineString([(lon_min, la), (lon_max, la)]))
                if xs.is_empty:
                    continue
                intervalos: list[tuple[float, float]] = []
                geoms = getattr(xs, "geoms", [xs])
                for g in geoms:
                    if g.geom_type == "LineString" and not g.is_empty:
                        coords = list(g.coords)
                        intervalos.append((min(c[0] for c in coords), max(c[0] for c in coords)))
                if not intervalos:
                    continue
                j0 = np.searchsorted(lons, [iv[0] for iv in intervalos], side="left")
                j1 = np.searchsorted(lons, [iv[1] for iv in intervalos], side="right")
                for a, b in zip(j0, j1):
                    if b <= a:
                        continue
                    linha = arr[i, a:b]
                    for px in linha:
                        ndvi = tabela.get((int(px[0]), int(px[1]), int(px[2])))
                        n_pixels_poligono += 1
                        if ndvi is not None and NDVI_VALIDO[0] <= ndvi <= NDVI_VALIDO[1]:
                            valores.append(ndvi)

        if not valores:
            raise SateliteIndisponivelError(
                f"nenhum pixel de NDVI válido para o município {codigo_ibge} em {dia}"
            )
        arr_v = np.asarray(valores, dtype=float)
        return {
            "ndvi_medio": float(arr_v.mean()),
            "ndvi_std": float(arr_v.std()),
            "ndvi_min": float(arr_v.min()),
            "ndvi_max": float(arr_v.max()),
            "n_pixels_validos": int(arr_v.size),
            "n_pixels_no_poligono": int(n_pixels_poligono),
            "fracao_cobertura_valida": float(arr_v.size / n_pixels_poligono) if n_pixels_poligono else 0.0,
            "data_composto": dia.isoformat(),
            "tiles": tiles_usados,
            "fonte": "NASA GIBS WMTS — MODIS_Terra_L3_NDVI_16Day",
            "url_colormap": COLORMAP_URL,
            "malha": f"{IBGE_MALHA.format(cod=codigo_ibge)}?formato=application/vnd.geo+json",
        }

    def ndvi_semanas(
        self, codigo_ibge: int, semanas: list[int]
    ) -> dict[int, dict[str, Any]]:
        """NDVI por semana epidemiológica (agregação temporal 16 dias -> semana).

        Cada semana ``AAAASS`` usa o composto de data igual ou imediatamente
        anterior à semana. A mesma data pode servir a mais de uma semana quando
        a composição não muda — é o dado realmente disponível no momento.
        """
        resultado: dict[int, dict[str, Any]] = {}
        for se in semanas:
            ano, semana = divmod(se, 100)
            try:
                dia = date.fromisocalendar(ano, semana, 1)
            except ValueError:
                dia = date(ano, 1, 1) + timedelta(weeks=semana - 1)
            try:
                resultado[se] = self.ndvi_municipio(codigo_ibge, dia)
            except SateliteIndisponivelError as exc:
                logger.warning("semana %s sem NDVI: %s", se, exc)
                resultado[se] = {}
        return resultado
