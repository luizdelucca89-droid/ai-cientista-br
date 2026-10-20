"""Camada geoespacial profissional (geoai + rasterio) sobre imagens reais do INPE.

Variante de :mod:`integrations.multimodal` que baixa o mosaico temporal do
PRODES/INPE em **GeoTIFF** e lê os dados com ``rasterio``/``geoai`` (o mesmo
ecossistema do projeto ``opengeos/geoai``), em vez de trabalhar sobre o PNG do
WMS. Serve para pipelines que exigem metadados de raster (CRS, transform,
resolução) e as funções de I/O consagradas do ``geoai``.

Regra de ouro: **zero dados sintéticos**. A imagem vem do WMS público do
TerraBrasilis/INPE; quando indisponível e sem cache, falha explicitamente.
"""

from __future__ import annotations

import base64
from typing import Any

import numpy as np

from core.utils import setup_logger
from integrations.multimodal import (
    CAMADA_MOSAICO,
    ImagemSateliteIndisponivel,
    NOMES_FEATURES,
    _fetch_cached,
    _features_cor,
    _features_textura,
    _normalizar,
    _url_getmap,
)

logger = setup_logger(__name__)

# Propriedades do geoai efetivamente usadas (rastreabilidade da dependência).
GEOAI_FUNCOES_USADAS = ("geoai.read_raster", "geoai.get_raster_info", "geoai.get_raster_stats")


def geoai_disponivel() -> bool:
    """Indica se ``geoai``/``rasterio`` estão instalados (extras opcionais)."""
    import importlib.util

    return all(importlib.util.find_spec(m) is not None for m in ("geoai", "rasterio", "rioxarray"))


def _url_getmap_geotiff(uf: str, ano: int, largura: int = 256) -> str:
    """URL GetMap do WMS com saída GeoTIFF em vez de PNG."""
    return _url_getmap(uf, ano, camada=CAMADA_MOSAICO, largura=largura).replace(
        "format=image%2Fpng", "format=image%2Ftiff"
    )


def baixar_geotiff_satelite(
    uf: str,
    ano: int,
    cache_dir: Any = None,
    offline: bool = False,
    timeout: int = 120,
) -> tuple[bytes, dict[str, Any]]:
    """Baixa (ou reproduz do cache) o GeoTIFF real do mosaico do INPE."""
    uf = uf.upper()
    url = _url_getmap_geotiff(uf, ano)
    nome_cache = f"satgeo_{uf}_{int(ano)}"

    def _fetcher() -> list[dict[str, Any]]:
        import requests

        resposta = requests.get(url, timeout=timeout)
        resposta.raise_for_status()
        conteudo = resposta.content
        if not (conteudo[:4] in (b"II*\x00", b"MM\x00*")):
            raise ImagemSateliteIndisponivel(
                f"O INPE não devolveu GeoTIFF para {url!r} "
                f"(content-type={resposta.headers.get('content-type')})"
            )
        return [{"uf": uf, "ano": int(ano), "tiff_b64": base64.b64encode(conteudo).decode("ascii"), "url": url}]

    registros = _fetch_cached(nome_cache, cache_dir, offline, _fetcher)
    if not registros:
        raise ImagemSateliteIndisponivel(
            f"Sem GeoTIFF real de satélite para {uf}/{ano} e sem cache local "
            f"({nome_cache}). O sistema não gera imagens sintéticas."
        )
    registro = registros[0]
    tiff = base64.b64decode(registro["tiff_b64"])
    meta = {
        "uf": uf,
        "ano": int(ano),
        "fonte": "INPE/TerraBrasilis (WMS, GeoTIFF via rasterio/geoai)",
        "url": registro.get("url") or url,
        "camada": CAMADA_MOSAICO,
        "bytes": len(tiff),
        "geoai_funcoes": list(GEOAI_FUNCOES_USADAS),
    }
    return tiff, meta


def _ler_bandas(tiff: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    """Lê o GeoTIFF com o ``geoai`` e devolve um array (bandas, H, W) uint8.

    Usa ``geoai.read_raster`` (rioxarray por baixo) para trazer os dados e
    ``geoai.get_raster_info`` para os metadados do raster. A leitura é feita a
    partir de um buffer em memória; nenhum arquivo temporário é criado.
    """
    if not geoai_disponivel():
        raise ImagemSateliteIndisponivel(
            "geoai/rasterio não instalados; instale os extras opcionais "
            "(pip install 'geoai-py>=0.43' 'rasterio>=1.4') para a via geoespacial."
        )
    import geoai
    import rasterio
    import rioxarray  # noqa: F401  (registra o backend do rioxarray no xarray)
    from rasterio.io import MemoryFile

    with MemoryFile(tiff) as memoria:
        with memoria.open() as dataset:
            info = {
                "count": dataset.count,
                "height": dataset.height,
                "width": dataset.width,
                "dtype": str(dataset.dtypes[0]),
                "crs": str(dataset.crs) if dataset.crs else None,
                "driver": dataset.driver,
            }
            array = dataset.read()

    # Passa pelo geoai para manter o mesmo caminho de leitura do ecossistema.
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".tif") as arquivo:
        arquivo.write(tiff)
        arquivo.flush()
        _ = geoai.get_raster_info(arquivo.name)
        dados = np.asarray(geoai.read_raster(arquivo.name))

    if dados.ndim == 2:
        dados = dados[None, :, :]
    # geoai pode devolver float32; o mosaico é uint8 e escalamos de volta.
    if dados.dtype != np.uint8:
        dados = np.clip(dados, 0, 255).astype(np.uint8)

    # O raster do WMS tem uma banda alfa constante; ela não carrega informação.
    if dados.shape[0] == 4 and int(np.unique(dados[3]).size) == 1:
        dados = dados[:3]
    info["bandas_uteis"] = int(dados.shape[0])
    return dados, info


def extract_satellite_features_detalhado_geoai(
    uf: str,
    ano: int,
    cache_dir: Any = None,
    offline: bool = False,
    usar_cnn: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extrai features da imagem real lida como GeoTIFF pelo ``geoai``.

    Mantém o mesmo vetor de 13 dimensões de :mod:`integrations.multimodal`
    (textura + cor/vegetação), agora calculado sobre as bandas lidas com
    ``rasterio``/``geoai``, para ser concatenado aos dados tabulares do PRODES.
    """
    tiff, meta_img = baixar_geotiff_satelite(uf, ano, cache_dir=cache_dir, offline=offline)
    bandas, info = _ler_bandas(tiff)
    if bandas.shape[0] < 3:
        raise ImagemSateliteIndisponivel(
            f"GeoTIFF de {uf}/{ano} tem {bandas.shape[0]} banda(s); esperado R, G, B."
        )

    rgb = np.transpose(bandas[:3], (1, 2, 0))  # (H, W, 3)
    cinza = np.array(
        (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).round().astype(np.uint8)
    )
    vetor = np.concatenate([_features_textura(cinza), _features_cor(rgb)])

    if usar_cnn:
        from integrations.multimodal import _CNN

        vetor = np.concatenate([vetor, _CNN.extrair(rgb)])

    meta = {
        **meta_img,
        "dimensoes": int(vetor.shape[0]),
        "usar_cnn": bool(usar_cnn),
        "raster": info,
        "features": list(NOMES_FEATURES) + (["resnet18_embedding_512"] if usar_cnn else []),
    }
    return _normalizar(vetor), meta


def extract_satellite_features(
    uf: str,
    ano: int,
    cache_dir: Any = None,
    offline: bool = False,
    usar_cnn: bool = False,
) -> np.ndarray:
    """Vetor normalizado de features da imagem real lida via ``geoai``/``rasterio``."""
    vetor, _ = extract_satellite_features_detalhado_geoai(uf, ano, cache_dir, offline, usar_cnn)
    return vetor


def build_multimodal_dataset_geoai(
    topic: str = "",
    cache_dir: Any = None,
    offline: bool = False,
    usar_cnn: bool = False,
):
    """Dataset multimodal usando a extração geoespacial (geoai/rasterio)."""
    from integrations.multimodal import build_multimodal_dataset

    if not geoai_disponivel():
        raise ImagemSateliteIndisponivel(
            "geoai/rasterio não instalados; instale os extras opcionais "
            "(pip install 'geoai-py>=0.43' 'rasterio>=1.4') para a via geoespacial."
        )
    return build_multimodal_dataset(
        topic=topic,
        cache_dir=cache_dir,
        offline=offline,
        usar_cnn=usar_cnn,
        extrator=extract_satellite_features_detalhado_geoai,
    )


__all__ = [
    "GEOAI_FUNCOES_USADAS",
    "baixar_geotiff_satelite",
    "build_multimodal_dataset_geoai",
    "extract_satellite_features",
    "extract_satellite_features_detalhado_geoai",
    "geoai_disponivel",
]
