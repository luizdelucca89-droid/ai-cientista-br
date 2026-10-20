"""Testes da camada multimodal (imagens de satélite reais do INPE).

Os testes marcados com ``network`` consultam o WMS do TerraBrasilis/INPE e
gravam as imagens reais em um cache temporário; os demais rodam sobre esse cache
e não tocam a rede. Nenhum pixel é sintetizado: quando a imagem real não está
disponível, o código deve falhar explicitamente.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from integrations.multimodal import (  # noqa: E402
    CAMADA_MOSAICO,
    NOMES_FEATURES,
    ImagemSateliteIndisponivel,
    _features_cor,
    _features_textura,
    _url_getmap,
    baixar_imagem_satelite,
    extract_satellite_features,
    extract_satellite_features_detalhado,
)
from integrations.multimodal_geo import (  # noqa: E402
    GEOAI_FUNCOES_USADAS,
    _url_getmap_geotiff,
    baixar_geotiff_satelite,
    extract_satellite_features_detalhado_geoai,
    geoai_disponivel,
)


# --------------------------------------------------------------------------- #
# Funções puras (sem rede)
# --------------------------------------------------------------------------- #
def _imagem_vegetacao() -> np.ndarray:
    """Gradiente verde sintético apenas para testar as funções puras de features."""
    altura, largura = 64, 64
    r = np.tile(np.linspace(20, 60, largura), (altura, 1))
    g = np.tile(np.linspace(120, 200, largura), (altura, 1))
    b = np.tile(np.linspace(20, 50, largura), (altura, 1))
    return np.dstack([r, g, b]).astype(np.uint8)


def test_url_getmap_usa_time_no_mosaico():
    url = _url_getmap("RO", 2021)
    assert "temporal_mosaic_legal_amazon" in url
    assert "TIME=2021-01-01T00%3A00%3A00Z" in url
    assert "CQL_FILTER" not in url


def test_url_getmap_cql_na_camada_vetorial():
    url = _url_getmap("PA", 2020, camada="prodes-legal-amz:yearly_deforestation")
    assert "CQL_FILTER" in url
    assert "year%3D2020" in url
    assert "state%3D%27PA%27" in url


def test_url_getmap_uf_desconhecida():
    with pytest.raises(ImagemSateliteIndisponivel):
        _url_getmap("XX", 2020)


def test_features_textura_dimensao_e_finitude():
    cinza = np.array(Image.fromarray(_imagem_vegetacao()).convert("L"))
    vetor = _features_textura(cinza)
    assert vetor.shape == (len(NOMES_FEATURES) - 3,)  # 5 props x (média+desvio)
    assert np.all(np.isfinite(vetor))


def test_features_cor_faixa_valida():
    vetor = _features_cor(_imagem_vegetacao())
    fracao, brilho, saturacao = vetor
    assert 0.0 <= fracao <= 1.0
    assert 0.0 <= brilho <= 1.0
    assert 0.0 <= saturacao <= 1.0
    # A imagem de teste é predominantemente verde.
    assert fracao > 0.9


# --------------------------------------------------------------------------- #
# Sem rede e sem cache: falha explícita, nunca dados sintéticos
# --------------------------------------------------------------------------- #
@pytest.fixture
def sem_cache_repo(tmp_path, monkeypatch):
    """Isola o cache: aponta o cache versionado para uma pasta vazia.

    O ``_candidate_dirs`` cai no ``data/dataset_cache`` do repositório quando o
    cache da execução não tem o registro; aqui zeramos esse fallback para
    simular "nenhuma imagem real disponível".
    """
    from core.experiments import datasets

    vazio = tmp_path / "cache_vazio"
    vazio.mkdir()
    monkeypatch.setattr(datasets, "DEFAULT_CACHE_DIR", vazio)
    return tmp_path


def test_offline_sem_cache_falha(sem_cache_repo):
    with pytest.raises(ImagemSateliteIndisponivel):
        baixar_imagem_satelite("RO", 2021, cache_dir=sem_cache_repo, offline=True)


def test_offline_sem_cache_falha_features(sem_cache_repo):
    with pytest.raises(ImagemSateliteIndisponivel):
        extract_satellite_features("RO", 2021, cache_dir=sem_cache_repo, offline=True)


def test_png_invalido_rejeitado(sem_cache_repo, monkeypatch):
    """Resposta que não é PNG deve ser rejeitada (evita gravar XML de erro)."""

    class RespostaFalsa:
        content = b"<ServiceExceptionReport>erro</ServiceExceptionReport>"
        headers = {"content-type": "application/vnd.ogc.se_xml"}

        def raise_for_status(self):
            return None

    monkeypatch.setattr("requests.get", lambda *a, **k: RespostaFalsa())
    with pytest.raises(ImagemSateliteIndisponivel):
        baixar_imagem_satelite("RO", 2021, cache_dir=sem_cache_repo, offline=False)


# --------------------------------------------------------------------------- #
# Cache real: usa uma imagem PNG real gravada no cache (fonte: INPE)
# --------------------------------------------------------------------------- #
@pytest.fixture()
def cache_com_imagem(tmp_path):
    """Grava no cache uma imagem real de satélite do INPE (se a rede permitir).

    Sem rede, o teste que depende desta fixture é pulado — não inventamos imagem.
    """
    import requests

    url = _url_getmap("RO", 2021)
    try:
        resposta = requests.get(url, timeout=60)
        resposta.raise_for_status()
        png = resposta.content
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"INPE indisponível: {exc}")
    if not png.startswith(b"\x89PNG"):
        pytest.skip("INPE não devolveu PNG")

    payload = {
        "fonte_gravada_em": "teste",
        "tipo": "observacoes_reais",
        "registros": [
            {
                "uf": "RO",
                "ano": 2021,
                "png_b64": base64.b64encode(png).decode("ascii"),
                "url": url,
            }
        ],
    }
    (tmp_path / "sat_RO_2021.json").write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


@pytest.mark.network
def test_baixar_imagem_real_do_inpe(tmp_path):
    png, meta = baixar_imagem_satelite("RO", 2021, cache_dir=tmp_path)
    assert png.startswith(b"\x89PNG")
    assert meta["fonte"].startswith("INPE")
    assert meta["bytes"] > 1000
    imagem = Image.open(io.BytesIO(png)).convert("RGB")
    assert imagem.size == (256, 256)
    # Uma imagem real de satélite tem variação espectral; uma tela branca não.
    assert np.array(imagem).std() > 5


@pytest.mark.network
def test_features_imagem_real_dimensao_e_norma(tmp_path):
    vetor, meta = extract_satellite_features_detalhado("RO", 2021, cache_dir=tmp_path)
    assert vetor.shape == (len(NOMES_FEATURES),)
    assert np.all(np.isfinite(vetor))
    assert abs(float(np.linalg.norm(vetor)) - 1.0) < 1e-9
    assert meta["dimensoes"] == len(NOMES_FEATURES)
    assert meta["camada"] == CAMADA_MOSAICO


@pytest.mark.network
def test_features_variam_por_ano(tmp_path):
    """Anos diferentes usam mosaicos diferentes; as features devem diferir."""
    v2021 = extract_satellite_features("RO", 2021, cache_dir=tmp_path)
    v2018 = extract_satellite_features("RO", 2018, cache_dir=tmp_path)
    assert not np.allclose(v2021, v2018)


def test_features_do_cache_offline(cache_com_imagem):
    vetor, meta = extract_satellite_features_detalhado(
        "RO", 2021, cache_dir=cache_com_imagem, offline=True
    )
    assert vetor.shape == (len(NOMES_FEATURES),)
    assert np.all(np.isfinite(vetor))
    assert meta["bytes"] > 1000


@pytest.mark.network
def test_cnn_resnet18_dimensao(tmp_path):
    """O embedding do ResNet18 (512-d) é acrescentado quando pedido."""
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch/torchvision não instalados (extra opcional).")
    vetor, meta = extract_satellite_features_detalhado(
        "RO", 2021, cache_dir=tmp_path, usar_cnn=True
    )
    assert vetor.shape == (len(NOMES_FEATURES) + 512,)
    assert np.all(np.isfinite(vetor))
    assert meta["usar_cnn"] is True


# --------------------------------------------------------------------------- #
# Camada geoespacial (geoai + rasterio)
# --------------------------------------------------------------------------- #
def test_url_getmap_geotiff():
    url = _url_getmap_geotiff("RO", 2021)
    assert "image%2Ftiff" in url
    assert "TIME=2021-01-01T00%3A00%3A00Z" in url


def test_geoai_geotiff_offline_sem_cache_falha(tmp_path):
    with pytest.raises(ImagemSateliteIndisponivel):
        baixar_geotiff_satelite("RO", 2021, cache_dir=tmp_path, offline=True)


@pytest.mark.network
def test_geoai_geotiff_real(tmp_path):
    if not geoai_disponivel():
        pytest.skip("geoai/rasterio não instalados.")
    tiff, meta = baixar_geotiff_satelite("RO", 2021, cache_dir=tmp_path)
    # Assinatura TIFF (little ou big endian).
    assert tiff[:4] in (b"II*\x00", b"MM\x00*")
    assert meta["bytes"] > 1000
    assert meta["fonte"].startswith("INPE")
    assert "geoai.read_raster" in meta["geoai_funcoes"]


@pytest.mark.network
def test_geoai_features_reais(tmp_path):
    if not geoai_disponivel():
        pytest.skip("geoai/rasterio não instalados.")
    vetor, meta = extract_satellite_features_detalhado_geoai("RO", 2021, cache_dir=tmp_path)
    assert vetor.shape == (len(NOMES_FEATURES),)
    assert abs(float(np.linalg.norm(vetor)) - 1.0) < 1e-9
    assert np.all(np.isfinite(vetor))
    # As bandas foram lidas pelo rasterio/geoai com metadados de raster.
    assert meta["raster"]["count"] >= 3
    assert meta["raster"]["width"] == 256
    assert set(GEOAI_FUNCOES_USADAS).issubset(set(meta["geoai_funcoes"]))


@pytest.mark.network
def test_dataset_multimodal_real(tmp_path):
    from integrations.multimodal import build_multimodal_dataset

    ds = build_multimodal_dataset(cache_dir=tmp_path, offline=False)
    if ds is None:
        pytest.skip("Sem imagens reais suficientes para montar o dataset multimodal.")
    assert ds.meta["modalidade"].startswith("multimodal")
    assert ds.meta["dimensao_features_imagem"] == len(NOMES_FEATURES)
    assert ds.X.shape[1] == len(ds.meta["variaveis"])
    assert np.all(np.isfinite(ds.X))
    assert set(np.unique(ds.y)).issubset({0, 1})
    assert ds.meta["tipo"] == "dados_reais"
