"""Integração multimodal: features de imagem de satélite real do INPE.

Cruza os dados tabulares reais do PRODES (área desmatada por UF e ano) com
features extraídas de **imagens de satélite reais** do Brasil, servidas pelo
GeoServer WMS do TerraBrasilis/INPE:

* ``prodes-legal-amz:temporal_mosaic_legal_amazon`` — mosaico temporal de
  imagens de satélite (composições Landsat) da Amazônia Legal;
* ``prodes-legal-amz:yearly_deforestation`` — polígonos anuais de desmatamento,
  usados para recortar a região de interesse por UF e ano via ``CQL_FILTER``.

Regra de ouro: **zero dados sintéticos**. A imagem vem de uma URL pública e
estável do INPE; quando o INPE está indisponível e não há cache, a função falha
explicitamente em vez de fabricar pixels.

Extraem-se duas famílias de features, ambas reais:

* **textura** (``skimage.feature.graycomatrix``): contraste, dissimilaridade,
  homogeneidade, energia e correlação agregados sobre distâncias e ângulos;
* **cor/vegetação**: fração de vegetação pelo índice ExG (Excess Green), brilho
  e saturação médios.

Com ``usar_cnn=True``, acrescenta-se o embedding de 512 dimensões do ResNet18
pré-treinado (``torchvision``). O padrão é ``False``: com poucas UF×ano, um vetor
de 512 dimensões causaria sobreajuste; o vetor interpretável é a escolha
estatisticamente defensável.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from core.utils import setup_logger

if TYPE_CHECKING:  # evita import circular com core.experiments.datasets
    from core.experiments.datasets import RealDataset

logger = setup_logger(__name__)

# --------------------------------------------------------------------------- #
# Fonte real: WMS do TerraBrasilis (INPE)
# --------------------------------------------------------------------------- #
WMS_URL = "https://terrabrasilis.dpi.inpe.br/geoserver/wms"
CAMADA_MOSAICO = "prodes-legal-amz:temporal_mosaic_legal_amazon"
CAMADA_DESMATAMENTO = "prodes-legal-amz:yearly_deforestation"

# Bounding boxes (minx, miny, maxx, maxy) em EPSG:4326, por UF da Amazônia Legal.
# Valores reais cobrindo a extensão de cada estado.
UF_BBOX: dict[str, tuple[float, float, float, float]] = {
    "AC": (-73.99, -11.15, -66.62, -7.11),
    "AM": (-73.80, -9.82, -56.10, 2.25),
    "AP": (-54.88, -1.24, -49.86, 4.44),
    "MA": (-48.76, -10.26, -41.80, -1.04),
    "MT": (-61.63, -18.04, -50.22, -7.35),
    "PA": (-58.90, -9.84, -46.03, 2.59),
    "RO": (-66.81, -13.69, -59.77, -7.97),
    "RR": (-64.83, 0.65, -59.77, 5.27),
    "TO": (-50.74, -13.47, -45.70, -5.17),
}

# Propriedades de Haralick usadas na textura.
PROPS_TEXTURA = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")
DISTANCIAS = (1, 2, 4, 8)
ANGULOS = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)

# Limiar do ExG para separar vegetação do restante (escala 0-255 normalizada).
LIMIAR_EXG = 0.02

NOMES_FEATURES = (
    [f"textura_{p}_media" for p in PROPS_TEXTURA]
    + [f"textura_{p}_desvio" for p in PROPS_TEXTURA]
    + ["fracao_vegetacao", "brilho_medio", "saturacao_media"]
)


class ImagemSateliteIndisponivel(RuntimeError):
    """A imagem real de satélite não pôde ser obtida; nada é fabricado."""


# --------------------------------------------------------------------------- #
# Download da imagem real
# --------------------------------------------------------------------------- #
def _url_getmap(uf: str, ano: int, camada: str = CAMADA_MOSAICO, largura: int = 256) -> str:
    """Monta a URL GetMap do WMS recortando a UF e o ano pedidos.

    A camada de mosaico é um raster temporal: o ano é selecionado por ``TIME``
    (``CQL_FILTER`` não é suportado em rasters). A camada vetorial de polígonos
    anuais aceita ``CQL_FILTER`` por ``year``/``state``.
    """
    bbox = UF_BBOX.get(uf.upper())
    if bbox is None:
        raise ImagemSateliteIndisponivel(f"UF sem bounding box conhecido: {uf!r}")
    minx, miny, maxx, maxy = bbox
    from urllib.parse import urlencode

    params = {
        "service": "WMS",
        "version": "1.1.1",
        "request": "GetMap",
        "layers": camada,
        "styles": "",
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "width": largura,
        "height": largura,
        "srs": "EPSG:4326",
        "format": "image/png",
    }
    if camada == CAMADA_MOSAICO:
        params["TIME"] = f"{int(ano)}-01-01T00:00:00Z"
    else:
        params["CQL_FILTER"] = f"year={int(ano)} AND state='{uf.upper()}'"
    return f"{WMS_URL}?{urlencode(params)}"


def _baixar_png(url: str, timeout: int = 120) -> bytes:
    import requests

    resposta = requests.get(url, timeout=timeout)
    resposta.raise_for_status()
    conteudo = resposta.content
    if not conteudo.startswith(b"\x89PNG"):
        raise ImagemSateliteIndisponivel(
            f"O INPE não devolveu PNG para {url!r} (content-type={resposta.headers.get('content-type')})"
        )
    return conteudo


def _fetch_cached(nome_cache: str, cache_dir: Any, offline: bool, fetcher: Any) -> list[dict[str, Any]]:
    """Atalho para o cache de observações reais (import tardio, sem ciclo)."""
    from core.experiments.datasets import _fetch_with_cache

    return _fetch_with_cache(nome_cache, cache_dir, offline, fetcher)


def _proveniencia_local(origem: str, url: str, **extra: Any) -> dict[str, Any]:
    """Metadados de proveniência (import tardio, sem ciclo)."""
    from core.experiments.datasets import _proveniencia

    return _proveniencia(origem, url, **extra)


def baixar_imagem_satelite(
    uf: str,
    ano: int,
    cache_dir: Any = None,
    offline: bool = False,
    timeout: int = 120,
) -> tuple[bytes, dict[str, Any]]:
    """Baixa (ou reproduz do cache) a imagem real de satélite da UF/ano.

    Devolve ``(png_bytes, meta)``. Sem rede e sem cache, levanta
    :class:`ImagemSateliteIndisponivel` — jamais gera pixels.
    """
    uf = uf.upper()
    url = _url_getmap(uf, ano)
    nome_cache = f"sat_{uf}_{int(ano)}"

    def _fetcher() -> list[dict[str, Any]]:
        png = _baixar_png(url, timeout=timeout)
        return [{"uf": uf, "ano": int(ano), "png_b64": base64.b64encode(png).decode("ascii"), "url": url}]

    registros = _fetch_cached(nome_cache, cache_dir, offline, _fetcher)
    if not registros:
        raise ImagemSateliteIndisponivel(
            f"Sem imagem real de satélite para {uf}/{ano} e sem cache local "
            f"({nome_cache}). O sistema não gera imagens sintéticas."
        )
    registro = registros[0]
    png = base64.b64decode(registro["png_b64"])
    meta = {
        "uf": uf,
        "ano": int(ano),
        "fonte": "INPE/TerraBrasilis (WMS)",
        "url": registro.get("url") or url,
        "camada": CAMADA_MOSAICO,
        "bytes": len(png),
    }
    return png, meta


# --------------------------------------------------------------------------- #
# Extração de features
# --------------------------------------------------------------------------- #
def _features_textura(cinza: np.ndarray) -> np.ndarray:
    """Features de textura de Haralick agregadas (média e desvio)."""
    from skimage.feature import graycomatrix, graycoprops

    quantizada = (cinza.astype(np.uint16) * 32 // 256).astype(np.uint8)
    matriz = graycomatrix(
        quantizada,
        distances=list(DISTANCIAS),
        angles=list(ANGULOS),
        levels=32,
        symmetric=True,
        normed=True,
    )
    valores = []
    for prop in PROPS_TEXTURA:
        grade = graycoprops(matriz, prop)  # (distâncias, ângulos)
        valores.extend([float(grade.mean()), float(grade.std())])
    return np.array(valores, dtype=float)


def _features_cor(rgb: np.ndarray) -> np.ndarray:
    """Fração de vegetação (ExG), brilho e saturação médios."""
    r = rgb[:, :, 0].astype(float) / 255.0
    g = rgb[:, :, 1].astype(float) / 255.0
    b = rgb[:, :, 2].astype(float) / 255.0
    exg = 2 * g - r - b
    fracao_vegetacao = float((exg > LIMIAR_EXG).mean())
    brilho = float((0.299 * r + 0.587 * g + 0.114 * b).mean())
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    saturacao = float(np.where(maxc > 0, (maxc - minc) / np.where(maxc > 0, maxc, 1), 0).mean())
    return np.array([fracao_vegetacao, brilho, saturacao], dtype=float)


@dataclass
class _ExtratorCNN:
    """ResNet18 pré-treinado usado como extrator de embedding (512-d)."""

    _modelo: Any = None

    def modelo(self) -> Any:
        if self._modelo is None:
            import torch
            from torchvision.models import ResNet18_Weights, resnet18

            rede = resnet18(weights=ResNet18_Weights.DEFAULT)
            rede.fc = torch.nn.Identity()
            rede.eval()
            self._modelo = rede
        return self._modelo

    def extrair(self, rgb: np.ndarray) -> np.ndarray:
        import torch
        from PIL import Image
        from torchvision import transforms

        pre = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        tensor = pre(Image.fromarray(rgb.astype(np.uint8))).unsqueeze(0)
        with torch.no_grad():
            saida = self.modelo()(tensor)
        return saida.squeeze(0).numpy().astype(float)


_CNN = _ExtratorCNN()


def _normalizar(vetor: np.ndarray) -> np.ndarray:
    """Normalização L2; devolve o vetor inalterado se a norma for zero."""
    norma = float(np.linalg.norm(vetor))
    return vetor / norma if norma > 0 else vetor


def extract_satellite_features_detalhado(
    uf: str,
    ano: int,
    cache_dir: Any = None,
    offline: bool = False,
    usar_cnn: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extrai o vetor de features da imagem real de satélite da UF/ano.

    Devolve ``(vetor_normalizado, meta)``. O vetor padrão tem 13 dimensões
    (textura + cor/vegetação); com ``usar_cnn=True`` acrescenta o embedding de
    512 dimensões do ResNet18, totalizando 525.
    """
    from PIL import Image

    png, meta_img = baixar_imagem_satelite(uf, ano, cache_dir=cache_dir, offline=offline)
    imagem = Image.open(io.BytesIO(png)).convert("RGB")
    rgb = np.array(imagem)

    textura = _features_textura(np.array(imagem.convert("L")))
    cor = _features_cor(rgb)
    vetor = np.concatenate([textura, cor])

    if usar_cnn:
        vetor = np.concatenate([vetor, _CNN.extrair(rgb)])

    meta = {
        **meta_img,
        "dimensoes": int(vetor.shape[0]),
        "usar_cnn": bool(usar_cnn),
        "features": NOMES_FEATURES + (["resnet18_embedding_512"] if usar_cnn else []),
    }
    return _normalizar(vetor), meta


def extract_satellite_features(
    uf: str,
    ano: int,
    cache_dir: Any = None,
    offline: bool = False,
    usar_cnn: bool = False,
) -> np.ndarray:
    """Devolve apenas o vetor normalizado de features da imagem real (API pedida)."""
    vetor, _ = extract_satellite_features_detalhado(uf, ano, cache_dir, offline, usar_cnn)
    return vetor


# --------------------------------------------------------------------------- #
# Dataset multimodal
# --------------------------------------------------------------------------- #
def build_multimodal_dataset(
    topic: str = "",
    cache_dir: Any = None,
    offline: bool = False,
    usar_cnn: bool = False,
    extrator: Any = None,
) -> RealDataset | None:
    """Combina o painel tabular do PRODES com features de imagem real do INPE.

    Cada linha é (UF, ano t): features tabulares do desmatamento observado mais o
    vetor de features da imagem de satélite real daquela UF naquele ano. O alvo é
    se a área desmatada em t+1 supera a de t. Devolve ``None`` quando não há
    imagem real disponível para as UF×ano do painel — nunca inventa pixels.

    ``extrator`` permite trocar a extração de features (por exemplo, a variante
    com ``geoai``/``rasterio`` em :mod:`integrations.multimodal_geo`); o padrão é
    a extração a partir do PNG do WMS.
    """
    from core.experiments.datasets import build_deforestation_dataset

    extrair = extrator or extract_satellite_features_detalhado
    tabular = build_deforestation_dataset(cache_dir=cache_dir, offline=offline)
    if tabular is None:
        logger.warning("Painel tabular do PRODES indisponível; multimodal não construído.")
        return None

    # Reconstrói o painel tabular mantendo a ordem cronológica do PRODES, para
    # alinhar cada linha com a sua feature de imagem.
    ordem = _ordem_temporal(tabular, cache_dir)

    linhas_imagem: dict[tuple[str, int], np.ndarray] = {}
    metas_imagem: dict[str, Any] = {}
    nomes_features: list[str] = list(NOMES_FEATURES)
    fonte_imagem = "INPE/TerraBrasilis (WMS)"
    camada_imagem = CAMADA_MOSAICO
    for uf, ano in ordem:
        if (uf, ano) in linhas_imagem:
            continue
        try:
            vetor, meta = extrair(uf, ano, cache_dir=cache_dir, offline=offline, usar_cnn=usar_cnn)
        except ImagemSateliteIndisponivel as exc:
            logger.warning("Imagem real ausente para %s/%s: %s", uf, ano, exc)
            continue
        linhas_imagem[(uf, ano)] = vetor
        metas_imagem[f"{uf}_{ano}"] = meta
        nomes_features = list(meta.get("features") or nomes_features)
        fonte_imagem = meta.get("fonte") or fonte_imagem
        camada_imagem = meta.get("camada") or camada_imagem

    if not linhas_imagem:
        logger.warning("Nenhuma imagem real de satélite disponível; multimodal indisponível.")
        return None

    X_tab, y = tabular.X, tabular.y
    variaveis = list(tabular.meta.get("variaveis", []))

    linhas: list[np.ndarray] = []
    alvos: list[int] = []
    chaves: list[tuple[str, int]] = []
    dimensao = next(iter(linhas_imagem.values())).shape[0]
    for i, (uf, ano) in enumerate(ordem):
        vetor = linhas_imagem.get((uf, ano))
        if vetor is None:
            continue
        linhas.append(np.concatenate([X_tab[i], vetor]))
        alvos.append(int(y[i]))
        chaves.append((uf, ano))

    if len(alvos) < 20:
        logger.warning("Apenas %s linhas multimodais completas; insuficiente.", len(alvos))
        return None

    X = np.array(linhas, dtype=float)
    y_final = np.array(alvos)
    meta = _proveniencia_local(
        f"INPE/PRODES (tabular) + {fonte_imagem} (imagens de satélite reais)",
        WMS_URL,
        n_amostras=int(len(y_final)),
        n_variaveis=int(X.shape[1]),
        variaveis=variaveis + [f"imagem_{n}" for n in nomes_features],
        alvo="area desmatada em t+1 > area em t (pressão crescente)",
        balanceamento={"positivos": int(y_final.sum()), "negativos": int((1 - y_final).sum())},
        anos=sorted({ano for _, ano in chaves}),
        n_ufs=len({uf for uf, _ in chaves}),
        split="temporal",
        modalidade="multimodal (tabular + imagem de satélite real)",
        dimensao_features_imagem=int(dimensao),
        camada_imagem=camada_imagem,
        imagens={k: v.get("url") for k, v in metas_imagem.items()},
        baseline=tabular.meta.get("baseline"),
    )
    logger.info(
        "Dataset multimodal real: %s amostras, %s UFs, %s features (%s de imagem)",
        len(y_final), meta["n_ufs"], X.shape[1], dimensao,
    )
    from core.experiments.datasets import RealDataset

    return RealDataset(X=X, y=y_final, meta=meta)


def _ordem_temporal(tabular: RealDataset, cache_dir: Any = None) -> list[tuple[str, int]]:
    """Reconstrói a ordem (UF, ano) das linhas a partir do cache do PRODES.

    O ``build_deforestation_dataset`` ordena por (ano, UF); reproduzimos a mesma
    ordenação e os mesmos filtros para alinhar tabular e imagem linha a linha.
    """
    import json

    from core.experiments.datasets import _candidate_dirs

    registros: list[dict[str, Any]] = []
    for base in _candidate_dirs(cache_dir):
        caminho = base / "prodes_desmatamento.json"
        if caminho.exists():
            registros = json.loads(caminho.read_text(encoding="utf-8")).get("registros", [])
            break

    por_ano: dict[int, dict[str, float]] = {}
    for r in registros:
        if r.get("ano") is None or not r.get("uf") or r.get("area_km2") is None:
            continue
        por_ano.setdefault(int(r["ano"]), {})[r["uf"]] = float(r["area_km2"])

    anos_validos = sorted(por_ano)
    chaves: list[tuple[str, int]] = []
    for i in range(1, len(anos_validos) - 1):
        ano_atual, ano_prox = anos_validos[i], anos_validos[i + 1]
        ano_ant = anos_validos[i - 1]
        for uf in por_ano[ano_atual]:
            if uf not in por_ano[ano_ant] or uf not in por_ano[ano_prox]:
                continue
            chaves.append((uf, ano_atual))
    return sorted(chaves, key=lambda c: (c[1], c[0]))


__all__ = [
    "CAMADA_DESMATAMENTO",
    "CAMADA_MOSAICO",
    "ImagemSateliteIndisponivel",
    "NOMES_FEATURES",
    "UF_BBOX",
    "WMS_URL",
    "baixar_imagem_satelite",
    "build_multimodal_dataset",
    "extract_satellite_features",
    "extract_satellite_features_detalhado",
]
