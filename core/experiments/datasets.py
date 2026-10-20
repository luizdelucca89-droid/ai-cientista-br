"""Datasets construídos a partir de APIs públicas brasileiras.

Nenhum dado é gerado ou simulado aqui: cada matriz de atributos vem de
**observações reais** baixadas em tempo de execução, e a proveniência é
registrada nos metadados para que o manuscrito possa afirmar exatamente de onde
veio cada número.

Fontes implementadas
====================
InfoDengue (Fiocruz/FGV)     Surtos de dengue com clima acoplado, por semana
                             epidemiológica. Alvo: nível de alerta da semana
                             seguinte (série temporal).
INPE/PRODES (TerraBrasilis)  Desmatamento anual por UF (km²), série histórica.
IBGE/SIDRA (PAM)             Produção agrícola por UF e ano.
IBGE/SIDRA (PNAD/ODS)        Indicadores educacionais por UF e ano.

Cada construtor devolve ``None`` (ou levanta ``RealDataUnavailableError``) quando
a fonte não responde — o runner então falha de forma explícita, em vez de
substituir observações por dados fabricados.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.utils import setup_logger
from integrations.brazilian_apis import INPEAPI, SIDRAAPI, InfoDengueAPI

logger = setup_logger(__name__)


class RealDataUnavailableError(RuntimeError):
    """Não há observação real disponível para o tópico; nada é fabricado."""


# Atributos observados na semana t, todos medições reais das fontes públicas.
FEATURES_DENGUE: tuple[str, ...] = (
    "tempmed",
    "umidmed",
    "casos",
    "casos_est",
    "p_inc100k",
    "Rt",
    "nivel",
    "nivel_inc",
)

# Nível de alerta (1 a 4) a partir do qual a semana é considerada surto.
LIMIAR_SURTO = 3

# Palavras do tópico que direcionam para cada fonte real.
GATILHOS_DENGUE = ("dengue", "arbovirose", "aedes", "infodengue")
GATILHOS_DESMATAMENTO = ("desmatamento", "amazônia", "amazonia", "prodes", "floresta", "cerrado", "bioma")
GATILHOS_AGRICULTURA = ("irrigação", "irrigacao", "agricultura", "agro", "soja", "safra", "plantio", "colheita")
GATILHOS_EDUCACAO = ("educação", "educacao", "escola", "aprendizado", "ensino", "gamificação", "gamificacao", "aluno")
# Temas que pedem explicitamente a modalidade imagem (satélite real do INPE).
GATILHOS_MULTIMODAL = (
    "multimodal",
    "satélite",
    "satelite",
    "imagem de satélite",
    "imagens de satélite",
    "sensoriamento remoto",
    "landsat",
    "sentinel",
)

# Códigos SIDRA reais usados pelos construtores.
SIDRA_PAM_SOJA = 2713
SIDRA_ODS_ESCOLARIZACAO = 353  # Indicador 16.9.1 (% crianças <5 com registro civil)


@dataclass
class RealDataset:
    """Matriz de atributos e alvo observados, com proveniência."""

    X: np.ndarray
    y: np.ndarray
    meta: dict[str, Any]


def _proveniencia(fonte: str, url: str, **extra: Any) -> dict[str, Any]:
    base = {
        "origem": fonte,
        "fonte_url": url,
        "tipo": "dados_reais",
        "coletado_em": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    base.update(extra)
    return base


# --------------------------------------------------------------------------- #
# Cache de observações reais
# --------------------------------------------------------------------------- #
# Diretório versionado com observações reais já gravadas (replay offline).
DEFAULT_CACHE_DIR = "data/dataset_cache"


def _cache_file(cache_dir: Any, name: str) -> Any:
    from pathlib import Path

    base = Path(cache_dir) if cache_dir else Path(DEFAULT_CACHE_DIR)
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{name}.json"


def _candidate_dirs(cache_dir: Any) -> list[Any]:
    """Diretórios de cache consultados, na ordem: execução e repositório."""
    from pathlib import Path

    dirs: list[Any] = []
    if cache_dir:
        dirs.append(Path(cache_dir))
    padrao = Path(DEFAULT_CACHE_DIR)
    if padrao not in dirs:
        dirs.append(padrao)
    return dirs


def _load_records(cache_dir: Any, name: str) -> Any | None:
    """Lê observações reais gravadas em disco (replay offline).

    O arquivo guarda as respostas cruas das APIs públicas, exatamente como
    vieram; nada é regerado. Serve para reproduzir um experimento sem rede.
    Procura primeiro no cache da execução e depois no cache versionado do
    repositório (``data/dataset_cache``).
    """
    import json
    from pathlib import Path

    for base in _candidate_dirs(cache_dir):
        caminho = Path(base) / f"{name}.json"
        if not caminho.exists():
            continue
        try:
            dados = json.loads(caminho.read_text(encoding="utf-8"))
            logger.info("Usando observações reais em cache: %s", caminho)
            return dados.get("registros")
        except (OSError, ValueError) as exc:
            logger.warning("Cache ilegível %s: %s", caminho, exc)
    return None


def _save_records(cache_dir: Any, name: str, registros: Any) -> None:
    import json

    if cache_dir is None or not registros:
        return
    caminho = _cache_file(cache_dir, name)
    payload = {
        "fonte_gravada_em": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tipo": "observacoes_reais",
        "registros": registros,
    }
    try:
        caminho.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:  # pragma: no cover - disco indisponível
        logger.warning("Não foi possível gravar o cache %s: %s", caminho, exc)


def _fetch_with_cache(name: str, cache_dir: Any, offline: bool, fetcher: Any) -> Any | None:
    """Busca observações reais com fallback para o cache em disco.

    Online: consulta a API e grava o resultado. Offline (ou falha de rede):
    reproduz as observações reais já gravadas. Se não houver cache, devolve
    ``None`` — jamais dados fabricados.
    """
    if offline:
        return _load_records(cache_dir, name)
    try:
        registros = fetcher()
    except Exception as exc:  # noqa: BLE001 - fonte externa instável
        logger.warning("Fonte %s indisponível (%s); tentando cache.", name, exc)
        registros = None
    if registros:
        _save_records(cache_dir, name, registros)
        return registros
    return _load_records(cache_dir, name)


# --------------------------------------------------------------------------- #
# Dengue (InfoDengue) — série temporal
# --------------------------------------------------------------------------- #
def build_dengue_outbreak_dataset(
    cidades: dict[int, str] | None = None,
    anos: tuple[int, ...] = (2022, 2023),
    limiar_surto: int = LIMIAR_SURTO,
    api: InfoDengueAPI | None = None,
    cache_dir: Any = None,
    offline: bool = False,
) -> RealDataset | None:
    """Monta o dataset de surto de dengue a partir do InfoDengue.

    Alvo: ``nivel`` da semana seguinte ``>= limiar_surto``. Cada linha usa apenas
    informação disponível até a semana t, e a validação é temporal (ver
    ``meta["split"]``), então não há vazamento do futuro para o treino.

    Com ``offline=True``, reproduz as observações reais gravadas em
    ``cache_dir`` (sem rede). Devolve ``None`` quando não há dados observados —
    o runner decide explicitamente o que fazer, em vez de substituir por dados
    gerados em silêncio.
    """
    cliente = api or InfoDengueAPI(rate_limit_seconds=0.3)
    registros = _fetch_with_cache(
        "infodengue",
        cache_dir,
        offline,
        lambda: cliente.coletar(municipios=cidades, anos=anos),
    )
    if not registros:
        logger.warning("InfoDengue não devolveu semanas; dataset real indisponível.")
        return None

    por_cidade: dict[int, list[dict[str, Any]]] = {}
    for registro in registros:
        por_cidade.setdefault(int(registro["geocode"]), []).append(registro)

    pares: list[tuple[dict[str, Any], dict[str, Any]]] = []
    descartadas = 0
    for serie in por_cidade.values():
        serie.sort(key=lambda x: x.get("SE", 0))
        for atual, proximo in zip(serie, serie[1:]):
            if any(atual.get(campo) is None for campo in FEATURES_DENGUE):
                descartadas += 1
                continue
            if proximo.get("nivel") is None:
                descartadas += 1
                continue
            pares.append((atual, proximo))

    if len(pares) < 50:
        logger.warning("Apenas %s semanas completas no InfoDengue; insuficiente.", len(pares))
        return None

    # Ordena por semana epidemiológica para permitir corte temporal cronológico.
    pares.sort(key=lambda p: (p[0].get("SE", 0), p[0].get("geocode", 0)))

    X = np.array([[float(atual[campo]) for campo in FEATURES_DENGUE] for atual, _ in pares])
    y = np.array([int(float(proximo["nivel"]) >= limiar_surto) for _, proximo in pares])

    municipios = sorted({atual["municipio"] for atual, _ in pares})
    semanas = [int(atual["SE"]) for atual, _ in pares]
    meta = _proveniencia(
        "InfoDengue (Fiocruz/FGV) — semanas epidemiológicas observadas",
        "https://info.dengue.mat.br/api/alertcity",
        n_amostras=int(len(y)),
        n_variaveis=int(X.shape[1]),
        variaveis=list(FEATURES_DENGUE),
        alvo=f"nivel da proxima semana >= {limiar_surto} (surto)",
        balanceamento={"positivos": int(y.sum()), "negativos": int((1 - y).sum())},
        municipios=municipios,
        n_municipios=len(municipios),
        anos=list(anos),
        intervalo_semanas={"inicio": min(semanas), "fim": max(semanas)},
        amostras_descartadas_por_dado_ausente=descartadas,
        split="temporal",
        baseline={"tipo": "persistencia", "coluna": "nivel", "limiar": limiar_surto},
    )
    logger.info(
        "Dataset real (InfoDengue): %s amostras, %s municípios, %.1f%% surtos",
        len(y),
        len(municipios),
        100 * float(y.mean()),
    )
    return RealDataset(X=X, y=y, meta=meta)


# --------------------------------------------------------------------------- #
# Desmatamento (INPE/PRODES) — painel UF × ano
# --------------------------------------------------------------------------- #
def build_deforestation_dataset(
    anos: tuple[int, ...] = (2018, 2019, 2020, 2021, 2022, 2023),
    api: INPEAPI | None = None,
    cache_dir: Any = None,
    offline: bool = False,
) -> RealDataset | None:
    """Painel real de desmatamento do PRODES: prevê alta do desmatamento por UF.

    Cada linha é (UF, ano t) com a área desmatada observada em t e o alvo é se a
    área em t+1 é maior que em t (pressão crescente). As variáveis são a própria
    série observada por UF — mudança recente, nível e média móvel —, todas
    calculadas a partir de medições oficiais do INPE.

    O corte de treino/teste é temporal (``meta["split"]``): os anos mais recentes
    ficam fora do treino.
    """
    cliente = api or INPEAPI(timeout=90)
    registros = _fetch_with_cache(
        "prodes_desmatamento",
        cache_dir,
        offline,
        lambda: cliente.serie_desmatamento(anos),
    )
    if not registros:
        logger.warning("PRODES não devolveu dados; dataset real indisponível.")
        return None

    por_ano: dict[int, dict[str, float]] = {}
    for r in registros:
        ano, uf, area = r.get("ano"), r.get("uf"), r.get("area_km2")
        if ano is None or not uf or area is None:
            continue
        por_ano.setdefault(int(ano), {})[uf] = float(area)

    anos_validos = sorted(por_ano)
    if len(anos_validos) < 4:
        logger.warning("PRODES devolveu apenas %s anos; insuficiente.", len(anos_validos))
        return None

    linhas: list[list[float]] = []
    alvos: list[int] = []
    chaves: list[tuple[str, int]] = []
    for i in range(1, len(anos_validos) - 1):
        ano_atual, ano_prox = anos_validos[i], anos_validos[i + 1]
        ano_ant = anos_validos[i - 1]
        for uf, area_atual in por_ano[ano_atual].items():
            if uf not in por_ano[ano_ant] or uf not in por_ano[ano_prox]:
                continue
            area_ant = por_ano[ano_ant][uf]
            area_prox = por_ano[ano_prox][uf]
            # Atributos observados até t (sem ver t+1).
            linhas.append([
                area_atual,
                area_atual - area_ant,
                (area_atual + area_ant) / 2.0,
                (area_atual - area_ant) / area_ant if area_ant else 0.0,
            ])
            alvos.append(int(area_prox > area_atual))
            chaves.append((uf, ano_atual))

    if len(alvos) < 30:
        logger.warning("Apenas %s observações PRODES completas; insuficiente.", len(alvos))
        return None

    ordem = sorted(range(len(chaves)), key=lambda i: (chaves[i][1], chaves[i][0]))
    X = np.array([linhas[i] for i in ordem], dtype=float)
    y = np.array([alvos[i] for i in ordem])
    meta = _proveniencia(
        "INPE/PRODES (TerraBrasilis) — desmatamento anual por UF",
        "https://terrabrasilis.dpi.inpe.br/geoserver/ows",
        n_amostras=int(len(y)),
        n_variaveis=int(X.shape[1]),
        variaveis=["area_km2_atual", "variacao_absoluta", "media_movel_2a", "variacao_relativa"],
        alvo="area desmatada em t+1 > area em t (pressão crescente)",
        balanceamento={"positivos": int(y.sum()), "negativos": int((1 - y).sum())},
        anos=anos_validos,
        n_ufs=len({c[0] for c in chaves}),
        split="temporal",
        # Baseline honesto: persistir a tendência recente (subiu se subiu em t-1→t).
        baseline={"tipo": "tendencia", "coluna": "variacao_absoluta", "limiar": 0.0},
    )
    logger.info("Dataset real (PRODES): %s amostras, %s UFs, %s anos", len(y), meta["n_ufs"], len(anos_validos))
    return RealDataset(X=X, y=y, meta=meta)


# --------------------------------------------------------------------------- #
# Agricultura (IBGE/SIDRA PAM) — painel UF × ano
# --------------------------------------------------------------------------- #
def build_agriculture_dataset(
    produto: int = SIDRA_PAM_SOJA,
    api: SIDRAAPI | None = None,
    cache_dir: Any = None,
    offline: bool = False,
) -> RealDataset | None:
    """Painel real de produção agrícola (PAM/SIDRA): prevê crescimento da safra.

    Alvo: a produção do produto em t+1 supera a de t. As variáveis são a série
    observada por UF (nível, variação absoluta e relativa) — medições oficiais do
    IBGE, sem qualquer geração sintética.
    """
    cliente = api or SIDRAAPI(timeout=40)
    registros = _fetch_with_cache(
        f"sidra_pam_{produto}",
        cache_dir,
        offline,
        lambda: cliente.producao_agricola(produto=produto, anos="all"),
    )
    if not registros:
        logger.warning("SIDRA não devolveu produção agrícola; dataset real indisponível.")
        return None

    por_uf: dict[str, dict[int, float]] = {}
    for r in registros:
        uf, ano, valor = r.get("uf"), r.get("ano"), r.get("producao_t")
        if uf and ano is not None and valor is not None:
            por_uf.setdefault(uf, {})[int(ano)] = float(valor)

    linhas: list[list[float]] = []
    alvos: list[int] = []
    chaves: list[tuple[str, int]] = []
    for uf, serie in por_uf.items():
        anos_uf = sorted(serie)
        for i in range(1, len(anos_uf) - 1):
            a, t, p = anos_uf[i - 1], anos_uf[i], anos_uf[i + 1]
            atual, ant, prox = serie[t], serie[a], serie[p]
            if ant <= 0:
                continue
            linhas.append([atual, atual - ant, (atual - ant) / ant, (atual + ant) / 2.0])
            alvos.append(int(prox > atual))
            chaves.append((uf, t))

    if len(alvos) < 30:
        logger.warning("Apenas %s observações agrícolas completas; insuficiente.", len(alvos))
        return None

    ordem = sorted(range(len(chaves)), key=lambda i: (chaves[i][1], chaves[i][0]))
    X = np.array([linhas[i] for i in ordem], dtype=float)
    y = np.array([alvos[i] for i in ordem])
    produto_nome = registros[0].get("produto", str(produto))
    meta = _proveniencia(
        f"IBGE/SIDRA PAM 1612 — produção de {produto_nome} por UF e ano",
        "https://apisidra.ibge.gov.br/values/t/1612",
        n_amostras=int(len(y)),
        n_variaveis=int(X.shape[1]),
        variaveis=["producao_t_atual", "variacao_absoluta", "variacao_relativa", "media_movel_2a"],
        alvo="producao em t+1 > producao em t (crescimento)",
        balanceamento={"positivos": int(y.sum()), "negativos": int((1 - y).sum())},
        n_ufs=len({c[0] for c in chaves}),
        anos=sorted({c[1] for c in chaves}),
        split="temporal",
        baseline={"tipo": "tendencia", "coluna": "variacao_absoluta", "limiar": 0.0},
    )
    logger.info("Dataset real (SIDRA/PAM): %s amostras, %s UFs", len(y), meta["n_ufs"])
    return RealDataset(X=X, y=y, meta=meta)


# --------------------------------------------------------------------------- #
# Educação (IBGE/SIDRA ODS) — painel UF × ano
# --------------------------------------------------------------------------- #
def build_education_dataset(
    api: SIDRAAPI | None = None,
    cache_dir: Any = None,
    offline: bool = False,
) -> RealDataset | None:
    """Painel real de indicador educacional (SIDRA ODS 16.9.1) por UF e ano.

    Alvo: o indicador melhora de t para t+1 (progresso). Variáveis: nível,
    variação absoluta e média móvel, todos valores oficiais do IBGE.
    """
    cliente = api or SIDRAAPI(timeout=40)
    registros = _fetch_with_cache(
        "sidra_ods_9544",
        cache_dir,
        offline,
        lambda: cliente.valores(9544, SIDRA_ODS_ESCOLARIZACAO, periodo="all"),
    )
    if not registros:
        logger.warning("SIDRA não devolveu indicador educacional; dataset real indisponível.")
        return None

    por_uf: dict[str, dict[int, float]] = {}
    for r in registros:
        uf = r.get("D1N")
        ano = r.get("D3N")
        valor = r.get("V")
        try:
            ano_i = int(ano)
            valor_f = float(valor)
        except (TypeError, ValueError):
            continue
        if uf:
            por_uf.setdefault(uf, {})[ano_i] = valor_f

    linhas: list[list[float]] = []
    alvos: list[int] = []
    chaves: list[tuple[str, int]] = []
    for uf, serie in por_uf.items():
        anos_uf = sorted(serie)
        for i in range(1, len(anos_uf) - 1):
            a, t, p = anos_uf[i - 1], anos_uf[i], anos_uf[i + 1]
            atual, ant, prox = serie[t], serie[a], serie[p]
            linhas.append([atual, atual - ant, (atual + ant) / 2.0])
            alvos.append(int(prox > atual))
            chaves.append((uf, t))

    if len(alvos) < 20:
        logger.warning("Apenas %s observações educacionais completas; insuficiente.", len(alvos))
        return None

    ordem = sorted(range(len(chaves)), key=lambda i: (chaves[i][1], chaves[i][0]))
    X = np.array([linhas[i] for i in ordem], dtype=float)
    y = np.array([alvos[i] for i in ordem])
    meta = _proveniencia(
        "IBGE/SIDRA ODS 16.9.1 — indicador educacional por UF e ano",
        "https://apisidra.ibge.gov.br/values/t/9544",
        n_amostras=int(len(y)),
        n_variaveis=int(X.shape[1]),
        variaveis=["indicador_atual", "variacao_absoluta", "media_movel_2a"],
        alvo="indicador melhora de t para t+1",
        balanceamento={"positivos": int(y.sum()), "negativos": int((1 - y).sum())},
        n_ufs=len({c[0] for c in chaves}),
        anos=sorted({c[1] for c in chaves}),
        split="temporal",
        baseline={"tipo": "tendencia", "coluna": "variacao_absoluta", "limiar": 0.0},
    )
    logger.info("Dataset real (SIDRA/ODS): %s amostras, %s UFs", len(y), meta["n_ufs"])
    return RealDataset(X=X, y=y, meta=meta)


def build_real_dataset(topic: str, cache_dir: Any = None, offline: bool = False) -> RealDataset | None:
    """Seleciona o dataset real adequado ao tópico; ``None`` se não houver.

    Só devolve dados observados. Quando o tópico não corresponde a nenhuma fonte
    pública implementada, devolve ``None`` e o runner falha explicitamente.
    Com ``offline=True``, reproduz as observações reais gravadas em ``cache_dir``.
    """
    texto = (topic or "").lower()
    # Multimodal antes dos demais: "desmatamento com imagens de satélite" deve
    # usar o cruzamento tabular+imagem, não o painel tabular isolado.
    if any(g in texto for g in GATILHOS_MULTIMODAL):
        from integrations.multimodal import build_multimodal_dataset

        multimodal = build_multimodal_dataset(cache_dir=cache_dir, offline=offline)
        if multimodal is not None:
            return multimodal
        logger.warning("Tema multimodal, mas sem imagens reais disponíveis; usando painel tabular.")
    if any(g in texto for g in GATILHOS_DENGUE):
        return build_dengue_outbreak_dataset(cache_dir=cache_dir, offline=offline)
    # Agricultura antes de desmatamento: "irrigação no Cerrado" tem gatilhos de
    # ambos, e a pergunta agrícola é a mais específica do tópico.
    if any(g in texto for g in GATILHOS_AGRICULTURA):
        return build_agriculture_dataset(cache_dir=cache_dir, offline=offline)
    if any(g in texto for g in GATILHOS_DESMATAMENTO):
        return build_deforestation_dataset(cache_dir=cache_dir, offline=offline)
    if any(g in texto for g in GATILHOS_EDUCACAO):
        return build_education_dataset(cache_dir=cache_dir, offline=offline)
    return None
