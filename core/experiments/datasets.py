"""Datasets construídos a partir de APIs públicas brasileiras.

O runner experimental original partia de um gerador sintético. Este módulo
fornece o caminho oposto: baixa **observações reais**, monta a matriz de
atributos e o alvo a partir delas e registra a proveniência nos metadados, de
modo que o manuscrito possa afirmar exatamente de onde veio cada número.

Primeiro caso implementado: prever surto de dengue na semana seguinte a partir
das semanas epidemiológicas do InfoDengue (Fiocruz/FGV), que já traz as
variáveis climáticas observadas (temperatura e umidade médias) usadas no modelo
de nowcasting da própria Fiocruz.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.utils import setup_logger
from integrations.brazilian_apis import InfoDengueAPI

logger = setup_logger(__name__)

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

# Palavras do tópico que direcionam para o dataset de dengue.
GATILHOS_DENGUE = ("dengue", "arbovirose", "aedes", "infodengue")


@dataclass
class RealDataset:
    """Matriz de atributos e alvo observados, com proveniência."""

    X: np.ndarray
    y: np.ndarray
    meta: dict[str, Any]


def build_dengue_outbreak_dataset(
    cidades: dict[int, str] | None = None,
    anos: tuple[int, ...] = (2022, 2023),
    limiar_surto: int = LIMIAR_SURTO,
    api: InfoDengueAPI | None = None,
) -> RealDataset | None:
    """Monta o dataset de surto de dengue a partir do InfoDengue.

    Alvo: ``nivel`` da semana seguinte ``>= limiar_surto``. Cada linha usa apenas
    informação disponível até a semana t, e a validação é temporal (ver
    ``meta["split"]``), então não há vazamento do futuro para o treino.

    Devolve ``None`` quando a fonte não devolve semanas suficientes — o runner
    decide explicitamente o que fazer, em vez de substituir por dados gerados em
    silêncio.
    """
    cliente = api or InfoDengueAPI(rate_limit_seconds=0.3)
    registros = cliente.coletar(municipios=cidades, anos=anos)
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
    meta = {
        "origem": "InfoDengue (Fiocruz/FGV) — semanas epidemiológicas observadas",
        "fonte_url": "https://info.dengue.mat.br/api/alertcity",
        "tipo": "dados_reais",
        "n_amostras": int(len(y)),
        "n_variaveis": int(X.shape[1]),
        "variaveis": list(FEATURES_DENGUE),
        "alvo": f"nivel da proxima semana >= {limiar_surto} (surto)",
        "balanceamento": {"positivos": int(y.sum()), "negativos": int((1 - y).sum())},
        "municipios": municipios,
        "n_municipios": len(municipios),
        "anos": list(anos),
        "intervalo_semanas": {"inicio": min(semanas), "fim": max(semanas)},
        "amostras_descartadas_por_dado_ausente": descartadas,
        "coletado_em": time.strftime("%Y-%m-%dT%H:%M:%S"),
        # Corte temporal: o runner treina no passado e testa no futuro.
        "split": "temporal",
        # Baseline honesto para série temporal: persistência do nível atual.
        "baseline": {"tipo": "persistencia", "coluna": "nivel", "limiar": limiar_surto},
    }
    logger.info(
        "Dataset real (InfoDengue): %s amostras, %s municípios, %.1f%% surtos",
        len(y),
        len(municipios),
        100 * float(y.mean()),
    )
    return RealDataset(X=X, y=y, meta=meta)


def build_real_dataset(topic: str) -> RealDataset | None:
    """Seleciona o dataset real adequado ao tópico; ``None`` se não houver.

    Só a dengue está implementada. Novos domínios entram aqui conforme ganham
    uma fonte pública com observações reais.
    """
    texto = (topic or "").lower()
    if any(gatilho in texto for gatilho in GATILHOS_DENGUE):
        return build_dengue_outbreak_dataset()
    return None
