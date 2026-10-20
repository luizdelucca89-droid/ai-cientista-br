"""Experimento de rigor estatístico: 3 cenários, validação temporal, SHAP.

Compara três cenários sobre dados reais (InfoDengue + NDVI/MODIS):

* **A (baseline):** Regressão Logística com atributos tabulares.
* **B (proposta):** Gradient Boosting com tabulares **+** satélite (NDVI).
* **C (ablação):** Gradient Boosting apenas com tabulares.

Protocolo:

* ``seed=42`` em tudo que tem aleatoriedade;
* validação **estritamente temporal** (``TimeSeriesSplit`` com janela expansiva),
  sem embaralhar — não há vazamento do futuro para o treino;
* métricas F1, AUC-ROC, precisão e revocação;
* intervalo de confiança 95% por bootstrap (1000 reamostragens);
* teste de Wilcoxon pareado (B vs A, B vs C) sobre F1 reamostrado;
* tamanho de efeito de Cohen (d) sobre as diferenças pareadas;
* importância de atributos por SHAP (TreeExplainer) para o cenário B.

Quando os dados reais refutam a hipótese, o resultado nulo é reportado como
está: nenhum número é ajustado para "melhorar" a história.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

SEED = 42
N_BOOTSTRAP = 1000
N_FOLDS = 5
TEST_FRACTION = 0.2


@dataclass
class ResultadoCenario:
    nome: str
    descricao: str
    n_features: int
    f1: float
    auc: float
    precision: float
    recall: float
    ic95: dict[str, tuple[float, float]] = field(default_factory=dict)
    f1_folds: list[float] = field(default_factory=list)
    f1_bootstrap: np.ndarray | None = field(default=None, repr=False)


def _metricas(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else float("nan"),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def _modelo(cenario: str) -> Any:
    if cenario == "A":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
        ])
    return GradientBoostingClassifier(random_state=SEED)


def _f1_bootstrap(
    y_true: np.ndarray, y_pred: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    n = len(y_true)
    valores = np.empty(N_BOOTSTRAP, dtype=float)
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        valores[b] = f1_score(y_true[idx], y_pred[idx], zero_division=0)
    return valores


def _ic95(valores: np.ndarray) -> tuple[float, float]:
    return (float(np.percentile(valores, 2.5)), float(np.percentile(valores, 97.5)))


def _cohen_d(diff: np.ndarray) -> float:
    """d de Cohen para amostras pareadas: média(diff) / desvio(diff)."""
    sd = float(np.std(diff, ddof=1))
    return float(np.mean(diff) / sd) if sd > 0 else 0.0


def executar_experimento(
    X: np.ndarray,
    y: np.ndarray,
    nomes_features: list[str],
    meta: dict[str, Any],
    diretorio_saida: Any = None,
) -> dict[str, Any]:
    """Roda os 3 cenários e devolve resultados prontos para o paper."""
    from scipy.stats import wilcoxon

    rng = np.random.default_rng(SEED)
    corte = int(len(y) * (1 - TEST_FRACTION))
    X_tr, X_te = X[:corte], X[corte:]
    y_tr, y_te = y[:corte], y[corte:]

    n_tab = len(meta.get("variaveis_tabulares") or [])
    cenarios_cfg = {
        "A": ("A (baseline)", "Regressão Logística, atributos tabulares", slice(0, n_tab)),
        "B": ("B (proposta)", "Gradient Boosting, tabulares + satélite (NDVI)", slice(0, X.shape[1])),
        "C": ("C (ablação)", "Gradient Boosting, apenas tabulares", slice(0, n_tab)),
    }

    resultados: dict[str, ResultadoCenario] = {}
    for chave, (nome, descricao, cols) in cenarios_cfg.items():
        modelo = _modelo(chave)
        modelo.fit(X_tr[:, cols], y_tr)
        prob = modelo.predict_proba(X_te[:, cols])[:, 1]
        pred = (prob >= 0.5).astype(int)
        m = _metricas(y_te, pred, prob)
        f1_boot = _f1_bootstrap(y_te, pred, rng)
        resultados[chave] = ResultadoCenario(
            nome=nome,
            descricao=descricao,
            n_features=(cols.stop - cols.start),
            f1=m["f1"],
            auc=m["auc"],
            precision=m["precision"],
            recall=m["recall"],
            ic95={"f1": _ic95(f1_boot)},
            f1_bootstrap=f1_boot,
        )

    # Validação cruzada temporal (janela expansiva) — estabilidade entre folds.
    tscv = TimeSeriesSplit(n_splits=N_FOLDS)
    for chave, (_, _, cols) in cenarios_cfg.items():
        f1s: list[float] = []
        for tr_idx, te_idx in tscv.split(X):
            modelo = _modelo(chave)
            modelo.fit(X[tr_idx][:, cols], y[tr_idx])
            pred = modelo.predict(X[te_idx][:, cols])
            f1s.append(float(f1_score(y[te_idx], pred, zero_division=0)))
        resultados[chave].f1_folds = f1s

    # Wilcoxon pareado sobre F1 reamostrado (B vs A e B vs C).
    testes: dict[str, Any] = {}
    f1_b = resultados["B"].f1_bootstrap
    for outro, rotulo in [("A", "B vs A"), ("C", "B vs C")]:
        diff = f1_b - resultados[outro].f1_bootstrap
        if np.allclose(diff, 0):
            stat, p = float("nan"), 1.0
        else:
            stat, p = wilcoxon(diff)
        testes[rotulo] = {
            "wilcoxon_stat": float(stat),
            "p_valor": float(p),
            "delta_f1_medio": float(np.mean(diff)),
            "ic95_delta": list(_ic95(diff)),
            "cohen_d": _cohen_d(diff),
        }

    delta_f1 = testes["B vs A"]["delta_f1_medio"]
    p_valor = testes["B vs A"]["p_valor"]
    ic_delta = testes["B vs A"]["ic95_delta"]
    # H1: satélite melhora com ΔF1 > 0.02 e p < 0.05.
    # O teste decisivo da hipótese é B vs C (mesmo algoritmo, com e sem
    # satélite): B vs A mistura o efeito do satélite com a troca de algoritmo
    # (LogReg -> Gradient Boosting). Reportamos os dois.
    delta_isolado = testes["B vs C"]["delta_f1_medio"]
    ic_inclui_zero = ic_delta[0] <= 0.0 <= ic_delta[1]
    hipotese = {
        "H0": "adição de satélite NÃO melhora (ΔF1 ≤ 0.02, p ≥ 0.05)",
        "H1": "adição de satélite melhora (ΔF1 > 0.02, p < 0.05)",
        "teste_principal": "B vs C (isola o satélite: mesmo algoritmo, com e sem NDVI)",
        "delta_f1_observado": delta_f1,
        "delta_f1_isolado_satelite": delta_isolado,
        "p_valor_observado": p_valor,
        "ic95_delta": list(ic_delta),
        "ic95_inclui_zero": ic_inclui_zero,
        "limiar_delta": 0.02,
        "alpha": 0.05,
        "decisao": "rejeitar H0 (favorece H1)" if (delta_f1 > 0.02 and p_valor < 0.05)
        else "não rejeitar H0 (resultado nulo)",
        "observacao_estatistica": (
            "O IC 95% do ΔF1 inclui zero, então a direção do efeito não está "
            "firmemente estabelecida, apesar do p-valor de Wilcoxon. Wilcoxon "
            "testa a mediana das diferenças pareadas (sensível ao sinal "
            "consistente); o IC bootstrap testa a magnitude média. Com efeito "
            "pequeno e variância alta os dois divergem, e o resultado prático "
            "é de melhora não confiável."
        ) if ic_inclui_zero else (
            "O IC 95% do ΔF1 não inclui zero."
        ),
    }

    saida: dict[str, Any] = {
        "seed": SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "n_folds": N_FOLDS,
        "split": "temporal (TimeSeriesSplit expansivo + holdout cronológico)",
        "n_treino": int(corte),
        "n_teste": int(len(y) - corte),
        "n_positivos_teste": int(y_te.sum()),
        "metricas": {
            c: {
                "nome": r.nome,
                "descricao": r.descricao,
                "n_features": r.n_features,
                "f1": r.f1,
                "auc": r.auc,
                "precision": r.precision,
                "recall": r.recall,
                "ic95_f1": list(r.ic95["f1"]),
                "f1_folds": r.f1_folds,
                "f1_folds_media": float(np.mean(r.f1_folds)),
                "f1_folds_std": float(np.std(r.f1_folds)),
            }
            for c, r in resultados.items()
        },
        "testes": testes,
        "hipotese": hipotese,
        "nomes_features": nomes_features,
        "proveniencia_dados": {
            k: v for k, v in meta.items()
            if k in ("origem", "fonte_url", "tipo", "coletado_em", "n_amostras",
                     "n_variaveis", "split", "intervalo_semanas", "n_municipios",
                     "defasagem_composto_dias", "fonte_satelite")
        },
    }

    # SHAP do cenário B (interpretação mecanicista).
    saida["shap"] = _shap_cenario_b(X_tr, X_te, y_tr, nomes_features)

    if diretorio_saida:
        destino = Path(diretorio_saida)
        destino.mkdir(parents=True, exist_ok=True)
        (destino / "resultados_experimento.json").write_text(
            json.dumps(saida, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
    return saida


def gerar_figuras(
    X: np.ndarray, y: np.ndarray, nomes: list[str], meta: dict[str, Any], destino: Any
) -> dict[str, str]:
    """Gera as figuras do paper a partir dos dados reais (ROC, SHAP, métricas)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    corte = int(len(y) * (1 - TEST_FRACTION))
    X_tr, X_te, y_tr, y_te = X[:corte], X[corte:], y[:corte], y[corte:]
    n_tab = len(meta.get("variaveis_tabulares") or [])
    gerados: dict[str, str] = {}

    # ROC dos três cenários.
    fig, ax = plt.subplots(figsize=(6, 5))
    for chave, cols in [("A", slice(0, n_tab)), ("B", slice(0, X.shape[1])), ("C", slice(0, n_tab))]:
        modelo = _modelo(chave)
        modelo.fit(X_tr[:, cols], y_tr)
        prob = modelo.predict_proba(X_te[:, cols])[:, 1]
        fpr, tpr, _ = roc_curve(y_te, prob)
        auc = roc_auc_score(y_te, prob)
        ax.plot(fpr, tpr, label=f"Cenário {chave} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Aleatório")
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos")
    ax.set_title("Curva ROC — validação temporal (holdout por semana)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    caminho = destino / "fig_roc.png"
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    gerados["roc"] = str(caminho)

    # SHAP do cenário B.
    info_shap = _shap_cenario_b(X_tr, X_te, y_tr, nomes)
    if info_shap.get("disponivel"):
        fig, ax = plt.subplots(figsize=(6, 5))
        itens = list(info_shap["importancias"].items())[::-1]
        ax.barh([k for k, _ in itens], [v for _, v in itens])
        ax.set_xlabel("|SHAP| médio (impacto na predição)")
        ax.set_title("Importância de atributos — Cenário B (GB + satélite)")
        fig.tight_layout()
        caminho = destino / "fig_shap.png"
        fig.savefig(caminho, dpi=150)
        plt.close(fig)
        gerados["shap"] = str(caminho)

    return gerados


def _shap_cenario_b(
    X_tr: np.ndarray, X_te: np.ndarray, y_tr: np.ndarray, nomes: list[str]
) -> dict[str, Any]:
    """Importância SHAP do cenário B (GB com satélite)."""
    try:
        import shap
    except ImportError:  # pragma: no cover
        return {"disponivel": False, "motivo": "shap não instalado"}
    modelo = GradientBoostingClassifier(random_state=SEED)
    modelo.fit(X_tr, y_tr)
    explainer = shap.TreeExplainer(modelo)
    valores = explainer.shap_values(X_te)
    if isinstance(valores, list):
        valores = valores[1]
    importancias = np.abs(valores).mean(axis=0)
    ordem = np.argsort(importancias)[::-1]
    return {
        "disponivel": True,
        "importancias": {
            nomes[i]: float(importancias[i]) for i in ordem
        },
        "ranking": [nomes[i] for i in ordem],
        "n_amostras_explicadas": int(X_te.shape[0]),
    }
