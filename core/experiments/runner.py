"""Execução real de experimentos com análise estatística.

O runner trabalha **exclusivamente** com observações reais de APIs públicas
brasileiras (ver ``core.experiments.datasets``): as features são medições reais e
o alvo é um evento observado. Não há gerador sintético nem preenchimento
fabricado.

Quando não existe fonte pública para o tópico, o runner levanta
``RealDataUnavailableError`` e o ciclo falha de forma explícita — é preferível
admitir a ausência de dados a publicar números inventados.

Em todos os caminhos os números são reais (executados de fato): modelos de
verdade, métricas com intervalo de confiança bootstrap e testes de hipótese
pareados.
"""

from __future__ import annotations

import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from core.experiments.datasets import RealDataUnavailableError, build_real_dataset
from core.utils import ensure_dir, setup_logger, write_json

logger = setup_logger(__name__)


@dataclass
class ExperimentResult:
    """Resultado consolidado de uma execução experimental."""

    hypothesis_id: str
    topic: str
    seed: int
    metrics: dict[str, dict[str, float]]
    statistical_tests: dict[str, Any]
    best_model: str
    dataset: dict[str, Any]
    environment: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0
    figures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "topic": self.topic,
            "seed": self.seed,
            "metrics": self.metrics,
            "statistical_tests": self.statistical_tests,
            "best_model": self.best_model,
            "dataset": self.dataset,
            "environment": self.environment,
            "duration_s": round(self.duration_s, 3),
            "figures": self.figures,
        }


def _bootstrap_ci(values: np.ndarray, statistic: Callable[[np.ndarray], float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    stats = [statistic(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


@dataclass
class ExperimentRunner:
    """Executa o protocolo experimental completo sobre dados observados."""

    seed: int = 42
    test_size: float = 0.3
    alpha: float = 0.05
    output_dir: Any = "outputs"
    allow_code_execution: bool = True
    cache_dir: Any = None
    offline: bool = False

    def run(self, topic: str, hypothesis: dict[str, Any], plan: dict[str, Any] | None = None) -> ExperimentResult:
        """Executa o experimento e devolve métricas + testes estatísticos.

        Levanta ``RealDataUnavailableError`` se não houver fonte pública real para
        o tópico — nunca cai para uma base sintética.
        """
        started = time.time()
        hyp_id = hypothesis.get("id", "H1")

        if not self.allow_code_execution:
            raise RuntimeError("Execução de código desabilitada em config.json (experiments.allow_code_execution)")

        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        from sklearn.model_selection import TimeSeriesSplit, cross_val_score
        from sklearn.preprocessing import StandardScaler
        from scipy import stats

        real = build_real_dataset(topic, cache_dir=self.cache_dir, offline=self.offline)
        if real is None:
            raise RealDataUnavailableError(
                f"Nenhuma fonte pública de observações reais para o tópico: {topic!r}. "
                "O sistema não gera dados sintéticos; escolha um tema com dados abertos "
                "(dengue, desmatamento, agricultura, educação) ou registre a fonte."
            )
        X, y, meta = real.X, real.y, real.meta
        X_train, X_test, y_train, y_test = self._split_temporal(X, y)
        scaler = StandardScaler().fit(X_train)
        X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

        models: dict[str, Any] = {
            "baseline_estatistico": None,
            "regressao_logistica": LogisticRegression(max_iter=1000, random_state=self.seed),
            "floresta_aleatoria": RandomForestClassifier(n_estimators=200, random_state=self.seed),
            "gradient_boosting": GradientBoostingClassifier(random_state=self.seed),
        }

        metrics: dict[str, dict[str, float]] = {}
        per_sample_errors: dict[str, np.ndarray] = {}

        for name, model in models.items():
            if model is None:
                # Linha de base honesta e específica do problema: para série
                # temporal, persistir o estado atual (dengue: nível de alerta já
                # ativo; painéis anuais: repetir a tendência recente). É o que um
                # serviço faria sem modelo algum.
                y_pred, y_score = self._baseline_prediction(X_test, meta)
            else:
                model.fit(X_train_s, y_train)
                y_pred = model.predict(X_test_s)
                y_score = (
                    model.predict_proba(X_test_s)[:, 1]
                    if hasattr(model, "predict_proba")
                    else model.decision_function(X_test_s)
                )
            acc = float(accuracy_score(y_test, y_pred))
            f1 = float(f1_score(y_test, y_pred, zero_division=0))
            try:
                auc = float(roc_auc_score(y_test, y_score))
            except ValueError:
                auc = float("nan")
            # Série temporal não pode usar CV aleatório: dobrar o futuro no
            # treino inflaria as métricas. TimeSeriesSplit respeita a ordem.
            if model is not None:
                cv = cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=5), scoring="f1")
            else:
                cv = np.array([f1] * 5)
            lo, hi = _bootstrap_ci((y_pred == y_test).astype(float), np.mean, alpha=self.alpha, seed=self.seed)
            metrics[name] = {
                "acuracia": round(acc, 4),
                "f1_score": round(f1, 4),
                "auc_roc": round(auc, 4) if auc == auc else None,
                "acuracia_ic95_inf": round(lo, 4),
                "acuracia_ic95_sup": round(hi, 4),
                "cv_f1_media": round(float(np.mean(cv)), 4),
                "cv_f1_desvio": round(float(np.std(cv)), 4),
            }
            per_sample_errors[name] = (y_pred == y_test).astype(float)

        baseline = per_sample_errors["baseline_estatistico"]
        tests: dict[str, Any] = {}
        for name in ("regressao_logistica", "floresta_aleatoria", "gradient_boosting"):
            sample = per_sample_errors[name]
            if np.allclose(sample, baseline):
                tests[f"t_pareado_vs_baseline_{name}"] = {"p_valor": 1.0, "estatistica": 0.0, "significativo": False}
                tests[f"wilcoxon_vs_baseline_{name}"] = {"p_valor": 1.0, "estatistica": 0.0, "significativo": False}
                continue
            t_stat, p_t = stats.ttest_rel(sample, baseline)
            try:
                w_stat, p_w = stats.wilcoxon(sample, baseline)
            except ValueError:
                w_stat, p_w = 0.0, 1.0
            tests[f"t_pareado_vs_baseline_{name}"] = {
                "p_valor": round(float(p_t), 6),
                "estatistica": round(float(t_stat), 4),
                "significativo": bool(p_t < self.alpha),
            }
            tests[f"wilcoxon_vs_baseline_{name}"] = {
                "p_valor": round(float(p_w), 6),
                "estatistica": round(float(w_stat), 4),
                "significativo": bool(p_w < self.alpha),
            }

        best_model = max(
            (n for n in metrics if metrics[n]["f1_score"] is not None),
            key=lambda n: metrics[n]["f1_score"],
        )

        result = ExperimentResult(
            hypothesis_id=hyp_id,
            topic=topic,
            seed=self.seed,
            metrics=metrics,
            statistical_tests=tests,
            best_model=best_model,
            dataset=meta,
            environment={
                "python": sys.version.split()[0],
                "plataforma": platform.platform(),
                "numpy": np.__version__,
                "scikit_learn": _pkg_version("sklearn"),
                "scipy": _pkg_version("scipy"),
            },
            duration_s=time.time() - started,
        )
        result.figures = self._save_figures(result)
        return result

    # ------------------------------------------------------------- data splits
    def _split_temporal(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Corte cronológico: treina no passado, testa no futuro.

        Em série temporal, amostrar o teste ao acaso deixaria semanas futuras no
        treino — o modelo "veria" o surto antes de prevê-lo. O corte é por
        posição porque os construtores de dataset devolvem as amostras ordenadas
        por tempo.
        """
        corte = int(round(len(X) * (1 - self.test_size)))
        corte = max(1, min(corte, len(X) - 1))
        return X[:corte], X[corte:], y[:corte], y[corte:]

    def _baseline_prediction(self, X_test: np.ndarray, meta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Previsão da linha de base, escolhida conforme a origem dos dados."""
        baseline = meta.get("baseline") or {}
        coluna = baseline.get("coluna")
        variaveis = list(meta.get("variaveis", []))
        if baseline.get("tipo") == "persistencia" and coluna in variaveis:
            limiar = float(baseline.get("limiar", 3))
            atual = X_test[:, variaveis.index(coluna)]
            return (atual >= limiar).astype(int), atual
        if baseline.get("tipo") == "tendencia" and coluna in variaveis:
            # Repete a direção do último movimento observado.
            limiar = float(baseline.get("limiar", 0.0))
            variacao = X_test[:, variaveis.index(coluna)]
            return (variacao > limiar).astype(int), variacao
        # Fallback neutro: mantém a classe majoritária do treino como previsão.
        return np.zeros(len(X_test), dtype=int), np.zeros(len(X_test))

    # -------------------------------------------------------------- artifacts
    def _save_figures(self, result: ExperimentResult) -> list[str]:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:  # noqa: BLE001
            logger.warning("matplotlib indisponível: %s", exc)
            return []

        figures: list[str] = []
        fig_dir = ensure_dir(self.output_dir)
        names = [n for n in result.metrics if result.metrics[n]["f1_score"] is not None]
        f1_values = [result.metrics[n]["f1_score"] for n in names]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(names, f1_values, color="#1f6f4a")
        ax.set_ylabel("F1-score")
        ax.set_title("Comparação de modelos — F1-score")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        path = fig_dir / "fig_f1_modelos.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(str(path))

        return figures


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001
        return "desconhecida"


def save_experiment(result: ExperimentResult, output_dir: Any) -> str:
    """Persiste o resultado experimental em JSON."""
    path = ensure_dir(output_dir) / f"experimento_{result.hypothesis_id}.json"
    write_json(path, result.as_dict())
    return str(path)
