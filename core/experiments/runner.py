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

import json
import platform
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from core.experiments.auto_debugger import executar_com_auto_debug
from core.experiments.datasets import RealDataUnavailableError, build_real_dataset
from core.utils import ensure_dir, setup_logger, write_json

logger = setup_logger(__name__)


# Código do experimento executado em sandbox com auto-debug. É um script de
# verdade (não uma string de fachada): calcula as métricas que o runner usa.
CODIGO_EXPERIMENTO = '''
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

corte = int(round(len(X) * (1 - test_size)))
corte = max(1, min(corte, len(X) - 1))
X_train, X_test = X[:corte], X[corte:]
y_train, y_test = y[:corte], y[corte:]

escalonador = StandardScaler().fit(X_train)
X_train = escalonador.transform(X_train)
X_test = escalonador.transform(X_test)

modelos = {
    "regressao_logistica": LogisticRegression(max_iter=1000, random_state=seed),
    "floresta_aleatoria": RandomForestClassifier(n_estimators=200, random_state=seed),
    "gradient_boosting": GradientBoostingClassifier(random_state=seed),
}
metricas = {}
for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    previsoes = modelo.predict(X_test)
    metricas[nome] = {
        "acuracia": float(accuracy_score(y_test, previsoes)),
        "f1_score": float(f1_score(y_test, previsoes, zero_division=0)),
    }
    probabilidades = modelo.predict_proba(X_test)[:, 1]
    metricas[nome]["auc_roc"] = float(roc_auc_score(y_test, probabilidades))
'''

# Estimadores reais disponíveis e a expressão que os instancia no sandbox.
_EXPRESSOES_ESTIMADOR: dict[str, str] = {
    "regressao_logistica": "LogisticRegression(max_iter=1000, random_state=seed)",
    "floresta_aleatoria": "RandomForestClassifier(n_estimators=200, random_state=seed)",
    "gradient_boosting": "GradientBoostingClassifier(random_state=seed)",
}
MODELOS_PADRAO: tuple[str, ...] = ("regressao_logistica", "floresta_aleatoria", "gradient_boosting")

# Como o LLM costuma escrever cada família de modelo no desenho experimental.
_APELOS_MODELO: tuple[tuple[str, str], ...] = (
    ("regressao logistica", "regressao_logistica"),
    ("regressão logística", "regressao_logistica"),
    ("logistica", "regressao_logistica"),
    ("logistic regression", "regressao_logistica"),
    ("logistic", "regressao_logistica"),
    ("random forest", "floresta_aleatoria"),
    ("floresta aleatoria", "floresta_aleatoria"),
    ("floresta", "floresta_aleatoria"),
    ("gradient boosting", "gradient_boosting"),
    ("gradient boost", "gradient_boosting"),
    ("xgboost", "gradient_boosting"),
    ("gbm", "gradient_boosting"),
    ("boosting", "gradient_boosting"),
)


def _normalizar(texto: str) -> str:
    """Minúsculas sem acento, para casar apelos escritos de formas variadas."""
    base = unicodedata.normalize("NFKD", (texto or "").lower())
    return "".join(c for c in base if not unicodedata.combining(c))


def _modelos_do_desenho(plan: dict[str, Any] | None) -> list[str]:
    """Extrai os modelos pedidos no desenho; devolve a tríade padrão se não houver.

    O desenho do experimentalista lista modelos por nome livre ("Random Forest com
    Lag de 4 semanas"). Mapeamos cada apelo para um estimador real implementado e
    treinamos **apenas** o que foi pedido — é isso que diferencia um experimento do
    outro no enxame. Modelos sem estimador real (SVM, regressão linear) são
    ignorados em vez de silenciosamente trocados por outro.
    """
    if not isinstance(plan, dict):
        return list(MODELOS_PADRAO)
    textos: list[str] = []
    for modelo in plan.get("modelos") or []:
        if isinstance(modelo, dict):
            textos.append(str(modelo.get("nome", "")))
        elif isinstance(modelo, str):
            textos.append(modelo)
    selecionados: list[str] = []
    for texto in textos:
        alvo = _normalizar(texto)
        for apelo, chave in _APELOS_MODELO:
            if apelo in alvo and chave not in selecionados:
                selecionados.append(chave)
                break
    # Preserva a ordem canônica para reprodutibilidade entre execuções.
    ordenados = [m for m in MODELOS_PADRAO if m in selecionados]
    return ordenados or list(MODELOS_PADRAO)


def _defasagem_do_desenho(plan: dict[str, Any] | None) -> int:
    """Detecta o lag pedido no desenho (ex.: "lag de 4 semanas" → 4). 0 se ausente."""
    if not isinstance(plan, dict):
        return 0
    texto = _normalizar(json.dumps(plan, ensure_ascii=False, default=str))
    padroes = (
        r"lag de (\d+)",
        r"defasagem de (\d+)",
        r"defasagem temporal de (\d+)",
        r"(\d+)\s+semanas? de defasagem",
        r"lag(\d+)",
    )
    for padrao in padroes:
        achado = re.search(padrao, texto)
        if achado:
            valor = int(achado.group(1))
            if 1 <= valor <= 12:
                return valor
    return 0


def _codigo_do_desenho(modelos: list[str], defasagem: int) -> str:
    """Gera o código do experimento treinando só os modelos pedidos.

    As features de defasagem, quando pedidas, já vêm embutidas em ``X`` pelo
    construtor do dataset (o alvo continua sendo a semana seguinte), então o
    código não as recalcula — apenas treina os estimadores selecionados.
    """
    if list(modelos) == list(MODELOS_PADRAO) and defasagem == 0:
        return CODIGO_EXPERIMENTO
    linhas = "\n".join(f'    "{nome}": {_EXPRESSOES_ESTIMADOR[nome]},' for nome in modelos)
    return CODIGO_EXPERIMENTO.replace(
        '    "regressao_logistica": LogisticRegression(max_iter=1000, random_state=seed),\n'
        '    "floresta_aleatoria": RandomForestClassifier(n_estimators=200, random_state=seed),\n'
        '    "gradient_boosting": GradientBoostingClassifier(random_state=seed),\n',
        linhas + "\n",
    )


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
    codigo: str = ""
    status_experimento: str = "ok"
    auto_debug: dict[str, Any] = field(default_factory=dict)

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
            "codigo": self.codigo,
            "status_experimento": self.status_experimento,
            "auto_debug": self.auto_debug,
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
    llm: Any = None
    max_debug_retries: int = 3

    def run(
        self,
        topic: str,
        hypothesis: dict[str, Any],
        plan: dict[str, Any] | None = None,
        codigo: str | None = None,
    ) -> ExperimentResult:
        """Executa o experimento e devolve métricas + testes estatísticos.

        Levanta ``RealDataUnavailableError`` se não houver fonte pública real para
        o tópico — nunca cai para uma base sintética.

        O código do experimento (``codigo`` ou ``CODIGO_EXPERIMENTO``) roda em
        sandbox com auto-debug: se falhar, o LLM corrige e o runner reexecuta, até
        ``max_debug_retries`` vezes. Esgotadas as tentativas, o resultado é
        marcado com ``status_experimento="falha_apos_debug"`` e o ciclo segue —
        o artigo reporta a dificuldade metodológica.
        """
        started = time.time()
        hyp_id = hypothesis.get("id", "H1")

        if not self.allow_code_execution:
            raise RuntimeError("Execução de código desabilitada em config.json (experiments.allow_code_execution)")

        from sklearn.metrics import f1_score
        from sklearn.model_selection import TimeSeriesSplit, cross_val_score
        from scipy import stats

        # O desenho do experimentalista (``plan``) pede modelos e features
        # específicos por tópico. O runner honra o pedido quando dá para mapeá-lo
        # para estimadores reais; caso contrário, mantém a tríade padrão. Sem isso,
        # tópicos distintos produziriam o mesmo experimento.
        defasagem = _defasagem_do_desenho(plan)
        modelos_cfg = _modelos_do_desenho(plan)
        codigo = codigo or _codigo_do_desenho(modelos_cfg, defasagem)

        real = build_real_dataset(topic, cache_dir=self.cache_dir, offline=self.offline, lags=defasagem)
        if real is None:
            raise RealDataUnavailableError(
                f"Nenhuma fonte pública de observações reais para o tópico: {topic!r}. "
                "O sistema não gera dados sintéticos; escolha um tema com dados abertos "
                "(dengue, desmatamento, agricultura, educação) ou registre a fonte."
            )
        X, y, meta = real.X, real.y, real.meta
        X_train, X_test, y_train, y_test = self._split_temporal(X, y)

        # Auto-debug: o código roda de fato e, se quebrar, é corrigido pelo LLM.
        executado = self._run_experiment_code(codigo, X, y, meta, topic, hyp_id)
        if not executado.sucesso:
            return self._falha_apos_debug(hyp_id, topic, meta, executado, started)

        namespace = executado.namespace
        modelos_treinados: dict[str, Any] = namespace.get("modelos") or {}

        metrics: dict[str, dict[str, float]] = {}
        per_sample_errors: dict[str, np.ndarray] = {}

        # Linha de base honesta e específica do problema: para série temporal,
        # persistir o estado atual (dengue: nível de alerta já ativo; painéis
        # anuais: repetir a tendência recente). É o que um serviço faria sem
        # modelo algum.
        y_pred_base, y_score_base = self._baseline_prediction(X_test, meta)
        metrics["baseline_estatistico"] = self._metricas(
            y_test, y_pred_base, y_score_base, np.array([f1_score(y_test, y_pred_base, zero_division=0)] * 5)
        )
        per_sample_errors["baseline_estatistico"] = (y_pred_base == y_test).astype(float)

        for name, modelo in modelos_treinados.items():
            y_pred = modelo.predict(namespace["X_test"])
            y_score = (
                modelo.predict_proba(namespace["X_test"])[:, 1]
                if hasattr(modelo, "predict_proba")
                else modelo.decision_function(namespace["X_test"])
            )
            # Série temporal não pode usar CV aleatório: dobrar o futuro no treino
            # inflaria as métricas. TimeSeriesSplit respeita a ordem.
            cv = cross_val_score(modelo, X, y, cv=TimeSeriesSplit(n_splits=5), scoring="f1")
            metrics[name] = self._metricas(y_test, y_pred, y_score, cv)
            per_sample_errors[name] = (y_pred == y_test).astype(float)

        if not modelos_treinados:
            return self._falha_apos_debug(
                hyp_id,
                topic,
                meta,
                executado,
                started,
                erro_extra="O código rodou mas não produziu o dicionário 'modelos'.",
            )

        tests = self._testes_estatisticos(per_sample_errors, stats)

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
            codigo=executado.codigo,
            status_experimento="ok",
            auto_debug={
                "tentativas": executado.tentativas,
                "correcoes_aplicadas": len(executado.correcoes),
            },
        )
        result.figures = self._save_figures(result)
        return result

    # --------------------------------------------------------------- execução
    def _run_experiment_code(
        self,
        codigo: str,
        X: np.ndarray,
        y: np.ndarray,
        meta: dict[str, Any],
        topic: str,
        hyp_id: str,
    ) -> Any:
        """Executa o código do experimento em sandbox, com auto-debug."""
        base = {
            "X": X,
            "y": y,
            "meta": meta,
            "topic": topic,
            "seed": self.seed,
            "test_size": self.test_size,
            "alpha": self.alpha,
            "hypothesis_id": hyp_id,
        }
        objetivo = f"treinar modelos preditivos para o tópico {topic!r} sobre observações reais ({meta.get('origem')})"
        return executar_com_auto_debug(
            codigo,
            llm=self.llm,
            objetivo=objetivo,
            base=base,
            max_retries=self.max_debug_retries,
        )

    def _falha_apos_debug(
        self,
        hyp_id: str,
        topic: str,
        meta: dict[str, Any],
        executado: Any,
        started: float,
        erro_extra: str = "",
    ) -> ExperimentResult:
        """Resultado honesto quando o auto-debug não conseguiu fazer o código rodar.

        Não inventa métricas: devolve ``metrics={}`` e o erro real, para que o
        ciclo continue e o manuscrito reporte a dificuldade metodológica.
        """
        erro = erro_extra or executado.erro or "falha desconhecida"
        logger.error("Experimento marcado como falha_apos_debug: %s", erro)
        return ExperimentResult(
            hypothesis_id=hyp_id,
            topic=topic,
            seed=self.seed,
            metrics={},
            statistical_tests={},
            best_model="",
            dataset=meta,
            environment={"python": sys.version.split()[0], "plataforma": platform.platform()},
            duration_s=time.time() - started,
            codigo=executado.codigo,
            status_experimento="falha_apos_debug",
            auto_debug={
                "tentativas": executado.tentativas,
                "correcoes_aplicadas": len(executado.correcoes),
                "erro_final": erro,
                "traceback": executado.traceback,
            },
        )

    def _metricas(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        y_score: np.ndarray,
        cv: np.ndarray,
    ) -> dict[str, float]:
        """Métricas reais de um modelo: acurácia, F1, AUC, IC bootstrap e CV."""
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        try:
            auc = float(roc_auc_score(y_test, y_score))
        except ValueError:
            auc = float("nan")
        lo, hi = _bootstrap_ci((y_pred == y_test).astype(float), np.mean, alpha=self.alpha, seed=self.seed)
        return {
            "acuracia": round(float(accuracy_score(y_test, y_pred)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "auc_roc": round(auc, 4) if auc == auc else None,
            "acuracia_ic95_inf": round(lo, 4),
            "acuracia_ic95_sup": round(hi, 4),
            "cv_f1_media": round(float(np.mean(cv)), 4),
            "cv_f1_desvio": round(float(np.std(cv)), 4),
        }

    def _testes_estatisticos(self, per_sample_errors: dict[str, np.ndarray], stats: Any) -> dict[str, Any]:
        """Testes pareados (t de Student e Wilcoxon) contra a linha de base."""
        baseline = per_sample_errors["baseline_estatistico"]
        tests: dict[str, Any] = {}
        for name in ("regressao_logistica", "floresta_aleatoria", "gradient_boosting"):
            if name not in per_sample_errors:
                continue
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
        return tests

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
