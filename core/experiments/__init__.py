"""Execução de experimentos e análise estatística."""

from core.experiments.runner import (
    ExperimentResult,
    ExperimentRunner,
    generate_synthetic_dataset,
    save_experiment,
)

__all__ = [
    "ExperimentResult",
    "ExperimentRunner",
    "generate_synthetic_dataset",
    "save_experiment",
]
