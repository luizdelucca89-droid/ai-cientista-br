"""Execução de experimentos e análise estatística."""

from core.experiments.datasets import (
    RealDataset,
    build_dengue_outbreak_dataset,
    build_real_dataset,
)
from core.experiments.runner import (
    ExperimentResult,
    ExperimentRunner,
    generate_synthetic_dataset,
    save_experiment,
)

__all__ = [
    "ExperimentResult",
    "ExperimentRunner",
    "RealDataset",
    "build_dengue_outbreak_dataset",
    "build_real_dataset",
    "generate_synthetic_dataset",
    "save_experiment",
]
