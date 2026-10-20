"""Execução de experimentos e análise estatística."""

from core.experiments.datasets import (
    RealDataUnavailableError,
    RealDataset,
    build_agriculture_dataset,
    build_dengue_outbreak_dataset,
    build_deforestation_dataset,
    build_education_dataset,
    build_real_dataset,
)
from core.experiments.runner import (
    ExperimentResult,
    ExperimentRunner,
    save_experiment,
)

__all__ = [
    "ExperimentResult",
    "ExperimentRunner",
    "RealDataUnavailableError",
    "RealDataset",
    "build_agriculture_dataset",
    "build_dengue_outbreak_dataset",
    "build_deforestation_dataset",
    "build_education_dataset",
    "build_real_dataset",
    "save_experiment",
]
