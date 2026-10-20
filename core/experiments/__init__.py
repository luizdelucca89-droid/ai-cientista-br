"""Execução de experimentos e análise estatística."""

from core.experiments.auto_debugger import (
    ErroDeAutoDebug,
    ResultadoExecucao,
    executar_com_auto_debug,
    sandbox_namespace,
)
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
    CODIGO_EXPERIMENTO,
    ExperimentResult,
    ExperimentRunner,
    save_experiment,
)

__all__ = [
    "CODIGO_EXPERIMENTO",
    "ErroDeAutoDebug",
    "ExperimentResult",
    "ExperimentRunner",
    "RealDataUnavailableError",
    "RealDataset",
    "ResultadoExecucao",
    "build_agriculture_dataset",
    "build_dengue_outbreak_dataset",
    "build_deforestation_dataset",
    "build_education_dataset",
    "build_real_dataset",
    "executar_com_auto_debug",
    "sandbox_namespace",
    "save_experiment",
]
