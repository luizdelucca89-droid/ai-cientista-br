"""Integrações com fontes de dados abertas."""

from integrations.brazilian_apis import (
    BancoCentralAPI,
    BrazilianDataAggregator,
    CrossrefAPI,
    IBGEAPI,
    LattesAPI,
    OpenAlexAPI,
    SciELOAPI,
)

__all__ = [
    "BancoCentralAPI",
    "BrazilianDataAggregator",
    "CrossrefAPI",
    "IBGEAPI",
    "LattesAPI",
    "OpenAlexAPI",
    "SciELOAPI",
]
