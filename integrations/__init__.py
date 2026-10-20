"""Integrações com fontes de dados abertas."""

from integrations.brazilian_apis import (
    BDTDAPI,
    BancoCentralAPI,
    BrazilianDataAggregator,
    CrossrefAPI,
    DATASUSAPI,
    DataverseAPI,
    IBGEAPI,
    INPEAPI,
    KaggleBRAPI,
    LattesAPI,
    OpenAlexAPI,
    PortalTransparenciaAPI,
    SIDRAAPI,
    SciELOAPI,
)
from integrations.multimodal import (
    ImagemSateliteIndisponivel,
    baixar_imagem_satelite,
    build_multimodal_dataset,
    extract_satellite_features,
    extract_satellite_features_detalhado,
)
from integrations.multimodal_geo import (
    baixar_geotiff_satelite,
    build_multimodal_dataset_geoai,
    geoai_disponivel,
)
from integrations.publishing import (
    IntegrityError,
    SubmissionPackage,
    generate_submission_package,
)

__all__ = [
    "BDTDAPI",
    "BancoCentralAPI",
    "BrazilianDataAggregator",
    "CrossrefAPI",
    "DATASUSAPI",
    "DataverseAPI",
    "IBGEAPI",
    "INPEAPI",
    "ImagemSateliteIndisponivel",
    "IntegrityError",
    "KaggleBRAPI",
    "LattesAPI",
    "OpenAlexAPI",
    "PortalTransparenciaAPI",
    "SIDRAAPI",
    "SciELOAPI",
    "SubmissionPackage",
    "baixar_geotiff_satelite",
    "baixar_imagem_satelite",
    "build_multimodal_dataset",
    "build_multimodal_dataset_geoai",
    "extract_satellite_features",
    "extract_satellite_features_detalhado",
    "generate_submission_package",
    "geoai_disponivel",
]
