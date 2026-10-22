"""Canonical spatial-transcriptomics method adapters."""

from .base import CanonicalAdapterResult, CanonicalSpatialAdapter
from .banksy import OfficialBANKSYAdapter
from .cellcharter import (
    CanonicalCellCharterAdapter,
    ControlledCellCharterAdapter,
    OfficialCellCharterAdapter,
    PrecomputedSCVICellCharterAdapter,
)
from .cellniche import OfficialCellNicheAdapter
from .stagate import OfficialSTAGATEAdapter

ADAPTERS = {
    "BANKSY_official": OfficialBANKSYAdapter,
    "STAGATE_official": OfficialSTAGATEAdapter,
    # CellCharter tracks
    "CellCharter_canonical": CanonicalCellCharterAdapter,
    "CellCharter_controlled_pca": ControlledCellCharterAdapter,
    "CellCharter_controlled": ControlledCellCharterAdapter,  # backward-compatible alias
    "CellCharter_official": OfficialCellCharterAdapter,  # DEPRECATED: Points to Controlled PCA wrapper for backward compatibility. Do not use for formal canonical benchmark.
    "CellCharter_scVI_precomputed": PrecomputedSCVICellCharterAdapter,
    "CellNiche_official": OfficialCellNicheAdapter,
}

__all__ = [
    "ADAPTERS",
    "CanonicalAdapterResult",
    "CanonicalSpatialAdapter",
    "OfficialBANKSYAdapter",
    "CanonicalCellCharterAdapter",
    "ControlledCellCharterAdapter",
    "OfficialCellCharterAdapter",
    "PrecomputedSCVICellCharterAdapter",
    "OfficialCellNicheAdapter",
    "OfficialSTAGATEAdapter",
]

