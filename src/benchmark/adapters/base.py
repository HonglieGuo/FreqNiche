from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import anndata as ad
import numpy as np
import torch


@dataclass
class CanonicalAdapterResult:
    method: str
    embedding: np.ndarray
    canonical_labels: np.ndarray | None = None
    refined_labels: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, np.ndarray] = field(default_factory=dict)

    def validate(self, n_spots: int) -> None:
        self.embedding = np.asarray(self.embedding, dtype=np.float32)
        if self.embedding.ndim != 2 or self.embedding.shape[0] != n_spots:
            raise ValueError(
                f"Invalid {self.method} embedding shape: {self.embedding.shape}"
            )
        if self.canonical_labels is not None:
            self.canonical_labels = np.asarray(self.canonical_labels)
            if self.canonical_labels.shape != (n_spots,):
                raise ValueError(
                    f"Invalid {self.method} canonical label shape: "
                    f"{self.canonical_labels.shape}"
                )
        if self.refined_labels is not None:
            self.refined_labels = np.asarray(self.refined_labels)
            if self.refined_labels.shape != (n_spots,):
                raise ValueError(
                    f"Invalid {self.method} refined label shape: "
                    f"{self.refined_labels.shape}"
                )
        if not np.isfinite(self.embedding).all():
            raise ValueError(f"Non-finite {self.method} embedding")
        if "canonical_candidate_labels" in self.extras:
            candidates = np.asarray(self.extras["canonical_candidate_labels"])
            if candidates.ndim != 2 or candidates.shape[1] != n_spots:
                raise ValueError(
                    f"Invalid {self.method} candidate label matrix: {candidates.shape}"
                )
        if "cluster_probabilities" in self.extras:
            probabilities = np.asarray(self.extras["cluster_probabilities"])
            if probabilities.ndim != 2 or probabilities.shape[0] != n_spots:
                raise ValueError(
                    f"Invalid {self.method} probability matrix: {probabilities.shape}"
                )
            if not np.isfinite(probabilities).all():
                raise ValueError(f"Non-finite {self.method} cluster probabilities")


class CanonicalSpatialAdapter(ABC):
    name: str

    @abstractmethod
    def fit_transform(
        self,
        adata: ad.AnnData,
        *,
        n_clusters: int,
        seed: int,
        device: torch.device,
    ) -> CanonicalAdapterResult:
        """Fit the official method on one complete slice and return embeddings."""
