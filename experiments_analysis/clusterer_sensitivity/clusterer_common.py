"""Shared contracts for the Human Skin clusterer-sensitivity experiment."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import anndata as ad
import igraph as ig
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "baselines/Baselines_outputs/runs/human_skin/2026-09-28/20260928_human_skin_clusterer_sensitivity"
HUMAN_SKIN_TRAINING_DIR = ROOT / "outputs/runs/freqniche/human_skin/2026-09-08/20260908_human_skin_training_final"
HUMAN_SKIN_BENCHMARK_DIR = ROOT / "baselines/Baselines_outputs/runs/human_skin/2026-09-18/20260918_human_skin_benchmark_final"
EMBEDDING_DIR = HUMAN_SKIN_BENCHMARK_DIR / "embeddings"
VAL_BARCODES = ROOT / "outputs/freqniche_wave_a_diagnostics/original_val_barcodes.csv"
TEST_METADATA = ROOT / "outputs/freqniche_wave_a_diagnostics/wave_a_main/test_centers_mapping.csv"
BOOTSTRAP_MANIFEST = HUMAN_SKIN_BENCHMARK_DIR / "bootstrap_manifest_val8k_10fold.json"
H5AD_PATH = ROOT / "datasets/h5ad/human_skin_merfish/human_skin_merfish.h5ad"
MCLUST_BATCH_SCRIPT = Path(__file__).with_name("mclust_batch.R")
CLUSTERER_ENV_LOCK = ROOT / "requirements.txt"

SEEDS = (42, 123, 2024)
K_CANDIDATES = tuple(range(5, 26))
RESOLUTION_CANDIDATES = tuple(round(x / 10.0, 1) for x in range(1, 21))
ORACLE_RESOLUTIONS = tuple(round(x / 20.0, 2) for x in range(1, 61))
CLUSTERERS = ("kmeans", "mclust", "louvain", "leiden")
N_NEIGHBORS = 15
MCLUST_PCA_DIM = 20
COMPARABLE_ENVIRONMENT_KEYS = (
    "numpy", "scipy", "scikit-learn", "scanpy", "anndata",
    "igraph", "leidenalg", "R", "mclust",
)


@dataclass(frozen=True)
class EmbeddingPair:
    method: str
    seed_label: str
    numeric_seed: int
    validation_path: Path
    test_path: Path


def compute_sha256(path: Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing required asset: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_ordered_barcodes_sha256(path: Path) -> str:
    """Hash the ordered barcode sequence using the frozen W24 contract.

    This is intentionally different from the physical CSV file SHA-256: the
    header and CSV byte serialization are excluded, while row order is bound.
    """
    frame = pd.read_csv(path, usecols=["cell_barcode"])
    if len(frame) != 8000 or frame["cell_barcode"].isna().any():
        raise RuntimeError(f"Invalid Validation barcode table: {path}")
    digest = hashlib.sha256()
    for barcode in frame["cell_barcode"].astype(str):
        digest.update(barcode.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def canonical_sha256(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sign_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "self_signature_sha256" in payload:
        raise ValueError("Payload already contains a signature")
    result = dict(payload)
    result["self_signature_sha256"] = canonical_sha256(payload)
    return result


def verify_signed_payload(payload: dict[str, Any]) -> dict[str, Any]:
    stored = payload.get("self_signature_sha256")
    if not isinstance(stored, str):
        raise RuntimeError("Missing self_signature_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "self_signature_sha256"}
    if canonical_sha256(unsigned) != stored:
        raise RuntimeError("Self-signature mismatch")
    return unsigned


def atomic_write_json(path: Path, payload: dict[str, Any], signed: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    body = sign_payload(payload) if signed else payload
    try:
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(body, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def relative_path(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def environment_versions() -> dict[str, str]:
    packages = ("numpy", "scipy", "scikit-learn", "scanpy", "anndata", "igraph", "leidenalg")
    versions = {name: importlib.metadata.version(name) for name in packages}
    r_query = subprocess.run(
        ["Rscript", "-e", "cat(as.character(getRversion()), '|', as.character(packageVersion('mclust')))"],
        capture_output=True, text=True, check=False,
    )
    if r_query.returncode != 0 or "|" not in r_query.stdout:
        raise RuntimeError(f"Unable to query R/mclust versions: {r_query.stdout} {r_query.stderr}")
    r_version, mclust_version = [item.strip() for item in r_query.stdout.split("|", 1)]
    return {
        "python": platform.python_version(),
        **versions,
        "R": r_version,
        "mclust": mclust_version,
    }


def assert_python_environment_matches_lock() -> dict[str, str]:
    if not CLUSTERER_ENV_LOCK.is_file():
        raise FileNotFoundError(CLUSTERER_ENV_LOCK)
    locked = {}
    for line in CLUSTERER_ENV_LOCK.read_text(encoding="utf-8").splitlines():
        if "==" in line and not line.lstrip().startswith("#"):
            name, version = line.strip().split("==", 1)
            locked[name.lower()] = version
    actual = environment_versions()
    for package in ("numpy", "scipy", "scikit-learn", "scanpy", "anndata", "igraph", "leidenalg"):
        expected = locked.get(package)
        if expected is None:
            raise RuntimeError(f"Package missing from clusterer environment lock: {package}")
        if actual[package] != expected:
            raise RuntimeError(
                f"Environment-lock mismatch for {package}: actual={actual[package]}, expected={expected}"
            )
    return actual


def load_signed_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return verify_signed_payload(json.load(handle))


def selection_dir(suite: str) -> Path:
    return OUTPUT_ROOT / ("local_freqniche_selection_v1" if suite == "freqniche" else "remote_baselines_selection_v1")


def evaluation_dir(suite: str) -> Path:
    return OUTPUT_ROOT / ("local_freqniche_evaluation_v1" if suite == "freqniche" else "remote_baselines_evaluation_v1")


def work_dir(suite: str) -> Path:
    return OUTPUT_ROOT / f".{suite}_selection_work_v1"


def freqniche_registry(include_test_verification: bool = True) -> list[EmbeddingPair]:
    """Only the frozen inductive model reported in the manuscript supplement."""
    pairs = [EmbeddingPair(
        method="FreqNiche-Inductive", seed_label=str(seed), numeric_seed=seed,
        validation_path=HUMAN_SKIN_TRAINING_DIR / f"trial_24_seed_{seed}/val_embeddings.npy",
        test_path=EMBEDDING_DIR / f"freqniche_st_v1_trial24_seed_{seed}_test_z.npy",
    ) for seed in SEEDS]
    for pair in pairs:
        if not pair.validation_path.is_file():
            raise FileNotFoundError(pair.validation_path)
        if include_test_verification and not pair.test_path.is_file():
            raise FileNotFoundError(pair.test_path)
    return pairs


def baseline_registry() -> list[EmbeddingPair]:
    pairs: list[EmbeddingPair] = []
    for seed in SEEDS:
        pairs.extend([
            EmbeddingPair(
                "CellCharter-Canonical", str(seed), seed,
                EMBEDDING_DIR / f"cellcharter_canonical_seed_{seed}_val_z.npy",
                EMBEDDING_DIR / f"cellcharter_canonical_seed_{seed}_test_z.npy",
            ),
            EmbeddingPair(
                "STAGATE-Controlled", str(seed), seed,
                EMBEDDING_DIR / f"stagate_controlled_seed_{seed}_val_z.npy",
                EMBEDDING_DIR / f"stagate_controlled_seed_{seed}_test_z.npy",
            ),
            EmbeddingPair(
                "CellNiche-Controlled", str(seed), seed,
                EMBEDDING_DIR / f"cellniche_controlled_seed_{seed}_val_z.npy",
                EMBEDDING_DIR / f"cellniche_controlled_seed_{seed}_test_z.npy",
            ),
        ])
    pairs.append(EmbeddingPair(
        "BANKSY-Controlled", "deterministic", 42,
        EMBEDDING_DIR / "banksy_controlled_val_z.npy",
        EMBEDDING_DIR / "banksy_controlled_test_z.npy",
    ))
    return pairs


def registry(suite: str, include_test_verification: bool = True) -> list[EmbeddingPair]:
    if suite == "freqniche":
        return freqniche_registry(include_test_verification=include_test_verification)
    if suite == "baselines":
        return baseline_registry()
    raise ValueError(f"Unknown suite: {suite}")


def validate_embedding(path: Path) -> np.ndarray:
    values = np.load(path, allow_pickle=False)
    if values.ndim != 2 or values.shape[0] != 8000 or values.shape[1] < 2:
        raise RuntimeError(f"Invalid embedding shape at {path}: {values.shape}")
    if not np.isfinite(values).all():
        raise RuntimeError(f"Non-finite embedding: {path}")
    return np.asarray(values, dtype=np.float32)


def fit_validation_pca(
    embedding: np.ndarray,
    source_embedding_sha256: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fit the frozen mclust PCA on the complete Validation embedding only."""
    values = np.asarray(embedding, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 8000 or values.shape[1] < MCLUST_PCA_DIM:
        raise RuntimeError(f"Embedding cannot support PCA{MCLUST_PCA_DIM}: {values.shape}")
    pca = PCA(n_components=MCLUST_PCA_DIM, svd_solver="full")
    projected = pca.fit_transform(values)
    if projected.shape != (8000, MCLUST_PCA_DIM) or not np.isfinite(projected).all():
        raise RuntimeError("Invalid Validation PCA projection")
    state = {
        "mean": np.asarray(pca.mean_, dtype=np.float64),
        "components": np.asarray(pca.components_, dtype=np.float64),
        "explained_variance": np.asarray(pca.explained_variance_, dtype=np.float64),
        "explained_variance_ratio": np.asarray(pca.explained_variance_ratio_, dtype=np.float64),
        "singular_values": np.asarray(pca.singular_values_, dtype=np.float64),
        "n_features_in": np.asarray([values.shape[1]], dtype=np.int64),
        "n_components": np.asarray([MCLUST_PCA_DIM], dtype=np.int64),
        "source_validation_embedding_sha256": np.asarray([source_embedding_sha256]),
        "fit_split": np.asarray(["validation"]),
        "protocol": np.asarray(["mclust_EEE_validation_fitted_PCA20_v1"]),
    }
    return np.asarray(projected, dtype=np.float64), state


def atomic_write_pca_state(path: Path, state: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("wb") as handle:
            np.savez_compressed(handle, **state)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def project_with_frozen_pca(
    embedding: np.ndarray,
    state_path: Path,
    expected_validation_embedding_sha256: str,
) -> np.ndarray:
    with np.load(state_path, allow_pickle=False) as payload:
        required = {
            "mean", "components", "explained_variance", "explained_variance_ratio",
            "singular_values", "n_features_in", "n_components",
            "source_validation_embedding_sha256", "fit_split", "protocol",
        }
        if set(payload.files) != required:
            raise RuntimeError(f"Unexpected PCA state keys: {payload.files}")
        source_sha = str(payload["source_validation_embedding_sha256"][0])
        fit_split = str(payload["fit_split"][0])
        protocol = str(payload["protocol"][0])
        mean = np.asarray(payload["mean"], dtype=np.float64)
        components = np.asarray(payload["components"], dtype=np.float64)
        explained_variance = np.asarray(payload["explained_variance"], dtype=np.float64)
        explained_variance_ratio = np.asarray(payload["explained_variance_ratio"], dtype=np.float64)
        singular_values = np.asarray(payload["singular_values"], dtype=np.float64)
        n_features = int(payload["n_features_in"][0])
        n_components = int(payload["n_components"][0])
    if source_sha != expected_validation_embedding_sha256:
        raise RuntimeError("Frozen PCA is bound to a different Validation embedding")
    if fit_split != "validation" or protocol != "mclust_EEE_validation_fitted_PCA20_v1":
        raise RuntimeError("Unexpected frozen PCA fitting protocol")
    if n_components != MCLUST_PCA_DIM or components.shape != (MCLUST_PCA_DIM, n_features):
        raise RuntimeError("Frozen PCA dimension mismatch")
    if mean.shape != (n_features,):
        raise RuntimeError("Frozen PCA mean dimension mismatch")
    if explained_variance.shape != (MCLUST_PCA_DIM,) or explained_variance_ratio.shape != (MCLUST_PCA_DIM,) or singular_values.shape != (MCLUST_PCA_DIM,):
        raise RuntimeError("Frozen PCA diagnostic dimension mismatch")
    if not all(np.isfinite(item).all() for item in (mean, components, explained_variance, explained_variance_ratio, singular_values)):
        raise RuntimeError("Frozen PCA state contains non-finite values")
    if not np.allclose(components @ components.T, np.eye(MCLUST_PCA_DIM), atol=1e-6, rtol=1e-6):
        raise RuntimeError("Frozen PCA components are not orthonormal")
    values = np.asarray(embedding, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != n_features:
        raise RuntimeError(f"Embedding/PCA feature mismatch: {values.shape}, expected {n_features}")
    projected = (values - mean) @ components.T
    if not np.isfinite(projected).all():
        raise RuntimeError("Frozen PCA projection contains non-finite values")
    return np.asarray(projected, dtype=np.float64)


def load_folds() -> tuple[list[np.ndarray], dict[str, Any]]:
    with BOOTSTRAP_MANIFEST.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("n_cells") != 8000 or manifest.get("n_bootstraps") != 10:
        raise RuntimeError("Unexpected Validation bootstrap manifest")
    expected_barcode_sha = manifest.get("validation_barcodes_sha256")
    actual_barcode_sha = compute_ordered_barcodes_sha256(VAL_BARCODES)
    if actual_barcode_sha != expected_barcode_sha:
        raise RuntimeError(
            "Validation barcode-order SHA mismatch: "
            f"actual={actual_barcode_sha}, expected={expected_barcode_sha}"
        )
    folds = [np.asarray(fold, dtype=np.int64) for fold in manifest["bootstrap_indices"]]
    if len(folds) != 10 or any(len(fold) != 6400 for fold in folds):
        raise RuntimeError("Unexpected Validation bootstrap fold dimensions")
    if any(fold.min() < 0 or fold.max() >= 8000 or len(np.unique(fold)) != len(fold) for fold in folds):
        raise RuntimeError("Invalid Validation bootstrap indices")
    return folds, manifest


def mean_pairwise_overlap_ari(labels: list[np.ndarray], folds: list[np.ndarray]) -> float:
    scores: list[float] = []
    for i in range(len(folds)):
        for j in range(i + 1, len(folds)):
            _, pos_i, pos_j = np.intersect1d(folds[i], folds[j], return_indices=True)
            if len(pos_i) < 2:
                raise RuntimeError("Bootstrap folds have insufficient overlap")
            scores.append(float(adjusted_rand_score(labels[i][pos_i], labels[j][pos_j])))
    return float(np.mean(scores))


def run_kmeans(values: np.ndarray, k: int, seed: int) -> np.ndarray:
    return KMeans(
        n_clusters=int(k), init="k-means++", n_init=20,
        max_iter=500, random_state=int(seed),
    ).fit_predict(np.asarray(values, dtype=np.float64)).astype(np.int64)


def run_graph_clusterer(
    values: np.ndarray,
    clusterer: str,
    resolution: float,
    seed: int,
) -> np.ndarray:
    return run_graph_resolution_sweep(values, clusterer, [resolution], seed)[float(resolution)]


def run_graph_resolution_sweep(
    values: np.ndarray,
    clusterer: str,
    resolutions: Iterable[float],
    seed: int,
) -> dict[float, np.ndarray]:
    """Build one kNN graph and evaluate every resolution on that same graph."""
    if clusterer not in {"louvain", "leiden"}:
        raise ValueError(clusterer)
    data = ad.AnnData(X=np.asarray(values, dtype=np.float32))
    sc.pp.neighbors(data, n_neighbors=N_NEIGHBORS, use_rep="X", random_state=int(seed))
    graph = None
    weights = None
    if clusterer == "louvain":
        upper = sp.triu(data.obsp["connectivities"], k=1).tocoo()
        graph = ig.Graph(
            n=data.n_obs,
            edges=list(zip(upper.row.tolist(), upper.col.tolist())),
            directed=False,
        )
        weights = upper.data.astype(float).tolist()
    outputs: dict[float, np.ndarray] = {}
    for resolution in resolutions:
        key = f"cluster_{str(float(resolution)).replace('.', '_')}"
        if clusterer == "leiden":
            sc.tl.leiden(
                data, resolution=float(resolution), key_added=key,
                random_state=int(seed), directed=False,
            )
        else:
            assert graph is not None and weights is not None
            ig.set_random_number_generator(random.Random(int(seed)))
            partition = graph.community_multilevel(
                weights=weights,
                resolution=float(resolution),
                return_levels=False,
            )
            outputs[float(resolution)] = np.asarray(partition.membership, dtype=np.int64)
            continue
        outputs[float(resolution)] = data.obs[key].astype(str).to_numpy().copy()
    return outputs


def run_mclust_batch(
    values: np.ndarray,
    tasks: list[tuple[str, int, int, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Run true R mclust EEE tasks in one isolated R process.

    Each task is (task_id, K, seed, row_indices). No sklearn fallback exists.
    """
    if not MCLUST_BATCH_SCRIPT.is_file():
        raise FileNotFoundError(MCLUST_BATCH_SCRIPT)
    with tempfile.TemporaryDirectory(prefix="freqniche_mclust_") as temp_text:
        temp = Path(temp_text)
        matrix_path = temp / "embedding.csv"
        task_path = temp / "tasks.csv"
        index_dir = temp / "indices"
        output_dir = temp / "labels"
        index_dir.mkdir()
        output_dir.mkdir()
        pd.DataFrame(np.asarray(values, dtype=np.float64)).to_csv(
            matrix_path, index=False, header=False
        )
        records = []
        expected_lengths: dict[str, int] = {}
        for task_id, k, seed, indices in tasks:
            safe_id = str(task_id)
            index_path = index_dir / f"{safe_id}.txt"
            np.savetxt(index_path, np.asarray(indices, dtype=np.int64), fmt="%d")
            records.append({"task_id": safe_id, "k": int(k), "seed": int(seed)})
            expected_lengths[safe_id] = int(len(indices))
        pd.DataFrame(records).to_csv(task_path, index=False)
        process = subprocess.Popen(
            ["Rscript", str(MCLUST_BATCH_SCRIPT), str(matrix_path), str(task_path), str(index_dir), str(output_dir)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        tail: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            print(f"    [R] {line.rstrip()}", flush=True)
            tail.append(line.rstrip())
            tail = tail[-50:]
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError("R mclust batch failed:\n" + "\n".join(tail))
        outputs: dict[str, np.ndarray] = {}
        for record in records:
            label_path = output_dir / f"{record['task_id']}.csv"
            if not label_path.is_file():
                raise RuntimeError(f"Missing mclust task output: {record['task_id']}")
            raw_labels = pd.read_csv(label_path, header=None).iloc[:, 0].to_numpy()
            labels = validate_mclust_classification(
                raw_labels,
                expected_length=expected_lengths[record["task_id"]],
                requested_g=int(record["k"]),
                task_id=record["task_id"],
            )
            outputs[record["task_id"]] = labels
        return outputs


def validate_mclust_classification(
    raw_labels: np.ndarray,
    expected_length: int,
    requested_g: int,
    task_id: str = "mclust",
) -> np.ndarray:
    """Validate mclust labels without requiring every requested component to be occupied."""
    values = np.asarray(raw_labels)
    if values.ndim != 1 or len(values) != int(expected_length):
        raise RuntimeError(
            f"Invalid mclust output length for {task_id}: "
            f"shape={values.shape}, expected={expected_length}"
        )
    try:
        numeric = values.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"Non-numeric mclust classification for {task_id}") from error
    if not np.isfinite(numeric).all() or not np.allclose(numeric, np.round(numeric)):
        raise RuntimeError(f"Non-finite or non-integer mclust classification for {task_id}")
    one_based = np.round(numeric).astype(np.int64)
    if np.any(one_based < 1) or np.any(one_based > int(requested_g)):
        raise RuntimeError(
            f"mclust labels outside requested G for {task_id}: "
            f"min={one_based.min()}, max={one_based.max()}, G={requested_g}"
        )
    labels = one_based - 1
    achieved_k = int(len(np.unique(labels)))
    if achieved_k < 1 or achieved_k > int(requested_g):
        raise RuntimeError(
            f"Invalid achieved mclust K for {task_id}: K={achieved_k}, G={requested_g}"
        )
    return labels


def selection_stability(
    embedding: np.ndarray,
    folds: list[np.ndarray],
    clusterer: str,
    numeric_seed: int,
) -> dict[str, Any]:
    if clusterer in {"kmeans", "mclust"}:
        candidates: Iterable[float | int] = K_CANDIDATES
    elif clusterer in {"louvain", "leiden"}:
        candidates = RESOLUTION_CANDIDATES
    else:
        raise ValueError(clusterer)

    labels_by_candidate: dict[str, list[np.ndarray]] = {}
    if clusterer == "mclust":
        tasks = []
        for candidate in candidates:
            for fold_id, indices in enumerate(folds):
                tasks.append((f"k{int(candidate)}_fold{fold_id}", int(candidate), numeric_seed + fold_id, indices))
        outputs = run_mclust_batch(embedding, tasks)
        for candidate in candidates:
            labels_by_candidate[str(candidate)] = [
                outputs[f"k{int(candidate)}_fold{fold_id}"] for fold_id in range(len(folds))
            ]
    elif clusterer == "kmeans":
        for candidate in candidates:
            candidate_labels = []
            for fold_id, indices in enumerate(folds):
                subset = embedding[indices]
                run_seed = numeric_seed + fold_id
                labels = run_kmeans(subset, int(candidate), run_seed)
                candidate_labels.append(labels)
            labels_by_candidate[str(candidate)] = candidate_labels
    else:
        labels_by_candidate = {str(candidate): [] for candidate in candidates}
        for fold_id, indices in enumerate(folds):
            sweep = run_graph_resolution_sweep(
                embedding[indices], clusterer, [float(c) for c in candidates], numeric_seed + fold_id
            )
            for candidate in candidates:
                labels_by_candidate[str(candidate)].append(sweep[float(candidate)])

    reports = {}
    for candidate in candidates:
        labels = labels_by_candidate[str(candidate)]
        achieved = [int(len(np.unique(item))) for item in labels]
        reports[str(candidate)] = {
            "parameter": float(candidate) if clusterer in {"louvain", "leiden"} else int(candidate),
            "parameter_semantics": (
                "requested_components_G" if clusterer == "mclust"
                else "requested_clusters_K" if clusterer == "kmeans"
                else "community_resolution"
            ),
            "mean_pairwise_overlap_ari": mean_pairwise_overlap_ari(labels, folds),
            "mean_clusters": float(np.mean(achieved)),
            "fold_cluster_counts": achieved,
            "folds_with_fewer_clusters_than_requested_G": (
                int(sum(value < int(candidate) for value in achieved))
                if clusterer == "mclust" else 0
            ),
        }
    valid = [
        candidate for candidate in candidates
        if 5 <= reports[str(candidate)]["mean_clusters"] <= 25
    ]
    if not valid:
        raise RuntimeError(f"No valid {clusterer} Validation candidate in cluster-count bounds")
    selected = max(
        valid,
        key=lambda candidate: (
            reports[str(candidate)]["mean_pairwise_overlap_ari"],
            -float(candidate),
        ),
    )
    return {
        "clusterer": clusterer,
        "selection_rule": "maximize mean pairwise ARI across shared Validation 80% folds; ties choose smaller parameter",
        "selected_parameter": float(selected) if clusterer in {"louvain", "leiden"} else int(selected),
        "selected_parameter_semantics": reports[str(selected)]["parameter_semantics"],
        "selected_stability": reports[str(selected)]["mean_pairwise_overlap_ari"],
        "candidate_reports": reports,
    }


def cluster_full(
    embedding: np.ndarray,
    clusterer: str,
    parameter: float | int,
    seed: int,
) -> np.ndarray:
    indices = np.arange(len(embedding), dtype=np.int64)
    if clusterer == "kmeans":
        return run_kmeans(embedding, int(parameter), seed)
    if clusterer == "mclust":
        return run_mclust_batch(embedding, [("full", int(parameter), seed, indices)])["full"]
    return run_graph_clusterer(embedding, clusterer, float(parameter), seed)


def select_graph_parameter_by_target_k(
    embedding: np.ndarray,
    clusterer: str,
    target_k: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    records = []
    labels_by_resolution = run_graph_resolution_sweep(
        embedding, clusterer, ORACLE_RESOLUTIONS, seed
    )
    for resolution in ORACLE_RESOLUTIONS:
        labels = labels_by_resolution[resolution]
        achieved = int(len(np.unique(labels)))
        records.append({
            "resolution": resolution,
            "achieved_k": achieved,
            "absolute_k_error": abs(achieved - int(target_k)),
        })
    selected = min(records, key=lambda row: (row["absolute_k_error"], row["resolution"]))
    return labels_by_resolution[selected["resolution"]], {
        "selection_rule": "minimize |K_pred-K_true|; ties choose smaller resolution; ARI/NMI excluded",
        "target_k": int(target_k),
        "selected_resolution": selected["resolution"],
        "selected_k": selected["achieved_k"],
        "absolute_k_error": selected["absolute_k_error"],
        "exact_k_match": selected["absolute_k_error"] == 0,
        "candidate_cluster_counts": records,
    }


def load_test_metadata() -> pd.DataFrame:
    metadata = pd.read_csv(TEST_METADATA)
    required = {"cell_barcode", "donor_id", "niche"}
    if len(metadata) != 8000 or not required.issubset(metadata.columns):
        raise RuntimeError("Invalid Test metadata contract")
    return metadata


def target_k_from_metadata(metadata: pd.DataFrame) -> int:
    value = int(metadata["niche"].dropna().astype(str).nunique())
    if value < 2:
        raise RuntimeError(f"Invalid Test target K: {value}")
    return value


def evaluate_labels(labels: np.ndarray, metadata: pd.DataFrame) -> dict[str, Any]:
    labels = np.asarray(labels)
    if len(labels) != len(metadata):
        raise RuntimeError("Label/metadata length mismatch")
    per_donor = {}
    aris, nmis = [], []
    for donor in sorted(metadata["donor_id"].astype(str).unique()):
        mask = metadata["donor_id"].astype(str).eq(donor).to_numpy()
        truth = metadata.loc[mask, "niche"].to_numpy()
        valid = ~pd.isna(truth)
        ari = float(adjusted_rand_score(truth[valid], labels[mask][valid]))
        nmi = float(normalized_mutual_info_score(truth[valid], labels[mask][valid]))
        aris.append(ari)
        nmis.append(nmi)
        per_donor[donor] = {"n_cells": int(valid.sum()), "ari": ari, "nmi": nmi}
    return {
        "donor_macro_ari": float(np.mean(aris)),
        "donor_macro_nmi": float(np.mean(nmis)),
        "worst_donor_ari": float(np.min(aris)),
        "donor_ari_sample_sd": float(np.std(aris, ddof=1)),
        "n_clusters": int(len(np.unique(labels))),
        "per_donor": per_donor,
    }


def method_seed_key(pair: EmbeddingPair) -> str:
    safe_method = pair.method.lower().replace("-", "_").replace(" ", "_")
    return f"{safe_method}__seed_{pair.seed_label}"


def input_record(path: Path) -> dict[str, str]:
    return {"path": relative_path(path), "sha256": compute_sha256(path)}
