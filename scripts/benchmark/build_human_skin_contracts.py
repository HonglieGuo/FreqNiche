#!/usr/bin/env python3
"""Build the frozen Human Skin Validation and Test center contracts.

The source data, split order, sample-isolated frozen graph and RandomState(42)
sampling contract uniquely determine this list. Both the ordered-barcode hash
and the original CSV file hash are checked against independent frozen records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "datasets/h5ad/human_skin_merfish/human_skin_merfish.h5ad"
SPLITS = ROOT / "datasets/h5ad/human_skin_merfish/donor_splits.csv"
GRAPH = ROOT / "outputs/cache/graphs/graph_7a7ba4e1e22eb69d49a210e2.npz"
TRAIN_OUTPUT = ROOT / "outputs/contracts/human_skin/train_centers_68940.json"
OUTPUT = ROOT / "outputs/contracts/human_skin/validation_barcodes_8k.csv"
LEGACY_OUTPUT = ROOT / "outputs/freqniche_wave_a_diagnostics/original_val_barcodes.csv"
TEST_OUTPUT = ROOT / "outputs/contracts/human_skin/test_centers_8k.csv"
LEGACY_TEST_OUTPUT = ROOT / "outputs/freqniche_wave_a_diagnostics/wave_a_main/test_centers_mapping.csv"

EXPECTED_ORDERED_HASH = "c19d1a9d0ff2694e7c902f6cf95eee6285256f16e36a5dc5445e3c1ec3aada56"
EXPECTED_TRAIN_ORDERED_HASH = "a117f218539d88e84e5e8e64ba1b916a105c1fd99aa237d3c8e6bd6fd23cf31a"
EXPECTED_FILE_HASH = "b37d6c6a4e2240eb5082be739ca2347f2d55fa6afbe7d36284b03d73ace051e9"
EXPECTED_TEST_FILE_HASH = "a32158b85155c6e820e59b953c50da57a336fd18427f2ee6b9f9d602f56583d0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_hash(barcodes: list[str]) -> str:
    digest = hashlib.sha256()
    for barcode in barcodes:
        digest.update(barcode.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def ensure_train_contract(barcodes: list[str], donor_counts: dict[str, int]) -> None:
    """Recreate the original 68,940 ordered centers without historical outputs."""
    if len(barcodes) != 68940 or len(set(barcodes)) != 68940:
        raise RuntimeError("Expected 68,940 unique Train centers")
    barcode_sha = ordered_hash(barcodes)
    if barcode_sha != EXPECTED_TRAIN_ORDERED_HASH:
        raise RuntimeError(f"Train barcode order differs from frozen contract: {barcode_sha}")
    if sum(donor_counts.values()) != len(barcodes):
        raise RuntimeError("Train donor counts do not sum to 68,940")

    if TRAIN_OUTPUT.is_file():
        existing = json.loads(TRAIN_OUTPUT.read_text(encoding="utf-8"))
        signature = existing.pop("sha256", None)
        computed = hashlib.sha256(json.dumps(existing, indent=2).encode("utf-8")).hexdigest()
        if signature != computed or existing.get("barcodes") != barcodes:
            raise RuntimeError(f"Existing Train center contract is invalid: {TRAIN_OUTPUT}")
        print(f"VERIFIED {TRAIN_OUTPUT.relative_to(ROOT) if TRAIN_OUTPUT.is_relative_to(ROOT) else TRAIN_OUTPUT}")
        return

    payload = {
        "n_centers": len(barcodes),
        "source_h5ad": str(DATASET.relative_to(ROOT)),
        "graph_fingerprint": GRAPH.stem.removeprefix("graph_"),
        "donor_counts": donor_counts,
        "barcodes": barcodes,
        "ordered_barcodes_sha256": barcode_sha,
    }
    payload["sha256"] = hashlib.sha256(json.dumps(payload, indent=2).encode("utf-8")).hexdigest()
    TRAIN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = TRAIN_OUTPUT.with_name(f".{TRAIN_OUTPUT.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, TRAIN_OUTPUT)
    print(f"CREATED {TRAIN_OUTPUT.relative_to(ROOT) if TRAIN_OUTPUT.is_relative_to(ROOT) else TRAIN_OUTPUT} | ORDERED_SHA256 {barcode_sha}")


def load_or_build_train_val_graph(dataset: ad.AnnData, mask: np.ndarray, obs: pd.DataFrame) -> sp.csr_matrix:
    """Use the frozen graph cache, or rebuild it from Train+Validation H5AD metadata.

    The expected fingerprint includes the graph implementation, ordered coordinates,
    sample IDs and gene names. A changed source cannot silently populate the frozen
    filename with a graph from a different contract.
    """
    from src.freqniche_st.human_skin_backend import get_graph_fingerprint, get_or_build_graph

    coords = np.asarray(dataset.obsm["spatial"])[mask]
    sample_ids = obs["sample_id"].astype(str).to_numpy()
    gene_names = list(dataset.var_names)
    fingerprint = get_graph_fingerprint(coords, sample_ids, gene_names, 50.0, 50.0 / 3.0)
    expected = GRAPH.stem.removeprefix("graph_")
    if fingerprint != expected:
        raise RuntimeError(
            f"Processed H5AD / graph-code fingerprint {fingerprint} does not match "
            f"frozen graph contract {expected}; refusing to create a misleading cache"
        )
    if GRAPH.is_file():
        with np.load(GRAPH) as cache:
            adjacency = sp.csr_matrix(
                (cache["A_data"], cache["A_indices"], cache["A_indptr"]),
                shape=tuple(cache["A_shape"]),
            )
        return adjacency
    _, adjacency, actual = get_or_build_graph(
        coords=coords,
        sample_ids=sample_ids,
        gene_names=gene_names,
        radius=50.0,
        cache_dir=str(GRAPH.parent),
    )
    if actual != expected or not GRAPH.is_file():
        raise RuntimeError("Graph cache rebuild did not publish the expected frozen asset")
    return adjacency


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train_validation", "test"), default="train_validation")
    parser.add_argument("--authorized", action="store_true", help="Required before creating Test centers")
    args = parser.parse_args()
    if args.phase == "test" and not args.authorized:
        parser.error("Test-center creation requires --authorized")

    splits = pd.read_csv(SPLITS)
    train_donors = splits.loc[splits["split"] == "train", "donor_id"].astype(str).to_numpy()
    val_donors = splits.loc[splits["split"].isin(["val", "validation"]), "donor_id"].astype(str).to_numpy()
    train_val_donors = set(train_donors) | set(val_donors)
    test_donors = splits.loc[splits["split"] == "test", "donor_id"].astype(str).to_numpy()

    dataset = ad.read_h5ad(DATASET, backed="r")
    mask = dataset.obs["donor_id"].astype(str).isin(train_val_donors).to_numpy()
    obs = dataset.obs.loc[mask]

    adjacency = load_or_build_train_val_graph(dataset, mask, obs)
    if adjacency.shape[0] != len(obs):
        raise RuntimeError(f"Frozen graph/Train+Validation size mismatch: {adjacency.shape[0]} vs {len(obs)}")
    non_isolated = np.diff(adjacency.indptr) > 1

    rng = np.random.RandomState(42)
    donor_values = obs["donor_id"].astype(str).to_numpy()
    train_selected: list[np.ndarray] = []
    train_counts: dict[str, int] = {}
    for donor in train_donors:
        candidates = np.flatnonzero((donor_values == donor) & non_isolated)
        if len(candidates) > 5000:
            candidates = rng.choice(candidates, size=5000, replace=False)
        train_selected.append(candidates)
        train_counts[str(donor)] = len(candidates)

    selected: list[np.ndarray] = []
    for donor in val_donors:
        candidates = np.flatnonzero((donor_values == donor) & non_isolated)
        if len(candidates) > 2000:
            candidates = rng.choice(candidates, size=2000, replace=False)
        if len(candidates) != 2000:
            raise RuntimeError(f"Validation donor {donor} produced {len(candidates)} centers")
        selected.append(candidates)

    indices = np.concatenate(selected)
    barcodes = obs.index.to_numpy()[indices].astype(str).tolist()
    if len(barcodes) != 8000 or len(set(barcodes)) != 8000:
        raise RuntimeError("Expected 8,000 unique Validation barcodes")
    actual_ordered_hash = ordered_hash(barcodes)
    if actual_ordered_hash != EXPECTED_ORDERED_HASH:
        raise RuntimeError(
            f"Ordered barcode hash mismatch: {actual_ordered_hash} != {EXPECTED_ORDERED_HASH}"
        )

    val_csv = pd.DataFrame({"cell_barcode": barcodes})
    if args.phase == "train_validation":
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        temporary = OUTPUT.with_suffix(f".tmp.{OUTPUT.suffix.lstrip('.')}")
        val_csv.to_csv(temporary, index=False)
        actual_file_hash = sha256_file(temporary)
    else:
        if not OUTPUT.is_file():
            raise FileNotFoundError(f"Frozen Validation centers missing: {OUTPUT}")
        actual_file_hash = sha256_file(OUTPUT)
    if actual_file_hash != EXPECTED_FILE_HASH:
        raise RuntimeError(f"CSV file hash mismatch: {actual_file_hash} != {EXPECTED_FILE_HASH}")
    if args.phase == "train_validation":
        train_barcodes = obs.index.to_numpy()[np.concatenate(train_selected)].astype(str).tolist()
        ensure_train_contract(train_barcodes, train_counts)
        temporary.replace(OUTPUT)

        if not os.environ.get("FREQNICHE_RUN_DIR"):
            LEGACY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            if LEGACY_OUTPUT.exists():
                if sha256_file(LEGACY_OUTPUT) != EXPECTED_FILE_HASH:
                    raise RuntimeError(f"Existing legacy barcode file is not the frozen asset: {LEGACY_OUTPUT}")
            else:
                shutil.copy2(OUTPUT, LEGACY_OUTPUT)
        print(f"REBUILT {OUTPUT.relative_to(ROOT) if OUTPUT.is_relative_to(ROOT) else OUTPUT}")
        print(f"ORDERED_SHA256 {actual_ordered_hash}")
        print(f"FILE_SHA256 {actual_file_hash}")
        print("Test centers not read or generated during Train/Validation preparation.")
        return

    # Continue the exact same random stream to reconstruct the frozen Test centers.
    full_obs = dataset.obs
    test_mask = full_obs["donor_id"].astype(str).isin(set(test_donors)).to_numpy()
    test_obs = full_obs.loc[test_mask]
    from src.freqniche_st.graph import build_global_spatial_graph

    _, test_adjacency = build_global_spatial_graph(
        coords=np.asarray(dataset.obsm["spatial"])[test_mask],
        sample_ids=test_obs["sample_id"].astype(str).to_numpy(),
        radius=50.0,
        sigma=50.0 / 3.0,
    )
    test_non_isolated = np.diff(test_adjacency.indptr) > 1
    test_donor_values = test_obs["donor_id"].astype(str).to_numpy()
    selected_test: list[np.ndarray] = []
    for donor in test_donors:
        candidates = np.flatnonzero((test_donor_values == donor) & test_non_isolated)
        if len(candidates) > 2000:
            candidates = rng.choice(candidates, size=2000, replace=False)
        if len(candidates) != 2000:
            raise RuntimeError(f"Test donor {donor} produced {len(candidates)} centers")
        selected_test.append(candidates)
    test_indices_local = np.concatenate(selected_test)
    test_barcodes = test_obs.index.to_numpy()[test_indices_local].astype(str)
    full_index = pd.Series(np.arange(dataset.n_obs, dtype=np.int64), index=dataset.obs_names.astype(str))
    global_indices = full_index.loc[test_barcodes].to_numpy(dtype=np.int64)
    niche_column = "niche" if "niche" in full_obs else "cellcharter_cluster.k_10"
    cell_type_column = "cell_type" if "cell_type" in full_obs else "cell_type_level_1"
    selected_obs = full_obs.loc[test_barcodes]
    mapping = pd.DataFrame(
        {
            "embedding_row": np.arange(len(test_barcodes), dtype=np.int64),
            "cell_barcode": test_barcodes,
            "context_row_index": global_indices,
            "global_row_index": global_indices,
            "donor_id": selected_obs["donor_id"].to_numpy(),
            "sample_id": selected_obs["sample_id"].to_numpy(),
            "cell_type": selected_obs[cell_type_column].to_numpy()
            if cell_type_column in selected_obs
            else "Unknown",
            "niche": selected_obs[niche_column].to_numpy(),
        }
    )
    TEST_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    test_temporary = TEST_OUTPUT.with_suffix(f".tmp.{TEST_OUTPUT.suffix.lstrip('.')}")
    mapping.to_csv(test_temporary, index=False)
    test_file_hash = sha256_file(test_temporary)
    if test_file_hash != EXPECTED_TEST_FILE_HASH:
        raise RuntimeError(
            f"Test centers CSV hash mismatch: {test_file_hash} != {EXPECTED_TEST_FILE_HASH}"
        )
    if TEST_OUTPUT.exists():
        if sha256_file(TEST_OUTPUT) != EXPECTED_TEST_FILE_HASH:
            raise RuntimeError(f"Existing Test centers are not the frozen asset: {TEST_OUTPUT}")
        test_temporary.unlink()
    else:
        test_temporary.replace(TEST_OUTPUT)
    if not os.environ.get("FREQNICHE_RUN_DIR"):
        LEGACY_TEST_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        if LEGACY_TEST_OUTPUT.exists():
            if sha256_file(LEGACY_TEST_OUTPUT) != EXPECTED_TEST_FILE_HASH:
                raise RuntimeError(f"Existing legacy Test centers file is not frozen: {LEGACY_TEST_OUTPUT}")
        else:
            shutil.copy2(TEST_OUTPUT, LEGACY_TEST_OUTPUT)

    print(f"REBUILT {OUTPUT.relative_to(ROOT)}")
    print(f"ORDERED_SHA256 {actual_ordered_hash}")
    print(f"FILE_SHA256 {actual_file_hash}")
    print(f"TEST_FILE_SHA256 {test_file_hash}")


if __name__ == "__main__":
    main()
