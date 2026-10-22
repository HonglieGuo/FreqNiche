"""Select clusterer hyperparameters from Validation stability only.

The runner is resumable at the method-by-seed shard level. Test embeddings and
all biological labels are outside this phase.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import clusterer_common as common


def build_contract(suite: str, pairs: list[common.EmbeddingPair]) -> dict:
    return {
        "contract_version": "1.0",
        "suite": suite,
        "protocol": {
            "clusterers": list(common.CLUSTERERS),
            "kmeans_k_candidates": list(common.K_CANDIDATES),
            "mclust_g_candidates": list(common.K_CANDIDATES),
            "resolution_candidates": list(common.RESOLUTION_CANDIDATES),
            "n_neighbors": common.N_NEIGHBORS,
            "selection": "maximize mean pairwise ARI across the shared ten Validation 80% folds; ties choose smaller parameter",
            "cluster_count_bounds": [5, 25],
            "kmeans": "k-means++; n_init=20; max_iter=500",
            "mclust": "Validation-fitted PCA20 then R mclust::Mclust; modelNames=EEE; requested G may yield fewer non-empty classes; no fallback",
            "mclust_preprocessing": {
                "fit_split": "complete Validation embedding",
                "n_components": common.MCLUST_PCA_DIM,
                "svd_solver": "full",
                "test_policy": "apply the frozen Validation mean/components; never refit on Test",
            },
        },
        "environment": common.assert_python_environment_matches_lock(),
        "code": {
            "runner": common.input_record(Path(__file__)),
            "common": common.input_record(Path(common.__file__)),
            "mclust_batch_R": common.input_record(common.MCLUST_BATCH_SCRIPT),
        },
        "shared_inputs": {
            "bootstrap_manifest": common.input_record(common.BOOTSTRAP_MANIFEST),
            "validation_barcodes": common.input_record(common.VAL_BARCODES),
            "validation_barcode_order_sha256": common.compute_ordered_barcodes_sha256(common.VAL_BARCODES),
            "environment_lock": common.input_record(common.CLUSTERER_ENV_LOCK),
        },
        "embeddings": {
            common.method_seed_key(pair): {
                "method": pair.method,
                "seed_label": pair.seed_label,
                "numeric_seed": pair.numeric_seed,
                "validation": common.input_record(pair.validation_path),
            }
            for pair in pairs
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only clusterer hyperparameter selection")
    parser.add_argument("--suite", choices=("freqniche", "baselines"), required=True)
    args = parser.parse_args()

    final = common.selection_dir(args.suite)
    work = common.work_dir(args.suite)
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite completed selection: {final}")
    common.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    pairs = common.registry(args.suite, include_test_verification=False)
    contract = build_contract(args.suite, pairs)
    contract_path = work / "run_contract.json"
    shard_dir = work / "shards"
    pca_dir = work / "pca"

    if work.exists():
        if not contract_path.is_file():
            raise RuntimeError(f"Unverifiable interrupted work directory: {work}")
        existing = common.load_signed_json(contract_path)
        if existing != contract:
            raise RuntimeError("Resume contract mismatch; refusing to mix code/protocol/assets")
    else:
        shard_dir.mkdir(parents=True)
        pca_dir.mkdir(parents=True)
        common.atomic_write_json(contract_path, contract)
    pca_dir.mkdir(parents=True, exist_ok=True)

    folds, _ = common.load_folds()

    summaries = {}
    for pair_index, pair in enumerate(pairs, start=1):
        key = common.method_seed_key(pair)
        shard_path = shard_dir / f"{key}.json"
        pca_path = pca_dir / f"{key}__validation_pca20.npz"
        if shard_path.exists():
            shard = common.load_signed_json(shard_path)
            if shard.get("validation_embedding_sha256") != common.compute_sha256(pair.validation_path):
                raise RuntimeError(f"Resumed shard embedding mismatch: {key}")
            pca_record = shard.get("mclust_preprocessing", {}).get("pca_state", {})
            if pca_record.get("path") != f"pca/{pca_path.name}":
                raise RuntimeError(f"Resumed shard PCA path mismatch: {key}")
            if common.compute_sha256(work / pca_record["path"]) != pca_record.get("sha256"):
                raise RuntimeError(f"Resumed shard PCA SHA mismatch: {key}")
            print(f"[{pair_index}/{len(pairs)}] resume verified: {key}", flush=True)
        else:
            print(f"[{pair_index}/{len(pairs)}] selecting clusterers: {key}", flush=True)
            embedding = common.validate_embedding(pair.validation_path)
            validation_embedding_sha = common.compute_sha256(pair.validation_path)
            mclust_embedding, pca_state = common.fit_validation_pca(
                embedding, validation_embedding_sha
            )
            common.atomic_write_pca_state(pca_path, pca_state)
            selections = {}
            for clusterer in common.CLUSTERERS:
                evaluated_embedding = mclust_embedding if clusterer == "mclust" else embedding
                suffix = " (Validation-fitted PCA20)" if clusterer == "mclust" else " (native embedding)"
                print(f"  - {clusterer}{suffix}", flush=True)
                selections[clusterer] = common.selection_stability(
                    evaluated_embedding, folds, clusterer, pair.numeric_seed
                )
                selections[clusterer]["input_preprocessing"] = (
                    "validation_fitted_PCA20" if clusterer == "mclust" else "native_embedding"
                )
                selections[clusterer]["evaluated_dimension"] = int(evaluated_embedding.shape[1])
            shard = {
                "shard_version": "1.0",
                "suite": args.suite,
                "method": pair.method,
                "seed_label": pair.seed_label,
                "numeric_seed": pair.numeric_seed,
                "validation_embedding_path": common.relative_path(pair.validation_path),
                "validation_embedding_sha256": validation_embedding_sha,
                "validation_embedding_shape": list(embedding.shape),
                "mclust_preprocessing": {
                    "protocol": "validation_fitted_PCA20",
                    "fit_split": "validation",
                    "native_dimension": int(embedding.shape[1]),
                    "evaluated_dimension": common.MCLUST_PCA_DIM,
                    "source_validation_embedding_sha256": validation_embedding_sha,
                    "pca_state": {
                        "path": f"pca/{pca_path.name}",
                        "sha256": common.compute_sha256(pca_path),
                    },
                },
                "selections": selections,
            }
            common.atomic_write_json(shard_path, shard)
        summaries[key] = {
            "method": shard["method"],
            "seed_label": shard["seed_label"],
            "numeric_seed": shard["numeric_seed"],
            "selected_parameters": {
                clusterer: shard["selections"][clusterer]["selected_parameter"]
                for clusterer in common.CLUSTERERS
            },
            "selected_parameter_semantics": {
                clusterer: shard["selections"][clusterer]["selected_parameter_semantics"]
                for clusterer in common.CLUSTERERS
            },
            "selected_stabilities": {
                clusterer: shard["selections"][clusterer]["selected_stability"]
                for clusterer in common.CLUSTERERS
            },
            "mclust_preprocessing": shard["mclust_preprocessing"],
            "shard_path": f"shards/{shard_path.name}",
            "shard_sha256": common.compute_sha256(shard_path),
        }

    manifest = {
        "manifest_version": "1.0",
        "analysis": "Human Skin clusterer-sensitivity Validation selection",
        "suite": args.suite,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "test_data_accessed": False,
        "labels_accessed": False,
        "contract_sha256": common.compute_sha256(contract_path),
        "records": summaries,
    }
    common.atomic_write_json(work / "selection_manifest.json", manifest)
    os.replace(work, final)
    print(f"Selection completed and published: {final}", flush=True)


if __name__ == "__main__":
    main()
