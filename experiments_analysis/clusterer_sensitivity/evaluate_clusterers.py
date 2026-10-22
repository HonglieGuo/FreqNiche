"""Evaluate four clusterers on Test with frozen and Oracle readouts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import clusterer_common as common


def load_selection(suite: str) -> tuple[dict, dict, dict[str, dict]]:
    root = common.selection_dir(suite)
    manifest = common.load_signed_json(root / "selection_manifest.json")
    if manifest.get("manifest_version") != "1.0" or manifest.get("suite") != suite:
        raise RuntimeError("Unexpected clusterer selection manifest")
    contract_path = root / "run_contract.json"
    if common.compute_sha256(contract_path) != manifest["contract_sha256"]:
        raise RuntimeError("Selection contract SHA mismatch")
    contract = common.load_signed_json(contract_path)
    if contract.get("environment") != common.environment_versions():
        raise RuntimeError("Runtime environment differs from the frozen Validation-selection environment")
    shards = {}
    for key, record in manifest["records"].items():
        shard_path = root / record["shard_path"]
        if common.compute_sha256(shard_path) != record["shard_sha256"]:
            raise RuntimeError(f"Selection shard SHA mismatch: {key}")
        shard = common.load_signed_json(shard_path)
        pca_record = shard.get("mclust_preprocessing", {}).get("pca_state", {})
        pca_path = root / pca_record.get("path", "")
        if not pca_path.is_file() or common.compute_sha256(pca_path) != pca_record.get("sha256"):
            raise RuntimeError(f"Selection PCA state SHA mismatch: {key}")
        shards[key] = shard
    return manifest, contract, shards


def save_assignment(path: Path, labels: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int32)
    with path.open("wb") as handle:
        np.save(handle, labels)
    return {"path": path.name, "sha256": common.compute_sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Test evaluation of frozen clusterer selections")
    parser.add_argument("--suite", choices=("freqniche", "baselines"), required=True)
    parser.add_argument("--authorized", action="store_true")
    args = parser.parse_args()
    if not args.authorized:
        parser.error("Test evaluation requires --authorized")

    final = common.evaluation_dir(args.suite)
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite Test evaluation: {final}")
    selection_manifest, selection_contract, shards = load_selection(args.suite)
    pairs = common.registry(args.suite)
    if set(shards) != {common.method_seed_key(pair) for pair in pairs}:
        raise RuntimeError("Selection/evaluation registry mismatch")

    metadata = common.load_test_metadata()
    target_k = common.target_k_from_metadata(metadata)
    staging = common.OUTPUT_ROOT / f".{args.suite}_evaluation_v1.tmp.{os.getpid()}"
    if staging.exists():
        raise RuntimeError(f"Stale evaluation staging directory: {staging}")
    assignment_dir = staging / "assignments"
    assignment_dir.mkdir(parents=True)
    try:
        rows = []
        records = {}
        assignment_hashes = {}
        input_embeddings = {}
        input_pca_states = {}
        for pair_index, pair in enumerate(pairs, start=1):
            key = common.method_seed_key(pair)
            print(f"[{pair_index}/{len(pairs)}] evaluating {key}", flush=True)
            embedding = common.validate_embedding(pair.test_path)
            input_embeddings[key] = common.input_record(pair.test_path)
            shard = shards[key]
            pca_record = shard["mclust_preprocessing"]["pca_state"]
            pca_path = common.selection_dir(args.suite) / pca_record["path"]
            input_pca_states[key] = common.input_record(pca_path)
            mclust_embedding = common.project_with_frozen_pca(
                embedding,
                pca_path,
                shard["validation_embedding_sha256"],
            )
            clusterer_records = {}
            for clusterer in common.CLUSTERERS:
                print(f"  - {clusterer}: frozen Validation parameter", flush=True)
                parameter = shard["selections"][clusterer]["selected_parameter"]
                evaluated_embedding = mclust_embedding if clusterer == "mclust" else embedding
                labels_main = common.cluster_full(
                    evaluated_embedding, clusterer, parameter, pair.numeric_seed
                )
                metrics_main = common.evaluate_labels(labels_main, metadata)
                main_path = assignment_dir / f"{key}__{clusterer}__validation_frozen.npy"
                main_assignment = save_assignment(main_path, labels_main)
                assignment_hashes[f"assignments/{main_path.name}"] = main_assignment["sha256"]

                print(f"  - {clusterer}: Oracle true-K diagnostic", flush=True)
                if clusterer == "kmeans":
                    labels_oracle = common.cluster_full(
                        evaluated_embedding, clusterer, target_k, pair.numeric_seed
                    )
                    achieved_k = int(len(np.unique(labels_oracle)))
                    oracle_selection = {
                        "target_k": target_k,
                        "selected_k": achieved_k,
                        "absolute_k_error": abs(achieved_k - target_k),
                        "exact_k_match": achieved_k == target_k,
                        "parameter": target_k,
                        "parameter_semantics": "requested_clusters_K",
                        "selection_rule": "Oracle K-means fixes requested K to Test true K; label identities and ARI/NMI excluded",
                    }
                elif clusterer == "mclust":
                    labels_oracle = common.cluster_full(
                        evaluated_embedding, clusterer, target_k, pair.numeric_seed
                    )
                    achieved_k = int(len(np.unique(labels_oracle)))
                    oracle_selection = {
                        "target_k": target_k,
                        "requested_g": target_k,
                        "selected_k": achieved_k,
                        "absolute_k_error": abs(achieved_k - target_k),
                        "exact_k_match": achieved_k == target_k,
                        "empty_component_collapse": achieved_k < target_k,
                        "parameter": target_k,
                        "parameter_semantics": "requested_components_G",
                        "selection_rule": "Oracle mclust fixes requested G to Test true K; achieved K counts non-empty classes; label identities and ARI/NMI excluded",
                    }
                else:
                    labels_oracle, oracle_selection = common.select_graph_parameter_by_target_k(
                        evaluated_embedding, clusterer, target_k, pair.numeric_seed
                    )
                    oracle_selection["parameter"] = oracle_selection["selected_resolution"]
                metrics_oracle = common.evaluate_labels(labels_oracle, metadata)
                oracle_path = assignment_dir / f"{key}__{clusterer}__oracle_true_k.npy"
                oracle_assignment = save_assignment(oracle_path, labels_oracle)
                assignment_hashes[f"assignments/{oracle_path.name}"] = oracle_assignment["sha256"]

                clusterer_records[clusterer] = {
                    "validation_frozen": {
                        "parameter": parameter,
                        "parameter_semantics": shard["selections"][clusterer]["selected_parameter_semantics"],
                        "input_preprocessing": shard["selections"][clusterer]["input_preprocessing"],
                        "evaluated_dimension": shard["selections"][clusterer]["evaluated_dimension"],
                        "validation_stability": shard["selections"][clusterer]["selected_stability"],
                        "metrics": metrics_main,
                        "assignment": {"path": f"assignments/{main_path.name}", "sha256": main_assignment["sha256"]},
                    },
                    "oracle_true_k": {
                        "input_preprocessing": shard["selections"][clusterer]["input_preprocessing"],
                        "evaluated_dimension": shard["selections"][clusterer]["evaluated_dimension"],
                        "selection": oracle_selection,
                        "metrics": metrics_oracle,
                        "assignment": {"path": f"assignments/{oracle_path.name}", "sha256": oracle_assignment["sha256"]},
                    },
                }
                for track, parameter_used, metrics in (
                    ("validation_frozen", parameter, metrics_main),
                    ("oracle_true_k", oracle_selection["parameter"], metrics_oracle),
                ):
                    rows.append({
                        "suite": args.suite,
                        "method": pair.method,
                        "seed": pair.seed_label,
                        "clusterer": clusterer,
                        "track": track,
                        "parameter": parameter_used,
                        "parameter_semantics": (
                            shard["selections"][clusterer]["selected_parameter_semantics"]
                            if track == "validation_frozen"
                            else oracle_selection.get("parameter_semantics", "community_resolution")
                        ),
                        "input_preprocessing": shard["selections"][clusterer]["input_preprocessing"],
                        "native_dimension": int(embedding.shape[1]),
                        "evaluated_dimension": int(evaluated_embedding.shape[1]),
                        "target_k": target_k if track == "oracle_true_k" else "",
                        "n_clusters": metrics["n_clusters"],
                        "donor_macro_ari": metrics["donor_macro_ari"],
                        "donor_macro_nmi": metrics["donor_macro_nmi"],
                        "worst_donor_ari": metrics["worst_donor_ari"],
                        "donor_ari_sample_sd": metrics["donor_ari_sample_sd"],
                    })
            records[key] = {
                "method": pair.method,
                "seed_label": pair.seed_label,
                "numeric_seed": pair.numeric_seed,
                "test_embedding": common.input_record(pair.test_path),
                "mclust_pca_state": {
                    "path": common.relative_path(pca_path),
                    "sha256": common.compute_sha256(pca_path),
                    "source_validation_embedding_sha256": shard["validation_embedding_sha256"],
                },
                "clusterers": clusterer_records,
            }

        csv_path = staging / "clusterer_results.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        report = {
            "report_version": "1.0",
            "analysis": "Human Skin four-clusterer sensitivity",
            "suite": args.suite,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "tracks": {
                "validation_frozen": "primary label-free sensitivity: hyperparameter selected only by Validation bootstrap stability",
                "oracle_true_k": "diagnostic only: Test true K is used, label identities and ARI/NMI are excluded from selection",
            },
            "target_k": target_k,
            "environment": selection_contract["environment"],
            "records": records,
            "inputs": {
                "runner": common.input_record(Path(__file__)),
                "common": common.input_record(Path(common.__file__)),
                "mclust_batch_R": common.input_record(common.MCLUST_BATCH_SCRIPT),
                "selection_manifest": common.input_record(common.selection_dir(args.suite) / "selection_manifest.json"),
                "test_metadata": common.input_record(common.TEST_METADATA),
                "embeddings": input_embeddings,
                "mclust_validation_fitted_pca_states": input_pca_states,
            },
            "outputs": {
                "clusterer_results.csv": common.compute_sha256(csv_path),
                **assignment_hashes,
            },
        }
        common.atomic_write_json(staging / "evaluation_report.json", report)
        os.replace(staging, final)
        print(f"Evaluation completed and published: {final}", flush=True)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
