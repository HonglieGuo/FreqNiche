#!/usr/bin/env python3
"""Bounded transductive SEDR / GraphST / SpaGCN spatial-domain comparison.

Each donor is fitted separately on its fixed Validation or Test center cohort.
This avoids cross-sample graph edges and dense million-cell matrices, but is a
separate center-only transductive track, not the full-context main benchmark.
Test is read only after a signed Validation resolution selection exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.benchmark.adapters.legacy_spatial import SOURCES, fit_official_embedding
from src.benchmark.adapters.common import source_tree_sha256
from src.benchmark.evaluation_w24 import select_unsupervised_resolution_r_star
from src.freqniche_st.runtime_paths import resolve_run_output
from src.benchmark.adapters import legacy_spatial, spatial_artifacts


DATASETS = {
    "human_skin": {
        "data": ROOT / "datasets/h5ad/human_skin_merfish/human_skin_merfish.h5ad",
        "val": ROOT / "outputs/contracts/human_skin/validation_barcodes_8k.csv",
        "test": ROOT / "outputs/contracts/human_skin/test_centers_8k.csv",
        "coords": "spatial",
    },
    "sea_ad": {
        "data": ROOT / "datasets/h5ad/sea_ad/sea_ad_mtg_merfish.h5ad",
        "val": "val_centers.json",
        "test": "test_centers.json",
        "coords": "spatial",
    },
    "cosmx_nsclc": {
        "data": ROOT / "datasets/h5ad/cosmx_human_nsclc/cosmx_human_nsclc.h5ad",
        "val": "val_centers.json",
        "test": "test_centers.json",
        "coords": "spatial_um",
    },
}
RESOLUTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signed_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(payload)
    value["self_signature_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def signed_read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    recorded = value.pop("self_signature_sha256")
    actual = hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()
    if recorded != actual:
        raise RuntimeError(f"Validation selection signature mismatch: {path}")
    return value


def _center_indices(dataset: str, split: str, obs_names: pd.Index, run_dir: Path) -> np.ndarray:
    path = DATASETS[dataset][split]
    path = Path(path) if isinstance(path, Path) else run_dir / path
    if dataset == "human_skin" and (run_dir / path.name).is_file():
        path = run_dir / path.name
    if not path.is_file():
        raise FileNotFoundError(f"Frozen {split} centers missing: {path}")
    if path.suffix == ".csv":
        barcodes = pd.read_csv(path)["cell_barcode"].astype(str).tolist()
        indices = obs_names.get_indexer(barcodes)
        if (indices < 0).any():
            raise RuntimeError(f"Center barcode absent from processed H5AD: {path}")
    else:
        centers = json.loads(path.read_text(encoding="utf-8"))
        indices = np.asarray(centers["global_indices"], dtype=np.int64)
        barcodes = centers.get("cell_barcodes")
        if barcodes is not None and not np.array_equal(obs_names[indices].astype(str), np.asarray(barcodes).astype(str)):
            raise RuntimeError(f"Center row/barcode mismatch: {path}")
    if len(indices) != len(np.unique(indices)) or (indices < 0).any():
        raise RuntimeError(f"Duplicate or invalid center indices: {path}")
    return np.asarray(indices, dtype=np.int64)


def _load_donor_center_data(dataset: str, split: str, run_dir: Path) -> dict[str, ad.AnnData]:
    config = DATASETS[dataset]
    source = ad.read_h5ad(config["data"], backed="r")
    try:
        indices = _center_indices(dataset, split, source.obs_names, run_dir)
        expected = "validation" if split == "val" else "test"
        if "split" in source.obs and not source.obs["split"].iloc[indices].isin(
            [expected, "val" if expected == "validation" else expected]
        ).all():
            raise RuntimeError(f"{split} centers include non-{expected} cells")
        metadata = source.obs.iloc[indices]
        groups: dict[str, ad.AnnData] = {}
        for donor in sorted(metadata["donor_id"].astype(str).unique()):
            donor_indices = indices[metadata["donor_id"].astype(str).to_numpy() == donor]
            block = source[donor_indices].to_memory()
            block.obsm["spatial"] = np.asarray(block.obsm[config["coords"]], dtype=np.float32).copy()
            groups[donor] = block
        return groups
    finally:
        source.file.close()


def _cluster(embedding: np.ndarray, resolution: float, seed: int) -> np.ndarray:
    cells = ad.AnnData(X=np.asarray(embedding, dtype=np.float32))
    sc.pp.neighbors(cells, n_neighbors=15, use_rep="X")
    sc.tl.leiden(cells, resolution=resolution, random_state=seed)
    return cells.obs["leiden"].astype(str).to_numpy()


def _code_binding(method: str) -> dict:
    return {
        "runner_sha256": sha256(Path(__file__)),
        "adapter_sha256": sha256(Path(legacy_spatial.__file__)),
        "artifact_writer_sha256": sha256(Path(spatial_artifacts.__file__)),
        "vendor_source_sha256": source_tree_sha256(SOURCES[method]),
    }


def _verify_identity(payload: dict, args, binding: dict) -> None:
    if (payload["method"], payload["dataset"], payload["seed"]) != (args.method, args.dataset, args.seed):
        raise RuntimeError("Saved phase identity mismatch")
    if not spatial_artifacts.code_binding_matches({key: payload.get(key) for key in binding}, binding):
        raise RuntimeError("Source changed or legacy phase has no complete assets")


def _verify_phase_assets(payload: dict, output: Path, groups: dict) -> None:
    records = payload.get("asset_manifests", {})
    if set(records) != set(groups):
        raise RuntimeError("Missing donor assets in saved phase")
    for donor, record in records.items():
        manifest_path = output / record["path"]
        if not manifest_path.resolve().is_relative_to(output.resolve()) or sha256(manifest_path) != record["sha256"]:
            raise RuntimeError(f"Donor asset manifest changed: {donor}")
        manifest = spatial_artifacts.verify_artifact(
            manifest_path.parent, expected_method=payload["method"], expected_seed=payload["seed"],
            expected_fingerprint=spatial_artifacts.data_fingerprint(groups[donor]),
        )
        if manifest["smoke"] or "clustering_resolution" not in manifest:
            raise RuntimeError("Formal phase has incomplete assignments or smoke assets")
        if payload.get("split") == "test":
            assignments = pd.read_csv(
                manifest_path.parent / "assignments.csv", dtype=str, keep_default_na=False,
            )
            if not np.array_equal(
                assignments["reference_label"].to_numpy(), groups[donor].obs["niche"].astype(str).to_numpy(),
            ):
                raise RuntimeError(f"Test reference labels changed since evaluation: {donor}")


def _record_asset(output: Path, directory: Path) -> dict:
    path = directory / "artifact_manifest.json"
    return {"path": path.relative_to(output).as_posix(), "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--method", choices=("sedr", "graphst", "spagcn"), required=True)
    parser.add_argument("--phase", choices=("validation", "test", "smoke"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--authorized", action="store_true")
    parser.add_argument("--run-dir", type=Path, help="New output directory with frozen center lists")
    parser.add_argument("--resume", action="store_true", help="Reuse only hash-verified complete donor assets")
    args = parser.parse_args()
    if args.phase == "test" and not args.authorized:
        parser.error("Test requires --authorized")

    run_dir = args.run_dir or resolve_run_output(f"baselines/Baselines_outputs/runs/additional_spatial/{args.dataset}")
    run_dir = run_dir if run_dir.is_absolute() else ROOT / run_dir
    output = run_dir / "additional_spatial" / args.method / f"seed_{args.seed}"
    binding = _code_binding(args.method)
    if args.phase == "smoke":
        groups = _load_donor_center_data(args.dataset, "val", run_dir)
        first = next(iter(groups.values()))[:80].copy()
        directory = output / "assets" / "smoke" / "fixture"
        z, protocol = fit_official_embedding(
            args.method, first, seed=args.seed, device=args.device, smoke=True,
            artifact_dir=directory, resume=args.resume,
        )
        spatial_artifacts.verify_artifact(directory, replay=True)
        print(f"SMOKE_PASS {args.method} {args.dataset} cells={len(z)} dim={z.shape[1]} protocol={protocol}")
        return

    if args.phase == "validation":
        target = output / "selection.json"
        groups = _load_donor_center_data(args.dataset, "val", run_dir)
        if target.exists():
            if not args.resume:
                raise FileExistsError(f"Validation selection already exists: {target}")
            saved = signed_read(target)
            _verify_identity(saved, args, binding)
            _verify_phase_assets(saved, output, groups)
            print(f"VERIFIED_RESUME {target}", flush=True)
            return
        selections, assets = {}, {}
        for donor, data in groups.items():
            directory = output / "assets" / "validation" / donor
            z, protocol = fit_official_embedding(
                args.method, data, seed=args.seed, device=args.device,
                artifact_dir=directory, resume=args.resume,
            )
            barcode_path = output / f"val_barcodes_{donor}.csv"
            barcode_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"cell_barcode": data.obs_names.astype(str)}).to_csv(barcode_path, index=False)
            r_star, report = select_unsupervised_resolution_r_star(
                z, resolutions=list(RESOLUTIONS),
                manifest_path=output / f"val_folds_{donor}.json",
                val_barcodes_path=barcode_path, seed=args.seed,
            )
            selections[donor] = {"r_star": float(r_star), "stability": report, "protocol": protocol}
            spatial_artifacts.save_assignments(directory, _cluster(z, r_star, args.seed), r_star)
            assets[donor] = _record_asset(output, directory)
            print(f"VALIDATION {args.method} donor={donor} r*={r_star}", flush=True)
        frozen_r = float(sorted(RESOLUTIONS, key=lambda r: (abs(r - np.median(
            [entry["r_star"] for entry in selections.values()])), r))[0])
        signed_write(target, {
            "method": args.method, "dataset": args.dataset, "seed": args.seed,
            "paradigm": "transductive_center_only_donor_wise",
            "frozen_r_star": frozen_r, "per_validation_donor": selections,
            **binding, "asset_manifests": assets,
        })
        print(f"FROZEN_SELECTION {target} r*={frozen_r}", flush=True)
        return

    selection_path = output / "selection.json"
    selection = signed_read(selection_path)
    _verify_identity(selection, args, binding)
    result_path = output / "test_result.json"
    groups = _load_donor_center_data(args.dataset, "test", run_dir)
    if result_path.exists():
        if not args.resume:
            raise FileExistsError(f"Test result already exists: {result_path}")
        saved = signed_read(result_path)
        _verify_identity(saved, args, binding)
        _verify_phase_assets(saved, output, groups)
        if saved["selection_sha256"] != sha256(selection_path):
            raise RuntimeError("Frozen validation selection changed")
        print(f"VERIFIED_RESUME {result_path}", flush=True)
        return
    per_donor, assets = {}, {}
    for donor, data in groups.items():
        directory = output / "assets" / "test" / donor
        z, protocol = fit_official_embedding(
            args.method, data, seed=args.seed, device=args.device,
            artifact_dir=directory, resume=args.resume,
        )
        labels = _cluster(z, selection["frozen_r_star"], args.seed)
        truth = data.obs["niche"].to_numpy()
        valid = ~pd.isna(truth)
        per_donor[donor] = {
            "ari": float(adjusted_rand_score(truth[valid], labels[valid])),
            "nmi": float(normalized_mutual_info_score(truth[valid], labels[valid])),
            "n_clusters": int(len(np.unique(labels))),
            "cells": int(len(labels)), "protocol": protocol,
        }
        spatial_artifacts.save_assignments(
            directory, labels, selection["frozen_r_star"], reference=truth,
        )
        assets[donor] = _record_asset(output, directory)
        print(f"TEST {args.method} donor={donor} ARI={per_donor[donor]['ari']:.4f}", flush=True)
    signed_write(result_path, {
        "method": args.method, "dataset": args.dataset, "seed": args.seed,
        "split": "test", "paradigm": "transductive_center_only_donor_wise",
        "evaluation_track": "additional_spatial_bounded",
        "clusterer": "leiden", "leiden_resolution": selection["frozen_r_star"],
        "donor_macro_ari": float(np.mean([v["ari"] for v in per_donor.values()])),
        "donor_macro_nmi": float(np.mean([v["nmi"] for v in per_donor.values()])),
        "worst_donor_ari": float(min(v["ari"] for v in per_donor.values())),
        "per_donor": per_donor, "selection_sha256": sha256(selection_path),
        **binding, "asset_manifests": assets,
    })
    print(f"TEST_RESULT {result_path}", flush=True)


if __name__ == "__main__":
    main()
