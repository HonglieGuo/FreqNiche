"""Bind downstream inputs to one new training/evaluation run, never archived paths."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_FILES = {
    "human_skin": "datasets/h5ad/human_skin_merfish/human_skin_merfish.h5ad",
    "sea_ad": "datasets/h5ad/sea_ad/sea_ad_mtg_merfish.h5ad",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_inputs(run_dir: Path) -> Path:
    """Snapshot center ordering and bind every embedding after Test inference."""
    run_dir = run_dir.resolve()
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    dataset = manifest["request"]["dataset"]
    if dataset not in DATA_FILES:
        raise ValueError(f"No downstream prediction/expression analyses are defined for {dataset}")
    seeds = manifest["request"]["seeds"]
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("Run seeds must be nonempty and unique")
    config = run_dir / "model_config.json"
    if sha256(config) != manifest.get("model_config_sha256"):
        raise ValueError("Run model configuration hash mismatch")
    width = json.loads(config.read_text(encoding="utf-8"))["d_model"]
    bundle = {"version": 1, "dataset": dataset, "seeds": seeds,
              "embedding_width": width, "files": {}}
    target = run_dir / "downstream_inputs.json"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite downstream input bindings: {target}")
    center_dir = run_dir / "downstream_inputs"
    center_dir.mkdir(exist_ok=True)
    all_barcodes = []
    for split in ("validation", "test"):
        if dataset == "human_skin":
            name = "validation_barcodes_8k.csv" if split == "validation" else "test_centers_8k.csv"
            source = ROOT / "outputs/contracts/human_skin" / name
            frame = pd.read_csv(source, dtype={"cell_barcode": str})
        else:
            source = run_dir / ("val_centers.json" if split == "validation" else "test_centers.json")
            values = json.loads(source.read_text(encoding="utf-8"))["cell_barcodes"]
            frame = pd.DataFrame({"cell_barcode": values})
        if frame.empty or frame["cell_barcode"].isna().any() or frame["cell_barcode"].duplicated().any():
            raise ValueError(f"Invalid {split} center ordering")
        all_barcodes.append(set(frame["cell_barcode"].astype(str)))
        destination = center_dir / f"{split}.csv"
        if dataset == "human_skin":
            shutil.copyfile(source, destination)
        else:
            frame.to_csv(destination, index=False)
        bundle["files"][split] = {"path": destination.relative_to(run_dir).as_posix(),
                                   "sha256": sha256(destination)}
        for seed in seeds:
            if dataset == "human_skin":
                relative = (f"trial_24_seed_{seed}/val_embeddings.npy" if split == "validation"
                            else f"embeddings/freqniche_st_v1_trial24_seed_{seed}_test_z.npy")
            else:
                prefix = "val" if split == "validation" else "test"
                relative = f"embeddings/{prefix}_embeddings_seed_{seed}.npy"
            path = run_dir / relative
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            if values.shape != (len(frame), width) or not np.isfinite(values).all():
                raise ValueError(f"Invalid {split} embedding for seed {seed}: {values.shape}")
            bundle["files"][f"{split}_{seed}"] = {"path": relative, "sha256": sha256(path)}
    if all_barcodes[0] & all_barcodes[1]:
        raise ValueError("Validation and Test centers overlap")
    bundle["dataset_sha256"] = sha256(ROOT / DATA_FILES[dataset])
    bundle["model_config_sha256"] = sha256(config)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    temporary.replace(target)
    return target


class RunInputs:
    """Validate provenance and ordering once before any downstream computation."""

    def __init__(self, run_dir: str | Path, dataset: str):
        self.run_dir = Path(run_dir).expanduser().resolve()
        path = self.run_dir / "downstream_inputs.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Complete Test inference, then run "
                "python scripts/benchmark/export_downstream_inputs.py --run-dir <run>."
            )
        self.bundle = json.loads(path.read_text(encoding="utf-8"))
        if self.bundle.get("version") != 1 or self.bundle.get("dataset") != dataset:
            raise ValueError("Downstream dataset/version mismatch")
        self.seeds = tuple(self.bundle["seeds"])
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Invalid downstream seeds")
        run_manifest = json.loads((self.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        if run_manifest.get("status") != "complete":
            raise ValueError("Downstream tasks require a completed model/Test run")
        if (run_manifest["request"]["dataset"] != dataset
                or tuple(run_manifest["request"]["seeds"]) != self.seeds):
            raise ValueError("Downstream/run manifest mismatch")
        config = self.run_dir / "model_config.json"
        if sha256(config) != self.bundle["model_config_sha256"] or sha256(config) != run_manifest["model_config_sha256"]:
            raise ValueError("Downstream model configuration mismatch")
        self.width = json.loads(config.read_text(encoding="utf-8"))["d_model"]
        if self.width != self.bundle["embedding_width"]:
            raise ValueError("Downstream embedding width mismatch")
        self.data_path = ROOT / DATA_FILES[dataset]
        if sha256(self.data_path) != self.bundle["dataset_sha256"]:
            raise ValueError("Downstream source H5AD hash mismatch")
        self._paths = {}
        for key, record in self.bundle["files"].items():
            source = (self.run_dir / record["path"]).resolve()
            if not source.is_relative_to(self.run_dir) or sha256(source) != record["sha256"]:
                raise ValueError(f"Downstream file binding mismatch: {key}")
            self._paths[key] = source
        self.centers = {}
        for split in ("validation", "test"):
            frame = pd.read_csv(self.path(split), dtype={"cell_barcode": str})
            if frame.empty or frame["cell_barcode"].isna().any() or frame["cell_barcode"].duplicated().any():
                raise ValueError(f"Invalid {split} barcodes")
            self.centers[split] = frame
            for seed in self.seeds:
                self.embedding_path(split, seed)
        if set(self.centers["validation"]["cell_barcode"]) & set(self.centers["test"]["cell_barcode"]):
            raise ValueError("Validation/Test barcode overlap")

    def path(self, key: str) -> Path:
        return self._paths[key]

    def embedding_path(self, split: str, seed: int) -> Path:
        path = self.path(f"{split}_{seed}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != (len(self.centers[split]), self.width) or not np.isfinite(array).all():
            raise ValueError(f"Invalid embedding: {path}")
        return path
