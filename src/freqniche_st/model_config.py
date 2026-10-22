"""One model-configuration contract for training and frozen embedding readouts."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
MODEL_PARAMETER_MAP = {
    "k_max": "k_max", "rho": "rho", "p_center_mask": "p_center_mask",
    "m_max": "m_max", "hidden_dim": "d_model",
    "n_encoder_layers": "n_enc", "n_diffusion_layers": "n_dit",
    "n_heads": "n_heads", "diffusion_timesteps": "timesteps",
    "learning_rate": "lr", "weight_decay": "wd",
    "covariance_weight": "lambda_cov",
}
INTEGER_KEYS = {"k_max", "m_max", "d_model", "n_enc", "n_dit", "n_heads", "timesteps"}


def manuscript_defaults(dataset: str) -> tuple[dict[str, int | float], dict[str, int | float]]:
    """Dataset recipes from manuscript Table S02 and supplementary training details.

    Human Skin/CosMx use the selected 128/3/2 base model and 50-epoch budget.
    SEA-AD/Xenium use the reported 256/6/4 expansion and 200-epoch budget.
    Return independent dictionaries so editable runs cannot mutate defaults.
    """
    if dataset not in {"human_skin", "sea_ad", "cosmx_nsclc", "xenium_prime5k"}:
        raise ValueError(f"No manuscript configuration for dataset {dataset!r}")
    with (ROOT / "configs/freqniche/default.yaml").open("r", encoding="utf-8") as handle:
        configuration = yaml.safe_load(handle)
    model = {key: configuration["model"][key] for key in MODEL_PARAMETER_MAP}
    training = {key: configuration["training"][key]
                for key in ("max_epochs", "patience", "batch_size", "min_delta")}
    if dataset in {"sea_ad", "xenium_prime5k"}:
        model.update(hidden_dim=256, n_encoder_layers=6, n_diffusion_layers=4, learning_rate=1e-4)
        training["max_epochs"] = 200
    internal_from_public(model)
    return model, training


def frozen_internal_config() -> dict[str, int | float]:
    with (ROOT / "configs/freqniche/default.yaml").open("r", encoding="utf-8") as handle:
        public = yaml.safe_load(handle)["model"]
    return {internal: public[external] for external, internal in MODEL_PARAMETER_MAP.items()}


def validate_internal_config(config: dict[str, Any]) -> dict[str, int | float]:
    required = set(MODEL_PARAMETER_MAP.values())
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError(f"Model configuration keys must be exactly {sorted(required)}")
    for key, value in config.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Model configuration {key} must be a finite number")
        if key in INTEGER_KEYS and (not isinstance(value, int) or value < 1):
            raise ValueError(f"Model configuration {key} must be a positive integer")
    if config["k_max"] < 2 or config["d_model"] % config["n_heads"]:
        raise ValueError("k_max must be at least 2 and hidden_dim must divide evenly by n_heads")
    if not (0 < config["rho"] < 1 and 0 < config["p_center_mask"] < 1):
        raise ValueError("rho and p_center_mask must be strictly between 0 and 1")
    if config["lr"] <= 0 or config["wd"] < 0 or config["lambda_cov"] < 0:
        raise ValueError("learning_rate must be positive; decay and covariance weight must be nonnegative")
    return dict(config)


def internal_from_public(public: dict[str, Any]) -> dict[str, int | float]:
    if set(public) - set(MODEL_PARAMETER_MAP):
        raise ValueError(f"Unknown public model keys: {sorted(set(public) - set(MODEL_PARAMETER_MAP))}")
    config = frozen_internal_config()
    config.update({MODEL_PARAMETER_MAP[key]: value for key, value in public.items()})
    return validate_internal_config(config)


def build_freqniche_model(in_features: int, config: dict[str, Any]):
    """Construct the same architecture for training, Validation and Test."""
    from .models import FreqNicheDiTModel

    cfg = validate_internal_config(config)
    return FreqNicheDiTModel(
        in_features=in_features, k_target=32, d_model=cfg["d_model"],
        n_enc_layers=cfg["n_enc"], n_dit_layers=cfg["n_dit"],
        n_heads=cfg["n_heads"], timesteps=cfg["timesteps"],
        lambda_cov=cfg["lambda_cov"], mode="full_freq",
    )


def load_model_config(path: Path | None = None, *, output_dir: Path | None = None) -> dict[str, int | float]:
    if path is None and output_dir is not None:
        candidate = Path(output_dir) / "model_config.json"
        if candidate.is_file():
            path = candidate
        elif os.environ.get("FREQNICHE_RUN_DIR"):
            raise FileNotFoundError(f"Orchestrated run is missing its model configuration: {candidate}")
    if path is None:
        return validate_internal_config(frozen_internal_config())
    path = Path(path)
    config = validate_internal_config(json.loads(path.read_text(encoding="utf-8")))
    if output_dir is not None:
        manifest_path = Path(output_dir) / "run_manifest.json"
        if os.environ.get("FREQNICHE_RUN_DIR") and not manifest_path.is_file():
            raise FileNotFoundError(f"Orchestrated run is missing its manifest: {manifest_path}")
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = manifest.get("model_config_sha256")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if expected != actual:
                raise RuntimeError(f"Model configuration SHA-256 mismatch: {path}")
    return config


def verify_training_config(record: dict[str, Any], config: dict[str, Any], *, asset: str) -> None:
    """Reject a checkpoint trained under different settings in a new run.

    Historical frozen assets did not record this field and retain their old
    checkpoint-hash checks; orchestrated runs require the complete record.
    """
    if os.environ.get("FREQNICHE_RUN_DIR") and record.get("model_config") != config:
        raise RuntimeError(f"Training model configuration mismatch for {asset}")


def inference_batch_size(output_dir: Path, historical_default: int) -> int:
    if not os.environ.get("FREQNICHE_RUN_DIR"):
        return historical_default
    manifest_path = Path(output_dir) / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Orchestrated run is missing its manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    batch_size = manifest.get("request", {}).get("batch_size")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"Invalid inference batch size in {manifest_path}")
    return batch_size
