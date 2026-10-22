"""Atomic training checkpoints with all random streams needed at epoch boundaries."""
from __future__ import annotations

import os
from pathlib import Path
import random

import numpy as np
import torch


def atomic_torch_save(payload: dict, path: Path) -> None:
    """Replace a checkpoint only after its complete contents reach the filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _encode_numpy(state: tuple) -> dict:
    # Use primitives instead of NumPy pickle globals so weights_only=True works.
    return {"algorithm": state[0], "keys": state[1].tolist(), "position": int(state[2]),
            "has_gauss": int(state[3]), "cached_gauss": float(state[4])}


def _decode_numpy(state: dict) -> tuple:
    return (state["algorithm"], np.asarray(state["keys"], dtype=np.uint32),
            state["position"], state["has_gauss"], state["cached_gauss"])


def _dataset_rngs(dataset) -> list:
    if hasattr(dataset, "datasets"):
        return [rng for child in dataset.datasets for rng in _dataset_rngs(child)]
    return [dataset.rng]


def capture_rng_state(generator, train_dataset, val_dataset, device) -> dict:
    return {"python": random.getstate(), "numpy": _encode_numpy(np.random.get_state()),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "loader": generator.get_state(),
            "train_datasets": [_encode_numpy(rng.get_state()) for rng in _dataset_rngs(train_dataset)],
            "val_datasets": [_encode_numpy(rng.get_state()) for rng in _dataset_rngs(val_dataset)]}


def restore_rng_state(state, generator, train_dataset, val_dataset, device) -> None:
    train_rngs, val_rngs = _dataset_rngs(train_dataset), _dataset_rngs(val_dataset)
    if len(train_rngs) != len(state["train_datasets"]) or len(val_rngs) != len(state["val_datasets"]):
        raise ValueError("Dataset random-stream count changed on resume")
    random.setstate(state["python"])
    np.random.set_state(_decode_numpy(state["numpy"]))
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda":
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count changed on resume")
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    generator.set_state(state["loader"])
    for rng, saved in zip(train_rngs, state["train_datasets"]):
        rng.set_state(_decode_numpy(saved))
    for rng, saved in zip(val_rngs, state["val_datasets"]):
        rng.set_state(_decode_numpy(saved))
