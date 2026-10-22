# Local reruns and saved assets for SEDR, GraphST and SpaGCN

A search of the original formal runs did not locate model weights or embeddings for these three methods. Those runs retained metrics, Validation resolutions and cell lists only. The rerun workflow uses the original frozen evaluation centers, saves complete assets and compares the new results with the original records separately.

## Direct execution

Run the following command from the repository root **inside Linux/WSL**, using an environment with the required baseline dependencies. The default selection is Human Skin, SEA-AD and CosMx; methods are SEDR, GraphST and SpaGCN; seeds are 42, 123 and 2024. Replace the interpreter path if your environment is installed elsewhere.

~~~bash
$HOME/.venvs/FreqNiche-spatial-baselines/bin/python scripts/benchmark/rerun_additional_spatial_assets.py --output-dir baselines/Baselines_outputs/additional_spatial_assets_20261016 --device cuda --resume
~~~

Use the same command for the first run and after an interruption. Completed donor models are reused after file checksums and input fingerprints are verified. Incomplete donor models are retrained. This workflow does not save optimizer state for resuming individual training epochs.

On 2026-10-16, a fix addressed nullable string indexes that could not be written to CosMx H5AD files. Only saved copies are converted: string indexes become ordinary string objects and string columns become categorical columns, preserving IDs, order and missing values. Training inputs, models and evaluation protocols are unchanged. The compatibility manifest `src/benchmark/adapters/spatial_artifact_compatibility.json` records the exact code hashes before and after this fix, allowing completed Human Skin and SEA-AD assets to be reused without rewriting the original assets or stage manifests. Temporary files from the failed first CosMx donor save were removed, so that donor requires retraining. Remaining tasks can continue without rerunning the 162 donor bundles already completed at that point.

The default workflow comprises 54 Validation/Test stage processes and produces 180 donor asset bundles:

| Dataset | Validation donors | Test donors | Bundles across three methods and three seeds |
| --- | ---: | ---: | ---: |
| Human Skin | 4 | 4 | 72 |
| SEA-AD | 5 | 5 | 90 |
| CosMx | 1 | 1 | 18 |

These methods fit independently for each donor, so a method/dataset/seed combination produces multiple donor models. Validation and Test models are saved separately.

SEDR and GraphST use the requested CUDA device. SpaGCN retains the CPU computation used by its official implementation. The actual device is recorded in each model file.

Append `--dry-run` to inspect the command plan, or `--preflight-only` to check data and frozen cell lists only. Neither option starts training.

## Output location

With the example command, outputs are written relative to the repository root:

~~~text
baselines/Baselines_outputs/additional_spatial_assets_20261016/
~~~

Example directory layout:

~~~text
additional_spatial_assets_20261016/
  asset_backfill.json
  asset_audit.json
  comparison.csv
  human_skin/
    logs/
    additional_spatial/
      graphst/
        seed_42/
          selection.json
          test_result.json
          assets/
            validation/<donor>/
            test/D028/
              model.pt
              model_inputs.pt
              embedding.npy
              native_embedding.npy
              preprocess_pca.joblib
              processed_input.h5ad
              cells.csv
              input_genes.csv
              assignments.csv
              artifact_manifest.json
~~~

Each donor bundle contains:

- **model.pt:** trained weights, model construction parameters, method, seed and actual device.
- **model_inputs.pt:** the features and graph tensors actually used for inference; SEDR additionally records the nodes that were randomly masked.
- **embedding.npy:** the embedding used for clustering and metric calculation.
- **native_embedding.npy:** required for GraphST only; stores the representation before the final PCA readout.
- **preprocess_pca.joblib:** fitted PCA objects, including SpaGCN's internal PCA.
- **processed_input.h5ad:** preprocessed expression, gene order, scaling statistics and spatial coordinates, without niche or cell-type supervision labels.
- **cells.csv:** cell barcodes, donors and physical samples in embedding-row order.
- **input_genes.csv:** gene order before preprocessing; the filtered order is stored in the H5AD.
- **assignments.csv:** per-cell clustering results; the Test stage additionally records reference labels used only for evaluation.
- **artifact_manifest.json:** file sizes, SHA-256 checksums, input fingerprints, software versions, embedding shapes and clustering resolutions.
- SpaGCN additionally saves compressed **graph_sparse.npz** or **graph_dense.npz** files.

Model bundles are first written to temporary directories. Weights are then reloaded and native embeddings replayed; bundles are published to their final donor directories only after verification. Metric JSON files are written after all donor assets and clustering results are complete.

## Completion and audit

Successful completion prints:

~~~text
BACKFILL_COMPLETE ... donor_bundles=180
~~~

To verify all saved files and replay inference from the weights:

~~~bash
$HOME/.venvs/FreqNiche-spatial-baselines/bin/python scripts/benchmark/rerun_additional_spatial_assets.py --output-dir baselines/Baselines_outputs/additional_spatial_assets_20261016 --device cuda --audit-only --replay
~~~

Check that `asset_backfill.json` has status `complete` and `asset_audit.json` has status `passed`. Progress and errors are recorded in each dataset's `logs/` directory.

`comparison.csv` lists the original manuscript ARI, new ARI/NMI, standard deviations and ARI differences by method and dataset. New results are written to a separate directory and do not automatically update the original metrics or manuscript. Dependency versions and numerical differences can affect rerun results; inspect the comparison before adopting them.

## Loading one donor

From the repository root, using the Python environment above:

~~~python
from pathlib import Path
import numpy as np
import pandas as pd
from src.benchmark.adapters.spatial_artifacts import verify_artifact

folder = Path(
    "baselines/Baselines_outputs/additional_spatial_assets_20261016/"
    "human_skin/additional_spatial/graphst/seed_42/assets/test/D028"
)
manifest = verify_artifact(folder, replay=True)
z = np.load(folder / "embedding.npy")
cells = pd.read_csv(folder / "cells.csv", dtype=str, keep_default_na=False)
assert len(z) == len(cells)
~~~

Embeddings fitted independently for different donors have not been aligned across donors. Do not treat them as a shared coordinate system for joint clustering. GraphST's original readout uses PCA `fit_transform`; reading the saved `embedding.npy` preserves the exact coordinates used in that run.

## Verification performed

- Saving and non-saving execution paths produced identical embeddings and random-number states for all three methods' upstream code on 80 synthetic cells.
- Small-sample save/reload checks covered the formal training branches, including SEDR DEC and SpaGCN Louvain initialization.
- Weight reloading and native-embedding replay check for corrupted files, modified weights, cell-order mismatches and input-fingerprint changes.
- End-to-end Validation, Test and unified-launcher checks confirmed that resuming completed runs does not retrain them, while missing weights or changed Test annotations prevent reuse.
- Frozen donors, Validation cell order and Test cell counts were prechecked for all three formal datasets.

Full-scale reruns are performed by the user. These small-sample checks do not replace the final asset audit after a complete run.
