# CosMx key design ablations

This directory implements the manuscript's key design ablation analysis. The legacy root-level shell wrapper has been removed; launch the Python scripts directly.

## Inputs and configuration

`run_cosmx_component_ablation.py` runs six configurations: Full, NoCoord, NoLow, NoHigh, Raw-only and Deterministic, each with seeds 42, 123 and 2024. The combined ablation in the manuscript is implemented separately in `run_cosmx_all_three_ablation.py`. `compute_cosmx_ablation_ami.py` computes additional AMI readouts, and `audit_cosmx_all_three.py` audits the combined-ablation assets.

The base model follows the manuscript: 128 dimensions, 3 encoder layers, 2 DiT layers, 8 attention heads and 50 diffusion steps, with learning rate and weight decay both set to 1e-3. Training uses at most 50 epochs, batch size 64, patience 7 and min_delta 1e-4.

This workflow requires the frozen CosMx data audit, graphs, whitener, center manifests and Validation bootstrap manifest registered by `COSMX_RUN` in the script. By default, it also reuses verified Full/NoCoord checkpoints. `--no-reuse-existing` retrains those weights but still requires the shared input assets. Downloading the H5AD alone does not prepare every ablation input. The scripts do not automatically replace historical paths or relax input-signature checks.

## Direct execution

From the repository root in Linux/WSL, choose a new run directory:

```bash
ABLATION_RUN="outputs/runs/ablations/cosmx_nsclc/$(TZ=Asia/Shanghai date +%Y-%m-%d)/$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)"
# Inspect the plan without reading data or writing results
python experiments_analysis/ablation/run_cosmx_component_ablation.py --output-dir "$ABLATION_RUN" --phase all --plan
# Verify frozen inputs without training
python experiments_analysis/ablation/run_cosmx_component_ablation.py --output-dir "$ABLATION_RUN" --phase preflight
# Complete Train/Validation and frozen Test evaluation
python experiments_analysis/ablation/run_cosmx_component_ablation.py --output-dir "$ABLATION_RUN" --phase all --authorized
```

Add `--no-reuse-existing` to the full-run command to retrain Full/NoCoord. After reopening a terminal, set `ABLATION_RUN` to the exact previously used directory before resuming:

```bash
python experiments_analysis/ablation/run_cosmx_component_ablation.py --output-dir "$ABLATION_RUN" --phase train_validation --resume
python experiments_analysis/ablation/run_cosmx_component_ablation.py --output-dir "$ABLATION_RUN" --phase test --authorized
python experiments_analysis/ablation/run_cosmx_component_ablation.py --output-dir "$ABLATION_RUN" --phase audit
```

All results are written under `outputs/runs/ablations/cosmx_nsclc/` and remain ignored by Git. The workflow verifies provenance, weights, cell order, Validation parameters and metrics after independent reloading. Results from changed run configurations or source code are not mixed. Across-seed standard deviation on a single Test donor does not measure cross-donor uncertainty.
