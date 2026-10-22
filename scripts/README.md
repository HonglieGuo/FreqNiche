# Pipeline scripts

`scripts/benchmark/` contains the dataset backends and analysis utilities used
by the public FreqNiche workflows. Launch training through `train.py` or
`reproduce.py`, and baseline experiments through `baselines/run_baselines.sh`.

## Core model workflow

| Stage | Retained scripts |
| --- | --- |
| Human Skin preparation and Test evaluation | `build_human_skin_contracts.py`, `eval_freqniche_champion.py` |
| SEA-AD preparation, training and evaluation | `build_sea_ad_cohorts.py`, `fit_sea_ad_whitener.py`, `train_sea_ad.py`, `eval_sea_ad_stability.py` |
| CosMx preparation, training and evaluation | `build_cosmx_cohorts.py`, `fit_cosmx_whitener.py`, `train_cosmx.py`, `eval_cosmx_stability.py` |
| Xenium preparation, training and evaluation | `build_xenium_audit_manifest.py`, `build_xenium_cohorts.py`, `build_xenium_graphs.py`, `fit_xenium_whitener.py`, `train_xenium.py`, `eval_xenium_stability.py`, `eval_xenium_test.py` |
| Verified downstream input export | `export_downstream_inputs.py` |

Human Skin training is implemented in `src/freqniche_st/human_skin_backend.py`.
The shared orchestrator in `src/freqniche_st/orchestration.py` selects the
appropriate backends, records manifests and logs, and collects metrics.

Both public model entry points use the dataset-specific manuscript defaults:
Human Skin/CosMx use 128 dimensions, 3 encoder layers, 2 DiT layers and a
learning rate of 1e-3 (maximum 50 epochs); SEA-AD/Xenium use 256 dimensions,
6 encoder layers, 4 DiT layers and a learning rate of 1e-4 (maximum 200 epochs).
The default seeds are 42, 123 and 2024.

## Baselines and supplementary analyses

- `run_banksy_benchmark.py`, `run_cellcharter_benchmark.py`,
  `run_cellniche_benchmark.py` and `run_stagate_benchmark.py` provide Human Skin
  baseline runs.
- `run_sea_ad_baselines.py`, `run_cosmx_baselines.py` and
  `run_xenium_baselines.py` provide dataset-specific baseline workflows.
- `run_additional_spatial_baselines.py` implements the bounded donor-wise
  SEDR/GraphST/SpaGCN comparison. `rerun_additional_spatial_assets.py` and
  [the asset guide](benchmark/ADDITIONAL_SPATIAL_ASSETS.md) document the complete
  saved-asset workflow. `run_cosmx_stagate_assets.py` provides the saved CosMx
  STAGATE assets used by supplementary analyses.
- `run_human_skin_linear_controls.py` provides Raw-PCA and DualFreq-PCA controls.
- `prepare_full_slice_manifests.py`, `run_human_skin_full_slice_baselines.py`,
  `select_full_slice_resolutions.py` and `export_full_slice_summary.py` support
  the full-slice analyses and figure inputs.
- `export_sea_ad_table5.py`, `export_cosmx_table6.py`,
  `generate_table7_synthesis.py` and `recompute_final_readouts.py` retain the
  manuscript table and frozen-readout utilities.
- `download_gmt_files.py` downloads pathway collections for expression analysis.
  The released collections are also available under `data/gmt/`.

These supplementary utilities may require existing embeddings, manifests or
result tables in addition to the processed H5AD data. Consult the
[analysis guide](../experiments_analysis/README.md) and
[baseline guide](../baselines/README.md) for scope and input requirements.
CosMx ablations are launched directly from
`experiments_analysis/ablation/run_cosmx_component_ablation.py`.

## Recovery

`resume_human_skin_run.py` and `resume_xenium_run.py` recover interrupted
Train/Validation runs using their original manifests and saved assets. Use
`--run-dir` to select a run and `--dry-run` to inspect the recovery plan.

Historical experiment-specific archives, calibration scripts and standalone
development smoke tests are omitted from the public source distribution.
Generated outputs belong under ignored run directories.
