# Data download and usage

Processed inputs are hosted at [Honglie/FreqNiche-datasets](https://huggingface.co/datasets/Honglie/FreqNiche-datasets). The GitHub repository provides the code, configurations and this guide; download H5AD files and their accompanying manifests and split tables separately. The release contains 12 H5AD files and 10 JSON/CSV files. The H5AD files total approximately 2.79 GB; allow additional memory and disk space for processing and graph caches.

## Download

Run the following commands from the **FreqNiche repository root**:

```bash
python -m pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Honglie/FreqNiche-datasets', repo_type='dataset', revision='98bfe081bfdb6529490bc4f49f69c3360bd516ee', local_dir='datasets/h5ad', allow_patterns=['cosmx_human_nsclc/*', 'human_skin_merfish/*', 'sea_ad/*', 'xenium_ffpe_lungbreast_gse311609/*', 'Xenium_Prime5K_Lung/*'])"
```

These commands work in PowerShell and Bash. The full commit hash pins the first data upload, preventing subsequent updates from changing the reproduction inputs. Public, ungated repositories can be downloaded without signing in. If a download is interrupted, rerun the same command. The downloader may create `.cache/huggingface/` metadata inside the destination; this metadata is not an experimental input.

To download a single dataset, use Python:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Honglie/FreqNiche-datasets",
    repo_type="dataset",
    revision="98bfe081bfdb6529490bc4f49f69c3360bd516ee",
    local_dir="datasets/h5ad",
    allow_patterns=["human_skin_merfish/*"],
)
```

Alternatively, download files individually from the web interface's **Files and versions** tab and place them in the structure below. Preserve all subdirectories and filename capitalization.

## File layout and experiment mapping

```text
datasets/h5ad/
├── human_skin_merfish/
│   ├── human_skin_merfish.h5ad
│   ├── dataset_manifest.json
│   └── donor_splits.csv
├── sea_ad/
│   ├── sea_ad_mtg_merfish.h5ad
│   ├── dataset_manifest.json
│   └── donor_splits.csv
├── cosmx_human_nsclc/
│   ├── cosmx_human_nsclc.h5ad
│   ├── dataset_manifest.json
│   └── donor_splits.csv
├── xenium_ffpe_lungbreast_gse311609/
│   ├── samples/                  # Six H5AD files: GSM9509134-GSM9509139
│   ├── dataset_manifest.json
│   ├── sample_manifest.csv
│   └── donor_splits.csv
└── Xenium_Prime5K_Lung/
    ├── xenium_prime5k_lung_roi1.h5ad
    ├── xenium_prime5k_lung_roi2.h5ad
    ├── xenium_prime5k_lung_roi3.h5ad
    └── xenium_roi_preprocess_summary.csv
```

| Configuration ID | Data directory | Purpose |
| --- | --- | --- |
| `human_skin` | `human_skin_merfish` | Main Human Skin MERFISH benchmark |
| `sea_ad` | `sea_ad` | SEA-AD MTG MERFISH spatial-domain identification and cortical-depth analysis |
| `cosmx_nsclc` | `cosmx_human_nsclc` | CosMx NSCLC cross-platform benchmark and coordinate ablation |
| `xenium_prime5k` | `xenium_ffpe_lungbreast_gse311609/samples` | External benchmark using six NSCLC Prime 5K samples |
| `xenium_prime5k` supplementary ROIs | `Xenium_Prime5K_Lung` | Three spatial case-study regions, each containing 10,000 cells |

**Scope of the Xenium manifests:** the top-level totals of 40 samples, 27 donors and cell counts in `dataset_manifest.json`, together with the accompanying CSV files, describe a larger source collection. They are not the file count or total size of this release. The formal cohort is specified by `required_samples` in `configs/datasets/xenium_prime5k.yaml` and contains only GSM9509134-GSM9509139. Other samples listed in the manifests are not included.

## Reading the data and reproducing experiments

After completing the environment setup in the root README, inspect a dataset in read-only mode:

```python
from pathlib import Path
import anndata as ad

path = Path("datasets/h5ad/human_skin_merfish/human_skin_merfish.h5ad")
adata = ad.read_h5ad(path, backed="r")
try:
    print("cells × genes:", adata.shape)
    print("obs:", adata.obs.columns.tolist())
    print("obsm:", list(adata.obsm.keys()))
finally:
    adata.file.close()
```

These are AnnData/H5AD scientific data files. Read them with `anndata.read_h5ad`; Hugging Face `datasets.load_dataset` is not required. Avoid converting large sparse expression matrices directly to dense arrays.

Preserve the released donor splits and sample boundaries, and apply the valid-label masks defined in the configurations and evaluation scripts. SEA-AD expression includes nonnegative fractional values from the source and is not entirely raw integer counts. CosMx uses `obsm['spatial_um']` for standard physical coordinates and `obsm['spatial_pixel']` for pixel coordinates; see its manifest for unit conventions. Do not rescale physical coordinates that have already been converted.

From the repository root, inspect the command plan before starting training, Test inference and downstream analyses:

```bash
python reproduce.py --dataset human_skin --dry-run
python reproduce.py --dataset human_skin --tasks neighborhood_composition_prediction niche_associated_expression
python reproduce.py --dataset sea_ad --tasks cortical_depth_prediction
```

The public entry point selects the dataset-specific manuscript configuration by default: Human Skin/CosMx use 128 dimensions, 3 encoder layers, 2 DiT layers, a learning rate of 1e-3 and at most 50 epochs; SEA-AD/Xenium use 256 dimensions, 6 encoder layers, 4 DiT layers, a learning rate of 1e-4 and at most 200 epochs. `--profile current` uses the editable parameters in `train.py`. `--stage model` runs model training and evaluation only. `--stage downstream --run-dir RUN_DIRECTORY` reads the selected run's input manifest and performs downstream analysis. Within-cell-type niche-associated expression analysis also requires the specified biological grouping conditions; see the [analysis guide](../experiments_analysis/README.md).

Historical checkpoints and results are not required in the public distribution for the main reproduction workflow. The entry point rebuilds graphs and center contracts, then exports `downstream_inputs.json` after Test inference, with barcode, dimension and hash checks. Downstream results are written to the selected run's `downstream/` directory. Supplementary clusterer sensitivity analyses and historical manuscript figure audits may require their own frozen assets and are separate from this workflow. Agreement of full-scale three-seed results requires a separate formal training rerun.

## Sources, citation and licensing

This release contains the processed inputs used by FreqNiche. Original data contributors retain their attribution and rights. Cite FreqNiche and the original data sources when using these inputs. The source entry for GSE311609 is [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE311609). Exact source versions, original publications and redistribution permissions for the other datasets still require verification and documentation by the publisher. The current workspace does not include the `datasets/raw/*/DOWNLOAD_INFO.md` files mentioned in an earlier README.

The Apache-2.0 label currently shown on Hugging Face does not by itself establish that all third-party datasets use that license. This guide grants no new rights to upstream data. The publisher should verify source terms and update the dataset-card metadata and per-dataset licensing information accordingly. The repository's MIT license applies to original FreqNiche code and does not override data-source terms.

Download API and dataset-card references: [Hub download guide](https://huggingface.co/docs/huggingface_hub/guides/download) and [Dataset Cards](https://huggingface.co/docs/hub/datasets-cards).
