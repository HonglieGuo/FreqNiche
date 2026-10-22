# 数据下载与使用

处理后的数据托管于 [Honglie/FreqNiche-datasets](https://huggingface.co/datasets/Honglie/FreqNiche-datasets)。GitHub 代码仓库保留配置和小型清单，H5AD 文件需要单独下载。当前发布包包含 12 个 H5AD 文件及 10 个 JSON/CSV 文件，H5AD 总大小约 2.79 GB；请另外预留运行时内存和图缓存空间。

## 下载

在 **FreqNiche 项目根目录**执行：

```bash
python -m pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Honglie/FreqNiche-datasets', repo_type='dataset', revision='98bfe081bfdb6529490bc4f49f69c3360bd516ee', local_dir='datasets/h5ad', allow_patterns=['cosmx_human_nsclc/*', 'human_skin_merfish/*', 'sea_ad/*', 'xenium_ffpe_lungbreast_gse311609/*', 'Xenium_Prime5K_Lung/*'])"
```

命令适用于 PowerShell 和 Bash。固定的完整 commit 对应首次数据上传，避免后续更新改变复现实验输入。公开且未设门槛的仓库无需登录；如下载中断，可重新执行相同命令。工具可能在目标目录生成 `.cache/huggingface/` 下载元数据，这不是实验输入。

仅下载某个数据集时，使用 Python：

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

也可以在网页 Files and versions 中逐个下载，按下面的结构放置；不要丢失子目录或改变大小写。

## 目录与实验对应关系

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
│   ├── samples/                  # 六个 GSM9509134–GSM9509139 H5AD
│   ├── dataset_manifest.json
│   ├── sample_manifest.csv
│   └── donor_splits.csv
└── Xenium_Prime5K_Lung/
    ├── xenium_prime5k_lung_roi1.h5ad
    ├── xenium_prime5k_lung_roi2.h5ad
    ├── xenium_prime5k_lung_roi3.h5ad
    └── xenium_roi_preprocess_summary.csv
```

| 配置 ID | 数据目录 | 用途 |
| --- | --- | --- |
| `human_skin` | `human_skin_merfish` | Human Skin MERFISH 主基准 |
| `sea_ad` | `sea_ad` | SEA-AD MTG MERFISH 皮层深度与消融实验 |
| `cosmx_nsclc` | `cosmx_human_nsclc` | CosMx NSCLC 跨平台与坐标消融实验 |
| `xenium_prime5k` | `xenium_ffpe_lungbreast_gse311609/samples` | 六个 NSCLC Prime 5K 样本的外部基准 |
| `xenium_prime5k` 补充 ROI | `Xenium_Prime5K_Lung` | 三个各 10,000 细胞的空间案例分析区域 |

**Xenium 清单范围：** `dataset_manifest.json` 的顶层 40 样本、27 供者和总细胞数，以及配套 CSV，记录的是较大的来源集合，不能作为本次发布包的文件数量或总量。本次正式队列由 `configs/datasets/xenium_prime5k.yaml` 的 `required_samples` 明确指定，仅含 GSM9509134–GSM9509139。清单中的其他样本不随包提供。

## 读取与复现

完成项目根目录 README 中的环境安装后，可先以只读方式检查数据：

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

这些是 AnnData/H5AD 科学数据文件，使用 `anndata.read_h5ad`，无需使用 Hugging Face `datasets.load_dataset`。不要直接将大型稀疏表达矩阵转成稠密矩阵。

保留发布的供者划分和样本边界，遵循配置与正式脚本的标签有效性掩码。SEA-AD 的表达矩阵包含来源中的非负小数值，并非全部为原始整数计数。CosMx 的标准物理坐标为 `obsm['spatial_um']`；像素坐标为 `obsm['spatial_pixel']`，具体单位约定见清单。不要对已转换的物理坐标重复缩放。

在项目根目录运行下列命令，先检查计划，再启动模型训练、Test 推理和下游分析：

```bash
python reproduce.py --dataset human_skin --dry-run
python reproduce.py --dataset human_skin --tasks neighborhood_composition_prediction niche_associated_expression
python reproduce.py --dataset sea_ad --tasks cortical_depth_prediction
```

公共入口默认按数据集选择论文配置（Human Skin/CosMx：128维、3/2层、学习率1e-3、最多50轮；SEA-AD/Xenium：256维、6/4层、学习率1e-4、最多200轮）；`--profile current` 使用可编辑的 `train.py` 参数。`--stage model` 仅执行模型训练与评估；`--stage downstream --run-dir 实际运行目录` 读取该次运行的输入清单再做下游分析。同一细胞类型内的生态位相关表达分析还需满足规定的生物学分组条件，详见[分析说明](../experiments_analysis/README.md)。

公开包不需要包含历史训练权重或结果。新入口会重建图和中心 contracts，执行 Test 后导出带有条码、维度及哈希校验的 `downstream_inputs.json`，下游结果写到本次运行的 `downstream/` 中。附加聚类敏感性及历史论文图件审计仍可能需要各自冻结的资产；它们不属于这条从头复现入口。完整数据规模的三种子结果是否一致，需要另做正式训练验证。

## 来源、引用与许可

本包是 FreqNiche 使用的处理后输入，原始数据的贡献者和权利归属不变。使用时应引用 FreqNiche 和各原始数据来源。GSE311609 的来源入口为 [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE311609)。其余来源的精确版本、原始论文和再分发许可仍需发布者补充核实；当前本地工作区未包含早期 README 提到的 `datasets/raw/*/DOWNLOAD_INFO.md`。

Hugging Face 页面目前标注的 Apache-2.0 不能单独证明第三方数据全部采用该许可；本说明不为上游数据授予新的许可。发布者应在核对各来源条款后更新数据卡元数据和逐数据集的许可说明。

下载 API 与数据卡格式参考：[Hub 下载文档](https://huggingface.co/docs/huggingface_hub/guides/download)、[Dataset Cards](https://huggingface.co/docs/hub/datasets-cards)。
