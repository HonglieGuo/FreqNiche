# SEDR、GraphST、SpaGCN 本地重跑与资产保存

完整检索未找到三种方法原正式运行的模型权重或嵌入。旧运行只保存了指标、验证分辨率和细胞清单。新重跑使用原冻结评价中心，保存完整资产，并另行比较新旧结果。

## 直接运行

在 Windows PowerShell 执行以下命令，使用本机已有的 WSL 基线环境。默认运行 Human Skin、SEA-AD、CosMx，方法为 SEDR、GraphST、SpaGCN，种子为 42、123、2024。

~~~powershell
$HOME/.venvs/FreqNiche-spatial-baselines/bin/python scripts/benchmark/rerun_additional_spatial_assets.py --output-dir baselines/Baselines_outputs/additional_spatial_assets_20261016 --device cuda --resume
~~~

首次运行和中断后续跑使用同一命令。已完成的供体模型先检查文件校验值与输入指纹，再直接复用；未完成的供体重新训练。这里不保存优化器逐轮续训状态。

2026-10-16 已修复 CosMx 的可空字符串索引无法写入 H5AD 的问题：仅将保存副本中的字符串索引转成普通字符串对象、字符串列转成分类列，保留编号、顺序与缺失值。没有修改训练输入、模型或评价协议。兼容清单 **src/benchmark/adapters/spatial_artifact_compatibility.json** 精确登记此次修复前后的代码哈希，允许复用已完成的 Human Skin、SEA-AD 资产；原资产和阶段清单不重写。CosMx 首个供体保存失败后临时文件已清理，因此该供体需要重新训练；其余未执行任务继续运行即可，无需重跑已完成的 162 个供体包。

默认共 54 个验证/测试阶段进程，生成 180 个供体资产包：

| 数据集 | 验证供体 | 测试供体 | 三方法 × 三种子的资产包数 |
|---|---:|---:|---:|
| Human Skin | 4 | 4 | 72 |
| SEA-AD | 5 | 5 | 90 |
| CosMx | 1 | 1 | 18 |

这些方法按供体独立拟合，所以一个“方法—数据集—种子”会对应多个供体模型。验证模型与测试模型分别保存。

SEDR、GraphST 使用请求的 CUDA 设备；SpaGCN 保留官方实现的 CPU 运算方式，实际设备记录在模型文件中。

仅查看计划时，在命令末尾增加 **--dry-run**；仅检查数据和冻结细胞清单时增加 **--preflight-only**。这两个选项不会启动训练。

## 输出位置

~~~text
D:\code\research-projects\SpaPEFT\baselines\Baselines_outputs\additional_spatial_assets_20261016
~~~

目录示例：

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
            validation/<供体>/
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

每个供体包包含：

- **model.pt**：训练后的权重、模型构造参数、方法、种子和实际设备。
- **model_inputs.pt**：本次推理实际使用的特征和图张量；SEDR 额外保存实际随机掩蔽的节点。
- **embedding.npy**：实际用于该次聚类与指标计算的嵌入。
- **native_embedding.npy**：仅 GraphST 需要，保存最终 PCA 读出前的表示。
- **preprocess_pca.joblib**：实际拟合的 PCA 对象，包含 SpaGCN 内部的 PCA。
- **processed_input.h5ad**：预处理后的表达、基因顺序、缩放统计量和空间坐标，不含生态位或细胞类型监督标签。
- **cells.csv**：嵌入逐行对应的细胞条形码、供体和物理样本。
- **input_genes.csv**：预处理前的基因顺序；过滤后的顺序见 H5AD。
- **assignments.csv**：逐细胞聚类结果；测试阶段额外记录仅用于评价的参考标签。
- **artifact_manifest.json**：文件大小、SHA-256、输入指纹、软件版本、嵌入形状和聚类分辨率。
- SpaGCN 另存压缩的 **graph_sparse.npz** 或 **graph_dense.npz**。

模型包写入临时目录，保存后实际重新加载权重并重放原生嵌入，通过检查才发布为正式供体目录。指标 JSON 仅在所有供体资产和聚类结果完成后写出。

## 完成与复核

成功结束会显示：

~~~text
BACKFILL_COMPLETE ... donor_bundles=180
~~~

可另行执行完整文件校验及权重重放：

~~~powershell
$HOME/.venvs/FreqNiche-spatial-baselines/bin/python scripts/benchmark/rerun_additional_spatial_assets.py --output-dir baselines/Baselines_outputs/additional_spatial_assets_20261016 --device cuda --audit-only --replay
~~~

检查 **asset_backfill.json** 的 status 为 complete，以及 **asset_audit.json** 的 status 为 passed。运行进度和错误保存在各数据集的 logs 目录中。

**comparison.csv** 按方法和数据集列出原论文 ARI、新结果 ARI/NMI、标准差及 ARI 差值。新结果写入独立目录，不自动修改原指标记录或论文。依赖版本和数值计算差异可能使重跑结果不同，需根据对照表确认后再采用。

## 加载一个供体

在项目根目录的上述 Python 环境中：

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

不同供体独立拟合的嵌入未经跨供体坐标对齐，不应直接当作共享坐标系联合聚类。GraphST 的原读出使用 PCA fit_transform，直接读取已保存的 embedding.npy 可以保持此次结果的准确坐标。

## 已验证的检查

- 三种方法的真实作者代码，80 个合成细胞：保存与未保存路径的嵌入和随机数状态一致。
- 正式训练分支的小样本保存/重载，包括 SEDR 的 DEC 和 SpaGCN 的 Louvain 初始化。
- 实际加载权重并重放原生嵌入，检查文件损坏、权重修改、细胞顺序与输入指纹。
- 验证、测试和统一启动器的端到端流程；完成后续跑不重新训练，缺失权重或改变测试标注会阻止复用。
- 正式三个数据集的冻结供体、验证细胞顺序与测试细胞数已预检查。

完整数据重跑由用户执行；这些小样本检查不替代最终运行结束后的资产审计。
