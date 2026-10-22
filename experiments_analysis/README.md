# 论文实验与下游分析

本目录按 `Paper/FreqNiche-zh/main.tex` 实际引用的正文和补充材料整理。任务名称使用论文中的科学含义，不再使用 A/B/C 字母编号。

## 任务名称与启动方式

| 论文中的名称 | 命令参数 `--tasks` | 代码目录 | 数据集 |
| --- | --- | --- | --- |
| 邻域细胞组成预测 | `neighborhood_composition_prediction` | `neighborhood_composition_prediction/` | Human Skin |
| 皮层深度预测（连续解剖梯度预测） | `cortical_depth_prediction` | `cortical_depth_prediction/` | SEA-AD |
| 同一细胞类型内的生态位相关表达分析 | `niche_associated_expression` | `niche_associated_expression/` | Human Skin Fibro |

论文中的跨供体空间域识别（SEA-AD）和跨供体细胞生态位识别（Human Skin/CosMx），已包含在模型 Test 评估中，由 `scripts/benchmark/` 的评估脚本完成；不需要另选一个预测任务。Xenium 无标签评估也属于模型评估流程。

在项目根目录安装环境、按 [数据说明](../datasets/README.md) 下载数据后运行：

```bash
# 训练、Test 推理、邻域细胞组成预测及生态位相关表达分析
python reproduce.py --dataset human_skin --tasks neighborhood_composition_prediction niche_associated_expression
# 训练、Test 推理、皮层深度预测
python reproduce.py --dataset sea_ad --tasks cortical_depth_prediction
# 只检查命令计划，不训练、不写结果
python reproduce.py --dataset human_skin --tasks neighborhood_composition_prediction --dry-run
```

单独分析已完成的运行时，使用程序打印的真实目录替换 `DATE/RUN_ID`：

```bash
python reproduce.py --stage downstream --run-dir outputs/runs/freqniche/human_skin/DATE/RUN_ID --tasks neighborhood_composition_prediction niche_associated_expression
python reproduce.py --stage downstream --run-dir outputs/runs/freqniche/sea_ad/DATE/RUN_ID --tasks cortical_depth_prediction
```

## 独立脚本与结果

每个目录的启动脚本为 `run_<任务名称>.py`，也可以直接运行：

```bash
python experiments_analysis/neighborhood_composition_prediction/run_neighborhood_composition_prediction.py --run-dir RUN
python experiments_analysis/cortical_depth_prediction/run_cortical_depth_prediction.py --run-dir RUN
python experiments_analysis/niche_associated_expression/run_niche_associated_expression.py --run-dir RUN
```

输入来自本次训练导出的 `downstream_inputs.json`，包含 H5AD、模型配置、条码和嵌入校验。新结果分别写入本次运行的 `downstream/neighborhood_composition_prediction/`、`downstream/cortical_depth_prediction/` 和 `downstream/niche_associated_expression/`，全部由 Git 忽略。

两项预测任务保留按验证供体分组的交叉验证，标准化和 PCA 仅在相应训练折拟合；验证与测试供体不得重叠。正常计算拒绝覆盖已有结果，`--plot-only` 仅按同一运行的结果重绘。

## 生态位相关表达分析

本分析对应正文“同一细胞类型内的生态位相关表达分析”。保持 Fibro 细胞类型、Seed 42、Leiden 参数、原始计数检查、GMT 哈希、供体级表达效应、pseudobulk 变化及通路重叠计算；数值计算集中于 `niche_associated_expression/expression_analysis.py`。

比较 STROMA-enriched 与 PERIVASC-enriched 两组；默认簇编号为 2 和 0。新训练可能置换编号，必须根据程序打印的生态位比例确认生物学含义，并通过 `--expression-stroma-cluster`、`--expression-perivasc-cluster` 显式指定。直接脚本对应 `--stroma-cluster`、`--perivasc-cluster`。两组必须分别满足相应生态位的多数富集，至少 3 个供体在每组各有 20 个 Fibro 细胞；不满足则停止，不能为获得有利效应而调整参数。

[分析协议](protocols/niche_associated_expression_protocol.md) 记录论文原始比较的统计口径。有效样本量是供体数，分析属于描述性事后分析；新运行记录实际簇号、细胞数、供体集合和嵌入维度，不套用历史数值。

## 正文和补充材料使用的其他代码

| 内容 | 保留代码 | 论文对应部分 |
| --- | --- | --- |
| 关键设计消融 | `ablation/` | 正文“关键设计消融分析” |
| 聚类算法敏感性与参考类别数诊断 | `clusterer_sensitivity/` | 补充材料“聚类算法敏感性实验” |
| Xenium 冻结推理时间与显存测量 | `benchmark_xenium_frozen_inference.py` | 补充材料“大规模数据下的无标签评估与可扩展性” |
| 线性对照核验 | `audit_human_skin_linear_controls.py` | Raw PCA、DualFreq-PCA 对照 |
| 全切片背景和推理资产 | `infer_full_slice/` | 正文物理切片图的上游资产 |
| 论文图件及共享样式 | `plot_figures/`、`figure_style.py` | 正文及补充图件 |
| 论文表格维护 | `prepare_014_paper_inputs.py`、`format_paper_table_ranks.py` | 论文表格的输入整理与格式 |

附加核验和论文图件脚本仍可能需要对应的既有基线嵌入、选择清单、结果表或 R/mclust；这些资产不随代码发布，也不会由三个预测/表达任务自动生成。

已清理未进入论文实验的转导自适应执行流程、未实施的图/表达扰动提案、旧重复图件入口、额外开发测试和历史固定资产专用的重复表达分析启动器。仍被论文图件引用的共享函数及表达效应计算已提取保留。

本轮只整理入口和论文使用范围，不改写论文文件。命令计划和合成数据检查不等于完整数据的三种子正式复跑。
