# CosMx 关键设计消融

对应论文“关键设计消融分析”。计算实现保留在本目录，原根目录 Shell 包装入口已删除，直接使用 Python 脚本启动。

## 输入与配置

`run_cosmx_component_ablation.py` 运行 Full、NoCoord、NoLow、NoHigh、Raw-only 和 Deterministic 六组配置，均使用种子42、123、2024。论文中的联合退化配置由 `run_cosmx_all_three_ablation.py` 单独提供。`compute_cosmx_ablation_ami.py` 补算 AMI，`audit_cosmx_all_three.py` 核验联合消融资产。

基础模型与论文一致：128维、3层编码器、2层DiT、8个注意力头、50扩散步，学习率和权重衰减为1e-3；训练预算50轮，batch size 64，patience 7，min_delta 1e-4。

该消融流程依赖脚本中 `COSMX_RUN` 登记的冻结 CosMx 数据审计、图、白化器、中心清单和验证 bootstrap 清单。默认还复用 Full/NoCoord 的已核验检查点；`--no-reuse-existing` 从头训练这些权重，但仍需上述公共输入资产。仅下载 H5AD 不等于已经准备好所有消融输入。本目录不会自动替换历史路径或放宽输入签名检查。

## 直接启动

在 Linux/WSL 的项目根目录选择新的运行目录：

```bash
ABLATION_RUN="outputs/runs/ablations/cosmx_nsclc/$(TZ=Asia/Shanghai date +%Y-%m-%d)/$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)"
# 只检查计划，不读数据、不写结果
python experiments_analysis/ablation/run_cosmx_component_ablation.py --output-dir "$ABLATION_RUN" --phase all --plan
# 只核对冻结输入，不训练
python experiments_analysis/ablation/run_cosmx_component_ablation.py --output-dir "$ABLATION_RUN" --phase preflight
# 完整训练/验证及冻结 Test 评估
python experiments_analysis/ablation/run_cosmx_component_ablation.py --output-dir "$ABLATION_RUN" --phase all --authorized
```

需要重训 Full/NoCoord 时，在完整运行命令中加入 `--no-reuse-existing`。重启终端后，必须将 `ABLATION_RUN` 设置为之前实际使用的目录才能恢复：

```bash
python experiments_analysis/ablation/run_cosmx_component_ablation.py --output-dir "$ABLATION_RUN" --phase train_validation --resume
python experiments_analysis/ablation/run_cosmx_component_ablation.py --output-dir "$ABLATION_RUN" --phase test --authorized
python experiments_analysis/ablation/run_cosmx_component_ablation.py --output-dir "$ABLATION_RUN" --phase audit
```

所有结果写入 `outputs/runs/ablations/cosmx_nsclc/`，保持 Git 忽略。正常流程核对来源、权重、细胞顺序、验证参数和独立重载指标；既有运行配置或源代码变化时不混用结果。单一测试供体上的跨种子标准差不代表跨供体不确定性。
