# CRC 简化版 — 需求文档（Requirements）

> **文档状态**：Draft v0.2  
> **创建目的**：为后续 vibe coding（spec → design → task → 实现）提供需求基线  
> **范围**：仅定义“简化 CRC”应做什么；**本阶段不涉及代码改动**  
> **最高优先级约束**：**简化不得以效果变差为代价**（见 §2 R0、§5 NFR-04）

---

## 1. 背景与动机

当前 CRC（Causal Residual Corrector）实现位于 `models/CRC.py` 与 `exp/exp_crc.py`，采用 **baseline → MTL 预训练 → Hybrid 校正** 的两阶段流程。功能上能提升基线预测并报告 NDR，但实现层叠过多启发式模块，导致：

- 超参数语义重叠（如 `q_val` 同时控制 STCL 层数与安全边界分位数）
- 验证集被多次复用（τ 估计、MLP early stopping、blend 网格搜索）
- 推理阶段仍需 `tau_scale` 事后调节，说明安全边界未在训练阶段稳定定稿
- 代码体量大（~675 行 `CRC.py`），难以 ablation 与维护

**本 spec 的目标**：在 **效果不劣于当前 CRC** 的前提下，删除或合并非必要 trick，使方法 **更简洁、可解释、可 ablation、可复现**。

> **重要**：简化是手段，不是目的。任何模块只有在 **ablation 证明移除后不降级** 时才能删除；若某 trick 对效果有贡献，应保留其行为（可重构实现，但不可牺牲指标）。

---

## 2. 核心价值主张（必须保留）

### R0 — 非劣化约束（最高优先级）

简化版 CRC 在 **标准基准矩阵**（见 §5 NFR-04）上，相对当前 legacy CRC（`crc-legacy` tag）必须同时满足：

| 指标 | 约束 |
|------|------|
| **MAE**（CRC 纠偏后） | ≤ legacy CRC MAE + `ε_mae`（`ε_mae = 0`，不允许变差） |
| **MSE**（CRC 纠偏后） | ≤ legacy CRC MSE + `ε_mse`（`ε_mse = 0`） |
| **NDR** | ≥ legacy CRC NDR − `ε_ndr`（`ε_ndr = 0` pp） |

允许的唯一例外：浮点/随机性导致的 **≤0.1% 相对波动**，须在 `spec/baseline_results.md` 中记录 3 次重复实验标准差后判定。

**若某次简化导致任一基准配置不达标 → 该简化不得合并**，必须回滚或保留原模块。

---

简化后的 CRC 仍应满足以下产品级/研究级承诺：

| ID | 承诺 | 说明 |
|----|------|------|
| R1 | **即插即用** | 任意已注册 baseline（DLinear、PatchTST、TimesNet 等）之上叠加校正器，不微调 baseline |
| R2 | **残差修正** | 学习 `r = y − ŷ_baseline`，最终预测 `ŷ_final = ŷ_baseline + r_corrected` |
| R3 | **安全修正** | 非线性修正必须有显式幅度上界（τ），避免过度修正 |
| R4 | **可度量安全性** | 测试阶段继续报告 **NDR**（非恶化率）及 MAE/MSE 相对基线的变化 |
| R5 | **与现有入口兼容** | `python run.py --model CRC --baseline_model <X>` 仍可运行；批量脚本可渐进迁移 |
| R6 | **效果优先于简洁** | 代码行数、训练时长等优化目标，均排在 R0 之后 |

---

## 3. 功能需求

### 3.1 训练流程

| ID | 需求 | 优先级 |
|----|------|--------|
| FR-01 | **阶段一**不变：训练或 warm-start 加载 baseline | P0 |
| FR-02 | **阶段二**简化为单阶段校正器训练：输入 `(W, ŷ, y)`，输出残差修正 | P0 |
| FR-03 | 校正器采用 **双路结构**：Ridge 线性残差（保底）+ 小型 MLP 非线性残差（增广） | P0 |
| FR-04 | 非线性残差在训练与推理时均经 **τ 限幅**：`clip(r_nl, −τ, +τ)` | P0 |
| FR-05 | τ 在**单一验证流程**中估计：基于线性修正后的验证残差分位数，**不得**再叠加方向门控、blend 网格搜索等二次验证调参 | P0 |
| FR-06 | 特征编码器 **优先精简**；STCL / 图编码 **仅当 ablation 证明可移除时才降级为可选** | P1 |
| FR-07 | 物理先验 **优先缩减**；完全移除须通过 ablation，若全量先验有贡献则保留或找等价精简子集 | P1 |
| FR-08 | MTL 四任务 **候选移除**；移除前须证明单任务残差预训练或端到端训练可达 R0 | P0 |
| FR-09 | 融合方式 **候选简化**为 `ŷ + r_lin + r_nl_clipped`；移除 blend 网格搜索前须证明 R0 成立 | P0 |
| FR-10 | checkpoint 格式与 `--crc_skip_retrain` 语义保持；旧 checkpoint **不保证**向后兼容（需在 README/spec 中声明） | P1 |

### 3.2 推理与评估

| ID | 需求 | 优先级 |
|----|------|--------|
| FR-11 | 测试流程输出：baseline MAE/MSE、CRC MAE/MSE、NDR、相对提升率 | P0 |
| FR-12 | 保留 **Correction coverage**（`|r_nl| > τ` 被 clip 的比例）作为机制可解释性指标 | P1 |
| FR-13 | **`--tau_scale` 候选移除**；仅当找到训练阶段等价方案且满足 R0 时才废弃（见 design.md §4.4） | P1 |
| FR-14 | 可选保留 `--use_dtw`、 `--inverse` 等与 LTF 对齐的评估选项 | P2 |

### 3.3 CLI 与配置

| ID | 需求 | 优先级 |
|----|------|--------|
| FR-15 | 废弃 `--q_val` 的双重语义；拆分为独立参数（见 design.md） | P0 |
| FR-16 | 新参数命名清晰、一参一意，文档化默认值 | P0 |
| FR-17 | 保留 `--warm_start`、`--crc_val_ratio`、`--crc_ckpt_dir`、`--crc_skip_retrain` | P1 |

---

## 4. 候选简化项（须 ablation 门禁通过才可移除）

以下模块是 **简化候选**，不是「默认删除」。每一项须先跑 **移除实验**，在 §5 基准矩阵上 **全部满足 R0** 后，才可从默认路径剔除。

| 候选移除项 | 原位置 | 简化动机 | 移除条件 |
|------------|--------|----------|----------|
| MTL 四任务 + uncertainty weighting | `PretrainableEncoderMTL`, `mtl_pretrain_trainer` | 复杂、难维护 | 单任务/无预训练替代方案在基准矩阵上 MAE/MSE/NDR 均不劣 |
| 多尺度物理先验（ACF/FFT/IQR/…） | `build_physics_priors_ms` | 特征工程味浓 | 移除后全矩阵不劣；或找到更小但等价的先验子集 |
| 方向门控 `_apply_dir_gate` | `HybridTrainer` | 硬编码规则 | 关闭后 NDR 不下降 |
| per-(node, horizon) blend 网格搜索 | `HybridTrainer.train` step 6 | 验证集过拟合风险 | 固定融合 `w1=w2=1` 后 MAE/MSE/NDR 不劣 |
| `improve_eps` 拒绝融合 | `HybridTrainer` | 随 blend 简化 | 同上 |
| `q_val → quantile_q` 离散映射 | `exp_crc.py` | 语义混乱 | 拆分参数后，用等价映射复现 legacy 指标 |
| `--tau_scale` 推理缩放 | `HybridTrainer.predict` | 事后 knob | 训练阶段 `--crc_tau_quantile` 调参可复现最优 legacy 指标 |
| `top_m` 残差相关特征选择 | `build_physics_priors_ms` | 非核心分支 | 移除后全矩阵不劣 |
| 训练/测试内嵌可视化 | `models/CRC.py` | 非模型逻辑 | **可立即移除**（不影响预测指标） |
| mask_ratio 重建预训练 | `PretrainableEncoderMTL.forward` | 随 MTL 简化 | 随 MTL ablation 一并判定 |

**判定为「暂时不可移除」时**：在 `spec/ablation_results.md` 记录原因，代码中保留该模块（允许重构实现，但行为等价）。

**最终默认配置 = legacy 行为 − 已通过 ablation 的模块集合**（而非预设的「最小架构」）。

---

## 5. 非功能需求

### 5.1 标准基准矩阵（R0 验收必跑）

| # | 数据集 | Baseline | pred_len | 优先级 |
|---|--------|----------|----------|--------|
| B1 | ETTh1 | DLinear | 96 | P0 |
| B2 | ETTh1 | DLinear | 336 | P0 |
| B3 | Weather | TimesNet | 96 | P0 |
| B4 | Weather | TimesNet | 336 | P1 |
| B5 | Electricity | DLinear | 96 | P1 |

**R0 判定**：P0 的 B1–B3 **全部**满足非劣化，才允许合并简化改动；P1 基准作为扩展验证。

每次 ablation / 重构后，在 `spec/baseline_results.md` 追加对比行：

```
| config | MAE | MSE | NDR | vs_legacy_MAE | vs_legacy_NDR | pass |
```

### 5.2 其他非功能需求

| ID | 需求 | 验收标准 | 优先级 |
|----|------|----------|--------|
| NFR-01 | **可读性** | 通过 ablation 后，核心逻辑尽量 ≤ 350 行；**若与 R0 冲突，允许更多行数** | P2 |
| NFR-02 | **可 ablation** | 每个模块可通过 CLI/feature flag 独立开关 | P0 |
| NFR-03 | **可复现** | 固定 seed 下，重复 3 次 NDR/MAE 相对波动 < 0.1% | P0 |
| NFR-04 | **效果不劣化（R0）** | 见 §2 R0 与 §5.1 基准矩阵 | **P0** |
| NFR-05 | **训练时长** | 在不违反 R0 前提下，阶段二耗时尽量缩短；**非硬性指标** | P2 |
| NFR-06 | **文档** | 每个保留/移除决策有 ablation 记录 | P0 |

---

## 6. 约束条件

1. **不修改** Time-Series-Library 其他任务路径（imputation、anomaly 等）
2. **不修改** PIR 相关代码（`run_pir.py`, `models/PIR.py`）；NDR 对比脚本可后续适配
3. 第一阶段实现 **不强制** 改 batch shell 脚本；脚本迁移列入 task.md 低优先级
4. 遵循仓库现有代码风格（PyTorch、numpy、现有 `Exp_*` 继承模式）

---

## 7. 成功标准（Definition of Done）

简化 CRC 项目视为需求达成，当且仅当：

- [ ] **R0 通过**：基准矩阵 B1–B3（至少）上 MAE/MSE/NDR 均不劣于 legacy
- [ ] 每一项已移除模块在 `spec/ablation_results.md` 中有 **pass** 记录
- [ ] 默认配置下，端到端 `run.py --model CRC` 可完成 train + test
- [ ] CLI 参数一参一意（已拆分 `q_val` 等双重语义参数）
- [ ] 测试日志包含 MAE/MSE/NDR/Correction coverage
- [ ] `spec/ablation_results.md` 含「保留模块清单」及「移除模块清单」及理由
- [ ] spec/task.md 中 P0 任务全部勾选完成

**注意**：若经完整 ablation 后，legacy 大部分模块仍需保留，则「简化」体现为 **代码重构 + 参数语义清晰 + 去除纯工程噪音（可视化等）**——这仍是合法的成功 outcome。

---

## 8. 利益相关方假设

| 角色 | 关注点 |
|------|--------|
| 研究者/作者 | 方法故事简洁、ablation 清晰、NDR 可辩护 |
| 开发者（future vibe coding） | spec 足够具体，可按 task 逐步执行而不歧义 |
| 实验者 | batch 脚本 eventually 可用，warm_start 仍省事 |

---

## 9. 开放问题（待 ablation 数据决议）

| # | 问题 | 决议方式 |
|---|------|----------|
| OQ-1 | STCL 能否替换为轻量 TemporalEncoder？ | **ablation 决定**；若 NDR/MAE 变差则保留 STCL |
| OQ-2 | blend 网格搜索能否改为固定权重？ | **ablation 决定**；变差则保留 blend 或找解析近似 |
| OQ-3 | MTL 四任务能否降为单任务？ | **ablation 决定** |
| OQ-4 | `tau_scale` 能否完全去掉？ | 在训练时用 `crc_tau_quantile` 复现 legacy 最优 τ 效果；做不到则保留 |
| OQ-5 | 物理先验最小子集是哪些？ | 逐特征 ablation，保留有贡献的特征 |
| OQ-6 | Ridge per-node 是否必须？ | 默认保留（低风险）；共享权重需 ablation |

---

## 10. 参考文件

- 当前实现：`models/CRC.py`, `exp/exp_crc.py`, `run.py`
- 评估指标：`utils/metrics.py`, `exp/exp_crc.py::test`
- 批量实验：`run_crc_batch.sh`, `run_crc_tau_scale_sweep.sh`
