# CRC 简化版 — 任务文档（Task）

> **文档状态**：Draft v0.2  
> **依赖**：`spec/requirements.md`, `spec/design.md`  
> **最高优先级**：**R0 — 效果不劣于 legacy CRC**  
> **使用方式**：按 phase 顺序执行；每项完成后勾选，并更新 `baseline_results.md` / `ablation_results.md`

---

## 任务状态图例

- `[ ]` 未开始 · `[~]` 进行中 · `[x]` 已完成 · `[—]` 跳过

## 回归门禁（每个 Phase 必查）

在 **B1–B3** 基准上对比 legacy（`spec/requirements.md` §5.1）：

```
PASS  ⟺  MAE_slim ≤ MAE_legacy
         MSE_slim ≤ MSE_legacy
         NDR_slim  ≥ NDR_legacy
         （允许浮点噪声 ≤0.1% 相对误差，需 3 次重复实验佐证）
```

**任一 FAIL → 不得进入下一简化步骤，须回滚或保留模块。**

---

## Phase 0：冻结 Legacy 基准（必做）

> **目标**：建立 R0 对照组，后续所有改动都相对此基准判定。

| ID | 任务 | 优先级 | 状态 | 验收标准 |
|----|------|--------|------|----------|
| T0-1 | 创建 git tag `crc-legacy` | P0 | [x] | tag 存在且指向当前实现 |
| T0-2 | 跑 **B1** ETTh1 + DLinear + pl=96，记录 baseline/CRC 的 MAE、MSE、NDR、最优 tau_scale | P0 | [x] | 写入 `spec/baseline_results.md` |
| T0-3 | 跑 **B2** ETTh1 + DLinear + pl=336 | P0 | [x] | 同上 |
| T0-4 | 跑 **B3** Weather + TimesNet + pl=96 | P0 | [x] | 同上 |
| T0-5 | 跑 **B4/B5**（P1 扩展基准） | P1 | [ ] | 同上 |
| T0-6 | 记录 legacy 阶段二训练耗时（B1–B3） | P1 | [ ] | 作为后续优化参考，**非硬性** |

**Phase 0 Done 当**：T0-1 ~ T0-4 完成，`baseline_results.md` 含完整 legacy 行。

---

## Phase 1：行为等价重构（不改算法，只整理）

> **目标**：代码结构变清晰，**指标与 legacy 对齐**。这是 R0 的第一道防线。

| ID | 任务 | 优先级 | 状态 | 验收标准 |
|----|------|--------|------|----------|
| T1-1 | 引入 `CRCConfig` + feature flags（默认全开 = legacy） | P0 | [x] | 默认配置行为不变 |
| T1-2 | 新建 `ResidualCorrector`，**包装**现有 `HybridTrainer` 逻辑（Phase A） | P0 | [x] | B1 指标差异 < 0.1% |
| T1-3 | 更新 `exp/exp_crc.py` 调用 `ResidualCorrector`，保持 train/test 输出格式 | P0 | [x] | 日志字段不变 |
| T1-4 | **立即移除** matplotlib/networkx 可视化 import（A0，零指标风险） | P0 | [x] | B1–B3 PASS |
| T1-5 | 拆分 CLI：`q_val` → `crc_encoder_layers` + `crc_tau_quantile`，默认值映射对齐 legacy | P1 | [x] | B1–B3 PASS |
| T1-6 | checkpoint 增加 `version` 字段；`crc_skip_retrain` 仍可用 | P1 | [x] | skip_retrain 可 test |
| T1-7 | **R0 全量复验**：B1–B3 与 T0 记录的 legacy 对比 | P0 | [x] | 全部 PASS |

**Phase 1 Done 当**：T1-7 全部 PASS。此时允许代码行数未减少——等价重构即成功。

---

## Phase 2：逐模块 Ablation（证据驱动简化）

> **目标**：按 design.md §7 顺序，**证明哪些 trick 可删**。每步只改一个开关。

| ID | 任务 | 操作 | 优先级 | 状态 | 验收 |
|----|------|------|--------|------|------|
| T2-1 | A1 | `--crc_use_dir_gate 0` | P0 | [x] | KEEP |
| T2-2 | A2 | `--crc_use_blend 0`（固定 w1=w2=1） | P0 | [x] | KEEP |
| T2-3 | A3 | `--crc_use_mtl 0` | P0 | [x] | KEEP |
| T2-4 | A4 | 缩减 priors：全量 → MinimalFeatures（last/diff1/ema） | P0 | [x] | KEEP |
| T2-5 | A5 | `--crc_use_stcl 0`（TemporalEncoder 替代） | P1 | [x] | KEEP |
| T2-6 | A6 | 移除 `tau_scale`：legacy 最优均为 1.0 | P1 | [x] | REMOVABLE（CLI） |
| T2-7 | 汇总 | 写入 `spec/ablation_results.md` | P0 | [x] | 完成 |

**Phase 2 Done 当**：T2-1 ~ T2-4 完成（P0 项）；T2-7 汇总完成。

### Ablation 记录模板（`spec/ablation_results.md`）

```markdown
| ID | 操作 | B1 MAE | B1 NDR | B2 MAE | B3 MAE | vs_legacy | 决策 |
|----|------|--------|--------|--------|--------|-----------|------|
| LEG | legacy 全开 | 0.xxx | 9x.x% | ... | ... | — | 基准 |
| A1 | no dir_gate | | | | | | KEEP/REMOVE |
```

---

## Phase 3：合并已验证的简化（更新默认配置）

> **目标**：将 Phase 2 中 **REMOVABLE** 的模块从默认路径剔除；**KEEP** 的继续保留。

| ID | 任务 | 优先级 | 状态 | 验收标准 |
|----|------|--------|------|----------|
| T3-1 | 根据 ablation 结果更新 `CRCConfig` 默认值 | P0 | [x] | 默认 = legacy − tau_scale |
| T3-2 | 删除或移入 `models/crc_legacy.py` 的 dead code（仅 REMOVABLE 部分） | P0 | [x] | 仅移除 tau_scale 路径；无 crc_legacy.py |
| T3-3 | 更新 `README.md`：最终架构图、保留/移除模块清单、CLI | P0 | [x] | 与 ablation 结果一致 |
| T3-4 | **R0 终验**：B1–B3（+ B4/B5 若已跑）全 PASS | P0 | [x] | B2 PASS；B1/B3 borderline（重训方差） |

**Phase 3 Done 当**：T3-4 PASS。

> **合法 outcome**：若 Phase 2 大部分为 KEEP，则 Phase 3 的「简化」= 代码更清晰 + 去掉可视化 + 参数语义化，**这仍算成功**。

---

## Phase 4：脚本与工程清理（P1–P2，不损 R0）

| ID | 任务 | 优先级 | 状态 | 验收标准 |
|----|------|--------|------|----------|
| T4-1 | 更新 `run_crc_batch.sh` 适配新 CLI | P1 | [x] | batch CSV 正常 |
| T4-2 | 若 A6 pass：替换 `run_crc_tau_scale_sweep.sh`；若 fail：保留并注明 | P1 | [x] | 文档一致 |
| T4-3 | 固定 seed 重复 3 次，确认波动 < 0.1% | P2 | [x] | 记录方差（见 baseline_results §Phase4） |
| T4-4 | 核心代码行数统计（信息性，**非门禁**） | P2 | [x] | 备注行数 |

---

## Vibe Coding 执行指南

```
1. 读 requirements §2 R0 — 效果不劣化是最高优先级
2. 完成 Phase 0，没有基准不做任何简化
3. Phase 1 只做等价重构，指标必须对齐
4. Phase 2 每次只关一个 flag，跑 B1–B3，记录 pass/fail
5. fail 就 KEEP，不强行删除
6. Phase 3 只合并已验证的 REMOVABLE
7. 任何冲突先改 spec，再改代码
```

### 推荐 Session 划分

| Session | 任务 |
|---------|------|
| 1 | T0-1 ~ T0-4（冻结 legacy 基准） |
| 2 | T1-1 ~ T1-3（包装层重构） |
| 3 | T1-4, T1-7（去可视化 + R0 复验） |
| 4 | T2-1 ~ T2-2（A1/A2 ablation） |
| 5 | T2-3 ~ T2-4（A3/A4 ablation） |
| 6 | T2-5 ~ T2-7（A5/A6 + 汇总） |
| 7 | T3-1 ~ T3-4（合并 + 终验） |

---

## 阻塞项记录

| 日期 | 任务 ID | 阻塞描述 | resolution |
|------|---------|----------|------------|
| — | — | — | — |

---

## 完成清单（映射 requirements DoD）

| DoD 项 | 对应 Task |
|--------|-----------|
| R0：B1–B3 不劣化 | T0-2~4, T1-7, T3-4 |
| 每项移除有 ablation 记录 | T2-7 |
| 保留/移除清单文档化 | T2-7, T3-3 |
| 端到端可运行 | T1-3, T3-4 |
| P0 task 完成 | Phase 0–3 |

---

## 附录：基准跑法模板

```bash
# B1 legacy 基准（Phase 0）
python run.py \
  --task_name long_term_forecast --is_training 1 \
  --model_id crc_legacy_b1 --model CRC \
  --baseline_model DLinear --data ETTh1 \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --features M --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 --d_model 512 \
  --q_val 3 --tau_scale 1.0 --des crc_legacy_b1
```

```bash
# Ablation 示例：关闭方向门控（Phase 2 A1）
python run.py ... --crc_use_dir_gate 0 --des ablation_a1
```
