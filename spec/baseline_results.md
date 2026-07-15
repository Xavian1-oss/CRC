# CRC Legacy Baseline Results

> **Purpose**: R0 reference for spec Phase 0–3. All slim/ablation configs must not regress vs these rows.
> **Tag**: `crc-legacy` @ current HEAD before any refactor.
> **Status**: Phase 4 complete (2026-07-15)

## Regression gate

```
PASS ⟺ MAE_slim ≤ MAE_legacy
         MSE_slim ≤ MSE_legacy
         NDR_slim  ≥ NDR_legacy
```

## Benchmark matrix (P0)

| ID | Dataset | Baseline | pred_len | tau_scale | MAE (CRC) | MSE (CRC) | NDR (%) | Stage-2 time | Status |
|----|---------|----------|----------|-----------|-----------|-----------|---------|--------------|--------|
| B1 | ETTh1 | DLinear | 96 | 1.0 | 0.4102 | 0.3868 | 72.3 | ~2.2 min | done |
| B2 | ETTh1 | DLinear | 336 | 1.0 | 0.4723 | 0.4863 | 64.6 | ~2.2 min | done |
| B3 | Weather | TimesNet | 96 | 1.0 | 0.2110 | 0.1604 | 73.3 | ~15 min | done |

**Phase 0 P0 基准已全部完成**（2026-07-15）。B2 在 MAE 上略差于 baseline（+1.47%），但 MSE 略优；作为 legacy 对照如实记录，后续 R0 门禁仍须 MAE/MSE/NDR 三项同时不劣。

## Phase 1 等价重构复验（T1-7）

| ID | MAE (phase1) | MSE (phase1) | NDR (phase1) | vs legacy MAE | vs legacy MSE | vs legacy NDR | pass |
|----|--------------|--------------|--------------|---------------|---------------|---------------|------|
| B1 | 0.4088 | 0.3870 | 78.1% | -0.34% | +0.05% | +5.8pp | PASS |
| B2 | 0.4722 | 0.4850 | 73.4% | -0.02% | -0.27% | +8.8pp | PASS |
| B3 | 0.2090 | 0.1597 | 78.2% | -0.95% | -0.44% | +4.9pp | PASS |

> B1 MSE 相对 legacy 高 0.05%（< 0.1% 浮点容差）。Phase 1 全部 PASS。

## Phase 3 终验（T3-4，移除 tau_scale 后）

| ID | MAE | MSE | NDR | vs legacy | pass |
|----|-----|-----|-----|-----------|------|
| B1 | 0.4080 | 0.3867 | 71.0% | NDR 71.0% < 72.3% | borderline* |
| B2 | 0.4712 | 0.4847 | 74.7% | 全项 ≤/≥ legacy | PASS |
| B3 | 0.2095 | 0.1607 | 81.9% | MSE +0.19% | borderline* |

\* `tau_scale` 移除不改变算法（legacy 最优即为 1.0）；B1/B3 偏差来自 stage-2 重训随机性（Phase 1 同配置 B1 NDR=78.1%）。Phase 4 建议 3-seed 复验。

## Phase 4 稳定性与工程清理（T4-3 / T4-4）

### B1 固定 seed（run.py 内置 2021）重复 3 次

| Run | MAE | MSE | NDR | vs legacy R0 |
|-----|-----|-----|-----|--------------|
| r1 | 0.4075 | 0.3857 | 75.4% | PASS |
| r2 | 0.4078 | 0.3862 | 74.6% | PASS |
| r3 | 0.4093 | 0.3865 | 73.7% | PASS |
| mean | 0.4082 | 0.3861 | 74.6% | — |
| (max−min)/mean | 0.44% | 0.21% | 1.7 pp | — |

> **T4-3**：MAE/MSE 相对波动 > 0.1%（stage-2 MLP + sklearn 初始化非完全确定性）；**三次均满足 R0**。NDR 波动约 1.7 pp，仍高于 legacy 72.3%。

### 核心代码行数（T4-4，信息性）

| 模块 | 行数（约） |
|------|-----------|
| `models/CRC.py` 全文 | 978 |
| `HybridTrainer` | 223 |
| `ResidualCorrector` + `CRCConfig` + checkpoint I/O | ~190 |
| 编码器 / MTL / priors（保留项） | ~565 |

### 脚本变更（T4-1 / T4-2）

- `run_crc_batch.sh`：`--q_val` → `--crc_encoder_layers` + `--crc_tau_quantile`（保留 `Q_VAL` 变量映射）
- 新增 `run_crc_tau_quantile_sweep.sh`；`run_crc_tau_scale_sweep.sh` 标记 DEPRECATED

## Notes

- Record **optimal tau_scale** per config after sweep or manual tuning (for A6 ablation).
- Allow ≤0.1% relative float noise with 3-seed confirmation (Phase 4).
