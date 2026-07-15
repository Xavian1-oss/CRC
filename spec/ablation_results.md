# CRC Ablation Results

> **Purpose**: Evidence-driven removal decisions for Phase 2–3.
> **R0 gate**: MAE/MSE ≤ legacy CRC; NDR ≥ legacy CRC (see `spec/baseline_results.md`).

## Legacy reference (Phase 0)

| ID | B1 MAE | B1 MSE | B1 NDR | B2 MAE | B2 MSE | B2 NDR | B3 MAE | B3 MSE | B3 NDR |
|----|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| LEG | 0.4102 | 0.3868 | 72.3% | 0.4723 | 0.4863 | 64.6% | 0.2110 | 0.1604 | 73.3% |

## Ablation matrix

| ID | 操作 | B1 MAE | B1 NDR | B2 MAE | B2 NDR | B3 MAE | B3 NDR | vs_legacy | 决策 |
|----|------|--------|--------|--------|--------|--------|--------|-----------|------|
| A1 | no dir_gate | 0.4084 | 72.0% | 0.4729 | 61.2% | 0.2102 | 79.8% | B1 NDR↓, B2 MAE/MSE/NDR↓, B3 MSE↑ | **KEEP** |
| A2 | no blend (w1=w2=1) | 0.4701 | 10.1% | 0.5741 | 15.2% | 0.2337 | 33.1% | 全矩阵大幅劣化 | **KEEP** |
| A3 | no MTL | 0.4076 | 78.9% | 0.4720 | 70.5% | 0.2125 | 77.8% | B1/B2/B3 MSE↑, B3 MAE↑ | **KEEP** |
| A4 | minimal priors | 0.4063 | 86.0% | 0.4667 | 81.4% | 0.2106 | 79.7% | B3 MSE 0.1615>0.1604 | **KEEP** |
| A5 | no STCL (TemporalEncoder) | 0.4094 | 69.5% | 0.4641 | 69.3% | 0.2088 | 75.0% | B1 MSE/NDR↓, B2 MSE↑ | **KEEP** |
| A6 | no tau_scale (legacy used 1.0) | — | — | — | — | — | — | legacy 最优 tau_scale=1.0，与默认一致 | **REMOVABLE*** |

## 保留 / 移除清单（Phase 3 填写）

- 多尺度物理先验全量集（A4 fail）
- STCL 图编码器（A5 fail：B1 MSE/NDR 劣化，B2 MSE 劣化）

### REMOVABLE
- **`--tau_scale` 推理旋钮**（A6）：Phase 0 全部基准最优 `tau_scale=1.0`（默认值），训练阶段 `--crc_tau_quantile` 已控制 τ；可在 Phase 3 废弃该 CLI（代码层保留 scale=1.0 固定）

### KEEP
- 方向门控 `_apply_dir_gate`（A1 fail）
- per-(node,h) blend 网格搜索（A2 fail）
- MTL 四任务预训练（A3 fail）
- 多尺度物理先验全量集（A4 fail：B3 MSE 0.1615>0.1604；B1/B2 虽更优但不满足全矩阵）
