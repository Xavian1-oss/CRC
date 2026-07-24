# CRC 审稿复验 — 需求文档（Requirements）

> **文档状态**：Draft v0.3（**全员 rebuttal + revision backlog**）  
> **来源**：`Reviews.pdf`（NeurIPS 2026 #7852）  
> **依赖**：`spec/baseline_results.md`、`spec/ablation_results.md`、`review_spec/rebuttal_results.md`  
> **实验状态**：Table R1（G0–G5 × B1–B3）**已完成**（2026-07-17）

---

## 1. 审稿人映射（OpenReview ID）

| ID | 评分 | 态度 | Rebuttal 文档 | 实验优先级 |
|----|------|------|---------------|------------|
| **gnqC** | 4 | 偏正面 | `rebuttal_gnqC.md` | P0 — Table R1 **已定稿** |
| **vaUX** | 4 (Sig 2) | 缺 baseline 叙事 | `rebuttal_vaUX.md` | P0 — **复用 Table R1**（G0/G1/G5） |
| **14dv** | **5 Accept** | 报告/统计 | `rebuttal_14dv.md` | P1 — multi-seed + PIR 检验 |
| **MkCF** | 4 | 局限/图/K | `rebuttal_MkCF.md` | P0 文字 + P1 K 敏感性 |
| **AC asfz** | Meta | 新颖性/控制/理论 | `rebuttal_AC.md` | P0 相关 work + **引用 Table R1** |

> 旧称「R1」= **gnqC**，非 OpenReview 序号。`rebuttal_r1.md` 为别名，内容同 `rebuttal_gnqC.md`。

---

## 2. 共享证据：Table R1（Gated Baseline Ladder）

**语义（回应 vaUX / AC）：**

| ID | 论文名称 | 对应 vaUX「三选一」 |
|----|----------|-------------------|
| **G0** | Frozen baseline，无 CRC | **Never update** |
| **G1** | Always-on（`--crc_use_blend 0`） | **Always accept update** |
| **G2** | Ridge + val gate（`--crc_ridge_gate_only 1`） | Simple validation-gated |
| **G5** | Full CRC | **Update when validation supports it** |

**数字**：见 `rebuttal_results.md`（G2/G4 来自 `logs/review/`；G1/G3/G5 来自 spec ablation/legacy）。

**覆盖的共识弱点（AC Meta + gnqC + vaUX）：**

- Missing always-on → **G1**
- Missing simple val-gated ridge → **G2**
- Isolating selective deployment → **G1/G2/G5 对比**

---

## 3. 按审稿人的回应目标

### 3.1 gnqC — R-REV-gnqC（P0，实验已完成）

| 关切 | 交付 |
|------|------|
| W1/Q4 Core vs stabilizers | 三层表 + G3/G4/G5 |
| W2/Q1 Val-gated baselines | Table R1 全文 |
| W3 Theory scope | Rebuttal acknowledge；revision 改 Appendix A |
| W4/Q3 Val mismatch | sensitivity 图 + recalibration + G2 B2 保守例 |
| W5/Q2 NDR vs MAE | G1/G2/G5 三角解读 |

### 3.2 vaUX — R-REV-vaUX（P0 复用 Table R1 + P1 revision）

| 关切 | Rebuttal | Revision |
|------|----------|----------|
| 缺 always-on / never | **G1 vs G0 vs G5** | Table 1 脚注或 appendix 对照 |
| Table 1 无 NDR | 引用 Table R1 NDR | **主表或 appendix 加 NDR 列** |
| 校准集如何构造 | 短答：LTF val split + `crc_val_ratio` | Methods 扩写 |
| 背景/数据集/ backbone 抽象 | 承诺扩写 | Intro + 数据集段落 |
| Significance 2 | 强调 problem + Table R1 证据 | 同上 |

### 3.3 14dv — R-REV-14dv（P1 实验）

| 关切 | 交付 |
|------|------|
| Multi-seed 方差 | **R14-E1**：主表子集或 B1–B3 G5 + G1，≥3 seeds，mean±std |
| PIR vs CRC 显著性（Table 3） | **R14-E2**：配对 bootstrap 或 Wilcoxon across datasets；revision 报告 |

Rebuttal：承诺 revision 数字 + 若截止前可给 **1 个 dataset preview**。

### 3.4 MkCF — R-REV-MkCF（P0 文字 + P1 实验）

| 关切 | Rebuttal | Revision / 实验 |
|------|----------|-----------------|
| Val–test 不匹配 | 同 gnqC W4 + G2 例 | Limitations 加粗 |
| KNN 图 K=5 | 说明 `top_k`/训练集相关图 | **RMK-E1**：K∈{3,5,7}  on B1 |
| 漂移/在线 | 讨论 scope；非本文主 claim | Limitations |
| 超参多/迁移成本 | 两阶段 + 默认全开；Table R1 简化路径 G2/G4 | README |
| Q1 recalibrate 频率 | stage-2 重训 / 新 val 比例 | `rebuttal_results.md` §Recalibration |
| Q3 NDR vs MSE | 同 gnqC W5 | — |
| Limitations 不足 | 扩写承诺 | Limitations 节 |

### 3.5 AC — R-REV-AC（P0 文档，P1 实验可选）

| ID | 关切 | 交付 |
|----|------|------|
| **AC-Q1** | TAFAS, SOLID, PETSA, FAC, FFORMA/FFORMA 新颖性 | `rebuttal_AC.md` §novelty + Related Work revision |
| **AC-Q1** | 为何不作 baseline | 设定差异（frozen backbone post-hoc vs TTA）；算力/代码；revision 讨论 |
| **AC-Q2** | Appendix A oversell | Rebuttal + revision 降调 abstract/Prop A.3–A.4 |
| Meta | Missing controls | **Table R1** |
| Meta | No seed / weak「26/28」 | 同 14dv + 诚实 TimeXer 负项 |

---

## 4. 实验矩阵（review_spec 范围）

### 4.1 已完成（P0）

| ID | 状态 | 日志 |
|----|------|------|
| G0–G5 × B1–B3 | [x] | `logs/review/g2_*`, `g4_*` + spec |

### 4.2 Rebuttal 期建议补（P1）

| ID | 任务 | 回应 |
|----|------|------|
| **R14-E1** | G5 (+可选 G1) × B1–B3，seeds `{2021,2022,2023}` 或 run.py 可配 | 14dv, AC |
| **R14-E2** | Table 3 PIR vs CRC 统计脚本 | 14dv |
| **RMK-E1** | B1 `top_k` sweep 3/5/7 | MkCF Q2 |

### 4.3 Revision 期（P2）

| ID | 任务 |
|----|------|
| **REV-T1** | 主表全 dataset × G0/G1/G5 或 appendix |
| **REV-T2** | Table 1 + NDR 列 |
| **REV-T3** | NDR–ΔMAE 图（G0–G5） |
| **REV-T4** | Blend zero_rate 统计 |
| **REV-T5** | Related work：TAFAS, SOLID, PETSA, FAC, FFORMA |
| **REV-T6** | Appendix A 措辞；Methods 校准集 |

---

## 5. 非目标

- 复现全部 TTA 方法作 baseline（revision 以讨论为主，可选 1–2 个若时间允许）
- 在线/漂移新 benchmark（MkCF 仅讨论）
- 再次执行 `spec/` 简化 R0

---

## 6. Definition of Done

### Rebuttal 提交（全员）

- [x] Table R1 完成
- [x] `rebuttal_gnqC.md` 定稿
- [ ] `rebuttal_vaUX.md` 定稿（引用 G0/G1/G5）
- [ ] `rebuttal_14dv.md` 定稿（承诺 + 可选 preview）
- [ ] `rebuttal_MkCF.md` 定稿
- [ ] `rebuttal_AC.md` 定稿（AC-Q1/Q2 + Meta）
- [ ] OpenReview 五条回复粘贴完成

### Revision（会后）

- [ ] R14-E1/E2、RMK-E1 完成并写入 `rebuttal_results.md` §Seeds
- [ ] REV-T1–T6 勾选

---

## 7. 基准 B1–B3（不变）

| ID | Dataset | Baseline | pred_len |
|----|---------|----------|----------|
| B1 | ETTh1 | DLinear | 96 |
| B2 | ETTh1 | DLinear | 336 |
| B3 | Weather | TimesNet | 96 |

---

## 8. 审稿条目 → 文档/实验速查

| 来源 | 文档 | 实验 |
|------|------|------|
| gnqC W1–W5, Q1–Q4 | rebuttal_gnqC | Table R1 |
| vaUX major Q | rebuttal_vaUX | Table R1 (G0/G1/G5) |
| 14dv Q | rebuttal_14dv | R14-E1, R14-E2 |
| MkCF W/Q | rebuttal_MkCF | RMK-E1 (P1) |
| AC-Q1, Meta | rebuttal_AC | 文案 + Table R1 |
| AC-Q2 | rebuttal_AC, gnqC W3 | 文案 |
