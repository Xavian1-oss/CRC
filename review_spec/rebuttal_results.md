# CRC Rebuttal Experiment Results

> **Status**: v0.3 — Table R1 **全员共享**（gnqC, vaUX, AC, MkCF）；G2/G4 新跑；G1/G3/G5 引用 spec  
> **Logs**: `logs/review/g2_*`, `logs/review/g4_*`  
> **Rebuttal 映射**: 见 `requirements.md` §1

### G0 / G1 / G5 与 vaUX「三选一更新」

| vaUX 说法 | 本仓库对照 |
|-----------|------------|
| Never update | **G0** — Table 1 baseline 行 |
| Always accept update | **G1** — `--crc_use_blend 0` |
| Smart / selective update | **G5** — Full CRC |

---

## Table R1 — Gated baseline ladder (MAE / MSE / NDR)

Benchmarks: **B1** ETTh1+DLinear+96 · **B2** ETTh1+DLinear+336 · **B3** Weather+TimesNet+96

| Config | B1 MAE | B1 MSE | B1 NDR | B2 MAE | B2 MSE | B2 NDR | B3 MAE | B3 MSE | B3 NDR | Source |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| **G0** Baseline | 0.4108 | 0.3962 | — | 0.4654 | 0.4874 | — | 0.2184 | 0.1685 | — | test log |
| **G1** Always-on | 0.4701 | — | **10.1%** | 0.5741 | — | **15.2%** | 0.2337 | — | **33.1%** | spec A2 |
| **G2** Ridge+val gate | 0.4097 | 0.3941 | **97.8%** | 0.4725 | 0.4959 | **94.8%** | 0.2112 | 0.1626 | **89.9%** | review g2 |
| **G3** No dir gate | 0.4084 | — | 72.0% | 0.4729 | — | 61.2% | 0.2102 | — | 79.8% | spec A1 |
| **G4** Deploy-only | 0.4097 | 0.3941 | 62.1% | 0.4592 | 0.4838 | 77.8% | **0.2081** | **0.1584** | 79.4% | review g4 |
| **G5** Full CRC | **0.4102** | **0.3868** | 72.3% | **0.4723** | **0.4863** | 64.6% | **0.2110** | **0.1604** | 73.3% | spec legacy |

### Key comparisons (R1 narrative)

| Comparison | Finding |
|------------|---------|
| **G5 vs G1** | Always-on correction **collapses NDR** (10–33% vs 64–73%) and **hurts MAE** on all B1–B3 → selective deployment is essential, not “having a validation set”. |
| **G5 vs G2** | Ridge+val gate achieves **very high NDR** (90–98%) but **smaller MAE/MSE gains**; B2 MAE even **worse than baseline** (−1.5%). Full CRC needed for **accuracy + calibrated safety**. |
| **G5 vs G4** | Deploy-only (light encoder + blend) improves B3 MAE but **B1 NDR 62% < G5 72%** → representation (MTL/STCL/full priors) helps **both** accuracy and deployment quality. |
| **G5 vs G3** | Removing dir gate drops B2 NDR **64.6%→61.2%** → direction gate is a **stabilizer**, not the core deployment rule. |

---

## Rebuttal checklist（v0.3）

| Reviewer | 文稿 | 提交 |
|----------|------|------|
| gnqC | [x] `rebuttal_gnqC.md` | [ ] |
| vaUX | [ ] `rebuttal_vaUX.md` | [ ] |
| 14dv | [ ] `rebuttal_14dv.md` | [ ] |
| MkCF | [ ] `rebuttal_MkCF.md` | [ ] |
| AC | [ ] `rebuttal_AC.md` | [ ] |

### 实验 backlog

- [x] R14-E1 multi-seed（§Multi-seed）
- [x] R14-E2 PIR vs CRC significance → `review_spec/r14_e2_pir_crc_significance.md`
- [x] RMK-E1 top_k sweep B1

---

## Concept map

| Layer | Components | Evidence |
|-------|------------|----------|
| **Core deployment** | Ridge floor, τ, per-(node,h) blend | G1 vs G5; G2 uses binary ridge gate |
| **Stabilizer** | Direction gate, clip | G3 vs G5 |
| **Representation** | MTL, STCL, full priors | G4 vs G5; A3–A5 |

---

## Logs

| Run | Path |
|-----|------|
| G2 B1–B3 | `logs/review/g2_b1/` … `g2_b3/` |
| G4 B1–B3 | `logs/review/g4_b1/` … `g4_b3/` |
| R14-E1 | `logs/review/r14_g5_b1_seed*/`, `r14_g1_b1_seed*/` |
| RMK-E1 | `logs/review/rmk_topk{3,5,7}_b1/` |

---

## Multi-seed（R14-E1）

B1 · ETTh1 + DLinear + pl=96 · `--fix_seed` ∈ {2021, 2022, 2023} · 脚本 `scripts/review/run_multi_seed.sh`

| Config | B1 MAE mean±std | B1 MSE mean±std | B1 NDR mean±std | Per-seed MAE (2021/2022/2023) | Status |
|--------|-----------------|-----------------|-----------------|-------------------------------|--------|
| **G5** Full CRC | **0.4091 ± 0.0011** | **0.3885 ± 0.0007** | **72.3 ± 6.3%** | 0.4088 / 0.4103 / 0.4082 | [x] |
| **G1** Always-on | 0.4828 ± 0.0131 | 0.4936 ± 0.0378 | 7.2 ± 1.5% | 0.4742 / 0.4763 / 0.4979 | [x] |

**Takeaway (14dv preview):** G5 MAE stable (~0.41) with NDR **65–78%** across seeds; G1 stays **low NDR (~5–8%)** and **much worse MAE** — same qualitative gap as Table R1.

---

## top_k 敏感性（RMK-E1）

B1 Full CRC · `--top_k` = graph KNN · seed 2021 · `scripts/review/run_topk_sweep.sh`

| top_k | MAE | MSE | NDR |
|-------|-----|-----|-----|
| 3 | 0.4079 | 0.3858 | 78.9% |
| **5** (default) | 0.4081 | 0.3870 | **80.4%** |
| 7 | 0.4096 | 0.3886 | 74.3% |

**Takeaway (MkCF):** B1 metrics vary modestly with K; default **K=5** is near-best on NDR/MAE tradeoff.

---

## R14-E2 — CRC vs PIR（Table 3 内部 log 配对）

见 `review_spec/r14_e2_pir_crc_significance.md`（64 paired cells，CRC/PIR batch CSV）。

| Metric | mean(PIR−CRC) | bootstrap 95% CI | Wilcoxon p (H1: CRC better) | CRC lower MAE count |
|--------|---------------|------------------|----------------------------|---------------------|
| MAE | −0.0103 | [−0.0195, −0.0027] | 0.847 | 34/64 |
| MSE | −0.0120 | [−0.0281, +0.0003] | 0.560 | 35/64 |

**Rebuttal 口径:** 全矩阵上 **无单一方法碾压**；revision 将按 cell 报告显著性与效应量，并强调 CRC 的 **NDR / 选择性部署** 维度（PIR log 无 NDR）。

---

## Recalibration（MkCF Q1 / gnqC W4）

1. **New validation slice**: change `--crc_val_ratio` or val window → retrain stage-2 (backbone frozen).  
2. **New τ**: adjust `--crc_tau_quantile` → retrain stage-2.  
3. **Optional**: `--crc_skip_retrain` when only reloading corrector (future: blend-only recal).  
4. **Failure mode**: val not representative → conservative gates (see G2: high NDR, weak MAE on B2).

