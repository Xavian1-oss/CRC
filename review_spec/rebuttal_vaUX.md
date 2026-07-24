# Rebuttal to Reviewer vaUX — Draft

> **Status**: Draft v0.3 — **复用 Table R1**（`rebuttal_results.md`）  
> **Priority**: P0 rebuttal（Major Q = missing baselines）

---

Dear Reviewer vaUX,

Thank you for the review. We agree that the **“when to deploy”** framing is central, and that **missing explicit update policies** weakens the empirical story. We address this below with **new controlled comparisons** shared across reviewers (Table R1 in our supplement / rebuttal PDF if allowed).

---

## Major suggestion — Always update / never update / selective update

We clarify three policies on benchmarks B1–B3:

| Policy | Implementation | B1 MAE | B1 NDR |
|--------|----------------|--------|--------|
| **Never update** | Frozen baseline only (**G0**) | 0.4108 | — |
| **Always accept update** | Full hybrid corrector, **no** validation blending (**G1**, `w1=w2=1`) | **0.4701** | **10.1%** |
| **Selective update (CRC)** | Validation-calibrated per-(variable, horizon) blend (**G5**) | **0.4102** | **72.3%** |

**Table 1 “baseline” rows are G0 (never update).** CRC rows are G5. The missing piece was **G1 (always-on correction)**—now included. G1 shows that always accepting corrections **hurts both MAE and NDR**, supporting the paper’s motivation.

We also include **G2 (ridge + validation gate only)** and **G4 (deploy-only encoder)** in the full ladder (see `rebuttal_results.md`), isolating validation selection vs. full CRC.

---

## Background / datasets / anomalies

We agree the paper should better ground tasks and backbones. **Revision:** expand dataset descriptions (ETT hourly/minute, Weather multivariate, etc.), clarify that benchmarks are **standard cleaned splits** (not heavy OOD stress tests), and discuss limitations for sensor anomalies / drift (see also MkCF-related Limitations).

---

## NDR missing in Table 1

We agree. **Revision:** add NDR to main or appendix tables; rebuttal reports NDR alongside MSE for B1–B3 in Table R1.

---

## Held-out calibration set

CRC uses the **same validation split as Time-Series-Library** long-term forecasting (chronological; no test leakage). Stage-2 calibrates blend weights and τ on this val set only. **`crc_val_ratio`** controls what fraction of val windows train the corrector (sensitivity in paper). Cost: one stage-2 pass after a **frozen** backbone—no backbone fine-tuning.

---

## Minor (PIR, structure)

We will expand PIR in Related Work, add a dedicated **Results** subsection after §4.2, and improve Limitations per your checklist.

---

Thank you again.

---

## 内部备注

- Significance 2：用 Table R1 + problem framing 回应；不 overclaim Traffic 等 headline numbers without seeds（见 14dv / AC）
