# Rebuttal to Reviewer gnqC — Final

> **OpenReview ID**: gnqC · **Alias**: `rebuttal_r1.md`  
> **Evidence**: `review_spec/rebuttal_results.md` (Table R1)

---

Dear Reviewer gnqC,

Thank you for the thoughtful and constructive review. We appreciate your positive view of the **selective deployment** framing and the usefulness of **NDR**. Below we address each weakness and question with **new controlled comparisons** (Table R1) in addition to our ablation study.

---

## W1 & Q4 — Core selective deployment vs. stabilizers

We agree that the paper should separate the **core deployment mechanism** from stabilizing details. We will add a subsection *“Core deployment vs. stabilizers vs. representation”* and revise Fig. 1 accordingly:

| Layer | Components | Role |
|-------|------------|------|
| **Core deployment** | Ridge floor; τ from validation; **per-(node, horizon) blending** (including shrink-to-base weights `(0,0)`) | Decides **where** to deploy correction |
| **Stabilizers** | Direction gate; hard τ clipping | Reduces unsafe nonlinear updates |
| **Representation** | MTL; STCL; multi-scale physics priors | Improves **how well** residuals are fit |

**Evidence (Table R1):**

- **G1 (always-on, no selective blending):** NDR **10.1% / 15.2% / 33.1%** vs. Full CRC **72.3% / 64.6% / 73.3%** — deployment selection is the main safety mechanism.
- **G3 (no direction gate):** B2 NDR **61.2%** vs. Full CRC **64.6%** — the direction gate is a **stabilizer**, not the primary deployment rule.
- **G4 (deploy-only, light encoder):** keeps blending/τ but removes MTL/STCL/full priors; B1 NDR **62.1%** vs. Full CRC **72.3%**, while B3 MAE is competitive — representation modules affect both accuracy and deployment quality.

---

## W2 & Q1 — Gain from full CRC vs. validation-based selection alone

We agree this comparison is central. We now report a **Gated Baseline Ladder** on B1–B3 (Table R1):

| Method | Description | B1 NDR | B2 NDR | B3 NDR | B1 MAE |
|--------|-------------|--------|--------|--------|--------|
| G0 | Baseline only | — | — | — | 0.4108 |
| **G1** | Always-on hybrid (`w1=w2=1`, no selection) | **10.1%** | **15.2%** | **33.1%** | 0.4701 |
| **G2** | **Ridge + validation gate** (`w1∈{0,1}`, `w2=0`) | **97.8%** | **94.8%** | **89.9%** | 0.4097 |
| **G4** | Deploy-only (blend+τ, light encoder) | 62.1% | 77.8% | 79.4% | 0.4097 |
| **G5** | **Full CRC** | 72.3% | 64.6% | 73.3% | **0.4102** |

**Takeaways:**

1. **Validation alone is not enough.** G1 uses the same corrector and validation split but **without** per-(node,h) selection; NDR collapses and MAE degrades sharply (e.g., B1 MAE **0.4701** vs. **0.4102**).

2. **Simple validation gating (G2) is safe but conservative.** Ridge+val gate yields **very high NDR** (90–98%) by rarely deploying harmful corrections, but **MAE/MSE gains are modest** (B2 MAE even **exceeds** baseline: 0.4725 vs. 0.4654). Full CRC’s nonlinear branch + full blend grid is needed for **stronger average accuracy** while maintaining good NDR.

3. **Full CRC combines deployment and representation.** G4 shows that even with selective blending, removing MTL/STCL/full priors lowers deployment quality on B1 (NDR 62.1% vs. 72.3%).

We will include Table R1 in the revision (main text or appendix).

---

## W3 — Scope of theoretical analysis

We agree. We will revise the theory section to state clearly that it provides **intuition under stated assumptions**, not a formal deployment guarantee. Main claims remain **empirical**, supported by Table R1 and NDR.

---

## W4, Q3 & Limitations — Validation representativeness

We will **elevate validation–test mismatch** in Limitations, as you suggest.

CRC calibrates deployment (blend weights, τ) on the validation window. If validation is not representative of test:

- **Expected behavior:** more conservative `(0,0)` weights → **NDR may remain high** (as in G2) but **average MAE gain shrinks**.
- **Mitigation:** our validation-ratio sensitivity analysis (Fig. in paper); **recalibration** by retraining stage-2 on a new validation slice (backbone unchanged) or adjusting τ quantile.

We will add a short “failure modes” paragraph with this conservative-regime example (G2 on B2: NDR 94.8% but MAE −1.5% vs. baseline).

---

## W5 & Q2 — NDR vs. MAE/MSE

We will add a *Practitioner’s guide*:

- **MAE/MSE:** global average accuracy.
- **NDR:** fraction of (horizon × variable) pairs where correction **does not worsen** local MAE vs. the backbone.

**High NDR does not imply maximum MAE reduction.** Example from Table R1:

- **G2 (ridge+val gate):** B1 NDR **97.8%**, MAE **0.4097** (small gain over baseline 0.4108).
- **G5 (Full CRC):** B1 NDR **72.3%**, MAE **0.4102**, MSE **0.3868** (stronger global accuracy).
- **G1 (always-on):** B1 NDR **10.1%**, MAE **0.4701** (unsafe correction).

**When prefer higher NDR over slightly better average error?** Production settings where local non-degradation is critical (heterogeneous multivariate series, auditable pipelines). CRC targets **both** when possible; G2 illustrates the explicit **safety–accuracy trade-off**.

---

## Responses to Questions (summary)

| Q | Answer |
|---|--------|
| **Q1** | Safety (NDR) vs. always-on: **G1→G5**. Simple val gate (G2): high NDR, limited MAE gain. Full CRC adds nonlinear+calibrated blend for accuracy. |
| **Q2** | See W5; G1/G2/G5 form a clear trade-off triangle. |
| **Q3** | Limitations + recalibration + G2 conservative failure on B2. |
| **Q4** | Core = blend+τ+ridge; stabilizers = dir gate; representation = MTL/STCL/priors (G4 evidence). |

---

## Limitations (your comment)

We will make validation–test mismatch **more prominent**, since the deployment rule is central to CRC.

---

## Summary

We hope Table R1 and the clarified three-layer decomposition address your main questions. The new **G2 (ridge+validation gate)** baseline shows that simple validation-based selection can achieve high NDR but does not replace the full CRC design for **balanced accuracy and safety**. The **G1 (always-on)** baseline shows that selective deployment—not validation alone—is essential for NDR.

Thank you again for the helpful feedback.

---

## 内部备注（勿提交）

- G2 实现：`--crc_ridge_gate_only 1`（`models/CRC.py`）
- 脚本：`scripts/review/run_g2_experiments.sh`, `run_r1_experiments.sh`
- 全员映射：`review_spec/requirements.md` v0.3
