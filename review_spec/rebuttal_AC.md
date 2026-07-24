# Response to Area Chair (asfz) — Draft

> **Status**: Draft v0.3  
> **Use**: Author response to **Meta Review** + AC-Q1/Q2 (OpenReview discussion)

---

Dear Area Chair and reviewers,

Thank you for the meta-review and for consolidating the reviewers’ concerns. We respond to the **meta-level weaknesses** and **AC follow-up questions** below. Detailed replies to **gnqC, vaUX, 14dv, and MkCF** are in separate rebuttal posts; this letter focuses on **cross-cutting** issues.

---

## Missing controls: always-on correction and validation-gated ridge

Reviewers **vaUX, gnqC**, and the **meta-review** noted missing **always-on** and **simple validation-gated** baselines.

We now report a **Gated Baseline Ladder** on three benchmarks (ETTh1/DLinear pl=96 & 336; Weather/TimesNet pl=96)—summary:

| Config | Role | B1 NDR (example) |
|--------|------|------------------|
| G0 | Never update (Table 1 baseline) | — |
| G1 | Always-on correction | **10.1%** |
| G2 | Ridge + validation gate only | **97.8%** |
| G5 | Full CRC | **72.3%** |

**Takeaway:** (i) **Selective deployment (G5 vs G1)** is essential for NDR and MAE—not merely having a validation set. (ii) **Simple ridge gating (G2)** achieves very high NDR but **does not match Full CRC on average accuracy** (e.g., B2 MAE worse than baseline). Full numbers: `review_spec/rebuttal_results.md` (authors’ supplement).

This directly addresses the meta-review’s “missing controls” point.

---

## AC-Q1 — Novelty vs. TAFAS, SOLID, PETSA, FAC, FFORMA/FFORMA

We thank the AC for these references. We will **expand Related Work** and clarify **what is new in CRC as a synthesis**:

| Line of work | Typical setting | CRC |
|--------------|-----------------|-----|
| **TAFAS, PETSA, FAC** | **Test-time adaptation** (backbone updates at test) | **Frozen backbone** + post-hoc residual corrector |
| **SOLID** | Calibration / **context-shift detection** | **Per-(variable, horizon) deployment** from val residuals + hybrid corrector |
| **PIR** (cited) | Instance-wise revision strength | Explicit **accept/reject/shrink** with **NDR** diagnostic |
| **FFORMA / FFORMS** | **Ensemble forecast combination** | Single backbone + selective **residual** deployment |

**Novelty (concise):** reframing post-hoc correction as **when to deploy** (validation-calibrated per-unit blend over ridge + gated nonlinear paths), with **NDR** exposing local harm that MSE hides—supported by new G1/G2/G5 ladder.

**Why not included as empirical baselines:** different problem settings (TTA adapts θ; FFORMA combines models); limited comparable code under **frozen-backbone post-hoc** protocol; revision will discuss and cite fairly. We are open to adding **one** closest TTA/TAFAS-style discussion comparison in revision if the AC recommends a specific frozen-backbone protocol.

**PIR difference (revision):** we will soften §4.3—PIR **does** modulate revision; CRC adds **explicit per-(var,h) blend grid**, ridge floor, and **NDR-centric** evaluation.

---

## AC-Q2 — Appendix A / “formal analysis”

We agree **Prop A.3** is largely by construction and **Prop A.4** does not imply the displayed bound as stated (Remark A.5). We will **revise the abstract and appendix** to present Appendix A as **design intuition under idealized assumptions**, not a deployment guarantee—aligned with gnqC’s W3.

---

## Missing statistics / “26 of 28” claim

We agree **seed variance** and **formal comparisons** should be reported (**14dv**). **Revision:** mean±std over ≥3 seeds; paired tests for PIR vs CRC (Table 3). We will **qualify** headline win counts: TimeXer column includes **negative cells**; small gains need uncertainty bands.

---

## Validation–test representativeness

Addressed in MkCF/gnqC rebuttals + expanded Limitations.

---

We hope these clarifications and the new baseline ladder address the AC’s reservations. We are committed to a revision that reflects this evidence.

Sincerely,  
The authors

---

## 内部备注

- 勿在 rebuttal 中攻击 reviewer；若 AC 某条与事实不符，礼貌纠正并引用 Table R1
