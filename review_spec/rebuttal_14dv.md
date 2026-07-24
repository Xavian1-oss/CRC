# Rebuttal to Reviewer 14dv — Draft

> **Status**: Draft v0.3  
> **Rating**: 5 Accept — weaknesses are **reporting**, not conceptual

---

Dear Reviewer 14dv,

Thank you for the strong support and clear summary of CRC. We address the two reporting gaps below.

---

## Q1 — Variability across seeds

We agree. **Revision:** we will report **mean ± std** (≥3 seeds) for Full CRC and key baselines on representative dataset–backbone pairs (at minimum ETTh1/DLinear and Weather/TimesNet from our internal B1–B3 ladder).

**Rebuttal-period status:** We completed **R14-E1** on B1 (ETTh1/DLinear/pl=96): over seeds {2021,2022,2023}, Full CRC (G5) gives **MAE 0.4091±0.0011** and **NDR 72.3±6.3%**; always-on (G1) gives **MAE 0.4828±0.0131** and **NDR 7.2±1.5%** (see `review_spec/rebuttal_results.md` §Multi-seed). This confirms the Table R1 gap is **stable across seeds**, not a single-seed artifact.

---

## Q2 — Statistical significance vs. PIR (Table 3)

We will add **paired statistical analysis** across datasets/horizons (e.g., bootstrap 95% CI on MAE/MSE differences, or Wilcoxon signed-rank on paired errors). **Rebuttal preview:** on 64 paired cells from our internal CRC/PIR batch logs, aggregate Wilcoxon tests do **not** show CRC uniformly dominating PIR on MAE alone (see `review_spec/r14_e2_pir_crc_significance.md`); **revision** will report **per-cell** significance, effect sizes, and pair with **NDR** (unavailable for PIR in our protocol).

We will not overclaim significance where effect sizes are negligible (consistent with your reading that weaknesses are reporting-related).

---

## Limitations

We appreciate “None” on limitations; we will still expand validation–test mismatch and graph prior limitations in revision for other reviewers’ concerns.

Thank you again.

---

## 内部备注

- 实验 ID：**R14-E1**, **R14-E2** in `requirements.md` §4.2
