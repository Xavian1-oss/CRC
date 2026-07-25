Dear Reviewer 14dv,

Thank you for the strong support and clear summary of CRC. We address the two reporting gaps below.

---

**Q1 — Variability across seeds**

We agree. **Revision:** we will report **mean ± std** over ≥3 seeds for Full CRC and key baselines on representative dataset–backbone pairs (at minimum ETTh1/DLinear and Weather/TimesNet).

**Rebuttal preview (B1; same CLI as unified Table R1):** In the **unified batch** (seed 2021), G5/G1 give MAE **0.4084** / **0.4738** and NDR **76.3%** / **8.2%**. **Additional three-seed reruns** give G5 MAE **0.4091 ± 0.0011** and NDR **72.3 ± 6.3%** (stage-2 variance); G1 remains **7.2 ± 1.5%** NDR with much higher MAE. The **G1 vs. G5 gap** is stable qualitatively across seeds.

| Config | MAE (mean ± std) | NDR (mean ± std) |
|--------|------------------|------------------|
| Full CRC (G5) | **0.4091 ± 0.0011** | **72.3 ± 6.3%** |
| Always-on (G1) | 0.4828 ± 0.0131 | 7.2 ± 1.5% |

G5 MAE is stable across seeds; the **large G1 vs. G5 gap in MAE and NDR** is consistent across seeds—not a single-seed artifact. We will extend multi-seed reporting to additional settings in the camera-ready.

---

**Q2 — Statistical significance vs. PIR (Table 3)**

We will add **paired statistical analysis** across datasets and horizons (bootstrap 95% CI on MAE/MSE differences and/or Wilcoxon signed-rank tests).

**Rebuttal preview:** On **64 paired cells** from our internal CRC and PIR runs (matched dataset × backbone × horizon), aggregate Wilcoxon tests on MAE **do not** show CRC uniformly dominating PIR (**34/64** cells with lower CRC MAE; Wilcoxon *p* ≈ 0.85). We will **not** overclaim global MAE superiority over PIR.

**Revision:** report **per-cell** significance, effect sizes, and uncertainty. **PIR adjusts revision magnitude after an update is accepted; CRC decides whether to deploy a candidate correction in the first place**, with **NDR** and validation-calibrated selective deployment—dimensions our PIR runs do not report—so Table 3 alone does not capture the full comparison. Small effect sizes will be stated explicitly, consistent with your view that the remaining gaps are **reporting-related**, not conceptual.

---

**Limitations**

We appreciate your “None” on limitations; we will still expand validation–test mismatch and static graph priors in the revision (raised by other reviewers).

Thank you again.
