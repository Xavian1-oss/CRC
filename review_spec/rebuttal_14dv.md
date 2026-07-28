Dear Reviewer 14dv,

Thank you for the strong support and clear summary of CRC. We address the two reporting gaps below.

---

**Q1 — Variability across seeds**

We agree. **Revision:** we will report **mean ± std** over ≥3 seeds for Full CRC and key baselines on representative dataset–backbone pairs (at minimum ETTh1/DLinear and Weather/TimesNet).

**Rebuttal preview (B1; same protocol/hyperparameters as Table R1, separate multi-seed job):** Table R1 is a **representative fixed-seed run** (G5/G1 MAE **0.3914** / **0.4738**; NDR **80.3%** / **8.2%**). The ladder seed is **predefined** (`fix_seed` 2021), not selected by performance. **Additional three-seed reruns** give G5 MAE **0.3991 ± 0.0011** and NDR **79.3 ± 6.3%**; G1 remains **7.2 ± 1.5%** NDR with much higher MAE—the **relative G1 vs. G5 trends** are consistent across seeds.

| Config | MAE (mean ± std) | NDR (mean ± std) |
|--------|------------------|------------------|
| Full CRC (G5) | **0.3991 ± 0.0011** | **79.3 ± 6.3%** |
| Always-deploy (G1) | 0.4828 ± 0.0131 | 7.2 ± 1.5% |

The **large G1 vs. G5 gap in MAE and NDR** is consistent across seeds—not a single-seed artifact. We will extend multi-seed reporting to additional settings in the camera-ready.

---

**Q2 — Statistical significance vs. PIR (Table 3)**

We will add **paired statistical analysis** across datasets and horizons (bootstrap 95% CI on MAE/MSE differences and/or Wilcoxon signed-rank tests), aligned with **Table 3’s seven-dataset scope** in the revision.

**Rebuttal preview (ETT subset, finer granularity than Table 3):** Table 3 reports **horizon-averaged** Δ over **28 dataset–backbone pairs** (7 datasets × 4 backbones). As an interim paired check on **64 cells**—**ETTh1, ETTh2, ETTm1, ETTm2** × 4 backbones × horizons {96, 192, 336, 720} (same frozen-backbone protocol as our internal PIR/CRC logs)—from a **Jul 2026 paired rerun**, CRC wins **40/64** (MAE) and **42/64** (MSE) with mean paired advantage to CRC (Δ=PIR−CRC > 0). **Sign tests** favor CRC on this subset (MAE *p*=0.03; MSE *p*=0.008). **Wilcoxon** signed-rank tests agree in direction (MAE *p*≈0.07—not significant at 0.05; MSE *p*≈0.003)—not uniform wins on every cell (~24 MAE cells still favor PIR). This **does not** replace Table 3 (e.g., ETTh1 is PIR-favored there under horizon averaging; Weather/Electricity/Traffic are outside the 64-cell subset). We **do not** claim CRC dominates PIR on all settings; we view **PIR and CRC as complementary** (instance-level **revision identification** vs **correction trustworthiness / deploy-or-reject** at (var,h)) and emphasize **NDR / selective deployment** (not in PIR logs).

**Revision:** report **per-cell** significance, effect sizes, and uncertainty on the **full Table 3 grid** (7 datasets). **PIR identifies which forecasts to revise and performs post-hoc revision; CRC decides whether a generated correction should be trusted and deployed** at (var,h), with **NDR** and validation-calibrated selective deployment—dimensions our PIR runs do not report—so Table 3 alone does not capture the full comparison. Small effect sizes will be stated explicitly, consistent with your view that the remaining gaps are **reporting-related**, not conceptual.

---

**Limitations**

We appreciate your “None” on limitations; we will still expand validation–test mismatch and static graph priors in the revision (raised by other reviewers).

Thank you again.
