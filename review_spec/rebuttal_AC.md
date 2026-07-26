Dear Area Chair and reviewers,

Thank you for the meta-review. We address meta-level concerns and AC questions below; per-reviewer detail is in separate posts.

---

**Missing controls: always-on correction and validation-gated ridge**

Reviewers noted missing always-on correction and a simple validation-gated baseline. **Table R1** (Gated Baseline Ladder), one batch: ETTh1/DLinear pl=96 & 336; Weather/TimesNet pl=96.

| Config | Role | B1 NDR | B1 MAE |
|--------|------|--------|--------|
| G0 | Never update (Table 1 baseline) | — | 0.4108 |
| G1 | Always-deploy* | **8.2%** | 0.4738 |
| G2 | Ridge + val gate† | **97.9%** | 0.4097 |
| G5 | Full CRC | **80.3%** | **0.3914** |

\*G1 uses validation training but **disables per-(variable, horizon) blend search** (fixed full weights)—this isolates **always-on deployment** of the corrector path, not “no validation.”  
†G2 uses the **same Stage-A representation stack as G5** (MTL/STCL/priors); only the **deployment head** differs (no MLP; binary validation gate on the ridge path, w2=0).

**(i) G5 vs. G1:** in our ladder, selective per-(var,h) calibration is the key factor explaining the observed gains in both NDR and MAE (B1–B3 NDR ~8–34% vs ~78–80%; B1 MAE 0.4738 vs 0.3914).

**(ii) G2 vs. G5:** high NDR (97.9% B1) but weaker MSE/MAE than G5 (B1 MSE 0.3941 vs 0.3780; B2 G2 MAE 0.4693 vs G0 0.4654, G5 0.4392)—see gnqC/vaUX for full Table R1.

**Multi-seed preview (B1):** Predefined seed **2021** (not post hoc); G5 **0.3914** / **80.3%** NDR; reruns **0.3991 ± 0.0011**, G1 NDR **7.2 ± 1.5%**. **G1 vs. G5 trends** hold across seeds.

---

**AC-Q1 — Novelty vs. TAFAS, SOLID, PETSA, FAC, FFORMA, PIR**

**Revision:** expand Related Work with the clarification below (Methods/Limitations as needed).

Prior lines mainly target adaptation, revision magnitude, or model selection. **The distinction is not whether a gate exists, but what is gated:** CRC controls whether a **generated residual correction** should be trusted and deployed under a **frozen backbone** (per-(var,h), validation-calibrated at test).

| Line | Primary decision | CRC emphasis |
|------|------------------|--------------|
| TAFAS, PETSA, FAC | How/whether to adapt the backbone at test | Frozen backbone + validation-calibrated deployment of post-hoc corrections |
| SOLID | Whether context shift warrants action | Per-(var,h) deployment from validation residuals + hybrid corrector |
| PIR | Revision magnitude after an update is accepted | Whether (and how) to deploy correction paths; NDR as local-harm diagnostic |
| FFORMA / FFORMS | Which forecaster to ensemble | Which correction pathway within one frozen forecaster |

CRC is **not merely a residual corrector**; it adds an explicit deployment decision layer **orthogonal to correction generation**—when and how residual corrections should be trusted and deployed. Per-(var,h) calibration and NDR expose local harm that global MSE hides.

**Empirical distinction (Table R1, B1–B3):** **G1 vs G5** (always-deploy vs selective NDR/MAE); **G2 vs G5** (high NDR, weaker B1 MSE)—deployment ladder on frozen backbones, not TTA/PETSA protocol benchmarks.

We do not run TTA/FFORMA in rebuttal; we will discuss in revision and can add one frozen-backbone comparison if the AC recommends a protocol.

**PIR (revision):** soften §4.3—PIR modulates revision strength; CRC calibrates **deployment** of residual corrections (per-(var,h) blend, ridge floor, NDR).

---

**AC-Q2 — Appendix A / “formal analysis”**

We agree Prop A.3 is largely by construction and Prop A.4 does not imply the displayed bound as stated (Remark A.5). Revision: present Appendix A as design intuition under idealized assumptions, not a deployment guarantee; soften abstract wording accordingly.

---

**Statistics (“26/28”) & validation–test representativeness**

We agree seed variance and statistical reporting should be strengthened. **Revision:** mean±std over ≥3 seeds on representative settings; **paired statistical analysis on key comparisons** (including Table 3 CRC vs PIR)—we will **not** treat the Table 1 **26/28** pair count alone as significance.

**PIR preview (≠26/28):** 64 ETT cells (14dv): **40/64** MAE, **42/64** MSE wins; Wilcoxon MAE *p*≈0.07 (n.s. at 0.05). **PIR and CRC are complementary**, not MAE-dominance. Full Table 3 analysis in revision.

Validation–test mismatch: deployment rules are calibrated on validation; under shift, CRC tends toward conservative gates (high NDR, reduced MAE gain). We will elevate this in Limitations with G2 conservative behavior and stage-2 recalibration (MkCF/gnqC).

We will incorporate these revisions into the updated paper.

Sincerely,  
The authors
