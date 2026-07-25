Dear Area Chair and reviewers,

Thank you for the meta-review. We address meta-level concerns and AC questions below; per-reviewer detail is in separate posts.

---

**Missing controls: always-on correction and validation-gated ridge**

Reviewers noted missing always-on correction and a simple validation-gated baseline. **Table R1** (Gated Baseline Ladder), one batch: ETTh1/DLinear pl=96 & 336; Weather/TimesNet pl=96.

| Config | Role | B1 NDR | B1 MAE |
|--------|------|--------|--------|
| G0 | Never update (Table 1 baseline) | — | 0.4108 |
| G1 | Always-on* | **8.2%** | 0.4738 |
| G2 | Ridge + val gate† | **97.9%** | 0.4097 |
| G5 | Full CRC | **80.3%** | **0.3914** |

\*G1 uses validation training but **disables per-(variable, horizon) blend search** (fixed full weights)—this isolates **always-on deployment** of the corrector path, not “no validation.”  
†G2 uses the **same Stage-A representation stack as G5** (MTL/STCL/priors); only the **deployment head** differs (no MLP; binary validation gate on the ridge path, w2=0).

**(i) G5 vs. G1:** selective per-(var,h) calibration is essential for both NDR and MAE (B1–B3 NDR ~8–34% vs ~78–80%; B1 MAE 0.4738 vs 0.3914).

**(ii) G2 vs. G5:** illustrates a safety–capacity trade-off—ridge-only gating yields very high NDR (97.9% B1; 97.1% B2) but remains conservative and does not match G5 on global accuracy (B1 MSE 0.3941 vs 0.3780; B2 G2 MAE 0.4693 vs G0 0.4654, G5 0.4392). G5 targets balanced ~78–80% NDR with stronger MSE/MAE on B1–B3 (full table in gnqC/vaUX posts).

**Multi-seed preview (B1):** Table R1 G5 (MAE **0.3914**, NDR **80.3%**); three-seed reruns **0.3991 ± 0.0011**, NDR **79.3 ± 6.3%**; G1 NDR **7.2 ± 1.5%**—G1 vs. G5 gap stable qualitatively.

---

**AC-Q1 — Novelty vs. TAFAS, SOLID, PETSA, FAC, FFORMA, PIR**

We will expand Related Work and clarify what is new in CRC as a synthesis. Existing lines primarily optimize correction generation, test-time adaptation, or revision magnitude; CRC treats correction deployment—whether and how candidate residual corrections should be trusted—as a separate decision problem calibrated on validation.

| Line | Typical setting | CRC |
|------|-----------------|-----|
| TAFAS, PETSA, FAC | Test-time adaptation (backbone updates at test) | Frozen backbone + post-hoc corrector with validation-calibrated deployment |
| SOLID | Calibration / context-shift detection | Per-(var,h) deployment from validation residuals + hybrid corrector |
| PIR | Magnitude of an accepted revision | Whether (and how) to deploy correction paths; NDR as local-harm diagnostic |
| FFORMA / FFORMS | Select among independent forecasters | Select correction pathways within a single frozen forecaster |

**Novelty (concise):** CRC augments residual correction with an explicit deployment decision layer, shifting emphasis from how to generate corrections to when they should be trusted and deployed (ridge + gated nonlinear candidates). It is not merely a stronger residual corrector; per-(var,h) calibration and NDR expose local harm that global MSE hides. The G1/G2/G5 ladder supports this framing empirically under shared Stage-A weights.

Why not TTA/FFORMA baselines here: different objectives/protocols (test-time θ updates; ensembling separate models) vs. our frozen-backbone post-hoc setting. We will discuss fairly in revision and can add one frozen-backbone comparison if the AC recommends a protocol.

**PIR (revision):** we will soften §4.3—PIR adjusts the magnitude of an accepted revision; CRC decides whether a candidate residual correction should be trusted and deployed in the first place, with per-(var,h) calibration, a ridge floor, and NDR-centric evaluation.

---

**AC-Q2 — Appendix A / “formal analysis”**

We agree Prop A.3 is largely by construction and Prop A.4 does not imply the displayed bound as stated (Remark A.5). Revision: present Appendix A as design intuition under idealized assumptions, not a deployment guarantee; soften abstract wording accordingly.

---

**Statistics (“26/28”) & validation–test representativeness**

We agree seed variance and formal comparisons should be reported. **Revision:** mean±std over ≥3 seeds; formal tests on Table 1/3 with uncertainty—we will **not** treat the 26/28 pair count alone as significance.

**PIR preview (≠26/28):** **64 ETT cells** (4 ETT × 4 backbones × 4 horizons; Jul 2026 rerun): CRC **40/64** MAE, **42/64** MSE wins; sign *p*=0.03/0.008; Wilcoxon *p*≈0.07/0.003—**ETT subset only** (see 14dv); full 7-dataset tests in revision.

Validation–test mismatch: deployment rules are calibrated on validation; under shift, CRC tends toward conservative gates (high NDR, reduced MAE gain). We will elevate this in Limitations with G2 conservative behavior and stage-2 recalibration (MkCF/gnqC).

We will incorporate these revisions into the updated paper.

Sincerely,  
The authors
