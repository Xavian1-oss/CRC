Dear Area Chair and reviewers,

Thank you for the meta-review and for consolidating the reviewers’ concerns. We respond to **meta-level weaknesses** and **AC follow-up questions** here. Detailed replies to individual reviewers are in separate posts.

---

**Missing controls: always-on correction and validation-gated ridge**

Reviewers noted missing **always-on** and **simple validation-gated** baselines. We now report a **Gated Baseline Ladder** on three benchmarks (ETTh1/DLinear pl=96 & 336; Weather/TimesNet pl=96), from **one unified experimental batch** (same protocol throughout):

| Config | Role | B1 NDR | B1 MAE |
|--------|------|--------|--------|
| G0 | Never update (Table 1 baseline) | — | 0.4108 |
| G1 | Always-on correction | **8.2%** | 0.4738 |
| G2 | Ridge + validation gate only* | **97.9%** | 0.4097 |
| G5 | Full CRC | **76.3%** | 0.4084 |

\*G2 uses the **same Stage-A representation stack as G5** (MTL/STCL/priors); only the **deployment head** is restricted (no MLP; binary validation gate on the ridge path, w2=0).

**(i)** **Selective deployment (G5 vs. G1)** is essential for NDR and MAE—not merely having a validation set.

**(ii)** **G2 vs. G5** illustrates a **safety–capacity trade-off**: G2 achieves very high NDR (**97.9%** on B1; **97.1%** on B2) but is **conservative**—it **does not recover the accuracy gains of full selective deployment** (B1 MSE **0.3941** vs. G5 **0.3880**; B2 MAE **0.4693** vs. baseline **0.4654**). Full CRC (G5) targets **moderate NDR (~76–79%) with stronger global MSE** on B1–B3.

Full B1–B3 MAE/MSE/NDR appear in our reviewer-specific rebuttals (Table R1).

**Multi-seed preview (B1):** Table R1 reports a **fixed-seed representative run** (G5 B1 NDR **76.3%**, MAE **0.4084**). **Additional three-seed reruns** (same CLI) give G5 MAE **0.4091 ± 0.0011**, NDR **72.3 ± 6.3%**; G1 remains **7.2 ± 1.5%** NDR with much higher MAE—the **G1 vs. G5 gap is stable qualitatively** (stage-2 training variance explains modest NDR spread).

---

**AC-Q1 — Novelty vs. TAFAS, SOLID, PETSA, FAC, FFORMA/FFORMA**

We will **expand Related Work** and clarify **what is new in CRC as a synthesis**:

| Line of work | Typical setting | CRC |
|--------------|-----------------|-----|
| **TAFAS, PETSA, FAC** | **Test-time adaptation** (backbone updates at test) | **Frozen backbone** + post-hoc corrector with **validation-calibrated deployment** |
| **SOLID** | Calibration / **context-shift detection** | **Per-(variable, horizon) deployment** from val residuals + hybrid corrector |
| **PIR** | **Magnitude** of an accepted revision (instance-wise strength) | **Whether** (and how) to deploy correction paths; **NDR** as a local-harm diagnostic |
| **FFORMA / FFORMS** | **Select among independent forecasters** | **Select correction pathways** (baseline / ridge / nonlinear) **within a single frozen forecaster** |

**Novelty (concise):** CRC is **not merely a stronger residual corrector**; it adds an explicit **deployment decision layer**—**when to deploy** which candidate correction—calibrated on validation (per-unit blend over ridge + gated nonlinear paths), with **NDR** exposing local harm that MSE hides. The new **G1/G2/G5 ladder** supports this framing empirically.

**Why not empirical baselines for TTA / FFORMA / etc.:** these lines optimize **different objectives under different protocols** (e.g., TTA adapts θ at test time; FFORMA combines separate models). Under our **frozen-backbone post-hoc** setting, they are **not direct apples-to-apples baselines**; we will discuss them fairly and cite them in revision. We are open to adding **one** closest comparison if the AC recommends a specific frozen-backbone protocol.

**PIR (revision):** we will soften §4.3—PIR primarily controls **how much** to revise **after** a revision is accepted; CRC introduces an explicit **whether-to-deploy** decision over residual corrections (plus ridge floor and NDR-centric evaluation).

---

**AC-Q2 — Appendix A / “formal analysis”**

We agree **Prop A.3** is largely by construction and **Prop A.4** does not imply the displayed bound as stated (Remark A.5). We will **revise the abstract and appendix** to present Appendix A as **design intuition under idealized assumptions**, not a deployment guarantee.

---

**Missing statistics / “26 of 28” claim**

We agree **seed variance** and **formal comparisons** should be reported. **Revision:** mean±std over ≥3 seeds; paired tests for CRC vs. PIR on Table 3 with **per-cell** effect sizes and uncertainty. We will **avoid presenting aggregate win counts without uncertainty bands** and will qualify small gains explicitly.

---

**Validation–test representativeness**

Addressed in individual rebuttals (conservative G2 behavior, recalibration, expanded Limitations).

---

We hope these clarifications and the gated baseline ladder address the meta-review concerns. We are committed to a revision that reflects this evidence.

Sincerely,  
The authors
