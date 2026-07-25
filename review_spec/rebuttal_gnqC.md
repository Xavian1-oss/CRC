Dear Reviewer gnqC,

Thank you for the thoughtful and constructive review. We appreciate your positive view of the **selective deployment** framing and the usefulness of **NDR**. Below we address each weakness and question with **new controlled comparisons** (Table R1 below) in addition to our ablation study.

**Table R1 (Gated Baseline Ladder).** Three benchmarks under a **single unified batch** (same protocol throughout): **B1** ETTh1 + DLinear, pred_len 96; **B2** ETTh1 + DLinear, pred_len 336; **B3** Weather + TimesNet, pred_len 96. **G0** = frozen backbone only; **G1** = always-on deployment*; **G2** = ridge + validation gate only†; **G4** = deploy-only (blend + τ, light encoder); **G5** = Full CRC.

\* **G1** uses validation training but **disables per-(variable, horizon) blend search** (fixed full weights)—this isolates **always-on deployment** of the corrector path, not “no validation.”  
† **G2** uses the **same Stage-A stack as G5** (MTL/STCL/priors); only the **deployment head** differs (no MLP; binary validation gate on the ridge path, w2=0)—a **deployment ablation**, not a minimal linear-only pipeline.

---

**W1 & Q4 — Core selective deployment vs. stabilizers**

We agree that the paper should separate the **core deployment mechanism** from stabilizing details. We will add a subsection *“Core deployment vs. stabilizers vs. representation”* and revise Fig. 1 accordingly:

| Layer | Components | Role |
|-------|------------|------|
| **Core deployment** | Ridge floor; τ from validation; **per-(node, horizon) blending** (including shrink-to-base weights (0,0)) | Decides **where** to deploy correction |
| **Stabilizers** | Direction gate; hard τ clipping | Reduces unsafe nonlinear updates |
| **Representation** | MTL; STCL; multi-scale physics priors | Improves **how well** residuals are fit |

**Evidence (Table R1):**

- **G1 (always-on):** NDR **8.2% / 10.6% / 33.9%** (B1/B2/B3) vs. Full CRC **80.3% / 79.7% / 78.5%** — selective deployment is the main safety mechanism.
- **G3 (no direction gate):** effect is **setting-dependent** (B1 NDR **81.8%** vs. G5 **80.3%**; B2 **79.7%** vs. **78.5%**); we classify the direction gate as a **secondary stabilizer**, not the primary deployment rule.
- **G4 (deploy-only):** B1 NDR **62.1%** vs. Full CRC **80.3%**; on B3 Full CRC achieves **best MAE/MSE** (**0.1903 / 0.1408** vs. G4 **0.2081 / 0.1584**) while G4 remains strong on NDR—representation mainly boosts **deployment quality** on B1, not uniformly on all metrics.

---

**W2 & Q1 — Full CRC vs. validation-based selection alone**

We agree this comparison is central. Summary of Table R1:

| Method | Description | B1 NDR | B2 NDR | B3 NDR | B1 MAE |
|--------|-------------|--------|--------|--------|--------|
| G0 | Baseline only | — | — | — | 0.4108 |
| **G1** | Always-on (no selection) | **8.2%** | **10.6%** | **33.9%** | 0.4738 |
| **G2** | Ridge + val gate only* | **97.9%** | **97.1%** | **90.0%** | 0.4097 |
| G4 | Deploy-only encoder | 62.1% | 77.8% | 79.4% | 0.4097 |
| **G5** | Full CRC | **80.3%** | **79.7%** | **78.5%** | **0.3914** |

**Takeaways:**

1. **Per-(node,h) deployment calibration is essential.** G1 keeps validation training but **disables blend grid search** (fixed full weights); NDR collapses and MAE degrades (B1 MAE **0.4738** vs. G5 **0.3914**). This is stronger than Table 2’s single-filter ablations—it isolates **always-on deployment** of the full corrector path.

2. **High safety alone is insufficient (G2 vs. G5).** Ridge-only validation gating yields **very high NDR** (90–98%) but remains **conservative** and **does not recover the accuracy gains of the full deployment strategy**: weaker B1 MSE than G5 (**0.3941** vs. **0.3780**); on B2, G2 MAE **0.4693** exceeds baseline **0.4654** (NDR **97.1%**) while G5 MAE **0.4392** improves over G0. Full CRC targets **balanced safety and accuracy**—**stronger MSE** with **moderate NDR (~78–80%)**.

3. **Full CRC combines deployment and representation.** G4 lowers B1 NDR (**62.1%** vs. **80.3%**); on B3, Full CRC yields the **strongest MAE/MSE** in the ladder.

We will include the full Table R1 in the revision (main text or appendix).

---

**W3 — Scope of theoretical analysis**

We agree. We will revise the theory section to state clearly that it provides **intuition under stated assumptions**, not a formal deployment guarantee. Main claims remain **empirical**, supported by Table R1 and NDR.

---

**W4, Q3 & Limitations — Validation representativeness**

We will **elevate validation–test mismatch** in Limitations.

CRC calibrates deployment on the validation window. If validation is not representative of test:

- **Expected behavior:** more conservative (0,0) blend weights → **NDR may stay high** (as in G2) but **average MAE gain shrinks**.
- **Mitigation:** validation-ratio sensitivity (Fig. in paper); **recalibration** by retraining stage-2 on a new validation slice (backbone frozen) or adjusting the τ quantile.

We will add a “failure modes” paragraph with G2 on B2: NDR **97.1%** but MAE **worse than baseline** (0.4693 vs. 0.4654).

---

**W5 & Q2 — NDR vs. MAE/MSE**

We will add a short *Practitioner’s guide*:

- **MAE/MSE:** global average accuracy.
- **NDR:** fraction of (horizon × variable) pairs where correction **does not worsen** local error vs. the backbone (**MAE at each unit**, ties count as non-degrading—same as §4.1).

**High NDR does not imply maximum MAE reduction** (Table R1):

- **G2:** B1 NDR **97.9%**, MAE **0.4097**, MSE **0.3941** (small MAE gain over baseline 0.4108).
- **G5:** B1 NDR **80.3%**, MAE **0.3914**, MSE **0.3780** (stronger MSE vs. G0/G2).
- **G1:** B1 NDR **8.2%**, MAE **0.4738**.

When is higher NDR preferred? Settings where **local non-degradation** matters (heterogeneous multivariate series, auditable pipelines). CRC targets both when possible; G2 illustrates an explicit **safety–accuracy trade-off**.

---

**Questions (summary)**

| Q | Answer |
|---|--------|
| **Q1** | Always-on vs. safety: **G1→G5**. **G2** (same Stage-A as G5, ridge-only gate): high NDR, **conservative**—weaker B1 MSE vs G5. Full CRC adds nonlinear + full blend for MSE/NDR balance. |
| **Q2** | See W5; G1/G2/G5 form a clear trade-off triangle. |
| **Q3** | Limitations + recalibration + G2 conservative example on B2. |
| **Q4** | Core = blend + τ + ridge; stabilizers = dir gate; representation = MTL/STCL/priors (**G4**). |

---

**Limitations (your comment)**

We will make validation–test mismatch **more prominent**, since the deployment rule is central to CRC.

---

**Summary**

We hope Table R1 and the three-layer decomposition address your main questions. **G2** (same representation as G5, restricted deployment head) achieves high NDR but is **too conservative** to match Full CRC on MSE/accuracy. **G1** shows that **per-unit deployment calibration**—not merely training a corrector on validation—is essential for NDR.

Thank you again for the helpful feedback.
