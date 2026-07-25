Dear Reviewer MkCF,

Thank you for the detailed review and for highlighting **when to deploy** as the core contribution. We respond to each point below.

**Table R1** (single unified batch, same protocol): B1 ETTh1/DLinear pl=96; B2 pl=336; B3 Weather/TimesNet pl=96.

---

**W1 — Validation–test mismatch**

We agree deployment rules are calibrated on validation and **may fail under distribution shift**. We will **expand Limitations** prominently.

**Evidence:** Our validation-ratio sensitivity analysis (paper) shows reasonable stability under varying val fractions. Under mismatch, CRC tends toward **conservative gates**—high NDR but reduced MAE gain. From Table R1, **G2** (*same Stage-A as G5; ridge-only validation gate*) on B2: **97.1% NDR** but MAE **0.4693** vs. baseline **0.4654**.

**Recalibration (Q1):** When the deployment context changes, recalibrate by **retraining stage-2 only** (backbone frozen)—e.g., a new validation slice or a new τ quantile. **Frequency:** when the **operational validation window** changes (new season or month of held-out data), not on every test batch.

---

**W2 — KNN graph (K=5)**

The adjacency uses **training-set correlation** with top-K neighbors (default **K=5**). It captures **static linear co-movement**, not dynamic lags—we will state this clearly as a **limitation**.

**Sensitivity (B1, Full CRC, seed 2021):**

| K | MAE | MSE | NDR |
|---|-----|-----|-----|
| 3 | 0.4079 | 0.3858 | 78.9% |
| **5 (default)** | 0.4081 | 0.3870 | **80.4%** |
| 7 | 0.4096 | 0.3886 | 74.3% |

Metrics vary modestly with K; **K=5** is near-best on this benchmark (standalone B1 rerun; NDR may differ by a few pp from unified Table R1 G5 due to stage-2 variance). We will include this table in the revision appendix.

---

**W3 — Online / strong drift benchmarks**

We agree standard ETT/Weather/Electricity/Traffic splits do not stress **online drift**. We scope CRC as **post-hoc correction under frozen backbones** with a single calibration window; **online recalibration** is future work (Limitations).

---

**W4 — Hyperparameters / migration cost**

CRC is two-stage: (1) train or warm-start the backbone once; (2) train the corrector on validation predictions. Table R1 includes **deployment ablations** (G2: restricted gate on same Stage-A stack; G4 deploy-only encoder). We will add an appendix **migration checklist** (backbone checkpoint + corrector weights).

---

**Q3 — NDR vs. MSE/MAE**

**NDR** is the fraction of (horizon × variable) pairs where correction **does not worsen** local MAE vs. the backbone; **MAE/MSE** are global averages. **High NDR does not imply maximum MAE reduction.** Table R1: **G2** on B1 has NDR **97.9%** but MSE **0.3941** vs. **G5** NDR **76.3%**, MSE **0.3880**; **G1** has NDR **8.2%**, MAE **0.4738**.

---

**Limitations (your comment)**

We will substantially expand Limitations (validation–test mismatch, static graph, drift, hyperparameters).

Thank you again.
