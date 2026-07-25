Dear Reviewer vaUX,

Thank you for the review. We agree that the **“when to deploy”** framing is central, and that **missing explicit update policies** weakened the empirical story. We address this with **new controlled comparisons** (Table R1).

**Table R1 setup.** **B1–B3** (ETTh1/DLinear @96 & @336; Weather/TimesNet @96); **G1–G5 from one unified batch** (same protocol).

---

**Major suggestion — Never update / always update / selective update**

| Policy | Config | B1 MAE | B1 NDR | B2 NDR | B3 NDR |
|--------|--------|--------|--------|--------|--------|
| **Never update** | G0 (Table 1 baseline) | 0.4108 | — | — | — |
| **Always accept update** | G1 (always-on corrector) | **0.4738** | **8.2%** | **10.6%** | **33.9%** |
| **Selective update (CRC)** | G5 (Full CRC) | **0.3914** | **80.3%** | **79.7%** | **78.5%** |

**Conceptually**, Table 1 **baseline / CRC** rows are **G0 (never update)** and **G5 (Full CRC)**; Table R1 **instantiates** that comparison on **B1–B3** (fixed dataset–backbone–**single pred_len** each—not the horizon-averaged cells in Table 1, so MAE/MSE **need not match** Table 1 numerically). The missing control was **G1**: same corrector stack but **no per-(variable, horizon) blend calibration** (`w1=w2=1` everywhere; τ / direction gate still on). G1 shows that **always deploying** the corrector **hurts both MAE and NDR**, supporting the paper’s motivation.

We also report **G2** and **G4** in the full ladder. **G2** (*same Stage-A as G5; ridge-only validation gate, no MLP*) reaches **97.9% / 97.1% / 90.0%** NDR—high safety—but **cannot match G5 on global MSE** (**B1 MSE 0.3941 vs. G5 0.3780**; **B2** G2 MAE **0.4693** vs. baseline **0.4654**, G5 **0.4392**). **G4** (light encoder, full blend) sits between G2 and G5 on NDR; **G5** is **best on B3 MAE/MSE** (full Table R1 in appendix).

---

**Background / datasets / anomalies**

We agree the paper should better ground tasks and backbones. **Revision:** expand dataset descriptions (ETT hourly/minute, Weather multivariate, etc.), clarify that benchmarks use **standard cleaned splits** (not heavy OOD stress tests), and discuss sensor anomalies / drift in Limitations.

---

**NDR missing in Table 1**

We agree. **Revision:** add NDR to the main table or appendix. In rebuttal we report NDR alongside MAE/MSE for B1–B3 in Table R1 (see above).

---

**Held-out calibration set**

CRC uses the **same chronological validation split as Time-Series-Library** long-term forecasting (no test leakage). Stage-2 calibrates blend weights and τ on validation predictions only. The validation fraction for corrector training follows our sensitivity analysis in the paper. The backbone remains **frozen**; cost is one stage-2 pass after backbone training.

---

**Minor (PIR, structure, limitations)**

We will expand PIR in Related Work, add a dedicated **Results** subsection after §4.2, and strengthen Limitations per your checklist.

---

Thank you again for the constructive feedback.
