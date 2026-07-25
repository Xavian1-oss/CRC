Dear Reviewer vaUX,

Thank you for the review. We agree that the **“when to deploy”** framing is central, and that **missing explicit update policies** weakened the empirical story. We address this with **new controlled comparisons** (Table R1).

**Table R1 setup.** **B1–B3** (ETTh1/DLinear @96 & @336; Weather/TimesNet @96); **G1–G5 from one unified batch** (same protocol).

---

**Major suggestion — Never update / always update / selective update**

| Policy | Config | B1 MAE | B1 NDR | B2 NDR | B3 NDR |
|--------|--------|--------|--------|--------|--------|
| **Never update** | G0 (Table 1 baseline) | 0.4108 | — | — | — |
| **Always accept update** | G1 (always-on corrector) | **0.4738** | **8.2%** | **10.6%** | **33.9%** |
| **Selective update (CRC)** | G5 (Full CRC) | **0.4084** | **76.3%** | **77.7%** | **78.5%** |

**Table 1 baseline rows correspond to G0 (never update); CRC rows correspond to G5.** The missing comparison was **G1 (always-on correction)**. G1 shows that always accepting corrections **hurts both MAE and NDR**, supporting the paper’s motivation.

We also report **G2** and **G4** in the full ladder. **G2** (*same Stage-A as G5; ridge-only validation gate, no MLP*) reaches **97.9% / 97.1% / 90.0%** NDR but is **conservative**—**B1 MSE 0.3941 vs. G5 0.3880**; **B2 MAE 0.4693 vs. baseline 0.4654**. **G4** (light encoder, full blend) sits between G2 and G5 on NDR/MAE (full Table R1 in appendix).

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
