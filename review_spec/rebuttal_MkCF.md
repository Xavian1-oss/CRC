# Rebuttal to Reviewer MkCF — Draft

> **Status**: Draft v0.3 — **Table R1** + Limitations / revision promises

---

Dear Reviewer MkCF,

Thank you for the detailed review and for highlighting **when to deploy** as the core contribution. We respond to each weakness below.

---

## W1 — Validation–test mismatch

We agree deployment rules are calibrated on validation and **may fail under distribution shift**. We will **expand Limitations** prominently.

**Evidence:** Our validation-ratio sensitivity analysis (paper) shows reasonable stability under varying val fractions; under mismatch, CRC tends toward **conservative gates**—high NDR but reduced MAE gain. Example from new Table R1: **G2 (ridge+val gate)** on B2 achieves **94.8% NDR** but **worse MAE than baseline** (0.4725 vs. 0.4654)—illustrating safe-but-conservative behavior.

**Recalibration (Q1):** When deployment context changes, recalibrate by **retraining stage-2 only** (frozen backbone): new validation slice via `--crc_val_ratio`, or new `--crc_tau_quantile`. Frequency: **whenever the operational validation window changes** (e.g., new season/month of held-out data)—not every test batch.

---

## W2 — KNN graph (K=5)

The adjacency uses **training-set correlation** with top-K neighbors (default K=5 via `--top_k`). It captures **static linear co-movement**, not dynamic lags—We will state this clearly as a **limitation**.

**Revision / RMK-E1:** we will report sensitivity for **K ∈ {3, 5, 7}** on ETTh1/B1 (Full CRC).

---

## W3 — Online / strong drift benchmarks

We agree standard ETT/Weather/Electricity/Traffic splits do not stress **online drift**. We scope CRC as **post-hoc correction under frozen backbones** with a single calibration val window; **online recalibration** is future work (discussed in Limitations).

---

## W4 — Hyperparameters / migration cost

CRC is two-stage: (1) train/warm-start backbone once; (2) train corrector on val predictions. Table R1 shows **simpler paths** (G2 ridge-gate, G4 deploy-only) for ablation; production defaults follow paper. We will add a **migration checklist** in appendix (backbone ckpt + `crc_corrector_val*.pth`).

---

## Q3 — NDR vs MSE/MAE

See shared practitioner guide in gnqC rebuttal: **NDR** = local non-degradation rate; **MAE/MSE** = global average. High NDR does not imply maximum MAE reduction (G2 vs G5 in Table R1).

---

## Limitations (your comment)

We will substantially expand Limitations (val–test, static graph, drift, hyperparameters).

Thank you again.

---

## 内部备注

- **RMK-E1** top_k sweep on B1: K∈{3,5,7} — MAE 0.4079–0.4096, NDR 74–80%; default K=5 near-best (`rebuttal_results.md` §top_k)
