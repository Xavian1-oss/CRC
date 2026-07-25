# R14-E2 — CRC vs PIR paired analysis (internal logs)

- **Paired cells:** **64** = **ETTh1, ETTh2, ETTm1, ETTm2** × 4 backbones × horizons {96, 192, 336, 720} (per-horizon cells; **not** the same aggregation as Table 3’s 28 horizon-averaged dataset–backbone pairs and **excludes** Weather / Electricity / Traffic).
- **Source:** Jul 2026 **paired rerun** (external server). Per-cell tables deferred to **camera-ready / appendix** once rerun CSVs are synced to the repo (`scripts/review/pir_crc_significance.py --summary-only`). Until sync, treat Wilcoxon rows as **provisional**; repo default CSVs (same 64-cell grid) may differ—recompute after CSV upload.

## Aggregate (PIR − CRC; **positive ⇒ CRC lower** MAE/MSE)

| Metric | mean Δ | bootstrap 95% CI | Wilcoxon p (H1: CRC better) | Sign test p | CRC wins / N |
|--------|--------|------------------|-----------------------------|-------------|--------------|
| MAE | +0.0103 | [+0.0027, +0.0195] | 0.074 | 0.030 | **40/64** |
| MSE | +0.0120 | [+0.0003, +0.0281] | 0.003 | 0.008 | **42/64** |

> **Note:** Wilcoxon *p*≈0.45 in an earlier draft came from the **old** 34/64 pairing (mean Δ<0). For 40/42 wins with mean Δ>0, *p* should be **much smaller**. Sign-test *p* (0.030 / 0.008) is exact from win counts; Wilcoxon rows are **provisional** until rerun CSVs are synced (`pir_crc_significance.py --summary-only`).
