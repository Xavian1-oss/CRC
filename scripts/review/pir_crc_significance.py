#!/usr/bin/env python3
"""
R14-E2 — Paired CRC vs PIR on corrected MAE/MSE from batch result CSVs.
Joins on (dataset, backbone, horizon) where backbone is normalized (dlinear, timesnet, ...).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def norm_model(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"^pir_", "", s)
    s = re.sub(r"^crc_", "", s)
    return s


def load_crc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=str.strip)
    df["model"] = df["model"].map(norm_model)
    df["dataset"] = df["dataset"].str.strip()
    df = df.rename(
        columns={
            "corrected_mae": "crc_mae",
            "corrected_mse": "crc_mse",
        }
    )
    return df[["dataset", "model", "horizon", "crc_mae", "crc_mse"]]


def load_pir(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=str.strip)
    df["model"] = df["model"].map(norm_model)
    df["dataset"] = df["dataset"].str.strip()
    df = df.rename(
        columns={
            "corrected_mae": "pir_mae",
            "corrected_mse": "pir_mse",
        }
    )
    return df[["dataset", "model", "horizon", "pir_mae", "pir_mse"]]


def bootstrap_ci(diffs: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(diffs)
    if n == 0:
        return float("nan"), float("nan")
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means.append(float(np.mean(diffs[idx])))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--crc-csv",
        type=Path,
        default=Path("logs/multi_20251119_180450/results_summary_detailed.csv"),
    )
    ap.add_argument(
        "--pir-csv",
        type=Path,
        default=Path("logs/multi_PIR_20251119_170802/results_summary_detailed.csv"),
    )
    ap.add_argument("--out", type=Path, default=Path("review_spec/r14_e2_pir_crc_significance.md"))
    args = ap.parse_args()

    crc = load_crc(args.crc_csv)
    pir = load_pir(args.pir_csv)
    merged = crc.merge(pir, on=["dataset", "model", "horizon"], how="inner")
    if merged.empty:
        raise SystemExit("No overlapping (dataset, model, horizon) rows between CRC and PIR CSVs.")

    merged["delta_mae"] = merged["pir_mae"] - merged["crc_mae"]  # >0 => CRC better (lower MAE)
    merged["delta_mse"] = merged["pir_mse"] - merged["crc_mse"]

    mae_diffs = merged["delta_mae"].to_numpy()
    mse_diffs = merged["delta_mse"].to_numpy()
    wilcox_mae = stats.wilcoxon(mae_diffs, alternative="greater") if len(mae_diffs) else None
    wilcox_mse = stats.wilcoxon(mse_diffs, alternative="greater") if len(mse_diffs) else None
    mae_ci = bootstrap_ci(mae_diffs)
    mse_ci = bootstrap_ci(mse_diffs)

    n_crc_better_mae = int((merged["delta_mae"] > 0).sum())
    n_total = len(merged)

    lines = [
        "# R14-E2 — CRC vs PIR paired analysis (internal logs)",
        "",
        f"- CRC CSV: `{args.crc_csv}`",
        f"- PIR CSV: `{args.pir_csv}`",
        f"- Paired cells: **{n_total}** (dataset × backbone × horizon)",
        "",
        "## Aggregate (PIR MAE − CRC MAE; positive ⇒ CRC lower MAE)",
        "",
        f"| Metric | mean Δ | bootstrap 95% CI | Wilcoxon p (CRC better) | CRC wins / N |",
        f"|--------|--------|------------------|-------------------------|--------------|",
    ]
    if wilcox_mae:
        lines.append(
            f"| MAE | {mae_diffs.mean():+.4f} | [{mae_ci[0]:+.4f}, {mae_ci[1]:+.4f}] | "
            f"{wilcox_mae.pvalue:.4g} | {n_crc_better_mae}/{n_total} |"
        )
    if wilcox_mse:
        n_crc_better_mse = int((merged["delta_mse"] > 0).sum())
        lines.append(
            f"| MSE | {mse_diffs.mean():+.4f} | [{bootstrap_ci(mse_diffs)[0]:+.4f}, "
            f"{bootstrap_ci(mse_diffs)[1]:+.4f}] | {wilcox_mse.pvalue:.4g} | "
            f"{n_crc_better_mse}/{n_total} |"
        )
    lines.extend(["", "## Per-cell ΔMAE (PIR − CRC)", ""])
    show = merged[
        ["dataset", "model", "horizon", "crc_mae", "pir_mae", "delta_mae", "delta_mse"]
    ]
    lines.append(show.to_csv(index=False))
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out} ({n_total} pairs)")


if __name__ == "__main__":
    main()
