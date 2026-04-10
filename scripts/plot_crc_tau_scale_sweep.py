#!/usr/bin/env python3
"""
Plot CRC inference-time tau_scale sweep from a master.log.

Goal (paper-ready): tell a single story
  knob (tau_scale) -> mechanism (clip rate |r_delta| > tau) -> outcomes (MSE / NDR).

Expected log lines (from exp/exp_crc.py):
  - "CRC Tau-Scale Sweep: ... tau_scale=X"
  - "CRC 纠偏后 -> MSE: a.bcd, MAE: e.fgh"
  - "NDR (非恶化率): xx.x%"
  - "Correction coverage (clip rate |r_delta|>tau): yy.y%"

Usage:
  python scripts/plot_crc_tau_scale_sweep.py --log logs/tau_scale_sweep_xxx/master.log
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SweepRow:
    tau_scale: float
    mse: float
    mae: float
    ndr: float
    clip_cov: float


_RE_TAU = re.compile(r"tau_scale=([0-9]*\.?[0-9]+)")
_RE_MSE_MAE = re.compile(r"CRC\s+纠偏后\s+->\s+MSE:\s*([0-9.]+),\s*MAE:\s*([0-9.]+)")
_RE_NDR = re.compile(r"NDR\s*\(非恶化率\):\s*([0-9.]+)%")
_RE_CLIP = re.compile(r"Correction coverage\s*\(clip rate \|r_delta\|>tau\):\s*([0-9.]+)%")


def parse_master_log(path: Path) -> List[SweepRow]:
    tau: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    ndr: Optional[float] = None
    clip: Optional[float] = None
    rows: List[SweepRow] = []

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = _RE_TAU.search(line)
        if m and ("Tau-Scale Sweep" in line or "--tau_scale" in line):
            tau = float(m.group(1))
            mse = mae = ndr = clip = None
            continue

        if tau is None:
            continue

        m = _RE_MSE_MAE.search(line)
        if m:
            mse, mae = float(m.group(1)), float(m.group(2))
            continue

        m = _RE_NDR.search(line)
        if m:
            ndr = float(m.group(1))
            continue

        m = _RE_CLIP.search(line)
        if m:
            clip = float(m.group(1))
            continue

        # finalize a block as soon as all metrics are available
        if (tau is not None) and (mse is not None) and (mae is not None) and (ndr is not None) and (clip is not None):
            rows.append(SweepRow(tau_scale=tau, mse=mse, mae=mae, ndr=ndr, clip_cov=clip))
            tau = None

    return rows


def _paper_style():
    mpl.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
        }
    )


def make_figure(rows: List[SweepRow], title: str):
    rows = sorted(rows, key=lambda r: r.tau_scale)
    x = np.array([r.tau_scale for r in rows], dtype=np.float64)
    mse = np.array([r.mse for r in rows], dtype=np.float64)
    ndr = np.array([r.ndr for r in rows], dtype=np.float64)
    clip = np.array([r.clip_cov for r in rows], dtype=np.float64)

    _paper_style()
    fig = plt.figure(figsize=(6.8, 4.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.4], hspace=0.18)

    # Panel A: outcomes
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title(title)
    ax1.plot(x, mse, marker="o", color="tab:blue", label="Accuracy (MSE, lower is better)")
    ax1.set_xscale("log", base=2)
    ax1.set_ylabel("MSE")
    ax1.set_xticklabels([])  # shared x

    # annotate best MSE point
    best_i = int(np.argmin(mse))
    ax1.scatter([x[best_i]], [mse[best_i]], color="tab:blue", zorder=5)
    ax1.annotate(
        f"best MSE @ {x[best_i]:g}",
        xy=(x[best_i], mse[best_i]),
        xytext=(8, -18),
        textcoords="offset points",
        fontsize=9,
        color="tab:blue",
        arrowprops=dict(arrowstyle="->", lw=0.8, color="tab:blue", alpha=0.8),
    )

    ax1b = ax1.twinx()
    ax1b.plot(x, ndr, marker="s", color="tab:orange", label="Safety proxy (NDR %, higher is better)")
    ax1b.set_ylabel("NDR (%)")
    ax1b.set_ylim(min(80.0, float(ndr.min()) - 1.0), min(100.0, float(ndr.max()) + 1.0))

    # merged legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True)

    # Panel B: mechanism
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(x, clip, marker="^", color="tab:green", label="Mechanism: clip rate (|rδ| > τ)")
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Inference knob: tau_scale (log2 scale)")
    ax2.set_ylabel("Clip rate (%)")
    ax2.set_ylim(0.0, min(100.0, float(clip.max()) + 5.0))
    ax2.legend(loc="upper right", frameon=True)

    # subtle causal arrow to link panels
    ax2.annotate(
        "larger tau_scale → larger τ → fewer clips",
        xy=(x[-1], clip[-1]),
        xytext=(-10, 10),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color="tab:green",
    )

    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, required=True, help="Path to tau_scale sweep master.log")
    ap.add_argument("--out", type=str, default="", help="Output prefix (without extension). Defaults next to log.")
    ap.add_argument("--title", type=str, default="CRC inference-time τ scaling: knob → mechanism → outcomes")
    args = ap.parse_args()

    log_path = Path(args.log)
    rows = parse_master_log(log_path)
    if not rows:
        raise SystemExit(f"No sweep rows parsed from: {log_path}")

    # Default: save figures to repo root (Time-series-library) to keep outputs centralized.
    # Include the sweep folder name to avoid overwriting across runs.
    if args.out:
        out_prefix = Path(args.out)
    else:
        sweep_tag = log_path.parent.name
        out_prefix = Path.cwd() / f"crc_tau_scale_sweep_{sweep_tag}"
    fig = make_figure(rows, title=args.title)
    fig.savefig(str(out_prefix) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out_prefix) + ".png", bbox_inches="tight")
    print(f"Saved: {out_prefix}.pdf")
    print(f"Saved: {out_prefix}.png")


if __name__ == "__main__":
    main()

