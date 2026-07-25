#!/usr/bin/env python3
"""Export unified Table R1 from logs/review/r1u_* into review_spec/rebuttal_results snippet."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = ROOT / "logs" / "review"

BASE_RE = re.compile(r"基线模型\(.*?\) -> MSE: ([0-9.]+), MAE: ([0-9.]+)")
CRC_RE = re.compile(r"CRC 纠偏后 -> MSE: ([0-9.]+), MAE: ([0-9.]+)")
NDR_RE = re.compile(r"NDR \(非恶化率\): ([0-9.]+)%")


def parse(path: Path) -> dict | None:
    t = path.read_text(errors="replace")
    b, c, n = BASE_RE.search(t), CRC_RE.search(t), NDR_RE.search(t)
    if not (b and c and n):
        return None
    return {
        "g0_mse": float(b.group(1)),
        "g0_mae": float(b.group(2)),
        "mse": float(c.group(1)),
        "mae": float(c.group(2)),
        "ndr": float(n.group(1)),
    }


def main() -> None:
    rows = {}
    for g in ("g1", "g2", "g3", "g4", "g5"):
        for b in ("b1", "b2", "b3"):
            p = LOG_ROOT / f"r1u_{g}_{b}" / "run.log"
            if p.exists():
                r = parse(p)
                if r:
                    rows[(g, b)] = r
    g0 = {b: rows.get(("g5", b), rows.get(("g1", b))) for b in ("b1", "b2", "b3")}
    for b in ("b1", "b2", "b3"):
        d = g0[b]
        if d:
            print(f"G0 {b.upper()} MAE={d['g0_mae']:.4f} MSE={d['g0_mse']:.4f}")
    for g in ("g1", "g2", "g3", "g4", "g5"):
        line = f"{g.upper()}"
        for b in ("b1", "b2", "b3"):
            d = rows.get((g, b))
            if d:
                line += f" | {b}: MAE={d['mae']:.4f} MSE={d['mse']:.4f} NDR={d['ndr']:.1f}%"
        print(line)


if __name__ == "__main__":
    main()
