#!/usr/bin/env python3
"""Parse logs/review/*/run.log for MAE/MSE/NDR lines."""
from __future__ import annotations

import re
import sys
from pathlib import Path

MAE_RE = re.compile(r"CRC 纠偏后 -> MSE: ([0-9.]+), MAE: ([0-9.]+)")
NDR_RE = re.compile(r"NDR \(非恶化率\): ([0-9.]+)%")


def parse_log(path: Path) -> dict | None:
    text = path.read_text(errors="replace")
    mae_m = MAE_RE.search(text)
    ndr_m = NDR_RE.search(text)
    if not mae_m or not ndr_m:
        return None
    mse, mae = float(mae_m.group(1)), float(mae_m.group(2))
    ndr = float(ndr_m.group(1))
    return {"mae": mae, "mse": mse, "ndr": ndr}


def main() -> None:
    root = Path(__file__).resolve().parents[2] / "logs" / "review"
    pattern = sys.argv[1] if len(sys.argv) > 1 else "*"
    rows = []
    for log in sorted(root.glob(f"{pattern}/run.log")):
        stats = parse_log(log)
        tag = log.parent.name
        if stats:
            rows.append((tag, stats))
        else:
            print(f"# skip (incomplete): {tag}", file=sys.stderr)
    print("tag\tMAE\tMSE\tNDR%")
    for tag, s in rows:
        print(f"{tag}\t{s['mae']:.4f}\t{s['mse']:.4f}\t{s['ndr']:.1f}")


if __name__ == "__main__":
    main()
