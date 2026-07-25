#!/usr/bin/env python3
"""Parse unified Table R1 logs and check rebuttal claim support."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = ROOT / "logs" / "review"
OUT_MD = ROOT / "review_spec" / "table_r1_unified_verification.md"

BASE_RE = re.compile(
    r"基线模型\(.*?\) -> MSE: ([0-9.]+), MAE: ([0-9.]+)"
)
CRC_RE = re.compile(r"CRC 纠偏后 -> MSE: ([0-9.]+), MAE: ([0-9.]+)")
NDR_RE = re.compile(r"NDR \(非恶化率\): ([0-9.]+)%")


def parse_log(path: Path) -> dict | None:
    text = path.read_text(errors="replace")
    b = BASE_RE.search(text)
    c = CRC_RE.search(text)
    n = NDR_RE.search(text)
    if not b or not c or not n:
        return None
    return {
        "g0_mse": float(b.group(1)),
        "g0_mae": float(b.group(2)),
        "mse": float(c.group(1)),
        "mae": float(c.group(2)),
        "ndr": float(n.group(1)),
    }


def load_unified() -> dict[tuple[str, str], dict]:
    data: dict[tuple[str, str], dict] = {}
    for g in ("g1", "g2", "g3", "g4", "g5"):
        for b in ("b1", "b2", "b3"):
            tag = f"r1u_{g}_{b}"
            p = LOG_ROOT / tag / "run.log"
            if not p.exists():
                continue
            stats = parse_log(p)
            if stats:
                data[(g, b)] = stats
    return data


def check(label: str, ok: bool, detail: str) -> tuple[bool, str]:
    mark = "PASS" if ok else "FAIL"
    return ok, f"- [{mark}] **{label}**: {detail}"


def main() -> int:
    data = load_unified()
    expected = 15
    if len(data) < expected:
        print(f"Incomplete: {len(data)}/{expected} runs parsed.", file=sys.stderr)
        for g in ("g1", "g2", "g3", "g4", "g5"):
            for b in ("b1", "b2", "b3"):
                if (g, b) not in data:
                    print(f"  missing r1u_{g}_{b}", file=sys.stderr)

    lines = [
        "# Table R1 unified verification",
        "",
        f"Parsed **{len(data)}** runs from `logs/review/r1u_*`.",
        "",
        "## Metrics (G0 from baseline row in each log)",
        "",
        "| Config | B1 MAE | B1 MSE | B1 NDR | B2 MAE | B2 MSE | B2 NDR | B3 MAE | B3 MSE | B3 NDR |",
        "|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|",
    ]

    def cell(g: str, b: str, field: str) -> str:
        d = data.get((g, b))
        if not d:
            return "—"
        if field == "g0_mae" and b == "b1":
            return f"{d['g0_mae']:.4f}"
        if field in d:
            v = d[field]
            if field == "ndr":
                return f"{v:.1f}%"
            return f"{v:.4f}"
        return "—"

    row_g0 = ["**G0**"]
    for b in ("b1", "b2", "b3"):
        d = data.get(("g5", b)) or data.get(("g1", b))
        if d:
            row_g0.extend([f"{d['g0_mae']:.4f}", f"{d['g0_mse']:.4f}", "—"])
        else:
            row_g0.extend(["—", "—", "—"])
    lines.append("| " + " | ".join(row_g0) + " |")

    for g, glab in [
        ("g1", "G1 always-on"),
        ("g2", "G2 ridge+gate"),
        ("g3", "G3 no dir"),
        ("g4", "G4 deploy-only"),
        ("g5", "G5 Full CRC"),
    ]:
        row = [f"**{glab}**"]
        for b in ("b1", "b2", "b3"):
            d = data.get((g, b))
            if d:
                row.extend([f"{d['mae']:.4f}", f"{d['mse']:.4f}", f"{d['ndr']:.1f}%"])
            else:
                row.extend(["—", "—", "—"])
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "## Claim checks (rebuttal narratives)", ""])
    checks: list[tuple[bool, str]] = []

    for b in ("b1", "b2", "b3"):
        g1, g5 = data.get(("g1", b)), data.get(("g5", b))
        if g1 and g5:
            checks.append(
                check(
                    f"G1 vs G5 NDR+MAE ({b.upper()})",
                    g1["ndr"] < g5["ndr"] - 20 and g1["mae"] > g5["mae"] + 0.02,
                    f"G1 NDR={g1['ndr']:.1f}% MAE={g1['mae']:.4f} vs G5 NDR={g5['ndr']:.1f}% MAE={g5['mae']:.4f}",
                )
            )

    g2b2, g0b2 = data.get(("g2", "b2")), data.get(("g5", "b2"))
    if g2b2 and g0b2:
        checks.append(
            check(
                "G2 conservative on B2 (MAE worse than G0)",
                g2b2["mae"] > g0b2["g0_mae"],
                f"G2 MAE={g2b2['mae']:.4f} vs G0={g0b2['g0_mae']:.4f}, NDR={g2b2['ndr']:.1f}%",
            )
        )

    g5b1, g2b1 = data.get(("g5", "b1")), data.get(("g2", "b1"))
    if g5b1 and g2b1:
        checks.append(
            check(
                "G5 vs G2 B1 MSE (Full CRC stronger global MSE)",
                g5b1["mse"] < g2b1["mse"] - 0.005,
                f"G5 MSE={g5b1['mse']:.4f} vs G2 MSE={g2b1['mse']:.4f} (MAE: G5={g5b1['mae']:.4f} G2={g2b1['mae']:.4f})",
            )
        )
        checks.append(
            check(
                "G2 higher NDR than G5 on B1",
                g2b1["ndr"] > g5b1["ndr"] + 10,
                f"G2 NDR={g2b1['ndr']:.1f}% vs G5={g5b1['ndr']:.1f}%",
            )
        )

    g4b1, g5b1 = data.get(("g4", "b1")), data.get(("g5", "b1"))
    if g4b1 and g5b1:
        checks.append(
            check(
                "G5 NDR > G4 on B1 (representation helps deployment)",
                g5b1["ndr"] > g4b1["ndr"] + 5,
                f"G5 NDR={g5b1['ndr']:.1f}% vs G4={g4b1['ndr']:.1f}%",
            )
        )

    g3b2, g5b2 = data.get(("g3", "b2")), data.get(("g5", "b2"))
    if g3b2 and g5b2:
        checks.append(
            check(
                "G5 NDR > G3 on B2 (dir gate stabilizer)",
                g5b2["ndr"] > g3b2["ndr"],
                f"G5 NDR={g5b2['ndr']:.1f}% vs G3={g3b2['ndr']:.1f}%",
            )
        )

    g5b1_g0 = data.get(("g5", "b1"))
    if g5b1_g0:
        checks.append(
            check(
                "G5 B1 MSE better than G0",
                g5b1_g0["mse"] < g5b1_g0["g0_mse"] - 0.005,
                f"G5 MSE={g5b1_g0['mse']:.4f} vs G0={g5b1_g0['g0_mse']:.4f}",
            )
        )

    all_ok = True
    for ok, line in checks:
        lines.append(line)
        if not ok:
            all_ok = False

    lines.extend(
        [
            "",
            f"**Overall narrative support:** {'YES' if all_ok and len(data) >= expected else 'PARTIAL — see FAIL lines'}",
            "",
        ]
    )

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_MD}")
    print("\n".join(l for _, l in checks))
    return 0 if all_ok and len(data) >= expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
