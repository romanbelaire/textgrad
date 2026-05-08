"""Aggregate DiffTG sweep outputs into a single table.

Scans `outputs/toy_inference_sweep/*/summary.json` and prints a TSV to
stdout with one row per run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise ValueError("Usage: python scripts/aggregate_sweep.py <sweep_root>")
    root = Path(sys.argv[1])
    rows: list[dict] = []
    for summary_path in sorted(root.glob("*/summary.json")):
        with open(summary_path) as f:
            s = json.load(f)
        rows.append({
            "run": summary_path.parent.name,
            "K": s["config"]["langevin"]["K"],
            "step_size": s["config"]["langevin"]["step_size"],
            "num_spans": s["config"]["span_select"]["num_spans"],
            "num_tasks": s["num_tasks"],
            "R_orig": f"{s['mean_R_orig']:.4f}",
            "R_final": f"{s['mean_R_final']:.4f}",
            "delta_R": f"{s['mean_delta_R']:.4f}",
            "accept_rate": f"{s['accept_rate']:.3f}",
            "cos_z0_zK": f"{s['mean_cos_z0_zK']:.3f}",
        })
    if not rows:
        raise SystemExit(f"No summaries found under {root}")
    cols = list(rows[0].keys())
    print("\t".join(cols))
    for r in rows:
        print("\t".join(str(r[c]) for c in cols))


if __name__ == "__main__":
    main()
