"""Generate a v1 validation sweep.

Reads `configs/toy_inference.yaml` as the base, writes variants under
`configs/_sweep/` differing by (K, step_size, num_spans), and writes a
submission script `sbatch/_sweep_submit.sh` that `sbatch`es each variant.

The baseline K=0 variant is included: with K=0 the Langevin loop runs zero
iterations, so DiffTG should be an identity op on reward (sanity check).
"""
from __future__ import annotations

import itertools
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "configs" / "toy_inference.yaml"
OUT_CFG = ROOT / "configs" / "_sweep"
OUT_SBATCH = ROOT / "sbatch" / "_sweep_submit.sh"
OUT_RUNS = "outputs/toy_inference_sweep"  # relative to ROOT at run time

K_GRID = [0, 5, 20]
STEP_GRID = [0.005, 0.01, 0.05]
NSPANS_GRID = [1, 3]


def main() -> None:
    with open(BASE) as f:
        base = yaml.safe_load(f)

    OUT_CFG.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "#!/bin/bash",
        "set -euo pipefail",
        f"mkdir -p {OUT_RUNS}",
    ]
    for K, step, nspans in itertools.product(K_GRID, STEP_GRID, NSPANS_GRID):
        if K == 0 and (step != STEP_GRID[0] or nspans != NSPANS_GRID[0]):
            # K=0 is an identity op regardless of step_size / num_spans; only
            # emit one config for it.
            continue
        tag = f"K{K}_step{step}_n{nspans}"
        cfg = dict(base)
        cfg["langevin"] = dict(base["langevin"])
        cfg["langevin"]["K"] = K
        cfg["langevin"]["step_size"] = step
        cfg["span_select"] = dict(base["span_select"])
        cfg["span_select"]["num_spans"] = nspans
        cfg_path = OUT_CFG / f"{tag}.yaml"
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        out_dir = f"{OUT_RUNS}/{tag}"
        lines.append(
            f"sbatch sbatch/run_difftg.sh {cfg_path.relative_to(ROOT)} {out_dir}"
        )

    with open(OUT_SBATCH, "w") as f:
        f.write("\n".join(lines) + "\n")
    OUT_SBATCH.chmod(0o755)
    print(f"Wrote {len([p for p in OUT_CFG.glob('*.yaml')])} configs to {OUT_CFG}")
    print(f"Submit with: bash {OUT_SBATCH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
