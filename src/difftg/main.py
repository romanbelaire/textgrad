"""DiffTG CLI.

Usage:
    python -m src.difftg.main <config.yaml> <output_dir>
"""
from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .base_lm import BaseLM
from .config import Config, load_config
from .diff_lm import build_diff_lm
from .difftg_step import difftg_step
from .reward import build_reward
from .span_select import build_selector, gen_text_has_answer_span_marker
from .tasks import build_tasks


def _seed_all(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    g = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    g.manual_seed(seed)
    return g


def _result_to_record(res) -> dict:
    return {
        "task_idx": res.task_idx,
        "prompt": res.traj.prompt,
        "gen_text_orig": res.traj.gen_text,
        "gen_text_final": res.final_gen_text,
        "R_orig": res.R_orig,
        "R_final": res.R_final,
        "delta_R_total": res.R_final - res.R_orig,
        "attempts": [asdict(a) for a in res.attempts],
    }


def _run_inference_loop(
    f,
    base_lm: BaseLM,
    diff_lm,
    selector,
    reward_fn,
    tasks: list,
    *,
    task_indices: list[int],
    trajs: list,
    g: torch.Generator,
    cfg: Config,
    r_orig_sum: float,
    r_final_sum: float,
    n_attempts: int,
    n_accepted: int,
    cos_sum: float,
    processed_count: int,
    total_tasks: int,
) -> tuple[float, float, int, int, float]:
    """Write one result per task; `trajs[k]` pairs with `tasks[k]` and `task_indices[k]`."""
    for k, task in enumerate(tasks):
        i = task_indices[k]
        res = difftg_step(
            base_lm=base_lm,
            diff_lm=diff_lm,
            selector=selector,
            reward_fn=reward_fn,
            task=task,
            langevin_cfg=cfg.langevin,
            num_spans=cfg.span_select.num_spans,
            task_idx=i,
            generator=g,
            traj=trajs[k],
        )
        f.write(json.dumps(_result_to_record(res)) + "\n")
        f.flush()
        r_orig_sum += res.R_orig
        r_final_sum += res.R_final
        for a in res.attempts:
            n_attempts += 1
            if a.accepted:
                n_accepted += 1
            cos_sum += a.cos_final
        processed_count += 1
        if processed_count % 10 == 0 or processed_count == total_tasks:
            print(
                f"[{processed_count}/{total_tasks}] "
                f"mean_R_orig={r_orig_sum / processed_count:.4f} "
                f"mean_R_final={r_final_sum / processed_count:.4f} "
                f"accept_rate={(n_accepted / max(n_attempts, 1)):.3f}",
                flush=True,
            )
    return r_orig_sum, r_final_sum, n_attempts, n_accepted, cos_sum


def run_inference(cfg: Config, out_dir: Path) -> None:
    g = _seed_all(cfg.task.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_lm = BaseLM(cfg.base_lm)
    diff_lm = build_diff_lm(cfg.diff_lm, base_lm)
    selector = build_selector(cfg.span_select, base_lm, cfg.task.seed)
    reward_fn = build_reward(cfg.reward)
    tasks = build_tasks(cfg.task)

    results_path = out_dir / "results.jsonl"
    summary_path = out_dir / "summary.json"

    t0 = time.time()
    r_orig_sum = 0.0
    r_final_sum = 0.0
    n_attempts = 0
    n_accepted = 0
    cos_sum = 0.0
    total_tasks = len(tasks)
    processed_count = 0

    with open(results_path, "w") as f:
        if cfg.span_select.kind == "answer":
            chunk = max(1, cfg.task.inference_chunk_size)
            offset = 0
            while offset < total_tasks:
                end = min(offset + chunk, total_tasks)
                chunk_tasks = tasks[offset:end]
                task_indices = list(range(offset, end))
                trajs = [base_lm.generate(t.prompt) for t in chunk_tasks]
                need_repair = [
                    trajs[j]
                    for j in range(len(trajs))
                    if not gen_text_has_answer_span_marker(trajs[j].gen_text)
                ]
                if need_repair:
                    base_lm.batch_repair_answer_markers(need_repair)
                r_orig_sum, r_final_sum, n_attempts, n_accepted, cos_sum = _run_inference_loop(
                    f,
                    base_lm,
                    diff_lm,
                    selector,
                    reward_fn,
                    chunk_tasks,
                    task_indices=task_indices,
                    trajs=trajs,
                    g=g,
                    cfg=cfg,
                    r_orig_sum=r_orig_sum,
                    r_final_sum=r_final_sum,
                    n_attempts=n_attempts,
                    n_accepted=n_accepted,
                    cos_sum=cos_sum,
                    processed_count=processed_count,
                    total_tasks=total_tasks,
                )
                processed_count += len(chunk_tasks)
                offset = end
        else:
            for i, task in enumerate(tasks):
                res = difftg_step(
                    base_lm=base_lm,
                    diff_lm=diff_lm,
                    selector=selector,
                    reward_fn=reward_fn,
                    task=task,
                    langevin_cfg=cfg.langevin,
                    num_spans=cfg.span_select.num_spans,
                    task_idx=i,
                    generator=g,
                )
                f.write(json.dumps(_result_to_record(res)) + "\n")
                f.flush()
                r_orig_sum += res.R_orig
                r_final_sum += res.R_final
                for a in res.attempts:
                    n_attempts += 1
                    if a.accepted:
                        n_accepted += 1
                    cos_sum += a.cos_final
                processed_count = i + 1
                if processed_count % 10 == 0 or processed_count == total_tasks:
                    print(
                        f"[{processed_count}/{total_tasks}] "
                        f"mean_R_orig={r_orig_sum / processed_count:.4f} "
                        f"mean_R_final={r_final_sum / processed_count:.4f} "
                        f"accept_rate={(n_accepted / max(n_attempts, 1)):.3f}",
                        flush=True,
                    )

    summary = {
        "num_tasks": len(tasks),
        "mean_R_orig": r_orig_sum / max(len(tasks), 1),
        "mean_R_final": r_final_sum / max(len(tasks), 1),
        "mean_delta_R": (r_final_sum - r_orig_sum) / max(len(tasks), 1),
        "num_attempts": n_attempts,
        "accept_rate": n_accepted / max(n_attempts, 1),
        "mean_cos_z0_zK": cos_sum / max(n_attempts, 1),
        "wall_seconds": time.time() - t0,
        "config": {
            "base_lm": asdict(cfg.base_lm),
            "diff_lm": asdict(cfg.diff_lm),
            "langevin": asdict(cfg.langevin),
            "span_select": asdict(cfg.span_select),
            "task": asdict(cfg.task),
            "reward": asdict(cfg.reward),
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {results_path} and {summary_path}", flush=True)


def run_training(cfg: Config, out_dir: Path) -> None:
    if cfg.training is None:
        raise ValueError("Training mode requires a `training:` config block.")
    from .trainer import run as trainer_run
    trainer_run(cfg, out_dir)


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 2:
        raise ValueError(f"Usage: python -m src.difftg.main <config.yaml> <output_dir>; got {argv}")
    cfg = load_config(argv[0])
    out_dir = Path(argv[1])
    if cfg.mode == "inference":
        run_inference(cfg, out_dir)
    elif cfg.mode == "training":
        run_training(cfg, out_dir)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode!r}")


if __name__ == "__main__":
    main()
