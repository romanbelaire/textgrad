"""Pretrain a `TrainableEmbedDenoiser` via denoising score-matching over
spans drawn from BaseLM rollouts.

Workflow:
    1. Build BaseLM + a fresh TrainableEmbedDenoiser.
    2. Collect a pool of (normalized) span embeddings by repeatedly sampling
       tasks, generating rollouts, and extracting random-sentence spans.
    3. For `num_steps` steps: sample a batch of clean spans, sample a random
       `t` and sigma, compute the score-matching loss, step.
    4. Save the denoiser state_dict.

Config example: `configs/pretrain_diff_lm.yaml`.
"""
from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from src.difftg.base_lm import BaseLM
from src.difftg.config import BaseLMConfig, DiffLMConfig, SpanSelectConfig, TaskConfig, _build
from src.difftg.diff_lm import TrainableEmbedDenoiser, build_diff_lm
from src.difftg.encoder import encode_span
from src.difftg.span_select import build_selector
from src.difftg.tasks import build_tasks


@dataclass(frozen=True)
class PretrainConfig:
    num_pool_rollouts: int
    num_steps: int
    batch_size: int
    lr: float
    max_span_tokens: int
    sigma_min: float
    sigma_max: float
    t_max: int
    ckpt_path: str


def _load(path: str):
    with open(path) as f:
        raw = yaml.safe_load(f)
    return (
        _build(BaseLMConfig, raw["base_lm"]),
        _build(DiffLMConfig, raw["diff_lm"]),
        _build(SpanSelectConfig, raw["span_select"]),
        _build(TaskConfig, raw["task"]),
        _build(PretrainConfig, raw["pretrain"]),
    )


def _collect_pool(base_lm, selector, tasks, max_span_tokens: int) -> list[torch.Tensor]:
    pool: list[torch.Tensor] = []
    for task in tasks:
        traj = base_lm.generate(task.prompt)
        spans = selector(traj, 0.0, 4)
        for (s, e) in spans:
            z = encode_span(base_lm, traj.gen_text[s:e])
            if z.size(0) < 2 or z.size(0) > max_span_tokens:
                continue
            pool.append(z.detach())
    if not pool:
        raise ValueError("Collected empty span pool; check task/selector config.")
    return pool


def _pad_batch(spans: list[torch.Tensor], pad_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad to [B, pad_len, d]; mask is [B, pad_len] with 1 for real tokens."""
    B = len(spans)
    d = spans[0].size(-1)
    device = spans[0].device
    batch = torch.zeros(B, pad_len, d, device=device, dtype=spans[0].dtype)
    mask = torch.zeros(B, pad_len, device=device, dtype=spans[0].dtype)
    for i, z in enumerate(spans):
        L = min(z.size(0), pad_len)
        batch[i, :L] = z[:L]
        mask[i, :L] = 1.0
    return batch, mask


def main() -> None:
    if len(sys.argv) != 3:
        raise ValueError("Usage: python scripts/pretrain_diff_lm.py <config.yaml> <output_dir>")
    cfg_path, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    base_cfg, diff_cfg, sel_cfg, task_cfg, pre_cfg = _load(cfg_path)

    if diff_cfg.kind != "trainable":
        raise ValueError("pretrain_diff_lm requires diff_lm.kind == 'trainable'.")

    torch.manual_seed(task_cfg.seed)
    random.seed(task_cfg.seed)

    base_lm = BaseLM(base_cfg)
    diff_lm = build_diff_lm(diff_cfg, base_lm)
    if not isinstance(diff_lm, TrainableEmbedDenoiser):
        raise TypeError(f"Expected TrainableEmbedDenoiser, got {type(diff_lm).__name__}.")

    selector = build_selector(sel_cfg, base_lm, task_cfg.seed)
    # Capped to num_pool_rollouts for data collection.
    sub_task_cfg = TaskConfig(
        kind=task_cfg.kind,
        num_tasks=pre_cfg.num_pool_rollouts,
        seed=task_cfg.seed,
        variant=task_cfg.variant,
        split=task_cfg.split,
    )
    tasks = build_tasks(sub_task_cfg)

    print(f"Collecting span pool from {len(tasks)} rollouts ...", flush=True)
    pool = _collect_pool(base_lm, selector, tasks, pre_cfg.max_span_tokens)
    print(f"Collected {len(pool)} spans.", flush=True)

    opt = torch.optim.Adam(diff_lm.parameters(), lr=pre_cfg.lr)
    diff_lm.train()

    log_path = out_dir / "pretrain.jsonl"
    t0 = time.time()
    with open(log_path, "w") as log_f:
        for step in range(pre_cfg.num_steps):
            idx = [random.randrange(len(pool)) for _ in range(pre_cfg.batch_size)]
            spans = [pool[i] for i in idx]
            pad_len = max(z.size(0) for z in spans)
            z0, mask = _pad_batch(spans, pad_len)

            t = random.randrange(pre_cfg.t_max)
            # log-uniform sigma schedule
            u = random.random()
            sigma = pre_cfg.sigma_min * (pre_cfg.sigma_max / pre_cfg.sigma_min) ** u

            eps = torch.randn_like(z0)
            z_noisy = F.normalize(z0 + sigma * eps, dim=-1)
            target = -(z_noisy - z0) / (sigma * sigma)
            pred = diff_lm(z_noisy, t)
            err = (pred - target) * mask.unsqueeze(-1)
            loss = (err.pow(2).sum()) / (mask.sum() * z0.size(-1))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diff_lm.parameters(), 1.0)
            opt.step()

            if step % 50 == 0 or step + 1 == pre_cfg.num_steps:
                rec = {"step": step, "loss": float(loss.item()), "t": t, "sigma": sigma}
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
                print(
                    f"[{step+1}/{pre_cfg.num_steps}] loss={loss.item():.5f} "
                    f"t={t} sigma={sigma:.4f}",
                    flush=True,
                )

    ckpt_path = Path(pre_cfg.ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(diff_lm.state_dict(), ckpt_path)
    summary = {
        "ckpt": str(ckpt_path),
        "pool_size": len(pool),
        "num_steps": pre_cfg.num_steps,
        "wall_seconds": time.time() - t0,
        "pretrain_cfg": asdict(pre_cfg),
        "diff_lm_cfg": asdict(diff_cfg),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved ckpt to {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
