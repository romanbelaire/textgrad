"""v2 training loop.

Implements spec sec. 4:
    4.1  REINFORCE on BaseLM span logits with `delta_R` as credit (LoRA).
    4.2  Optional: fine-tune DiffLM so it reproduces accepted moves.
    4.3  Optional: train a span-level critic in embedding space.

Driven by `configs/toy_train.yaml`. Entry point: `run(cfg, out_dir)`.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

from .base_lm import BaseLM, Trajectory
from .config import Config
from .diff_lm import TrainableEmbedDenoiser, build_diff_lm
from .encoder import decode_span, encode_span, replace_span
from .langevin import run_langevin
from .reward import build_reward
from .span_select import build_selector
from .tasks import build_tasks


class SpanCritic(nn.Module):
    """Tiny MLP on mean-pooled z0 predicting delta_R."""

    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z0: torch.Tensor) -> torch.Tensor:
        # z0: [L, d] -> scalar
        pooled = z0.mean(dim=0)
        return self.net(pooled).squeeze(-1)


def _wrap_lora(base_lm: BaseLM, cfg: Config) -> None:
    tr = cfg.training
    lora = LoraConfig(
        r=tr.lora_r,
        lora_alpha=tr.lora_alpha,
        lora_dropout=tr.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    base_lm.model = get_peft_model(base_lm.model, lora)
    base_lm.model.print_trainable_parameters()


def _seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    g = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    g.manual_seed(seed)
    return g


def run(cfg: Config, out_dir: Path) -> None:
    if cfg.training is None:
        raise ValueError("Trainer requires a `training:` config block.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tr = cfg.training
    g = _seed(cfg.task.seed)

    base_lm = BaseLM(cfg.base_lm)
    diff_lm = build_diff_lm(cfg.diff_lm, base_lm)
    selector = build_selector(cfg.span_select, base_lm, cfg.task.seed)
    reward_fn = build_reward(cfg.reward)
    tasks = build_tasks(cfg.task)
    if len(tasks) < tr.num_outer_steps:
        raise ValueError(
            f"num_outer_steps ({tr.num_outer_steps}) > task pool ({len(tasks)}); "
            f"increase task.num_tasks."
        )

    if tr.train_base_lm:
        _wrap_lora(base_lm, cfg)
        base_lm.model.train()

    params: list[torch.nn.Parameter] = []
    if tr.train_base_lm:
        params += [p for p in base_lm.model.parameters() if p.requires_grad]

    critic: SpanCritic | None = None
    if tr.train_critic:
        d = base_lm.embedding_matrix.shape[-1]
        critic = SpanCritic(d=d, hidden=tr.critic_hidden).to(base_lm.device)
        params += list(critic.parameters())

    if tr.train_diff_lm:
        if not isinstance(diff_lm, TrainableEmbedDenoiser):
            raise TypeError(
                "train_diff_lm=True requires diff_lm.kind='trainable' "
                f"(got {type(diff_lm).__name__})."
            )
        diff_lm.train()
        params += list(diff_lm.parameters())

    if not params:
        raise ValueError("Trainer has zero trainable parameters; check training flags.")
    opt = torch.optim.Adam(params, lr=tr.lr)

    log_path = out_dir / "train.jsonl"
    log_f = open(log_path, "w")
    t0 = time.time()
    n_accepted = 0
    n_attempts = 0

    for step in range(tr.num_outer_steps):
        task = tasks[step]
        with torch.no_grad():
            traj = base_lm.generate(task.prompt)
        R_orig = float(reward_fn(traj.gen_text, task))
        try:
            spans = selector(traj, R_orig, cfg.span_select.num_spans)
        except ValueError:
            continue

        opt.zero_grad(set_to_none=True)
        total_loss = torch.tensor(0.0, device=base_lm.device)
        step_records: list[dict] = []

        for (s, e) in spans:
            with torch.no_grad():
                z0 = encode_span(base_lm, traj.gen_text[s:e])
                z_k, _info = run_langevin(z0, diff_lm, cfg.langevin, generator=g)
                new_span, _ = decode_span(base_lm, z_k)
                cand_text = replace_span(traj.gen_text, s, e, new_span)
                R_new = float(reward_fn(cand_text, task))
                delta_R = R_new - R_orig

            n_attempts += 1
            if R_new >= R_orig:
                n_accepted += 1

            # 4.1 Policy REINFORCE on BaseLM span logits.
            if tr.train_base_lm and delta_R > 0:
                cand_traj = Trajectory(
                    prompt=traj.prompt, chat_prompt=traj.chat_prompt, gen_text=cand_text
                )
                new_char_e = s + len(new_span)
                log_prob = base_lm.logprob_span(cand_traj, s, new_char_e)
                total_loss = total_loss - delta_R * log_prob

            # 4.2 DiffLM self-distillation: push DiffLM's one-step sample toward z_k.
            if tr.train_diff_lm and delta_R > 0:
                # DiffLM.sample(z0) defined as z0 + score(z0, 0); cosine distance.
                score0 = diff_lm.score(z0, 0)
                sampled = F.normalize(z0 + score0, dim=-1)
                cos = (sampled * z_k.detach()).sum(dim=-1)
                total_loss = total_loss + (1.0 - cos).mean()

            # 4.3 Span critic regression.
            if tr.train_critic:
                pred = critic(z0)
                total_loss = total_loss + (pred - float(delta_R)) ** 2

            step_records.append({
                "span": [s, e],
                "delta_R": delta_R,
                "accepted": R_new >= R_orig,
                "orig": traj.gen_text[s:e],
                "new": new_span,
            })

        if total_loss.requires_grad:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, tr.grad_clip)
            opt.step()

        rec = {
            "step": step,
            "R_orig": R_orig,
            "loss": float(total_loss.item()) if total_loss.requires_grad else 0.0,
            "accept_rate_running": n_accepted / max(n_attempts, 1),
            "spans": step_records,
        }
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        if step % 20 == 0 or step + 1 == tr.num_outer_steps:
            print(
                f"[{step+1}/{tr.num_outer_steps}] loss={rec['loss']:.4f} "
                f"R_orig={R_orig:.3f} accept={rec['accept_rate_running']:.3f}",
                flush=True,
            )

        if tr.save_every > 0 and (step + 1) % tr.save_every == 0:
            _save(base_lm, diff_lm, critic, out_dir / f"ckpt_step{step+1}", tr)

    log_f.close()
    _save(base_lm, diff_lm, critic, out_dir / "ckpt_final", tr)
    summary = {
        "num_outer_steps": tr.num_outer_steps,
        "num_attempts": n_attempts,
        "accept_rate_final": n_accepted / max(n_attempts, 1),
        "wall_seconds": time.time() - t0,
        "training_cfg": asdict(tr),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def _save(base_lm: BaseLM, diff_lm, critic, path: Path, tr) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if tr.train_base_lm:
        base_lm.model.save_pretrained(path / "base_lm_lora")
    if tr.train_diff_lm:
        torch.save(diff_lm.state_dict(), path / "diff_lm.pt")
    if tr.train_critic and critic is not None:
        torch.save(critic.state_dict(), path / "critic.pt")
