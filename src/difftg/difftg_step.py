"""Inference-time DiffTextGradStep.

One call to `difftg_step` = one "textual gradient" analog: perturb selected
spans of a single trajectory via on-sphere Langevin in embedding space, accept
edits that do not decrease reward. This is spec sec. "Inference-time
TextGrad alternative" (v1 deliverable).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .base_lm import BaseLM, Trajectory
from .encoder import decode_span, encode_span, replace_span
from .langevin import run_langevin
from .config import LangevinConfig


@dataclass
class SpanAttempt:
    char_start: int
    char_end: int
    orig_span_text: str
    new_span_text: str
    R_new: float
    delta_R: float
    accepted: bool
    cos_final: float  # cosine(z_0, z_K) averaged over tokens
    step_norms: list[float] = field(default_factory=list)


@dataclass
class DiffTGResult:
    task_idx: int
    traj: Trajectory
    final_gen_text: str
    R_orig: float
    R_final: float
    attempts: list[SpanAttempt]


@torch.no_grad()
def difftg_step(
    base_lm: BaseLM,
    diff_lm,
    selector,
    reward_fn,
    task,
    langevin_cfg: LangevinConfig,
    num_spans: int,
    task_idx: int = 0,
    generator: torch.Generator | None = None,
) -> DiffTGResult:
    """Run one inference-time DiffTextGradStep for a single task.

    Matches spec pseudocode:

        traj = BaseLM.generate(x)
        R_orig = R(traj)
        for (s, e) in SelectSpan(traj)[:num_spans]:
            z0 = normalize(EncodeSpan(traj[s:e]))
            z  = Langevin(z0, K, step_size)
            traj' = replace_span(traj, s, e, DecodeSpan(z))
            if R(traj') >= R_orig: accept
        return traj, R
    """
    traj = base_lm.generate(task.prompt)
    R_orig = float(reward_fn(traj.gen_text, task))

    spans = selector(traj, R_orig, num_spans)

    cur_text = traj.gen_text
    cur_R = R_orig
    attempts: list[SpanAttempt] = []
    for (s, e) in spans:
        orig_span = cur_text[s:e]
        z0 = encode_span(base_lm, orig_span)
        z_k, info = run_langevin(z0, diff_lm, langevin_cfg, generator=generator)
        new_span, _ = decode_span(base_lm, z_k)
        cand_text = replace_span(cur_text, s, e, new_span)
        R_new = float(reward_fn(cand_text, task))
        delta_R = R_new - cur_R
        accept = R_new >= cur_R
        attempts.append(
            SpanAttempt(
                char_start=s,
                char_end=e,
                orig_span_text=orig_span,
                new_span_text=new_span,
                R_new=R_new,
                delta_R=delta_R,
                accepted=accept,
                cos_final=info["cos_to_z0"][-1] if info["cos_to_z0"] else 1.0,
                step_norms=info["step_norm"],
            )
        )
        if accept:
            cur_text = cand_text
            cur_R = R_new

    return DiffTGResult(
        task_idx=task_idx,
        traj=traj,
        final_gen_text=cur_text,
        R_orig=R_orig,
        R_final=cur_R,
        attempts=attempts,
    )
