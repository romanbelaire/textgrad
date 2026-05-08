"""Frozen dataclass configs for DiffTG.

Loaded from YAML (or JSON); every subsystem has its own sub-config. We do not
silently fill in missing keys or coerce types; if a key is missing or has the
wrong type the constructor will raise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BaseLMConfig:
    model_name: str
    device: str
    max_new_tokens: int
    temperature: float


@dataclass(frozen=True)
class DiffLMConfig:
    kind: str  # "gaussian" | "trainable" | "pretrained"
    # trainable-only
    d_model: int | None = None
    n_layers: int | None = None
    n_heads: int | None = None
    t_max: int | None = None
    ckpt_path: str | None = None
    # pretrained-only
    pretrained_name: str | None = None
    bridge_dim: int | None = None


@dataclass(frozen=True)
class LangevinConfig:
    K: int
    step_size: float
    tangent_project: bool


@dataclass(frozen=True)
class SpanSelectConfig:
    kind: str  # "random" | "lowest_logprob" | "judge"
    num_spans: int
    # random-only
    unit: str | None = None  # "sentence" | "line"
    # lowest_logprob-only
    window_tokens: int | None = None
    # judge-only
    judge_url: str | None = None
    judge_model: str | None = None


@dataclass(frozen=True)
class TaskConfig:
    kind: str  # "toy" | "gsm8k"
    num_tasks: int
    seed: int
    # toy-only
    variant: str | None = None
    # gsm8k-only
    split: str | None = None


@dataclass(frozen=True)
class RewardConfig:
    kind: str  # "toy_string" | "exact_match"


@dataclass(frozen=True)
class TrainingConfig:
    num_outer_steps: int
    lr: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    grad_clip: float
    train_base_lm: bool
    train_diff_lm: bool
    train_critic: bool
    critic_hidden: int
    save_every: int


@dataclass(frozen=True)
class Config:
    mode: str  # "inference" | "training"
    base_lm: BaseLMConfig
    diff_lm: DiffLMConfig
    langevin: LangevinConfig
    span_select: SpanSelectConfig
    task: TaskConfig
    reward: RewardConfig
    training: TrainingConfig | None = None


def _build(cls, data: dict[str, Any]):
    return cls(**data)


def load_config(path: str | Path) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(
        mode=raw["mode"],
        base_lm=_build(BaseLMConfig, raw["base_lm"]),
        diff_lm=_build(DiffLMConfig, raw["diff_lm"]),
        langevin=_build(LangevinConfig, raw["langevin"]),
        span_select=_build(SpanSelectConfig, raw["span_select"]),
        task=_build(TaskConfig, raw["task"]),
        reward=_build(RewardConfig, raw["reward"]),
        training=_build(TrainingConfig, raw["training"]) if "training" in raw else None,
    )
