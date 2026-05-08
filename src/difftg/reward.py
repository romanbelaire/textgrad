"""Scalar reward functions R(trajectory, task).

A Reward implements `__call__(gen_text: str, task) -> float` returning a
scalar in [0, 1]. The `task` object is opaque to the reward except that it
must expose `task.answer` as a ground-truth string (for the two rewards we
ship today).
"""
from __future__ import annotations

import difflib
import re
from typing import Protocol

from .config import RewardConfig


class Reward(Protocol):
    def __call__(self, gen_text: str, task) -> float: ...


_ANSWER_TAG_RE = re.compile(r"\[ANSWER\](.*?)\[/ANSWER\]", re.IGNORECASE | re.DOTALL)
_ANSWER_RE = re.compile(r"answer\s*[:=]\s*([^\n]+)", re.IGNORECASE)


def _extract_answer(gen_text: str) -> str:
    tag = _ANSWER_TAG_RE.search(gen_text)
    if tag is not None:
        return tag.group(1).strip()
    m = _ANSWER_RE.search(gen_text)
    if m is None:
        # No well-formed answer marker; treat entire last non-empty line as answer.
        lines = [ln.strip() for ln in gen_text.splitlines() if ln.strip()]
        return lines[-1] if lines else ""
    return m.group(1).strip()


class ToyStringReward:
    """Soft reward for toy string tasks. Returns `difflib.SequenceMatcher`
    ratio between the extracted answer and ground truth. Exact match -> 1.0.
    """

    def __call__(self, gen_text: str, task) -> float:
        pred = _extract_answer(gen_text)
        truth = task.answer
        if not truth:
            raise ValueError("Task.answer is empty; toy reward requires a non-empty truth.")
        if pred == truth:
            return 1.0
        return float(difflib.SequenceMatcher(None, pred, truth).ratio())


_GSM8K_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


class ExactMatchReward:
    """Binary reward: 1.0 iff the last number in `gen_text` equals `task.answer`."""

    def __call__(self, gen_text: str, task) -> float:
        answer_text = _extract_answer(gen_text)
        nums = _GSM8K_NUMBER_RE.findall(answer_text)
        if not nums:
            nums = _GSM8K_NUMBER_RE.findall(gen_text)
        if not nums:
            return 0.0
        pred = nums[-1].rstrip(".")
        return 1.0 if pred == str(task.answer).strip() else 0.0


def build_reward(cfg: RewardConfig) -> Reward:
    if cfg.kind == "toy_string":
        return ToyStringReward()
    if cfg.kind == "exact_match":
        return ExactMatchReward()
    raise ValueError(f"Unknown reward kind: {cfg.kind!r}")
