"""Synthetic toy tasks for validating the DiffTG pipeline.

Each task has a `prompt` (the user query) and an `answer` (the ground-truth
string the reward will compare against the extracted final answer from the
model's output).

Variants:
    - `digit_shift`: add 1 (mod 10) to each digit of a 4-digit number.
    - `reverse_string`: reverse a 6-letter random string.
    - `sum_digits`: sum the digits of a 4-digit number.
"""
from __future__ import annotations

import random
import string
from dataclasses import dataclass

from ..config import TaskConfig


@dataclass(frozen=True)
class ToyTask:
    prompt: str
    answer: str


def _digit_shift(rng: random.Random) -> ToyTask:
    digits = "".join(str(rng.randint(0, 9)) for _ in range(4))
    shifted = "".join(str((int(d) + 1) % 10) for d in digits)
    prompt = (
        f"Take the 4-digit number {digits} and add 1 (mod 10) to each digit "
        f"individually. Show brief reasoning, then on a final line write "
        f"`[ANSWER]<result>[/ANSWER]` where <result> is the new 4-digit string."
    )
    return ToyTask(prompt=prompt, answer=shifted)


def _reverse_string(rng: random.Random) -> ToyTask:
    s = "".join(rng.choice(string.ascii_lowercase) for _ in range(6))
    rev = s[::-1]
    prompt = (
        f"Reverse the string `{s}`. Show brief reasoning, then on a final "
        f"line write `[ANSWER]<result>[/ANSWER]`."
    )
    return ToyTask(prompt=prompt, answer=rev)


def _sum_digits(rng: random.Random) -> ToyTask:
    digits = "".join(str(rng.randint(0, 9)) for _ in range(4))
    total = sum(int(d) for d in digits)
    prompt = (
        f"Compute the sum of the digits of {digits}. Show brief reasoning, "
        f"then on a final line write `[ANSWER]<result>[/ANSWER]`."
    )
    return ToyTask(prompt=prompt, answer=str(total))


_VARIANTS = {
    "digit_shift": _digit_shift,
    "reverse_string": _reverse_string,
    "sum_digits": _sum_digits,
}


def build_toy_tasks(cfg: TaskConfig) -> list[ToyTask]:
    if cfg.variant is None:
        raise ValueError("TaskConfig.variant must be set for toy tasks.")
    if cfg.variant not in _VARIANTS:
        raise ValueError(f"Unknown toy variant: {cfg.variant!r}; known: {sorted(_VARIANTS)}")
    gen = _VARIANTS[cfg.variant]
    rng = random.Random(cfg.seed)
    return [gen(rng) for _ in range(cfg.num_tasks)]
