"""GSM8K task loader (stretch goal).

Uses HuggingFace `datasets` to pull the `gsm8k` benchmark. Answer field in
GSM8K is a full explanation ending with `#### <number>`; we extract the final
number as ground truth for `ExactMatchReward`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from datasets import load_dataset

from ..config import TaskConfig


@dataclass(frozen=True)
class GSM8KTask:
    prompt: str
    answer: str  # the numeric answer as a string


_FINAL_NUM_RE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")


def build_gsm8k_tasks(cfg: TaskConfig) -> list[GSM8KTask]:
    split = cfg.split if cfg.split is not None else "test"
    ds = load_dataset("gsm8k", "main", split=split)
    tasks: list[GSM8KTask] = []
    for i, ex in enumerate(ds):
        if i >= cfg.num_tasks:
            break
        m = _FINAL_NUM_RE.search(ex["answer"])
        if m is None:
            raise ValueError(f"GSM8K example {i} has no `#### <num>` marker: {ex['answer']!r}")
        prompt = (
            f"{ex['question']}\n\n"
            f"Reason step by step, then on a final line write `[ANSWER]<number>[/ANSWER]`."
        )
        tasks.append(GSM8KTask(prompt=prompt, answer=m.group(1)))
    return tasks
