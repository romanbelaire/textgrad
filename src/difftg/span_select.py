"""`SelectSpan` strategies.

A span selector takes a `Trajectory` plus the current reward and returns a list
of `(char_start, char_end)` indices into `traj.gen_text`. All spans are
non-empty and end > start.

Implementations:
    - `AnswerSpanSelector`: return the value part of an `answer: ...` line.
      Only useful for tasks whose reward is a regex match on this marker; it
      guarantees the perturbed span overlaps the reward-determining text.
    - `RandomSentenceSelector`: split on sentence / line terminators; sample.
    - `LowestLogprobSelector`: pick the window of N tokens with lowest mean
      BaseLM log-probability.
    - `JudgeSelector`: call an OpenAI-compatible judge LLM to return a span
      it believes most likely to be wrong.
"""
from __future__ import annotations

import json
import random
import re
import urllib.request
from typing import Protocol

import torch

from .base_lm import BaseLM, Trajectory
from .config import SpanSelectConfig


Span = tuple[int, int]


class SelectSpan(Protocol):
    def __call__(self, traj: Trajectory, R_orig: float, n: int) -> list[Span]: ...


# --- answer line ------------------------------------------------------------

_ANSWER_TAG_RE = re.compile(r"\[ANSWER\](.*?)\[/ANSWER\]", re.IGNORECASE | re.DOTALL)
_ANSWER_VALUE_RE = re.compile(r"answer\s*[:=]\s*([^\n]+)", re.IGNORECASE)
_FOUR_DIGIT_RE = re.compile(r"(?<!\d)(\d{4})(?!\d)")


def gen_text_has_answer_span_marker(gen_text: str) -> bool:
    """True iff `AnswerSpanSelector` would find a span without raising (ignoring toy 4-digit fallback)."""
    return _ANSWER_TAG_RE.search(gen_text) is not None or _ANSWER_VALUE_RE.search(gen_text) is not None


class AnswerSpanSelector:
    """Return the char span of the final-answer value in gen_text.

    The selector always returns at most one span: the substring captured by
    group(1) of `/\\[ANSWER\\](.*?)\\[/ANSWER\\]/is` or
    `/answer\\s*[:=]\\s*([^\\n]+)/i`. `n > 1` is ignored.
    Fails loudly if no answer marker is present.
    """

    def __call__(self, traj: Trajectory, R_orig: float, n: int) -> list[Span]:
        tag = _ANSWER_TAG_RE.search(traj.gen_text)
        if tag is not None:
            return [(tag.start(1), tag.end(1))]

        m = _ANSWER_VALUE_RE.search(traj.gen_text)
        if m is not None:
            return [(m.start(1), m.end(1))]

        if "Take the 4-digit number" not in traj.prompt:
            raise ValueError(
                "AnswerSpanSelector: gen_text has no `[ANSWER]...[/ANSWER]` or `answer: ...` marker; "
                f"gen_text={traj.gen_text!r}"
            )

        # Toy fallback: if there is no explicit answer marker, use the last
        # standalone 4-digit number in the generation.
        four_digit = list(_FOUR_DIGIT_RE.finditer(traj.gen_text))
        if four_digit:
            last = four_digit[-1]
            return [(last.start(1), last.end(1))]

        # If no 4-digit output exists, append an explicit incorrect candidate so
        # downstream span edits/reward extraction can continue deterministically.
        prompt_four_digit = _FOUR_DIGIT_RE.search(traj.prompt)
        fallback = prompt_four_digit.group(1) if prompt_four_digit is not None else "0000"
        traj.gen_text = traj.gen_text.rstrip() + f"\n[ANSWER]{fallback}[/ANSWER]"
        m2 = _ANSWER_TAG_RE.search(traj.gen_text)
        if m2 is None:
            raise ValueError(
                "AnswerSpanSelector: failed to synthesize fallback answer marker; "
                f"gen_text={traj.gen_text!r}"
            )
        return [(m2.start(1), m2.end(1))]


# --- random -----------------------------------------------------------------

_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?\n]?")


class RandomSentenceSelector:
    def __init__(self, unit: str, seed: int):
        if unit not in ("sentence", "line"):
            raise ValueError(f"Unknown unit: {unit}")
        self.unit = unit
        self.rng = random.Random(seed)

    def _spans(self, text: str) -> list[Span]:
        if self.unit == "sentence":
            matches = _SENTENCE_RE.finditer(text)
            out = [(m.start(), m.end()) for m in matches if m.group().strip()]
        else:  # line
            out = []
            i = 0
            for line in text.splitlines(keepends=True):
                stripped = line.strip()
                if stripped:
                    j = i + len(line.rstrip("\n"))
                    out.append((i, j))
                i += len(line)
        return out

    def __call__(self, traj: Trajectory, R_orig: float, n: int) -> list[Span]:
        cands = self._spans(traj.gen_text)
        if not cands:
            raise ValueError(f"No candidate spans in gen_text: {traj.gen_text!r}")
        n = min(n, len(cands))
        return self.rng.sample(cands, n)


# --- lowest log-prob --------------------------------------------------------


class LowestLogprobSelector:
    """Pick the window of `window_tokens` consecutive tokens with the lowest
    mean log-probability under BaseLM; return its char span."""

    def __init__(self, base_lm: BaseLM, window_tokens: int):
        if window_tokens < 1:
            raise ValueError(f"window_tokens must be >= 1, got {window_tokens}")
        self.base_lm = base_lm
        self.window = window_tokens

    def __call__(self, traj: Trajectory, R_orig: float, n: int) -> list[Span]:
        per_tok, offsets = self.base_lm.per_token_logprobs(traj.chat_prompt, traj.gen_text)
        G = per_tok.size(0)
        if G == 0:
            raise ValueError("gen_text tokenizes to zero tokens.")
        w = min(self.window, G)
        # Rolling sum via cumulative sum.
        cs = per_tok.cumsum(0)
        # window sum at start i is cs[i+w-1] - (cs[i-1] if i>0 else 0)
        window_sums = cs[w - 1:] - torch.cat([torch.zeros(1, device=cs.device), cs[:-w]], dim=0)
        # Sort ascending by log-prob mean.
        order = window_sums.argsort().tolist()
        spans: list[Span] = []
        chosen_starts: list[int] = []
        for start in order:
            if len(spans) >= n:
                break
            # De-duplicate overlapping windows.
            if any(abs(start - s) < w for s in chosen_starts):
                continue
            end = start + w
            char_s = offsets[start][0]
            char_e = offsets[end - 1][1]
            if char_e <= char_s:
                continue
            spans.append((char_s, char_e))
            chosen_starts.append(start)
        if not spans:
            raise ValueError("Failed to select any spans via LowestLogprob.")
        return spans


# --- judge ------------------------------------------------------------------


_JUDGE_PROMPT = (
    "You are auditing an AI's response. Identify the single contiguous "
    "substring of the response that is most likely to be incorrect or could "
    "be improved. Return ONLY a JSON object of the form "
    '{"start": <int>, "end": <int>} giving character indices into the response '
    "(0-indexed, end-exclusive). No commentary.\n\n"
    "RESPONSE:\n---\n{response}\n---"
)


class JudgeSelector:
    """Ask an OpenAI-compatible judge LLM for a char range to perturb.

    Uses the `/v1/chat/completions` endpoint. Fails loudly if the judge
    returns malformed JSON or out-of-range indices.
    """

    def __init__(self, judge_url: str, judge_model: str):
        self.url = judge_url.rstrip("/") + "/v1/chat/completions"
        self.model = judge_model

    def _call(self, user_msg: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": user_msg}],
            "temperature": 0.0,
        }
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]

    def __call__(self, traj: Trajectory, R_orig: float, n: int) -> list[Span]:
        spans: list[Span] = []
        for _ in range(n):
            reply = self._call(_JUDGE_PROMPT.format(response=traj.gen_text))
            obj = json.loads(reply.strip())
            s, e = int(obj["start"]), int(obj["end"])
            if not (0 <= s < e <= len(traj.gen_text)):
                raise ValueError(
                    f"Judge returned invalid span ({s}, {e}) for gen_text of length {len(traj.gen_text)}"
                )
            spans.append((s, e))
        return spans


# --- factory ----------------------------------------------------------------


def build_selector(cfg: SpanSelectConfig, base_lm: BaseLM, seed: int) -> SelectSpan:
    if cfg.kind == "answer":
        return AnswerSpanSelector()
    if cfg.kind == "random":
        return RandomSentenceSelector(unit=cfg.unit or "sentence", seed=seed)
    if cfg.kind == "lowest_logprob":
        if cfg.window_tokens is None:
            raise ValueError("window_tokens must be set for lowest_logprob selector.")
        return LowestLogprobSelector(base_lm=base_lm, window_tokens=cfg.window_tokens)
    if cfg.kind == "judge":
        if cfg.judge_url is None or cfg.judge_model is None:
            raise ValueError("judge_url and judge_model must be set for judge selector.")
        return JudgeSelector(judge_url=cfg.judge_url, judge_model=cfg.judge_model)
    raise ValueError(f"Unknown span_select kind: {cfg.kind!r}")
