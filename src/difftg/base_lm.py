"""Wrapper around a HuggingFace causal LM.

Exposes: `generate`, `logprob_span`, and the input-embedding matrix (plus a
cached L2-normalized view for cosine-argmax decoding). All char indices are
into the generated text only; the prompt is never edited.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import BaseLMConfig

_ANSWER_TAG_SYSTEM_INSTRUCTION = (
    "Always wrap the final answer in [ANSWER] and [/ANSWER] tags. "
    "Use exactly one tagged final answer segment."
)


@dataclass
class Trajectory:
    """A single BaseLM rollout.

    `prompt` is the user-role content passed in. `chat_prompt` is the exact
    string fed to the model after applying the chat template. `gen_text` is the
    decoded generation (no special tokens).
    """

    prompt: str
    chat_prompt: str
    gen_text: str


class BaseLM:
    def __init__(self, cfg: BaseLMConfig):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.bfloat16,
        ).to(cfg.device)
        self.model.eval()
        self.device = torch.device(cfg.device)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self._embed_weight_norm: torch.Tensor | None = None

    @property
    def embedding_matrix(self) -> torch.Tensor:
        """Raw input-embedding weight, shape [V, d]."""
        return self.model.get_input_embeddings().weight

    @property
    def embedding_matrix_norm(self) -> torch.Tensor:
        """L2-normalized input-embedding weight, cached. Detached from grads."""
        if self._embed_weight_norm is None:
            w = self.embedding_matrix.detach()
            self._embed_weight_norm = F.normalize(w, dim=-1)
        return self._embed_weight_norm

    def _chat_prompt(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": _ANSWER_TAG_SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.no_grad()
    def generate(self, prompt: str) -> Trajectory:
        chat_prompt = self._chat_prompt(prompt)
        ids = self.tokenizer(chat_prompt, return_tensors="pt").input_ids.to(self.device)
        do_sample = self.cfg.temperature > 0.0
        out = self.model.generate(
            ids,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=do_sample,
            temperature=self.cfg.temperature if do_sample else 1.0,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen_ids = out[0, ids.size(1):]
        gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return Trajectory(prompt=prompt, chat_prompt=chat_prompt, gen_text=gen_text)

    def logprob_span(self, traj: Trajectory, char_s: int, char_e: int) -> torch.Tensor:
        """Differentiable sum of BaseLM log-probs of the gen tokens whose
        offsets fall inside [char_s, char_e). Used by v2 REINFORCE.
        """
        prompt_ids = self.tokenizer(
            traj.chat_prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids[0].to(self.device)

        gen_enc = self.tokenizer(
            traj.gen_text,
            return_tensors="pt",
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        gen_ids = gen_enc.input_ids[0].to(self.device)
        offsets = gen_enc.offset_mapping[0].tolist()

        # Tokens whose char span overlaps [char_s, char_e).
        span_idx = [
            i for i, (a, b) in enumerate(offsets)
            if a < char_e and b > char_s
        ]
        if not span_idx:
            raise ValueError(
                f"No gen tokens fall inside char range [{char_s}, {char_e}); "
                f"gen_text length={len(traj.gen_text)}"
            )
        i0, i1 = span_idx[0], span_idx[-1] + 1  # exclusive end

        full = torch.cat([prompt_ids, gen_ids], dim=0).unsqueeze(0)
        logits = self.model(full).logits[0]  # [T, V]
        P = prompt_ids.size(0)
        # logits[t] predicts token at position t+1. Token gen_ids[j] sits at
        # position P+j, so its predicting logits row is P+j-1.
        target_logits = logits[P + i0 - 1 : P + i1 - 1]  # [L_span, V]
        target_ids = gen_ids[i0:i1]
        log_probs = F.log_softmax(target_logits.float(), dim=-1)
        return log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1).sum()

    @torch.no_grad()
    def per_token_logprobs(
        self, chat_prompt: str, gen_text: str
    ) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        """Log-prob of each gen token given prompt+prefix. Returns (logprobs
        [G], offsets [G] as (char_start, char_end) into gen_text). Used by the
        lowest-logprob span selector.
        """
        prompt_ids = self.tokenizer(
            chat_prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids[0].to(self.device)
        gen_enc = self.tokenizer(
            gen_text,
            return_tensors="pt",
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        gen_ids = gen_enc.input_ids[0].to(self.device)
        offsets = [tuple(o) for o in gen_enc.offset_mapping[0].tolist()]

        full = torch.cat([prompt_ids, gen_ids], dim=0).unsqueeze(0)
        logits = self.model(full).logits[0]
        P = prompt_ids.size(0)
        G = gen_ids.size(0)
        target_logits = logits[P - 1 : P + G - 1]
        log_probs = F.log_softmax(target_logits.float(), dim=-1)
        per_tok = log_probs.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1)
        return per_tok, offsets
