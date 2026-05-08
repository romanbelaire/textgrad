"""Encode a text span into L2-normalized embeddings on the unit sphere; decode
a matrix of L2-normalized embeddings back to text via cosine-argmax into the
BaseLM input-embedding matrix.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .base_lm import BaseLM


def encode_span(base_lm: BaseLM, text_span: str) -> torch.Tensor:
    """Tokenize `text_span` (no special tokens), look up BaseLM input
    embeddings for those ids, L2-normalize per-token. Returns [L, d].

    Raises if the span tokenizes to zero tokens.
    """
    ids = base_lm.tokenizer(
        text_span, add_special_tokens=False, return_tensors="pt"
    ).input_ids[0].to(base_lm.device)
    if ids.numel() == 0:
        raise ValueError(f"Empty span after tokenization: {text_span!r}")
    z = base_lm.embedding_matrix[ids]  # [L, d]
    return F.normalize(z, dim=-1)


def decode_span(base_lm: BaseLM, z: torch.Tensor) -> tuple[str, torch.Tensor]:
    """Cosine-argmax each row of `z` (expected L2-normalized) into the
    BaseLM embedding matrix; decode the resulting token ids with
    `skip_special_tokens=True`.

    Returns `(text, ids)` where ids is [L].
    """
    if z.dim() != 2:
        raise ValueError(f"decode_span expects z of shape [L, d], got {tuple(z.shape)}")
    embed_norm = base_lm.embedding_matrix_norm  # [V, d]
    sims = z.to(embed_norm.dtype) @ embed_norm.T  # [L, V]
    ids = sims.argmax(dim=-1)  # [L]
    text = base_lm.tokenizer.decode(ids.tolist(), skip_special_tokens=True)
    return text, ids


def replace_span(gen_text: str, char_s: int, char_e: int, new_text: str) -> str:
    return gen_text[:char_s] + new_text + gen_text[char_e:]
