"""Pluggable DiffLM backends.

All implementations expose:

    score(z: [L, d] | [B, L, d], t: int) -> same-shape tensor

where `z` is assumed L2-normalized per-token (on the unit sphere in each row).

Backends:
    - `GaussianNoiseDiffLM` (v1): score = 0; pure Langevin noise.
    - `TrainableEmbedDenoiser` (v1.5): small Transformer score net over the
      BaseLM input-embedding space. Fit with denoising score matching.
    - `PretrainedDiffLMAdapter` (v1.5): shim for an external continuous
      diffusion LM; a linear bridge maps between its embedding space and the
      BaseLM's. Concrete wrappers are deferred until we pick a model; this
      class raises on `score` if no wrapper is registered for the name.
"""
from __future__ import annotations

import math
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_lm import BaseLM
from .config import DiffLMConfig


class DiffLM(Protocol):
    def score(self, z: torch.Tensor, t: int) -> torch.Tensor: ...


# --- v1 baseline -------------------------------------------------------------


class GaussianNoiseDiffLM:
    """score(z, t) = 0. Equivalent to pure isotropic Langevin noise; a useful
    ablation baseline and the v1 default."""

    def __init__(self):
        pass

    def score(self, z: torch.Tensor, t: int) -> torch.Tensor:
        return torch.zeros_like(z)

    def parameters(self):
        return iter(())

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        if state:
            raise ValueError("GaussianNoiseDiffLM has no state to load.")


# --- v1.5 trainable ----------------------------------------------------------


class _SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim: int, t_max: int):
        super().__init__()
        self.dim = dim
        self.t_max = t_max

    def forward(self, t: int, device, dtype) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / max(half - 1, 1)
        )
        x = torch.tensor([float(t) / max(self.t_max, 1)], device=device, dtype=torch.float32) * freqs
        emb = torch.cat([x.sin(), x.cos()], dim=-1).to(dtype)
        if emb.numel() < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.numel()))
        return emb  # [dim]


class TrainableEmbedDenoiser(nn.Module):
    """Small Transformer score network over BaseLM input embeddings.

    Input: z of shape [L, d] or [B, L, d] (L2-normalized).
    Output: score of same shape.

    The time index `t` is embedded sinusoidally and added broadcast-wise.
    """

    def __init__(self, d: int, n_layers: int, n_heads: int, t_max: int):
        super().__init__()
        self.d = d
        self.time_embed = _SinusoidalTimeEmbed(d, t_max)
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=4 * d,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d, d)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z: torch.Tensor, t: int) -> torch.Tensor:
        squeeze = False
        if z.dim() == 2:
            z = z.unsqueeze(0)
            squeeze = True
        t_emb = self.time_embed(t, z.device, z.dtype)  # [d]
        h = z + t_emb.view(1, 1, -1)
        h = self.transformer(h)
        out = self.out_proj(h)
        return out.squeeze(0) if squeeze else out

    def score(self, z: torch.Tensor, t: int) -> torch.Tensor:
        return self.forward(z, t)

    def score_matching_loss(
        self,
        z0: torch.Tensor,
        t: int,
        sigma: float,
    ) -> torch.Tensor:
        """Denoising score-matching loss for a single time step.

        Given clean z0 (L2-normalized per row, shape [B, L, d]) and noise level
        sigma, we corrupt with Gaussian noise, re-normalize, and regress the
        target score -(z_noised - z0) / sigma^2. This is a simplified Euclidean
        target; for spherical geometry we rely on re-normalization to keep
        samples near the manifold.
        """
        if z0.dim() == 2:
            z0 = z0.unsqueeze(0)
        eps = torch.randn_like(z0)
        z_noisy = F.normalize(z0 + sigma * eps, dim=-1)
        target = -(z_noisy - z0) / (sigma * sigma)
        pred = self.forward(z_noisy, t)
        return F.mse_loss(pred, target)


# --- v1.5 pretrained adapter -------------------------------------------------


_PRETRAINED_REGISTRY: dict[str, type] = {}


def register_pretrained(name: str):
    def _reg(cls):
        _PRETRAINED_REGISTRY[name] = cls
        return cls
    return _reg


@register_pretrained("bert_reconstructor")
class BertReconstructorDenoiser(nn.Module):
    """Use a HuggingFace BERT-family encoder as a rough continuous denoiser.

    Inputs are treated as soft input-embeddings to BERT; the score is the
    residual between BERT's contextualized output and its input:

        score(z, t) = BertEncoder(z + pos_embed) - z

    This is not a true diffusion LM but gives a concrete pretrained baseline
    to compare against `GaussianNoiseDiffLM` and `TrainableEmbedDenoiser`.
    Replace with a real continuous-embedding diffusion LM wrapper (Plaid /
    LD4LG / TESS) when one is selected; register it via `@register_pretrained`.
    """

    HF_MODEL = "bert-base-uncased"

    def __init__(self, cfg: "DiffLMConfig", base_lm: BaseLM):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(
            self.HF_MODEL,
            torch_dtype=torch.bfloat16,
        ).to(base_lm.device)
        self.bert.eval()
        for p in self.bert.parameters():
            p.requires_grad = False
        self.hidden = self.bert.config.hidden_size

    def score(self, z: torch.Tensor, t: int) -> torch.Tensor:
        squeeze = False
        if z.dim() == 2:
            z = z.unsqueeze(0)
            squeeze = True
        B, L, _ = z.shape
        pos_ids = torch.arange(L, device=z.device).unsqueeze(0).expand(B, L)
        pos_emb = self.bert.embeddings.position_embeddings(pos_ids)
        type_ids = torch.zeros(B, L, dtype=torch.long, device=z.device)
        type_emb = self.bert.embeddings.token_type_embeddings(type_ids)
        h = self.bert.embeddings.LayerNorm(z + pos_emb + type_emb)
        attn_mask = torch.ones(B, L, device=z.device, dtype=z.dtype)
        ext_mask = self.bert.get_extended_attention_mask(attn_mask, (B, L))
        enc = self.bert.encoder(h, attention_mask=ext_mask).last_hidden_state
        score = enc - z
        return score.squeeze(0) if squeeze else score


class PretrainedDiffLMAdapter(nn.Module):
    """Wraps an external continuous-embedding diffusion LM.

    Concrete wrappers are registered via `@register_pretrained(name)`. If the
    pretrained model's embedding space differs from BaseLM's, a trainable
    linear bridge projects back and forth:

        bridge_in:  base_d -> pretrained_d
        bridge_out: pretrained_d -> base_d

    The bridge is initialised as near-identity (or zero-padded identity) and
    can be fit by the trainer together with the pretrained model kept frozen.
    """

    def __init__(self, cfg: DiffLMConfig, base_lm: BaseLM):
        super().__init__()
        if cfg.pretrained_name is None:
            raise ValueError("DiffLMConfig.pretrained_name must be set for PretrainedDiffLMAdapter.")
        if cfg.pretrained_name not in _PRETRAINED_REGISTRY:
            raise NotImplementedError(
                f"No pretrained DiffLM wrapper registered for {cfg.pretrained_name!r}. "
                f"Known: {sorted(_PRETRAINED_REGISTRY)}. Register one via @register_pretrained."
            )
        wrapper_cls = _PRETRAINED_REGISTRY[cfg.pretrained_name]
        self.inner = wrapper_cls(cfg, base_lm)

        base_d = base_lm.embedding_matrix.shape[-1]
        pre_d = cfg.bridge_dim if cfg.bridge_dim is not None else base_d
        self.bridge_in = nn.Linear(base_d, pre_d, bias=False)
        self.bridge_out = nn.Linear(pre_d, base_d, bias=False)
        # Initialise as a (possibly rectangular) identity-like map.
        with torch.no_grad():
            k = min(base_d, pre_d)
            self.bridge_in.weight.zero_()
            self.bridge_out.weight.zero_()
            eye = torch.eye(k)
            self.bridge_in.weight[:k, :k].copy_(eye)
            self.bridge_out.weight[:k, :k].copy_(eye)

    def score(self, z: torch.Tensor, t: int) -> torch.Tensor:
        z_in = self.bridge_in(z)
        s_in = self.inner.score(z_in, t)
        return self.bridge_out(s_in)


# --- factory -----------------------------------------------------------------


def build_diff_lm(cfg: DiffLMConfig, base_lm: BaseLM) -> DiffLM:
    if cfg.kind == "gaussian":
        return GaussianNoiseDiffLM()
    if cfg.kind == "trainable":
        d = cfg.d_model if cfg.d_model is not None else base_lm.embedding_matrix.shape[-1]
        if d != base_lm.embedding_matrix.shape[-1]:
            raise ValueError(
                f"TrainableEmbedDenoiser must match BaseLM embedding dim "
                f"({base_lm.embedding_matrix.shape[-1]}); got d_model={d}."
            )
        model = TrainableEmbedDenoiser(
            d=d,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            t_max=cfg.t_max,
        ).to(base_lm.device)
        if cfg.ckpt_path is not None:
            model.load_state_dict(torch.load(cfg.ckpt_path, map_location=base_lm.device))
        return model
    if cfg.kind == "pretrained":
        return PretrainedDiffLMAdapter(cfg, base_lm).to(base_lm.device)
    raise ValueError(f"Unknown DiffLM kind: {cfg.kind!r}")
