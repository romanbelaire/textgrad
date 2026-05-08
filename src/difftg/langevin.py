"""K-step Langevin dynamics on the unit sphere (per-token).

Textbook Langevin in R^d:

    z_{t+1} = z_t + eps * score(z_t, t) + sqrt(2*eps) * eta,   eta ~ N(0, I_d)

In embedding space d is large (~1k-4k), so ||sqrt(2*eps)*eta|| ~ sqrt(2*eps*d)
which swamps the unit sphere for any reasonable `step_size` unless we compensate
for d. We rescale the noise by 1/sqrt(d) so that the total per-step increment
has norm on the order of sqrt(2*eps), matching the textbook one-dim intuition:

    z_{t+1} = normalize( z_t + eps * score(z_t, t) + sqrt(2*eps / d) * eta )

`eta` is optionally projected onto the tangent space to the sphere at `z_t`
before being used, following spec sec. 3b. Per-step arc length is then
approximately sqrt(2*eps) and cos(z_0, z_K) ~ exp(-K*eps) for small eps.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .config import LangevinConfig


def project_to_tangent(eps: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Remove the radial component of `eps` at each point `z` on the sphere.
    Both tensors have shape [L, d]; `z` is expected L2-normalized per-row.
    """
    radial = (eps * z).sum(dim=-1, keepdim=True) * z
    return eps - radial


def run_langevin(
    z0: torch.Tensor,
    diff_lm,  # DiffLM protocol: has .score(z, t) -> Tensor
    cfg: LangevinConfig,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict]:
    """Run K Langevin steps starting from `z0` (assumed L2-normalized).

    Returns `(z_K, info)` where info holds per-step diagnostics:
        - `cos_to_z0`: cosine similarity to z0, mean over tokens, per-step.
        - `step_norm`: ||z_{t+1} - z_t|| mean over tokens, per-step.
    """
    if z0.dim() != 2:
        raise ValueError(f"run_langevin expects z0 of shape [L, d], got {tuple(z0.shape)}")
    z = z0.clone()
    d = z.shape[-1]
    noise_scale = math.sqrt(2.0 * cfg.step_size / d)
    cos_hist: list[float] = []
    step_hist: list[float] = []
    for t in range(cfg.K):
        eps = torch.randn(z.shape, device=z.device, dtype=z.dtype, generator=generator)
        if cfg.tangent_project:
            eps = project_to_tangent(eps, z)
        score = diff_lm.score(z, t)
        z_new = z + cfg.step_size * score + noise_scale * eps
        z_new = F.normalize(z_new, dim=-1)
        cos_hist.append(float((z_new * z0).sum(dim=-1).mean().item()))
        step_hist.append(float((z_new - z).norm(dim=-1).mean().item()))
        z = z_new
    return z, {"cos_to_z0": cos_hist, "step_norm": step_hist}
