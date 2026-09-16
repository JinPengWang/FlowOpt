"""
Entropic Optimal Transport --- the single Flow Matching ingredient
=================================================================

Public entry: ``sinkhorn_coupling(source, target, w_target, reg)``.

Everything else (greedy matching, neural vector field, online MLP training)
that used to live in this package has been removed. The algorithm does not
benefit from a "neural" baseline or a "greedy" baseline; both were merely
side-channel experiments that did not survive scrutiny. The canonical FM
algorithm only needs the Sinkhorn step implemented in `optimizer.py`.

For users that want a one-liner import:

    from flowopt.ot import sinkhorn_coupling
"""

from __future__ import annotations

import torch

__all__ = ["sinkhorn_coupling"]


def sinkhorn_coupling(source: torch.Tensor,
                      target: torch.Tensor,
                      w_target: torch.Tensor,
                      reg: float,
                      max_iter: int = 60) -> torch.Tensor:
    """Stabilized Sinkhorn-Knopp OT coupling.

    Solves  min_{Pi >= 0} <Pi, C> + reg * H(Pi)
               subject to  Pi 1 = 1/N,  Pi^T 1 = w_target.

    Returns Pi*  (|source| x |target|).

    Notes
    -----
    * The cost matrix is re-scaled by its median so that ``reg`` is a
      meaningful "entropy weight" regardless of the problem scale.
    * Update is in the log domain with ``1e-30`` floors, which avoids
      catastrophic underflow under float32.
    """
    N, _ = source.shape
    M, _ = target.shape

    s_sq = (source ** 2).sum(dim=-1, keepdim=True)
    t_sq = (target ** 2).sum(dim=-1, keepdim=True).t()
    cost = (s_sq + t_sq - 2.0 * source @ target.t()).clamp_min(0.0)
    cost = cost / (cost.median() + 1e-12)

    p = torch.full((N,), 1.0 / N, device=source.device, dtype=source.dtype)
    q = w_target / (w_target.sum() + 1e-12)

    u = torch.zeros(N, device=source.device, dtype=source.dtype)
    v = torch.zeros(M, device=source.device, dtype=source.dtype)
    log_p = torch.log(p + 1e-30)
    log_q = torch.log(q + 1e-30)

    for _ in range(max_iter):
        u_new = reg * (log_p - torch.logsumexp((v.unsqueeze(0) - cost) / reg, dim=1))
        v = reg * (log_q - torch.logsumexp((u_new.unsqueeze(1) - cost) / reg, dim=0))
        u = u_new

    return torch.exp((u.unsqueeze(1) + v.unsqueeze(0) - cost) / reg)
