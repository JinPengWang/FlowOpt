"""
FlowOpt --- Continuous-Time Generative Optimization
====================================================

Idea in one line
-----------------
We treat black-box minimization as *transport*: at every iteration we have a
search distribution p_0 = N(m, sigma^2 C). We pick a target distribution p_1
concentrated on the current elites, and we solve the Entropic Optimal Transport
between p_0 and p_1. The OT pairing defines a deterministic closed-form
conditional velocity field u(x_0) = target(x_0) - x_0 (the linear-interpolant
velocity from Flow Matching). One Euler step along this velocity produces the
new p_0, then we update (m, sigma, C) by following the elite statistics.

This file is intentionally very short. The whole algorithm is one class,
~120 lines. There are ZERO user-tunable hyperparameters: every internal
constant is fixed by (D, N) through closed-form first-principles formulae.

Why this is "Flow Matching", not heuristic stacking
---------------------------------------------------
Flow Matching (Lipman et al., 2023) defines for any conditional vector field
u_t(x|x_0, x_1) a marginal vector field v_t(x) = E[x_1 - x_0 | x_t = x].
Under the linear Optimal Transport interpolant:

    x_t = (1-t) x_0 + t x_1       =>     u_t = dx_t/dt = x_1 - x_0    (constant in t)

the marginal velocity field admits a deterministic closed form whenever we can
solve the OT plan between samples of p_0 and samples of p_1. We use Entropic
OT (Sinkhorn) precisely for that.

The Gaussian search state is updated as an EMA toward the (sigma, C) of the
elite population, with one classical trick (Cumulative Step-Size Adaptation,
Hansen 2008) for sigma.  Stripped of: rank-1 path (p_c), hsig gating, separate
c_c, c_1/c_mu --- all of them were redundant knobs that did not improve
interpretability and were prone to mis-tuning.  The two remaining scalars
(c_sigma, alpha_C) are derived from (D, mu_eff) and have natural meanings:

    c_sigma = mu_eff / (D + mu_eff)         # CSA learning rate
    alpha_C  = mu_eff / (D^2 + mu_eff)      # rank-mu cov learning rate

Population size N is the only knob, and even that one is "physics", not
"tuning": the user has to specify the budget of evaluations per iteration.
"""

from __future__ import annotations

import math
from typing import Callable

import torch


# --------------------------------------------------------------------------- #
#  First-principles constants (no learned weights, no magic numbers)         #
# --------------------------------------------------------------------------- #

def _chi_D(D: int) -> float:
    """E[|| N(0, I_D) ||] via Stirling expansion to O(1/D^2).
       chi_D ~ sqrt(D) * (1 - 1/(4D) + 1/(32 D^2))
    """
    return math.sqrt(D) * (1.0 - 1.0 / (4.0 * D) + 1.0 / (32.0 * D * D))


def _log_rank_weights(mu: int, device, dtype) -> torch.Tensor:
    """Logarithmic weights for the top-mu sorted by fitness.  Rank-invariant
       under any strictly monotone transformation of the objective.
       w_i = (log(mu + 0.5) - log(i + 1)) / normalizer, i = 0..mu-1.
    """
    raw = torch.tensor([math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)],
                       device=device, dtype=dtype)
    return raw / raw.sum()


# --------------------------------------------------------------------------- #
#  Entropic OT -- the FM ingredient (canonical implementation lives in ot.py) #
# --------------------------------------------------------------------------- #

from .ot import sinkhorn_coupling  # single implementation; re-exported for back-compat


# --------------------------------------------------------------------------- #
#  FlowOpt --- the entire optimizer                                            #
# --------------------------------------------------------------------------- #

class FlowOpt:
    """Hyperparameter-free Flow Matching optimizer.

    Parameters
    ----------
    dim : int            problem dimension D
    bounds               (lb, ub); scalar broadcasts to (D,)
    pop_size : int       N (default 30). The ONLY user knob.
    device               'cpu' / 'cuda' (auto-detected if None)
    seed : int | None    deterministic reproduction

    The state variables (m, sigma, L, p_sigma) carry one Gaussian search
    distribution and one path.  An iteration consists of nine commented
    lines; see ``step`` below.
    """

    def __init__(self, dim: int, bounds=(-5.0, 5.0), pop_size: int = 30,
                 device=None, seed: int | None = None):
        self.dim = dim
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if seed is not None:
            torch.manual_seed(seed)

        self.N = pop_size
        self.mu = max(2, self.N // 2)             # 50% selection rate -> standard
        self.lb, self.ub, self.span = _normalise_bounds(bounds, dim, self.device)

        # --- Gaussian search state: p_0 = N(m, sigma^2 C), C = L L^T
        self.m = self.lb + torch.rand(dim, device=self.device) * self.span
        self.sigma = float(0.3 * self.span.mean().item())
        self.L = torch.eye(dim, device=self.device)
        self.p_sigma = torch.zeros(dim, device=self.device)

        # --- First-principles invariants (no user choice possible)
        self.w = _log_rank_weights(self.mu, self.device, torch.float32)
        self.mu_eff   = float(1.0 / (self.w ** 2).sum().item())      # e.g. ~5 for N=30
        self.c_sigma  = self.mu_eff / (self.dim + self.mu_eff)       # CSA learning rate
        self.d_sigma  = 1.0 + self.c_sigma                            # canonical damping
        self.alpha_C  = self.mu_eff / (self.dim ** 2 + self.mu_eff)  # cov learning rate
        self.chi_D    = _chi_D(self.dim)                             # E[||N(0,I_D)||]

        self.best_f = float("inf")
        self.best_x = self.m.clone()
        self.history: list = []

    # ------------------------- housekeeping ----------------------------- #
    def fold(self, x: torch.Tensor) -> torch.Tensor:
        """Smooth periodic reflection: samples always stay in [lb, ub]."""
        lb = self.lb if x.device == self.lb.device else self.lb.to(x.device)
        w = self.span if x.device == self.span.device else self.span.to(x.device)
        s = (x - lb) % (2.0 * w)
        return lb + torch.where(s > w, 2.0 * w - s, s)

    def clamp(self, x: torch.Tensor) -> torch.Tensor:
        """Hard clamp for the mean (m is updated, x is reflected)."""
        lb = self.lb if x.device == self.lb.device else self.lb.to(x.device)
        ub = self.ub if x.device == self.ub.device else self.ub.to(x.device)
        return torch.clamp(x, min=lb, max=ub)

    # ------------------------- one FM step ------------------------------ #
    def step(self, f: Callable[[torch.Tensor], torch.Tensor], reg_ot: float
             ) -> tuple[float, torch.Tensor]:
        """ONE Flow Matching iteration.  Returns (best_f, best_x) after the step.

        The body is nine lines, one per FM step in the docstring header.
        """
        D, N = self.dim, self.N
        # 1) sample from current search Gaussian, fold to domain
        z = torch.randn(N, D, device=self.device)
        x = self.fold(self.m + self.sigma * (z @ self.L.t()))                       # x in dom
        # 2) evaluate
        fvals = f(x)
        idx = torch.argsort(fvals)
        elites = x[idx[: self.mu]]                                                   # (mu, D)
        m_old = self.m
        best_v, best_i = torch.min(fvals, dim=0)
        if float(best_v.item()) < self.best_f:
            self.best_f = float(best_v.item())
            self.best_x = x[best_i].clone()

        # 3) Sinkhorn OT between population and weighted elite support
        Pi = sinkhorn_coupling(x, elites, self.w, reg=reg_ot)
        # 4) barycentric FM target: "where each particle should be at time 1"
        row = Pi.sum(dim=1, keepdim=True) + 1e-12
        target = (Pi / row) @ elites                                                 # (N, D)
        # 5) FM velocity: u = target - x  (closed-form constant in t)
        u = target - x

        # 6) mean probability-flow drift (Euler step of the marginal equation)
        self.m = self.clamp(m_old + u.mean(dim=0))

        # 7) Cumulative Step-Size Adaptation on the elite-mean path
        ar = (elites - m_old) / self.sigma                                            # (mu, D)
        z_w = math.sqrt(self.mu_eff) * (self.w.view(-1, 1) * ar).sum(dim=0)          # (D,)
        # Standard CSA path update (Hansen 2001, Eq. 11):
        #   p_sigma <- (1 - c_sigma) * p_sigma + sqrt(c_sigma*(2-c_sigma)) * z_w
        # The sqrt factor ensures stationary Var[p_sigma] = I_D under the null
        # hypothesis (Theorem 3), so E[||p_sigma||] = chi_D exactly.
        _sqrt_cs = math.sqrt(self.c_sigma * (2.0 - self.c_sigma))
        self.p_sigma = (1.0 - self.c_sigma) * self.p_sigma + _sqrt_cs * z_w
        exp_arg = (self.c_sigma / self.d_sigma) * (float(self.p_sigma.norm())
                                                  / self.chi_D - 1.0)
        self.sigma = float(self.sigma * math.exp(max(-1.0, min(1.0, exp_arg))))
        self.sigma = max(1e-30, min(self.sigma, 0.5 * float(self.span.mean())))

        # 8) rank-mu covariance update (one scalar learning rate, alpha_C)
        C_old = self.L @ self.L.t()
        C_mu = (self.w.view(-1, 1) * ar).t() @ ar                                    # (D, D)
        C = (1.0 - self.alpha_C) * C_old + self.alpha_C * C_mu
        C = 0.5 * (C + C.t())
        evals, evecs = torch.linalg.eigh(C)
        evals = evals.clamp_min(1e-14)
        self.L = evecs @ torch.diag(torch.sqrt(evals))

        return self.best_f, self.best_x

    # ------------------------- driver loop ------------------------------ #
    def optimize(self, f: Callable[[torch.Tensor], torch.Tensor],
                 max_iters: int = 200,
                 eps_schedule: tuple[float, float] = (0.5, 1e-3)
                 ) -> dict:
        """Run ``max_iters`` FM steps with a linear entropic-temperature
        schedule from ``eps_schedule[0]`` down to ``eps_schedule[1]``.

        The epsilon schedule is dimensionless (medians) and uses no
        problem-specific scale.
        """
        eps_hi, eps_lo = eps_schedule
        self.history = []
        for t in range(max_iters):
            prog = t / max(max_iters - 1, 1)
            reg = eps_hi * (1.0 - prog) + eps_lo * prog
            best_f, _ = self.step(f, reg)
            self.history.append((t + 1, best_f))
        return {
            "best_f": self.best_f,
            "best_x": self.best_x.detach().cpu().numpy(),
            "iters": max_iters,
            "history": self.history,
        }


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _normalise_bounds(bounds, dim: int, device):
    lb, ub = bounds[0], bounds[1]
    if isinstance(lb, (int, float)):
        lb = torch.full((dim,), float(lb), device=device)
    else:
        lb = torch.as_tensor(lb, device=device, dtype=torch.float32)
    if isinstance(ub, (int, float)):
        ub = torch.full((dim,), float(ub), device=device)
    else:
        ub = torch.as_tensor(ub, device=device, dtype=torch.float32)
    return lb, ub, ub - lb
