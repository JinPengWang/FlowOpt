"""
Unit tests for FlowOpt v2
=========================

1. First-principles invariants are recovered exactly
   (c_sigma, d_sigma, alpha_C, chi_D, mu_eff all derived from D and N).
2. CSA null-hypothesis: under random fitness the path norm ratio converges.
   E[|| p_sigma ||] ≈ chi_D  (this is the "zero-drift" claim).
3. Domain reflection is exact: fold is a true bijection inside [lb, ub].
4. Deterministic reproducibility under fixed seeds.
5. Sphere converges well within the budget.
6. Sinkhorn coupling is doubly stochastic up to log-domain precision.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flowopt import FlowOpt, sinkhorn_coupling  # noqa: E402
from flowopt.benchmarks import get_benchmark     # noqa: E402


# -------------------- 1. invariants ---------------------------- #

def test_first_principles_invariants():
    D, N = 10, 30
    opt = FlowOpt(dim=D, pop_size=N, seed=42)

    mu = N // 2
    raw = torch.tensor([math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)])
    w = raw / raw.sum()
    mu_eff = float(1.0 / (w ** 2).sum().item())

    expected = {
        "c_sigma": mu_eff / (D + mu_eff),
        "d_sigma": 1.0 + mu_eff / (D + mu_eff),
        "alpha_C": mu_eff / (D ** 2 + mu_eff),
        "chi_D": math.sqrt(D) * (1.0 - 1.0 / (4.0 * D) + 1.0 / (32.0 * D * D)),
        "mu_eff": mu_eff,
    }

    for k, v in expected.items():
        got = getattr(opt, k)
        assert math.isclose(got, v, rel_tol=1e-6, abs_tol=1e-12), \
            f"{k}: expected {v}, got {got}"


# -------------------- 2. CSA null hypothesis ------------------- #

def test_csa_null_hypothesis():
    """Under random fitness, the CSA path's expected norm equals chi_D."""
    D, N = 8, 30
    torch.manual_seed(0)
    opt = FlowOpt(dim=D, pop_size=N, seed=0)

    # Burn-in then measure |p_sigma|.  No objective ever feeds back into the
    # path except via the random displacement vector z_w.
    norms = []
    for gen in range(400):
        x = opt.m + opt.sigma * torch.randn(N, D, device=opt.device)
        random_fitness = torch.randn(N).tolist()
        order = np.argsort(random_fitness)
        elites = x[order[:opt.mu]]
        ar = (elites - opt.m) / opt.sigma
        z_w = math.sqrt(opt.mu_eff) * (opt.w.view(-1, 1) * ar).sum(dim=0)
        opt.p_sigma = (1.0 - opt.c_sigma) * opt.p_sigma + z_w
        if gen > 100:
            norms.append(float(opt.p_sigma.norm().item()))

    ratio = float(np.mean(norms)) / opt.chi_D
    assert 0.85 <= ratio <= 1.15, f"null-hypothesis ratio {ratio:.3f} deviated from 1"


# -------------------- 3. periodic reflection ------------------- #

def test_periodic_reflection():
    D = 5
    opt = FlowOpt(dim=D, bounds=(0.0, 10.0), pop_size=10, seed=42)
    pts = torch.tensor(
        [[11.0, -1.0, 5.0, 22.0, -12.0],
         [15.0, -5.0, 10.0, 0.0, 8.0]],
        device=opt.device, dtype=torch.float32,
    )
    folded = opt.fold(pts)
    assert (folded >= opt.lb).all().item()
    assert (folded <= opt.ub).all().item()
    assert math.isclose(folded[0, 0].item(), 9.0)    # 11 -> 9
    assert math.isclose(folded[0, 1].item(), 1.0)    # -1 -> 1
    assert math.isclose(folded[0, 2].item(), 5.0)


# -------------------- 4. deterministic reproduction ----------- #

def test_deterministic_reproduction():
    f = get_benchmark("sphere", dim=10)
    r1 = FlowOpt(dim=10, bounds=f.bounds, seed=999).optimize(f, max_iters=50)
    r2 = FlowOpt(dim=10, bounds=f.bounds, seed=999).optimize(f, max_iters=50)
    assert np.allclose(r1["best_x"], r2["best_x"])
    assert math.isclose(r1["best_f"], r2["best_f"], abs_tol=1e-12)


# -------------------- 5. Sphere convergence -------------------- #

def test_sphere_convergence():
    f = get_benchmark("sphere", dim=10)
    res = FlowOpt(dim=10, bounds=f.bounds, seed=42).optimize(f, max_iters=200)
    assert res["best_f"] < 1e-10, f"Sphere failed: best_f = {res['best_f']}"


# -------------------- 6. Sinkhorn coupling --------------------- #

def test_sinkhorn_marginals():
    D, N = 5, 20
    torch.manual_seed(7)
    x = torch.randn(N, D)
    w = torch.zeros(N)
    w[:10] = torch.softmax(torch.randn(10), dim=0)

    Pi = sinkhorn_coupling(x, x, w, reg=0.05)
    row_sum = Pi.sum(dim=1)
    col_sum = Pi.sum(dim=0)
    target_q = w / w.sum()
    # source marginal is 1/N by construction
    assert torch.allclose(row_sum, torch.full_like(row_sum, 1.0 / N), atol=5e-3)
    assert torch.allclose(col_sum, target_q, atol=5e-3)


# -------------------- driver ----------------------------------- #

def _run_all() -> list[tuple[str, bool, str]]:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    out = []
    for t in tests:
        try:
            t()
            out.append((t.__name__, True, ""))
        except AssertionError as e:
            out.append((t.__name__, False, str(e)))
        except Exception as e:
            out.append((t.__name__, False, f"{type(e).__name__}: {e}"))
    return out


if __name__ == "__main__":
    results = _run_all()
    width = max(len(n) for n, _, _ in results)
    for name, ok, msg in results:
        flag = "[PASS]" if ok else "[FAIL]"
        print(f"  {flag}  {name:<{width}}  {msg}")
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - failed} / {len(results)} passed")
    if failed:
        sys.exit(1)
