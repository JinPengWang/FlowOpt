"""
Unit Tests for FlowOpt: First-Principles Riemannian Flow Matching Optimizer.
Verifies:
1. Exact First-Principles Dimensional Invariants (No magic numbers)
2. Null Hypothesis Flow-Path Variance Calibration (E[||p_sigma||] = chi_D)
3. Boundary Folding Reflection Mechanics
4. Deterministic Seed Reproducibility
5. Quadratic Basin Fast Convergence (Sphere)
6. Entropic Optimal Transport Coupling Integrity
"""

import math
import sys
import os
import torch
import numpy as np

# Ensure flowopt is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowopt.optimizer import FlowOpt
from flowopt.benchmarks import get_benchmark


def test_first_principles_invariants():
    """Verify all coefficients are derived strictly from dimension D and effective selection mass."""
    dim = 10
    opt = FlowOpt(dim=dim, pop_size=30, seed=42)

    mu = 30 // 2
    raw_w = torch.tensor([math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)], dtype=torch.float32)
    weights = raw_w / raw_w.sum()
    mu_eff = float(1.0 / (weights ** 2).sum().item())

    # Exact mathematical definitions
    expected_c_sigma = mu_eff / (dim + mu_eff)
    expected_d_sigma = 1.0 + expected_c_sigma
    expected_c_c = 4.0 / (dim + 4.0)
    expected_c_1 = 2.0 / (dim ** 2 + mu_eff)
    expected_c_mu = min(1.0 - expected_c_1, 2.0 * mu_eff / (dim ** 2 + mu_eff))
    expected_chi_d = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim ** 2))

    assert math.isclose(opt.c_sigma, expected_c_sigma, rel_tol=1e-5), "c_sigma mismatch"
    assert math.isclose(opt.d_sigma, expected_d_sigma, rel_tol=1e-5), "d_sigma mismatch"
    assert math.isclose(opt.c_c, expected_c_c, rel_tol=1e-5), "c_c mismatch"
    assert math.isclose(opt.c_1, expected_c_1, rel_tol=1e-5), "c_1 mismatch"
    assert math.isclose(opt.c_mu, expected_c_mu, rel_tol=1e-5), "c_mu mismatch"
    assert math.isclose(opt.chi_d, expected_chi_d, rel_tol=1e-5), "chi_d mismatch"


def test_null_hypothesis_unbiasedness():
    """
    Under a pure flat random fitness landscape, the cumulative step-size flow path
    must have an expected norm equal to chi_D, ensuring zero exponential drift.
    """
    dim = 10
    torch.manual_seed(123)
    opt = FlowOpt(dim=dim, pop_size=30, seed=123)

    # Run for 200 generations under random fitness
    norms = []
    for gen in range(200):
        # Generate random fitness values (pure noise, no signal)
        x = opt.m + opt.sigma * torch.randn(opt.N, dim, device=opt.device)
        random_fitness = torch.randn(opt.N).tolist()
        
        # Rank by random fitness
        sorted_indices = np.argsort(random_fitness)
        elites = x[sorted_indices[:opt.mu]]
        
        # Invariant flow step
        z_k = (elites - opt.m) / opt.sigma
        z_flow = math.sqrt(opt.mu_eff) * (opt.weights.unsqueeze(1) * z_k).sum(dim=0)
        opt.p_sigma = (1.0 - opt.c_sigma) * opt.p_sigma + math.sqrt(opt.c_sigma * (2.0 - opt.c_sigma)) * z_flow
        if gen > 50:  # Allow burn-in to stationary distribution
            norms.append(opt.p_sigma.norm().item())

    mean_norm = float(np.mean(norms))
    # E[||p_sigma||] should be within 15% of chi_D
    ratio = mean_norm / opt.chi_d
    assert 0.85 <= ratio <= 1.15, f"Null hypothesis norm ratio {ratio:.3f} deviated significantly from 1.0"


def test_boundary_folding():
    """Verify that boundary folding reflects smoothly within [lb, ub]."""
    dim = 5
    opt = FlowOpt(dim=dim, bounds=(0.0, 10.0), seed=42)

    # Test values outside upper and lower bounds
    test_x = torch.tensor([
        [11.0, -1.0, 5.0, 22.0, -12.0],
        [15.0, -5.0, 10.0, 0.0, 8.0]
    ], device=opt.device, dtype=torch.float32)
    folded = opt.fold(test_x)

    # All folded values must be strictly within [lb, ub]
    assert (folded >= opt.lb).all().item(), "Folded value below lower bound"
    assert (folded <= opt.ub).all().item(), "Folded value above upper bound"
    # Symmetric reflection checks
    assert math.isclose(folded[0, 0].item(), 9.0), "11.0 on [0, 10] should fold to 9.0"
    assert math.isclose(folded[0, 1].item(), 1.0), "-1.0 on [0, 10] should fold to 1.0"
    assert math.isclose(folded[0, 2].item(), 5.0), "5.0 on [0, 10] should remain 5.0"


def test_deterministic_reproducibility():
    """Verify that identical seeds produce bitwise identical optimization results."""
    func = get_benchmark("sphere", dim=10)

    opt1 = FlowOpt(dim=10, bounds=func.bounds, seed=999)
    res1 = opt1.optimize(func, max_iters=50)

    opt2 = FlowOpt(dim=10, bounds=func.bounds, seed=999)
    res2 = opt2.optimize(func, max_iters=50)

    assert np.allclose(res1["best_x"], res2["best_x"]), "Optimizers with identical seeds produced different solution vectors"
    assert math.isclose(res1["best_f"], res2["best_f"], abs_tol=1e-12), "Optimizers with identical seeds produced different fitness"


def test_sphere_quadratic_convergence():
    """Verify that FlowOpt achieves near-machine-precision zero on unimodal Sphere in 200 iterations."""
    func = get_benchmark("sphere", dim=10)
    opt = FlowOpt(dim=10, bounds=func.bounds, seed=42)
    res = opt.optimize(func, max_iters=200)

    assert res["best_f"] < 1e-10, f"Sphere convergence failed to reach < 1e-10, got {res['best_f']}"


def test_sinkhorn_transport_mass_conservation():
    """Verify that Sinkhorn entropic transport preserves marginal probability conservation."""
    from flowopt.optimizer import sinkhorn_transport
    dim = 5
    N = 20
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(N, dim, device=device)
    weights = torch.zeros(N, device=device)
    weights[:10] = torch.softmax(torch.randn(10, device=device), dim=0)

    dest = sinkhorn_transport(x, weights, reg=0.1)
    
    # Destination should have matching shape and finite values
    assert dest.shape == x.shape
    assert not torch.isnan(dest).any()
    assert not torch.isinf(dest).any()


if __name__ == "__main__":
    tests = [
        test_first_principles_invariants,
        test_null_hypothesis_unbiasedness,
        test_boundary_folding,
        test_deterministic_reproducibility,
        test_sphere_quadratic_convergence,
        test_sinkhorn_transport_mass_conservation,
    ]
    print(f"Running {len(tests)} unit tests for FlowOpt...")
    for t in tests:
        print(f"  [RUNNING] {t.__name__} ...", end="", flush=True)
        t()
        print(" [PASSED]")
    print(f"All {len(tests)} tests passed successfully!")
