import torch
import numpy as np

class BenchmarkFunction:
    def __init__(self, name, dim, bounds, optimal_value=0.0):
        self.name = name
        self.dim = dim
        self.bounds = bounds
        self.optimal_value = optimal_value

    def __call__(self, x):
        raise NotImplementedError


class Sphere(BenchmarkFunction):
    """f(x) = sum(x_i^2)"""
    def __init__(self, dim=30):
        super().__init__("Sphere", dim, (-5.12, 5.12), optimal_value=0.0)

    def __call__(self, x):
        return torch.sum(x ** 2, dim=-1)


class Rosenbrock(BenchmarkFunction):
    """f(x) = sum(100 * (x_{i+1} - x_i^2)^2 + (x_i - 1)^2)"""
    def __init__(self, dim=30):
        super().__init__("Rosenbrock", dim, (-2.048, 2.048), optimal_value=0.0)

    def __call__(self, x):
        x0 = x[..., :-1]
        x1 = x[..., 1:]
        return torch.sum(100.0 * (x1 - x0 ** 2) ** 2 + (x0 - 1.0) ** 2, dim=-1)


class Rastrigin(BenchmarkFunction):
    """f(x) = 10 * d + sum(x_i^2 - 10 * cos(2 * pi * x_i))"""
    def __init__(self, dim=30):
        super().__init__("Rastrigin", dim, (-5.12, 5.12), optimal_value=0.0)

    def __call__(self, x):
        d = x.shape[-1]
        return 10.0 * d + torch.sum(x ** 2 - 10.0 * torch.cos(2.0 * np.pi * x), dim=-1)


class Ackley(BenchmarkFunction):
    """f(x) = -20 exp(-0.2 sqrt(1/d sum x_i^2)) - exp(1/d sum cos(2 pi x_i)) + 20 + e"""
    def __init__(self, dim=30):
        super().__init__("Ackley", dim, (-32.768, 32.768), optimal_value=0.0)

    def __call__(self, x):
        d = x.shape[-1]
        sum_sq = torch.sum(x ** 2, dim=-1)
        sum_cos = torch.sum(torch.cos(2.0 * np.pi * x), dim=-1)
        term1 = -20.0 * torch.exp(-0.2 * torch.sqrt(sum_sq / d + 1e-12))
        term2 = -torch.exp(sum_cos / d)
        return term1 + term2 + 20.0 + np.e


class Griewank(BenchmarkFunction):
    """f(x) = 1 + 1/4000 sum x_i^2 - prod cos(x_i / sqrt(i))"""
    def __init__(self, dim=30):
        super().__init__("Griewank", dim, (-600.0, 600.0), optimal_value=0.0)

    def __call__(self, x):
        d = x.shape[-1]
        device = x.device
        i_vec = torch.sqrt(torch.arange(1, d + 1, device=device, dtype=x.dtype))
        sum_term = torch.sum(x ** 2, dim=-1) / 4000.0
        prod_term = torch.prod(torch.cos(x / i_vec), dim=-1)
        return sum_term - prod_term + 1.0


class Schwefel(BenchmarkFunction):
    """f(x) = 418.9829 * d - sum(x_i * sin(sqrt(|x_i|)))"""
    def __init__(self, dim=30):
        super().__init__("Schwefel", dim, (-500.0, 500.0), optimal_value=0.0)

    def __call__(self, x):
        d = x.shape[-1]
        term = torch.sum(x * torch.sin(torch.sqrt(torch.abs(x) + 1e-12)), dim=-1)
        return 418.9829 * d - term


class Levy(BenchmarkFunction):
    """Levy benchmark function with multiple local optima."""
    def __init__(self, dim=30):
        super().__init__("Levy", dim, (-10.0, 10.0), optimal_value=0.0)

    def __call__(self, x):
        w = 1.0 + (x - 1.0) / 4.0
        d = x.shape[-1]
        w1 = w[..., 0]
        wd = w[..., -1]
        w_mid = w[..., :-1]
        
        term1 = torch.sin(np.pi * w1) ** 2
        term2 = torch.sum((w_mid - 1.0) ** 2 * (1.0 + 10.0 * torch.sin(np.pi * w_mid + 1.0) ** 2), dim=-1)
        term3 = (wd - 1.0) ** 2 * (1.0 + torch.sin(2.0 * np.pi * wd) ** 2)
        return term1 + term2 + term3


class LennardJonesCluster(BenchmarkFunction):
    """
    Lennard-Jones molecular potential energy minimization.
    A notoriously challenging real-world non-convex optimization problem in chemical physics.
    Particles in 3D: N_atoms = dim // 3.
    V(r) = 4 * sum_{i < j} [ (1/r_ij)^12 - (1/r_ij)^6 ]
    """
    def __init__(self, n_atoms=6):
        dim = n_atoms * 3
        super().__init__(f"LennardJones_{n_atoms}", dim, (-2.5, 2.5), optimal_value=-12.712)
        self.n_atoms = n_atoms

    def __call__(self, x):
        # x: (B, 3 * n_atoms)
        B = x.shape[0]
        coords = x.view(B, self.n_atoms, 3)
        
        # Pairwise distance matrix
        diff = coords.unsqueeze(2) - coords.unsqueeze(1) # (B, n, n, 3)
        dist_sq = torch.sum(diff ** 2, dim=-1) # (B, n, n)
        
        # Upper triangular mask for i < j
        mask = torch.triu(torch.ones(self.n_atoms, self.n_atoms, device=x.device, dtype=torch.bool), diagonal=1)
        
        dist_sq_pairs = dist_sq[:, mask] # (B, pairs)
        dist_sq_pairs = torch.clamp(dist_sq_pairs, min=1e-6)
        
        r2_inv = 1.0 / dist_sq_pairs
        r6_inv = r2_inv ** 3
        r12_inv = r6_inv ** 2
        
        v_pairs = 4.0 * (r12_inv - r6_inv)
        energy = torch.sum(v_pairs, dim=-1)
        return energy


def get_all_benchmarks(dim=30):
    return [
        Sphere(dim=dim),
        Rosenbrock(dim=dim),
        Rastrigin(dim=dim),
        Ackley(dim=dim),
        Griewank(dim=dim),
        Schwefel(dim=dim),
        Levy(dim=dim),
        LennardJonesCluster(n_atoms=dim // 3 if dim >= 6 else 4)
    ]
