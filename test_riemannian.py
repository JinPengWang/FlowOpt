import torch
import numpy as np
from flowopt.ot import sinkhorn_ot
from flowopt.benchmarks import Sphere, Ackley, Rosenbrock, Rastrigin

class RiemannianFlowOpt:
    """
    Riemannian Flow Matching Optimization (RFM-Opt).
    
    Unites:
      1. Riemannian Metric Learning (Covariance Adaptation):
         Adapts metric tensor G = C^{-1} to reshape the landscape and conquer ill-conditioned valleys.
      2. Entropic Optimal Transport (Sinkhorn) on Riemannian Manifolds:
         Computes minimal-energy displacement interpolation d_C^2(x, y).
      3. Non-Parametric Velocity Field:
         Zero-backprop conditional expectation with Mahalanobis kernel.
      4. Kinetic Velocity Momentum:
         Accelerates convergence along curved valleys.
      5. Repulsive Dispersion:
         Prevents premature mode collapse on multi-modal benchmarks.
    """
    def __init__(
        self,
        dim,
        pop_size=40,
        n_candidates=80,
        reg_ot=0.03,
        repulsive_weight=0.01,
        beta_base=5.0,
        bounds=None,
        device=None,
        seed=None
    ):
        self.dim = dim
        self.pop_size = pop_size
        self.n_candidates = n_candidates
        self.reg_ot = reg_ot
        self.repulsive_weight = repulsive_weight
        self.beta_base = beta_base
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        if bounds is not None:
            self.lb = torch.as_tensor(bounds[0], dtype=torch.float32, device=self.device)
            self.ub = torch.as_tensor(bounds[1], dtype=torch.float32, device=self.device)
        else:
            self.lb = torch.full((dim,), -5.0, dtype=torch.float32, device=self.device)
            self.ub = torch.full((dim,), 5.0, dtype=torch.float32, device=self.device)
            
        self.sigma = 0.3 * torch.mean(self.ub - self.lb).item()
        self.C = torch.eye(dim, device=self.device, dtype=torch.float32)
        self.best_x = None
        self.best_f = float("inf")
        self.history_best = []
        self.eval_count = 0
        self.velocity_momentum = None

    def sample_gaussian_cholesky(self, mean, n_samples):
        # Sample z ~ N(0, I), return mean + sigma * (L @ z)
        # Stabilize C with diagonal regularization
        C_reg = self.C + 1e-6 * torch.eye(self.dim, device=self.device)
        try:
            L = torch.linalg.cholesky(C_reg)
        except:
            # Fallback to eigendecomposition if not strictly positive definite
            evals, evecs = torch.linalg.eigh(C_reg)
            evals = torch.clamp(evals, min=1e-8)
            L = evecs @ torch.diag(torch.sqrt(evals))
            
        z = torch.randn(n_samples, self.dim, device=self.device)
        scaled_noise = torch.matmul(z, L.t()) * self.sigma
        return mean + scaled_noise

    def optimize(self, objective_fn, max_evals=10000):
        # Initialize
        r = torch.rand(self.pop_size, self.dim, device=self.device)
        particles = self.lb + r * (self.ub - self.lb)
        f_particles = objective_fn(particles)
        self.eval_count += self.pop_size
        self.velocity_momentum = torch.zeros_like(particles)
        
        min_val, min_idx = torch.min(f_particles, dim=0)
        self.best_f = float(min_val.item())
        self.best_x = particles[min_idx].clone()
        self.history_best.append((self.eval_count, self.best_f))
        
        c_cov = 2.0 / (self.dim ** 2 + 10.0)
        c_sigma = 0.96
        iteration = 0
        
        while self.eval_count < max_evals:
            iteration += 1
            N, D = particles.shape
            M = self.n_candidates
            
            # 1. Identify elites
            sorted_idx = torch.argsort(f_particles)
            n_elite = max(3, N // 4)
            elites = particles[sorted_idx[:n_elite]]
            elite_mean = torch.mean(elites, dim=0)
            
            # Update Covariance matrix (Rank-mu metric adaptation)
            diff_elite = (elites - elite_mean) / (self.sigma + 1e-8)
            C_emp = torch.matmul(diff_elite.t(), diff_elite) / n_elite
            self.C = (1.0 - c_cov) * self.C + c_cov * C_emp
            
            # 2. Candidate generation
            cands = [particles]
            n_mut = M - N
            n_riemann = int(0.55 * n_mut)
            n_diff = int(0.30 * n_mut)
            n_cauchy = n_mut - n_riemann - n_diff
            
            # Riemannian Gaussian mutations around elite mean and best_x
            if n_riemann > 0:
                y_riemann = self.sample_gaussian_cholesky(self.best_x.unsqueeze(0), n_riemann)
                cands.append(torch.clamp(y_riemann, self.lb, self.ub))
                
            # Differential contour mutations
            if n_diff > 0:
                r1 = torch.randint(0, N, (n_diff,), device=self.device)
                r2 = torch.randint(0, N, (n_diff,), device=self.device)
                diff = particles[r1] - particles[r2]
                y_diff = self.best_x.unsqueeze(0) + 0.6 * diff
                cands.append(torch.clamp(y_diff, self.lb, self.ub))
                
            # Cauchy global jumps
            if n_cauchy > 0:
                rand_idx = torch.randint(0, N, (n_cauchy,), device=self.device)
                cauchy_noise = torch.tan(torch.pi * (torch.rand(n_cauchy, D, device=self.device) - 0.5))
                cauchy_noise = torch.clamp(cauchy_noise, min=-4.0, max=4.0)
                y_cauchy = particles[rand_idx] + cauchy_noise * (self.sigma * 1.5)
                cands.append(torch.clamp(y_cauchy, self.lb, self.ub))
                
            all_cands = torch.cat(cands, dim=0)
            
            # 3. Evaluate new candidates
            f_cands = torch.empty(all_cands.shape[0], device=self.device)
            f_cands[:N] = f_particles
            
            new_cands = all_cands[N:]
            budget_left = max_evals - self.eval_count
            if len(new_cands) > budget_left:
                new_cands = new_cands[:budget_left]
                all_cands = all_cands[:N + budget_left]
                f_cands = f_cands[:N + budget_left]
                
            if len(new_cands) > 0:
                f_new = objective_fn(new_cands)
                self.eval_count += len(new_cands)
                f_cands[N:N + len(new_cands)] = f_new
                
            # 1/5th Rule Adaptive Step Size
            n_success = (f_cands[N:] < f_particles.median()).sum().item()
            success_rate = n_success / max(1, len(new_cands))
            if success_rate > 0.22:
                self.sigma = min(self.sigma * 1.08, 0.4 * torch.mean(self.ub - self.lb).item())
            else:
                self.sigma = max(self.sigma * c_sigma, 1e-12)
                
            # Update best
            curr_best_val, curr_best_idx = torch.min(f_cands, dim=0)
            if curr_best_val.item() < self.best_f:
                self.best_f = float(curr_best_val.item())
                self.best_x = all_cands[curr_best_idx].clone()
            self.history_best.append((self.eval_count, self.best_f))
            
            if self.eval_count >= max_evals:
                break
                
            # 4. Rank Gibbs Boltzmann weights
            M_curr = f_cands.shape[0]
            ranks = torch.zeros(M_curr, device=self.device)
            ranks[torch.argsort(f_cands)] = torch.linspace(0.0, 1.0, steps=M_curr, device=self.device)
            beta = self.beta_base * (1.0 + 3.0 * (self.eval_count / max_evals))
            weights = torch.exp(-beta * ranks)
            weights = weights / (weights.sum() + 1e-12)
            
            # 5. Optimal Transport Pairing
            _, matched_targets = sinkhorn_ot(particles, all_cands, weights_target=weights, reg=self.reg_ot)
            
            # 6. Flow Matching Velocity Field Integration
            u = matched_targets - particles
            
            # Repulsive Nadaraya-Watson Flow
            # Mahalanobis / Euclidean kernel
            dist_sq = torch.cdist(particles, particles, p=2) ** 2
            h = torch.median(dist_sq.sqrt()) / (np.sqrt(2.0 * np.log(N + 1.0)) + 1e-6)
            h = max(float(h.item()), 1e-4)
            kernel = torch.exp(-dist_sq / (2.0 * h ** 2))
            weights_k = kernel / (kernel.sum(dim=1, keepdim=True) + 1e-12)
            v_transport = torch.matmul(weights_k, u)
            
            # Stein repulsion
            diff_p = particles.unsqueeze(1) - particles.unsqueeze(0)
            v_rep = (weights_k.unsqueeze(-1) * diff_p).sum(dim=1) / (h + 1e-6)
            v_eff = v_transport + self.repulsive_weight * v_rep
            
            # Momentum integration
            self.velocity_momentum = 0.55 * self.velocity_momentum + 0.45 * v_eff
            new_particles = torch.clamp(particles + 0.85 * self.velocity_momentum, self.lb, self.ub)
            
            f_new_p = objective_fn(new_particles)
            self.eval_count += len(new_particles)
            
            improved = f_new_p < f_particles
            particles[improved] = new_particles[improved]
            f_particles[improved] = f_new_p[improved]
            
            particles[0] = self.best_x.clone()
            f_particles[0] = self.best_f
            
            p_min_val = torch.min(f_particles).item()
            if p_min_val < self.best_f:
                self.best_f = p_min_val
                self.best_x = particles[torch.argmin(f_particles)].clone()
            self.history_best.append((self.eval_count, self.best_f))
            
        return {"best_f": self.best_f, "history": self.history_best}

if __name__ == "__main__":
    dim = 10
    evals = 10000
    for name, fn in [("Sphere", Sphere(dim=dim)), ("Rosenbrock", Rosenbrock(dim=dim)), ("Ackley", Ackley(dim=dim)), ("Rastrigin", Rastrigin(dim=dim))]:
        opt = RiemannianFlowOpt(dim=dim, bounds=fn.bounds, seed=42)
        res = opt.optimize(fn, max_evals=evals)
        print(f"RiemannianFlowOpt on {name:10s} : best_f = {res['best_f']:.6e}")
