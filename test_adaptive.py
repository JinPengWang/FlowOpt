import torch
import numpy as np
from flowopt.ot import sinkhorn_ot
from flowopt.vector_field import NonParametricVectorField
from flowopt.benchmarks import Sphere, Ackley, Rosenbrock, Rastrigin

class AdaptiveFlowOpt:
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
            
        self.vector_field = NonParametricVectorField(repulsive_weight=self.repulsive_weight)
        self.sigma = 0.3 * torch.mean(self.ub - self.lb).item()
        self.best_x = None
        self.best_f = float("inf")
        self.history_best = []
        self.eval_count = 0
        self.velocity_momentum = None

    def optimize(self, objective_fn, max_evals=10000):
        # 1. Initialize
        r = torch.rand(self.pop_size, self.dim, device=self.device)
        particles = self.lb + r * (self.ub - self.lb)
        f_particles = objective_fn(particles)
        self.eval_count += self.pop_size
        self.velocity_momentum = torch.zeros_like(particles)
        
        min_val, min_idx = torch.min(f_particles, dim=0)
        self.best_f = float(min_val.item())
        self.best_x = particles[min_idx].clone()
        self.history_best.append((self.eval_count, self.best_f))
        
        iteration = 0
        c_sigma = 0.95
        
        while self.eval_count < max_evals:
            iteration += 1
            N, D = particles.shape
            M = self.n_candidates
            
            # Elites
            sorted_idx = torch.argsort(f_particles)
            n_elite = max(3, N // 4)
            elites = particles[sorted_idx[:n_elite]]
            elite_std = torch.std(elites, dim=0) + 1e-8
            
            # 2. Candidate generation
            cands = [particles] # retain particles
            n_mut = M - N
            
            # Differential flow direction
            n_diff = int(0.35 * n_mut)
            n_gauss = int(0.45 * n_mut)
            n_cauchy = n_mut - n_diff - n_gauss
            
            if n_diff > 0:
                r1 = torch.randint(0, N, (n_diff,), device=self.device)
                r2 = torch.randint(0, N, (n_diff,), device=self.device)
                diff = particles[r1] - particles[r2]
                y_diff = self.best_x.unsqueeze(0) + 0.7 * diff
                cands.append(torch.clamp(y_diff, self.lb, self.ub))
                
            if n_gauss > 0:
                # Multi-scale Gaussian centered at elites
                rand_elite_idx = torch.randint(0, n_elite, (n_gauss,), device=self.device)
                noise = torch.randn(n_gauss, D, device=self.device) * (self.sigma * (elite_std / (torch.mean(elite_std) + 1e-8)))
                y_gauss = elites[rand_elite_idx] + noise
                cands.append(torch.clamp(y_gauss, self.lb, self.ub))
                
            if n_cauchy > 0:
                rand_idx = torch.randint(0, N, (n_cauchy,), device=self.device)
                cauchy_noise = torch.tan(torch.pi * (torch.rand(n_cauchy, D, device=self.device) - 0.5))
                cauchy_noise = torch.clamp(cauchy_noise, min=-5.0, max=5.0)
                y_cauchy = particles[rand_idx] + cauchy_noise * (self.sigma * 2.0)
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
                
            # Success check for 1/5th adaptive step-size
            n_successful = (f_cands[N:] < f_particles.median()).sum().item()
            success_ratio = n_successful / max(1, len(new_cands))
            if success_ratio > 0.25:
                self.sigma = min(self.sigma * 1.1, 0.5 * torch.mean(self.ub - self.lb).item())
            else:
                self.sigma = max(self.sigma * c_sigma, 1e-12)
                
            # Update best
            c_min_val, c_min_idx = torch.min(f_cands, dim=0)
            if c_min_val.item() < self.best_f:
                self.best_f = float(c_min_val.item())
                self.best_x = all_cands[c_min_idx].clone()
            self.history_best.append((self.eval_count, self.best_f))
            
            if self.eval_count >= max_evals:
                break
                
            # 4. Rank Gibbs
            M_curr = f_cands.shape[0]
            ranks = torch.zeros(M_curr, device=self.device)
            ranks[torch.argsort(f_cands)] = torch.linspace(0.0, 1.0, steps=M_curr, device=self.device)
            beta = self.beta_base * (1.0 + 3.0 * (self.eval_count / max_evals))
            weights = torch.exp(-beta * ranks)
            weights = weights / (weights.sum() + 1e-12)
            
            # 5. Sinkhorn Optimal Transport
            _, matched_targets = sinkhorn_ot(particles, all_cands, weights_target=weights, reg=self.reg_ot)
            
            # 6. Flow Matching Velocity Field Integration
            u = matched_targets - particles
            v = self.vector_field.evaluate(particles, particles, u)
            
            # Velocity momentum
            self.velocity_momentum = 0.5 * self.velocity_momentum + 0.5 * v
            new_particles = torch.clamp(particles + 0.85 * self.velocity_momentum, self.lb, self.ub)
            
            f_new_p = objective_fn(new_particles)
            self.eval_count += len(new_particles)
            
            improved = f_new_p < f_particles
            particles[improved] = new_particles[improved]
            f_particles[improved] = f_new_p[improved]
            
            # Maintain best
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
        opt = AdaptiveFlowOpt(dim=dim, bounds=fn.bounds, seed=42)
        res = opt.optimize(fn, max_evals=evals)
        print(f"AdaptiveFlowOpt on {name:10s} : best_f = {res['best_f']:.6e}")
