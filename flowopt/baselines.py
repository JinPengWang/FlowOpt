import torch
import numpy as np
import cma

class BaseOptimizer:
    def __init__(self, dim, bounds, pop_size=50, seed=None):
        self.dim = dim
        self.lb = bounds[0]
        self.ub = bounds[1]
        self.pop_size = pop_size
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    def optimize(self, objective_fn, max_evals=10000):
        raise NotImplementedError


class CMAESOptimizer(BaseOptimizer):
    """Covariance Matrix Adaptation Evolution Strategy (CMA-ES)"""
    def optimize(self, objective_fn, max_evals=10000):
        x0 = np.random.uniform(self.lb, self.ub, self.dim)
        sigma0 = 0.3 * (self.ub - self.lb)
        opts = {
            'bounds': [self.lb, self.ub],
            'popsize': self.pop_size,
            'maxfevals': max_evals,
            'verbose': -9,
            'seed': self.seed if self.seed is not None else 0
        }
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        
        eval_count = 0
        best_f = float("inf")
        best_x = None
        history = []
        
        while not es.stop() and eval_count < max_evals:
            candidates = es.ask()
            # Evaluate using PyTorch objective
            t_candidates = torch.tensor(np.array(candidates), dtype=torch.float32)
            if hasattr(objective_fn, 'device'):
                t_candidates = t_candidates.to(objective_fn.device)
            fitness = objective_fn(t_candidates).detach().cpu().numpy()
            es.tell(candidates, fitness)
            eval_count += len(candidates)
            
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_f:
                best_f = float(fitness[min_idx])
                best_x = np.copy(candidates[min_idx])
            history.append((eval_count, best_f))
            
        return {"best_x": best_x, "best_f": best_f, "evals": eval_count, "history": history}


class DifferentialEvolutionOptimizer(BaseOptimizer):
    """Differential Evolution (DE/rand/1/bin)"""
    def __init__(self, dim, bounds, pop_size=50, F=0.8, CR=0.9, seed=None):
        super().__init__(dim, bounds, pop_size, seed)
        self.F = F
        self.CR = CR

    def optimize(self, objective_fn, max_evals=10000):
        pop = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        t_pop = torch.tensor(pop, dtype=torch.float32)
        if hasattr(objective_fn, 'device'):
            t_pop = t_pop.to(objective_fn.device)
        fitness = objective_fn(t_pop).detach().cpu().numpy()
        
        eval_count = self.pop_size
        best_idx = np.argmin(fitness)
        best_f = float(fitness[best_idx])
        best_x = np.copy(pop[best_idx])
        history = [(eval_count, best_f)]
        
        while eval_count < max_evals:
            trials = np.empty_like(pop)
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = pop[a] + self.F * (pop[b] - pop[c])
                mutant = np.clip(mutant, self.lb, self.ub)
                
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                trials[i] = trial
                
            t_trials = torch.tensor(trials, dtype=torch.float32)
            if hasattr(objective_fn, 'device'):
                t_trials = t_trials.to(objective_fn.device)
            f_trials = objective_fn(t_trials).detach().cpu().numpy()
            eval_count += self.pop_size
            
            # Selection
            better = f_trials < fitness
            pop[better] = trials[better]
            fitness[better] = f_trials[better]
            
            curr_best_idx = np.argmin(fitness)
            if fitness[curr_best_idx] < best_f:
                best_f = float(fitness[curr_best_idx])
                best_x = np.copy(pop[curr_best_idx])
            history.append((eval_count, best_f))
            
        return {"best_x": best_x, "best_f": best_f, "evals": eval_count, "history": history}


class PSOOptimizer(BaseOptimizer):
    """Particle Swarm Optimization (PSO)"""
    def __init__(self, dim, bounds, pop_size=50, w=0.729, c1=1.49445, c2=1.49445, seed=None):
        super().__init__(dim, bounds, pop_size, seed)
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def optimize(self, objective_fn, max_evals=10000):
        pos = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        v_max = 0.2 * (self.ub - self.lb)
        vel = np.random.uniform(-v_max, v_max, (self.pop_size, self.dim))
        
        t_pos = torch.tensor(pos, dtype=torch.float32)
        if hasattr(objective_fn, 'device'):
            t_pos = t_pos.to(objective_fn.device)
        fitness = objective_fn(t_pos).detach().cpu().numpy()
        
        eval_count = self.pop_size
        pbest_pos = np.copy(pos)
        pbest_f = np.copy(fitness)
        
        gbest_idx = np.argmin(pbest_f)
        best_f = float(pbest_f[gbest_idx])
        best_x = np.copy(pbest_pos[gbest_idx])
        history = [(eval_count, best_f)]
        
        while eval_count < max_evals:
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            
            vel = (self.w * vel + 
                   self.c1 * r1 * (pbest_pos - pos) + 
                   self.c2 * r2 * (best_x - pos))
            vel = np.clip(vel, -v_max, v_max)
            pos = np.clip(pos + vel, self.lb, self.ub)
            
            t_pos = torch.tensor(pos, dtype=torch.float32)
            if hasattr(objective_fn, 'device'):
                t_pos = t_pos.to(objective_fn.device)
            fitness = objective_fn(t_pos).detach().cpu().numpy()
            eval_count += self.pop_size
            
            # Update pbest
            improved = fitness < pbest_f
            pbest_f[improved] = fitness[improved]
            pbest_pos[improved] = pos[improved]
            
            curr_best_idx = np.argmin(pbest_f)
            if pbest_f[curr_best_idx] < best_f:
                best_f = float(pbest_f[curr_best_idx])
                best_x = np.copy(pbest_pos[curr_best_idx])
            history.append((eval_count, best_f))
            
        return {"best_x": best_x, "best_f": best_f, "evals": eval_count, "history": history}


class CBOOptimizer(BaseOptimizer):
    """Consensus-Based Optimization (CBO) (Carrillo et al.)"""
    def __init__(self, dim, bounds, pop_size=50, alpha=5.0, sigma=0.4, dt=0.1, seed=None):
        super().__init__(dim, bounds, pop_size, seed)
        self.alpha = alpha
        self.sigma = sigma
        self.dt = dt

    def optimize(self, objective_fn, max_evals=10000):
        particles = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        
        t_p = torch.tensor(particles, dtype=torch.float32)
        if hasattr(objective_fn, 'device'):
            t_p = t_p.to(objective_fn.device)
        fitness = objective_fn(t_p).detach().cpu().numpy()
        
        eval_count = self.pop_size
        best_idx = np.argmin(fitness)
        best_f = float(fitness[best_idx])
        best_x = np.copy(particles[best_idx])
        history = [(eval_count, best_f)]
        
        while eval_count < max_evals:
            # Gibbs consensus point
            f_shift = fitness - np.min(fitness)
            weights = np.exp(-self.alpha * f_shift / (np.std(fitness) + 1e-8))
            weights = weights / (np.sum(weights) + 1e-12)
            v_alpha = np.sum(weights[:, None] * particles, axis=0)
            
            # SDE update: dx = -lambda * (x - v_alpha) dt + sigma * |x - v_alpha| dW
            diff = particles - v_alpha
            drift = -diff * self.dt
            diffusion = self.sigma * np.abs(diff) * np.sqrt(self.dt) * np.random.randn(self.pop_size, self.dim)
            
            particles = np.clip(particles + drift + diffusion, self.lb, self.ub)
            
            t_p = torch.tensor(particles, dtype=torch.float32)
            if hasattr(objective_fn, 'device'):
                t_p = t_p.to(objective_fn.device)
            fitness = objective_fn(t_p).detach().cpu().numpy()
            eval_count += self.pop_size
            
            curr_best_idx = np.argmin(fitness)
            if fitness[curr_best_idx] < best_f:
                best_f = float(fitness[curr_best_idx])
                best_x = np.copy(particles[curr_best_idx])
            history.append((eval_count, best_f))
            
        return {"best_x": best_x, "best_f": best_f, "evals": eval_count, "history": history}


class CEMOptimizer(BaseOptimizer):
    """Cross-Entropy Method (CEM)"""
    def __init__(self, dim, bounds, pop_size=50, elite_ratio=0.2, seed=None):
        super().__init__(dim, bounds, pop_size, seed)
        self.n_elite = max(2, int(pop_size * elite_ratio))

    def optimize(self, objective_fn, max_evals=10000):
        mean = np.random.uniform(self.lb, self.ub, self.dim)
        std = 0.5 * (self.ub - self.lb) * np.ones(self.dim)
        
        eval_count = 0
        best_f = float("inf")
        best_x = None
        history = []
        
        while eval_count < max_evals:
            samples = np.random.normal(mean, std, (self.pop_size, self.dim))
            samples = np.clip(samples, self.lb, self.ub)
            
            t_s = torch.tensor(samples, dtype=torch.float32)
            if hasattr(objective_fn, 'device'):
                t_s = t_s.to(objective_fn.device)
            fitness = objective_fn(t_s).detach().cpu().numpy()
            eval_count += self.pop_size
            
            elite_indices = np.argsort(fitness)[:self.n_elite]
            elites = samples[elite_indices]
            
            mean = np.mean(elites, axis=0)
            std = np.std(elites, axis=0) + 1e-5
            
            curr_best = fitness[elite_indices[0]]
            if curr_best < best_f:
                best_f = float(curr_best)
                best_x = np.copy(elites[0])
            history.append((eval_count, best_f))
            
        return {"best_x": best_x, "best_f": best_f, "evals": eval_count, "history": history}
