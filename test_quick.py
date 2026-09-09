import torch
from flowopt.optimizer import FlowOpt
from flowopt.baselines import CMAESOptimizer, DifferentialEvolutionOptimizer, PSOOptimizer, CBOOptimizer, CEMOptimizer
from flowopt.benchmarks import Rastrigin, Sphere, Ackley

def test_optimizers():
    dim = 10
    max_evals = 2000
    
    benchmarks = [
        ("Sphere", Sphere(dim=dim)),
        ("Rastrigin", Rastrigin(dim=dim)),
        ("Ackley", Ackley(dim=dim))
    ]
    
    for name, fn in benchmarks:
        print(f"\n================ Testing on {name} (D={dim}) ================")
        
        # 1. FlowOpt (OT Closed-Form)
        opt_flow = FlowOpt(dim=dim, pop_size=40, n_candidates=80, bounds=fn.bounds, mode="ot_closed_form", seed=42)
        res_flow = opt_flow.optimize(fn, max_evals=max_evals)
        print(f"FlowOpt (OT Closed-Form) : Best f = {res_flow['best_f']:.6e} (evals: {res_flow['evals']})")
        
        # 2. CMA-ES
        opt_cma = CMAESOptimizer(dim=dim, bounds=fn.bounds, pop_size=40, seed=42)
        res_cma = opt_cma.optimize(fn, max_evals=max_evals)
        print(f"CMA-ES                   : Best f = {res_cma['best_f']:.6e} (evals: {res_cma['evals']})")
        
        # 3. DE
        opt_de = DifferentialEvolutionOptimizer(dim=dim, bounds=fn.bounds, pop_size=40, seed=42)
        res_de = opt_de.optimize(fn, max_evals=max_evals)
        print(f"Differential Evolution   : Best f = {res_de['best_f']:.6e} (evals: {res_de['evals']})")
        
        # 4. PSO
        opt_pso = PSOOptimizer(dim=dim, bounds=fn.bounds, pop_size=40, seed=42)
        res_pso = opt_pso.optimize(fn, max_evals=max_evals)
        print(f"PSO                      : Best f = {res_pso['best_f']:.6e} (evals: {res_pso['evals']})")
        
        # 5. CBO
        opt_cbo = CBOOptimizer(dim=dim, bounds=fn.bounds, pop_size=40, seed=42)
        res_cbo = opt_cbo.optimize(fn, max_evals=max_evals)
        print(f"CBO                      : Best f = {res_cbo['best_f']:.6e} (evals: {res_cbo['evals']})")

if __name__ == "__main__":
    test_optimizers()
