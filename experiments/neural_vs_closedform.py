import time
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import numpy as np
from flowopt.optimizer import FlowOpt
from flowopt.benchmarks import Sphere, Rastrigin, Ackley

def run_neural_vs_closedform_experiment(dim=10, max_evals=2000, n_runs=3):
    print("=" * 70)
    print(f"EXPERIMENT: Neural Vector Field vs. Closed-Form OT Vector Field (D={dim})")
    print("=" * 70)
    
    benchmarks = [
        ("Sphere", Sphere(dim=dim)),
        ("Rastrigin", Rastrigin(dim=dim)),
        ("Ackley", Ackley(dim=dim))
    ]
    
    results = {}
    
    for name, fn in benchmarks:
        results[name] = {"closed_form": {"fitness": [], "time": []}, "neural": {"fitness": [], "time": []}}
        
        for run in range(n_runs):
            seed = 42 + run
            
            # 1. Closed-Form OT-FlowOpt
            t0 = time.time()
            opt_cf = FlowOpt(dim=dim, bounds=fn.bounds, mode="ot_closed_form", seed=seed)
            res_cf = opt_cf.optimize(fn, max_evals=max_evals)
            t_cf = time.time() - t0
            results[name]["closed_form"]["fitness"].append(res_cf["best_f"])
            results[name]["closed_form"]["time"].append(t_cf)
            
            # 2. Neural FlowOpt (Online MLP training)
            t0 = time.time()
            opt_nn = FlowOpt(dim=dim, bounds=fn.bounds, mode="neural", seed=seed)
            res_nn = opt_nn.optimize(fn, max_evals=max_evals)
            t_nn = time.time() - t0
            results[name]["neural"]["fitness"].append(res_nn["best_f"])
            results[name]["neural"]["time"].append(t_nn)
            
        cf_f_mean = np.mean(results[name]["closed_form"]["fitness"])
        cf_t_mean = np.mean(results[name]["closed_form"]["time"])
        nn_f_mean = np.mean(results[name]["neural"]["fitness"])
        nn_t_mean = np.mean(results[name]["neural"]["time"])
        
        speedup = nn_t_mean / max(cf_t_mean, 1e-6)
        
        print(f"\nBenchmark: {name}")
        print(f"  Closed-Form OT-FM : Best f = {cf_f_mean:.4e} | Wall-clock Time = {cf_t_mean:.2f}s")
        print(f"  Neural MLP-FM     : Best f = {nn_f_mean:.4e} | Wall-clock Time = {nn_t_mean:.2f}s")
        print(f"  Speedup Factor    : {speedup:.1f}x FASTER with Closed-Form!")
        
    return results

if __name__ == "__main__":
    run_neural_vs_closedform_experiment(dim=10, max_evals=2000, n_runs=3)
