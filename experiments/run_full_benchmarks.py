import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import time
import torch
import numpy as np
from scipy import stats

from flowopt.optimizer import FlowOpt
from flowopt.baselines import (
    CMAESOptimizer,
    DifferentialEvolutionOptimizer,
    PSOOptimizer,
    CBOOptimizer,
    CEMOptimizer
)
from flowopt.benchmarks import (
    Sphere,
    Rosenbrock,
    Rastrigin,
    Ackley,
    Griewank,
    Schwefel,
    Levy,
    LennardJonesCluster
)

def run_benchmarks(dim=30, max_evals=10000, n_runs=5, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    
    benchmarks = [
        Sphere(dim=dim),
        Rosenbrock(dim=dim),
        Rastrigin(dim=dim),
        Ackley(dim=dim),
        Griewank(dim=dim),
        Schwefel(dim=dim),
        Levy(dim=dim),
        LennardJonesCluster(n_atoms=dim // 3 if dim >= 6 else 4)
    ]
    
    optimizers = {
        "FlowOpt (Ours)": lambda d, b, s: FlowOpt(dim=d, bounds=b, seed=s),
        "CMA-ES": lambda d, b, s: CMAESOptimizer(dim=d, bounds=b, seed=s),
        "DE": lambda d, b, s: DifferentialEvolutionOptimizer(dim=d, bounds=b, seed=s),
        "PSO": lambda d, b, s: PSOOptimizer(dim=d, bounds=b, seed=s),
        "CBO": lambda d, b, s: CBOOptimizer(dim=d, bounds=b, seed=s),
        "CEM": lambda d, b, s: CEMOptimizer(dim=d, bounds=b, seed=s)
    }
    
    summary = {}
    
    print("=" * 85)
    print(f"STARTING COMPREHENSIVE BENCHMARK EVALUATION (Dim={dim}, Evals={max_evals}, Runs={n_runs})")
    print("=" * 85)
    
    for fn in benchmarks:
        fn_name = fn.name
        summary[fn_name] = {}
        print(f"\n>>> Running Benchmark: {fn_name} (Bounds: {fn.bounds}, Optimum: {fn.optimal_value})")
        
        for opt_name, opt_factory in optimizers.items():
            fitness_list = []
            runtime_list = []
            histories = []
            
            for run_idx in range(n_runs):
                seed = 1000 * (run_idx + 1) + 42
                t0 = time.time()
                try:
                    opt = opt_factory(fn.dim, fn.bounds, seed)
                    res = opt.optimize(fn, max_evals=max_evals)
                    elapsed = time.time() - t0
                    fitness_list.append(res["best_f"])
                    runtime_list.append(elapsed)
                    # subsample history to 100 points
                    hist = res.get("history", [])
                    if len(hist) > 100:
                        step = len(hist) // 100
                        hist = hist[::step]
                    histories.append(hist)
                except Exception as e:
                    print(f"Error in {opt_name} on {fn_name}: {e}")
                    fitness_list.append(float("nan"))
                    runtime_list.append(0.0)
                    histories.append([])
                    
            fit_arr = np.array(fitness_list)
            mean_f = np.nanmean(fit_arr)
            std_f = np.nanstd(fit_arr)
            mean_t = np.mean(runtime_list)
            
            summary[fn_name][opt_name] = {
                "mean_fitness": float(mean_f),
                "std_fitness": float(std_f),
                "mean_time": float(mean_t),
                "all_fitness": [float(x) for x in fit_arr],
                "histories": histories
            }
            print(f"  {opt_name:<16} | Mean: {mean_f:12.4e} ± {std_f:10.4e} | Time: {mean_t:5.2f}s")
            
        # Statistical significance test (Wilcoxon rank-sum or Mann-Whitney U test vs FlowOpt)
        flow_scores = summary[fn_name]["FlowOpt (Ours)"]["all_fitness"]
        print(f"  --- Significance Tests vs FlowOpt (Ours) ---")
        for opt_name in optimizers:
            if opt_name == "FlowOpt (Ours)":
                continue
            other_scores = summary[fn_name][opt_name]["all_fitness"]
            try:
                stat, pval = stats.mannwhitneyu(flow_scores, other_scores, alternative="two-sided")
                signif = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "n.s."))
                print(f"    vs {opt_name:<12}: p = {pval:.4f} ({signif})")
                summary[fn_name][opt_name]["p_value_vs_flowopt"] = float(pval)
            except Exception as e:
                summary[fn_name][opt_name]["p_value_vs_flowopt"] = 1.0
                
    # Save full results to JSON
    json_path = os.path.join(output_dir, f"benchmark_results_d{dim}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll benchmark results successfully exported to {json_path}")
    return summary

if __name__ == "__main__":
    # Test on D=10 and D=30
    run_benchmarks(dim=10, max_evals=6000, n_runs=5, output_dir="results")
