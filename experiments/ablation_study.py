import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import torch
import numpy as np

from flowopt.optimizer import FlowOpt
from flowopt.benchmarks import Sphere, Rosenbrock, Rastrigin, Ackley

def run_ablation_study(dim=10, max_evals=5000, n_runs=5, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    
    benchmarks = [
        Sphere(dim=dim),
        Rosenbrock(dim=dim),
        Rastrigin(dim=dim),
        Ackley(dim=dim)
    ]
    
    # Define Ablation Variants
    variants = {
        "FlowOpt (Full Model)": {
            "mode": "ot_closed_form", "momentum": 0.5, "repulsive_weight": 0.015, "reg_ot": 0.03
        },
        "w/o Optimal Transport (Greedy)": {
            "mode": "greedy", "momentum": 0.5, "repulsive_weight": 0.015, "reg_ot": 0.03
        },
        "w/o Kinetic Momentum (gamma=0)": {
            "mode": "ot_closed_form", "momentum": 0.0, "repulsive_weight": 0.015, "reg_ot": 0.03
        },
        "w/o Repulsive Dispersion (alpha=0)": {
            "mode": "ot_closed_form", "momentum": 0.5, "repulsive_weight": 0.0, "reg_ot": 0.03
        },
        "Euler ODE (1-step)": {
            "mode": "ot_closed_form", "momentum": 0.5, "repulsive_weight": 0.015, "reg_ot": 0.03
        }
    }
    
    results = {}
    print("=" * 80)
    print(f"ABLATION STUDY: Systematic Component Contribution Analysis (Dim={dim})")
    print("=" * 80)
    
    for fn in benchmarks:
        fn_name = fn.name
        results[fn_name] = {}
        print(f"\n--- Benchmark: {fn_name} ---")
        
        for v_name, v_params in variants.items():
            fitness_list = []
            for run_idx in range(n_runs):
                seed = 2000 * (run_idx + 1) + 42
                opt = FlowOpt(
                    dim=dim,
                    pop_size=25,
                    n_candidates=50,
                    mode=v_params["mode"],
                    momentum=v_params["momentum"],
                    repulsive_weight=v_params["repulsive_weight"],
                    reg_ot=v_params["reg_ot"],
                    bounds=fn.bounds,
                    seed=seed
                )
                res = opt.optimize(fn, max_evals=max_evals)
                fitness_list.append(res["best_f"])
                
            mean_f = float(np.mean(fitness_list))
            std_f = float(np.std(fitness_list))
            results[fn_name][v_name] = {"mean": mean_f, "std": std_f, "all": fitness_list}
            print(f"  {v_name:<36} | Mean f(x): {mean_f:12.4e} ± {std_f:10.4e}")
            
    # Export ablation results
    out_file = os.path.join(output_dir, "ablation_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAblation study results exported to {out_file}")
    return results

if __name__ == "__main__":
    run_ablation_study(dim=10, max_evals=5000, n_runs=5)
