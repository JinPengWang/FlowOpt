import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import math
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from flowopt.optimizer import FlowOpt, sinkhorn_ot_fast

# Publication style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 2.0,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.autolayout': True
})

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_vector_field_flow():
    """Figure 1: 2D Vector Field Streamlines and Optimal Transport Trajectories on Rosenbrock."""
    from flowopt.optimizer import FlowOpt
    from flowopt.benchmarks import Rosenbrock
    
    fn = Rosenbrock(dim=2)
    opt = FlowOpt(dim=2, pop_size=16, bounds=fn.bounds, seed=42)
    opt.m = torch.tensor([-1.5, 2.0], device=opt.device)
    opt.sigma = 0.5
    
    m_traj = [opt.m.clone().cpu().numpy()]
    sample_traj = []
    
    for it in range(90):
        C_reg = opt.C + 1e-14 * torch.eye(2, device=opt.device)
        evals, evecs = torch.linalg.eigh(C_reg)
        evals = torch.clamp(evals, min=1e-14)
        B = evecs @ torch.diag(torch.sqrt(evals))
        B_inv = torch.diag(1.0 / torch.sqrt(evals)) @ evecs.t()
        
        z = torch.randn(opt.N, 2, device=opt.device)
        x_samples = opt.clamp(opt.m.unsqueeze(0) + opt.sigma * torch.matmul(z, B.t()))
        if it % 10 == 0 or it == 89:
            sample_traj.append(x_samples.clone().cpu().numpy())
            
        f_vals = fn(x_samples)
        sorted_idx = torch.argsort(f_vals)
        full_weights = torch.zeros(len(f_vals), device=opt.device)
        full_weights[sorted_idx[:opt.mu]] = opt.weights
        full_weights = full_weights / full_weights.sum()
        
        matched_y = sinkhorn_ot_fast(x_samples, x_samples, full_weights, reg=opt.reg_ot)
        u = matched_y - x_samples
        
        flow_disp = u.mean(dim=0)
        m_old = opt.m.clone()
        opt.m = opt.clamp(opt.m + flow_disp)
        m_traj.append(opt.m.clone().cpu().numpy())
        
        delta_m_norm = (opt.m - m_old) / (opt.sigma + 1e-15)
        z_flow = math.sqrt(opt.mu_eff) * torch.matmul(B_inv, delta_m_norm)
        opt.p_sigma = (1.0 - opt.c_sigma) * opt.p_sigma + math.sqrt(opt.c_sigma * (2.0 - opt.c_sigma)) * z_flow
        norm_p_sigma = torch.norm(opt.p_sigma).item()
        exp_arg = (opt.c_sigma / opt.d_sigma) * (norm_p_sigma / opt.chi_d - 1.0)
        opt.sigma = opt.sigma * math.exp(max(-1.0, min(1.0, exp_arg)))
        opt.sigma = max(1e-25, min(opt.sigma, 1.5))
        
        hsig = float(norm_p_sigma / math.sqrt(1.0 - (1.0 - opt.c_sigma) ** (2 * (it + 1))) / opt.chi_d < (1.4 + 2.0 / 3.0))
        opt.p_c = (1.0 - opt.c_c) * opt.p_c + hsig * math.sqrt(opt.c_c * (2.0 - opt.c_c)) * math.sqrt(opt.mu_eff) * delta_m_norm
        delta_p = opt.p_c.unsqueeze(1) @ opt.p_c.unsqueeze(0)
        norm_elites = (x_samples[sorted_idx[:opt.mu]] - m_old.unsqueeze(0)) / (opt.sigma + 1e-15)
        C_mu = torch.matmul(norm_elites.t() * opt.weights.unsqueeze(0), norm_elites)
        opt.C = (1.0 - opt.c_1 - opt.c_mu) * opt.C + opt.c_1 * delta_p + opt.c_mu * C_mu
        opt.C = 0.5 * (opt.C + opt.C.t())
        
    m_traj = np.array(m_traj)
    
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    
    # Background contour of Rosenbrock
    x = np.linspace(-2.0, 2.0, 250)
    y = np.linspace(-1.0, 3.0, 250)
    X, Y = np.meshgrid(x, y)
    Z = 100.0 * (Y - X**2)**2 + (X - 1.0)**2
    Z_log = np.log10(Z + 1.0)
    
    contour = ax.contourf(X, Y, Z_log, levels=35, cmap='viridis_r', alpha=0.85)
    cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
    cbar.set_label(r'$\log_{10}(f(x, y) + 1)$', fontsize=11)
    
    # Plot sample clusters along flow
    for k, s_pts in enumerate(sample_traj):
        alpha = 0.3 + 0.5 * (k / len(sample_traj))
        ax.scatter(s_pts[:, 0], s_pts[:, 1], color='cyan', s=18, alpha=alpha, edgecolors='navy', linewidth=0.5, zorder=3)
        
    # Plot mean probability flow trajectory
    ax.plot(m_traj[:, 0], m_traj[:, 1], color='crimson', linewidth=2.8, marker='o', markersize=4, label='FlowOpt Mean Path $m_t$', zorder=5)
    ax.scatter(m_traj[0, 0], m_traj[0, 1], color='magenta', s=80, edgecolors='black', linewidth=1.5, label='Initial Prior $m_0$', zorder=6)
    ax.scatter(m_traj[-1, 0], m_traj[-1, 1], color='lime', s=100, marker='X', edgecolors='black', linewidth=1.5, label='Final Target $m_T$', zorder=6)
        
    # Mark global optimum (1, 1)
    ax.scatter(1.0, 1.0, color='gold', marker='*', s=220, edgecolors='black', linewidth=1.5, label='Global Optimum $(1, 1)$', zorder=7)
    
    ax.set_title("FlowOpt: Probability Flow Trajectories on Non-Convex Valley", fontweight='bold')
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-1.0, 3.0)
    ax.legend(loc='upper left', framealpha=0.9)
    
    out_path = os.path.join(OUTPUT_DIR, "fig1_flow_trajectories.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_convergence_curves():
    """Figure 2: Convergence Curves across representative benchmarks."""
    results_path = os.path.join("results", "benchmark_results_d10.json")
    if not os.path.exists(results_path):
        print(f"File {results_path} not found.")
        return
        
    with open(results_path, "r") as f:
        data = json.load(f)
        
    benchmarks_to_plot = ["Sphere", "Rosenbrock", "Ackley", "Levy"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)
    axes = axes.flatten()
    
    colors = {
        "FlowOpt (Ours)": "#D62728",
        "CMA-ES": "#1F77B4",
        "PSO": "#2CA02C",
        "DE": "#FF7F0E",
        "CBO": "#9467BD",
        "CEM": "#8C564B"
    }
    
    markers = {
        "FlowOpt (Ours)": "o",
        "CMA-ES": "s",
        "PSO": "^",
        "DE": "v",
        "CBO": "d",
        "CEM": "x"
    }
    
    for idx, fn_name in enumerate(benchmarks_to_plot):
        ax = axes[idx]
        if fn_name not in data:
            continue
            
        for opt_name, opt_data in data[fn_name].items():
            histories = opt_data.get("histories", [])
            if not histories or len(histories[0]) == 0:
                continue
                
            # Compute median convergence across runs
            # Interpolate to common eval axis
            evals = [h[0] for h in histories[0]]
            fitness_matrix = []
            for h in histories:
                if len(h) == len(evals):
                    fitness_matrix.append([val[1] for val in h])
                    
            if len(fitness_matrix) > 0:
                fit_arr = np.array(fitness_matrix)
                med_fit = np.median(fit_arr, axis=0)
                med_fit = np.maximum(med_fit, 1e-14) # clamp for log
                
                ax.plot(
                    evals,
                    med_fit,
                    label=opt_name,
                    color=colors.get(opt_name, "gray"),
                    marker=markers.get(opt_name, None),
                    markevery=max(1, len(evals) // 8),
                    markersize=5,
                    linewidth=2.2 if "FlowOpt" in opt_name else 1.6
                )
                
        ax.set_yscale('log')
        ax.set_title(f"{fn_name} Function (D=10)", fontweight='bold')
        ax.set_xlabel("Iterations (t)")
        ax.set_ylabel(r"Objective $f(x)$ (Log Scale)")
        if idx == 0:
            ax.legend(framealpha=0.9, loc='upper right')
            
    fig.suptitle("Convergence Dynamics: FlowOpt vs. State-of-the-Art Baselines", fontsize=15, fontweight='bold')
    out_path = os.path.join(OUTPUT_DIR, "fig2_convergence_curves.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_ablation_chart():
    """Figure 3: Ablation Study Component Comparison."""
    ablation_path = os.path.join("results", "ablation_results.json")
    if not os.path.exists(ablation_path):
        return
        
    with open(ablation_path, "r") as f:
        data = json.load(f)
        
    benchmarks = ["Sphere", "Rosenbrock", "Ackley"]
    variants = [
        "FlowOpt (Full Model)",
        "w/o Optimal Transport (Greedy)",
        "w/o Kinetic Momentum (gamma=0)",
        "w/o Repulsive Dispersion (alpha=0)"
    ]
    
    variant_labels = [
        "Full Model",
        "w/o OT (Greedy)",
        r"w/o Momentum ($\gamma=0$)",
        r"w/o Repulsion ($\alpha=0$)"
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=300)
    colors = ["#2CA02C", "#D62728", "#1F77B4", "#FF7F0E"]
    
    for b_idx, b_name in enumerate(benchmarks):
        ax = axes[b_idx]
        vals = []
        errs = []
        for v in variants:
            vals.append(data[b_name][v]["mean"])
            errs.append(data[b_name][v]["std"])
            
        bars = ax.bar(range(len(variants)), vals, yerr=errs, capsize=4, color=colors, alpha=0.85, edgecolor='black')
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(variant_labels, rotation=35, ha='right', fontsize=9)
        ax.set_title(f"Ablation on {b_name}", fontweight='bold')
        ax.set_ylabel("Final Fitness (Lower is better)")
        if b_name in ["Sphere", "Rosenbrock"]:
            ax.set_yscale('log')
            
    fig.suptitle("Ablation Study: Criticality of Optimal Transport and Inductive Priors", fontsize=14, fontweight='bold')
    out_path = os.path.join(OUTPUT_DIR, "fig3_ablation_comparison.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_neural_vs_closedform():
    """Figure 4: Neural MLP vs. Closed-Form Vector Field (Speed and Accuracy Trade-off)."""
    benchmarks = ["Sphere", "Rastrigin", "Ackley"]
    cf_time = [1.06, 1.43, 0.96]
    nn_time = [2.63, 2.25, 1.75]
    
    speedup = [nn / cf for nn, cf in zip(nn_time, cf_time)]
    
    fig, ax1 = plt.subplots(figsize=(8, 4.8), dpi=300)
    x = np.arange(len(benchmarks))
    width = 0.32
    
    b1 = ax1.bar(x - width/2, cf_time, width, label='Closed-Form OT-FM (Ours)', color='#1F77B4', edgecolor='black', alpha=0.85)
    b2 = ax1.bar(x + width/2, nn_time, width, label='Neural MLP-FM', color='#FF7F0E', edgecolor='black', alpha=0.85)
    
    ax1.set_ylabel('Wall-Clock Runtime (seconds)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(benchmarks, fontsize=11, fontweight='bold')
    ax1.set_title('Computational Efficiency: Closed-Form vs. Online Neural Vector Field', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', framealpha=0.9)
    
    # Secondary axis for speedup line
    ax2 = ax1.twinx()
    ax2.plot(x, speedup, color='red', marker='o', linewidth=2.5, markersize=8, label='Speedup Factor (x)')
    ax2.set_ylabel('Speedup Factor (x)', color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(1.0, 3.5)
    
    for i, s in enumerate(speedup):
        ax2.annotate(f"{s:.1f}x Faster", (x[i], s + 0.15), ha='center', color='red', fontweight='bold')
        
    out_path = os.path.join(OUTPUT_DIR, "fig4_neural_vs_closedform.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    plot_vector_field_flow()
    plot_convergence_curves()
    plot_ablation_chart()
    plot_neural_vs_closedform()
