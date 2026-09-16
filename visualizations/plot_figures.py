"""
Publication figures for FlowOpt v2.

Generates four PNG+PDF panels under ``figures/``:
  1. Probability-flow geodesics on Rosenbrock  (single trajectory of m)
  2. Convergence curves vs. baselines           (Sphere, Rosenbrock, Ackley, Levy)
  3. CSA null-hypothesis behaviour              (path norm vs chi_D under random fitness)
  4. Sinkhorn coupling visualisation           (a 2-D example with 6 particles)

No third-party plotting libraries beyond matplotlib.  All figures use
Nature/IEEE-style typography and vector PDF export.
"""

from __future__ import annotations

import math
import os
import sys
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flowopt import FlowOpt, sinkhorn_coupling  # noqa: E402
from flowopt.benchmarks import (                 # noqa: E402
    Sphere, Rosenbrock, Ackley, Levy, LennardJonesCluster,
)

mpl.rcParams.update({
    "font.size": 9.5,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "axes.labelsize": 10.5,
    "axes.titlesize": 11,
    "axes.linewidth": 0.6,
    "axes.grid": True,
    "grid.alpha": 0.30,
    "grid.linestyle": "--",
    "lines.linewidth": 1.7,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 110,
})

OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)
PANEL_LABELS = list("abcd")


# --------------------------------------------------------------------------- #
#  Figure 1 — probability-flow geodesics on Rosenbrock                        #
# --------------------------------------------------------------------------- #

def fig1_flow_trajectory():
    fn = Rosenbrock(dim=2)
    opt = FlowOpt(dim=2, pop_size=20, bounds=fn.bounds, seed=42)
    # Manually start from a far-away point to show the trajectory
    opt.m = torch.tensor([-1.6, 2.2], device=opt.device)

    traj = [opt.m.clone().cpu().numpy()]
    for it in range(85):
        reg_t = 0.5 * (1.0 - it / 84.0) + 1e-3
        opt.step(fn, reg_ot=reg_t)
        traj.append(opt.m.clone().cpu().numpy())
    traj = np.array(traj)

    gx = np.linspace(-2.0, 2.0, 200)
    gy = np.linspace(-1.0, 3.0, 200)
    GX, GY = np.meshgrid(gx, gy)
    GZ = 100.0 * (GY - GX ** 2) ** 2 + (GX - 1.0) ** 2

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    cf = ax.contourf(GX, GY, np.log10(GZ + 1.0), levels=24, cmap="viridis_r", alpha=0.85)
    fig.colorbar(cf, ax=ax, fraction=0.045, pad=0.025,
                 label=r"$\log_{10}\bigl(f(x_1,x_2)+1\bigr)$")
    ax.plot(traj[:, 0], traj[:, 1], color="white", lw=1.4, alpha=0.7)
    ax.plot(traj[:, 0], traj[:, 1], color="#D62728", lw=2.2, label="Flow mean path")
    ax.scatter(traj[0, 0], traj[0, 1], s=85, color="#FFA500",
               edgecolor="black", lw=0.7, zorder=5, label=r"start $\mathbf{m}_0$")
    ax.scatter(traj[-1, 0], traj[-1, 1], s=110, color="#2CA02C", marker="P",
               edgecolor="black", lw=0.7, zorder=5, label=r"end $\mathbf{m}_T$")
    ax.scatter([1.0], [1.0], s=160, marker="*", color="gold",
               edgecolor="black", lw=1.0, zorder=6, label=r"ground state $(1,1)$")
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-1.0, 3.0)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("Probability-flow geodesics on the Rosenbrock valley")
    ax.legend(loc="upper left", framealpha=0.92, fontsize=8.5)
    ax.text(0.02, 0.96, "a", transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="top")

    for ext in ("png", "pdf"):
        p = os.path.join(OUT, f"fig1_flow_trajectory.{ext}")
        fig.savefig(p, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"  saved {p}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Figure 2 — convergence comparison (if benchmark json exists)               #
# --------------------------------------------------------------------------- #

def fig2_convergence_curves(json_path=None):
    if json_path is None:
        json_path = os.path.join(ROOT, "results", "benchmark_results_d10.json")
    if not os.path.exists(json_path):
        print("  [fig2] skipped (no benchmark json yet)")
        return
    with open(json_path) as f:
        data = json.load(f)

    benches = ["Sphere", "Rosenbrock", "Ackley", "Levy"]
    palette = {"FlowOpt (Ours)": "#D62728", "CMA-ES": "#1F77B4", "PSO": "#2CA02C",
               "DE": "#FF7F0E", "CBO": "#9467BD", "CEM": "#7F7F7F"}
    floor = {"Sphere": 1e-15, "Levy": 1e-15, "Ackley": 1e-6, "Rosenbrock": 1e-3}

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6))
    axes = axes.flatten()
    for i, name in enumerate(benches):
        ax = axes[i]
        if name not in data:
            ax.set_visible(False)
            continue
        for opt_name, opt_data in data[name].items():
            histories = opt_data.get("histories", [])
            if not histories:
                continue
            evals = [h[0] for h in histories[0]]
            mat = np.array([[v for _, v in h] for h in histories
                            if len(h) == len(evals)], dtype=float)
            if mat.size == 0:
                continue
            med = np.maximum(np.median(mat, axis=0), floor[name])
            ax.plot(evals, med, color=palette.get(opt_name, "#7F7F7F"),
                    lw=2.6 if "FlowOpt" in opt_name else 1.4,
                    label=opt_name)
        ax.set_yscale("log")
        ax.set_title(name)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("best  f(x)")
        ax.text(0.02, 0.96, PANEL_LABELS[i], transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top")
        if i == 0:
            ax.legend(loc="upper right", fontsize=7.5)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(OUT, f"fig2_convergence_curves.{ext}")
        fig.savefig(p, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"  saved {p}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Figure 3 — CSA null-hypothesis check                                         #
# --------------------------------------------------------------------------- #

def fig3_csa_null_hypothesis():
    D, N = 10, 30
    torch.manual_seed(0)
    opt = FlowOpt(dim=D, pop_size=N, seed=0)

    norms = []
    burn = 100
    for gen in range(800):
        # random fitness, no signal
        x = opt.m + opt.sigma * torch.randn(N, D, device=opt.device)
        rf = torch.randn(N).tolist()
        order = np.argsort(rf)
        elites = x[order[: opt.mu]]
        ar = (elites - opt.m) / opt.sigma
        z_w = math.sqrt(opt.mu_eff) * (opt.w.view(-1, 1) * ar).sum(dim=0)
        opt.p_sigma = (1.0 - opt.c_sigma) * opt.p_sigma + z_w
        if gen >= burn:
            norms.append(float(opt.p_sigma.norm().item()))

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.hist(norms, bins=30, color="#1F77B4", alpha=0.85, density=True,
            label=r"empirical $\|p_\sigma\|$")
    ax.axvline(opt.chi_D, color="#D62728", lw=2.0,
               label=fr"$\chi_D = {opt.chi_D:.3f}$")
    ax.set_xlabel(r"$\|p_\sigma\|$ under random fitness")
    ax.set_ylabel("density")
    ax.set_title("CSA null-hypothesis: step-size path keeps chi_D")
    ax.legend(loc="upper right")
    ax.text(0.02, 0.96, "a", transform=ax.transAxes, fontsize=12, fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(OUT, f"fig3_csa_null.{ext}")
        fig.savefig(p, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"  saved {p}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Figure 4 — Sinkhorn coupling visualisation                                  #
# --------------------------------------------------------------------------- #

def fig4_sinkhorn_coupling():
    torch.manual_seed(0)
    source = torch.tensor([[0.2, 0.3], [0.4, 0.7], [0.9, 0.2],
                           [1.1, 0.6], [1.5, 0.3], [1.7, 1.0]])
    target = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
                           [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    w_t = torch.tensor([0.05, 0.05, 0.05, 0.05, 0.30, 0.50])
    Pi = sinkhorn_coupling(source, target, w_t, reg=0.05).detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4),
                             gridspec_kw={"width_ratios": [1.0, 1.4]})
    ax = axes[0]
    ax.imshow(Pi, cmap="viridis", aspect="auto")
    ax.set_xlabel("target j")
    ax.set_ylabel("source i")
    ax.set_title("Coupling  $\\Pi^*$  (Sinkhorn)")
    for i in range(Pi.shape[0]):
        for j in range(Pi.shape[1]):
            if Pi[i, j] > 0.04:
                ax.text(j, i, f"{Pi[i,j]:.2f}", ha="center", va="center",
                        color="white" if Pi[i, j] < 0.12 else "black", fontsize=8)

    ax = axes[1]
    src = source.numpy()
    tgt = target.numpy()
    ax.scatter(src[:, 0], src[:, 1], c="#1F77B4", s=70, label="source", zorder=3)
    ax.scatter(tgt[:, 0], tgt[:, 1], c=tgt[:, 1] + 0.3, cmap="Reds",
               s=140, label="target", zorder=3)
    for i, t in enumerate(tgt):
        ax.annotate(str(i), (t[0], t[1]), color="white",
                    ha="center", va="center", fontsize=8, fontweight="bold")
    for i in range(Pi.shape[0]):
        for j in range(Pi.shape[1]):
            if Pi[i, j] > 0.02:
                ax.plot([src[i, 0], tgt[j, 0]], [src[i, 1], tgt[j, 1]],
                        color="gray", alpha=float(Pi[i, j] * 6), lw=1.2)
    ax.set_xlim(-0.4, 2.4)
    ax.set_ylim(-0.4, 1.4)
    ax.set_title("edges weighted by $\\Pi^*$")
    ax.legend(loc="lower right")
    ax.text(0.02, 0.96, "a", transform=ax.transAxes, fontsize=12, fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(OUT, f"fig4_sinkhorn_coupling.{ext}")
        fig.savefig(p, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"  saved {p}")
    plt.close(fig)


if __name__ == "__main__":
    fig1_flow_trajectory()
    fig2_convergence_curves()
    fig3_csa_null_hypothesis()
    fig4_sinkhorn_coupling()
    print("done.")
