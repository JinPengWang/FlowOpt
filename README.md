# FlowOpt

**Continuous-Time Generative Optimization via Entropic Optimal Transport Flow Matching.**

A black-box continuous optimizer whose every internal constant is fixed by
the problem dimension and the population size. There are no knobs to tune.

```python
import torch
from flowopt import FlowOpt

def sphere(x):
    return torch.sum(x ** 2, dim=-1)

opt = FlowOpt(dim=10, bounds=(-5.12, 5.12))      # zero tuning
result = opt.optimize(sphere, max_iters=200)
print(result["best_f"])                          # -> ~ 0.0
```

---

## The Algorithm

We view every iteration as one step of a **Flow Matching** generative process
between two distributions:

| role | what it is |
|---|---|
| `p_0` (source) | current search Gaussian `N(m, sigma^2 C)` |
| `p_1` (target) | empirical elite distribution (top-mu by `f(x)`) |
| `x_t` | linear OT interpolant `(1-t) x_0 + t x_1` |
| `u_t` | FM conditional velocity `x_1 - x_0` (constant in `t`) |
| `Pi^*` | entropic OT coupling (Sinkhorn-Knopp) between the two |
| `target_i` | barycentric FM target for each particle |
| `m, sigma, C` | parameters of `p_0` updated by following the elites |

Update rule — exactly **nine lines** (see `flowopt/optimizer.py`):

1. draw `x_i ~ N(m, sigma^2 C)`
2. evaluate `f(x_i)`, rank, pick top-`mu` elites with log-weights `w`
3. solve Sinkhorn `Pi^*` between population and elites
4. barycentric target `t_i = (Pi^* / row_sum) @ elites`
5. velocity `u_i = t_i - x_i`  *(FM closed form)*
6. mean drift `m <- m + mean(u_i)`  *(one Euler step)*
7. CSA path  `p_sigma`  ->  `sigma <- sigma * exp((c_sigma/d_sigma) * (||p_sigma||/chi_D - 1))`
8. rank-mu cov  `C <- (1-alpha_C) C + alpha_C Cov_w(elites)`

Every internal constant comes from `(D, mu_eff) = (D, N)`:

```
mu          = N // 2
w_i         = log(mu + 0.5) - log(i + 1)          (rank-invariant)
mu_eff      = 1 / sum w_i^2
c_sigma     = mu_eff / (D + mu_eff)               # CSA learning rate
d_sigma     = 1 + c_sigma
alpha_C     = mu_eff / (D^2 + mu_eff)             # rank-mu cov rate
chi_D       = sqrt(D) (1 - 1/(4D) + 1/(32 D^2))   # E[||N(0, I_D)||]
```

Zero tunable hyperparameters. The population size `N` is the only argument
the user sets, and it is determined by the evaluation budget, not by tuning.

---

## Why Flow Matching, not heuristics

`pi`-coupled straight OT paths minimise the kinetic action of the flow.
That is what every other ingredient does *not* give you: greedy matching
creates crossings, convex combinations smooth out the elite signal, neural
velocity fields waste time on training and introduce instabilities.

| alternative | what goes wrong |
|---|---|
| greedy nearest-neighbour | crossing paths -> non-monotonic convergence |
| isotropic Gaussian update | flat search on ill-conditioned Rosenbrock |
| neural MLP velocity | 100x slower, unstable on small budgets |
| PSO / DE / CBO | depends on hand-tuned (w, c1, c2, F, CR, alpha, ...) |

A small heads-up: on Rosenbrock, a pure rank-mu covariance update without
a rank-1 direction memory converges ~3-5x slower than CMA-ES at equal budget;
the gap closes with more iterations.  See `tutorial.ipynb` Cell 10 for the
honest discussion.

---

## Installation

```bash
# 1. create venv (recommended)
python -m venv flowopt-env
flowopt-env/Scripts/activate      # Windows
source flowopt-env/bin/activate   # Linux / macOS

# 2. install PyTorch with CUDA support FIRST.
#    `pip install -r requirements.txt` alone will install the CPU-only torch
#    from the default PyPI index, which is almost never what you want.
#    Choose a CUDA build that matches your driver / toolkit:
#
#      cu124  -- NVIDIA driver >= 525.60 (recommended for Ampere/Ada/Hopper)
#      cu118  -- legacy driver (CUDA 11.8 toolkit, e.g. RTX 30xx older setups)
#      cu126  -- newer driver required
#      cu128  -- Linux only
#
#    Check what your machine has with `nvidia-smi` (top right shows the
#    "CUDA Version" the driver supports).

pip install --index-url https://download.pytorch.org/whl/cu124 torch

# 3. everything else from requirements.txt
pip install -r requirements.txt
```

If you genuinely want a CPU-only install (e.g. on a laptop with no NVIDIA
GPU), step 2 is enough on its own and step 3 brings in the rest.

Tested with Python 3.13, PyTorch 2.6.0 + CUDA 12.4 (RTX 3060 sm_86) and
the CPU-only fallback.  The optimizer auto-detects CUDA when available:

```python
opt = FlowOpt(dim=100, bounds=(-5, 5), pop_size=64)   # uses cuda:0 if present
opt = FlowOpt(dim=10,  bounds=(-5, 5), pop_size=30, device='cpu')  # force CPU
```

### Using it in VS Code

The venv is named `flowopt-env` (it cannot be called `flowopt` because that is
the package directory).  VS Code does **not** auto-discover venvs with
non-standard names, so three things are needed.  All three are already in the
repo:

1. `ipykernel` is installed in the venv (listed in `requirements.txt`) --
   without it the environment never appears in the kernel picker.
2. The kernel is registered with Jupyter:
   ```bash
   flowopt-env/Scripts/python.exe -m ipykernel install --user --name flowopt-env --display-name "Python (flowopt-env)"
   ```
3. `.vscode/settings.json` sets `python.defaultInterpreterPath` to the venv, and
   `tutorial.ipynb` has `"kernelspec": {"name": "flowopt-env"}` in its metadata.

Then open `tutorial.ipynb`, click **Select Kernel** (top-right) -> **Python
Environments** -> **Python (flowopt-env)**.  If it is still missing, run
*Python: Select Interpreter* from the command palette and pick
`./flowopt-env/Scripts/python.exe`, then restart the kernel.

---

## Project layout

```
.
|-- flowopt/
|   |-- __init__.py
|   |-- optimizer.py        # the entire algorithm  (~ 180 lines)
|   |-- ot.py               # Sinkhorn-Knopp          (~  60 lines)
|   |-- benchmarks.py       # test functions          (Sphere, Rosenbrock, ...)
|   `-- baselines.py        # CMA-ES, DE, PSO, CBO, CEM (for comparisons)
|-- tests/
|   `-- test_flowopt.py     # six invariant / behavior checks
|-- visualizations/
|   `-- plot_figures.py     # publication-quality figures (PDF + PNG)
|-- experiments/
|   `-- run_full_benchmarks.py  # head-to-head vs. baselines
|-- paper/                  # LaTeX manuscript + supplementary material
|-- figures/                # generated output (PNG + PDF)
|-- results/                # JSON traces from the benchmark runs
|-- requirements.txt
`-- README.md
```

Run the unit tests with:

```bash
python tests/test_flowopt.py
```

Run the head-to-head benchmark with:

```bash
python experiments/run_full_benchmarks.py
```

Build the figures with:

```bash
python visualizations/plot_figures.py
```

For an in-depth walkthrough, open the tutorial notebook:

```bash
jupyter notebook tutorial.ipynb
```

---

## License

MIT.
