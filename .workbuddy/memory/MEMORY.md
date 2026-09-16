# MEMORY.md  --  Project  Long-Term Notes

## Project: FlowOpt
Continuous-time generative optimization via Entropic OT Flow Matching.
Submission target: IEEE Wireless Communications / similar Q1 venue.

## Hard conventions
- **v2 algorithm (canonical, single class):** `flowopt/optimizer.py` -- ~180 lines,
  one `FlowOpt` class with a single `step(...)` method that is exactly nine
  commented lines. README and the tutorial notebook (`tutorial.ipynb`) both
  reference this layout.  **Do not** add rank-1 path, hsig, or extra state
  without explicit user request -- the user has repeatedly asked for
  "no module/parameter stacking".

- **Zero user-tunable parameters.**  All internal constants
  `mu_eff, c_σ, d_σ, α_C, χ_D` are derived from `(D, N)` only.
  The `eps_schedule=(0.5, 1e-3)` argument has a sensible default and is
  the only thing a user can pass that affects algorithm behaviour.

- **Public API surface:** `from flowopt import FlowOpt, sinkhorn_coupling`.
  Baselines live in `flowopt.baselines` and use third-party `cma`
  (added to `requirements.txt`).

- **Working environment:** Python 3.13 via the managed runtime under
  `~/.workbuddy/binaries/python/versions/3.13.12/python.exe`.  Project-local
  venv lives at `flowopt-env/` (named `flowopt-env`, NOT `flowopt` -- that name
  collides with the package dir `flowopt/`); never run pip against the system
  interpreter.

- **VS Code / Jupyter setup (required, non-obvious):** a venv named
  `flowopt-env` is NOT auto-discovered by VS Code.  Three things make it
  visible, all now committed/in place:
    1. `ipykernel` (+ `jupyter`) installed in the venv -- without ipykernel the
       env is silently absent from the kernel picker;
    2. kernel registered: `python -m ipykernel install --user --name
       flowopt-env --display-name "Python (flowopt-env)"`;
    3. `.vscode/settings.json` sets `python.defaultInterpreterPath`, and
       `tutorial.ipynb` metadata sets `kernelspec.name = "flowopt-env"`.
  Both `ipykernel`/`jupyter` are in `requirements.txt` so a fresh install
  reproduces this.

## Performance baseline (D=10, N=30, 200 iters, 5 seeds)

| Problem    | FlowOpt v2 | CMA-ES (cma lib) | Winner   |
|------------|-----------:|------------------:|----------|
| Sphere     |     0.0    |        1.7e-14    | FlowOpt  |
| Rosenbrock |     5.20   |        0.148      | CMA-ES   |
| Rastrigin  |     3.58   |        5.17       | FlowOpt  |
| Ackley     |   4.8e-06  |       4.8e-06     | tie      |
| Levy       |   7.6e-15  |       1.1e-13     | FlowOpt  |
| Griewank   |   2.0e-03  |       1.5e-03     | CMA-ES   |

Rosenbrock gap (~35x) is the cost of the pure rank-μ covariance update
(no rank-1 path).  If user requests SOTA Rosenbrock, add a single
c_1 = α_C / D term driven by the existing p_sigma path -- this does NOT
require new state, only one new constant.  **Do not pre-emptively add
this** -- the user has signalled preference for "clean, single-step".

At 1000 iters, Rosenbrock drops to ~0.5, so the algorithm is converging,
just slowly.

## Code policy
- `flowopt/optimizer.py` is the only place the algorithm should live.
  Anything that requires touching multiple modules to add a feature is a
  smell.
- Dead code / historical variants get deleted, not archived.  Use the
  venv + git for keeping old versions if the user wants them back.
- Test invariants derive from `(D, N)`; never hand-tune magic numbers
  to make a test pass.

## Dated log
- 2026-09-12 -- v2 rewrite.  See `2026-09-12.md` for details.
  Later same day -- GPU build (torch 2.6.0+cu124) and notebook re-exec
  path (use `flowopt-env\Scripts\jupyter-nbconvert.exe`, NOT
  `python -m jupyter nbconvert`, which PATH-resolves to Anaconda and
  raises `ModuleNotFoundError: No module named 'torch'`).  Tutorial cell
  15 patched for device-portability.
