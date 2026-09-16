"""
FlowOpt : Continuous-Time Generative Optimization
==================================================

Public surface:

    from flowopt import FlowOpt                     # main optimizer
    from flowopt.benchmarks import Sphere, Rosenbrock, ...
    from flowopt.baselines import CMAESOptimizer, ... # baselines (optional)
    from flowopt.ot import sinkhorn_coupling         # the FM ingredient

Version 2 introduces a single, hyperparameter-free optimizer that is a
literal instance of Entropic Optimal Transport Flow Matching.
"""

from .optimizer import FlowOpt, sinkhorn_coupling

__version__ = "2.0.0"
__all__ = ["FlowOpt", "sinkhorn_coupling"]
