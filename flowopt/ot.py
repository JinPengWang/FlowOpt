import torch
import numpy as np

def sinkhorn_ot(x_source, x_target, weights_target=None, reg=0.03, max_iter=30, tol=1e-4):
    """
    Computes the Entropic Optimal Transport coupling matrix between source and target particles
    using the stabilized Sinkhorn-Knopp algorithm.
    """
    N, D = x_source.shape
    M, _ = x_target.shape
    device = x_source.device
    
    # Source distribution: uniform over particles 1/N
    p = torch.full((N,), 1.0 / N, device=device, dtype=x_source.dtype)
    
    # Target distribution: normalized Gibbs weights
    if weights_target is None:
        q = torch.full((M,), 1.0 / M, device=device, dtype=x_source.dtype)
    else:
        q = weights_target / (weights_target.sum() + 1e-12)
        
    # Cost matrix C_ij = ||x_i - y_j||^2
    x_source_sq = (x_source ** 2).sum(dim=-1, keepdim=True) # (N, 1)
    x_target_sq = (x_target ** 2).sum(dim=-1, keepdim=True).t() # (1, M)
    cost = x_source_sq + x_target_sq - 2.0 * torch.matmul(x_source, x_target.t()) # (N, M)
    cost = torch.clamp(cost, min=0.0)
    
    # Normalize cost for numerical stability
    cost_scale = torch.median(cost) + 1e-6
    cost_norm = cost / cost_scale
    
    u = torch.zeros(N, device=device, dtype=x_source.dtype)
    v = torch.zeros(M, device=device, dtype=x_source.dtype)
    u_prev = u
    
    for it in range(max_iter):
        mat1 = (-cost_norm + v.unsqueeze(0)) / reg
        u = reg * (torch.log(p + 1e-16) - torch.logsumexp(mat1, dim=1))
        
        mat2 = (-cost_norm + u.unsqueeze(1)) / reg
        v = reg * (torch.log(q + 1e-16) - torch.logsumexp(mat2, dim=0))
        
        if it % 5 == 4:
            if torch.max(torch.abs(u - u_prev)) < tol:
                break
            u_prev = u
            
    # Compute coupling Pi = exp((u + v - cost_norm) / reg)
    log_pi = (u.unsqueeze(1) + v.unsqueeze(0) - cost_norm) / reg
    coupling = torch.exp(log_pi)
    coupling = coupling / (coupling.sum() + 1e-12) # enforce sum to 1
    
    # Compute barycentric projection: matched_targets[i] = sum_j (Pi_ij / p_i) * y_j
    coupling_cond = coupling / (coupling.sum(dim=1, keepdim=True) + 1e-12) # (N, M)
    matched_targets = torch.matmul(coupling_cond, x_target) # (N, D)
    
    return coupling, matched_targets

def greedy_nearest_ot(x_source, x_target, weights_target=None):
    """
    Greedy nearest neighbor matching with Gibbs weighting.
    Ultra-fast O(N * M) approximation when Sinkhorn is bypassed.
    """
    N, D = x_source.shape
    M, _ = x_target.shape
    x_source_sq = (x_source ** 2).sum(dim=-1, keepdim=True)
    x_target_sq = (x_target ** 2).sum(dim=-1, keepdim=True).t()
    dist = x_source_sq + x_target_sq - 2.0 * torch.matmul(x_source, x_target.t())
    
    if weights_target is not None:
        # Bias distance towards higher Gibbs weights
        log_w = torch.log(weights_target + 1e-12).unsqueeze(0)
        dist = dist - 0.5 * log_w
        
    nearest_idx = torch.argmin(dist, dim=1)
    matched_targets = x_target[nearest_idx]
    return nearest_idx, matched_targets
