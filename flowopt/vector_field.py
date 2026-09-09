import torch
import torch.nn as nn
import numpy as np

class NonParametricVectorField:
    """
    Closed-form non-parametric velocity field estimator for Flow Matching.
    Computes v(x, t) = E[x_1 - x_0 | x_t = x] via Nadaraya-Watson kernel regression
    with optimal transport paired particles, augmented with a repulsive dispersion term.
    
    This eliminates the need for training a neural network, reducing per-iteration
    compute from seconds to microseconds.
    """
    def __init__(self, repulsive_weight=0.01, adaptive_bandwidth=True, min_bandwidth=1e-3):
        self.repulsive_weight = repulsive_weight
        self.adaptive_bandwidth = adaptive_bandwidth
        self.min_bandwidth = min_bandwidth

    def compute_bandwidth(self, x_t):
        N, D = x_t.shape
        if N <= 1:
            return torch.tensor(1.0, device=x_t.device, dtype=x_t.dtype)
        # Pairwise distance median heuristic
        x_sq = (x_t ** 2).sum(dim=-1, keepdim=True)
        dist_sq = x_sq + x_sq.t() - 2.0 * torch.matmul(x_t, x_t.t())
        dist_sq = torch.clamp(dist_sq, min=0.0)
        dist = torch.sqrt(dist_sq + 1e-12)
        
        # Take median of upper triangular entries
        triu_idx = torch.triu_indices(N, N, offset=1)
        pairwise_dists = dist[triu_idx[0], triu_idx[1]]
        if len(pairwise_dists) > 0:
            median_dist = torch.median(pairwise_dists)
            h = median_dist / (np.sqrt(2.0 * np.log(N + 1.0)) + 1e-6)
            h = torch.clamp(h, min=self.min_bandwidth)
        else:
            h = torch.tensor(1.0, device=x_t.device, dtype=x_t.dtype)
        return h

    def evaluate(self, x_query, x_t, velocities, bandwidth=None):
        """
        Evaluate the velocity field at query positions x_query.
        
        Args:
            x_query: Tensor (Q, D) - evaluation points
            x_t: Tensor (N, D) - current trajectory points at time t
            velocities: Tensor (N, D) - conditional velocity u = x_1 - x_0
            bandwidth: scalar bandwidth h (optional, auto-computed if None)
            
        Returns:
            v_field: Tensor (Q, D) - evaluated continuous velocity vector field
        """
        Q, D = x_query.shape
        N, _ = x_t.shape
        
        if bandwidth is None:
            h = self.compute_bandwidth(x_t)
        else:
            h = bandwidth
            
        # Pairwise distance between query points and particle positions at time t
        q_sq = (x_query ** 2).sum(dim=-1, keepdim=True) # (Q, 1)
        p_sq = (x_t ** 2).sum(dim=-1, keepdim=True).t() # (1, N)
        dist_sq = q_sq + p_sq - 2.0 * torch.matmul(x_query, x_t.t()) # (Q, N)
        dist_sq = torch.clamp(dist_sq, min=0.0)
        
        # Gaussian kernel weights: K_ij = exp(-||x_q - x_t_i||^2 / (2 h^2))
        kernel = torch.exp(-dist_sq / (2.0 * (h ** 2) + 1e-12)) # (Q, N)
        weights = kernel / (kernel.sum(dim=1, keepdim=True) + 1e-12) # (Q, N)
        
        # Flow Matching transport velocity: v_transport = sum_i w_i * u_i
        v_transport = torch.matmul(weights, velocities) # (Q, D)
        
        # Repulsive dispersion term to prevent premature particle collapse:
        # v_rep = (alpha / h^2) * sum_i w_i * (x_query - x_t_i)
        if self.repulsive_weight > 0:
            diff = x_query.unsqueeze(1) - x_t.unsqueeze(0) # (Q, N, D)
            repulsion = (weights.unsqueeze(-1) * diff).sum(dim=1) / (h + 1e-6) # (Q, D)
            v_field = v_transport + self.repulsive_weight * repulsion
        else:
            v_field = v_transport
            
        return v_field


class MLPVectorFieldNet(nn.Module):
    """
    Lightweight MLP parameterizing the velocity field v_theta(x, t) for Flow Matching.
    Input: (x, t) -> Output: v (D-dimensional velocity vector)
    """
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )
        
    def forward(self, x, t):
        # x: (B, D), t: (B, 1)
        inp = torch.cat([x, t], dim=-1)
        return self.net(inp)


class NeuralVectorField:
    """
    Online Parametric Flow Matching Vector Field learner.
    Trains an MLP online via least-squares Flow Matching loss:
    L_FM(theta) = E_{t, i} || v_theta(x_t, t) - u_i ||^2
    """
    def __init__(self, dim, hidden_dim=64, lr=1e-2, epochs=30):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.model = None

    def fit_and_evaluate(self, x_source, x_target, x_query, t_eval=0.5):
        """
        Trains the neural vector field on paired particles (x_0, x_1) and evaluates at x_query.
        """
        device = x_source.device
        N, D = x_source.shape
        self.model = MLPVectorFieldNet(dim=D, hidden_dim=self.hidden_dim).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        u = x_target - x_source # (N, D)
        
        # Train for a few epochs
        for _ in range(self.epochs):
            t = torch.rand(N, 1, device=device) # (N, 1)
            x_t = (1.0 - t) * x_source + t * x_target # (N, D)
            v_pred = self.model(x_t, t)
            loss = torch.mean((v_pred - u) ** 2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Evaluate
        self.model.eval()
        with torch.no_grad():
            t_tensor = torch.full((x_query.shape[0], 1), t_eval, device=device)
            v_query = self.model(x_query, t_tensor)
        return v_query
