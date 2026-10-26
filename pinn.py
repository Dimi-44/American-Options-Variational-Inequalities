"""
Physics-informed neural network for the American put (thesis, Chapter 5).

Calendar-time form, with B the Black-Scholes residual (thesis, eq. 5.1):

    B[u] = u_t + 1/2 sigma^2 S^2 u_SS + r S u_S - r u,
    min{-B[u], u - Phi} = 0,   u(S, T) = Phi(S) = max(K - S, 0)       (thesis, eq. 5.2).

Architecture (thesis, Sections 5.3.2 and 5.3.4): 4 hidden layers of 64 tanh units and the
boundary-conditioned output u = K exp(-S/S_max) + (1 - exp(-S/S_max)) f_theta(S, t)
(eq. 5.4), so that u(0, t) = K holds exactly. Everything is computed in units of K
(S/K, u/K), which leaves the equation unchanged and keeps the network inputs and
outputs of order one.

Losses (`loss=`):
  * "vi" (default; thesis, Section 5.3.3): the residual of the variational inequality,
        L = 100 * mean(min(-B[u], u - Phi)^2)                         (eq. 5.3)
          + 50 * L_term + 50 * L_bnd,
    with L_term = mean((u(S, T) - Phi(S))^2) (eq. 5.5) and
    L_bnd = mean((u(0, t) - K)^2) + 0.1 * mean(u(S_max, t)^2). The first term of L_bnd is
    zero by construction of the output layer, so relative to the VI term this is
    eq. (5.7) with lambda_term = 0.5 and lambda_far = 0.05.
  * "original": the loss of the June 2025 submission, kept as a baseline,
        L = 1 * ||B[u]||^2 + 10 * ||(Phi - u)^+||^2 + 30 * ||(u - Phi) B[u]||^2
          + 50 * L_term + 50 * L_bnd.
    Its PDE term asks B[u] = 0 everywhere, including the exercise region where
    B[Phi] = -rK, so it competes with the obstacle constraint. Note: because the loss is
    computed in units of K, the product term scales as K^4 while the other terms scale as
    K^2, so the complementarity weight here is effectively 30 / K^2 of the weight the
    same formula has in price units.
    Pass `weights=REWEIGHTED` for the reweighted variant (0.7 / 117 / 97), the weights the
    author used in an earlier perpetual-put experiment with the same loss.

Training: Adam, learning rate 1e-3 divided by 5 after 60% and again after 85% of the
iterations; 4096 interior, 512 terminal and 256 boundary points, redrawn every 100
iterations (see `_sample`).
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

# weights of the original (June 2025) loss; term and bnd are also used by the "vi" loss
WEIGHTS = dict(pde=1.0, ineq=10.0, comp=30.0, term=50.0, bnd=50.0)
# reweighted variant of the original loss (author's earlier perpetual-put experiment)
REWEIGHTED = dict(pde=0.7, ineq=117.0, comp=97.0, term=50.0, bnd=50.0)


class PINN(nn.Module):
    def __init__(self, width=64, depth=4, x_max=2.0, T=1.0):
        super().__init__()
        layers, d_in = [], 2
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), nn.Tanh()]
            d_in = width
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)
        self.x_max, self.T = x_max, T

    def forward(self, x, t):
        """x = S/K, t in [0, T]; returns u/K."""
        z = torch.stack([x / self.x_max * 2 - 1, t / self.T * 2 - 1], dim=-1)
        f = self.net(z).squeeze(-1)
        w = torch.exp(-x / self.x_max)
        return w + (1 - w) * f            # u(0, t) = K exactly (K = 1 in scaled units)


def _sample(n, x_max, T, gen):
    """Interior collocation points, biased towards the strike and towards t = 0 and t = T."""
    half = n // 2
    x_uni = torch.rand(half, generator=gen) * x_max
    x_atm = (1.0 + 0.25 * torch.randn(n - half, generator=gen)).clamp(0.0, x_max)
    t_uni = torch.rand(half, generator=gen) * T
    beta = torch.distributions.Beta(0.5, 0.5)
    t_end = beta.sample((n - half,)) * T
    return torch.cat([x_uni, x_atm]), torch.cat([t_uni, t_end])


def train_pinn(K=50.0, r=0.05, sigma=0.2, T=1.0, S_max=100.0, loss="vi", vi_weight=100.0, weights=None,
               iterations=20_000, lr=1e-3, n_interior=4096, n_terminal=512, n_boundary=256,
               width=64, depth=4, resample_every=100, seed=0, log_every=0):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    W = dict(WEIGHTS, **(weights or {}))
    x_max = S_max / K
    model = PINN(width, depth, x_max, T)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(0.6 * iterations), int(0.85 * iterations)], 0.2)
    payoff = lambda x: torch.clamp(1.0 - x, min=0.0)
    t0 = time.perf_counter()
    history = []

    for it in range(iterations):
        if it % resample_every == 0:
            x, t = _sample(n_interior, x_max, T, gen)
            x.requires_grad_(True)
            t.requires_grad_(True)
            x_T = torch.rand(n_terminal, generator=gen) * x_max
            t_b = torch.rand(n_boundary, generator=gen) * T

        u = model(x, t)
        u_t, u_x = torch.autograd.grad(u.sum(), (t, x), create_graph=True)
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        B = u_t + 0.5 * sigma**2 * x**2 * u_xx + r * x * u_x - r * u
        gap = u - payoff(x)

        l_term = ((model(x_T, torch.full_like(x_T, T)) - payoff(x_T))**2).mean()
        l_bnd = ((model(torch.zeros_like(t_b), t_b) - 1.0)**2).mean() \
            + 0.1 * (model(torch.full_like(t_b, x_max), t_b)**2).mean()
        if loss == "vi":
            # residual of the variational inequality itself: min{-B[u], u - Phi} = 0
            l_vi = (torch.minimum(-B, gap)**2).mean()
            total = vi_weight * l_vi + W["term"] * l_term + W["bnd"] * l_bnd
        elif loss == "original":
            l_pde = (B**2).mean()
            l_ineq = (torch.relu(-gap)**2).mean()
            l_comp = ((gap * B)**2).mean()
            total = (W["pde"] * l_pde + W["ineq"] * l_ineq + W["comp"] * l_comp
                     + W["term"] * l_term + W["bnd"] * l_bnd)
        else:
            raise ValueError(f"unknown loss {loss!r}; use 'vi' or 'original'")

        opt.zero_grad()
        total.backward()
        opt.step()
        sched.step()
        if log_every and it % log_every == 0:
            history.append((it, total.item()))
            print(f"  it {it:6d}  loss {total.item():.3e}", flush=True)

    model.eval()
    return model, {"time": time.perf_counter() - t0, "history": history, "K": K, "T": T}


def evaluate(model, info, S, t=0.0):
    """Option value at prices S (array) and calendar time t."""
    K = info["K"]
    with torch.no_grad():
        x = torch.tensor(np.asarray(S, dtype=np.float32) / K)
        u = model(x, torch.full_like(x, t))
    return u.numpy().astype(np.float64) * K


def vi_terms(model, info, S, t, r=0.05, sigma=0.2):
    """The two arguments of min{-B[u], u - Phi} (in units of K) at prices S and calendar time t."""
    K = info["K"]
    x = torch.tensor(np.asarray(S, dtype=np.float32) / K, requires_grad=True)
    tt = torch.full_like(x, float(t), requires_grad=True)
    u = model(x, tt)
    u_t, u_x = torch.autograd.grad(u.sum(), (tt, x), create_graph=True)
    u_xx = torch.autograd.grad(u_x.sum(), x)[0]
    B = u_t + 0.5 * sigma**2 * x**2 * u_xx + r * x * u_x - r * u
    gap = u - torch.clamp(1.0 - x, min=0.0)
    return (-B).detach().numpy(), gap.detach().numpy()


def exercise_boundary(model, info, times, S_grid, r=0.05, sigma=0.2):
    """Early-exercise boundary of the network, read off from the VI itself.

    A point belongs to the exercise region when the min in min{-B[u], u - Phi} is
    attained by the payoff term (u - Phi < -B[u]); for the exact solution this is
    exactly the set {V = Phi}. The boundary is the largest such S below the strike.
    """
    K = info["K"]
    out = []
    for t in times:
        neg_B, gap = vi_terms(model, info, S_grid, t, r, sigma)
        ex = (gap < neg_B) & (S_grid < K) & (S_grid > 0)
        out.append(S_grid[ex].max() if ex.any() else np.nan)
    return np.array(out)


def load_pinn(path, K=50.0, T=1.0, S_max=100.0, width=64, depth=4):
    model = PINN(width, depth, S_max / K, T)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model, {"K": K, "T": T, "time": float("nan")}
