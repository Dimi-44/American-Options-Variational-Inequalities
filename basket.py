"""
Extension: the American put in d dimensions.

Test problem: an American put on the GEOMETRIC average of d correlated stocks,

    payoff  Phi(S) = (K - G)^+,      G = (S_1 * ... * S_d)^(1/d),

where every S_i follows a GBM with rate r, volatility sigma and pairwise
correlation rho (no dividends). The geometric average of correlated GBMs is
itself a GBM, with

    sigma_G^2 = sigma^2 (1 + (d - 1) rho) / d,     q_G = (sigma^2 - sigma_G^2) / 2,

so the d-dimensional problem has an EXACT one-dimensional equivalent: an
American put on G with volatility sigma_G and dividend yield q_G. That 1-D
problem is priced to ~1e-4 with QuantLib and serves as the reference in every
dimension. The solvers below do not use this reduction; they work in the full
d-dimensional space.

Solvers:
  * pinn_basket   : the VI-residual PINN of the 1-D case, in d dimensions
  * lsm_basket    : Longstaff-Schwartz Monte Carlo (Bermudan, out-of-sample)
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn


# ------------------------------------------------------------------------------
# Exact reference via the 1-D reduction
# ------------------------------------------------------------------------------
def geometric_params(d, sigma, rho):
    sig_g = sigma * math.sqrt((1 + (d - 1) * rho) / d)
    q_g = 0.5 * (sigma**2 - sig_g**2)
    return sig_g, q_g


def reference_price(G0, d, K=50.0, r=0.05, sigma=0.2, rho=0.25, T=1.0,
                    exercise="american", n_dates=50, grid=2000):
    """American (or Bermudan with n_dates equally spaced dates) put on G, via QuantLib."""
    import QuantLib as ql
    sig_g, q_g = geometric_params(d, sigma, rho)
    today = ql.Date(1, ql.January, 2024)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    days = T * 365
    if exercise == "american":
        ex = ql.AmericanExercise(today, today + int(round(days)))
    else:
        ex = ql.BermudanExercise([today + int(round(days * i / n_dates)) for i in range(0, n_dates + 1)])  # incl. today, as in LSM
    opt = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Put, K), ex)
    proc = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(G0)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, q_g, dc)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, r, dc)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), sig_g, dc)))
    opt.setPricingEngine(ql.FdBlackScholesVanillaEngine(proc, grid, grid))
    return opt.NPV()


def correlation(d, rho):
    C = np.full((d, d), rho)
    np.fill_diagonal(C, 1.0)
    return C


# ------------------------------------------------------------------------------
# PINN in d dimensions
# ------------------------------------------------------------------------------
class BasketPINN(nn.Module):
    """u(t, x) in units of K, with x = log(S/K) in R^d.

    The network output is used directly (a smooth function); the terminal payoff
    is imposed through the loss, as in the 1-D PINN (thesis, eq. 5.5). Building the payoff into the
    output (u = Phi + (1 - t/T) f) does NOT work: u = Phi then has zero residual at
    every collocation point, because the payoff's kink -- where the PDE fails --
    has measure zero and is never sampled."""

    def __init__(self, d, width=128, depth=4, T=1.0, x_scale=0.5):
        super().__init__()
        layers, d_in = [], d + 1
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), nn.Tanh()]
            d_in = width
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)
        self.T, self.x_scale = T, x_scale

    @staticmethod
    def payoff(x):
        return torch.clamp(1.0 - torch.exp(x.mean(dim=-1)), min=0.0)

    def forward(self, t, x):
        z = torch.cat([(t / self.T * 2 - 1).unsqueeze(-1), x / self.x_scale], dim=-1)
        return self.net(z).squeeze(-1)


def _bs_operator(model, t, x, r, chol):
    """B[u] = u_t + 1/2 sum_ij Sigma_ij u_{x_i x_j} + (r - sigma^2/2) sum_i u_{x_i} - r u,
    with Sigma = chol @ chol.T, computed exactly through d directional second derivatives."""
    u = model(t, x)
    u_t, g = torch.autograd.grad(u.sum(), (t, x), create_graph=True)
    second = torch.zeros_like(u)
    for k in range(chol.shape[1]):
        lk = chol[:, k]
        dk = g @ lk                                             # derivative along l_k
        hk = torch.autograd.grad(dk.sum(), x, create_graph=True)[0]
        second = second + hk @ lk                               # l_k^T H l_k
    drift = (r - 0.5 * torch.diagonal(chol @ chol.T)) @ g.T     # sum_i (r - Sigma_ii/2) u_i
    return u, u_t + 0.5 * second + drift - r * u


def _sample(n, d, T, r, sigma, chol_np, gen, spread=1.5, base=0.25):
    """Collocation points where the solution matters: around the risk-neutral
    distribution of log-prices started near S0 = K, widened."""
    t = torch.rand(n, generator=gen) * T
    z = torch.randn(n, d, generator=gen)
    w = torch.randn(n, d, generator=gen)
    chol = torch.tensor(chol_np, dtype=torch.float32)
    x = ((r - 0.5 * sigma**2) * t).unsqueeze(-1) \
        + spread * torch.sqrt(t).unsqueeze(-1) * (z @ chol.T) \
        + base * w
    return t, x


def pinn_basket(d, K=50.0, r=0.05, sigma=0.2, rho=0.25, T=1.0, iterations=20_000,
                n_points=4096, n_terminal=1024, w_vi=100.0, w_term=50.0, lr=1e-3, width=128, depth=4, resample_every=100, seed=0,
                log_every=0, threads=None):
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    chol_np = np.linalg.cholesky(sigma**2 * correlation(d, rho))
    chol = torch.tensor(chol_np, dtype=torch.float32)
    model = BasketPINN(d, width, depth, T)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(0.6 * iterations), int(0.85 * iterations)], 0.2)
    t0 = time.perf_counter()
    for it in range(iterations):
        if it % resample_every == 0:
            t, x = _sample(n_points, d, T, r, sigma, chol_np, gen)
            t.requires_grad_(True)
            x.requires_grad_(True)
            _, x_T = _sample(n_terminal, d, T, r, sigma, chol_np, gen)
            x_T = x_T + ((r - 0.5 * sigma**2) * T)          # spread as at maturity
            t_T = torch.full((n_terminal,), T)
        u, B = _bs_operator(model, t, x, r, chol)
        gap = u - model.payoff(x)
        l_vi = (torch.minimum(-B, gap) ** 2).mean()            # VI residual, as in 1-D
        l_term = ((model(t_T, x_T) - model.payoff(x_T)) ** 2).mean()
        loss = w_vi * l_vi + w_term * l_term
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if log_every and it % log_every == 0:
            print(f"  d={d} it {it:6d} loss {loss.item():.3e}  ({time.perf_counter() - t0:.0f}s)", flush=True)
    model.eval()
    return model, {"time": time.perf_counter() - t0, "K": K, "T": T}


def pinn_price(model, info, S0):
    """Prices at t = 0 for an array of spot vectors S0 (n, d)."""
    x = torch.tensor(np.log(np.asarray(S0, dtype=np.float64) / info["K"]), dtype=torch.float32)
    with torch.no_grad():
        u = model(torch.zeros(x.shape[0]), x)
    return u.numpy().astype(np.float64) * info["K"]


# ------------------------------------------------------------------------------
# Longstaff-Schwartz Monte Carlo
# ------------------------------------------------------------------------------
def _features(S, K):
    """Constant, linear, squared and cross terms of S/K, plus the payoff."""
    X = S / K
    n, d = X.shape
    cols = [np.ones(n)] + [X[:, i] for i in range(d)] + [X[:, i] * X[:, j] for i in range(d) for j in range(i, d)]
    G = np.exp(np.log(X).mean(axis=1))
    cols.append(np.maximum(1 - G, 0))
    return np.column_stack(cols)


def _paths(S0, n, d, r, sigma, rho, T, n_dates, rng):
    dt = T / n_dates
    L = np.linalg.cholesky(correlation(d, rho))
    logS = np.log(np.asarray(S0, dtype=np.float64)) * np.ones((n, d))
    out = np.empty((n_dates + 1, n, d))
    out[0] = np.exp(logS)
    for k in range(1, n_dates + 1):
        logS = logS + (r - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * rng.standard_normal((n, d)) @ L.T
        out[k] = np.exp(logS)
    return out


def lsm_basket(S0, d, K=50.0, r=0.05, sigma=0.2, rho=0.25, T=1.0, n_dates=50,
               n_train=100_000, n_test=100_000, seed=0):
    """Bermudan price with n_dates exercise dates. The exercise rule is fitted on
    training paths and applied to independent test paths (a lower bound)."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    disc = math.exp(-r * T / n_dates)
    train = _paths(S0, n_train, d, r, sigma, rho, T, n_dates, rng)
    test = _paths(S0, n_test, d, r, sigma, rho, T, n_dates, rng)
    pay = lambda S: np.maximum(K - np.exp(np.log(S).mean(axis=-1)), 0.0)
    cf = [pay(train[-1]), pay(test[-1])]
    for k in range(n_dates - 1, 0, -1):
        cf = [c * disc for c in cf]
        ex_tr = pay(train[k])
        itm = ex_tr > 0
        if itm.sum() < 50:
            continue
        A = _features(train[k][itm], K)
        coef = np.linalg.lstsq(A, cf[0][itm], rcond=None)[0]
        for j, P in enumerate((train, test)):
            ex = pay(P[k])
            m = ex > 0
            cont = _features(P[k][m], K) @ coef
            idx = np.where(m)[0][ex[m] > cont]
            cf[j][idx] = ex[idx]
    vals = cf[1] * disc
    # exercise at t = 0 if it beats continuation
    price = max(float(vals.mean()), float(pay(np.asarray(S0, dtype=np.float64)[None, :])[0]))
    return price, float(vals.std() / math.sqrt(n_test)), time.perf_counter() - t0
