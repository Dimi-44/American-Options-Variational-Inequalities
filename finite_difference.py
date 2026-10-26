"""
Grid-based solvers for the American put written as a parabolic variational
inequality (thesis, Chapters 3-4).

Time-to-expiry form (q = T - t), with the elliptic operator written with the
sign convention of Chapters 1-2 (A is monotone/coercive):

    A V = -1/2 sigma^2 S^2 V_SS - r S V_S + r V,                       (thesis, eq. 3.11)

    min{ V_q + A V,  V - psi } = 0,     V(0, S) = psi(S) = max(K - S, 0),  (eq. 3.14 / 4.6)
    V(q, 0) = K,   V(q, S_max) = 0.

Both solvers use the Crank-Nicolson discretisation of Chapter 4 with central
differences in S, plus Rannacher start-up: the first 4 steps are fully implicit,
which removes the oscillations that Crank-Nicolson produces from the payoff kink.
(The start-up is an implementation detail that the thesis does not describe.)

* penalty_fd   : penalised equation  V_q + A V + (1/eps) * min(V - psi, 0) = 0, with
                 the penalty at the new time level, solved by Newton (policy)
                 iteration at every step: each iteration is one tridiagonal solve,
                 and it stops when the set of penalised nodes no longer changes
                 (thesis, Section 4.1.3; Forsyth & Vetzal, 2002).
* projected_sor: the linear complementarity problem of each time step (eq. 4.7),
                 solved by projected SOR (eqs. 4.8-4.9; Cryer, 1971), with Young's
                 optimal relaxation parameter omega. This is the efficient form of the
                 projected fixed point V = P_K(V - alpha * residual), P_K(z) = max(z, psi).
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.linalg import solve_banded

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*a, **k):
        return a[0] if a and callable(a[0]) else (lambda f: f)


def _grid(K, S_max, M):
    S = np.linspace(0.0, S_max, M + 1)
    return S, np.maximum(K - S, 0.0)


def _operator_coefficients(S, r, sigma, dS):
    """Tridiagonal coefficients of A on interior nodes: (A V)_i = l V_{i-1} + d V_i + u V_{i+1}."""
    s = S[1:-1]
    a = 0.5 * sigma**2 * s**2 / dS**2
    b = r * s / (2 * dS)
    lower = -(a - b)
    diag = 2 * a + r
    upper = -(a + b)
    return lower, diag, upper


def _apply_A(V, lower, diag, upper):
    return lower * V[:-2] + diag * V[1:-1] + upper * V[2:]


def _time_steps(N, rannacher):
    """theta for every step: 1 (implicit) for the Rannacher steps, then 0.5."""
    return [1.0 if n < rannacher else 0.5 for n in range(N)]


def _exercise_boundary(S, V, psi, K, tol=1e-6):
    """Largest S < K where V equals the payoff (the free boundary S*)."""
    ex = (V - psi <= tol) & (S < K) & (S > 0)
    return float(S[ex].max()) if ex.any() else float("nan")


# ------------------------------------------------------------------------------
# Penalty method
# ------------------------------------------------------------------------------
def penalty_fd(K=50.0, r=0.05, sigma=0.2, T=1.0, S_max=100.0, M=400, N=400,
               eps=1e-6, rannacher=4, max_newton=50, record_boundary=True):
    t0 = time.perf_counter()
    S, psi = _grid(K, S_max, M)
    dS, dq = S_max / M, T / N
    lower, diag, upper = _operator_coefficients(S, r, sigma, dS)
    V = psi.copy()
    boundary = [K] if record_boundary else None
    newton_its = 0
    psi_in = psi[1:-1]

    for theta in _time_steps(N, rannacher):
        old = V.copy()
        # boundary values at the new time level
        V[0], V[-1] = K, 0.0
        rhs_base = old[1:-1] / dq - (1 - theta) * _apply_A(old, lower, diag, upper)
        # contributions of the known boundary values to the implicit part
        rhs_base[0] -= theta * lower[0] * V[0]
        rhs_base[-1] -= theta * upper[-1] * V[-1]

        x = old[1:-1].copy()
        for _ in range(max_newton):
            newton_its += 1
            active = x < psi_in                      # penalty switched on where V < psi
            pen = active / eps
            ab = np.zeros((3, len(x)))
            ab[0, 1:] = theta * upper[:-1]
            ab[1, :] = 1.0 / dq + theta * diag + pen
            ab[2, :-1] = theta * lower[1:]
            rhs = rhs_base + pen * psi_in
            x_new = solve_banded((1, 1), ab, rhs)
            if np.array_equal(x_new < psi_in, active):
                x = x_new
                break
            x = x_new
        V[1:-1] = x
        if record_boundary:
            boundary.append(_exercise_boundary(S, V, psi, K, tol=1e-5))

    return {
        "S": S, "V": V, "boundary": np.array(boundary) if record_boundary else None,
        "q": np.linspace(0, T, N + 1), "time": time.perf_counter() - t0,
        "iterations": newton_its, "min_violation": float((V - psi).min()),
    }


# ------------------------------------------------------------------------------
# Projection method (projected SOR)
# ------------------------------------------------------------------------------
@njit(cache=True)
def _psor_solve(x, rhs, lo, di, up, psi, omega, tol, max_iter):
    n = x.shape[0]
    it = 0
    for it in range(1, max_iter + 1):
        err = 0.0
        for i in range(n):
            s = rhs[i]
            if i > 0:
                s -= lo[i] * x[i - 1]
            if i < n - 1:
                s -= up[i] * x[i + 1]
            gs = s / di[i]
            new = x[i] + omega * (gs - x[i])
            if new < psi[i]:
                new = psi[i]                       # projection P_K
            d = abs(new - x[i])
            if d > err:
                err = d
            x[i] = new
        if err < tol:
            break
    return it


def optimal_omega(lo, di, up):
    """Young's optimal relaxation parameter for the tridiagonal matrix."""
    off = np.sqrt(np.maximum(lo[1:] * up[:-1], 0.0) / (di[:-1] * di[1:]))
    J = np.diag(off, 1) + np.diag(off, -1)
    rho = float(np.max(np.abs(np.linalg.eigvalsh(J))))
    return 2.0 / (1.0 + math.sqrt(max(1.0 - rho**2, 0.0)))


def projected_sor(K=50.0, r=0.05, sigma=0.2, T=1.0, S_max=100.0, M=400, N=400,
                  omega="auto", tol=1e-9, rannacher=4, max_iter=10_000, record_boundary=True):
    t0 = time.perf_counter()
    S, psi = _grid(K, S_max, M)
    dS, dq = S_max / M, T / N
    lower, diag, upper = _operator_coefficients(S, r, sigma, dS)
    V = psi.copy()
    boundary = [K] if record_boundary else None
    sweeps = 0
    psi_in = psi[1:-1].copy()
    omegas = {}

    for theta in _time_steps(N, rannacher):
        old = V.copy()
        V[0], V[-1] = K, 0.0
        lo = theta * lower
        di = 1.0 / dq + theta * diag
        up = theta * upper
        if theta not in omegas:
            omegas[theta] = optimal_omega(lo, di, up) if omega == "auto" else float(omega)
        rhs = old[1:-1] / dq - (1 - theta) * _apply_A(old, lower, diag, upper)
        rhs[0] -= lo[0] * V[0]
        rhs[-1] -= up[-1] * V[-1]
        x = np.maximum(old[1:-1].copy(), psi_in)
        sweeps += _psor_solve(x, rhs, lo, di, up, psi_in, omegas[theta], tol, max_iter)
        V[1:-1] = x
        if record_boundary:
            boundary.append(_exercise_boundary(S, V, psi, K))

    return {
        "S": S, "V": V, "boundary": np.array(boundary) if record_boundary else None,
        "q": np.linspace(0, T, N + 1), "time": time.perf_counter() - t0,
        "iterations": sweeps, "omega": omegas.get(0.5), "min_violation": float((V - psi).min()),
    }


def price_at(result, S0):
    return float(np.interp(S0, result["S"], result["V"]))
