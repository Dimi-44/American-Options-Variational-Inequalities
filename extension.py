"""
Extension: the American put in d = 1, 2, 3, 5, 10 dimensions.

American put on the geometric average of d correlated stocks (see pvi/basket.py),
priced by the d-dimensional VI-PINN and by Longstaff-Schwartz Monte Carlo, and
compared with the exact price from the 1-D reduction.

    python extension.py            # uses trained networks in results/extension/ if present
    python extension.py --retrain  # retrain (about 3 hours on a 2-core CPU in total)

Parameters: K = 50, r = 5%, sigma = 20% for every stock, pairwise correlation 0.25, T = 1.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pvi.basket import (BasketPINN, geometric_params, lsm_basket, pinn_basket, pinn_price,
                        reference_price)

DIMS = [1, 2, 3, 5, 10]
P = dict(K=50.0, r=0.05, sigma=0.2, rho=0.25, T=1.0)
ITERATIONS, POINTS = 20_000, 2048
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "extension")
C = {"ref": "#1f1f1e", "pinn": "#1baf7a", "lsm": "#2a78d6", "muted": "#6b6a63", "grid": "#e4e3dd"}


def test_points(d, seed=1):
    """5 symmetric spots (all stocks equal, G0 = 40..60) and 10 asymmetric ones."""
    rng = np.random.default_rng(seed)
    sym = np.array([[g] * d for g in (40, 45, 50, 55, 60)], float)
    asym = np.exp(np.log(rng.uniform(40, 60, (10, 1))) + rng.normal(0, 0.15, (10, d)))
    return np.vstack([sym, asym])


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel(xlabel, color=C["muted"])
    ax.set_ylabel(ylabel, color=C["muted"])
    ax.grid(True, color=C["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def main(retrain):
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    rows, curves = [], {}
    for d in DIMS:
        pts = test_points(d)
        G = np.exp(np.log(pts).mean(axis=1))
        ref_am = np.array([reference_price(g, d, **P) for g in G])
        ref_be = np.array([reference_price(g, d, **P, exercise="bermudan", n_dates=50) for g in G])

        weights = os.path.join(OUT, f"pinn_d{d}.pt")
        if retrain or not os.path.exists(weights):
            print(f"d = {d}: training PINN ...", flush=True)
            model, info = pinn_basket(d, **P, iterations=ITERATIONS, n_points=POINTS, log_every=5000)
            torch.save(model.state_dict(), weights)
            with open(weights + ".time", "w") as f:
                f.write(str(info["time"]))
        else:
            model = BasketPINN(d, T=P["T"])
            model.load_state_dict(torch.load(weights))
            model.eval()
            info = {"K": P["K"], "T": P["T"], "time": float(open(weights + ".time").read())}
        v_pinn = pinn_price(model, info, pts)

        lsm = [lsm_basket(p, d, **P, n_dates=50) for p in pts[:5]]       # symmetric spots only
        v_lsm = np.array([x[0] for x in lsm])
        se_lsm = np.array([x[1] for x in lsm])
        t_lsm = float(np.mean([x[2] for x in lsm]))

        err_pinn = np.abs(v_pinn - ref_am)
        rows.append(dict(
            d=d, sigma_G=geometric_params(d, P["sigma"], P["rho"])[0],
            ref_atm=ref_am[2], pinn_atm=v_pinn[2], lsm_atm=v_lsm[2], lsm_se_atm=se_lsm[2],
            pinn_max_err_sym=err_pinn[:5].max(), pinn_max_err_asym=err_pinn[5:].max(),
            pinn_min_gap=float((v_pinn - np.maximum(P["K"] - G, 0)).min()),
            lsm_max_err_vs_bermudan=np.abs(v_lsm - ref_be[:5]).max(),
            lsm_max_err_vs_american=np.abs(v_lsm - ref_am[:5]).max(),
            pinn_train_seconds=info["time"], lsm_seconds_per_price=t_lsm,
            grid_points_200_per_axis=200.0 ** d * 200,
        ))
        g_line = np.linspace(35, 65, 61)
        curves[d] = dict(G=g_line,
                         ref=np.array([reference_price(g, d, **P) for g in g_line]),
                         pinn=pinn_price(model, info, np.repeat(g_line[:, None], d, axis=1)),
                         lsm_G=G[:5], lsm=v_lsm, lsm_se=se_lsm)
        r = rows[-1]
        print(f"d={d:>2}  ref {r['ref_atm']:.4f}  PINN {r['pinn_atm']:.4f}  LSM {r['lsm_atm']:.4f}  | "
              f"PINN max err sym {r['pinn_max_err_sym']:.4f} asym {r['pinn_max_err_asym']:.4f}  "
              f"LSM max err (Bermudan ref) {r['lsm_max_err_vs_bermudan']:.4f}  | "
              f"PINN {r['pinn_train_seconds'] / 60:.0f} min, LSM {r['lsm_seconds_per_price']:.1f} s/price", flush=True)

    with open(os.path.join(OUT, "extension.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---------------------------------------------------------------- figure: accuracy and cost vs d
    dims = [r["d"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), dpi=150)
    a1.plot(dims, [max(r["pinn_max_err_sym"], r["pinn_max_err_asym"]) for r in rows], "o-", color=C["pinn"],
            lw=2, label="PINN (15 spots, one training)")
    a1.plot(dims, [r["lsm_max_err_vs_bermudan"] for r in rows], "o-", color=C["lsm"], lw=2,
            label="LSM Monte Carlo (5 spots, one run each)")
    a1.set_yscale("log")
    a1.legend(frameon=False, fontsize=9)
    style(a1, "Largest price error vs number of assets", "Number of assets d", "max |error| (log)")
    a2.plot(dims, [r["pinn_train_seconds"] for r in rows], "o-", color=C["pinn"], lw=2, label="PINN: training (all spots)")
    a2.plot(dims, [r["lsm_seconds_per_price"] for r in rows], "o-", color=C["lsm"], lw=2, label="LSM: per price")
    a2.set_yscale("log")
    a2.legend(frameon=False, fontsize=9)
    style(a2, "Runtime vs number of assets", "Number of assets d", "seconds (log)")
    for a in (a1, a2):
        a.set_xticks(dims)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "extension_scaling.png"))
    plt.close(fig)

    # ---------------------------------------------------------------- figure: price along the diagonal, d = 10
    dmax = max(DIMS)
    cv = curves[dmax]
    fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=150)
    ax.plot(cv["G"], np.maximum(P["K"] - cv["G"], 0), color=C["muted"], ls="--", lw=1, label="Payoff")
    ax.plot(cv["G"], cv["ref"], color=C["ref"], lw=2, label="Exact (1-D reduction)")
    ax.plot(cv["G"], cv["pinn"], color=C["pinn"], lw=1.8, label="PINN")
    ax.errorbar(cv["lsm_G"], cv["lsm"], yerr=2 * cv["lsm_se"], fmt="o", color=C["lsm"], ms=5, label="LSM (±2 s.e.)")
    ax.legend(frameon=False, fontsize=9)
    style(ax, f"American put on the geometric average of {dmax} stocks (all stocks equal)",
          "Geometric average G", "Option value")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "extension_d10.png"))
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true")
    main(ap.parse_args().retrain)
