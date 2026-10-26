"""
American put as a parabolic variational inequality (thesis, Chapters 3-5): penalty
finite differences and projected SOR (Chapter 4) and a physics-informed neural network
(Chapter 5), checked against QuantLib.

    python benchmark.py               # everything; uses the trained PINNs in results/ if present
    python benchmark.py --retrain     # also retrain the three PINNs (~13 min each on a laptop CPU)
    python benchmark.py --skip-pinn   # grid methods only (about a minute)

Parameters (those of the original June 2025 results chapter): K = 50, r = 5%,
sigma = 20%, T = 1, S_max = 100. Writes results/*.csv and results/*.png.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pvi.finite_difference import penalty_fd, price_at, projected_sor
from pvi.reference import quantlib_american_put, quantlib_european_put

P = dict(K=50.0, r=0.05, sigma=0.2, T=1.0)
S_MAX = 100.0
SPOTS = [30, 40, 45, 50, 55, 60, 70]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

C = {"ref": "#1f1f1e", "pen": "#2a78d6", "psor": "#eb6834", "pinn": "#1baf7a",
     "pinn_original": "#e87ba4", "muted": "#6b6a63", "grid": "#e4e3dd"}


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel(xlabel, color=C["muted"])
    ax.set_ylabel(ylabel, color=C["muted"])
    ax.grid(True, color=C["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def write_csv(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main(skip_pinn, retrain):
    os.makedirs(OUT, exist_ok=True)

    # ------------------------------------------------------------------ references
    ql_fd = {S: quantlib_american_put(S, **P, engine="fd", steps=4000) for S in SPOTS}
    ql_crr = {S: quantlib_american_put(S, **P, engine="crr", steps=200) for S in SPOTS}
    euro = {S: quantlib_european_put(S, **P) for S in SPOTS}
    fine = projected_sor(**P, S_max=S_MAX, M=4000, N=2000)          # reference curve and boundary
    print("Reference: QuantLib FD 4000x4000 (prices), PSOR M=4000 N=2000 (curves, boundary)")
    print(f"  fine-grid PSOR vs QuantLib: max |diff| = "
          f"{max(abs(price_at(fine, S) - ql_fd[S]) for S in SPOTS):.1e}")

    # ------------------------------------------------------------------ grid methods
    conv_rows = []
    for M in (100, 200, 400, 800, 1600):
        for name, solver in (("penalty", penalty_fd), ("psor", projected_sor)):
            res = solver(**P, S_max=S_MAX, M=M, N=M)
            err = max(abs(price_at(res, S) - ql_fd[S]) for S in SPOTS)
            conv_rows.append([name, M, M, err, res["time"], res["iterations"], res["boundary"][-1]])
    write_csv("grid_convergence.csv",
              ["method", "M", "N", "max_abs_error", "seconds", "iterations", "boundary_T"], conv_rows)

    pen = penalty_fd(**P, S_max=S_MAX, M=800, N=800)
    psor = projected_sor(**P, S_max=S_MAX, M=800, N=800)

    # ------------------------------------------------------------------ PINNs
    pinns = {}
    if not skip_pinn:
        import torch
        from pvi.pinn import REWEIGHTED, evaluate, exercise_boundary, load_pinn, train_pinn
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        variants = {"vi": dict(loss="vi"), "original": dict(loss="original"),
                    "original_reweighted": dict(loss="original", weights=REWEIGHTED)}
        for loss, kw in variants.items():
            weights = os.path.join(OUT, f"pinn_{loss}.pt")
            if retrain or not os.path.exists(weights):
                print(f"Training PINN ({loss} loss) ...", flush=True)
                model, info = train_pinn(**P, S_max=S_MAX, iterations=20_000, **kw)
                torch.save(model.state_dict(), weights)
                with open(weights + ".time", "w") as f:
                    f.write(str(info["time"]))
            else:
                print(f"Loading trained PINN ({loss} loss) from {weights}", flush=True)
                model, info = load_pinn(weights, K=P["K"], T=P["T"], S_max=S_MAX)
                if os.path.exists(weights + ".time"):
                    info["time"] = float(open(weights + ".time").read())
            q_grid = np.linspace(0, P["T"], 41)
            s_grid = np.linspace(0.05, S_MAX, 4000)
            pinns[loss] = {
                "prices": {S: float(evaluate(model, info, [S])[0]) for S in SPOTS},
                "curve": evaluate(model, info, fine["S"]),
                "q": q_grid,
                "boundary": exercise_boundary(model, info, P["T"] - q_grid, s_grid),
                "time": info["time"],
            }

    # ------------------------------------------------------------------ price table
    cols = [("QuantLib FD", ql_fd), ("QuantLib CRR 200", ql_crr), ("European", euro),
            ("Penalty FD", {S: price_at(pen, S) for S in SPOTS}),
            ("PSOR", {S: price_at(psor, S) for S in SPOTS})]
    for loss, label in (("vi", "PINN"), ("original", "PINN (original loss)"),
                        ("original_reweighted", "PINN (original loss, reweighted)")):
        if loss in pinns:
            cols.append((label, pinns[loss]["prices"]))
    write_csv("prices.csv", ["S0"] + [c[0] for c in cols], [[S] + [c[1][S] for c in cols] for S in SPOTS])

    print(f"\n{'S0':>4}" + "".join(f"{c[0]:>20}" for c in cols))
    for S in SPOTS:
        print(f"{S:>4}" + "".join(f"{c[1][S]:>20.4f}" for c in cols))
    print("\nmax |error| vs QuantLib FD over S0 in", SPOTS)
    for name, d in cols[1:]:
        if name == "European":
            continue
        print(f"  {name:<22}{max(abs(d[S] - ql_fd[S]) for S in SPOTS):.4f}")
    print(f"\nexercise boundary at t = 0 (one year to expiry): reference {fine['boundary'][-1]:.2f}, "
          f"penalty {pen['boundary'][-1]:.2f}, PSOR {psor['boundary'][-1]:.2f}"
          + "".join(f", PINN ({k}) {v['boundary'][-1]:.2f}" for k, v in pinns.items()))
    print(f"runtime: penalty {pen['time']:.2f}s, PSOR {psor['time']:.2f}s"
          + "".join(f", PINN ({k}) {v['time']:.0f}s" for k, v in pinns.items()))

    # ------------------------------------------------------------------ figure 1: values and errors
    S = fine["S"]
    ref = fine["V"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150)
    a1.plot(S, np.maximum(P["K"] - S, 0), color=C["muted"], lw=1, ls="--", label="Payoff")
    a1.plot(S, ref, color=C["ref"], lw=2, label="Reference")
    a1.plot(S, np.interp(S, pen["S"], pen["V"]), color=C["pen"], lw=1.5, ls=(0, (4, 3)), label="Penalty FD")
    for k, lab in (("vi", "PINN"), ("original", "PINN, original loss")):
        if k in pinns:
            a1.plot(S, pinns[k]["curve"], color=C["pinn" if k == "vi" else "pinn_original"], lw=1.8, label=lab)
    a1.set_xlim(20, 80)
    a1.set_ylim(-0.5, 30)
    style(a1, "American put value one year before expiry", "Stock price S", "Option value")
    a1.legend(frameon=False, fontsize=9)

    floor = 1e-6   # errors below this are drawn at the floor
    for name, V, col in (("Penalty FD", np.interp(S, pen["S"], pen["V"]), C["pen"]),
                         ("PSOR", np.interp(S, psor["S"], psor["V"]), C["psor"])):
        a2.plot(S, np.maximum(np.abs(V - ref), floor), color=col, lw=1.5, label=name)
    for k, lab in (("vi", "PINN"), ("original", "PINN, original loss")):
        if k in pinns:
            a2.plot(S, np.maximum(np.abs(pinns[k]["curve"] - ref), floor),
                    color=C["pinn" if k == "vi" else "pinn_original"], lw=1.5, label=lab)
    a2.set_yscale("log")
    a2.set_xlim(0, S_MAX)
    style(a2, "Absolute error against the reference", "Stock price S", "|error| (log scale)")
    a2.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "option_value.png"))
    plt.close(fig)

    # ------------------------------------------------------------------ figure 2: exercise boundary
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
    ax.plot(fine["q"], fine["boundary"], color=C["ref"], lw=2, label="Reference")
    ax.plot(pen["q"], pen["boundary"], color=C["pen"], lw=1.5, ls=(0, (4, 3)), label="Penalty FD")
    for k, lab in (("vi", "PINN"), ("original", "PINN, original loss")):
        if k in pinns:
            ax.plot(pinns[k]["q"], pinns[k]["boundary"], color=C["pinn" if k == "vi" else "pinn_original"],
                    lw=1.8, marker="o", markersize=3, label=lab)
    style(ax, "Early-exercise boundary S*", "Time to expiry (years)", "Critical stock price S*")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "exercise_boundary.png"))
    plt.close(fig)
    write_csv("exercise_boundary_reference.csv", ["time_to_expiry", "S_star"],
              zip(fine["q"][::50], fine["boundary"][::50]))

    # ------------------------------------------------------------------ figure 3: convergence
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), dpi=150)
    for name, col, lab in (("penalty", C["pen"], "Penalty FD"), ("psor", C["psor"], "PSOR")):
        rows = [r for r in conv_rows if r[0] == name]
        a1.plot([r[1] for r in rows], [r[3] for r in rows], "o-", color=col, lw=2, label=lab)
        a2.plot([r[1] for r in rows], [r[4] for r in rows], "o-", color=col, lw=2, label=lab)
    for a in (a1, a2):
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xticks([100, 200, 400, 800, 1600])
        a.set_xticklabels(["100", "200", "400", "800", "1600"])
        a.minorticks_off()
        a.legend(frameon=False, fontsize=9)
    style(a1, "Max price error vs grid size (M = N)", "Grid points M", "max |error|")
    style(a2, "Runtime vs grid size", "Grid points M", "seconds")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "grid_convergence.png"))
    plt.close(fig)
    print(f"\nwrote figures and tables to {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-pinn", action="store_true", help="grid methods only")
    ap.add_argument("--retrain", action="store_true",
                    help="train the PINNs even if saved weights exist in results/")
    a = ap.parse_args()
    main(a.skip_pinn, a.retrain)
