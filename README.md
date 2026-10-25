# American Options as Variational Inequalities

Pricing the American put as a **parabolic variational inequality** with three methods: a penalty method with finite differences, a projection method (projected SOR) and a physics-informed neural network (PINN). Each is checked against independent reference prices. An **extension** then tests the PINN on the same problem in up to 10 dimensions, where grid methods are no longer possible.

This repository is the computational part of my BSc thesis at the Athens University of Economics and Business (Department of Statistics, 2025), *Pricing the Risk of Decision Using Variational Inequalities: An American Type Derivatives Example*, supervised by Professor A. N. Yannacopoulos. The theory is in the thesis, and this repository replaces its numerical results chapter.

📄 **Thesis (Chapters 1–5, revised):** [`thesis/Dallas_BSc_thesis_variational_inequalities.pdf`](thesis/Dallas_BSc_thesis_variational_inequalities.pdf)

<p align="center">
  <img src="results/option_value.png" width="900" alt="American put value and pricing error of the three methods">
</p>

## Where things are

| Thesis | Topic | In this repository |
|---|---|---|
| Ch. 1 | Elliptic VIs: Lions–Stampacchia, penalization, regularity, free boundary | (theory) |
| Ch. 2 | Parabolic VIs, Bochner spaces, viscosity solutions | (theory) |
| Ch. 3 | The American put as a parabolic VI | the problem solved below |
| Ch. 4 | Penalization + finite differences; Minty, projection, PSOR | [`pvi/finite_difference.py`](pvi/finite_difference.py) |
| Ch. 5 | Physics-informed neural networks for the PVI | [`pvi/pinn.py`](pvi/pinn.py) |
| (Results) | Comparison against reference prices | [`benchmark.py`](benchmark.py), [`results/`](results/) |
| (Extension) | The PINN in 2–10 dimensions vs Monte Carlo | [`extension.py`](extension.py), [`pvi/basket.py`](pvi/basket.py), [`results/extension/`](results/extension/) |

## The problem

An American put with strike $K$ can be exercised at any time before maturity $T$. Its value $V(q, S)$, as a function of time to expiry $q = T - t$ and stock price $S$, solves the **parabolic variational inequality** (thesis, eq. 3.14):

```math
\min\Big\{ \partial_q V + A V,\; V - \psi \Big\} = 0,
\qquad
A V = -\tfrac12\sigma^2 S^2\, \partial_{SS} V - r S\, \partial_S V + r V,
\qquad
\psi(S) = (K - S)^+ ,
```

with $V(0, S) = \psi(S)$, $V(q, 0) = K$ and $V(q, S_{\max}) = 0$.

At every point, either the Black–Scholes equation holds (continuation region: keep the option) or $V = \psi$ (exercise region: exercise now). The two regions are separated by the unknown **early-exercise boundary** $S^\ast(q)$. Because the value function is only $C^1$, not $C^2$, across that boundary, it is not a classical solution of any single PDE (Section 3.2.3), which is why the problem is posed as a variational inequality.

**Parameters** (as in the thesis): $K = 50$, $r = 5\%$, $\sigma = 20\%$, $T = 1$ year, $S_{\max} = 100$.

## Methods

| Method | How the constraint $V \ge \psi$ is enforced | Implementation |
|---|---|---|
| **Penalty FD** | Adds $\tfrac{1}{\varepsilon}\min(V - \psi, 0)$ to the PDE ($\varepsilon = 10^{-6}$). The penalty is treated implicitly and solved by Newton/policy iteration each step. | Crank–Nicolson with Rannacher start-up, tridiagonal solves |
| **Projected SOR** | Projection $P_K(z) = \max(z, \psi)$ applied node by node inside an SOR sweep (Cryer, 1971). This is the efficient form of the fixed point $V = P_K(V - \alpha\,\cdot\text{residual})$ from Minty's reformulation. | Same discretisation; Young's optimal $\omega$ |
| **PINN** | The loss is the residual of the VI itself, $\lVert\min(-\mathcal{B}[u],\, u - \psi)\rVert^2$, plus terminal and boundary terms. | 4 × 64 tanh network, output layer enforcing $u(0,t)=K$, Adam, 20 000 iterations |
| **Reference** | QuantLib's finite-difference engine on a 4000 × 4000 grid. Independent of this code, and agrees with a fine-grid PSOR solution to $10^{-4}$. | `pvi/reference.py` |

## Results

Option values one year before expiry (`python benchmark.py`):

| $S_0$ | Reference | European | Penalty FD | PSOR | PINN (VI residual) | PINN, thesis loss, thesis weights | PINN, thesis loss, reweighted |
|---|---|---|---|---|---|---|---|
| 30 | **20.0000** | 17.5887 | 20.0000 | 20.0000 | 19.9943 | 19.2997 | 19.8937 |
| 40 | **10.0000** | 8.4912 | 10.0000 | 10.0000 | 9.9706 | 9.5624 | 9.9436 |
| 45 | **5.7462** | 5.1071 | 5.7462 | 5.7462 | 5.7432 | 5.6684 | 5.8516 |
| 50 | **3.0451** | 2.7868 | 3.0450 | 3.0450 | 3.0593 | 3.0499 | 3.1386 |
| 55 | **1.4932** | 1.3929 | 1.4931 | 1.4931 | 1.5138 | 1.5053 | 1.5674 |
| 60 | **0.6835** | 0.6460 | 0.6835 | 0.6835 | 0.7064 | 0.6992 | 0.7260 |
| 70 | **0.1216** | 0.1168 | 0.1216 | 0.1216 | 0.1410 | 0.1273 | 0.1325 |
| **max error** | | | **0.0001** | **0.0001** | **0.029** | **0.70** | **0.11** |
| **runtime** | | | **0.08 s** | **0.28 s** | **13 min** | **13 min** | **12 min** |

Grid methods use $M = N = 800$. Runtimes are on a laptop-class CPU. The exercise boundary one year before expiry is $S^\ast \approx 40.4$.

### 1. The two grid methods are accurate and fast

Penalty FD and PSOR agree with the reference to $10^{-4}$ at every price. Their error falls about fourfold each time the grid is doubled, until it reaches the accuracy of the reference itself. The penalty method is the faster of the two on fine grids (0.09 s against 0.34 s at $M = 800$), because each Newton step is a direct tridiagonal solve, while PSOR needs more sweeps as the grid is refined. Both trace the exercise boundary exactly.

<p align="center">
  <img src="results/grid_convergence.png" width="820" alt="Error and runtime against grid size">
</p>

### 2. The PINN works, and the design of its loss is what matters

The thesis builds the variational inequality into the PINN's loss: a PDE term, an inequality term $\lVert(\psi - u)^+\rVert^2$ and a complementarity term $\lVert(u - \psi)\,\mathcal{B}[u]\rVert^2$, together with a boundary-conditioned output layer, adaptive sampling, and terminal and boundary losses. All of that structure is kept here. The three PINN columns differ only in how the interior is penalised:

* **Thesis loss, thesis weights (1 / 10 / 30).** The PDE term $\lVert\mathcal{B}[u]\rVert^2$ asks the Black–Scholes equation to hold everywhere, including the exercise region, where the true solution has $\mathcal{B}[\psi] = -rK$. With these weights it pulls the price *below the payoff* deep in the money (19.30 at $S = 30$, where exercising immediately gives 20).
* **Thesis loss, reweighted (0.7 / 117 / 97).** These are the weights from the author's perpetual-put experiment, which validated the same loss against the closed-form perpetual solution. Letting the inequality and complementarity terms dominate cuts the error about sevenfold, to 0.11, though the price still dips slightly below the payoff deep in the money.
* **VI residual.** Writing the thesis's own equation (3.14), $\min\{-\mathcal{B}[u], u - \psi\} = 0$, directly as the loss, $\lVert\min(-\mathcal{B}[u],\, u - \psi)\rVert^2$, removes the dependence on hand-tuned weights and brings the error down to 0.03.

In one dimension, however, the PINN does not compete with the grid methods or QuantLib: it is thousands of times slower (13 minutes against a fraction of a second) and much less accurate. Its advantage, being mesh-free, only matters in higher dimensions, which is what the extension below tests.

### 3. The exercise boundary is much harder for the PINN than the price

<p align="center">
  <img src="results/exercise_boundary.png" width="640" alt="Early-exercise boundary">
</p>

The grid methods recover $S^\ast(q)$ exactly. The PINN's boundary is about 2 units too high, even though its prices are accurate to 0.03. This is not a bug: near the boundary the option value differs from the payoff only quadratically, $V - \psi \approx \tfrac12\Gamma\,(S - S^\ast)^2$, so a tiny error in value moves the apparent boundary a long way. The network's boundary is read off the VI itself, as the set where $u - \psi < -\mathcal{B}[u]$, with no tolerance to tune.

## Extension: the PINN in higher dimensions

In one dimension the grid methods are unbeatable. The argument for a mesh-free method is that grids stop working when there are several underlying assets: with 200 points per asset and 200 time steps, a grid needs

| assets $d$ | 1 | 2 | 3 | 5 | 10 |
|---|---|---|---|---|---|
| grid points | $4\times10^{4}$ | $8\times10^{6}$ | $1.6\times10^{9}$ | $6.4\times10^{13}$ | $2\times10^{25}$ |

This extension tests whether the PINN really keeps working as $d$ grows.

**Test problem.** An American put on the **geometric average** of $d$ stocks, $G = (S_1 \cdots S_d)^{1/d}$, with payoff $(K - G)^+$. Every stock has volatility 20% and each pair has correlation $\rho = 0.25$; the other parameters are as above. The geometric average of correlated GBMs is itself a GBM, with

```math
\sigma_G^2 = \sigma^2\,\frac{1 + (d-1)\rho}{d}, \qquad q_G = \tfrac12\left(\sigma^2 - \sigma_G^2\right),
```

so the $d$-dimensional problem has an **exact one-dimensional equivalent**: an American put on $G$ with volatility $\sigma_G$ and dividend yield $q_G$. QuantLib prices that to $10^{-4}$, which gives a true reference price in every dimension. Neither solver uses this shortcut; both work in the full $d$-dimensional space.

**Methods.**
* **PINN.** The same VI-residual loss in log-prices $x_i = \ln(S_i/K)$. The $d$-dimensional Black–Scholes operator is computed exactly through $d$ directional second derivatives. Collocation points (2048 per batch) are drawn around the risk-neutral distribution of the stocks. The network has 4 × 128 tanh units and is trained for 20 000 iterations. One training prices every spot.
* **Longstaff–Schwartz Monte Carlo (LSM).** 50 exercise dates, 100 000 training and 100 000 independent pricing paths, and a quadratic regression basis plus the payoff. It is the standard high-dimensional method. LSM prices a Bermudan option, so its error is measured against the Bermudan reference with the same dates.

**Results** (`python extension.py`). The PINN is tested on 15 spots (5 with all stocks equal, 10 with the stocks spread apart); LSM is run separately at the 5 symmetric spots.

| $d$ | Exact, $G_0 = 50$ | PINN | LSM (± s.e.) | PINN max error | LSM max error | PINN training | LSM per price |
|---|---|---|---|---|---|---|---|
| 1 | 3.0450 | 3.0790 | 3.0392 ± 0.011 | 0.092 | 0.010 | 13 min | 0.9 s |
| 2 | 2.3621 | 2.3944 | 2.3375 ± 0.009 | 0.074 | 0.024 | 19 min | 1.9 s |
| 3 | 2.0844 | 2.1234 | 2.0558 ± 0.008 | 0.051 | 0.024 | 25 min | 2.5 s |
| 5 | 1.8338 | 1.8640 | 1.8205 ± 0.007 | 0.073 | 0.009 | 37 min | 4.6 s |
| 10 | 1.6230 | 1.6468 | 1.6024 ± 0.006 | 0.053 | 0.017 | 68 min | 16.8 s |

<p align="center">
  <img src="results/extension/extension_scaling.png" width="900" alt="Error and runtime against the number of assets">
</p>
<p align="center">
  <img src="results/extension/extension_d10.png" width="640" alt="PINN, LSM and exact price for 10 assets">
</p>

**What the extension shows:**

* **The PINN solves the 10-dimensional variational inequality, and its accuracy does not degrade with dimension.** The largest error stays between 0.05 and 0.09 (about 3–4% of the at-the-money price) from $d = 1$ to $d = 10$, on a problem where a grid would need $10^{25}$ points.
* **Its cost grows roughly linearly, not exponentially.** Training time rises from 13 to 68 minutes, mainly because the second derivatives need one extra backward pass per asset.
* **LSM is still more accurate and much cheaper per price**, with errors of about 0.01–0.025 in a few seconds per price. The PINN's advantage is that one training gives the price at *every* spot and time, plus the Greeks by automatic differentiation. At $d = 10$, one PINN training costs about as much as 240 LSM prices, so it pays off when many prices or sensitivities are needed, such as for hedging or risk reports.
* **A lesson from the thesis showed up in practice.** Building the payoff into the network output, $u = \psi + (1 - t/T) f_\theta$, looks like an obvious shortcut, but the network then learns $u = \psi$ with zero loss. The payoff's kink, where the equation fails, has measure zero and is never sampled. This is the non-smoothness that Chapter 3 discusses. The network has to stay smooth, with the payoff imposed through the loss.

**Limitations.**
* The geometric basket is a convenient benchmark *because* it has a one-dimensional structure; arithmetic baskets or max-options are harder for every method.
* At $d = 1$ this PINN (0.09) is less accurate than the one in the main results (0.03), which uses the thesis's boundary-conditioned output layer on a bounded domain; that output layer has no direct $d$-dimensional analogue.
* Each result comes from a single training run on a CPU; a GPU would shorten training considerably.

## About the thesis PDF

The PDF contains Chapters 1–5 and Appendix A.1 of the thesis, revised in 2026. While building this repository I checked the thesis against the working code and corrected the errors I found. The main ones:

* **Sign of the Black–Scholes operator (Ch. 3–4).** It was written as $A V = +\tfrac12\sigma^2S^2V_{SS} + \dots$ inside $\partial_q V + AV \ge 0$, which makes the equation run backwards in time; in the submitted version, eq. (3.20) contradicted eq. (3.12). The operator now carries the sign convention of Chapters 1–2, and the complementarity form (3.9) is fixed accordingly.
* **The PINN loss (Ch. 5)** was replaced by the VI residual, as explained above.
* **Numerical schemes (Ch. 4).** The penalty is now treated implicitly; explicit treatment is unstable for small $\varepsilon$. A description of projected SOR was added.
* **Theory (Ch. 1–2, App. A.1):**
  * corrected statement of the Aubin–Lions lemma;
  * a correct hemicontinuous-but-discontinuous example;
  * a well-defined coercivity condition and consistent penalty function $\beta(s) = \min(s,0)$;
  * the missing uniform penalty bound in the $H^2$-regularity proof;
  * a complete proof of the Lions–Stampacchia theorem (Galerkin approximation and Minty's lemma instead of an infinite-dimensional Brouwer argument).
* **Cross-references:** all "??" and stale references were fixed.

The original Chapter 6 (Results) is not included; this repository replaces it. Its numbers were wrong for specific, fixable reasons, found by re-running the original notebooks:

* **QuantLib was set up correctly, but its curves were mislabelled.** The plotting loop passed time as the option's *maturity*, so the curve labelled $t = 0$ was an option expiring immediately and the one labelled $t = 1$ was today's price. The exercise boundary read off those curves (35–37) should have been 40.4, and the at-the-money price is 3.045.
* **The projection code was one line from correct.** The interest rate entered both halves of the Crank–Nicolson step with full weight, discounting twice. Fixing that line moves the price from 2.96 to 3.043.
* **The penalty code had the operator's sign error and a misaligned tridiagonal matrix.** Its final `max(V, payoff)` step meant it was really a projection method; the result was identical for every penalty parameter. With both fixed it also gives 3.044.
* **The comparison "MAE vs binomial" used a hand-written tree that does not price the option,** so those numbers carried no information.

## Running

```bash
pip install -r requirements.txt
python benchmark.py               # tables and figures (uses the trained PINNs in results/)
python benchmark.py --retrain     # also retrain both PINNs (~13 min each on a CPU)
python benchmark.py --skip-pinn   # grid methods only
python extension.py               # extension, d = 1..10 (uses trained networks in results/extension/)
python extension.py --retrain     # retrain them (about 3 hours on a 2-core CPU)
pytest -q                         # tests
```

```python
from pvi.finite_difference import penalty_fd, price_at
res = penalty_fd(K=50, r=0.05, sigma=0.2, T=1.0, S_max=100, M=800, N=800)
price_at(res, 50)        # 3.0450
res["boundary"][-1]      # exercise boundary one year before expiry, ~40.4
```

```
pvi/finite_difference.py   penalty method and projected SOR (Crank–Nicolson, Rannacher start-up)
pvi/pinn.py                physics-informed neural network (thesis loss, reweighted thesis loss, VI residual)
pvi/reference.py           QuantLib reference prices
pvi/basket.py              extension: d-dimensional PINN, LSM Monte Carlo, exact 1-D reference
benchmark.py               main results, writes results/
extension.py               extension, writes results/extension/
results/                   figures, CSV tables, trained network weights
thesis/                    thesis PDF (Chapters 1–5, revised)
tests/                     pytest suite
```

## References

* Yannacopoulos, A. N., & Tsekrekos, A. E. (2020). *Variational Inequalities and Applications in Finance*. Springer.
* Kinderlehrer, D., & Stampacchia, G. (1980). *An Introduction to Variational Inequalities and Their Applications*. Academic Press.
* Cryer, C. W. (1971). The solution of a quadratic programming problem using systematic overrelaxation. *SIAM Journal on Control*, 9(3), 385–392.
* Forsyth, P. A., & Vetzal, K. R. (2002). Quadratic convergence for valuing American options using a penalty method. *SIAM Journal on Scientific Computing*, 23(6), 2095–2122.
* Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks. *Journal of Computational Physics*, 378, 686–707.
* Longstaff, F. A., & Schwartz, E. S. (2001). Valuing American options by simulation: a simple least-squares approach. *The Review of Financial Studies*, 14(1), 113–147.
* Sirignano, J., & Spiliopoulos, K. (2018). DGM: A deep learning algorithm for solving partial differential equations. *Journal of Computational Physics*, 375, 1339–1364.

## License

Code: MIT, see [LICENSE](LICENSE). The thesis PDF remains © the author.
