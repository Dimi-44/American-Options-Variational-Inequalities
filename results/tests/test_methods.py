import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from pvi.finite_difference import penalty_fd, price_at, projected_sor  # noqa: E402

P = dict(K=50.0, r=0.05, sigma=0.2, T=1.0, S_max=100.0)
# QuantLib FdBlackScholesVanillaEngine(4000, 4000)
REFERENCE = {30: 20.0, 45: 5.7462, 50: 3.0451, 60: 0.6835, 70: 0.1216}


@pytest.mark.parametrize("solver", [penalty_fd, projected_sor])
def test_grid_methods_match_reference(solver):
    res = solver(**P, M=400, N=400)
    for S, ref in REFERENCE.items():
        assert price_at(res, S) == pytest.approx(ref, abs=1e-3)


@pytest.mark.parametrize("solver", [penalty_fd, projected_sor])
def test_constraint_and_boundary(solver):
    res = solver(**P, M=400, N=400)
    assert res["min_violation"] > -1e-5                 # V >= payoff (penalty: up to O(eps))
    assert res["boundary"][-1] == pytest.approx(40.45, abs=0.3)


def test_american_above_european_deep_in_the_money():
    res = penalty_fd(**P, M=200, N=200)
    assert price_at(res, 30) == pytest.approx(20.0, abs=1e-5)    # immediate exercise (penalty: O(eps))
    assert price_at(res, 30) > 17.58                             # European put is 17.59


def test_error_decreases_with_grid():
    errs = [abs(price_at(penalty_fd(**P, M=m, N=m), 50) - REFERENCE[50]) for m in (100, 200, 400)]
    assert errs[0] > errs[1] > errs[2]


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "results", "pinn_vi.pt")),
                    reason="trained PINN weights not present")
def test_trained_pinn():
    from pvi.pinn import evaluate, load_pinn
    model, info = load_pinn(os.path.join(ROOT, "results", "pinn_vi.pt"))
    values = evaluate(model, info, list(REFERENCE))
    assert np.max(np.abs(values - np.array(list(REFERENCE.values())))) < 0.05


def test_basket_reference_reduces_to_one_dimension():
    from pvi.basket import geometric_params, reference_price
    assert geometric_params(1, 0.2, 0.25) == pytest.approx((0.2, 0.0))
    assert reference_price(50.0, 1, grid=800) == pytest.approx(REFERENCE[50], abs=2e-3)


def test_lsm_basket_close_to_reference():
    from pvi.basket import lsm_basket, reference_price
    price, se, _ = lsm_basket([50.0, 50.0], 2, n_train=20_000, n_test=20_000, n_dates=20)
    ref = reference_price(50.0, 2, exercise="bermudan", n_dates=20, grid=800)
    assert abs(price - ref) < 4 * se + 0.03
