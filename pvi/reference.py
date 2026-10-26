"""Reference prices from QuantLib (independent of the solvers in this repository)."""
from __future__ import annotations

import QuantLib as ql


def quantlib_american_put(S0, K=50.0, r=0.05, sigma=0.2, T=1.0, engine="fd", steps=4000):
    """engine = "fd" -> FdBlackScholesVanillaEngine(steps, steps)
       engine = "crr" -> Cox-Ross-Rubinstein binomial tree with `steps` steps."""
    today = ql.Date(1, ql.January, 2024)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    maturity = today + int(round(T * 365))
    option = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Put, K),
                              ql.AmericanExercise(today, maturity))
    process = ql.BlackScholesProcess(
        ql.QuoteHandle(ql.SimpleQuote(S0)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, r, dc)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), sigma, dc)))
    if engine == "crr":
        option.setPricingEngine(ql.BinomialVanillaEngine(process, "crr", steps))
    else:
        option.setPricingEngine(ql.FdBlackScholesVanillaEngine(process, steps, steps))
    return option.NPV()


def quantlib_european_put(S0, K=50.0, r=0.05, sigma=0.2, T=1.0):
    today = ql.Date(1, ql.January, 2024)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    option = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Put, K),
                              ql.EuropeanExercise(today + int(round(T * 365))))
    process = ql.BlackScholesProcess(
        ql.QuoteHandle(ql.SimpleQuote(S0)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, r, dc)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), sigma, dc)))
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return option.NPV()
