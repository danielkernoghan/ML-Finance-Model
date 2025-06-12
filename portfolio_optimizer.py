import cvxpy as cp
import numpy as np

def optimize_portfolio(returns, max_weight=0.4, risk_aversion=0.5):
    mu = returns.mean().values
    Sigma = returns.cov().values
    n = len(mu)

    w = cp.Variable(n)
    risk = cp.quad_form(w, Sigma)
    expected_return = mu @ w

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= max_weight
    ]

    prob = cp.Problem(cp.Maximize(expected_return - risk_aversion * risk), constraints)
    prob.solve()

    weights = w.value
    tickers = returns.columns
    return dict(zip(tickers, weights.round(4)))
