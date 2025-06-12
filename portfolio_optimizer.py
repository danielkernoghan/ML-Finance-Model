import numpy as np
import pandas as pd
from scipy.optimize import minimize

def optimize_portfolio(returns: pd.DataFrame):
    if returns is None or returns.empty or len(returns.columns) < 1:
        print("No return data to optimize on.")
        return {}

    tickers = returns.columns.tolist()
    cov_matrix = returns.cov()
    mean_returns = returns.mean()
    num_assets = len(tickers)

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = num_assets * [1. / num_assets]

    try:
        result = minimize(portfolio_volatility, initial_guess, method='SLSQP',
                          bounds=bounds, constraints=constraints)
        if not result.success:
            raise ValueError("Optimization failed")
        weights = result.x
        return dict(zip(tickers, weights.round(4)))
    except Exception as e:
        print(f"Portfolio optimization failed: {e}")
        return {}
