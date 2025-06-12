from news_sentiment import get_sentiment
from financial_data import get_returns
from portfolio_optimizer import optimize_portfolio

tickers = ["XUU.TO", "VCN.TO", "XQQ.TO", "ZCN.TO", "XIU.TO", "CLG.TO", "CCOM.TO"]

# 1. Get sentiment scores and classify recommendations
sentiments = {ticker: get_sentiment(ticker) for ticker in tickers}
recommendations = {}

for ticker, score in sentiments.items():
    if score > 0.2:
        recommendations[ticker] = "BUY"
    elif score < -0.2:
        recommendations[ticker] = "SELL"
    else:
        recommendations[ticker] = "HOLD"

# 2. Only optimize over BUY tickers
buy_tickers = [t for t in tickers if recommendations[t] == "BUY"]

# 3. Get historical returns
returns = get_returns(buy_tickers)

# 4. Optionally adjust returns based on sentiment (comment out if not using)
for ticker in returns.columns:
    returns[ticker] *= (1 + sentiments[ticker])  

# 5. Optimize portfolio
portfolio_weights = optimize_portfolio(returns)

# 6. Final output
print("\n--- Sentiment Recommendations ---")
for t in tickers:
    print(f"{t}: {sentiments[t]:.2f} → {recommendations[t]}")

print("\n--- Optimized Portfolio Allocation ---")
for t, w in portfolio_weights.items():
    print(f"{t}: {w*100:.2f}%")
