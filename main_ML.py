import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from financial_data import get_returns
from portfolio_optimizer import optimize_portfolio

tickers = ["XUU.TO", "VCN.TO", "XQQ.TO", "ZCN.TO", "XIU.TO", "CLG.TO"]

# Parameters
lookback_days = 90
future_window = 5
return_threshold = 0.01  # 1%

def fetch_data(ticker):
    df = yf.download(ticker, period=f"{lookback_days+future_window*2}d", interval='1d', auto_adjust=True)
    if df.empty:
        return None
    df = df[['Close']].dropna()
    df['return'] = df['Close'].pct_change()
    df['ma5'] = df['Close'].rolling(5).mean()
    df['ma10'] = df['Close'].rolling(10).mean()
    df['volatility'] = df['return'].rolling(5).std()
    df['momentum'] = df['Close'] / df['Close'].shift(5) - 1

    # Labeling: future return
    df['future_return'] = df['Close'].shift(-future_window) / df['Close'] - 1
    df['label'] = df['future_return'].apply(lambda x: 1 if x > return_threshold else (-1 if x < -return_threshold else 0))
    df.dropna(inplace=True)

    features = ['return', 'ma5', 'ma10', 'volatility', 'momentum']
    return df[features + ['label']]

def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Step 1: Prepare data
all_data = {}
for ticker in tickers:
    df = fetch_data(ticker)
    if df is not None and len(df) > 30:
        all_data[ticker] = df

# Step 2: Train per-ticker models
models = {}
recommendations = {}
for ticker, df in all_data.items():
    X = df.drop(columns='label')
    y = df['label']
    model = train_model(X[:-1], y[:-1])
    models[ticker] = model

    # Step 3: Predict last row
    prediction = model.predict(X.iloc[[-1]])[0]
    label_map = {1: "BUY", 0: "HOLD", -1: "SELL"}
    recommendations[ticker] = label_map[prediction]

# Step 4: Get returns for optimization
buy_tickers = [t for t, rec in recommendations.items() if rec == "BUY"]
if not buy_tickers:
    print("No BUY recommendations. Using all tickers for optimization.")
    buy_tickers = list(all_data.keys())

returns = get_returns(buy_tickers)

# Step 5: Optimize
portfolio_weights = optimize_portfolio(returns)

# Step 6: Output
print("\n--- Machine Learning Recommendations ---")
for t in tickers:
    print(f"{t}: {recommendations.get(t, 'No Data')}")

print("\n--- Optimized Portfolio Allocation ---")
for t, w in portfolio_weights.items():
    print(f"{t}: {w*100:.2f}%")
