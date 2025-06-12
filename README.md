# Financial Portfolio Advisor

This project is an intelligent tool for evaluating stocks, ETFs, and commodities using a combination of historical data and live financial data. It outputs actionable **buy/hold/sell** signals and portfolio optimization recommendations.

Version 1 also holds the original version of the model which aimed to include news sentiment. Workign on sourcing a reliable news API.

---

## Features

- **ML Model**: Uses historical financial indicators and sentiment analysis to generate recommendations.
- **Portfolio Optimizer**: Suggests allocations to maximize expected return or Sharpe ratio.

---

## Project Structure

| File/Folder            | Description                                  |
|------------------------|----------------------------------------------|
| `financial_data.py`    | Pulls and processes market datav             |
| `main_ML.py`           | Trains and applies ML model for predictions  |
| `portfolio_optimizer.py`| Solves allocation problem using optimization |
| `Version_1/`           | Archive of earlier experiments (optional)    |

---
