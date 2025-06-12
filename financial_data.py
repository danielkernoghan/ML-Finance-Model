import yfinance as yf

def get_returns(tickers, period='6mo', interval='1d'):
    data = yf.download(tickers, period=period, interval=interval, auto_adjust=True)['Close']
    returns = data.pct_change().dropna()
    return returns
