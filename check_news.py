import yfinance as yf
ticker = "GOTO.JK"
t = yf.Ticker(ticker)
print(f"News for {ticker}:")
print(t.news)
