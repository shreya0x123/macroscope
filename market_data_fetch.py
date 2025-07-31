import yfinance as yf
import pandas as pd
from datetime import datetime

ticker = "^NSEI"  # or "^GSPC" for S&P 500
today = datetime.today().strftime('%Y-%m-%d')

data = yf.download(ticker, start="2023-01-01", end=today, auto_adjust=False)

if "Adj Close" not in data.columns:
    print("❌ 'Adj Close' not found. Check auto_adjust setting.")
    exit()

data["daily_return"] = data["Adj Close"].pct_change()
data["next_day_return"] = data["daily_return"].shift(-1)
data["target"] = data["next_day_return"].apply(lambda x: 1 if x > 0 else 0)

df = data.reset_index()[["Date", "daily_return", "next_day_return", "target"]]
df.to_csv("market_returns.csv", index=False)
print("✅ Market data saved to market_returns.csv")
