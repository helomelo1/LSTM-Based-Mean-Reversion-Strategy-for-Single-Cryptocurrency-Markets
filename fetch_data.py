import os
import polars as pl
import yfinance as yf

ticker = "RELIANCE.NS"
stock = yf.Ticker(ticker)

data = stock.history(period="1y")

df = pl.DataFrame({
    'Date': data.index,
    'Open': data['Open'].values,
    'High': data['High'].values,
    'Low': data['Low'].values,
    'Close': data['Close'].values,
    'Volume': data['Volume'].values
})

# print(df.head)

output_dir = "data/"
os.makedirs(output_dir, exist_ok=True)
df.write_csv(f"{output_dir}/{ticker}.csv")

print(f"Data Saved to {output_dir}.")