import torch
import numpy as np
import polars as pl
import pandas as pd
import yfinance as yf
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


def download_data(ticker: str, start="", end="", interval="") -> pl.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False, auto_adjust=False)
    df.dropna(inplace=True)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    
    df = df.reset_index()
    
    pl_df = pl.from_pandas(df)
    
    if 'Adj Close' in pl_df.columns:
        pl_df = pl_df.with_columns(pl.col("Adj Close").alias("AdjClose")).drop("Adj Close")
    elif 'Close' in pl_df.columns and 'AdjClose' not in pl_df.columns:
        pl_df = pl_df.with_columns(pl.col("Close").alias("AdjClose"))
    
    return pl_df


def add_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns([
        (pl.col("Close") / pl.col("Close").shift(1)).log().alias("LogRet")
    ])

    df = df.with_columns([
        pl.col("LogRet").rolling_std(window_size=5).alias("Volatility"),
        ((pl.col("Volume") - pl.col("Volume").rolling_mean(window_size=20)) /
         pl.col("Volume").rolling_std(window_size=20)).alias("VolumeZ")
    ])

    return df.drop_nulls()


def prepare_features(df: pl.DataFrame):
    df = df.with_columns([
        pl.col("LogRet").shift(-1).alias("Target")
    ]).drop_nulls()

    features = ["Open", "High", "Low", "Close", "Volume", "LogRet", "Volatility", "VolumeZ"]

    X = df.select(features).to_numpy()
    y = df.select("Target").to_numpy()

    return X, y, df.select("Target").height


class TS_Dataset(Dataset):
    def __init__(self, X, y, seq_len=60):
        self.seq_len = seq_len
        self.X = []
        self.y = []

        for i in range(seq_len, len(X)):
            self.X.append(X[i - seq_len:i])
            self.y.append(y[i])

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)


    def __len__(self):
        return len(self.X)
    

    def __getitem__(self, index):
        return self.X[index], self.y[index]
    

def load_dataset(ticker: str, seq_len=60, train_split=0.8, start="", end="", interval=""):
    df = download_data(ticker, start=start, end=end, interval=interval)
    df = add_features(df)
    X, y, _ = prepare_features(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    split_idx = int(len(X_scaled) * train_split)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    train_ds = TS_Dataset(X_train, y_train, seq_len=seq_len)
    test_ds = TS_Dataset(X_test, y_test, seq_len=seq_len)

    return train_ds, test_ds, scaler, df