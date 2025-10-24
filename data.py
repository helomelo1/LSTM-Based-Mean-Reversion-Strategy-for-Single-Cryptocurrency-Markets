import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def fetch_data(ticker="ETH-USD", interval="1h", period="720d"):
    df = yf.download(ticker, interval=interval, period=period)
    return df


def compute_features(df, lookback=24):
    df = df.copy()

    # Calculating mean and distance from the same to eventually calculate Z-Score
    df['SMA'] = df['Close'].rolling(window=lookback).mean()
    df['Dist_from_SMA'] = df['Close'] - df['SMA']

    # Calculating Z-Score
    rolling_std = df['Close'].rolling(window=lookback).std()
    df['Z-Score'] = (df['Close'] - df['SMA']) / rolling_std

    # Calculating Returns
    df['Returns'] = df['Close'].pct_change()

    # Calculating Volatility
    df['Volatility'] = df['Returns'].rolling(window=lookback).std()

    # Drop NaN values
    df = df.dropna()

    return df


def prepare_seq(df, seq_len=24, target_col='Returns'):
    feature_cols = ['Dist_from_SMA', 'Z_Score', 'Returns', 'Volatility']
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[feature_cols])

    X = []
    y = []

    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i - seq_len : i])
        y.append(df[target_col].iloc[i])

    X = np.array(X)
    y = np.array(y)

    return X, y, scaler


def split_data(X, y, train_ratio=0.70, val_ratio=0.15):
    train_size = int(train_ratio * len(X))
    val_size = int(val_ratio * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test