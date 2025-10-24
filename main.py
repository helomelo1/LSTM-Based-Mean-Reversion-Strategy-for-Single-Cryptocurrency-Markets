import torch
import numpy as np
import pandas as pd

from data import fetch_data, compute_features, prepare_seq, split_data
from model import AttentionBiLSTM, create_data_loaders, train_model, predict
from backtest import VectorBTMeanReversion


def main():
    # ==========================
    # ⚙️ Parameters
    # ==========================
    CONFIG = {
        "ticker": "ETH-USD",
        "interval": "1h",
        "period": "720d",
        "lookback": 24,           # SMA + Z-Score window
        "seq_len": 24,            # Sequence length for LSTM
        "hidden_size": 64,
        "epochs": 30,
        "learning_rate": 0.001,
        "z_score_entry": 2.0,
        "z_score_exit": 0.5,
        "pred_threshold": 0.0,
        "initial_capital": 10000,
        "fees": 0.001
    }

    # ==========================
    # 1️⃣ Data Preparation
    # ==========================
    print("Fetching and preparing data...")
    df = fetch_data(
        ticker=CONFIG["ticker"],
        interval=CONFIG["interval"],
        period=CONFIG["period"]
    )
    df = compute_features(df, lookback=CONFIG["lookback"])

    X, y, scaler = prepare_seq(df, seq_len=CONFIG["seq_len"], target_col='Returns')
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # ==========================
    # 2️⃣ Model Training
    # ==========================
    print("\nInitializing model...")
    model = AttentionBiLSTM(
        n_features=X.shape[2],
        hidden_size=CONFIG["hidden_size"],
        num_layers=2,
        dropout=0.3
    )

    train_loader, val_loader = create_data_loaders(X_train, y_train, X_val, y_val)
    results = train_model(model, train_loader, val_loader, epochs=CONFIG["epochs"], lr=CONFIG["learning_rate"])

    print("\nTraining completed!")

    # ==========================
    # 3️⃣ Prediction
    # ==========================
    print("\nGenerating predictions...")
    preds = predict(model, X_test).flatten()

    df_for_backtest = df.iloc[-len(X_test):].copy()
    df_for_backtest['prediction'] = preds

    # ==========================
    # 4️⃣ Backtesting
    # ==========================
    print("\nRunning VectorBT backtest...")
    backtester = VectorBTMeanReversion(
        initial_capital=CONFIG["initial_capital"],
        fees=CONFIG["fees"]
    )

    portfolio = backtester.run_backtest(
        df_for_backtest,
        predictions=df_for_backtest['prediction'],
        z_score_entry=CONFIG["z_score_entry"],
        z_score_exit=CONFIG["z_score_exit"],
        pred_threshold=CONFIG["pred_threshold"]
    )

    backtester.print_results()
    # backtester.plot_results()
    # backtester.analyze_returns()
    # backtester.plot_positions(df_for_backtest)

    # ==========================
    # 5️⃣ Save outputs
    # ==========================
    df_for_backtest.to_csv("final_predictions_with_signals.csv", index=True)
    print("\nSaved final results to 'final_predictions_with_signals.csv'")


if __name__ == "__main__":
    main()
