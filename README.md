# LSTM-Based Mean Reversion Strategy for Single Cryptocurrency Markets

## Overview

This project implements a quantitative trading strategy for cryptocurrency markets (specifically ETH-USD) that combines a classic statistical arbitrage signal (Z-Score) with a deep learning model (Bidirectional LSTM with Attention) to filter trading signals.

The core idea is to identify when a price has moved "abnormally" far from its recent average (mean reversion) and then use an LSTM model to predict whether the price will actually revert, helping to avoid "false" signals.

## The Strategy: "Rubber Band" + "AI Filter"

1.  **The "Rubber Band" (Z-Score):** We calculate a Z-Score for the price based on its recent average (`SMA`) and volatility (`rolling_std`).
    * A high Z-Score ($> 2.0$) means the price is "weirdly high" and the "rubber band" is stretched. We bet it will **fall** (Go Short).
    * A low Z-Score ($< -2.0$) means the price is "weirdly low" and the "rubber band" is stretched. We bet it will **rise** (Go Long).

2.  **The "AI Filter" (Attention BiLSTM):** The Z-Score alone can be noisy. The LSTM model is trained on a sequence of features (`Z_Score`, `Returns`, `Volatility`, etc.) to predict the *next* return.
    * **Buy Signal:** The Z-Score is $<-2.0$ **AND** the LSTM predicts a *positive* return.
    * **Sell Signal:** The Z-Score is $> 2.0$ **AND** the LSTM predicts a *negative* return.

## File Structure

* `main.py`: The main entry point. It runs the entire pipeline from data fetching to backtesting.
* `data.py`: Handles fetching data from `yfinance`, calculating features (SMA, Z-Score, etc.), and preparing sequences for the LSTM.
* `model.py`: Defines the `AttentionBiLSTM` model architecture, the PyTorch `Dataset`, and the `train_model` and `predict` functions.
* `backtest.py`: Uses the `vectorbt` library to simulate the trading strategy based on the generated signals, calculate performance metrics, and plot results.

## How It Works (The Pipeline)

The `main.py` script executes these steps in order:

1.  **Fetch Data:** Downloads 1-hour ETH-USD data from `yfinance`.
2.  **Compute Features:** Calculates `SMA`, `Z_Score`, `Returns`, and `Volatility`.
3.  **Prepare Sequences:** Creates rolling windows (sequences) of data (e.g., 24 hours) as the $X$ features and the *next* hour's return as the $y$ target.
4.  **Split Data:** Splits the dataset into training, validation, and test sets.
5.  **Train Model:** Initializes and trains the `AttentionBiLSTM` model to predict the next return. The best model (based on validation loss) is saved to `best_attention_bilstm_model.pth`.
6.  **Generate Predictions:** Uses the trained model to generate predictions on the unseen test data.
7.  **Run Backtest:** The `VectorBTMeanReversion` class takes the Z-Scores and the model's predictions to generate final trade signals (entries and exits).
8.  **Print Results:** The backtest simulation is run, and a full report of the strategy's performance (Total Return, Win Rate, Max Drawdown, etc.) is printed to the console.

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd lstm-based-mean-reversion-strategy
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    Create a `requirements.txt` file with the following content:
    ```
    torch
    numpy
    pandas
    yfinance
    scikit-learn
    vectorbt
    matplotlib
    seaborn
    ```
    Then, install them:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1.  **Run the entire pipeline:**
    ```bash
    python main.py
    ```

2.  **Tweak Parameters (Optional):**
    You can easily change the strategy's behavior by editing the `CONFIG` dictionary at the top of `main.py`:

    ```python
    CONFIG = {
        "ticker": "ETH-USD",      # Cryptocurrency to trade
        "interval": "1h",         # Data frequency
        "period": "720d",         # How much data to download (720 days)
        "lookback": 24,           # Window for SMA + Z-Score (24 hours)
        "seq_len": 24,            # Sequence length for LSTM (24 hours)
        "hidden_size": 64,        # Number of neurons in LSTM
        "epochs": 30,             # Max training epochs
        "z_score_entry": 2.0,     # Z-Score level to trigger a trade
        "z_score_exit": 0.5,      # Z-Score level to close a trade
        ...
    }
    ```

### Outputs

* **Console:** A detailed backtest performance report.
* `best_attention_bilstm_model.pth`: The trained PyTorch model weights.
* `final_predictions_with_signals.csv`: A CSV file containing the test data, the model's predictions, and the final trade signals (long/short entries/exits).
* **Plots (if enabled):** By un-commenting the `plot_` functions in `main.py`, you can generate and save plots of the portfolio value, drawdowns, and trade signals.