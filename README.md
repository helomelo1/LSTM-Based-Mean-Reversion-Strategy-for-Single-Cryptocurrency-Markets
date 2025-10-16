# LSTM-Based Quantitative Trading and Performance Evaluation System

## 📜 Project Description

This project implements a quantitative trading system using a Long Short-Term Memory (LSTM) neural network to forecast financial time series data. The system generates trading signals based on the model's predictions, backtests the trading strategy, and evaluates its performance using key financial metrics. This repository provides a complete pipeline for building, training, and evaluating a trading bot.

---

## 🚀 Key Features

* **LSTM-Based Forecasting:** Utilizes an LSTM model to capture temporal dependencies in financial data.
* **Trading Signal Generation:** Generates buy/sell signals based on the forecasted price movements.
* **Backtesting Engine:** A robust backtesting system to simulate the trading strategy on historical data.
* **Performance Evaluation:** Calculates and analyzes key performance metrics such as Sharpe Ratio, Sortino Ratio, and Maximum Drawdown.
* **Statistical Analysis:** Includes notebooks for statistical tests like Augmented Dickey-Fuller for stationarity and Johansen for cointegration.

---

## 📂 File Descriptions

* **`models.py`**: Contains the core Python classes for the trading bot, including the LSTM model, data preprocessing, backtesting, and performance evaluation.
* **`Integration_Cointegration_and_Stationarity.ipynb`**: A Jupyter Notebook for performing statistical analysis on time series data, including tests for stationarity and cointegration.
* **`README.md`**: This file, providing an overview of the project.

---

## ⚙️ Methodology

1.  **Data Preprocessing:** The financial time series data is loaded, cleaned, and scaled. It is then transformed into a suitable format for the LSTM model.
2.  **Model Training:** An LSTM model is trained on the historical data to learn patterns and forecast future price movements.
3.  **Signal Generation:** The trained model predicts future prices. Based on these predictions, trading signals (buy, sell, or hold) are generated.
4.  **Backtesting:** The trading strategy is simulated on a test set of historical data to evaluate its performance.
5.  **Performance Evaluation:** The backtesting results are analyzed using various performance metrics to assess the profitability and risk of the strategy.

---

## 🔧 Installation and Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/helomelo1/lstm-based-quantitative-trading-and-performance-evaluation-system.git](https://github.com/helomelo1/lstm-based-quantitative-trading-and-performance-evaluation-system.git)
    cd lstm-based-quantitative-trading-and-performance-evaluation-system
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🏃‍♀️ How to Run

1.  **Statistical Analysis (Optional):**
    * Open and run the `Integration_Cointegration_and_Stationarity.ipynb` notebook in a Jupyter environment to perform preliminary analysis on your data.

2.  **Run the Trading Bot:**
    * Utilize the classes and functions in `models.py` to create an instance of the trading bot, train the LSTM model, and run the backtest.
    ```python
    # Example usage in a Python script
    from models import TradingBot

    # Initialize the bot with your data and parameters
    bot = TradingBot(data_path='your_data.csv', ...)

    # Run the backtest
    bot.backtest_strategy()

    # Get the performance metrics
    bot.get_performance()
    ```

---

## 📦 Dependencies

You will need to create a `requirements.txt` file with the following content: