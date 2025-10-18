import torch
import numpy as np
import polars as pl
import vectorbt as vbt
from train_utils import volatility_target_positions


class Backtester:
    def __init__(self, target_vol=0.10, alpha=10.0, max_leverage=2.0, fees=0.0005):
        self.target_vol = target_vol
        self.alpha = alpha
        self.max_leverage = max_leverage
        self.fees = fees

    
    def run(self, preds, reals, freq="", plot=True):
        preds = np.asarray(preds)
        reals = np.asarray(reals)

        positions = volatility_target_positions(preds, reals, target_vol=self.target_vol, alpha=self.alpha, max_leverage=self.max_leverage)

        df = pl.DataFrame({
            "returns": reals,
            "positions": positions
        })
        df = df.with_columns([(1 + pl.col("returns")).cumprod().alias("price")])

        price_pd = df["price"].to_pandas()
        weights_pd = df["positions"].to_pandas()

        # Building Portfolio
        pf = vbt.Portfolio.from_weights(
            close = price_pd,
            weights = weights_pd,
            fees = self.fees,
            freq = freq
        )

        stats = pf.stats()
        print("\n---- BACKTEST SUMMARY ----")
        print(stats)

        return {
            "portfolio": pf,
            "stats": stats,
            "positions": weights_pd,
            "price": price_pd,
        }
    

    def from_model_output(model, dataloader, device="cpu", **kwargs):
        model.eval()
        preds, reals = [], []
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(x_batch)
                preds.extend(outputs.cpu().numpy().flatten())
                reals.extend(y_batch.cpu().numpy().flatten())

        backtester = Backtester(**kwargs)
        return backtester.run(preds, reals)