import torch
import numpy as np


def train_model(model, optimizer, criterion, train_loader, test_loader, device, epochs=20, verbose=True):
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

        if verbose:
            print(f"Epoch [{epoch+1}/{epochs}]  Train Loss: {avg_train_loss:.6f}  Val Loss: {avg_val_loss:.6f}")

    return train_losses, val_losses


def predict(model, data_loader, device):
    model.eval()
    preds = [], reals = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            preds.extend(outputs.cpu().numpy().flatten())
            reals.extend(y_batch.cpu().numpy().flatten())

    preds = np.array(preds)
    reals = np.array(reals)
    return preds, reals


def volatility_target_positions(preds, returns, target_vol=0.10, window=20, alpha=10.0, max_leverage=2.0, threshold=0.0):
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy().flatten()
    
    preds = np.nan_to_num(preds, nan=0.0)
    returns = np.nan_to_num(returns, nan=0.0)

    rolling_vol = np.array([
        np.std(returns[max(0, i - window):i]) for i in range(len(returns))
    ])
    rolling_vol = np.maximum(rolling_vol, 1e-6) * np.sqrt(252)

    raw_signal = np.tanh(alpha * preds)
    raw_signal[np.abs(raw_signal) < threshold] = 0.0

    pos = raw_signal * (target_vol / rolling_vol)
    pos = np.clip(pos, -max_leverage, max_leverage)

    return pos


def compute_metrics(preds, reals, target_vol=0.10, window=20, alpha=10.0, max_leverage=2.0, threshold=0.0, freq=252):
    positions = volatility_target_positions(preds, reals, target_vol, window, alpha, max_leverage, threshold)
    strategy_returns = positions * reals

    # Portfolio metrics
    ann_return = np.mean(strategy_returns) * freq
    ann_vol = np.std(strategy_returns) * np.sqrt(freq)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    # Drawdown
    cum_curve = np.cumprod(1 + strategy_returns)
    running_max = np.maximum.accumulate(cum_curve)
    drawdown = (cum_curve - running_max) / running_max
    max_drawdown = drawdown.min()

    # Turnover
    turnover = np.mean(np.abs(np.diff(positions)))

    metrics = {
        "Sharpe Ratio": sharpe,
        "Annualized Return": ann_return,
        "Volatility": ann_vol,
        "Max Drawdown": max_drawdown,
        "Turnover": turnover
    }

    return metrics