import os
import torch
import numpy as np
from tqdm import tqdm


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


def compute_metrics(preds, reals):
    # Long Short signals conversion
    positions = np.sign(preds)
    realized_returns = positions * reals

    cum_returns = np.sum(realized_returns)
    volatility = np.std(realized_returns)
    sharpe = cum_returns / volatility if volatility > 0 else np.nan

    # Drawdown
    cum_curve = np.cumsum(realized_returns)
    running_max = np.maximum.accumulate(cum_curve)
    drawdown = np.min(cum_curve - running_max)

    # Turnover
    turnover = np.mean(np.abs(np.diff(positions)))

    # Metrics Dict
    metrics = {
        "Sharpe Ratio": sharpe,
        "Cumulative Return": cum_returns,
        "Volatility": volatility,
        "Drawdown": drawdown,
        "Turnover": turnover
    }

    return metrics