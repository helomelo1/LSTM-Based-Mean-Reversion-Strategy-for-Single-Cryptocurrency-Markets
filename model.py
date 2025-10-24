import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AttentionLayer(nn.Module):
    def __init(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        attention_scores = self.attention(lstm_output)
        attention_weights = torch.softmax(attention_scores, dim=1)\

        context_vector = torch.sum(attention_weights * lstm_output)

        return context_vector, attention_weights


class AttentionBiLSTM(nn.Module):
    def __init__(self, n_features=4, hidden_size=64, num_layers=2, dropout=0.3):
        super(AttentionBiLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.bilstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.attention = AttentionLayer(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        lstm_out, (hidden, cell) = self.bilstm(x)
        context_vector, attention_weights = self.attention(lstm_out)
        normalized = self.batch_norm(context_vector)
        dropped = self.dropout(normalized)
        pred = self.fc(dropped)

        return pred.squeeze(), attention_weights


def create_data_loaders(X_train, y_train, X_val, y_val):
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)

    train_loader = DataLoader(train_loader, batch_size=32)
    val_loader = DataLoader(val_loader, batch_size=32)

    return train_loader, val_loader


def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, device=DEVICE):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 15
    patience_cnt = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            preds, _ = model(X_batch)
            loss = criterion(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                if isinstance(model, AttentionBiLSTM):
                    predictions, _ = model(X_batch)
                else:
                    predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()
        
        # Calculate average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_attention_bilstm_model.pth')
            print(f'  → Best model saved (val_loss: {val_loss:.6f})')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping at epoch {epoch+1}')
                break
    model.load_state_dict(torch.load('best_attention_bilstm_model.pth'))
    
    return {'train_losses': train_losses, 'val_losses': val_losses}


def predict(model, X, device=DEVICE, return_attention=False):
    model = model.to(device)
    model.eval()
    
    X_tensor = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        if isinstance(model, AttentionBiLSTM):
            predictions, attention_weights = model(X_tensor)
            if return_attention:
                return predictions.cpu().numpy(), attention_weights.cpu().numpy()
        else:
            predictions = model(X_tensor)
    
    return predictions.cpu().numpy()


def load_model(model_path='best_attention_bilstm_model.pth', n_features=4, hidden_size=64, device='cpu', use_attention=True):
    if use_attention:
        model = AttentionBiLSTM(n_features=n_features, hidden_size=hidden_size, num_layers=2, dropout=0.3)
    else:
        model = SimpleLSTM(n_features=n_features, hidden_size=hidden_size, num_layers=1, dropout=0.2)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")
    return model