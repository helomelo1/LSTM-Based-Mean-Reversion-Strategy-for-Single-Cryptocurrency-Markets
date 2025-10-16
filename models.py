import torch
import torch.nn as nn


def get_device():
    """Return 'cuda' if available, else 'cpu'."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_model(model_class, input_size, **kwargs):
    """
    Initialize model, optimizer, and loss.

    Parameters
    ----------
    model_class : class
        LSTMModel or AttentionLSTMModel
    input_size : int
        Number of input features per timestep
    kwargs : dict
        Model hyperparameters (hidden_size, num_layers, etc.)
    """
    device = get_device()
    model = model_class(input_size=input_size, **kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=kwargs.get("lr", 1e-3))
    criterion = nn.MSELoss()
    return model, optimizer, criterion, device


class LSTMModel(nn.Module):
    """
    Standard LSTM for predicting next-day return or price.
    """

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, bidirectional=False):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional
        )
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction_factor, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)
        last_hidden = lstm_out[:, -1, :]  # take last timestep
        out = self.fc(last_hidden)
        return out


class AttentionLSTMModel(nn.Module):
    """
    LSTM with additive attention mechanism.
    The model learns weights for each timestep in the sequence,
    allowing it to focus on important time periods.
    """

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, bidirectional=False):
        super(AttentionLSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        direction_factor = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Attention layers
        self.attn = nn.Linear(hidden_size * direction_factor, hidden_size * direction_factor)
        self.context_vector = nn.Linear(hidden_size * direction_factor, 1, bias=False)

        # Final fully-connected output layer
        self.fc = nn.Linear(hidden_size * direction_factor, 1)

    def forward(self, x):
        """
        Forward pass through Attention-LSTM.
        Input: (batch, seq_len, input_size)
        Output: (batch, 1)
        """
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)
        
        # Attention weights (additive attention)
        attn_scores = torch.tanh(self.attn(lstm_out))       # (batch, seq_len, hidden)
        attn_weights = torch.softmax(self.context_vector(attn_scores), dim=1)  # (batch, seq_len, 1)
        
        # Weighted sum of hidden states
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden)
        
        # Final output
        out = self.fc(context)
        return out