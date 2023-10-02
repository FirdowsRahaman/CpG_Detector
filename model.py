import torch
import torch.nn as nn

class CpGCounter(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, num_classes=1):
        super(CpGCounter, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, hidden_size//2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.liner = nn.Linear(hidden_size//2, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take the last time step output
        fc_out = self.relu(self.fc(lstm_out))
        fc_out = self.dropout(fc_out)
        logits = self.liner(fc_out)
        return logits
