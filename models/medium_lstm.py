import torch
import torch.nn as nn

class MediumLSTM(nn.Module):
    """Medium LSTM for mid-tier devices"""
    def __init__(self, input_size=14, hidden_size=64, num_layers=2, num_classes=2):
        super(MediumLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        self.fc1 = nn.Linear(hidden_size, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # x shape: (batch, timesteps, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last timestep output
        last_output = lstm_out[:, -1, :]
        
        x = self.dropout(torch.relu(self.fc1(last_output)))
        x = self.fc2(x)
        
        return x