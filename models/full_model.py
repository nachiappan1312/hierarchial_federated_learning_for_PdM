import torch
import torch.nn as nn

class FullCNNLSTM(nn.Module):
    """Full CNN-LSTM hybrid for powerful devices"""
    def __init__(self, input_channels=14, num_classes=2):
        super(FullCNNLSTM, self).__init__()
        
        # CNN layers for feature extraction
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        
        # LSTM layers for temporal modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # x shape: (batch, timesteps, features)
        x = x.transpose(1, 2)  # -> (batch, features, timesteps)
        
        # CNN feature extraction
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        # Prepare for LSTM: (batch, channels, time) -> (batch, time, channels)
        x = x.transpose(1, 2)
        
        # LSTM temporal modeling
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        
        # Classification head
        x = self.dropout1(self.relu(self.fc1(last_output)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x