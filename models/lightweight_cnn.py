import torch
import torch.nn as nn

class LightweightCNN(nn.Module):
    """Lightweight CNN for resource-constrained devices"""
    def __init__(self, input_channels=14, num_classes=2):
        super(LightweightCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7, 128)  # Adjust based on input size
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # x shape: (batch, timesteps, features)
        x = x.transpose(1, 2)  # -> (batch, features, timesteps)
        
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        
        x = self.flatten(x)
        x = self.dropout(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)