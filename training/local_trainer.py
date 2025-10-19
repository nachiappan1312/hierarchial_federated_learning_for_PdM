import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Tuple, Optional

class LocalTrainer:
    """Handles local training on edge devices"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        learning_rate: float = 0.001,
        batch_size: int = 64
    ):
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader with training data
            epoch: Current epoch number
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 5,
        validation_split: float = 0.2
    ) -> Dict:
        """
        Complete training procedure
        
        Args:
            X_train: Training features
            y_train: Training labels
            epochs: Number of epochs
            validation_split: Fraction of data for validation
            
        Returns:
            Dictionary with training history
        """
        # Split into train and validation
        val_size = int(len(X_train) * validation_split)
        train_size = len(X_train) - val_size
        
        X_train_split = X_train[:-val_size] if val_size > 0 else X_train
        y_train_split = y_train[:-val_size] if val_size > 0 else y_train
        X_val = X_train[-val_size:] if val_size > 0 else None
        y_val = y_train[-val_size:] if val_size > 0 else None
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_split),
            torch.LongTensor(y_train_split)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validation
            if X_val is not None:
                val_loss, val_acc = self.validate(X_val, y_val)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
        
        return history
    
    def validate(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[float, float]:
        """
        Validate model
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Tuple of (validation_loss, validation_accuracy)
        """
        self.model.eval()
        
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
