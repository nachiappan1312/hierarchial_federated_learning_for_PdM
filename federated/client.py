# federated/client.py - FINAL COMPLETE VERSION

import torch
import torch.nn as nn
import torch.optim as optim
from models.full_model import FullCNNLSTM
from models.pruning import AdaptivePruning
import numpy as np

class IoTDevice:
    """Simulates an IoT device in federated learning"""
    
    def __init__(self, device_id, data, labels, capability_score, gateway_id):
        self.device_id = device_id
        self.X = torch.FloatTensor(data)
        self.y = torch.LongTensor(labels)
        self.capability_score = capability_score
        self.gateway_id = gateway_id
        
        # Store input dimensions
        if len(data.shape) == 3:
            self.input_channels = data.shape[2]
        elif len(data.shape) == 2:
            self.input_channels = data.shape[1]
        else:
            self.input_channels = 14
        
        # FIXED: Always use 2 classes for consistency with global model
        self.num_classes = 2  # Binary classification
        
        # Model will be created during training
        self.model = None
        self.pruner = AdaptivePruning()
        
        # Training stats
        self.loss_history = []
        self.health_score = 1.0
    
    def _create_model(self, input_channels, num_classes):
        """Create model with specified dimensions"""
        return FullCNNLSTM(
            input_channels=input_channels,
            num_classes=num_classes
        )
    
    def train_local_model(self, global_weights, epochs=5, batch_size=64, lr=0.001):
        """Perform local training"""
        
        # Check if we have enough data and classes
        unique_labels = torch.unique(self.y)
        if len(unique_labels) < 2:
            print(f"Warning: Device {self.device_id} has only {len(unique_labels)} class(es). Skipping training.")
            # Return current weights or global weights
            if self.model is not None:
                return {
                    'weights': self.model.state_dict(),
                    'num_samples': len(self.X),
                    'quality_weight': 0.1,  # Low quality
                    'health_score': self.health_score,
                    'loss': 10.0,
                    'capability_score': self.capability_score
                }
            else:
                return {
                    'weights': global_weights,
                    'num_samples': len(self.X),
                    'quality_weight': 0.1,
                    'health_score': 1.0,
                    'loss': 10.0,
                    'capability_score': self.capability_score
                }
        
        # Extract dimensions from global model
        if 'conv1.weight' in global_weights:
            global_input_channels = global_weights['conv1.weight'].shape[1]
        else:
            global_input_channels = self.input_channels
        
        if 'fc3.weight' in global_weights:
            global_num_classes = global_weights['fc3.weight'].shape[0]
        else:
            global_num_classes = self.num_classes
        
        # Create or recreate model with correct dimensions
        if (self.model is None or 
            self._get_model_input_channels() != global_input_channels or
            self._get_model_output_classes() != global_num_classes):
            self.model = self._create_model(global_input_channels, global_num_classes)
        
        # Adjust data dimensions if needed
        if self.input_channels != global_input_channels:
            self.X = self._adjust_input_dimensions(self.X, global_input_channels)
            self.input_channels = global_input_channels
        
        # Load global weights
        try:
            self.model.load_state_dict(global_weights, strict=True)
        except Exception as e:
            print(f"Warning: Device {self.device_id} failed to load weights: {e}")
            # Keep the newly initialized model
            pass
        
        # Calculate health score
        recent_data = self.X[-100:].numpy() if len(self.X) > 100 else self.X.numpy()
        self.health_score = self.pruner.calculate_health_score(
            recent_data.reshape(-1, recent_data.shape[-1]),
            thresholds=self._get_thresholds()
        )
        
        # Adjust pruning based on device capability
        base_pruning_ratio = self.pruner.get_pruning_ratio(self.health_score)
        
        if self.capability_score < 0.4:
            pruning_ratio = min(base_pruning_ratio * 1.5, 0.7)
        elif self.capability_score < 0.7:
            pruning_ratio = base_pruning_ratio
        else:
            pruning_ratio = base_pruning_ratio * 0.5
        
        # Apply pruning
        self.model = self.pruner.prune_model(self.model, pruning_ratio)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Adjust batch size and epochs based on capability
        if self.capability_score < 0.4:
            actual_batch_size = max(16, batch_size // 4)
            actual_epochs = max(2, epochs // 2)
        elif self.capability_score < 0.7:
            actual_batch_size = max(32, batch_size // 2)
            actual_epochs = epochs
        else:
            actual_batch_size = batch_size
            actual_epochs = epochs
        
        # Ensure we have enough data
        if len(self.X) < actual_batch_size:
            actual_batch_size = len(self.X)
        
        dataset = torch.utils.data.TensorDataset(self.X, self.y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=actual_batch_size, shuffle=True
        )
        
        self.model.train()
        epoch_losses = []
        
        for epoch in range(actual_epochs):
            batch_losses = []
            for batch_X, batch_y in dataloader:
                try:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    batch_losses.append(loss.item())
                except Exception as e:
                    print(f"Warning: Device {self.device_id} batch training error: {e}")
                    continue
            
            if batch_losses:
                epoch_loss = np.mean(batch_losses)
                epoch_losses.append(epoch_loss)
        
        if epoch_losses:
            self.loss_history.extend(epoch_losses)
            avg_loss = np.mean(epoch_losses)
        else:
            avg_loss = 10.0
        
        # Calculate quality weight
        val_loss = self._validate()
        quality_weight = np.exp(-2.0 * val_loss) * (1 + 1.5 * (1 - self.health_score))
        
        # Make pruning permanent
        self.model = self.pruner.remove_pruning(self.model)
        
        return {
            'weights': self.model.state_dict(),
            'num_samples': len(self.X),
            'quality_weight': quality_weight,
            'health_score': self.health_score,
            'loss': avg_loss,
            'capability_score': self.capability_score
        }
    
    def _get_model_input_channels(self):
        """Get input channels from current model"""
        if self.model is None:
            return None
        return self.model.conv1.in_channels
    
    def _get_model_output_classes(self):
        """Get output classes from current model"""
        if self.model is None:
            return None
        return self.model.fc3.out_features
    
    def _adjust_input_dimensions(self, X, target_channels):
        """Adjust input tensor dimensions to match target"""
        current_channels = X.shape[2]
        
        if current_channels == target_channels:
            return X
        elif current_channels > target_channels:
            return X[:, :, :target_channels]
        else:
            padding = torch.zeros(X.shape[0], X.shape[1], target_channels - current_channels)
            return torch.cat([X, padding], dim=2)
    
    def _validate(self):
        """Simple validation on 20% of data"""
        val_size = int(0.2 * len(self.X))
        if val_size == 0:
            val_size = min(10, len(self.X))
        
        val_X = self.X[-val_size:]
        val_y = self.y[-val_size:]
        
        # Check if validation set has both classes
        unique_val_labels = torch.unique(val_y)
        if len(unique_val_labels) < 2:
            return 1.0  # Return moderate loss
        
        self.model.eval()
        with torch.no_grad():
            try:
                outputs = self.model(val_X)
                criterion = nn.CrossEntropyLoss()
                val_loss = criterion(outputs, val_y).item()
            except Exception as e:
                val_loss = 1.0
        
        return val_loss
    
    def _get_thresholds(self):
        """Get anomaly thresholds"""
        data = self.X.numpy().reshape(-1, self.X.shape[-1])
        thresholds = []
        for i in range(data.shape[1]):
            col_data = data[:, i]
            col_mean = np.mean(col_data)
            col_std = np.std(col_data)
            thresholds.append({
                'mean': abs(col_mean) + 2 * col_std,
                'std': 2 * col_std if col_std > 0 else 1.0,
                'trend': 0.1
            })
        return thresholds