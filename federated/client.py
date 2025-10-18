import torch
import torch.nn as nn
import torch.optim as optim
from models.lightweight_cnn import LightweightCNN
from models.medium_lstm import MediumLSTM
from models.full_model import FullCNNLSTM
from models.pruning import AdaptivePruning

class IoTDevice:
    """Simulates an IoT device in federated learning"""
    
    def __init__(self, device_id, data, labels, capability_score, gateway_id):
        self.device_id = device_id
        self.X = torch.FloatTensor(data)
        self.y = torch.LongTensor(labels)
        self.capability_score = capability_score
        self.gateway_id = gateway_id
        
        # Assign model based on capability
        self.model = self._assign_model()
        self.pruner = AdaptivePruning()
        
        # Training stats
        self.loss_history = []
        self.health_score = 1.0
    
    def _assign_model(self):
        """Assign model architecture based on device capability"""
        input_size = self.X.shape[2]
        
        if self.capability_score < 0.4:
            return LightweightCNN(input_channels=input_size)
        elif self.capability_score < 0.7:
            return MediumLSTM(input_size=input_size)
        else:
            return FullCNNLSTM(input_channels=input_size)
    
    def train_local_model(self, global_weights, epochs=5, batch_size=64, lr=0.001):
        """Perform local training"""
        # Load global weights
        self.model.load_state_dict(global_weights)
        
        # Calculate health score and apply adaptive pruning
        recent_data = self.X[-100:].numpy() if len(self.X) > 100 else self.X.numpy()
        self.health_score = self.pruner.calculate_health_score(
            recent_data.reshape(-1, recent_data.shape[-1]),
            thresholds=self._get_thresholds()
        )
        
        pruning_ratio = self.pruner.get_pruning_ratio(self.health_score)
        self.model = self.pruner.prune_model(self.model, pruning_ratio)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        dataset = torch.utils.data.TensorDataset(self.X, self.y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        self.model.train()
        epoch_losses = []
        
        for epoch in range(epochs):
            batch_losses = []
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            
            epoch_loss = np.mean(batch_losses)
            epoch_losses.append(epoch_loss)
        
        self.loss_history.extend(epoch_losses)
        
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
            'loss': np.mean(epoch_losses)
        }
    
    def _validate(self):
        """Simple validation on 20% of data"""
        val_size = int(0.2 * len(self.X))
        val_X = self.X[-val_size:]
        val_y = self.y[-val_size:]
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(val_X)
            criterion = nn.CrossEntropyLoss()
            val_loss = criterion(outputs, val_y).item()
        
        return val_loss
    
    def _get_thresholds(self):
        """Get anomaly thresholds (simplified)"""
        data = self.X.numpy().reshape(-1, self.X.shape[-1])
        thresholds = []
        for i in range(data.shape[1]):
            thresholds.append({
                'mean': np.abs(data[:, i].mean()) + 2 * data[:, i].std(),
                'std': 2 * data[:, i].std(),
                'trend': 0.1
            })
        return thresholds