import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from copy import deepcopy

class CloudServer:
    """Central cloud server coordinating federated learning"""
    
    def __init__(self, model_config, device='cpu'):
        self.device = device
        self.model_config = model_config
        
        # Initialize global model (using full model as global)
        from models.full_model import FullCNNLSTM
        self.global_model = FullCNNLSTM(**model_config).to(device)
        
        self.round = 0
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'communication_cost': []
        }
    
    def get_global_weights(self):
        """Return current global model weights"""
        return deepcopy(self.global_model.state_dict())
    
    def aggregate_edge_updates(self, edge_updates):
        """
        Perform cloud-level aggregation with failure-imminent priority
        
        Args:
            edge_updates: List of dicts from edge gateways containing
                         'weights', 'num_samples', 'priority_weight'
        """
        if not edge_updates:
            return
        
        # Calculate total weighted samples
        total_weighted = 0
        for update in edge_updates:
            weighted = update['num_samples'] * update['priority_weight']
            update['final_weight'] = weighted
            total_weighted += weighted
        
        # Aggregate with priority weighting
        new_weights = {}
        first_weights = edge_updates[0]['weights']
        
        for key in first_weights.keys():
            new_weights[key] = torch.zeros_like(first_weights[key])
            
            for update in edge_updates:
                weight = update['final_weight'] / total_weighted
                new_weights[key] += update['weights'][key] * weight
        
        # Update global model
        self.global_model.load_state_dict(new_weights)
        self.round += 1
        
        # Track communication cost (simplified)
        total_params = sum(p.numel() for p in self.global_model.parameters())
        comm_cost_mb = (total_params * 4 * len(edge_updates)) / (1024 ** 2)
        self.training_history['communication_cost'].append(comm_cost_mb)
    
    def evaluate(self, X_test, y_test, batch_size=64):
        """Evaluate global model on test set"""
        self.global_model.eval()
        
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_test),
            torch.LongTensor(y_test)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
        
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                outputs = self.global_model(batch_X.to(self.device))
                _, predicted = torch.max(outputs.data, 1)
                
                total += batch_y.size(0)
                correct += (predicted.cpu() == batch_y).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.numpy())
        
        accuracy = correct / total
        
        # Calculate additional metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        self.training_history['accuracy'].append(accuracy)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
