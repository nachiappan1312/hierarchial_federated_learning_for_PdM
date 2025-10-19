import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np

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
        
        # Aggregate with priority weighting and proper type handling
        new_weights = {}
        first_weights = edge_updates[0]['weights']
        
        for key in first_weights.keys():
            first_tensor = first_weights[key]
            new_weights[key] = torch.zeros_like(first_tensor)
            
            # Handle different tensor types appropriately
            if 'num_batches_tracked' in key:
                # For num_batches_tracked, take the maximum
                max_val = first_tensor.clone()
                for update in edge_updates:
                    if update['weights'][key] > max_val:
                        max_val = update['weights'][key]
                new_weights[key] = max_val
                
            elif 'running_mean' in key or 'running_var' in key:
                # For running statistics, do weighted average
                for update in edge_updates:
                    weight = update['final_weight'] / total_weighted
                    # Convert to float for computation
                    tensor_val = update['weights'][key].float()
                    new_weights[key] += tensor_val * weight
                # Convert back to original dtype
                new_weights[key] = new_weights[key].to(first_tensor.dtype)
                
            else:
                # For regular parameters (weights and biases)
                for update in edge_updates:
                    weight = update['final_weight'] / total_weighted
                    param_tensor = update['weights'][key]
                    
                    # Ensure float type for arithmetic
                    if param_tensor.dtype in [torch.long, torch.int]:
                        param_tensor = param_tensor.float()
                    
                    new_weights[key] += param_tensor * weight
                
                # Maintain original dtype
                if first_tensor.dtype != new_weights[key].dtype:
                    new_weights[key] = new_weights[key].to(first_tensor.dtype)
        
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
        
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.LongTensor(y_test).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
        
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                try:
                    outputs = self.global_model(batch_X)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(batch_y.cpu().numpy())
                except Exception as e:
                    print(f"Warning: Evaluation batch error: {e}")
                    continue
        
        if total == 0:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
        
        accuracy = correct / total
        
        # Calculate additional metrics
        try:
            from sklearn.metrics import precision_score, recall_score, f1_score
            precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        except Exception as e:
            print(f"Warning: Metric calculation error: {e}")
            precision = accuracy
            recall = accuracy
            f1 = accuracy
        
        self.training_history['accuracy'].append(accuracy)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }