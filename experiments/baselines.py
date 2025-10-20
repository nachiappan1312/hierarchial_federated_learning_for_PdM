import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict
from collections import defaultdict


class CentralizedTrainer:
    """Baseline: Centralized learning (all data in cloud)"""
    
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = torch.FloatTensor(X_train)
        self.y_train = torch.LongTensor(y_train)
        self.X_test = torch.FloatTensor(X_test)
        self.y_test = torch.LongTensor(y_test)
    
    def train(self, epochs=50, batch_size=64, lr=0.001, verbose=True):
        """Train centralized model"""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        results = {'epoch': [], 'train_loss': [], 'test_accuracy': []}
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            
            # Evaluate
            accuracy = self.evaluate()
            
            results['epoch'].append(epoch + 1)
            results['train_loss'].append(avg_loss)
            results['test_accuracy'].append(accuracy)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, "
                      f"Accuracy: {accuracy:.4f}")
        
        return results
    
    def evaluate(self):
        """Evaluate on test set"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            outputs = self.model(self.X_test)
            _, predicted = torch.max(outputs.data, 1)
            total = self.y_test.size(0)
            correct = (predicted == self.y_test).sum().item()
        
        return correct / total


class StandardFedAvg:
    """Baseline: Standard FedAvg without hierarchical aggregation"""
    
    def __init__(self, cloud_server, devices, X_test, y_test):
        self.cloud_server = cloud_server
        self.devices = devices
        self.X_test = X_test
        self.y_test = y_test
    
    def train(self, num_rounds=100, local_epochs=5, verbose=True):
        """Execute standard FedAvg"""
        results = {
            'round': [],
            'accuracy': [],
            'f1_score': [],
            'communication_mb': []
        }
        
        for round_num in range(num_rounds):
            if verbose and (round_num + 1) % 10 == 0:
                print(f"Round {round_num + 1}/{num_rounds}")
            
            # Device training
            device_updates = []
            global_weights = self.cloud_server.get_global_weights()
            
            for device in self.devices:
                update = device.train_local_model(
                    global_weights=global_weights,
                    epochs=local_epochs
                )
                device_updates.append(update)
            
            # Simple FedAvg aggregation (no quality weighting)
            total_samples = sum(u['num_samples'] for u in device_updates)
            new_weights = {}
            
            # FIXED: Proper type handling for different tensor types
            first_weights = device_updates[0]['weights']
            
            for key in first_weights.keys():
                first_tensor = first_weights[key]
                
                # Handle different tensor types appropriately
                if 'num_batches_tracked' in key:
                    # For num_batches_tracked, take the maximum (it's a counter)
                    new_weights[key] = first_tensor.clone()
                    for update in device_updates[1:]:
                        if update['weights'][key] > new_weights[key]:
                            new_weights[key] = update['weights'][key]
                
                elif 'running_mean' in key or 'running_var' in key:
                    # For running statistics, do weighted average
                    new_weights[key] = torch.zeros_like(first_tensor).float()
                    for update in device_updates:
                        weight = update['num_samples'] / total_samples
                        tensor_val = update['weights'][key].float()
                        new_weights[key] += tensor_val * weight
                    # Convert back to original dtype
                    new_weights[key] = new_weights[key].to(first_tensor.dtype)
                
                else:
                    # For regular parameters (weights, biases)
                    new_weights[key] = torch.zeros_like(first_tensor).float()
                    
                    for update in device_updates:
                        weight = update['num_samples'] / total_samples
                        param_tensor = update['weights'][key]
                        
                        # Ensure float type for arithmetic
                        if param_tensor.dtype in [torch.long, torch.int]:
                            param_tensor = param_tensor.float()
                        
                        new_weights[key] += param_tensor * weight
                    
                    # Maintain original dtype
                    if first_tensor.dtype != new_weights[key].dtype:
                        new_weights[key] = new_weights[key].to(first_tensor.dtype)
            
            self.cloud_server.global_model.load_state_dict(new_weights)
            
            # Evaluate
            metrics = self.cloud_server.evaluate(self.X_test, self.y_test)
            
            results['round'].append(round_num + 1)
            results['accuracy'].append(metrics['accuracy'])
            results['f1_score'].append(metrics['f1_score'])
            
            # Communication cost (higher than hierarchical)
            total_params = sum(
                p.numel() for p in self.cloud_server.global_model.parameters()
            )
            comm_mb = (total_params * 4 * len(self.devices) * 2) / (1024 ** 2)
            results['communication_mb'].append(comm_mb)
        
        return results