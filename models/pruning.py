import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np

class AdaptivePruning:
    """Implements adaptive temporal pruning"""
    
    def __init__(self, p_max=0.6, p_min=0.1):
        self.p_max = p_max
        self.p_min = p_min
    
    def calculate_health_score(self, sensor_data, thresholds):
        """
        Calculate device health score based on sensor readings
        
        Args:
            sensor_data: Recent sensor readings
            thresholds: Anomaly thresholds for each sensor
        
        Returns:
            health_score: Float in [0, 1], where 1 = healthy
        """
        anomaly_scores = []
        
        for i, (data, threshold) in enumerate(zip(sensor_data.T, thresholds)):
            # Check if sensor values exceed thresholds
            if np.abs(data.mean()) > threshold['mean']:
                anomaly_scores.append(1)
            if data.std() > threshold['std']:
                anomaly_scores.append(1)
            if len(data) > 1:
                trend = np.polyfit(range(len(data)), data, 1)[0]
                if np.abs(trend) > threshold['trend']:
                    anomaly_scores.append(1)
        
        anomaly_rate = sum(anomaly_scores) / max(len(anomaly_scores), 1)
        health_score = 1 / (1 + np.exp(5 * (anomaly_rate - 0.5)))
        
        return health_score
    
    def get_pruning_ratio(self, health_score):
        """Determine pruning ratio based on health score"""
        return self.p_max * health_score + self.p_min * (1 - health_score)
    
    def prune_model(self, model, pruning_ratio):
        """Apply structured pruning to model"""
        parameters_to_prune = []
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv1d):
                parameters_to_prune.append((module, 'weight'))
            elif isinstance(module, nn.Linear):
                parameters_to_prune.append((module, 'weight'))
            elif isinstance(module, nn.LSTM):
                # Prune LSTM weight matrices
                for weight_name in ['weight_ih_l0', 'weight_hh_l0']:
                    if hasattr(module, weight_name):
                        parameters_to_prune.append((module, weight_name))
        
        # Apply L1 unstructured pruning
        for module, param_name in parameters_to_prune:
            prune.l1_unstructured(module, name=param_name, amount=pruning_ratio)
        
        return model
    
    def remove_pruning(self, model):
        """Make pruning permanent"""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Linear, nn.LSTM)):
                try:
                    prune.remove(module, 'weight')
                except:
                    pass
        return model