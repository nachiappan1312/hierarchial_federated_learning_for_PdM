# models/pruning.py

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
            sensor_data: Recent sensor readings (n_samples, n_features)
            thresholds: Anomaly thresholds for each sensor
        
        Returns:
            health_score: Float in [0, 1], where 1 = healthy
        """
        if len(sensor_data) == 0 or len(thresholds) == 0:
            return 1.0
        
        anomaly_scores = []
        
        # Ensure we don't go out of bounds
        n_features = min(sensor_data.shape[1], len(thresholds))
        
        for i in range(n_features):
            try:
                data = sensor_data[:, i]
                threshold = thresholds[i]
                
                # Check if sensor values exceed thresholds
                mean_val = np.mean(data)
                std_val = np.std(data)
                
                if abs(mean_val) > threshold.get('mean', float('inf')):
                    anomaly_scores.append(1)
                if std_val > threshold.get('std', float('inf')):
                    anomaly_scores.append(1)
                    
                # Check trend
                if len(data) > 1:
                    try:
                        trend = np.polyfit(range(len(data)), data, 1)[0]
                        if abs(trend) > threshold.get('trend', float('inf')):
                            anomaly_scores.append(1)
                    except:
                        pass
            except Exception as e:
                # Skip this feature if there's an error
                continue
        
        if len(anomaly_scores) == 0:
            return 1.0
        
        anomaly_rate = sum(anomaly_scores) / max(len(anomaly_scores), 1)
        health_score = 1 / (1 + np.exp(5 * (anomaly_rate - 0.5)))
        
        return float(health_score)
    
    def get_pruning_ratio(self, health_score):
        """Determine pruning ratio based on health score"""
        pruning_ratio = self.p_max * health_score + self.p_min * (1 - health_score)
        return float(np.clip(pruning_ratio, 0.0, 0.9))
    
    def prune_model(self, model, pruning_ratio):
        """Apply structured pruning to model"""
        if pruning_ratio <= 0 or pruning_ratio >= 1:
            return model
        
        try:
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
                try:
                    prune.l1_unstructured(module, name=param_name, amount=pruning_ratio)
                except Exception as e:
                    # Skip if pruning fails for this layer
                    continue
        
        except Exception as e:
            print(f"Warning: Pruning failed: {e}")
        
        return model
    
    def remove_pruning(self, model):
        """Make pruning permanent"""
        try:
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv1d, nn.Linear, nn.LSTM)):
                    try:
                        prune.remove(module, 'weight')
                    except:
                        pass
                    
                    # Also try to remove from LSTM specific weights
                    if isinstance(module, nn.LSTM):
                        for weight_name in ['weight_ih_l0', 'weight_hh_l0']:
                            try:
                                prune.remove(module, weight_name)
                            except:
                                pass
        except Exception as e:
            print(f"Warning: Remove pruning failed: {e}")
        
        return model