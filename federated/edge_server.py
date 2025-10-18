import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from copy import deepcopy

class EdgeGateway:
    """Edge gateway that performs intermediate aggregation"""
    
    def __init__(self, gateway_id, outlier_threshold=2.5):
        self.gateway_id = gateway_id
        self.outlier_threshold = outlier_threshold
        self.devices = []
        self.aggregated_weights = None
        self.total_samples = 0
        self.priority_weight = 1.0
    
    def register_device(self, device):
        """Register a device to this gateway"""
        self.devices.append(device)
    
    def aggregate_device_updates(self, device_updates):
        """
        Perform edge-level aggregation with outlier filtering
        
        Args:
            device_updates: List of dicts with 'weights', 'num_samples', 
                           'quality_weight', 'health_score'
        """
        if not device_updates:
            return None
        
        # Step 1: Outlier detection and filtering
        filtered_updates = self._filter_outliers(device_updates)
        
        if not filtered_updates:
            filtered_updates = device_updates  # Keep all if all are outliers
        
        # Step 2: Calculate aggregation weights
        total_weighted_samples = 0
        for update in filtered_updates:
            weighted_samples = update['num_samples'] * update['quality_weight']
            total_weighted_samples += weighted_samples
            update['aggregation_weight'] = weighted_samples
        
        # Step 3: Weighted aggregation
        aggregated = {}
        first_weights = filtered_updates[0]['weights']
        
        for key in first_weights.keys():
            aggregated[key] = torch.zeros_like(first_weights[key])
            
            for update in filtered_updates:
                weight = update['aggregation_weight'] / total_weighted_samples
                aggregated[key] += update['weights'][key] * weight
        
        # Step 4: Calculate priority weight for cloud aggregation
        # Higher priority for gateways with devices showing degradation
        degradation_count = sum(
            1 for u in filtered_updates if u['health_score'] < 0.7
        )
        self.priority_weight = 1.0 + 2.0 * (degradation_count / len(filtered_updates))
        
        self.aggregated_weights = aggregated
        self.total_samples = sum(u['num_samples'] for u in filtered_updates)
        
        return {
            'weights': aggregated,
            'num_samples': self.total_samples,
            'priority_weight': self.priority_weight,
            'num_devices': len(filtered_updates)
        }
    
    def _filter_outliers(self, device_updates):
        """Filter outlier model updates using statistical methods"""
        if len(device_updates) < 3:
            return device_updates
        
        # Calculate pairwise distances between model updates
        distances = []
        for i, update_i in enumerate(device_updates):
            dist_sum = 0
            for j, update_j in enumerate(device_updates):
                if i != j:
                    dist = self._model_distance(
                        update_i['weights'], 
                        update_j['weights']
                    )
                    dist_sum += dist
            distances.append(dist_sum / (len(device_updates) - 1))
        
        # Statistical outlier detection
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        threshold = mean_dist + self.outlier_threshold * std_dist
        
        filtered = [
            update for update, dist in zip(device_updates, distances)
            if dist <= threshold
        ]
        
        return filtered if filtered else device_updates
    
    def _model_distance(self, weights1, weights2):
        """Calculate L2 distance between two model weight dictionaries"""
        distance = 0.0
        for key in weights1.keys():
            if key in weights2:
                distance += torch.sum((weights1[key] - weights2[key]) ** 2).item()
        return np.sqrt(distance)
    
    def compress_weights(self, compression_ratio=0.3):
        """Apply quantization for communication efficiency"""
        if self.aggregated_weights is None:
            return None
        
        compressed = {}
        for key, tensor in self.aggregated_weights.items():
            # Quantize to int8
            min_val = tensor.min()
            max_val = tensor.max()
            scale = (max_val - min_val) / 255.0
            
            quantized = ((tensor - min_val) / scale).round().to(torch.uint8)
            
            compressed[key] = {
                'quantized': quantized,
                'scale': scale,
                'min': min_val,
                'shape': tensor.shape
            }
        
        return compressed
    
    def decompress_weights(self, compressed):
        """Decompress quantized weights"""
        decompressed = {}
        for key, data in compressed.items():
            quantized = data['quantized'].float()
            decompressed[key] = quantized * data['scale'] + data['min']
        return decompressed

