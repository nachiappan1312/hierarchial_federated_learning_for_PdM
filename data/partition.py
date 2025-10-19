# data/partition.py

import numpy as np
from collections import defaultdict

def partition_data(X, y, num_devices=50, heterogeneity='high'):
    """
    Create non-IID partitions simulating heterogeneous IoT deployment
    Ensures each device has at least some samples from each class
    
    Args:
        X: Feature array (samples, timesteps, features)
        y: Labels
        num_devices: Number of virtual devices
        heterogeneity: 'low', 'medium', or 'high'
        
    Returns:
        List of tuples (device_X, device_y) for each device
    """
    
    device_data = []
    
    # Get class indices
    class_indices = defaultdict(list)
    for idx, label in enumerate(y):
        class_indices[int(label)].append(idx)
    
    num_classes = len(class_indices)
    
    # Ensure we have data for each class
    min_samples_per_class = max(10, len(X) // (num_devices * num_classes * 2))
    
    # Calculate device sizes
    if heterogeneity == 'high':
        sizes = np.random.lognormal(mean=6, sigma=1, size=num_devices)
    elif heterogeneity == 'medium':
        sizes = np.random.lognormal(mean=6, sigma=0.5, size=num_devices)
    else:
        sizes = np.ones(num_devices) * (len(X) / num_devices)
    
    sizes = (sizes / sizes.sum() * len(X)).astype(int)
    
    # Ensure minimum size per device
    min_size = min_samples_per_class * num_classes * 2
    sizes = np.maximum(sizes, min_size)
    
    # Adjust last device to use all data
    if sizes.sum() < len(X):
        sizes[-1] += len(X) - sizes.sum()
    elif sizes.sum() > len(X):
        # Reduce sizes proportionally
        sizes = (sizes / sizes.sum() * len(X)).astype(int)
        sizes[-1] = len(X) - sizes[:-1].sum()
    
    # Create device partitions
    for i in range(num_devices):
        if heterogeneity == 'high':
            if num_classes == 2:
                # Skewed but ensure minimum from each class
                class_probs = np.random.dirichlet([0.5, 2])
                class_probs = np.maximum(class_probs, 0.2)  # At least 20% from each
                class_probs = class_probs / class_probs.sum()
            else:
                class_probs = np.random.dirichlet([0.5] * num_classes)
        elif heterogeneity == 'medium':
            if num_classes == 2:
                class_probs = np.random.dirichlet([1, 1.5])
                class_probs = np.maximum(class_probs, 0.3)
                class_probs = class_probs / class_probs.sum()
            else:
                class_probs = np.random.dirichlet([1.0] * num_classes)
        else:
            class_probs = np.ones(num_classes) / num_classes
        
        device_indices = []
        target_size = sizes[i]
        
        # First, ensure minimum samples from each class
        for class_id in class_indices:
            if len(class_indices[class_id]) >= min_samples_per_class:
                min_samples = np.random.choice(
                    class_indices[class_id],
                    size=min_samples_per_class,
                    replace=False
                )
                device_indices.extend(min_samples.tolist())
                class_indices[class_id] = [
                    idx for idx in class_indices[class_id] 
                    if idx not in min_samples
                ]
        
        # Then distribute remaining according to probabilities
        remaining_size = target_size - len(device_indices)
        
        for class_id, prob in enumerate(class_probs):
            if class_id not in class_indices or len(class_indices[class_id]) == 0:
                continue
                
            n_samples = int(remaining_size * prob)
            n_samples = min(n_samples, len(class_indices[class_id]))
            
            if n_samples > 0:
                sampled_indices = np.random.choice(
                    class_indices[class_id],
                    size=n_samples,
                    replace=False
                )
                device_indices.extend(sampled_indices.tolist())
                
                class_indices[class_id] = [
                    idx for idx in class_indices[class_id] 
                    if idx not in sampled_indices
                ]
        
        # Add any remaining needed samples
        if len(device_indices) < target_size:
            remaining = []
            for class_id in class_indices:
                remaining.extend(class_indices[class_id])
            
            if remaining:
                shortage = min(target_size - len(device_indices), len(remaining))
                extra = np.random.choice(remaining, size=shortage, replace=False)
                device_indices.extend(extra.tolist())
                
                for class_id in class_indices:
                    class_indices[class_id] = [
                        idx for idx in class_indices[class_id] 
                        if idx not in extra
                    ]
        
        if len(device_indices) > 0:
            device_indices = np.array(device_indices)
            device_X = X[device_indices]
            device_y = y[device_indices]
            
            # Verify we have both classes
            unique_labels = np.unique(device_y)
            if len(unique_labels) >= num_classes:
                device_data.append((device_X, device_y))
            else:
                # Skip devices with only one class
                # Return indices to pool
                for idx in device_indices:
                    label = int(y[idx])
                    class_indices[label].append(idx)
    
    # Handle remaining data - distribute to existing devices
    remaining_indices = []
    for class_id in class_indices:
        remaining_indices.extend(class_indices[class_id])
    
    if remaining_indices and len(device_data) > 0:
        # Distribute evenly across devices
        for idx in remaining_indices:
            device_idx = np.random.randint(0, len(device_data))
            old_X, old_y = device_data[device_idx]
            new_X = np.concatenate([old_X, X[idx:idx+1]])
            new_y = np.concatenate([old_y, y[idx:idx+1]])
            device_data[device_idx] = (new_X, new_y)
    
    return device_data