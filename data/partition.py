import numpy as np
from collections import defaultdict

def partition_data(X, y, num_devices=50, heterogeneity='high'):
    """
    Create non-IID partitions simulating heterogeneous IoT deployment
    
    Args:
        X: Feature array (samples, timesteps, features)
        y: Labels
        num_devices: Number of virtual devices
        heterogeneity: 'low', 'medium', or 'high'
    """
    device_data = []
    
    # Strategy 1: Partition by label distribution (class imbalance)
    class_indices = defaultdict(list)
    for idx, label in enumerate(y):
        class_indices[label].append(idx)
    
    # Strategy 2: Create different sample sizes (quantity heterogeneity)
    if heterogeneity == 'high':
        sizes = np.random.lognormal(mean=6, sigma=1, size=num_devices)
    elif heterogeneity == 'medium':
        sizes = np.random.lognormal(mean=6, sigma=0.5, size=num_devices)
    else:
        sizes = np.ones(num_devices) * (len(X) / num_devices)
    
    sizes = (sizes / sizes.sum() * len(X)).astype(int)
    
    # Strategy 3: Non-IID class distribution
    for i in range(num_devices):
        # Create skewed class distribution
        if heterogeneity == 'high':
            class_probs = np.random.dirichlet([0.5, 2])  # Highly skewed
        elif heterogeneity == 'medium':
            class_probs = np.random.dirichlet([1, 1.5])
        else:
            class_probs = np.array([0.5, 0.5])
        
        device_indices = []
        target_size = sizes[i]
        
        for class_id, prob in enumerate(class_probs):
            n_samples = int(target_size * prob)
            if len(class_indices[class_id]) < n_samples:
                n_samples = len(class_indices[class_id])
            
            sampled = np.random.choice(
                class_indices[class_id],
                size=n_samples,
                replace=False
            )
            device_indices.extend(sampled)
            
            # Remove sampled indices
            class_indices[class_id] = [
                idx for idx in class_indices[class_id] 
                if idx not in sampled
            ]
        
        device_X = X[device_indices]
        device_y = y[device_indices]
        
        device_data.append((device_X, device_y))
    
    return device_data