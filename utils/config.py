import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class Config:
    """Central configuration for the federated learning system"""
    
    # ========== Data Settings ==========
    dataset: str = 'FD003'
    window_size: int = 30
    stride: int = 15
    degradation_threshold: int = 125
    data_dir: str = 'data/raw'
    
    # ========== Federated Learning Settings ==========
    num_rounds: int = 100
    local_epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.003
    client_participation_rate: float = 1.0
    
    # ========== Architecture Settings ==========
    num_devices: int = 50
    num_gateways: int = 10
    devices_per_gateway: int = 5
    
    # ========== Model Settings ==========
    input_channels: int = 14
    num_classes: int = 2
    
    # ========== Heterogeneity Settings ==========
    heterogeneity_level: str = 'high'  # 'low', 'medium', 'high'
    capability_distribution: str = 'beta'  # 'beta', 'uniform', 'fixed'
    
    # ========== Adaptive Pruning Settings ==========
    enable_adaptive_pruning: bool = False
    p_max: float = 0.6
    p_min: float = 0.1
    pruning_method: str = 'l1_unstructured'  # 'l1_unstructured', 'structured'
    
    # ========== Priority Weighting Settings ==========
    enable_priority_weighting: bool = True
    rul_threshold: int = 50
    priority_strength: float = 2.0
    
    # ========== Edge Aggregation Settings ==========
    enable_edge_aggregation: bool = True
    outlier_threshold: float = 2.5
    quality_alpha: float = 1.5
    quality_beta: float = 2.0
    enable_compression: bool = True
    compression_ratio: float = 0.3
    
    # ========== Device Capability Settings ==========
    lightweight_threshold: float = 0.4
    medium_threshold: float = 0.7
    
    # ========== Training Settings ==========
    device: str = 'cpu'  # 'cpu', 'cuda', 'mps'
    random_seed: int = 42
    validation_split: float = 0.2
    
    # ========== Logging Settings ==========
    results_dir: str = 'results'
    logs_dir: str = 'logs'
    save_models: bool = True
    log_interval: int = 10
    verbose: bool = True
    
    # ========== Experiment Settings ==========
    run_ablation: bool = False
    run_baselines: bool = True
    save_plots: bool = True
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    
    def save(self, path: str):
        """Save configuration to file"""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load configuration from file"""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def update(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown config parameter '{key}'")
    
    def print_config(self):
        """Print current configuration"""
        print("="*60)
        print("CONFIGURATION")
        print("="*60)
        
        sections = {
            'Data': ['dataset', 'window_size', 'stride', 'degradation_threshold'],
            'Federated Learning': ['num_rounds', 'local_epochs', 'batch_size', 'learning_rate'],
            'Architecture': ['num_devices', 'num_gateways', 'devices_per_gateway'],
            'Model': ['input_channels', 'num_classes'],
            'Adaptive Pruning': ['enable_adaptive_pruning', 'p_max', 'p_min'],
            'Priority Weighting': ['enable_priority_weighting', 'rul_threshold', 'priority_strength'],
            'Edge Aggregation': ['enable_edge_aggregation', 'outlier_threshold', 'enable_compression'],
        }
        
        for section, keys in sections.items():
            print(f"\n{section}:")
            for key in keys:
                if hasattr(self, key):
                    value = getattr(self, key)
                    print(f"  {key:30} {value}")
        
        print("="*60)


# Create default configuration instance
default_config = Config()
