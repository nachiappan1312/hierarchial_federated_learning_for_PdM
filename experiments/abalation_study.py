import sys
sys.path.append('..')

import numpy as np
from data.preprocess import load_cmapss_data, partition_data
from federated.cloud_server import CloudServer
from federated.edge_server import EdgeGateway
from federated.client import IoTDevice
from training.federated_trainer import FederatedTrainer
from utils.visualization import ResultsVisualizer

def ablation_study():
    """Run ablation study to isolate component contributions"""
    
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_cmapss_data('FD003')
    device_data = partition_data(X_train, y_train, num_devices=30)
    
    model_config = {'input_channels': X_train.shape[2], 'num_classes': 2}
    results_dict = {}
    
    configurations = [
        {
            'name': 'Full Method',
            'edge_aggregation': True,
            'adaptive_pruning': True,
            'priority_weighting': True
        },
        {
            'name': 'No Adaptive Pruning',
            'edge_aggregation': True,
            'adaptive_pruning': False,
            'priority_weighting': True
        },
        {
            'name': 'No Edge Aggregation',
            'edge_aggregation': False,
            'adaptive_pruning': True,
            'priority_weighting': True
        },
        {
            'name': 'No Priority Weighting',
            'edge_aggregation': True,
            'adaptive_pruning': True,
            'priority_weighting': False
        }
    ]
    
    for config in configurations:
        print(f"\nRunning: {config['name']}")
        
        # Initialize with configuration
        cloud_server = CloudServer(model_config)
        edge_gateways = [EdgeGateway(i) for i in range(6)] if config['edge_aggregation'] else []
        
        devices = []
        for i, (data, labels) in enumerate(device_data):
            device = IoTDevice(
                device_id=i,
                data=data,
                labels=labels,
                capability_score=np.random.beta(2, 5),
                gateway_id=i % 6 if config['edge_aggregation'] else 0
            )
            # Modify device to disable pruning if needed
            if not config['adaptive_pruning']:
                device.pruner.p_max = 0.0
                device.pruner.p_min = 0.0
            
            devices.append(device)
        
        # Modify cloud server for priority weighting
        if not config['priority_weighting']:
            # Override aggregate method to ignore priority weights
            original_aggregate = cloud_server.aggregate_edge_updates
            def no_priority_aggregate(edge_updates):
                for update in edge_updates:
                    update['priority_weight'] = 1.0
                return original_aggregate(edge_updates)
            cloud_server.aggregate_edge_updates = no_priority_aggregate
        
        trainer = FederatedTrainer(cloud_server, edge_gateways, devices, X_test, y_test)
        results = trainer.train(num_rounds=50, local_epochs=5, verbose=False)
        results_dict[config['name']] = results
    
    # Visualize
    visualizer = ResultsVisualizer()
    visualizer.plot_training_curves(results_dict, save_path='results/ablation_study.png')
    
    print("\nAblation study complete. Results saved to results/ablation_study.png")

if __name__ == "__main__":
    ablation_study()