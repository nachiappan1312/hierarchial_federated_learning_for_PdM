import sys
sys.path.append('..')

import argparse
import numpy as np
import torch
from data.preprocess import load_cmapss_data, partition_data
from federated.cloud_server import CloudServer
from federated.edge_server import EdgeGateway
from federated.client import IoTDevice
from training.federated_trainer import FederatedTrainer
from experiments.baselines import CentralizedTrainer, StandardFedAvg
from models.full_model import FullCNNLSTM
from utils.visualization import ResultsVisualizer
import json

def run_full_experiment(args):
    """Run complete experiment with all baselines"""
    
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_cmapss_data('FD003')
    
    print("Creating data partitions...")
    device_data = partition_data(
        X_train, y_train,
        num_devices=args.devices,
        heterogeneity=args.heterogeneity
    )
    
    model_config = {'input_channels': X_train.shape[2], 'num_classes': 2}
    results_dict = {}
    
    # 1. Proposed Method
    print("\n" + "="*70)
    print("RUNNING: Proposed Hierarchical FL")
    print("="*70)
    
    cloud_server = CloudServer(model_config)
    edge_gateways = [EdgeGateway(i) for i in range(args.gateways)]
    
    devices = []
    for i, (data, labels) in enumerate(device_data):
        device = IoTDevice(
            device_id=i,
            data=data,
            labels=labels,
            capability_score=np.random.beta(2, 5),
            gateway_id=i % args.gateways
        )
        devices.append(device)
    
    trainer = FederatedTrainer(cloud_server, edge_gateways, devices, X_test, y_test)
    proposed_results = trainer.train(num_rounds=args.rounds, local_epochs=args.epochs)
    results_dict['Proposed (Hierarchical)'] = proposed_results
    
    # 2. Standard FedAvg
    print("\n" + "="*70)
    print("RUNNING: Standard FedAvg Baseline")
    print("="*70)
    
    cloud_server_fedavg = CloudServer(model_config)
    fedavg_trainer = StandardFedAvg(cloud_server_fedavg, devices, X_test, y_test)
    fedavg_results = fedavg_trainer.train(num_rounds=args.rounds, local_epochs=args.epochs, verbose=False)
    results_dict['FedAvg'] = fedavg_results
    
    # 3. Centralized (if requested)
    if args.centralized:
        print("\n" + "="*70)
        print("RUNNING: Centralized Baseline")
        print("="*70)
        
        centralized_model = FullCNNLSTM(**model_config)
        centralized_trainer = CentralizedTrainer(
            centralized_model, X_train, y_train, X_test, y_test
        )
        centralized_results = centralized_trainer.train(epochs=args.rounds * args.epochs // 5)
        
        # Convert to comparable format
        centralized_results['round'] = [i*5 for i in centralized_results['epoch']]
        centralized_results['accuracy'] = centralized_results['test_accuracy']
        centralized_results['communication_mb'] = [0] * len(centralized_results['epoch'])
        results_dict['Centralized'] = centralized_results
    
    # Save results
    print("\nSaving results...")
    with open('results/experiment_results.json', 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        serializable_results = {}
        for method, results in results_dict.items():
            serializable_results[method] = {
                k: [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
                for k, v in results.items()
            }
        json.dump(serializable_results, f, indent=2)
    
    # Generate visualizations
    print("Generating visualizations...")
    visualizer = ResultsVisualizer()
    visualizer.plot_training_curves(results_dict, save_path='results/comparison.png')
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    for method_name, results in results_dict.items():
        print(f"\n{method_name}:")
        print(f"  Final Accuracy: {results['accuracy'][-1]:.4f}")
        if 'f1_score' in results:
            print(f"  Final F1-Score: {results['f1_score'][-1]:.4f}")
        if 'communication_mb' in results and sum(results['communication_mb']) > 0:
            print(f"  Total Communication: {sum(results['communication_mb']):.2f} MB")
    
    print("\nResults saved to results/experiment_results.json")
    print("Visualizations saved to results/comparison.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run federated learning experiments')
    parser.add_argument('--rounds', type=int, default=100, help='Number of training rounds')
    parser.add_argument('--epochs', type=int, default=5, help='Local epochs per round')
    parser.add_argument('--devices', type=int, default=50, help='Number of IoT devices')
    parser.add_argument('--gateways', type=int, default=10, help='Number of edge gateways')
    parser.add_argument('--heterogeneity', type=str, default='high', 
                       choices=['low', 'medium', 'high'], help='Data heterogeneity level')
    parser.add_argument('--centralized', action='store_true', help='Include centralized baseline')
    
    args = parser.parse_args()
    run_full_experiment(args)