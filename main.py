#!/usr/bin/env python3
"""
Main entry point for Hierarchical Federated Learning for Predictive Maintenance
"""

import sys
import os
import numpy as np
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import configurations
from utils.config import Config
from utils.metrics import MetricsCalculator
from utils.visualization import ResultsVisualizer

# Import data handling
from data.preprocess import load_cmapss_data
from data.partition import partition_data

# Import federated components
from federated.cloud_server import CloudServer
from federated.edge_server import EdgeGateway
from federated.client import IoTDevice

# Import training
from training.federated_trainer import FederatedTrainer
from training.evaluation import ModelEvaluator

# Import baselines
from experiments.baselines import CentralizedTrainer, StandardFedAvg


def setup_experiment(config: Config):
    """Setup experiment with given configuration"""
    
    # Set random seeds for reproducibility
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)
    
    # Set device
    if config.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        config.device = 'cpu'
    
    print(f"Using device: {config.device}")
    
    return config


def load_data(config: Config):
    """Load and preprocess data"""
    
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    try:
        # Try to load real NASA C-MAPSS data
        X_train, y_train, X_test, y_test = load_cmapss_data(
            dataset=config.dataset,
            data_dir=config.data_dir
        )
        print(f"✓ Loaded NASA C-MAPSS {config.dataset} dataset")
        data_source = "NASA C-MAPSS"
        
    except Exception as e:
        print(f"Could not load NASA C-MAPSS data: {e}")
        print("Generating synthetic data for demonstration...")
        
        # Generate synthetic data
        np.random.seed(config.random_seed)
        n_train = 10000
        n_test = 2000
        
        X_train = np.random.randn(n_train, config.window_size, config.input_channels)
        y_train = np.random.randint(0, config.num_classes, n_train)
        X_test = np.random.randn(n_test, config.window_size, config.input_channels)
        y_test = np.random.randint(0, config.num_classes, n_test)
        
        print("✓ Generated synthetic data")
        data_source = "Synthetic"
    
    print(f"\nData shapes:")
    print(f"  Training:   {X_train.shape}")
    print(f"  Testing:    {X_test.shape}")
    print(f"  Features:   {config.input_channels}")
    print(f"  Classes:    {config.num_classes}")
    print(f"  Source:     {data_source}")
    
    return X_train, y_train, X_test, y_test, data_source


def create_federated_setup(X_train, y_train, config: Config):
    """Create federated learning setup"""
    
    print("\n" + "="*70)
    print("CREATING FEDERATED SETUP")
    print("="*70)
    
    # Partition data for devices
    print(f"\nPartitioning data for {config.num_devices} devices...")
    device_data = partition_data(
        X_train, y_train,
        num_devices=config.num_devices,
        heterogeneity=config.heterogeneity_level
    )
    print("✓ Data partitioned")
    
    # Create cloud server
    print("\nInitializing cloud server...")
    model_config = {
        'input_channels': config.input_channels,
        'num_classes': config.num_classes
    }
    cloud_server = CloudServer(model_config, device=config.device)
    print(f"✓ Cloud server initialized")
    print(f"  Global model parameters: {sum(p.numel() for p in cloud_server.global_model.parameters()):,}")
    
    # Create edge gateways
    print(f"\nCreating {config.num_gateways} edge gateways...")
    edge_gateways = [
        EdgeGateway(gateway_id=i, outlier_threshold=config.outlier_threshold)
        for i in range(config.num_gateways)
    ]
    print("✓ Edge gateways created")
    
    # Create IoT devices
    print(f"\nCreating {config.num_devices} IoT devices...")
    devices = []
    capability_scores = []
    
    for i, (data, labels) in enumerate(device_data):
        # Assign capability score based on distribution
        if config.capability_distribution == 'beta':
            capability_score = np.random.beta(2, 5)
        elif config.capability_distribution == 'uniform':
            capability_score = np.random.uniform(0, 1)
        else:
            capability_score = 0.5
        
        capability_scores.append(capability_score)
        
        device = IoTDevice(
            device_id=i,
            data=data,
            labels=labels,
            capability_score=capability_score,
            gateway_id=i % config.num_gateways
        )
        
        # Apply configuration settings
        if not config.enable_adaptive_pruning:
            device.pruner.p_max = 0.0
            device.pruner.p_min = 0.0
        
        devices.append(device)
    
    print("✓ IoT devices created")
    
    # Print device statistics
    print(f"\nDevice statistics:")
    print(f"  Lightweight ({config.lightweight_threshold:.1f}): "
          f"{sum(1 for s in capability_scores if s < config.lightweight_threshold)} devices")
    print(f"  Medium ({config.medium_threshold:.1f}): "
          f"{sum(1 for s in capability_scores if config.lightweight_threshold <= s < config.medium_threshold)} devices")
    print(f"  Full capacity: "
          f"{sum(1 for s in capability_scores if s >= config.medium_threshold)} devices")
    
    # Data distribution statistics
    sample_counts = [len(d[0]) for d in device_data]
    print(f"\nData distribution:")
    print(f"  Min samples:     {min(sample_counts)}")
    print(f"  Max samples:     {max(sample_counts)}")
    print(f"  Mean samples:    {np.mean(sample_counts):.1f}")
    print(f"  Std samples:     {np.std(sample_counts):.1f}")
    
    return cloud_server, edge_gateways, devices


def run_proposed_method(cloud_server, edge_gateways, devices, X_test, y_test, config):
    """Run the proposed hierarchical federated learning method"""
    
    print("\n" + "="*70)
    print("TRAINING: PROPOSED HIERARCHICAL FL METHOD")
    print("="*70)
    
    trainer = FederatedTrainer(
        cloud_server=cloud_server,
        edge_gateways=edge_gateways,
        devices=devices,
        X_test=X_test,
        y_test=y_test
    )
    
    results = trainer.train(
        num_rounds=config.num_rounds,
        local_epochs=config.local_epochs,
        verbose=config.verbose
    )
    
    return results


def run_baselines(devices, X_train, y_train, X_test, y_test, config):
    """Run baseline methods for comparison"""
    
    baseline_results = {}
    
    if not config.run_baselines:
        print("\nSkipping baselines (disabled in config)")
        return baseline_results
    
    # Standard FedAvg
    print("\n" + "="*70)
    print("TRAINING: STANDARD FEDAVG BASELINE")
    print("="*70)
    
    from models.full_model import FullCNNLSTM
    model_config = {
        'input_channels': config.input_channels,
        'num_classes': config.num_classes
    }
    
    cloud_server_baseline = CloudServer(model_config, device=config.device)
    baseline_trainer = StandardFedAvg(
        cloud_server=cloud_server_baseline,
        devices=devices,
        X_test=X_test,
        y_test=y_test
    )
    
    fedavg_results = baseline_trainer.train(
        num_rounds=config.num_rounds,
        local_epochs=config.local_epochs,
        verbose=False
    )
    baseline_results['FedAvg'] = fedavg_results
    
    print(f"\n✓ FedAvg completed")
    print(f"  Final Accuracy: {fedavg_results['accuracy'][-1]:.4f}")
    print(f"  Final F1-Score: {fedavg_results['f1_score'][-1]:.4f}")
    
    # Centralized (optional - can be slow)
    if config.run_ablation:
        print("\n" + "="*70)
        print("TRAINING: CENTRALIZED BASELINE")
        print("="*70)
        
        centralized_model = FullCNNLSTM(**model_config)
        centralized_trainer = CentralizedTrainer(
            centralized_model, X_train, y_train, X_test, y_test
        )
        
        # Train for equivalent number of epochs
        total_epochs = config.num_rounds * config.local_epochs // 5
        centralized_results = centralized_trainer.train(
            epochs=total_epochs,
            batch_size=config.batch_size,
            lr=config.learning_rate,
            verbose=False
        )
        
        # Convert to comparable format
        centralized_results['round'] = [i*5 for i in centralized_results['epoch']]
        centralized_results['accuracy'] = centralized_results['test_accuracy']
        centralized_results['f1_score'] = [0.0] * len(centralized_results['epoch'])
        centralized_results['communication_mb'] = [0.0] * len(centralized_results['epoch'])
        
        baseline_results['Centralized'] = centralized_results
        
        print(f"\n✓ Centralized completed")
        print(f"  Final Accuracy: {centralized_results['accuracy'][-1]:.4f}")
    
    return baseline_results


def evaluate_models(cloud_server, X_test, y_test, config):
    """Comprehensive model evaluation"""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE EVALUATION")
    print("="*70)
    
    evaluator = ModelEvaluator(cloud_server.global_model, device=config.device)
    
    # Get detailed metrics
    metrics = evaluator.evaluate(X_test, y_test, return_predictions=True)
    
    print(f"\nDetailed Metrics:")
    print(f"  Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  Precision:       {metrics['precision']:.4f}")
    print(f"  Recall:          {metrics['recall']:.4f}")
    print(f"  F1-Score:        {metrics['f1_score']:.4f}")
    
    if metrics.get('auc_roc'):
        print(f"  AUC-ROC:         {metrics['auc_roc']:.4f}")
    
    # Per-class metrics
    print(f"\nPer-Class Metrics:")
    for i, (p, r, f) in enumerate(zip(
        metrics['precision_per_class'],
        metrics['recall_per_class'],
        metrics['f1_per_class']
    )):
        print(f"  Class {i}: Precision={p:.4f}, Recall={r:.4f}, F1={f:.4f}")
    
    # Classification report
    print(f"\nClassification Report:")
    report = evaluator.get_classification_report(
        X_test, y_test,
        target_names=['Normal', 'Degrading'] if config.num_classes == 2 else None
    )
    print(report)
    
    # Save confusion matrix
    if config.save_plots:
        cm_path = os.path.join(config.results_dir, 'confusion_matrix.png')
        evaluator.plot_confusion_matrix(
            X_test, y_test,
            target_names=['Normal', 'Degrading'] if config.num_classes == 2 else None,
            save_path=cm_path
        )
    
    return metrics


def calculate_additional_metrics(proposed_results, baseline_results, devices, config):
    """Calculate additional metrics for analysis"""
    
    print("\n" + "="*70)
    print("CALCULATING ADDITIONAL METRICS")
    print("="*70)
    
    metrics_calc = MetricsCalculator()
    additional_metrics = {}
    
    # Fairness analysis
    device_accuracies = []
    for device in devices:
        if len(device.X) > 0:
            # Simple validation accuracy
            from training.local_trainer import LocalTrainer
            trainer = LocalTrainer(device.model)
            _, acc = trainer.validate(device.X.numpy(), device.y.numpy())
            device_accuracies.append(acc)
    
    if device_accuracies:
        fairness_index = metrics_calc.calculate_jain_fairness_index(device_accuracies)
        additional_metrics['fairness'] = {
            'jain_index': fairness_index,
            'mean_accuracy': np.mean(device_accuracies),
            'std_accuracy': np.std(device_accuracies),
            'min_accuracy': np.min(device_accuracies),
            'max_accuracy': np.max(device_accuracies)
        }
        
        print(f"\nFairness Metrics:")
        print(f"  Jain's Index:     {fairness_index:.4f}")
        print(f"  Mean Accuracy:    {np.mean(device_accuracies):.4f}")
        print(f"  Std Accuracy:     {np.std(device_accuracies):.4f}")
        print(f"  Min Accuracy:     {np.min(device_accuracies):.4f}")
        print(f"  Max Accuracy:     {np.max(device_accuracies):.4f}")
    
    # Communication cost analysis
    total_params = sum(p.numel() for p in devices[0].model.parameters())
    
    comm_proposed = metrics_calc.calculate_communication_cost(
        model_params=total_params,
        num_devices=config.num_devices,
        num_rounds=config.num_rounds,
        compression_ratio=config.compression_ratio if config.enable_compression else 1.0
    )
    
    comm_fedavg = metrics_calc.calculate_communication_cost(
        model_params=total_params,
        num_devices=config.num_devices,
        num_rounds=config.num_rounds,
        compression_ratio=1.0
    )
    
    additional_metrics['communication'] = {
        'proposed': comm_proposed,
        'fedavg': comm_fedavg,
        'reduction_percent': (1 - comm_proposed['total_mb'] / comm_fedavg['total_mb']) * 100
    }
    
    print(f"\nCommunication Analysis:")
    print(f"  Proposed Total:   {comm_proposed['total_mb']:.2f} MB")
    print(f"  FedAvg Total:     {comm_fedavg['total_mb']:.2f} MB")
    print(f"  Reduction:        {additional_metrics['communication']['reduction_percent']:.1f}%")
    
    # Convergence analysis
    if proposed_results:
        convergence = metrics_calc.calculate_convergence_metrics(
            proposed_results['accuracy'],
            threshold=0.9
        )
        additional_metrics['convergence'] = convergence
        
        print(f"\nConvergence Analysis:")
        print(f"  Converged:        {convergence['converged']}")
        if convergence['converged']:
            print(f"  Round:            {convergence['convergence_round']}")
        print(f"  Final Accuracy:   {convergence['final_accuracy']:.4f}")
    
    return additional_metrics


def visualize_results(proposed_results, baseline_results, additional_metrics, config):
    """Generate visualizations"""
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    if not config.save_plots:
        print("Plotting disabled in config")
        return
    
    visualizer = ResultsVisualizer()
    
    # Combine all results
    all_results = {'Proposed (Hierarchical)': proposed_results}
    all_results.update(baseline_results)
    
    # Main comparison plot
    plot_path = os.path.join(config.results_dir, 'training_curves.png')
    visualizer.plot_training_curves(all_results, save_path=plot_path)
    print(f"✓ Saved training curves to {plot_path}")
    
    # Additional plots can be added here
    
    return


def save_results(proposed_results, baseline_results, additional_metrics, config, data_source):
    """Save all results to files"""
    
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    import json
    from datetime import datetime
    
    # Create results dictionary
    results_dict = {
        'timestamp': datetime.now().isoformat(),
        'config': config.to_dict(),
        'data_source': data_source,
        'proposed_results': proposed_results,
        'baseline_results': baseline_results,
        'additional_metrics': additional_metrics
    }
    
    # Convert numpy types to native Python types
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    results_dict = convert_numpy(results_dict)
    
    # Save to JSON
    results_path = os.path.join(config.results_dir, 'experiment_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"✓ Results saved to {results_path}")
    
    # Save configuration
    config_path = os.path.join(config.results_dir, 'config.json')
    config.save(config_path)
    
    return results_path


def print_final_summary(proposed_results, baseline_results, additional_metrics):
    """Print final summary of results"""
    
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    
    # Proposed method
    print("\nProposed Method (Hierarchical FL):")
    print(f"  Final Accuracy:        {proposed_results['accuracy'][-1]:.4f}")
    print(f"  Final F1-Score:        {proposed_results['f1_score'][-1]:.4f}")
    
    if 'communication' in additional_metrics:
        comm = additional_metrics['communication']
        print(f"  Total Communication:   {comm['proposed']['total_mb']:.2f} MB")
        print(f"  Communication Reduction: {comm['reduction_percent']:.1f}%")
    
    if 'fairness' in additional_metrics:
        print(f"  Jain's Fairness Index: {additional_metrics['fairness']['jain_index']:.4f}")
    
    if 'convergence' in additional_metrics:
        conv = additional_metrics['convergence']
        if conv['converged']:
            print(f"  Convergence Round:     {conv['convergence_round']}")
    
    # Baselines comparison
    if baseline_results:
        print("\nBaseline Comparison:")
        for method_name, results in baseline_results.items():
            print(f"\n  {method_name}:")
            print(f"    Final Accuracy:      {results['accuracy'][-1]:.4f}")
            if 'f1_score' in results and results['f1_score']:
                print(f"    Final F1-Score:      {results['f1_score'][-1]:.4f}")
            
            # Calculate improvement
            if proposed_results:
                improvement = (proposed_results['accuracy'][-1] - results['accuracy'][-1]) * 100
                print(f"    Improvement:         {improvement:+.2f}%")
    
    print("\n" + "="*70)


def main():
    """Main execution function"""
    
    print("="*70)
    print("HIERARCHICAL FEDERATED LEARNING FOR PREDICTIVE MAINTENANCE")
    print("="*70)
    
    # Load configuration
    config = Config()
    
    # Parse command line arguments (optional)
    import argparse
    parser = argparse.ArgumentParser(description='Federated Learning for Predictive Maintenance')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--rounds', type=int, help='Number of rounds')
    parser.add_argument('--devices', type=int, help='Number of devices')
    parser.add_argument('--no-baselines', action='store_true', help='Skip baseline methods')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    
    args = parser.parse_args()
    
    # Load config from file if specified
    if args.config:
        config = Config.load(args.config)
    
    # Override with command line arguments
    if args.rounds:
        config.num_rounds = args.rounds
    if args.devices:
        config.num_devices = args.devices
        config.num_gateways = max(2, args.devices // 5)
    if args.no_baselines:
        config.run_baselines = False
    
    # Quick test mode
    if args.quick:
        config.num_rounds = 10
        config.num_devices = 10
        config.num_gateways = 2
        config.run_baselines = False
        print("\n⚡ QUICK TEST MODE ⚡")
    
    # Print configuration
    config.print_config()
    
    # Setup experiment
    config = setup_experiment(config)
    
    # Load data
    X_train, y_train, X_test, y_test, data_source = load_data(config)
    
    # Create federated setup
    cloud_server, edge_gateways, devices = create_federated_setup(
        X_train, y_train, config
    )
    
    # Run proposed method
    proposed_results = run_proposed_method(
        cloud_server, edge_gateways, devices,
        X_test, y_test, config
    )
    
    # Run baselines
    baseline_results = run_baselines(
        devices, X_train, y_train, X_test, y_test, config
    )
    
    # Comprehensive evaluation
    final_metrics = evaluate_models(cloud_server, X_test, y_test, config)
    
    # Calculate additional metrics
    additional_metrics = calculate_additional_metrics(
        proposed_results, baseline_results, devices, config
    )
    
    # Visualize results
    visualize_results(proposed_results, baseline_results, additional_metrics, config)
    
    # Save results
    results_path = save_results(
        proposed_results, baseline_results, additional_metrics, config, data_source
    )
    
    # Print final summary
    print_final_summary(proposed_results, baseline_results, additional_metrics)
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nResults saved to: {results_path}")
    print(f"Visualizations in: {config.results_dir}/")
    print("\n✓ All done! 🎉")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)