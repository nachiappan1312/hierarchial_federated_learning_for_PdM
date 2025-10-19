from collections import defaultdict

class FederatedTrainer:
    """Orchestrates the complete federated training process"""
    
    def __init__(self, cloud_server, edge_gateways, devices, X_test, y_test):
        self.cloud_server = cloud_server
        self.edge_gateways = edge_gateways
        self.devices = devices
        self.X_test = X_test
        self.y_test = y_test
        
        # Assign devices to gateways
        for device in devices:
            gateway_id = device.gateway_id
            self.edge_gateways[gateway_id].register_device(device)
    
    def train(self, num_rounds=100, local_epochs=5, verbose=True):
        """
        Execute federated training
        
        Args:
            num_rounds: Number of federated rounds
            local_epochs: Number of local training epochs per device
            verbose: Print progress
        """
        results = {
            'round': [],
            'accuracy': [],
            'f1_score': [],
            'communication_mb': [],
            'train_time': []
        }
        
        for round_num in range(num_rounds):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Round {round_num + 1}/{num_rounds}")
                print(f"{'='*60}")
            
            import time
            start_time = time.time()
            
            # Phase 1: Device-level training
            print("Phase 1: Device-level local training...")
            device_updates_by_gateway = defaultdict(list)
            
            global_weights = self.cloud_server.get_global_weights()
            
            for device in self.devices:
                update = device.train_local_model(
                    global_weights=global_weights,
                    epochs=local_epochs
                )
                device_updates_by_gateway[device.gateway_id].append(update)
            
            # Phase 2: Edge-level aggregation
            print("Phase 2: Edge gateway aggregation...")
            edge_updates = []
            
            for gateway in self.edge_gateways:
                updates = device_updates_by_gateway[gateway.gateway_id]
                if updates:
                    aggregated = gateway.aggregate_device_updates(updates)
                    if aggregated:
                        edge_updates.append(aggregated)
            
            # Phase 3: Cloud-level aggregation
            print("Phase 3: Cloud server global aggregation...")
            self.cloud_server.aggregate_edge_updates(edge_updates)
            
            # Evaluation
            print("Evaluating global model...")
            metrics = self.cloud_server.evaluate(self.X_test, self.y_test)
            
            train_time = time.time() - start_time
            
            # Record results
            results['round'].append(round_num + 1)
            results['accuracy'].append(metrics['accuracy'])
            results['f1_score'].append(metrics['f1_score'])
            results['communication_mb'].append(
                self.cloud_server.training_history['communication_cost'][-1]
            )
            results['train_time'].append(train_time)
            
            if verbose:
                print(f"\nResults:")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  F1-Score: {metrics['f1_score']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  Communication: {results['communication_mb'][-1]:.2f} MB")
                print(f"  Time: {train_time:.2f}s")
        
        return results