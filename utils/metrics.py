import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class MetricsCalculator:
    """Calculate various metrics for federated learning"""
    
    @staticmethod
    def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy"""
        return accuracy_score(y_true, y_pred)
    
    @staticmethod
    def calculate_per_class_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """Calculate per-class precision, recall, f1-score"""
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        return {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1_score': f1.tolist(),
            'support': support.tolist()
        }
    
    @staticmethod
    def calculate_jain_fairness_index(values: List[float]) -> float:
        """
        Calculate Jain's fairness index
        
        Args:
            values: List of values (e.g., per-device accuracies)
            
        Returns:
            Fairness index between 0 and 1 (1 = perfectly fair)
        """
        if not values or len(values) == 0:
            return 0.0
        
        values = np.array(values)
        numerator = np.sum(values) ** 2
        denominator = len(values) * np.sum(values ** 2)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    @staticmethod
    def calculate_communication_cost(
        model_params: int,
        num_devices: int,
        num_rounds: int,
        compression_ratio: float = 1.0
    ) -> Dict:
        """
        Calculate communication cost
        
        Args:
            model_params: Number of model parameters
            num_devices: Number of devices
            num_rounds: Number of training rounds
            compression_ratio: Compression ratio (1.0 = no compression)
            
        Returns:
            Dictionary with communication metrics
        """
        # Assume 4 bytes per float32 parameter
        bytes_per_param = 4
        
        # Upload + Download per round per device
        total_bytes = model_params * bytes_per_param * num_devices * num_rounds * 2
        total_bytes *= compression_ratio
        
        total_mb = total_bytes / (1024 ** 2)
        total_gb = total_bytes / (1024 ** 3)
        
        return {
            'total_bytes': total_bytes,
            'total_mb': total_mb,
            'total_gb': total_gb,
            'per_round_mb': total_mb / num_rounds,
            'per_device_mb': total_mb / num_devices
        }
    
    @staticmethod
    def calculate_convergence_metrics(
        accuracy_history: List[float],
        threshold: float = 0.9
    ) -> Dict:
        """
        Calculate convergence metrics
        
        Args:
            accuracy_history: List of accuracies over rounds
            threshold: Convergence threshold (e.g., 0.9 = 90% of final accuracy)
            
        Returns:
            Dictionary with convergence metrics
        """
        if not accuracy_history:
            return {'converged': False}
        
        final_accuracy = accuracy_history[-1]
        target_accuracy = final_accuracy * threshold
        
        # Find round where threshold was reached
        converged_round = None
        for i, acc in enumerate(accuracy_history):
            if acc >= target_accuracy:
                converged_round = i + 1
                break
        
        return {
            'converged': converged_round is not None,
            'convergence_round': converged_round,
            'final_accuracy': final_accuracy,
            'target_accuracy': target_accuracy,
            'total_rounds': len(accuracy_history)
        }
    
    @staticmethod
    def calculate_early_detection_rate(
        predictions: np.ndarray,
        true_labels: np.ndarray,
        rul_values: np.ndarray,
        threshold: int = 50
    ) -> Dict:
        """
        Calculate early detection rate for predictive maintenance
        
        Args:
            predictions: Predicted labels
            true_labels: True labels
            rul_values: Remaining useful life values
            threshold: RUL threshold for "early" detection
            
        Returns:
            Dictionary with early detection metrics
        """
        # Find samples where failure was correctly predicted with RUL > threshold
        correct_predictions = (predictions == true_labels) & (true_labels == 1)
        early_detections = correct_predictions & (rul_values > threshold)
        
        # Calculate rates
        total_failures = np.sum(true_labels == 1)
        early_detected = np.sum(early_detections)
        
        if total_failures == 0:
            return {
                'early_detection_rate': 0.0,
                'early_detected': 0,
                'total_failures': 0
            }
        
        early_detection_rate = early_detected / total_failures
        
        # Average lead time for early detections
        if early_detected > 0:
            avg_lead_time = np.mean(rul_values[early_detections])
        else:
            avg_lead_time = 0.0
        
        return {
            'early_detection_rate': early_detection_rate,
            'early_detected': int(early_detected),
            'total_failures': int(total_failures),
            'avg_lead_time': float(avg_lead_time)
        }

