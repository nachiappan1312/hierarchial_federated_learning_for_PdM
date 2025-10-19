import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int = 64,
        return_predictions: bool = False
    ) -> Dict:
        """
        Comprehensive model evaluation
        
        Args:
            X_test: Test features
            y_test: Test labels
            batch_size: Batch size for evaluation
            return_predictions: Whether to return predictions
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        # Create data loader
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.LongTensor(y_test)
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        all_predictions = []
        all_probabilities = []
        all_labels = []
        total_loss = 0.0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                
                # Get predictions and probabilities
                probabilities = torch.softmax(output, dim=1)
                _, predicted = torch.max(output.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        metrics = {
            'loss': total_loss / len(test_loader),
            'accuracy': accuracy_score(all_labels, all_predictions),
            'precision': precision_score(all_labels, all_predictions, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_predictions, average='weighted', zero_division=0),
            'f1_score': f1_score(all_labels, all_predictions, average='weighted', zero_division=0),
        }
        
        # Add per-class metrics
        precision_per_class = precision_score(all_labels, all_predictions, average=None, zero_division=0)
        recall_per_class = recall_score(all_labels, all_predictions, average=None, zero_division=0)
        f1_per_class = f1_score(all_labels, all_predictions, average=None, zero_division=0)
        
        metrics['precision_per_class'] = precision_per_class.tolist()
        metrics['recall_per_class'] = recall_per_class.tolist()
        metrics['f1_per_class'] = f1_per_class.tolist()
        
        # AUC-ROC (for binary classification)
        if len(np.unique(all_labels)) == 2:
            try:
                metrics['auc_roc'] = roc_auc_score(all_labels, all_probabilities[:, 1])
            except:
                metrics['auc_roc'] = None
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(all_labels, all_predictions).tolist()
        
        if return_predictions:
            metrics['predictions'] = all_predictions
            metrics['probabilities'] = all_probabilities
            metrics['labels'] = all_labels
        
        return metrics
    
    def get_classification_report(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        target_names: Optional[list] = None
    ) -> str:
        """
        Get detailed classification report
        
        Args:
            X_test: Test features
            y_test: Test labels
            target_names: Names for each class
            
        Returns:
            Classification report string
        """
        metrics = self.evaluate(X_test, y_test, return_predictions=True)
        
        if target_names is None:
            target_names = [f"Class {i}" for i in range(len(np.unique(y_test)))]
        
        report = classification_report(
            metrics['labels'],
            metrics['predictions'],
            target_names=target_names
        )
        
        return report
    
    def plot_confusion_matrix(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        target_names: Optional[list] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix
        
        Args:
            X_test: Test features
            y_test: Test labels
            target_names: Names for each class
            save_path: Path to save plot
        """
        metrics = self.evaluate(X_test, y_test)
        cm = np.array(metrics['confusion_matrix'])
        
        if target_names is None:
            target_names = [f"Class {i}" for i in range(len(cm))]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names
        )
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.close()
    
    def calculate_rul_metrics(
        self,
        X_test: np.ndarray,
        y_true_rul: np.ndarray,
        y_pred_rul: np.ndarray
    ) -> Dict:
        """
        Calculate Remaining Useful Life prediction metrics
        
        Args:
            X_test: Test features
            y_true_rul: True RUL values
            y_pred_rul: Predicted RUL values
            
        Returns:
            Dictionary with RUL metrics
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(y_true_rul, y_pred_rul)
        mse = mean_squared_error(y_true_rul, y_pred_rul)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_rul, y_pred_rul)
        
        # Scoring function from PHM08 challenge
        def score_function(y_true, y_pred):
            d = y_pred - y_true
            score = np.sum(np.where(d < 0, np.exp(-d/13) - 1, np.exp(d/10) - 1))
            return score
        
        score = score_function(y_true_rul, y_pred_rul)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'phm_score': score
        }
