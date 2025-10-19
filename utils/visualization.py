import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class ResultsVisualizer:
    """Visualize experimental results"""
    
    @staticmethod
    def plot_training_curves(results_dict, save_path='results/training_curves.png'):
        """
        Plot training curves for multiple methods
        
        Args:
            results_dict: Dict mapping method names to their results
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy over rounds
        ax = axes[0, 0]
        for method_name, results in results_dict.items():
            if 'round' in results:
                ax.plot(results['round'], results['accuracy'], 
                       label=method_name, marker='o', markersize=3)
        ax.set_xlabel('Round')
        ax.set_ylabel('Accuracy')
        ax.set_title('Test Accuracy vs Training Round')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # F1-Score over rounds
        ax = axes[0, 1]
        for method_name, results in results_dict.items():
            if 'f1_score' in results:
                ax.plot(results['round'], results['f1_score'], 
                       label=method_name, marker='o', markersize=3)
        ax.set_xlabel('Round')
        ax.set_ylabel('F1-Score')
        ax.set_title('F1-Score vs Training Round')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Communication cost
        ax = axes[1, 0]
        for method_name, results in results_dict.items():
            if 'communication_mb' in results:
                cumulative_comm = np.cumsum(results['communication_mb'])
                ax.plot(results['round'], cumulative_comm, 
                       label=method_name, marker='o', markersize=3)
        ax.set_xlabel('Round')
        ax.set_ylabel('Cumulative Communication (MB)')
        ax.set_title('Communication Cost')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Final comparison bar chart
        ax = axes[1, 1]
        methods = list(results_dict.keys())
        final_accuracy = [results_dict[m]['accuracy'][-1] for m in methods]
        colors = plt.cm.Set3(range(len(methods)))
        bars = ax.bar(methods, final_accuracy, color=colors)
        ax.set_ylabel('Final Accuracy')
        ax.set_title('Final Model Comparison')
        ax.set_ylim([min(final_accuracy) - 0.05, 1.0])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved training curves to {save_path}")
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, save_path='results/confusion_matrix.png'):
        """Plot confusion matrix"""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Degrading'],
                   yticklabels=['Normal', 'Degrading'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved confusion matrix to {save_path}")
