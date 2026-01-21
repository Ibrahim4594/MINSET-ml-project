"""
Training Visualizer - Plot and save training metrics
"""
import json
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.use('Agg')  # Use non-interactive backend


class TrainingVisualizer:
    """Visualize and save training history."""

    def __init__(self, output_dir="visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def plot_training_history(self, history, model_id="default"):
        """
        Plot training and validation accuracy/loss.

        Args:
            history: Keras training history object
            model_id: Model identifier for file naming
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot accuracy
        axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Plot loss
        axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
        axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / f"training_history_{model_id}.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"Training history saved to {output_path}")
        plt.close()

        return str(output_path)

    def plot_model_comparison(self, models_info):
        """
        Compare multiple models.

        Args:
            models_info: List of dicts with model info
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        model_names = [m['name'] for m in models_info]
        accuracies = [m['accuracy'] for m in models_info]
        colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe']

        bars = ax.bar(model_names, accuracies, color=colors[:len(model_names)])

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Test Accuracy', fontsize=12)
        ax.set_ylim([min(accuracies) * 0.99, 1.0])
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / "model_comparison.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"Model comparison saved to {output_path}")
        plt.close()

        return str(output_path)

    def plot_confusion_matrix(self, cm, model_id="default"):
        """
        Plot confusion matrix.

        Args:
            cm: Confusion matrix (10x10 for MNIST)
            model_id: Model identifier
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax)

        tick_marks = range(10)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

        # Add text annotations
        for i in range(10):
            for j in range(10):
                text = ax.text(j, i, cm[i, j],
                             ha="center", va="center",
                             color="white" if cm[i, j] > cm.max() / 2 else "black",
                             fontsize=8)

        plt.tight_layout()
        output_path = self.output_dir / f"confusion_matrix_{model_id}.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"Confusion matrix saved to {output_path}")
        plt.close()

        return str(output_path)

    def plot_inference_time_distribution(self, times, model_id="default"):
        """
        Plot inference time distribution.

        Args:
            times: List of inference times in milliseconds
            model_id: Model identifier
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        axes[0].hist(times, bins=50, color='#667eea', alpha=0.7, edgecolor='black')
        axes[0].set_title('Inference Time Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Time (ms)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].axvline(sum(times) / len(times), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {sum(times)/len(times):.2f}ms')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)

        # Box plot
        axes[1].boxplot(times, vert=True)
        axes[1].set_title('Inference Time Statistics', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Time (ms)', fontsize=12)
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / f"inference_distribution_{model_id}.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"Inference distribution saved to {output_path}")
        plt.close()

        return str(output_path)

    def save_metrics_json(self, metrics, model_id="default"):
        """Save metrics as JSON."""
        output_path = self.output_dir / f"metrics_{model_id}.json"
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {output_path}")
        return str(output_path)
