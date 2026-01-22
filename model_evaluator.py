"""
Model Evaluator - Comprehensive model evaluation and metrics
"""
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import tensorflow as tf
from tensorflow import keras


class ModelEvaluator:
    """Evaluate model performance on test data."""

    def __init__(self, model_path: str = "models/mnist_model.h5"):
        self.model = keras.models.load_model(model_path)
        self.metrics_dir = Path("metrics")
        self.metrics_dir.mkdir(exist_ok=True)

    def evaluate_full(self, x_test, y_test):
        """Full evaluation with all metrics."""
        print("Evaluating model on test set...")

        # Predictions
        y_pred_proba = self.model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
        }

        # Per-class metrics
        class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        metrics['classification_report'] = class_report
        metrics['confusion_matrix'] = cm.tolist()

        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print("="*60)

        return metrics

    def per_digit_metrics(self, x_test, y_test):
        """Get metrics for each digit."""
        y_pred_proba = self.model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        per_digit = {}
        for digit in range(10):
            mask = y_test == digit
            if mask.sum() > 0:
                acc = accuracy_score(y_test[mask], y_pred[mask])
                per_digit[str(digit)] = {
                    'accuracy': float(acc),
                    'samples': int(mask.sum())
                }

        return per_digit

    def save_metrics(self, metrics: dict, name: str = "evaluation"):
        """Save metrics to file."""
        path = self.metrics_dir / f"{name}.json"
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {path}")
        return path

    def get_model_info(self):
        """Get model information."""
        return {
            'total_params': int(self.model.count_params()),
            'trainable_params': int(sum([tf.size(w).numpy() for w in self.model.trainable_weights])),
            'layers': len(self.model.layers),
            'model_name': self.model.name if hasattr(self.model, 'name') else 'mnist_model',
        }
