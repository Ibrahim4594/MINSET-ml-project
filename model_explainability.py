"""
Model Explainability - Understand model predictions
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class ModelExplainer:
    """Explain model predictions using various techniques."""

    def __init__(self, model):
        self.model = model
        self.output_dir = Path("explanations")
        self.output_dir.mkdir(exist_ok=True)

    def get_saliency_map(self, image_array):
        """
        Compute saliency map using gradient-based method.

        Args:
            image_array: Input image (1, 784)

        Returns:
            Saliency map (28, 28)
        """
        import tensorflow as tf

        # Convert to tensor
        image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(image_tensor)
            predictions = self.model(image_tensor)
            max_class = tf.argmax(predictions[0])
            class_channel = predictions[0, max_class]

        # Get gradients
        grads = tape.gradient(class_channel, image_tensor)

        # Take absolute value and reshape
        saliency = tf.abs(grads[0])
        saliency = tf.reshape(saliency, (28, 28))

        return saliency.numpy()

    def get_activation_map(self, image_array, layer_name=None):
        """
        Get activation map from intermediate layer.

        Args:
            image_array: Input image
            layer_name: Layer name (default: last hidden layer)

        Returns:
            Activation map
        """
        import tensorflow as tf

        if layer_name is None:
            # Use second-to-last layer
            layer_name = self.model.layers[-2].name

        # Create intermediate model
        intermediate_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer(layer_name).output
        )

        # Get activations
        activations = intermediate_model(image_array)

        # Average across feature maps
        activation_map = tf.reduce_mean(activations, axis=-1)[0]

        return activation_map.numpy()

    def plot_saliency_map(self, image_array, predicted_digit, output_path=None):
        """
        Plot saliency map visualization.

        Args:
            image_array: Input image
            predicted_digit: Predicted digit
            output_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Original image
        original = image_array[0].reshape(28, 28)
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Saliency map
        saliency = self.get_saliency_map(image_array)
        axes[1].imshow(saliency, cmap='hot')
        axes[1].set_title('Saliency Map')
        axes[1].axis('off')

        # Overlay
        axes[2].imshow(original, cmap='gray', alpha=0.5)
        axes[2].imshow(saliency, cmap='hot', alpha=0.5)
        axes[2].set_title(f'Predicted: {predicted_digit}')
        axes[2].axis('off')

        if output_path is None:
            output_path = self.output_dir / f"saliency_{predicted_digit}.png"

        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"Saliency map saved to {output_path}")
        plt.close()

        return str(output_path)

    def get_neuron_importance(self):
        """
        Get importance of neurons in the first dense layer.

        Returns:
            Array of neuron importance scores
        """
        # Get weights of first dense layer
        first_dense = None
        for layer in self.model.layers:
            if hasattr(layer, 'kernel'):
                first_dense = layer
                break

        if first_dense is None:
            return None

        weights = first_dense.kernel.numpy()  # Shape: (784, 128)

        # Compute importance as sum of absolute weights
        importance = np.sum(np.abs(weights), axis=1)  # Shape: (784,)

        return importance

    def plot_neuron_importance(self, output_path=None):
        """Plot neuron importance heatmap."""
        importance = self.get_neuron_importance()

        if importance is None:
            print("Could not compute neuron importance")
            return None

        # Reshape to image
        importance_image = importance.reshape(28, 28)

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(importance_image, cmap='viridis')
        ax.set_title('Input Neuron Importance', fontsize=14, fontweight='bold')
        ax.set_xlabel('Pixel X')
        ax.set_ylabel('Pixel Y')
        plt.colorbar(im, ax=ax, label='Importance')

        if output_path is None:
            output_path = self.output_dir / "neuron_importance.png"

        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"Neuron importance map saved to {output_path}")
        plt.close()

        return str(output_path)

    def get_feature_attribution(self, image_array, method='occlusion'):
        """
        Compute feature attribution using occlusion or other methods.

        Args:
            image_array: Input image (1, 784)
            method: 'occlusion' or 'gradient'

        Returns:
            Attribution scores (784,)
        """
        if method == 'occlusion':
            return self._occlusion_attribution(image_array)
        elif method == 'gradient':
            return self._gradient_attribution(image_array)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _occlusion_attribution(self, image_array, occlusion_size=4):
        """
        Compute attribution by occluding pixels.

        Faster version using sliding window.
        """
        # Get baseline prediction
        baseline_pred = self.model.predict(image_array, verbose=0)
        baseline_class = np.argmax(baseline_pred[0])
        baseline_confidence = baseline_pred[0, baseline_class]

        # Reshape image for occlusion
        image_2d = image_array[0].reshape(28, 28)
        attribution = np.zeros((28, 28))

        # Slide window
        step = occlusion_size
        for i in range(0, 28 - occlusion_size, step):
            for j in range(0, 28 - occlusion_size, step):
                # Create occluded image
                occluded = image_2d.copy()
                occluded[i:i+occlusion_size, j:j+occlusion_size] = 0

                # Get prediction
                occluded_flat = occluded.reshape(1, 784).astype(np.float32)
                pred = self.model.predict(occluded_flat, verbose=0)
                confidence = pred[0, baseline_class]

                # Attribution = drop in confidence
                drop = baseline_confidence - confidence
                attribution[i:i+occlusion_size, j:j+occlusion_size] = drop

        return attribution.flatten()

    def _gradient_attribution(self, image_array):
        """Compute attribution using gradients."""
        import tensorflow as tf

        saliency = self.get_saliency_map(image_array)
        return saliency.flatten()

    def explain_prediction(self, image_array, predicted_digit):
        """
        Generate comprehensive explanation for a prediction.

        Returns:
            Dictionary with explanation data
        """
        explanation = {
            'digit': predicted_digit,
            'saliency_map_path': self.plot_saliency_map(image_array, predicted_digit),
            'neuron_importance_path': self.plot_neuron_importance(),
        }

        # Compute attributions
        try:
            occlusion_attr = self.get_feature_attribution(image_array, method='occlusion')
            explanation['occlusion_attribution'] = occlusion_attr.tolist()
        except Exception as e:
            print(f"Could not compute occlusion attribution: {e}")

        return explanation

    def generate_report(self, image_array, predicted_digit, model_accuracy=None):
        """
        Generate a text report explaining the prediction.

        Args:
            image_array: Input image
            predicted_digit: Predicted digit
            model_accuracy: Model accuracy (optional)

        Returns:
            Report string
        """
        report = f"""
=== PREDICTION EXPLANATION REPORT ===

Predicted Digit: {predicted_digit}
Input Shape: {image_array.shape}

Model Analysis:
- Total layers: {len(self.model.layers)}
- Total parameters: {self.model.count_params():,}
"""

        if model_accuracy is not None:
            report += f"- Model accuracy: {model_accuracy:.2%}\n"

        report += """
Explanation Methods Used:
1. Saliency Maps (gradient-based)
   - Shows which pixels most affect the prediction
   - Generated from input gradients with respect to predicted class

2. Neuron Importance
   - Shows which pixels are most important for the model
   - Based on connection weights in first layer

3. Occlusion Attribution
   - Shows impact of occluding different regions
   - Measures drop in confidence when regions are hidden

Output Files:
- saliency_[digit].png: Visualization of prediction-influencing regions
- neuron_importance.png: Heatmap of important pixel regions

Interpretation:
- Brighter regions in saliency maps indicate stronger influence on prediction
- Compare multiple methods for robust understanding
- Use for model debugging and validation

=================================
"""
        return report
