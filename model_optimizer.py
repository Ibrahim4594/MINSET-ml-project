"""
Model Optimizer - Techniques to optimize model size and speed
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path


class ModelOptimizer:
    """Optimize models through quantization and compression."""

    def __init__(self):
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)

    def quantize_model(self, model_path, output_path=None, representative_data=None):
        """
        Quantize model to int8 format using TensorFlow Lite.

        Args:
            model_path: Path to original .h5 model
            output_path: Path for quantized .tflite model
            representative_data: Data for quantization calibration

        Returns:
            Path to quantized model
        """
        if output_path is None:
            output_path = self.models_dir / "mnist_model_quantized.tflite"

        print(f"\nQuantizing model to TFLite format...")

        # Load the model
        model = keras.models.load_model(model_path)

        # Convert to TFLite with quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # Enable quantization optimization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Set target spec for quantization
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        # Provide representative data for quantization
        if representative_data is not None:
            def representative_dataset():
                for data in representative_data.batch(1).take(100):
                    yield [tf.cast(data, tf.float32)]
            converter.representative_dataset = representative_dataset

        # Convert
        quantized_model = converter.convert()

        # Save quantized model
        with open(output_path, 'wb') as f:
            f.write(quantized_model)

        original_size = os.path.getsize(model_path) / (1024 * 1024)
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)
        compression_ratio = (1 - quantized_size / original_size) * 100

        print(f"Original model size: {original_size:.2f} MB")
        print(f"Quantized model size: {quantized_size:.2f} MB")
        print(f"Compression: {compression_ratio:.1f}%")

        return str(output_path)

    def prune_model(self, model, pruning_rate=0.3):
        """
        Apply magnitude-based pruning to model.

        Args:
            model: Keras model
            pruning_rate: Fraction of weights to prune

        Returns:
            Pruned model
        """
        print(f"\nApplying pruning at {pruning_rate*100}% rate...")

        try:
            import tensorflow_model_optimization as tfmot

            # Create pruning config
            pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=pruning_rate,
                begin_step=0,
                end_step=1000
            )

            # Apply pruning
            pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
                model,
                pruning_schedule=pruning_schedule
            )

            print("Model pruning applied successfully")
            return pruned_model

        except ImportError:
            print("TensorFlow Model Optimization not installed. Skipping pruning.")
            return model

    def calculate_model_size_reduction(self, original_path, optimized_path):
        """Calculate size reduction from optimization."""
        original_size = os.path.getsize(original_path)
        optimized_size = os.path.getsize(optimized_path)
        reduction = (1 - optimized_size / original_size) * 100

        return {
            'original_size_kb': original_size / 1024,
            'optimized_size_kb': optimized_size / 1024,
            'reduction_percent': reduction
        }

    def estimate_speedup(self, original_model, optimized_model_path):
        """
        Estimate inference speedup from optimization.
        Note: Actual speedup depends on hardware.
        """
        import time

        # Create dummy input
        dummy_input = np.random.randn(1, 784).astype(np.float32)

        # Benchmark original model
        start = time.time()
        for _ in range(100):
            original_model.predict(dummy_input, verbose=0)
        original_time = (time.time() - start) / 100

        # Benchmark is for inference time estimation
        print(f"\nOriginal model inference time: {original_time*1000:.2f}ms")
        print("Note: Actual speedup depends on hardware and runtime")

        return original_time

    def create_inference_script(self, tflite_model_path, output_script="inference_optimized.py"):
        """Create a script for using the optimized model."""
        script_content = '''"""
Optimized MNIST inference using TFLite quantized model
"""
import numpy as np
import tensorflow as tf


class OptimizedMNISTPredictor:
    """Predict using quantized TFLite model."""

    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def preprocess(self, image_array):
        """Preprocess image for quantized model."""
        # Quantized models expect int8 input (-128 to 127)
        # Scale from float32 [0, 1] to int8 range
        input_scale, input_zero_point = self.input_details[0]["quantization"]
        if input_scale > 0:
            image_int8 = (image_array / input_scale).astype(np.int8) + input_zero_point
        else:
            # If no quantization params, use regular conversion
            image_int8 = (image_array * 127).astype(np.int8)
        return image_int8

    def predict(self, image_data):
        """Make prediction on image."""
        # Preprocess
        processed = self.preprocess(image_data)

        # Set input
        self.interpreter.set_tensor(self.input_details[0]["index"],
                                    processed.reshape(1, 784))

        # Run inference
        self.interpreter.invoke()

        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])

        # Dequantize output if needed
        output_scale, output_zero_point = self.output_details[0]["quantization"]
        if output_scale > 0:
            output_float = (output_data.astype(np.float32) - output_zero_point) * output_scale
        else:
            output_float = output_data.astype(np.float32) / 127

        return np.argmax(output_float[0])


if __name__ == "__main__":
    # Example usage
    predictor = OptimizedMNISTPredictor("models/mnist_model_quantized.tflite")

    # Create dummy input
    dummy_input = np.random.randn(784).astype(np.float32)
    dummy_input = (dummy_input - np.min(dummy_input)) / (np.max(dummy_input) - np.min(dummy_input))

    # Predict
    result = predictor.predict(dummy_input)
    print(f"Predicted digit: {result}")
'''

        with open(output_script, 'w') as f:
            f.write(script_content)

        print(f"Inference script created: {output_script}")
        return output_script
