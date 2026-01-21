"""
Quantize trained MNIST model for faster inference and smaller size
Run this after training with: python quantize_model.py
"""
import os
from model_optimizer import ModelOptimizer


def quantize_mnist_model():
    """Quantize the trained MNIST model."""

    model_path = "models/mnist_model.h5"

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run train_model.py first")
        return

    optimizer = ModelOptimizer()

    # Quantize the model
    quantized_path = optimizer.quantize_model(model_path)

    # Create optimized inference script
    optimizer.create_inference_script(quantized_path)

    print("\nQuantization complete!")
    print(f"Quantized model: {quantized_path}")
    print("\nYou can now use the quantized model for even faster inference")


if __name__ == "__main__":
    quantize_mnist_model()
