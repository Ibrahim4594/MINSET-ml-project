"""
MNIST Model Training Script
Trains a lightweight neural network on MNIST dataset and saves it for inference.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def train_mnist_model():
    """Train and save a lightweight model on MNIST dataset."""

    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize pixel values to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Flatten images from 28x28 to 784
    x_train_flat = x_train.reshape(-1, 28 * 28)
    x_test_flat = x_test.reshape(-1, 28 * 28)

    print(f"Training set shape: {x_train_flat.shape}")
    print(f"Test set shape: {x_test_flat.shape}")

    # Build a small neural network (2-3 layers)
    print("\nBuilding model...")
    model = keras.Sequential([
        layers.Dense(128, activation="relu", input_shape=(784,)),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(10, activation="softmax")  # 10 digit classes (0-9)
    ])

    # Compile model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\nModel Summary:")
    model.summary()

    # Train model
    print("\nTraining model...")
    history = model.fit(
        x_train_flat,
        y_train,
        batch_size=128,
        epochs=15,
        verbose=1,
        validation_split=0.1
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(x_test_flat, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/mnist_model.h5"
    model.save(model_path)
    print(f"\nModel saved to {model_path}")

    # Print model file size
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Model size: {model_size_mb:.2f} MB")

    return model

if __name__ == "__main__":
    train_mnist_model()
