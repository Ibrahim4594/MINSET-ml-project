"""
Advanced Training - Hyperparameter tuning and custom training
"""
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from data_augmentation import create_augmentation_generator
from model_manager import ModelManager
from training_visualizer import TrainingVisualizer


class AdvancedTrainer:
    """Advanced training with customizable hyperparameters."""

    def __init__(self):
        self.model_manager = ModelManager()
        self.visualizer = TrainingVisualizer()
        self.history = None

    def build_custom_model(self, architecture):
        """
        Build model from custom architecture specification.

        Args:
            architecture: List of layer specs
            Example: [
                {'type': 'dense', 'units': 256, 'activation': 'relu'},
                {'type': 'dropout', 'rate': 0.3},
                {'type': 'dense', 'units': 128, 'activation': 'relu'},
                {'type': 'dropout', 'rate': 0.2},
                {'type': 'dense', 'units': 10, 'activation': 'softmax'}
            ]
        """
        model = keras.Sequential()
        model.add(layers.Input(shape=(784,)))

        for i, layer_spec in enumerate(architecture):
            layer_type = layer_spec.get('type', 'dense')

            if layer_type == 'dense':
                model.add(layers.Dense(
                    units=layer_spec.get('units', 128),
                    activation=layer_spec.get('activation', 'relu')
                ))
            elif layer_type == 'dropout':
                model.add(layers.Dropout(
                    rate=layer_spec.get('rate', 0.2)
                ))
            elif layer_type == 'batchnorm':
                model.add(layers.BatchNormalization())
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

        return model

    def train_with_config(self, config):
        """
        Train model with custom configuration.

        Args:
            config: Dictionary with training parameters
        """
        print(f"\n{'='*60}")
        print("ADVANCED TRAINING WITH CUSTOM CONFIG")
        print(f"{'='*60}")

        # Extract config
        architecture = config.get('architecture', self._default_architecture())
        learning_rate = config.get('learning_rate', 0.001)
        batch_size = config.get('batch_size', 128)
        epochs = config.get('epochs', 20)
        use_augmentation = config.get('augmentation', True)
        optimizer_name = config.get('optimizer', 'adam')
        loss_function = config.get('loss', 'sparse_categorical_crossentropy')

        print(f"\nConfiguration:")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Batch size: {batch_size}")
        print(f"  Epochs: {epochs}")
        print(f"  Augmentation: {use_augmentation}")
        print(f"  Optimizer: {optimizer_name}")
        print(f"  Loss: {loss_function}")

        # Load data
        print(f"\nLoading MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
        x_train_flat = x_train.reshape(-1, 28 * 28)
        x_test_flat = x_test.reshape(-1, 28 * 28)

        # Build model
        print(f"\nBuilding model...")
        model = self.build_custom_model(architecture)

        # Compile
        optimizer = self._get_optimizer(optimizer_name, learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=loss_function,
            metrics=['accuracy']
        )

        print(f"\nModel Summary:")
        model.summary()

        # Train
        print(f"\nTraining model...")
        start_time = time.time()

        if use_augmentation:
            print("Using data augmentation...")
            augmentor = create_augmentation_generator()
            x_train_reshaped = x_train.reshape(-1, 28, 28, 1)

            history = model.fit(
                augmentor.flow(x_train_reshaped, y_train, batch_size=batch_size),
                steps_per_epoch=len(x_train) // batch_size,
                epochs=epochs,
                validation_split=0.1,
                verbose=1
            )
        else:
            history = model.fit(
                x_train_flat,
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=0.1,
                verbose=1
            )

        training_time = time.time() - start_time

        # Evaluate
        print(f"\nEvaluating on test set...")
        test_loss, test_accuracy = model.evaluate(x_test_flat, y_test, verbose=0)

        print(f"\nResults:")
        print(f"  Test accuracy: {test_accuracy:.4f}")
        print(f"  Test loss: {test_loss:.4f}")
        print(f"  Training time: {training_time:.1f}s")

        # Save model
        os.makedirs("models", exist_ok=True)
        model_path = "models/mnist_model_custom.h5"
        model.save(model_path)

        # Register model
        model_id = self.model_manager.register_model(
            model_path=model_path,
            accuracy=float(test_accuracy),
            loss=float(test_loss),
            training_time=training_time,
            epochs=epochs,
            batch_size=batch_size
        )

        # Visualize
        self.visualizer.plot_training_history(history, model_id=model_id)

        self.history = history
        return model, history

    def _default_architecture(self):
        """Return default model architecture."""
        return [
            {'type': 'dense', 'units': 128, 'activation': 'relu'},
            {'type': 'dropout', 'rate': 0.2},
            {'type': 'dense', 'units': 64, 'activation': 'relu'},
            {'type': 'dropout', 'rate': 0.2},
            {'type': 'dense', 'units': 10, 'activation': 'softmax'}
        ]

    def _get_optimizer(self, name, learning_rate):
        """Get optimizer from name."""
        if name == 'adam':
            return keras.optimizers.Adam(learning_rate=learning_rate)
        elif name == 'sgd':
            return keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif name == 'rmsprop':
            return keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            return keras.optimizers.Adam(learning_rate=learning_rate)

    def hyperparameter_sweep(self, configs):
        """
        Train multiple models with different configurations.

        Args:
            configs: List of configuration dictionaries
        """
        results = []

        for i, config in enumerate(configs):
            print(f"\n\n{'#'*60}")
            print(f"Configuration {i+1}/{len(configs)}")
            print(f"{'#'*60}")

            try:
                model, history = self.train_with_config(config)
                test_acc = history.history['val_accuracy'][-1]

                results.append({
                    'config': config,
                    'accuracy': test_acc,
                    'status': 'success'
                })

                print(f"✓ Completed with accuracy: {test_acc:.4f}")
            except Exception as e:
                print(f"✗ Failed: {e}")
                results.append({
                    'config': config,
                    'accuracy': 0,
                    'status': 'failed',
                    'error': str(e)
                })

        # Print summary
        print(f"\n\n{'='*60}")
        print("HYPERPARAMETER SWEEP SUMMARY")
        print(f"{'='*60}")

        for i, result in enumerate(results, 1):
            status = "✓" if result['status'] == 'success' else "✗"
            acc = result.get('accuracy', 0)
            print(f"{status} Config {i}: Accuracy = {acc:.4f}")

        # Find best
        best = max([r for r in results if r['status'] == 'success'],
                  key=lambda x: x['accuracy'])
        print(f"\nBest: {best['accuracy']:.4f}")

        return results


def example_config():
    """Return example training configuration."""
    return {
        'architecture': [
            {'type': 'dense', 'units': 256, 'activation': 'relu'},
            {'type': 'batchnorm'},
            {'type': 'dropout', 'rate': 0.3},
            {'type': 'dense', 'units': 128, 'activation': 'relu'},
            {'type': 'batchnorm'},
            {'type': 'dropout', 'rate': 0.2},
            {'type': 'dense', 'units': 10, 'activation': 'softmax'}
        ],
        'learning_rate': 0.001,
        'batch_size': 128,
        'epochs': 20,
        'augmentation': True,
        'optimizer': 'adam',
        'loss': 'sparse_categorical_crossentropy'
    }


if __name__ == "__main__":
    # Example: Train with custom config
    trainer = AdvancedTrainer()

    # Single custom training
    config = example_config()
    trainer.train_with_config(config)

    # Or do hyperparameter sweep
    # configs = [
    #     {...config1...},
    #     {...config2...},
    # ]
    # results = trainer.hyperparameter_sweep(configs)
