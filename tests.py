"""
Unit tests for MNIST digit recognition system
Run with: python -m pytest tests.py -v
"""
import os
import json
import base64
import unittest
from io import BytesIO
import numpy as np
from PIL import Image

# Test imports
try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TF = True
except ImportError:
    HAS_TF = False


class TestDataGeneration(unittest.TestCase):
    """Test data generation and preprocessing."""

    def test_image_generation(self):
        """Test creating a dummy MNIST-like image."""
        # Create a simple test image
        img = Image.new('L', (28, 28), color=255)  # White background
        img_array = np.array(img, dtype='float32') / 255.0

        # Check shape
        self.assertEqual(img_array.shape, (28, 28))

        # Check value range
        self.assertTrue(np.all(img_array >= 0))
        self.assertTrue(np.all(img_array <= 1))

    def test_image_to_base64(self):
        """Test converting image to base64."""
        img = Image.new('L', (28, 28), color=0)
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode()

        # Check base64 format
        self.assertIn('iVBORw0KGgo', img_base64)  # PNG magic bytes in base64

    def test_image_preprocessing(self):
        """Test image preprocessing pipeline."""
        # Create test image
        img = Image.new('L', (28, 28), color=200)
        img_array = np.array(img, dtype='float32') / 255.0

        # Normalize
        img_normalized = img_array / 255.0 if np.max(img_array) > 1 else img_array

        # Invert
        img_inverted = 1.0 - img_normalized

        # Flatten
        img_flat = img_inverted.reshape(1, 784)

        # Check shape
        self.assertEqual(img_flat.shape, (1, 784))

        # Check values
        self.assertTrue(np.all(img_flat >= 0))
        self.assertTrue(np.all(img_flat <= 1))


class TestModelManager(unittest.TestCase):
    """Test model manager functionality."""

    def setUp(self):
        """Set up test fixtures."""
        from model_manager import ModelManager
        self.manager = ModelManager(models_dir="test_models")

    def tearDown(self):
        """Clean up test files."""
        import shutil
        if os.path.exists("test_models"):
            shutil.rmtree("test_models")

    def test_metadata_loading(self):
        """Test loading metadata."""
        # Should create empty metadata
        self.assertEqual(len(self.manager.metadata["models"]), 0)
        self.assertIsNone(self.manager.metadata["current"])

    def test_model_registration(self):
        """Test registering a model."""
        # Create dummy model file
        os.makedirs("test_models", exist_ok=True)
        dummy_model_path = "test_models/dummy_model.h5"
        with open(dummy_model_path, 'w') as f:
            f.write("dummy")

        # Register model
        model_id = self.manager.register_model(
            model_path=dummy_model_path,
            accuracy=0.95,
            loss=0.15,
            training_time=120,
            epochs=10,
            batch_size=32
        )

        # Verify registration
        self.assertEqual(model_id, "v1")
        self.assertIn(model_id, self.manager.metadata["models"])
        self.assertEqual(self.manager.metadata["current"], model_id)

    def test_model_info_retrieval(self):
        """Test retrieving model info."""
        # Create and register dummy model
        os.makedirs("test_models", exist_ok=True)
        dummy_model_path = "test_models/dummy_model.h5"
        with open(dummy_model_path, 'w') as f:
            f.write("dummy")

        model_id = self.manager.register_model(
            model_path=dummy_model_path,
            accuracy=0.99,
            loss=0.05,
            training_time=100,
            epochs=15,
            batch_size=128
        )

        # Retrieve info
        info = self.manager.get_model_info(model_id)

        self.assertEqual(info["accuracy"], 0.99)
        self.assertEqual(info["loss"], 0.05)
        self.assertEqual(info["epochs"], 15)


class TestPredictionService(unittest.TestCase):
    """Test prediction service."""

    def test_image_preprocessing_valid(self):
        """Test preprocessing valid image."""
        from prediction_service import PredictionService

        # Create dummy model
        if HAS_TF:
            dummy_model = keras.Sequential([
                keras.layers.Dense(10, input_shape=(784,))
            ])
        else:
            self.skipTest("TensorFlow not available")

        service = PredictionService(dummy_model)

        # Create test image
        img = Image.new('L', (28, 28), color=0)
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_base64 = f"data:image/png;base64,{base64.b64encode(img_bytes.getvalue()).decode()}"

        # Preprocess
        result = service.preprocess_single_image(img_base64)

        # Check result
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (1, 784))

    def test_image_preprocessing_invalid(self):
        """Test preprocessing invalid image data."""
        from prediction_service import PredictionService

        if HAS_TF:
            dummy_model = keras.Sequential([
                keras.layers.Dense(10, input_shape=(784,))
            ])
        else:
            self.skipTest("TensorFlow not available")

        service = PredictionService(dummy_model)

        # Try invalid base64
        result = service.preprocess_single_image("invalid_base64_data")

        self.assertIsNone(result)


class TestTrainingVisualizer(unittest.TestCase):
    """Test training visualization."""

    def setUp(self):
        """Set up test fixtures."""
        from training_visualizer import TrainingVisualizer
        self.visualizer = TrainingVisualizer(output_dir="test_visualizations")

    def tearDown(self):
        """Clean up test files."""
        import shutil
        if os.path.exists("test_visualizations"):
            shutil.rmtree("test_visualizations")

    def test_metrics_saving(self):
        """Test saving metrics as JSON."""
        metrics = {
            'accuracy': 0.95,
            'loss': 0.15,
            'epochs': 10
        }

        path = self.visualizer.save_metrics_json(metrics, model_id="test")

        # Verify file was created
        self.assertTrue(os.path.exists(path))

        # Load and verify
        with open(path, 'r') as f:
            loaded_metrics = json.load(f)
        self.assertEqual(loaded_metrics['accuracy'], 0.95)


class TestDataAugmentation(unittest.TestCase):
    """Test data augmentation techniques."""

    def test_noise_addition(self):
        """Test adding noise to images."""
        from data_augmentation import add_noise

        img = np.ones((28, 28))
        noisy_img = add_noise(img, noise_factor=0.1)

        # Should have same shape
        self.assertEqual(noisy_img.shape, img.shape)

        # Should be different
        self.assertFalse(np.allclose(img, noisy_img))

    def test_mixup_batch(self):
        """Test mixup augmentation."""
        from data_augmentation import mixup_batch

        x_batch = np.random.randn(4, 784).astype('float32')
        y_batch = np.eye(10)[[1, 3, 7, 9]]

        x_mixed, y_mixed = mixup_batch(x_batch, y_batch)

        # Check shapes
        self.assertEqual(x_mixed.shape, x_batch.shape)
        self.assertEqual(y_mixed.shape, y_batch.shape)

        # Should be interpolated
        self.assertFalse(np.array_equal(x_mixed, x_batch))


class TestModelOptimizer(unittest.TestCase):
    """Test model optimization."""

    def setUp(self):
        """Set up test fixtures."""
        from model_optimizer import ModelOptimizer
        self.optimizer = ModelOptimizer()

    def test_size_reduction_calculation(self):
        """Test size reduction calculation."""
        # Create dummy files
        os.makedirs("models", exist_ok=True)

        orig_path = "models/test_original.h5"
        opt_path = "models/test_optimized.tflite"

        with open(orig_path, 'w') as f:
            f.write("x" * 1000)  # 1000 bytes

        with open(opt_path, 'w') as f:
            f.write("y" * 500)  # 500 bytes

        stats = self.optimizer.calculate_model_size_reduction(orig_path, opt_path)

        # Check reduction
        self.assertEqual(stats['reduction_percent'], 50.0)

        # Cleanup
        os.remove(orig_path)
        os.remove(opt_path)


class TestIntegration(unittest.TestCase):
    """Integration tests."""

    def test_end_to_end_preprocessing(self):
        """Test end-to-end image preprocessing."""
        # Create test image
        img = Image.new('RGB', (100, 100), color='red')

        # Convert to base64
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_base64 = f"data:image/png;base64,{base64.b64encode(img_bytes.getvalue()).decode()}"

        # Should be valid base64
        self.assertTrue(len(img_base64) > 0)
        self.assertIn('data:image/png;base64,', img_base64)


# Performance benchmarks
class TestPerformance(unittest.TestCase):
    """Performance tests."""

    def test_preprocessing_speed(self):
        """Test image preprocessing is fast."""
        import time
        from prediction_service import PredictionService

        if not HAS_TF:
            self.skipTest("TensorFlow not available")

        dummy_model = keras.Sequential([
            keras.layers.Dense(10, input_shape=(784,))
        ])
        service = PredictionService(dummy_model)

        # Create test images
        images = []
        for _ in range(10):
            img = Image.new('L', (28, 28), color=0)
            img_bytes = BytesIO()
            img.save(img_bytes, format='PNG')
            img_base64 = f"data:image/png;base64,{base64.b64encode(img_bytes.getvalue()).decode()}"
            images.append(img_base64)

        # Time preprocessing
        start = time.time()
        for img_data in images:
            service.preprocess_single_image(img_data)
        elapsed = (time.time() - start) / 10 * 1000  # ms per image

        print(f"Average preprocessing time: {elapsed:.2f}ms")

        # Should be reasonably fast
        self.assertLess(elapsed, 100)  # Should be less than 100ms


if __name__ == '__main__':
    unittest.main()
