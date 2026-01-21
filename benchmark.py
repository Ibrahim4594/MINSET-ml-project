"""
Performance Benchmarking - Measure model and system performance
Run with: python benchmark.py
"""
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from io import BytesIO
import base64
from model_manager import ModelManager


class Benchmarker:
    """Benchmark various components of the system."""

    def __init__(self, model_path="models/mnist_model.h5"):
        self.model_path = model_path
        self.model = None
        self.results = {}

    def load_model(self):
        """Load model for benchmarking."""
        if not os.path.exists(self.model_path):
            print(f"Error: Model not found at {self.model_path}")
            return False

        self.model = keras.models.load_model(self.model_path)
        print(f"Model loaded from {self.model_path}")
        return True

    def benchmark_inference(self, num_iterations=100):
        """Benchmark single inference time."""
        if self.model is None:
            print("Error: Model not loaded")
            return None

        # Create dummy input
        dummy_input = np.random.randn(1, 784).astype(np.float32)

        # Warmup
        self.model.predict(dummy_input, verbose=0)

        # Benchmark
        times = []
        print(f"\nBenchmarking {num_iterations} inferences...")

        for i in range(num_iterations):
            start = time.time()
            self.model.predict(dummy_input, verbose=0)
            elapsed = (time.time() - start) * 1000  # Convert to ms
            times.append(elapsed)

            if (i + 1) % 25 == 0:
                print(f"  Completed {i + 1}/{num_iterations}")

        times = np.array(times)

        results = {
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'mean_ms': float(np.mean(times)),
            'median_ms': float(np.median(times)),
            'std_ms': float(np.std(times)),
            'p95_ms': float(np.percentile(times, 95)),
            'p99_ms': float(np.percentile(times, 99)),
            'throughput_fps': float(1000 / np.mean(times)),
        }

        self.results['inference'] = results
        return results

    def benchmark_batch_inference(self, batch_sizes=[1, 4, 8, 16, 32]):
        """Benchmark batch inference with different batch sizes."""
        if self.model is None:
            print("Error: Model not loaded")
            return None

        results = {}
        print(f"\nBenchmarking batch inference...")

        for batch_size in batch_sizes:
            # Create batch input
            batch_input = np.random.randn(batch_size, 784).astype(np.float32)

            # Warmup
            self.model.predict(batch_input, verbose=0)

            # Benchmark
            times = []
            for _ in range(20):
                start = time.time()
                self.model.predict(batch_input, verbose=0)
                elapsed = (time.time() - start) * 1000 / batch_size  # Per-sample time
                times.append(elapsed)

            avg_time = np.mean(times)
            results[batch_size] = {
                'avg_time_ms': float(avg_time),
                'throughput_fps': float(1000 / avg_time),
                'batch_throughput_fps': float(batch_size * 1000 / (avg_time * batch_size))
            }
            print(f"  Batch size {batch_size}: {avg_time:.2f}ms per sample, {1000/avg_time:.0f} FPS")

        self.results['batch_inference'] = results
        return results

    def benchmark_image_processing(self, num_iterations=100):
        """Benchmark image preprocessing."""
        from prediction_service import PredictionService

        service = PredictionService(self.model)

        # Create test image
        img = Image.new('L', (280, 280), color=100)
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_base64 = f"data:image/png;base64,{base64.b64encode(img_bytes.getvalue()).decode()}"

        # Benchmark preprocessing
        times = []
        print(f"\nBenchmarking image preprocessing ({num_iterations} iterations)...")

        for i in range(num_iterations):
            start = time.time()
            service.preprocess_single_image(img_base64)
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)

        times = np.array(times)

        results = {
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'mean_ms': float(np.mean(times)),
            'median_ms': float(np.median(times)),
            'std_ms': float(np.std(times)),
        }

        self.results['image_processing'] = results
        return results

    def benchmark_end_to_end(self, num_iterations=50):
        """Benchmark complete prediction pipeline."""
        from prediction_service import PredictionService

        service = PredictionService(self.model)

        # Create test image
        img = Image.new('L', (280, 280), color=100)
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_base64 = f"data:image/png;base64,{base64.b64encode(img_bytes.getvalue()).decode()}"

        # Benchmark
        times = []
        print(f"\nBenchmarking end-to-end prediction ({num_iterations} iterations)...")

        for i in range(num_iterations):
            start = time.time()
            service.predict_single(img_base64)
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)

            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{num_iterations}")

        times = np.array(times)

        results = {
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'mean_ms': float(np.mean(times)),
            'median_ms': float(np.median(times)),
            'std_ms': float(np.std(times)),
            'p95_ms': float(np.percentile(times, 95)),
            'p99_ms': float(np.percentile(times, 99)),
            'throughput_fps': float(1000 / np.mean(times)),
        }

        self.results['end_to_end'] = results
        return results

    def benchmark_model_size(self):
        """Benchmark model size."""
        if not os.path.exists(self.model_path):
            return None

        size_bytes = os.path.getsize(self.model_path)
        size_mb = size_bytes / (1024 * 1024)
        size_kb = size_bytes / 1024

        results = {
            'size_bytes': size_bytes,
            'size_kb': float(size_kb),
            'size_mb': float(size_mb),
        }

        self.results['model_size'] = results
        return results

    def print_results(self):
        """Print benchmark results in a formatted table."""
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)

        # Inference timing
        if 'inference' in self.results:
            print("\n1. Single Inference Timing")
            print("-" * 80)
            r = self.results['inference']
            print(f"  Min:        {r['min_ms']:.2f} ms")
            print(f"  Max:        {r['max_ms']:.2f} ms")
            print(f"  Mean:       {r['mean_ms']:.2f} ms")
            print(f"  Median:     {r['median_ms']:.2f} ms")
            print(f"  Std Dev:    {r['std_ms']:.2f} ms")
            print(f"  P95:        {r['p95_ms']:.2f} ms")
            print(f"  P99:        {r['p99_ms']:.2f} ms")
            print(f"  Throughput: {r['throughput_fps']:.0f} FPS")

        # Batch inference
        if 'batch_inference' in self.results:
            print("\n2. Batch Inference Timing")
            print("-" * 80)
            for batch_size, metrics in self.results['batch_inference'].items():
                print(f"  Batch size {batch_size}: {metrics['avg_time_ms']:.2f}ms/sample, {metrics['throughput_fps']:.0f} FPS")

        # Image processing
        if 'image_processing' in self.results:
            print("\n3. Image Preprocessing")
            print("-" * 80)
            r = self.results['image_processing']
            print(f"  Mean:       {r['mean_ms']:.2f} ms")
            print(f"  Median:     {r['median_ms']:.2f} ms")
            print(f"  Std Dev:    {r['std_ms']:.2f} ms")

        # End-to-end
        if 'end_to_end' in self.results:
            print("\n4. End-to-End Prediction")
            print("-" * 80)
            r = self.results['end_to_end']
            print(f"  Min:        {r['min_ms']:.2f} ms")
            print(f"  Max:        {r['max_ms']:.2f} ms")
            print(f"  Mean:       {r['mean_ms']:.2f} ms")
            print(f"  Median:     {r['median_ms']:.2f} ms")
            print(f"  P95:        {r['p95_ms']:.2f} ms")
            print(f"  P99:        {r['p99_ms']:.2f} ms")
            print(f"  Throughput: {r['throughput_fps']:.0f} FPS")

        # Model size
        if 'model_size' in self.results:
            print("\n5. Model Size")
            print("-" * 80)
            r = self.results['model_size']
            print(f"  Size:       {r['size_mb']:.2f} MB ({r['size_kb']:.0f} KB)")

        print("\n" + "=" * 80)

    def save_results(self, filename="benchmark_results.txt"):
        """Save results to file."""
        import json

        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\nResults saved to {filename}")


def main():
    """Run all benchmarks."""
    print("MNIST Model Benchmarking Suite")
    print("=" * 80)

    benchmarker = Benchmarker()

    # Load model
    if not benchmarker.load_model():
        return

    # Run benchmarks
    benchmarker.benchmark_model_size()
    benchmarker.benchmark_inference(num_iterations=100)
    benchmarker.benchmark_batch_inference()
    benchmarker.benchmark_image_processing(num_iterations=50)
    benchmarker.benchmark_end_to_end(num_iterations=50)

    # Print results
    benchmarker.print_results()

    # Save results
    benchmarker.save_results()


if __name__ == "__main__":
    main()
