"""
Batch Processor - Handle batch predictions efficiently
"""
import numpy as np
from typing import List, Dict, Tuple
import time


class BatchProcessor:
    """Process predictions in batches."""

    @staticmethod
    def prepare_batch(images: List[np.ndarray]) -> Tuple[np.ndarray, List[int]]:
        """
        Prepare batch for model input.

        Args:
            images: List of preprocessed image arrays

        Returns:
            Stacked array and original indices
        """
        batch = np.stack(images, axis=0)
        indices = list(range(len(images)))
        return batch, indices

    @staticmethod
    def process_batch_predictions(predictions: np.ndarray, batch_indices: List[int],
                                  threshold: float = 0.0) -> List[Dict]:
        """
        Post-process batch predictions.

        Args:
            predictions: Model predictions array
            batch_indices: Original batch indices
            threshold: Confidence threshold (0.0 = all, >0.0 = filtered)

        Returns:
            List of prediction results
        """
        results = []

        for idx, pred in enumerate(predictions):
            digit = int(np.argmax(pred))
            confidence = float(pred[digit])

            # Filter by threshold
            if confidence < threshold:
                continue

            results.append({
                'index': batch_indices[idx],
                'digit': digit,
                'confidence': confidence,
                'top_3': BatchProcessor._get_top_n(pred, 3)
            })

        return results

    @staticmethod
    def _get_top_n(predictions: np.ndarray, n: int = 3) -> List[Dict]:
        """Get top N predictions."""
        top_indices = np.argsort(predictions)[-n:][::-1]
        return [
            {'digit': int(idx), 'confidence': float(predictions[idx])}
            for idx in top_indices
        ]

    @staticmethod
    def chunk_batch(items: List, chunk_size: int) -> List[List]:
        """Split batch into chunks."""
        return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


class BatchStats:
    """Statistics for batch processing."""

    def __init__(self):
        self.total_processed = 0
        self.total_time_ms = 0
        self.successful = 0
        self.failed = 0
        self.timings = []

    def add_batch(self, count: int, time_ms: float, success_count: int):
        """Add batch processing result."""
        self.total_processed += count
        self.total_time_ms += time_ms
        self.successful += success_count
        self.failed += (count - success_count)
        self.timings.append(time_ms)

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if not self.timings:
            return {}

        timings_array = np.array(self.timings)

        return {
            'total_processed': self.total_processed,
            'successful': self.successful,
            'failed': self.failed,
            'success_rate_percent': round(
                self.successful / self.total_processed * 100, 2
            ) if self.total_processed > 0 else 0,
            'total_time_ms': round(self.total_time_ms, 2),
            'avg_batch_time_ms': round(np.mean(timings_array), 2),
            'min_batch_time_ms': round(np.min(timings_array), 2),
            'max_batch_time_ms': round(np.max(timings_array), 2),
            'throughput_items_per_sec': round(
                self.total_processed / (self.total_time_ms / 1000), 2
            ) if self.total_time_ms > 0 else 0,
        }

    def print_summary(self):
        """Print summary statistics."""
        summary = self.get_summary()
        if not summary:
            return

        print("\n" + "="*60)
        print("BATCH PROCESSING STATISTICS")
        print("="*60)
        print(f"Total Processed:      {summary['total_processed']}")
        print(f"Successful:           {summary['successful']}")
        print(f"Failed:               {summary['failed']}")
        print(f"Success Rate:         {summary['success_rate_percent']}%")
        print(f"Total Time:           {summary['total_time_ms']:.2f}ms")
        print(f"Avg Batch Time:       {summary['avg_batch_time_ms']:.2f}ms")
        print(f"Throughput:           {summary['throughput_items_per_sec']:.1f} items/sec")
        print("="*60 + "\n")


class StreamBatchProcessor:
    """Process streaming predictions."""

    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self.buffer = []
        self.callbacks = []

    def add_item(self, item):
        """Add item to buffer."""
        self.buffer.append(item)
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self):
        """Process buffered items."""
        if self.buffer:
            self._process_batch(self.buffer)
            self.buffer = []

    def _process_batch(self, batch):
        """Process batch internally."""
        for callback in self.callbacks:
            callback(batch)

    def on_batch_complete(self, callback):
        """Register callback for batch completion."""
        self.callbacks.append(callback)
