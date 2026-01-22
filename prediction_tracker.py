"""
Prediction Tracker - Track and analyze prediction history
"""
import json
from datetime import datetime
from pathlib import Path
from collections import Counter


class PredictionTracker:
    """Track prediction history and statistics."""

    def __init__(self, max_records: int = 10000):
        self.predictions = []
        self.max_records = max_records
        self.data_dir = Path("prediction_data")
        self.data_dir.mkdir(exist_ok=True)

    def add_prediction(self, digit: int, confidence: float, inference_time_ms: float,
                      from_cache: bool = False, timestamp: str = None):
        """Add prediction to history."""
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        prediction = {
            'timestamp': timestamp,
            'digit': digit,
            'confidence': confidence,
            'inference_time_ms': inference_time_ms,
            'from_cache': from_cache
        }

        self.predictions.append(prediction)

        # Keep only recent predictions
        if len(self.predictions) > self.max_records:
            self.predictions = self.predictions[-self.max_records:]

    def get_statistics(self):
        """Get prediction statistics."""
        if not self.predictions:
            return {}

        digits = [p['digit'] for p in self.predictions]
        confidences = [p['confidence'] for p in self.predictions]
        times = [p['inference_time_ms'] for p in self.predictions]
        cache_hits = sum(1 for p in self.predictions if p['from_cache'])

        import numpy as np

        return {
            'total_predictions': len(self.predictions),
            'unique_digits_predicted': len(set(digits)),
            'most_common_digit': Counter(digits).most_common(1)[0][0] if digits else None,
            'confidence_stats': {
                'mean': float(np.mean(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences)),
                'std': float(np.std(confidences))
            },
            'timing_stats': {
                'mean_ms': float(np.mean(times)),
                'min_ms': float(np.min(times)),
                'max_ms': float(np.max(times)),
                'std_ms': float(np.std(times))
            },
            'cache_hits': cache_hits,
            'cache_hit_rate_percent': round(cache_hits / len(self.predictions) * 100, 2)
        }

    def get_digit_distribution(self):
        """Get distribution of predicted digits."""
        if not self.predictions:
            return {}

        digits = [p['digit'] for p in self.predictions]
        counter = Counter(digits)

        total = len(self.predictions)
        return {
            digit: {
                'count': count,
                'percentage': round(count / total * 100, 2)
            }
            for digit, count in sorted(counter.items())
        }

    def get_recent_predictions(self, limit: int = 100):
        """Get recent predictions."""
        return self.predictions[-limit:]

    def save_history(self, filename: str = None):
        """Save prediction history to file."""
        if filename is None:
            filename = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

        path = self.data_dir / filename
        with open(path, 'w') as f:
            for pred in self.predictions:
                f.write(json.dumps(pred) + '\n')

        print(f"Prediction history saved to {path}")
        return path

    def load_history(self, filepath: str):
        """Load prediction history from file."""
        path = Path(filepath)
        if not path.exists():
            print(f"File not found: {path}")
            return False

        with open(path, 'r') as f:
            for line in f:
                pred = json.loads(line)
                self.predictions.append(pred)

        print(f"Loaded {len(self.predictions)} predictions from {path}")
        return True

    def print_statistics(self):
        """Print statistics."""
        stats = self.get_statistics()
        dist = self.get_digit_distribution()

        print("\n" + "="*60)
        print("PREDICTION STATISTICS")
        print("="*60)
        print(f"Total Predictions: {stats['total_predictions']}")
        print(f"Unique Digits:     {stats['unique_digits_predicted']}")
        print(f"Most Common:       {stats['most_common_digit']}")
        print(f"Cache Hit Rate:    {stats['cache_hit_rate_percent']}%")

        print("\nConfidence Statistics:")
        print(f"  Mean:  {stats['confidence_stats']['mean']:.4f}")
        print(f"  Min:   {stats['confidence_stats']['min']:.4f}")
        print(f"  Max:   {stats['confidence_stats']['max']:.4f}")

        print("\nTiming Statistics (ms):")
        print(f"  Mean:  {stats['timing_stats']['mean_ms']:.2f}ms")
        print(f"  Min:   {stats['timing_stats']['min_ms']:.2f}ms")
        print(f"  Max:   {stats['timing_stats']['max_ms']:.2f}ms")

        print("\nDigit Distribution:")
        for digit, data in sorted(dist.items()):
            print(f"  {digit}: {data['count']:4d} ({data['percentage']:5.1f}%)")

        print("="*60 + "\n")

    def clear_history(self):
        """Clear all history."""
        self.predictions = []
