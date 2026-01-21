"""
Cache Manager - LRU cache for predictions to improve performance
"""
import hashlib
from collections import OrderedDict
import numpy as np


class PredictionCache:
    """LRU cache for model predictions."""

    def __init__(self, max_size=100):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of cached predictions
        """
        self.max_size = max_size
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _hash_input(self, image_array):
        """Generate hash for image array."""
        # Quantize to reduce hash sensitivity to small variations
        quantized = (image_array * 100).astype(np.uint8)
        hash_bytes = hashlib.md5(quantized.tobytes()).digest()
        return hash_bytes.hex()[:16]

    def get(self, image_array):
        """
        Get prediction from cache.

        Args:
            image_array: Input image array

        Returns:
            Cached prediction or None if not in cache
        """
        key = self._hash_input(image_array)

        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]

        self.misses += 1
        return None

    def put(self, image_array, prediction):
        """
        Store prediction in cache.

        Args:
            image_array: Input image array
            prediction: Model prediction result
        """
        key = self._hash_input(image_array)

        if key in self.cache:
            # Update existing
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.max_size:
            # Remove least recently used
            self.cache.popitem(last=False)

        self.cache[key] = prediction

    def get_stats(self):
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0

        return {
            'hits': self.hits,
            'misses': self.misses,
            'total_requests': total,
            'hit_rate_percent': hit_rate,
            'cache_size': len(self.cache),
            'max_size': self.max_size
        }

    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def __len__(self):
        """Return cache size."""
        return len(self.cache)

    def __repr__(self):
        """Return cache statistics string."""
        stats = self.get_stats()
        return f"Cache(size={stats['cache_size']}/{stats['max_size']}, hit_rate={stats['hit_rate_percent']:.1f}%)"


class BatchPredictionCache:
    """Cache for batch predictions."""

    def __init__(self, max_size=50):
        """
        Initialize batch cache.

        Args:
            max_size: Maximum number of cached batch results
        """
        self.max_size = max_size
        self.cache = OrderedDict()

    def _hash_batch(self, images_array):
        """Generate hash for batch of images."""
        combined = tuple(np.sum(img) for img in images_array)
        hash_str = hashlib.md5(str(combined).encode()).hexdigest()
        return hash_str[:16]

    def get(self, images_array):
        """Get batch prediction from cache."""
        key = self._hash_batch(images_array)

        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]

        return None

    def put(self, images_array, predictions):
        """Store batch predictions in cache."""
        key = self._hash_batch(images_array)

        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)

        self.cache[key] = predictions

    def clear(self):
        """Clear the cache."""
        self.cache.clear()

    def __len__(self):
        """Return cache size."""
        return len(self.cache)


class CachedPredictionService:
    """Prediction service with caching."""

    def __init__(self, prediction_service, cache_size=100):
        """
        Initialize cached service.

        Args:
            prediction_service: Base prediction service
            cache_size: Cache size
        """
        self.service = prediction_service
        self.cache = PredictionCache(max_size=cache_size)
        self.batch_cache = BatchPredictionCache(max_size=cache_size // 2)

    def predict_single(self, image_data):
        """
        Make prediction with caching.

        Args:
            image_data: Base64 encoded image

        Returns:
            Prediction result
        """
        # Preprocess to get array
        image_array = self.service.preprocess_single_image(image_data)

        if image_array is None:
            return {'error': 'Failed to process image'}

        # Check cache
        cached_result = self.cache.get(image_array)
        if cached_result is not None:
            cached_result['from_cache'] = True
            return cached_result

        # Make prediction
        result = self.service.predict_single(image_data)

        if 'digit' in result:
            # Cache successful prediction
            self.cache.put(image_array, result)
            result['from_cache'] = False

        return result

    def predict_batch(self, images_data):
        """
        Make batch predictions with caching.

        Args:
            images_data: List of base64 encoded images

        Returns:
            Batch prediction result
        """
        # For batch caching, use the raw result
        result = self.service.predict_batch(images_data)

        # Add cache stats to result
        result['cache_stats'] = self.cache.get_stats()

        return result

    def get_cache_stats(self):
        """Get cache statistics."""
        return {
            'single_cache': self.cache.get_stats(),
            'batch_cache': {
                'size': len(self.batch_cache),
                'max_size': self.batch_cache.max_size
            }
        }

    def clear_cache(self):
        """Clear all caches."""
        self.cache.clear()
        self.batch_cache.clear()
