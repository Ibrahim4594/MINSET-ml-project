"""
Prediction Service - Handle single and batch predictions
"""
import io
import base64
import time
import numpy as np
from PIL import Image


class PredictionService:
    """Service for making predictions on MNIST images."""

    def __init__(self, model):
        self.model = model

    def preprocess_single_image(self, image_data):
        """
        Preprocess a single image for prediction.

        Args:
            image_data: Base64 encoded PNG image

        Returns:
            Preprocessed numpy array or None if error
        """
        try:
            # Decode base64
            image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
            image = Image.open(io.BytesIO(image_bytes))

            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')

            # Resize to 28x28
            image = image.resize((28, 28), Image.Resampling.LANCZOS)

            # Convert to numpy array and normalize
            image_array = np.array(image, dtype='float32') / 255.0

            # Invert colors
            image_array = 1.0 - image_array

            # Flatten to 784 dimensions
            image_array = image_array.reshape(1, 784)

            return image_array
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            return None

    def predict_single(self, image_data):
        """
        Make a prediction on a single image.

        Returns:
            Dict with prediction results
        """
        start_time = time.time()

        processed_image = self.preprocess_single_image(image_data)
        if processed_image is None:
            return {'error': 'Failed to process image'}

        predictions = self.model.predict(processed_image, verbose=0)
        inference_time = (time.time() - start_time) * 1000

        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_digit])

        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3 = [
            {
                'digit': int(idx),
                'confidence': float(predictions[0][idx])
            }
            for idx in top_3_indices
        ]

        return {
            'digit': predicted_digit,
            'confidence': confidence,
            'top_3': top_3,
            'inference_time_ms': inference_time,
            'timestamp': time.time()
        }

    def predict_batch(self, images_data):
        """
        Make predictions on multiple images.

        Args:
            images_data: List of base64 encoded PNG images

        Returns:
            List of prediction results
        """
        results = []
        total_start = time.time()

        for i, image_data in enumerate(images_data):
            try:
                result = self.predict_single(image_data)
                result['index'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'index': i,
                    'error': str(e)
                })

        total_time = (time.time() - total_start) * 1000
        avg_time = total_time / len(images_data) if images_data else 0

        return {
            'predictions': results,
            'total_time_ms': total_time,
            'average_time_ms': avg_time,
            'count': len(images_data),
            'successful': len([r for r in results if 'digit' in r])
        }

    def get_prediction_confidence_stats(self, predictions):
        """
        Calculate confidence statistics from predictions.

        Args:
            predictions: List of prediction results

        Returns:
            Statistics dict
        """
        confidences = [p['confidence'] for p in predictions if 'confidence' in p]

        if not confidences:
            return {}

        return {
            'mean_confidence': float(np.mean(confidences)),
            'min_confidence': float(np.min(confidences)),
            'max_confidence': float(np.max(confidences)),
            'std_confidence': float(np.std(confidences))
        }
