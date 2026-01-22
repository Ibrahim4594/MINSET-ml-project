"""
Validators - Input validation and sanitization
"""
import base64
import io
from PIL import Image
import numpy as np


class ValidationError(Exception):
    """Validation error exception."""
    pass


class InputValidator:
    """Validate and sanitize user inputs."""

    @staticmethod
    def validate_base64_image(image_data: str, max_size_mb: float = 5.0) -> bool:
        """
        Validate base64 encoded image.

        Args:
            image_data: Base64 encoded image data
            max_size_mb: Maximum allowed file size in MB

        Returns:
            True if valid
        """
        if not isinstance(image_data, str):
            raise ValidationError("Image data must be a string")

        if not image_data.startswith('data:image'):
            raise ValidationError("Invalid image format. Must be data URI")

        try:
            # Remove data URI header
            header, data = image_data.split(',', 1)

            # Decode base64
            image_bytes = base64.b64decode(data)

            # Check size
            size_mb = len(image_bytes) / (1024 * 1024)
            if size_mb > max_size_mb:
                raise ValidationError(f"Image too large: {size_mb:.2f}MB > {max_size_mb}MB")

            # Try to open as image
            image = Image.open(io.BytesIO(image_bytes))

            # Check image mode
            if image.mode not in ('L', 'RGB', 'RGBA'):
                raise ValidationError(f"Unsupported image mode: {image.mode}")

            return True

        except base64.binascii.Error:
            raise ValidationError("Invalid base64 encoding")
        except Exception as e:
            raise ValidationError(f"Invalid image data: {str(e)}")

    @staticmethod
    def validate_batch_images(images: list, max_batch_size: int = 100) -> bool:
        """Validate batch of images."""
        if not isinstance(images, list):
            raise ValidationError("Images must be a list")

        if len(images) == 0:
            raise ValidationError("Empty image list")

        if len(images) > max_batch_size:
            raise ValidationError(f"Batch too large: {len(images)} > {max_batch_size}")

        for i, img in enumerate(images):
            try:
                InputValidator.validate_base64_image(img)
            except ValidationError as e:
                raise ValidationError(f"Image {i} is invalid: {str(e)}")

        return True

    @staticmethod
    def validate_image_array(array: np.ndarray, expected_shape: tuple = (28, 28)) -> bool:
        """Validate numpy image array."""
        if not isinstance(array, np.ndarray):
            raise ValidationError("Array must be numpy ndarray")

        if len(array.shape) != len(expected_shape):
            raise ValidationError(f"Wrong array dimensions: {array.shape} != {expected_shape}")

        if array.shape != expected_shape:
            raise ValidationError(f"Wrong array shape: {array.shape} != {expected_shape}")

        # Check value range
        if np.min(array) < 0 or np.max(array) > 1:
            raise ValidationError("Array values must be in range [0, 1]")

        return True

    @staticmethod
    def validate_prediction_result(result: dict) -> bool:
        """Validate prediction result."""
        required_fields = {'digit', 'confidence', 'inference_time_ms'}

        missing = required_fields - set(result.keys())
        if missing:
            raise ValidationError(f"Missing fields in result: {missing}")

        if not isinstance(result['digit'], int) or not (0 <= result['digit'] <= 9):
            raise ValidationError(f"Invalid digit: {result['digit']}")

        if not isinstance(result['confidence'], (int, float)) or not (0 <= result['confidence'] <= 1):
            raise ValidationError(f"Invalid confidence: {result['confidence']}")

        if not isinstance(result['inference_time_ms'], (int, float)) or result['inference_time_ms'] < 0:
            raise ValidationError(f"Invalid inference time: {result['inference_time_ms']}")

        return True


class ErrorHandler:
    """Handle and format errors."""

    @staticmethod
    def format_error(error: Exception, status_code: int = 400) -> dict:
        """Format error for API response."""
        error_type = type(error).__name__

        if isinstance(error, ValidationError):
            status_code = 400
        elif isinstance(error, FileNotFoundError):
            status_code = 404
        elif isinstance(error, MemoryError):
            status_code = 507
        else:
            status_code = 500

        return {
            'error': str(error),
            'error_type': error_type,
            'status_code': status_code
        }

    @staticmethod
    def validate_and_handle(validation_func, *args, **kwargs):
        """Validate with error handling."""
        try:
            return validation_func(*args, **kwargs)
        except ValidationError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Validation error: {str(e)}"

        return True, None
