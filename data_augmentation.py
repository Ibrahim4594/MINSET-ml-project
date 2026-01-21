"""
Data Augmentation - Techniques to improve model generalization
"""
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_augmentation_generator():
    """
    Create data augmentation generator for training.

    Augmentations:
    - Rotation: ±15 degrees
    - Width/Height shift: ±10%
    - Zoom: ±10%
    - Shear: ±10%
    """
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1,
        fill_mode='nearest'
    )


def apply_augmentation(x_train, y_train, batch_size=128, steps_per_epoch=None):
    """
    Apply data augmentation to training data.

    Args:
        x_train: Training images (normalized)
        y_train: Training labels
        batch_size: Batch size for augmentation
        steps_per_epoch: Steps per epoch (None = use all data)

    Returns:
        Augmented data generator
    """
    augmentor = create_augmentation_generator()

    # Reshape for ImageDataGenerator (expects images as 28x28x1)
    x_train_reshaped = x_train.reshape(-1, 28, 28, 1)

    return augmentor.flow(
        x_train_reshaped,
        y_train,
        batch_size=batch_size,
        shuffle=True
    )


def add_noise(x, noise_factor=0.1):
    """Add Gaussian noise to images."""
    return x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x.shape)


def elastic_transform(x, alpha=30, sigma=5):
    """Apply elastic deformation to images."""
    from scipy.ndimage import gaussian_filter, map_coordinates

    shape = x.shape
    dx = gaussian_filter((np.random.random(shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.random(shape) * 2 - 1), sigma) * alpha

    x_coords, y_coords = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x_coords + dx, (-1, 1)), np.reshape(y_coords + dy, (-1, 1))

    return map_coordinates(x, indices, order=1, mode='reflect').reshape(shape)


def mixup_batch(x_batch, y_batch, alpha=0.2):
    """
    Apply Mixup augmentation to a batch.
    Interpolates between random pairs of samples.
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = x_batch.shape[0]

    index = np.random.permutation(batch_size)
    x_mixed = lam * x_batch + (1 - lam) * x_batch[index, :]
    y_mixed = lam * y_batch + (1 - lam) * y_batch[index]

    return x_mixed, y_mixed
