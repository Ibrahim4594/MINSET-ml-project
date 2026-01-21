# Usage Examples

Complete examples for using the MNIST digit recognition system.

## Basic Usage

### Web Interface

1. Open http://localhost:5000 in your browser
2. Draw a digit (0-9) on the canvas
3. Click "Predict" to get the model's prediction
4. View confidence scores and top-3 predictions

### Drawing Tools

- **Brush**: Draw with customizable pen size (2-30px)
- **Eraser**: Remove parts of your drawing
- **Undo/Redo**: Use Ctrl+Z / Ctrl+Y or buttons
- **Clear**: Start over with a fresh canvas
- **Upload**: Load images from your computer

## API Usage

### Single Prediction

```python
import requests
import base64
from PIL import Image
from io import BytesIO

# Load image
img = Image.open("digit.png").convert('L').resize((280, 280))
img_bytes = BytesIO()
img.save(img_bytes, format='PNG')
img_base64 = f"data:image/png;base64,{base64.b64encode(img_bytes.getvalue()).decode()}"

# Send request
response = requests.post('http://localhost:5000/predict', json={
    'image': img_base64
})

# Get result
result = response.json()
print(f"Predicted digit: {result['digit']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Batch Predictions

```python
import requests
import base64
from PIL import Image
from io import BytesIO

# Prepare multiple images
images = []
for img_path in ['digit1.png', 'digit2.png', 'digit3.png']:
    img = Image.open(img_path).convert('L').resize((280, 280))
    img_bytes = BytesIO()
    img.save(img_bytes, format='PNG')
    img_base64 = f"data:image/png;base64,{base64.b64encode(img_bytes.getvalue()).decode()}"
    images.append(img_base64)

# Send batch request
response = requests.post('http://localhost:5000/predict_batch', json={
    'images': images
})

# Get results
result = response.json()
print(f"Processed: {result['count']} images")
print(f"Successful: {result['successful']}")
print(f"Average time: {result['average_time_ms']:.2f}ms")
```

## Training Examples

### Default Training

```bash
python train_model.py
```

Trains model with default architecture and saves to `models/mnist_model.h5`.

### Custom Training

```python
from advanced_training import AdvancedTrainer

trainer = AdvancedTrainer()

config = {
    'architecture': [
        {'type': 'dense', 'units': 256, 'activation': 'relu'},
        {'type': 'batchnorm'},
        {'type': 'dropout', 'rate': 0.3},
        {'type': 'dense', 'units': 128, 'activation': 'relu'},
        {'type': 'dropout', 'rate': 0.2},
        {'type': 'dense', 'units': 10, 'activation': 'softmax'}
    ],
    'learning_rate': 0.001,
    'batch_size': 64,
    'epochs': 30,
    'augmentation': True,
    'optimizer': 'adam'
}

model, history = trainer.train_with_config(config)
```

### Hyperparameter Sweep

```python
from advanced_training import AdvancedTrainer

trainer = AdvancedTrainer()

configs = [
    {'learning_rate': 0.001, 'batch_size': 128, 'epochs': 15},
    {'learning_rate': 0.0005, 'batch_size': 64, 'epochs': 20},
    {'learning_rate': 0.002, 'batch_size': 256, 'epochs': 10},
]

results = trainer.hyperparameter_sweep(configs)
```

## Performance Analysis

### Benchmark Model

```bash
python benchmark.py
```

Generates comprehensive performance report including:
- Single inference timing
- Batch inference performance
- Image preprocessing time
- End-to-end prediction latency
- Model size analysis

### Using Performance Monitor

```python
from logger import get_loggers

loggers = get_loggers()
perf_monitor = loggers['performance']

# Start timing
perf_monitor.start_timer('prediction')

# ... do work ...

# End timing
elapsed = perf_monitor.end_timer('prediction')

# Print summary
perf_monitor.print_summary()
```

## Model Optimization

### Quantization

```bash
python quantize_model.py
```

Creates quantized model for faster inference and smaller size.

### Using Quantized Model

```python
from model_optimizer import ModelOptimizer

optimizer = ModelOptimizer()
quantized_path = optimizer.quantize_model('models/mnist_model.h5')

# Use optimized inference script
from inference_optimized import OptimizedMNISTPredictor

predictor = OptimizedMNISTPredictor(quantized_path)
digit = predictor.predict(image_data)
```

## Model Explainability

### Generate Explanations

```python
from model_explainability import ModelExplainer
import numpy as np

# Load model and create explainer
from tensorflow import keras
model = keras.models.load_model('models/mnist_model.h5')
explainer = ModelExplainer(model)

# Load image
image_array = np.random.randn(1, 784)  # Your image here

# Generate explanation
explanation = explainer.explain_prediction(image_array, predicted_digit=7)

# Generate report
report = explainer.generate_report(image_array, 7, model_accuracy=0.99)
print(report)
```

## Caching

### Using Prediction Cache

```python
from cache_manager import CachedPredictionService
from prediction_service import PredictionService

# Create cached service
service = PredictionService(model)
cached_service = CachedPredictionService(service, cache_size=100)

# Use it (automatically caches results)
result1 = cached_service.predict_single(image_data1)
result2 = cached_service.predict_single(image_data1)  # From cache!

# Get stats
stats = cached_service.get_cache_stats()
print(f"Cache hit rate: {stats['single_cache']['hit_rate_percent']:.1f}%")
```

## Data Augmentation

### Apply Augmentations

```python
from data_augmentation import *
import numpy as np

# Create sample image
image = np.random.randn(28, 28)

# Apply different augmentations
noisy = add_noise(image, noise_factor=0.1)
elastic = elastic_transform(image, alpha=30)

# Use ImageDataGenerator
from data_augmentation import apply_augmentation
aug_generator = apply_augmentation(x_train, y_train, batch_size=128)
```

## Testing

### Run Unit Tests

```bash
# Run all tests
pytest tests.py -v

# Run specific test class
pytest tests.py::TestPredictionService -v

# Run with coverage
pytest tests.py --cov=. --cov-report=html
```

### Write Custom Test

```python
import unittest
from prediction_service import PredictionService

class TestCustom(unittest.TestCase):
    def test_my_feature(self):
        # Your test here
        pass

if __name__ == '__main__':
    unittest.main()
```

## Docker Usage

### Build Image

```bash
docker build -t mnist-recognition .
```

### Run Container

```bash
docker run -p 5000:5000 mnist-recognition
```

### Docker Compose

```bash
docker-compose up
```

## Logging

### Access Logs

```bash
# View application logs
cat logs/mnist_app.log

# View structured JSON logs
cat logs/mnist_app_structured.jsonl
```

### Custom Logging

```python
from logger import get_loggers

loggers = get_loggers()
logger = loggers['main']
request_logger = loggers['request']

logger.info("Starting training", {'epochs': 20, 'batch_size': 128})
request_logger.log_prediction(digit=7, confidence=0.99, duration_ms=8.5)
```

## API Documentation

Visit http://localhost:5000/apidocs for interactive Swagger documentation.

## More Help

- Check README.md for setup instructions
- See CONTRIBUTING.md for development guidelines
- Open an issue for questions or bugs
