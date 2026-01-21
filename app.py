"""
Flask Backend for MNIST Digit Recognition
Accepts base64 encoded images and returns predictions.
"""
import os
import base64
import io
import time
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__, template_folder=".", static_folder=".")
CORS(app)

# Global model variable
model = None

def load_model():
    """Load the trained MNIST model."""
    global model
    model_path = "models/mnist_model.h5"

    if not os.path.exists(model_path):
        return False, "Model file not found. Please run train_model.py first."

    try:
        model = keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return True, "Model loaded successfully"
    except Exception as e:
        return False, f"Error loading model: {str(e)}"

def preprocess_image(image_data):
    """
    Preprocess image data for model inference.
    Args:
        image_data: Base64 encoded PNG image
    Returns:
        Preprocessed numpy array or None if error
    """
    try:
        # Decode base64
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')

        # Resize to 28x28
        image = image.resize((28, 28), Image.Resampling.LANCZOS)

        # Convert to numpy array and normalize
        image_array = np.array(image, dtype='float32') / 255.0

        # Invert colors (white digit on black background -> black digit on white)
        image_array = 1.0 - image_array

        # Flatten to 784 dimensions
        image_array = image_array.reshape(1, 784)

        return image_array
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

@app.route('/')
def index():
    """Serve the HTML frontend."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict digit from uploaded image.
    Expects JSON with 'image' field containing base64 encoded PNG.
    """
    global model

    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()

        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        image_data = data['image']

        # Preprocess image
        processed_image = preprocess_image(image_data)

        if processed_image is None:
            return jsonify({'error': 'Failed to process image'}), 400

        # Make prediction
        start_time = time.time()
        predictions = model.predict(processed_image, verbose=0)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms

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

        return jsonify({
            'digit': predicted_digit,
            'confidence': confidence,
            'top_3': top_3,
            'inference_time_ms': inference_time
        })

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    # Load model on startup
    success, message = load_model()
    print(message)

    if not success:
        print("Warning: Starting without model. Please run train_model.py first.")

    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
