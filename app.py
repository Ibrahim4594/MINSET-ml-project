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
from prediction_service import PredictionService

app = Flask(__name__, template_folder=".", static_folder=".")
CORS(app)

# Global variables
model = None
prediction_service = None

def load_model():
    """Load the trained MNIST model."""
    global model, prediction_service
    model_path = "models/mnist_model.h5"

    if not os.path.exists(model_path):
        return False, "Model file not found. Please run train_model.py first."

    try:
        model = keras.models.load_model(model_path)
        prediction_service = PredictionService(model)
        print(f"Model loaded from {model_path}")
        return True, "Model loaded successfully"
    except Exception as e:
        return False, f"Error loading model: {str(e)}"

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
    if prediction_service is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()

        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        image_data = data['image']

        # Make prediction using service
        result = prediction_service.predict_single(image_data)

        if 'error' in result:
            return jsonify(result), 400

        return jsonify(result)

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict digits from multiple images.
    Expects JSON with 'images' field containing list of base64 encoded PNGs.
    """
    if prediction_service is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()

        if 'images' not in data:
            return jsonify({'error': 'No images provided'}), 400

        images_data = data['images']

        if not isinstance(images_data, list):
            return jsonify({'error': 'Images must be a list'}), 400

        if len(images_data) == 0:
            return jsonify({'error': 'Empty images list'}), 400

        # Make batch predictions
        result = prediction_service.predict_batch(images_data)

        return jsonify(result)

    except Exception as e:
        print(f"Error in batch prediction: {str(e)}")
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
