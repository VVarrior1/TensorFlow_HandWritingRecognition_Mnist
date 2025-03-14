import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, render_template
import base64
from PIL import Image
import io
from waitress import serve


app = Flask(__name__)

# Load the model
MODEL_PATH = 'models/mnist_model.h5'

def load_mnist_model():
    """Load the trained MNIST model if it exists."""
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    return None

model = load_mnist_model()

def preprocess_image_data(image_data):
    """Preprocess the image data from canvas."""
    try:
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('L')
        
        # Directly resize to 28x28 to preserve quality
        image = image.resize((28, 28), Image.Resampling.BICUBIC)
        
        # Convert to numpy array and normalize
        img_array = np.array(image).astype('float32') / 255.0
        
        # Invert if background is light (MNIST has white digits on black background)
        if np.mean(img_array) > 0.5:
            img_array = 1.0 - img_array
            
        # Enhance contrast to match MNIST style
        min_val = np.min(img_array)
        max_val = np.max(img_array)
        if max_val > min_val:
            img_array = (img_array - min_val) / (max_val - min_val)
        
        # Reshape for model
        return img_array.reshape(1, 28, 28, 1)
        
    except Exception as e:
        app.logger.error(f"Error processing image: {e}")
        return None

@app.route('/')
def index():
    """Render the main page."""
    model_loaded = model is not None
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict the digit from the drawn image."""
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 400
    
    # Get image data from request
    image_data = request.json.get('image')
    if not image_data:
        return jsonify({'error': 'No image data provided'}), 400
    
    # Preprocess the image
    processed_image = preprocess_image_data(image_data)
    if processed_image is None:
        return jsonify({'error': 'Error processing image'}), 400
    
    try:
        # Make prediction
        prediction = model.predict(processed_image)
        predicted_digit = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]) * 100)
        
        return jsonify({
            'digit': predicted_digit,
            'confidence': confidence
        })
    except Exception as e:
        app.logger.error(f"Error making prediction: {e}")
        return jsonify({'error': 'Error making prediction'}), 500

@app.route('/train')
def train_info():
    """Provide information about training the model."""
    return render_template('train.html')

if __name__ == '__main__':
    # Create required directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # Check if model is loaded
    if model is None:
        print("Warning: Model not found. Please train the model first by running 'python mnist_handwriting_recognition.py'")
    else:
        print("Model loaded successfully!")
    
    # Use production server and port
    serve(app, host='0.0.0.0', port=8080) 