import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, render_template, redirect, url_for
import base64
from PIL import Image
import io
from waitress import serve


app = Flask(__name__)

# Model paths
CNN_MODEL_PATH = 'models/mnist_model.h5'
SIMPLE_MODEL_PATH = 'models/simple_mnist_model.h5'

# Model types and their corresponding paths
MODEL_PATHS = {
    'cnn': CNN_MODEL_PATH,
    'simple': SIMPLE_MODEL_PATH
}

# Default to CNN model if available, otherwise simple model
def get_default_model_type():
    if os.path.exists(CNN_MODEL_PATH):
        return 'cnn'
    elif os.path.exists(SIMPLE_MODEL_PATH):
        return 'simple'
    return None

current_model_type = get_default_model_type()
current_model_path = MODEL_PATHS.get(current_model_type) if current_model_type else None

def load_mnist_model(model_path):
    """Load the trained MNIST model if it exists."""
    if model_path and os.path.exists(model_path):
        return load_model(model_path)
    return None

model = load_mnist_model(current_model_path)

def preprocess_image_data(image_data, model_type='cnn'):
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
        if model_type == 'simple':
            return img_array.reshape(1, 28*28)
        else:
            return img_array.reshape(1, 28, 28, 1)
        
    except Exception as e:
        app.logger.error(f"Error processing image: {e}")
        return None

@app.route('/')
def index():
    """Render the main page."""
    # Check which models are available
    available_models = {}
    for model_type, path in MODEL_PATHS.items():
        available_models[model_type] = os.path.exists(path)
    
    return render_template(
        'index.html', 
        model_loaded=model is not None,
        available_models=available_models,
        current_model=current_model_type
    )

@app.route('/switch_model/<model_type>')
def switch_model(model_type):
    """Switch between available models."""
    global model, current_model_path, current_model_type
    
    if model_type in MODEL_PATHS and os.path.exists(MODEL_PATHS[model_type]):
        current_model_type = model_type
        current_model_path = MODEL_PATHS[model_type]
        model = load_mnist_model(current_model_path)
    
    return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict():
    """Predict the digit from the drawn image."""
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train a model first.'}), 400
    
    # Get image data from request
    image_data = request.json.get('image')
    if not image_data:
        return jsonify({'error': 'No image data provided'}), 400
    
    # Preprocess the image
    processed_image = preprocess_image_data(image_data, current_model_type)
    if processed_image is None:
        return jsonify({'error': 'Error processing image'}), 400
    
    try:
        # Make prediction
        prediction = model.predict(processed_image)
        predicted_digit = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]) * 100)
        
        return jsonify({
            'digit': predicted_digit,
            'confidence': confidence,
            'model_type': current_model_type
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
    
    # Check which models are available
    available_models = []
    for model_type, path in MODEL_PATHS.items():
        if os.path.exists(path):
            available_models.append(model_type.upper())
    
    if not available_models:
        print("Warning: No models found. Please train at least one model first by running:")
        print("  python train.py --model cnn          # for CNN model")
        print("  python train.py --model simple       # for simple model")
    else:
        print(f"Models available: {', '.join(available_models)}")
        if current_model_type:
            print(f"Currently using: {current_model_type.upper()} model")
        else:
            print("No model selected.")
    
    # Use production server and port
    serve(app, host='0.0.0.0', port=8080) 