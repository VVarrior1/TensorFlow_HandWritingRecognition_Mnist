import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import sys
import os

def preprocess_image(image_path):
    """Preprocess a custom image to match MNIST format."""
    try:
        # Open the image
        img = Image.open(image_path).convert('L').resize((28,28))
        
        # Invert if needed (MNIST has white digits on black background)
        # Check if the image is mostly white (background) or mostly black
        if np.mean(img) > 128:
            img = ImageOps.invert(img)
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        # Reshape to match model input shape (1, 28, 28, 1)
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def predict_digit(model, image_array):
    """Predict the digit in the image using the trained model."""
    # Make prediction
    prediction = model.predict(image_array)
    
    # Get the predicted class (digit)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    return predicted_digit, confidence

def display_prediction(image_path, image_array, predicted_digit, confidence):
    """Display the image and prediction."""
    plt.figure(figsize=(6, 3))
    
    # Display original image
    plt.subplot(1, 2, 1)
    img = Image.open(image_path).convert('L')
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    # Display preprocessed image
    plt.subplot(1, 2, 2)
    plt.imshow(image_array.reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted_digit}\nConfidence: {confidence:.2f}%")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to test the model with a custom image."""
    # Check if model exists
    model_path = 'models/mnist_model.h5'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        print("Please train the model first by running 'python mnist_handwriting_recognition.py'")
        return
    
    # Check if image path is provided
    if len(sys.argv) < 2:
        print("Usage: python test_custom_image.py <path_to_image>")
        return
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return
    
    # Load the trained model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    # Preprocess the image
    print(f"Processing image: {image_path}")
    image_array = preprocess_image(image_path)
    if image_array is None:
        return
    
    # Make prediction
    predicted_digit, confidence = predict_digit(model, image_array)
    print(f"Prediction: {predicted_digit}")
    print(f"Confidence: {confidence:.2f}%")
    
    # Display the result
    display_prediction(image_path, image_array, predicted_digit, confidence)

if __name__ == "__main__":
    main() 