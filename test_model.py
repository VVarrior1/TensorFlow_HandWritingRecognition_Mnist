import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

# Load the model
model_path = 'models/mnist_model.h5'
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found.")
    exit(1)

print(f"Loading model from {model_path}...")
model = load_model(model_path)

# Load MNIST test data
print("Loading MNIST test data...")
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the data
x_test = x_test.astype('float32') / 255.0
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Create a directory for test outputs
if not os.path.exists('test_outputs'):
    os.makedirs('test_outputs')

# Test the model on a few samples
num_samples = 5
for i in range(num_samples):
    # Get a sample
    sample = x_test[i:i+1]
    true_label = y_test[i]
    
    # Save the sample image
    plt.figure(figsize=(3, 3))
    plt.imshow(sample[0].reshape(28, 28), cmap='gray')
    plt.title(f"True Label: {true_label}")
    plt.savefig(f'test_outputs/sample_{i}.png')
    plt.close()
    
    # Make prediction
    prediction = model.predict(sample)
    predicted_label = np.argmax(prediction[0])
    confidence = np.max(prediction[0]) * 100
    
    print(f"Sample {i}:")
    print(f"  True label: {true_label}")
    print(f"  Predicted label: {predicted_label}")
    print(f"  Confidence: {confidence:.2f}%")
    print(f"  Raw prediction: {prediction[0]}")
    print()
    
    # Save prediction distribution
    plt.figure(figsize=(8, 4))
    plt.bar(range(10), prediction[0])
    plt.xticks(range(10))
    plt.xlabel('Digit')
    plt.ylabel('Probability')
    plt.title(f'Prediction Distribution (True: {true_label}, Pred: {predicted_label})')
    plt.savefig(f'test_outputs/prediction_{i}.png')
    plt.close()

print("Test completed. Check the 'test_outputs' directory for visualizations.")

# Create a custom test digit (a simple 5)
print("\nCreating and testing a custom digit...")
custom_digit = np.zeros((28, 28, 1), dtype=np.float32)

# Draw a simple "5" pattern
custom_digit[5:20, 5:20, 0] = 1.0  # Top horizontal line
custom_digit[5:15, 5:7, 0] = 1.0   # Left vertical line (top half)
custom_digit[15:17, 5:20, 0] = 1.0 # Middle horizontal line
custom_digit[17:25, 18:20, 0] = 1.0 # Right vertical line (bottom half)
custom_digit[23:25, 5:20, 0] = 1.0 # Bottom horizontal line

# Save the custom digit
plt.figure(figsize=(3, 3))
plt.imshow(custom_digit.reshape(28, 28), cmap='gray')
plt.title("Custom Digit (5)")
plt.savefig('test_outputs/custom_digit.png')
plt.close()

# Make prediction on custom digit
custom_prediction = model.predict(np.array([custom_digit]))
custom_predicted_label = np.argmax(custom_prediction[0])
custom_confidence = np.max(custom_prediction[0]) * 100

print(f"Custom digit:")
print(f"  Expected label: 5")
print(f"  Predicted label: {custom_predicted_label}")
print(f"  Confidence: {custom_confidence:.2f}%")
print(f"  Raw prediction: {custom_prediction[0]}")

# Save custom prediction distribution
plt.figure(figsize=(8, 4))
plt.bar(range(10), custom_prediction[0])
plt.xticks(range(10))
plt.xlabel('Digit')
plt.ylabel('Probability')
plt.title(f'Custom Digit Prediction (Pred: {custom_predicted_label})')
plt.savefig('test_outputs/custom_prediction.png')
plt.close() 