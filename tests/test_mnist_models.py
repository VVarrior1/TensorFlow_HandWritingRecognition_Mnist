import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import os
import argparse

def load_and_prepare_test_data(model_type='simple'):
    """Load and prepare the MNIST test dataset."""
    print("Loading MNIST test dataset...")
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize data
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape data based on model type
    if model_type == 'simple':
        # Flatten for simple model
        x_test_reshaped = x_test.reshape(x_test.shape[0], 28*28)
    else:
        # Keep 2D structure for CNN model
        x_test_reshaped = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    # Convert labels to categorical
    y_test_cat = to_categorical(y_test, 10)
    
    return x_test, x_test_reshaped, y_test, y_test_cat

def test_model_accuracy(model, x_test, y_test_cat):
    """Test the model's accuracy on the test set."""
    print("\nEvaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    return test_accuracy, test_loss

def visualize_predictions(model, x_test_orig, x_test_reshaped, y_test, num_samples=5):
    """Visualize predictions for a few samples."""
    # Create output directory if it doesn't exist
    output_dir = os.path.join('tests', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nVisualizing predictions for {num_samples} samples...")
    
    for i in range(num_samples):
        # Get a sample
        sample = x_test_reshaped[i:i+1]
        true_label = y_test[i]
        
        # Save the sample image
        plt.figure(figsize=(3, 3))
        plt.imshow(x_test_orig[i], cmap='gray')
        plt.title(f"True Label: {true_label}")
        plt.savefig(os.path.join(output_dir, f"sample_{i}.png"))
        plt.close()
        
        # Make prediction
        prediction = model.predict(sample, verbose=0)
        predicted_label = np.argmax(prediction[0])
        confidence = np.max(prediction[0]) * 100
        
        print(f"Sample {i}:")
        print(f"  True label: {true_label}")
        print(f"  Predicted label: {predicted_label}")
        print(f"  Confidence: {confidence:.2f}%")
        
        # Save prediction distribution
        plt.figure(figsize=(8, 4))
        plt.bar(range(10), prediction[0])
        plt.xticks(range(10))
        plt.xlabel('Digit')
        plt.ylabel('Probability')
        plt.title(f'Prediction Distribution (True: {true_label}, Pred: {predicted_label})')
        plt.savefig(os.path.join(output_dir, f"prediction_{i}.png"))
        plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def test_custom_digit(model, model_type='simple'):
    """Test the model on a custom-drawn digit."""
    output_dir = os.path.join('tests', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nTesting model on a custom digit (5)...")
    
    # Create a custom test digit (a simple 5)
    custom_digit = np.zeros((28, 28), dtype=np.float32)
    
    # Draw a simple "5" pattern
    custom_digit[5:20, 5:20] = 1.0  # Top horizontal line
    custom_digit[5:15, 5:7] = 1.0   # Left vertical line (top half)
    custom_digit[15:17, 5:20] = 1.0 # Middle horizontal line
    custom_digit[17:25, 18:20] = 1.0 # Right vertical line (bottom half)
    custom_digit[23:25, 5:20] = 1.0 # Bottom horizontal line
    
    # Save the custom digit
    plt.figure(figsize=(3, 3))
    plt.imshow(custom_digit, cmap='gray')
    plt.title("Custom Digit (5)")
    plt.savefig(os.path.join(output_dir, "custom_digit.png"))
    plt.close()
    
    # Prepare for model input
    if model_type == 'simple':
        # Flatten for simple model
        custom_input = custom_digit.reshape(1, 28*28)
    else:
        # Keep 2D structure for CNN model
        custom_input = custom_digit.reshape(1, 28, 28, 1)
    
    # Make prediction
    custom_prediction = model.predict(custom_input, verbose=0)
    custom_predicted_label = np.argmax(custom_prediction[0])
    custom_confidence = np.max(custom_prediction[0]) * 100
    
    print(f"Custom digit:")
    print(f"  Expected label: 5")
    print(f"  Predicted label: {custom_predicted_label}")
    print(f"  Confidence: {custom_confidence:.2f}%")
    
    # Save custom prediction distribution
    plt.figure(figsize=(8, 4))
    plt.bar(range(10), custom_prediction[0])
    plt.xticks(range(10))
    plt.xlabel('Digit')
    plt.ylabel('Probability')
    plt.title(f'Custom Digit Prediction (Pred: {custom_predicted_label})')
    plt.savefig(os.path.join(output_dir, "custom_prediction.png"))
    plt.close()

def main():
    """Main function to test MNIST models."""
    parser = argparse.ArgumentParser(description='Test MNIST handwriting recognition models')
    parser.add_argument('--model', type=str, choices=['simple', 'cnn'], default='simple',
                        help='Model type to test (simple or cnn)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize predictions')
    parser.add_argument('--custom', action='store_true',
                        help='Test on custom digit')
    args = parser.parse_args()
    
    # Determine model path based on type
    if args.model == 'simple':
        model_path = 'models/simple_mnist_model.h5'
        print("Testing Simple MNIST Model")
    else:
        model_path = 'models/mnist_model.h5'
        print("Testing CNN MNIST Model")
    
    print("=" * 40)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        print(f"Please make sure you've trained the {args.model} model first.")
        return
    
    # Load the model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    # Load and prepare test data
    x_test_orig, x_test_reshaped, y_test, y_test_cat = load_and_prepare_test_data(args.model)
    
    # Test model accuracy
    test_accuracy, test_loss = test_model_accuracy(model, x_test_reshaped, y_test_cat)
    
    # Visualize predictions if requested
    if args.visualize:
        visualize_predictions(model, x_test_orig, x_test_reshaped, y_test)
    
    # Test on custom digit if requested
    if args.custom:
        test_custom_digit(model, args.model)
    
    print("\nTesting completed.")

if __name__ == "__main__":
    main() 