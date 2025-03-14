"""
MNIST Handwriting Recognition Model Training Script

This script trains a model for MNIST handwriting recognition using the models_module.
"""

import argparse
import os
import tensorflow as tf
import numpy as np
from models_module import get_model, load_and_prepare_mnist_data

def main():
    """Train and save an MNIST model."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train MNIST handwriting recognition models')
    parser.add_argument('--model', type=str, choices=['simple', 'cnn'], default='cnn',
                        help='Model type to train (simple or cnn). Default: cnn')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs to train for. Default: model-specific (5 for simple, 15 for cnn)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training. Default: 128')
    args = parser.parse_args()
    
    model_type = args.model
    
    # Print header
    if model_type == 'simple':
        print("Simple MNIST Handwriting Recognition")
        print("=" * 40)
    else:
        print("CNN MNIST Handwriting Recognition Model Training")
        print("=" * 50)
    
    # Load and prepare data
    data = load_and_prepare_mnist_data(model_type)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data
    
    # Get the model
    model_class = get_model(model_type)
    if model_class is None:
        print(f"Error: Unknown model type '{model_type}'")
        return
    
    model = model_class()
    model.build()
    
    print("\nTraining model...")
    
    # Determine epochs if not specified
    epochs = args.epochs
    if epochs is None:
        epochs = 5 if model_type == 'simple' else 15
    
    # Train the model
    model.train(x_train, y_train, x_val, y_val, epochs=epochs, batch_size=args.batch_size)
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"\nModel saved as 'models/{model.model_filename}'")

if __name__ == "__main__":
    main() 