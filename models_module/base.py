"""
Base module for MNIST handwriting recognition models.

This module contains common functionality used by all models.
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
import os

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_and_prepare_mnist_data(model_type='cnn'):
    """
    Load and prepare the MNIST dataset.
    
    Args:
        model_type (str): Type of model ('simple' or 'cnn') to prepare data for
        
    Returns:
        tuple: ((x_train, y_train), (x_val, y_val), (x_test, y_test))
    """
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape data based on model type
    if model_type == 'simple':
        # Flatten for simple model
        x_train = x_train.reshape(x_train.shape[0], 28*28)
        x_test = x_test.reshape(x_test.shape[0], 28*28)
    else:
        # Keep 2D structure for CNN model
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    # Convert labels to categorical
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    # Split training data to create validation set
    validation_split = 0.1
    split_idx = int(x_train.shape[0] * (1 - validation_split))
    x_val = x_train[split_idx:]
    y_val = y_train[split_idx:]
    x_train = x_train[:split_idx]
    y_train = y_train[:split_idx]
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

class BaseModel:
    """Base class for all MNIST models."""
    
    def __init__(self):
        """Initialize the model."""
        self.model = None
        self.model_type = None
        self.model_filename = None
    
    def build(self):
        """Build the model architecture. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement build()")
    
    def train(self, x_train, y_train, x_val, y_val, epochs=10, batch_size=128):
        """
        Train the model.
        
        Args:
            x_train: Training data
            y_train: Training labels
            x_val: Validation data
            y_val: Validation labels
            epochs (int): Number of epochs to train for
            batch_size (int): Batch size for training
            
        Returns:
            History object from model.fit()
        """
        if self.model is None:
            self.build()
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Train with early stopping and model checkpointing
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'models/{self.model_filename}',
                save_best_only=True,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]
        
        return self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
    
    def evaluate(self, x_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            x_test: Test data
            y_test: Test labels
            
        Returns:
            tuple: (test_loss, test_accuracy)
        """
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build() first.")
        
        return self.model.evaluate(x_test, y_test, verbose=0)
    
    def save(self):
        """Save the model to disk."""
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build() first.")
        
        os.makedirs('models', exist_ok=True)
        self.model.save(f'models/{self.model_filename}') 