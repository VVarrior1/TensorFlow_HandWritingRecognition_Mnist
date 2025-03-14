"""
Simple model for MNIST handwriting recognition.

This module contains a simple feedforward neural network for MNIST digit recognition.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from .base import BaseModel

class SimpleModel(BaseModel):
    """Simple feedforward neural network for MNIST digit recognition."""
    
    def __init__(self):
        """Initialize the SimpleModel."""
        super().__init__()
        self.model_type = 'simple'
        self.model_filename = 'simple_mnist_model.h5'
    
    def build(self):
        """Build the simple model architecture."""
        self.model = Sequential([
            # Simple feedforward network with one hidden layer
            Dense(128, activation='relu', input_shape=(28*28,)),
            Dense(10, activation='softmax')
        ])
        
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, x_train, y_train, x_val, y_val, epochs=5, batch_size=128):
        """
        Train the simple model.
        
        Args:
            x_train: Training data
            y_train: Training labels
            x_val: Validation data
            y_val: Validation labels
            epochs (int): Number of epochs to train for (default: 5)
            batch_size (int): Batch size for training (default: 128)
            
        Returns:
            History object from model.fit()
        """
        return super().train(x_train, y_train, x_val, y_val, epochs, batch_size) 