"""
CNN model for MNIST handwriting recognition.

This module contains a Convolutional Neural Network (CNN) for MNIST digit recognition.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from .base import BaseModel

class CNNModel(BaseModel):
    """Convolutional Neural Network for MNIST digit recognition."""
    
    def __init__(self):
        """Initialize the CNNModel."""
        super().__init__()
        self.model_type = 'cnn'
        self.model_filename = 'mnist_model.h5'
    
    def build(self):
        """Build the CNN model architecture."""
        self.model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(10, activation='softmax')
        ])
        
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, x_train, y_train, x_val, y_val, epochs=7, batch_size=128):
        """
        Train the CNN model.
        
        Args:
            x_train: Training data
            y_train: Training labels
            x_val: Validation data
            y_val: Validation labels
            epochs (int): Number of epochs to train for (default: 15)
            batch_size (int): Batch size for training (default: 128)
            
        Returns:
            History object from model.fit()
        """
        return super().train(x_train, y_train, x_val, y_val, epochs, batch_size) 