"""
MNIST Handwriting Recognition Models Module

This module contains different model architectures for MNIST handwriting recognition.
"""

from .base import load_and_prepare_mnist_data
from .simple import SimpleModel
from .cnn import CNNModel

# Dictionary of available models
AVAILABLE_MODELS = {
    'simple': SimpleModel,
    'cnn': CNNModel
}

def get_model(model_name):
    """
    Get a model instance by name.
    
    Args:
        model_name (str): Name of the model to get ('simple' or 'cnn')
        
    Returns:
        Model class or None if model_name is not recognized
    """
    return AVAILABLE_MODELS.get(model_name) 