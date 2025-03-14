# Handwriting Digit Recognition Web App

A web application that recognizes handwritten digits using neural networks trained on the MNIST dataset. Users can draw digits on a canvas and get real-time predictions.

<img src="demo.png" alt="Demo of the Handwriting Recognition App" width="400"/>

## Features

- Interactive drawing canvas
- Real-time digit recognition
- Two model options:
  - Simple feedforward neural network (faster training)
  - Convolutional Neural Network (CNN) (higher accuracy)
- Clean and intuitive user interface
- Production-ready deployment setup
- Modular architecture for easy extension with new models

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/handwriting-recognition.git
cd handwriting-recognition
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── app.py                  # Main Flask web application
├── train.py                # Training script for all models
├── models/                 # Saved model files
├── models_module/          # Model definitions
│   ├── __init__.py         # Module initialization
│   ├── base.py             # Base model class and common functions
│   ├── simple.py           # Simple feedforward model
│   ├── cnn.py              # CNN model
├── templates/              # HTML templates
│   ├── index.html          # Main application page
│   ├── train.html          # Training information page
├── tests/                  # Test scripts
│   ├── test_mnist_models.py # Comprehensive test script
│   ├── outputs/            # Test visualizations
```

## Usage

1. Train the models:

```bash
# Train the CNN model (default)
python train.py

# Train the simple model
python train.py --model simple

# Customize training parameters
python train.py --model cnn --epochs 20 --batch-size 64
```

2. Start the web server:

```bash
python app.py
```

3. Open your browser and navigate to:

```
http://localhost:8080
```

## Testing

The project includes a comprehensive testing framework in the `tests` directory.

### Running Tests

```bash
# Test the simple model (default)
python tests/test_mnist_models.py --model simple

# Test the CNN model
python tests/test_mnist_models.py --model cnn

# Test with visualizations
python tests/test_mnist_models.py --visualize

# Test with custom digit
python tests/test_mnist_models.py --custom

# Test with all options
python tests/test_mnist_models.py --model cnn --visualize --custom
```

Test results will be displayed in the console, and visualizations will be saved to the `tests/outputs` directory.

See the [tests/README.md](tests/README.md) file for more detailed testing information.

## Model Architectures

### CNN Model

The CNN model consists of:

- 2 Convolutional layers with MaxPooling
- Dense layer with dropout for regularization
- Output layer for 10 digits
- Typically achieves 98-99% accuracy on MNIST test set

### Simple Model

The simple model consists of:

- Single hidden layer with 128 neurons
- Output layer for 10 digits
- Typically achieves 95-97% accuracy on MNIST test set
- Trains much faster than the CNN model

## Adding New Models

To add a new model:

1. Create a new file in the `models_module` directory (e.g., `advanced_cnn.py`)
2. Define a new model class that inherits from `BaseModel`
3. Implement the required methods (`__init__`, `build`, etc.)
4. Add the model to the `AVAILABLE_MODELS` dictionary in `models_module/__init__.py`

## Acknowledgments

- MNIST dataset from TensorFlow
- TensorFlow and Keras teams
- Flask web framework
