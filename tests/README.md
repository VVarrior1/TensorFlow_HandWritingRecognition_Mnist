# MNIST Model Testing

This directory contains scripts for testing the MNIST handwriting recognition models.

## Main Test Script

The main test script is `test_mnist_models.py`, which can test both the simple and CNN models.

### Usage

```bash
# Test the simple model (default)
python tests/test_mnist_models.py

# Test the CNN model
python tests/test_mnist_models.py --model cnn

# Test with visualizations
python tests/test_mnist_models.py --visualize

# Test with custom digit
python tests/test_mnist_models.py --custom

# Test with all options
python tests/test_mnist_models.py --model cnn --visualize --custom
```

### Command Line Arguments

- `--model`: Specify which model to test (`simple` or `cnn`). Default is `simple`.
- `--visualize`: Generate visualizations of test predictions.
- `--custom`: Test the model on a custom-drawn digit (a simple "5").

### Output

Test results will be displayed in the console, and visualizations will be saved to the `tests/outputs` directory.

## Test Outputs

The `outputs` directory will contain:

- Sample images from the MNIST test set
- Prediction distribution visualizations
- Custom digit visualization and prediction

## Examples

1. To quickly check the accuracy of the simple model:

   ```bash
   python tests/test_mnist_models.py
   ```

2. To compare the CNN model's performance with visualizations:

   ```bash
   python tests/test_mnist_models.py --model cnn --visualize
   ```

3. To test both models on a custom digit:
   ```bash
   python tests/test_mnist_models.py --model simple --custom
   python tests/test_mnist_models.py --model cnn --custom
   ```
