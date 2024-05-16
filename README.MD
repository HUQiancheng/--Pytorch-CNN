### README

# CNN with PyTorch

This project implements a Convolutional Neural Network (CNN) using PyTorch, designed for image classification tasks. The project includes custom dataset loading, model definition, training, and evaluation, with detailed visualization of training progress and model predictions.

## Project Structure

```
.
├── data
│   └── dataset
│       ├── 0
│       ├── 1
│       ├── 2
│       ├── 3
│       ├── 4
│       ├── 5
│       ├── 6
│       ├── 7
│       ├── 8
│       └── 9
├── src
│   ├── data
│   │   ├── CNN_Dataset.py
│   ├── inference
│   │   ├── CNN_Inference.py
│   ├── model
│   │   ├── CNNModel.py
│   ├── training
│   │   ├── train_CNN.py
│   ├── util
│   │   ├── visualization.py
│   │   ├── model.py
│   ├── __init__.py
├── train.py
└── README.md
```

## Modules Description

### data
- **CNN_Dataset.py**: Defines the `CNN_Dataset` class for loading and processing image data. Includes data augmentation and normalization.

### inference
- **CNN_Inference.py**: Contains the `load_model` function for loading a trained model from a checkpoint.

### model
- **CNNModel.py**: Defines the `CNNModel` class, which contains the architecture of the CNN used in this project.

### training
- **train_CNN.py**: Contains the `train` function for training the model and the `create_dataloader` function for creating data loaders for training and validation datasets.

### util
- **visualization.py**: Contains functions for visualizing training history (`plot_training_history`) and model predictions (`plot_predictions`).
- **model.py**: Contains the `summarize_model` function for printing a summary of the model.

### Main Script
- **train.py**: Main script to train and evaluate the model. Handles dataset loading, model instantiation, training, and evaluation.

## Requirements

- Python 3.7+
- PyTorch 1.6.0+
- torchvision
- matplotlib
- numpy
- scikit-learn

You can install the required packages using:
```
pip install -r requirements.txt
```

## Usage

1. **Prepare Dataset**: Ensure your dataset is structured correctly within the `data/dataset` directory, with subdirectories for each class (0, 1, 2, ..., 9) containing the corresponding images.

2. **Training the Model**:
    - Run the `train.py` script to start training the model.
    - The script will check if a pre-trained model exists. If found, it will ask whether to use it or train a new model.
    - The training progress, including loss and accuracy, will be printed and visualized.

    ```bash
    python train.py
    ```

3. **Visualizing Training History and Predictions**:
    - After training, the script will plot the training and validation loss and accuracy.
    - It will also display a grid of images with their predicted labels.

## Functions Overview

### `CNN_Dataset`
- **`__init__(self, root_dir, transform=None, train=True, train_ratio=0.8)`**:
    Initializes the dataset by loading image paths and labels, and splits the data into training and validation sets.
- **`__getitem__(self, idx)`**:
    Returns a preprocessed image and its label.
- **`__len__(self)`**:
    Returns the total number of images in the dataset.

### `load_model`
- **`load_model(model_path)`**:
    Loads a trained model from the specified path.

### `CNNModel`
- **`__init__(self)`**:
    Initializes the CNN layers.
- **`forward(self, x)`**:
    Defines the forward pass through the network.

### `train`
- **`train(model, trainloader, valloader, criterion, optimizer, device, epochs=5)`**:
    Trains the model using the specified parameters and records training history.

### `create_dataloader`
- **`create_dataloader(root_dir, transform, batch_size=64, train=True, train_ratio=0.8)`**:
    Creates data loaders for the training and validation datasets.

### `plot_predictions`
- **`plot_predictions(model, dataloader, device)`**:
    Plots the model's predictions on a grid of images.

### `plot_training_history`
- **`plot_training_history(history)`**:
    Plots the training and validation loss and accuracy over epochs.

### `summarize_model`
- **`summarize_model(model)`**:
    Prints a summary of the model's layers and parameters.

## Conclusion

This project provides a comprehensive implementation of a CNN using PyTorch, complete with data handling, training, evaluation, and visualization capabilities. The modular structure allows for easy extension and modification, making it a solid foundation for further development in image classification tasks.