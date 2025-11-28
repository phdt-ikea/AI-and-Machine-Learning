"""
PyTorch Neural Network Training Example

This script demonstrates a complete PyTorch workflow for training a neural network
on the Fashion-MNIST dataset. It includes data loading, model definition, training,
and evaluation.

The Fashion-MNIST dataset consists of 28x28 grayscale images of 10 fashion categories:
- T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

Key Components
--------------
- **Data Loading**: Downloads and prepares Fashion-MNIST dataset with DataLoader
- **Model Architecture**: Simple feedforward neural network with 2 hidden layers
- **Training Loop**: Implements backpropagation and parameter optimization
- **Evaluation**: Tests model performance on held-out test set

Model Architecture
------------------
- Input: 28x28 grayscale images (784 features after flattening)
- Hidden Layer 1: 512 neurons with ReLU activation
- Hidden Layer 2: 512 neurons with ReLU activation
- Output Layer: 10 classes (logits)

Training Configuration
----------------------
- Optimizer: Stochastic Gradient Descent (SGD)
- Learning Rate: 0.001
- Loss Function: Cross-Entropy Loss
- Batch Size: 64
- Epochs: 5

Dependencies
------------
- torch: PyTorch deep learning framework
- torchvision: Computer vision datasets and transforms

Usage
-----
Run this script to train and evaluate the neural network:
    $ python pytorch_example.py

Notes
-----
- The script automatically detects and uses available accelerators (GPU/TPU)
- Training progress is printed every 100 batches
- Test accuracy and loss are reported after each epoch
"""

# Import libraries
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets  # type: ignore
from torchvision.transforms import ToTensor  # type: ignore

# Dataset
# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# Create data loaders.
BATCH_SIZE = 64
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)
for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Creating Models
device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    and torch.accelerator.current_accelerator() is not None
    else "cpu"
)
print(f"Using {device} device")


# Define model
class NeuralNetwork(nn.Module):
    """
    Feedforward neural network for Fashion-MNIST classification.
    
    This network consists of a flattening layer followed by three fully-connected
    layers with ReLU activations. The architecture is designed for 28x28 image
    classification into 10 categories.
    
    Architecture
    ------------
    - Flatten: Converts 28x28 image to 784-dimensional vector
    - Linear(784, 512) + ReLU: First hidden layer
    - Linear(512, 512) + ReLU: Second hidden layer
    - Linear(512, 10): Output layer (logits for 10 classes)
    
    Attributes
    ----------
    flatten : nn.Flatten
        Layer to flatten input images into 1D vectors.
    linear_relu_stack : nn.Sequential
        Sequential container of linear and ReLU layers.
    
    Methods
    -------
    forward(x)
        Performs forward pass through the network.
    
    Examples
    --------
    >>> model = NeuralNetwork()
    >>> x = torch.randn(64, 1, 28, 28)  # batch of 64 images
    >>> logits = model(x)
    >>> logits.shape
    torch.Size([64, 10])
    """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        """
        Forward pass through the neural network.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, 28, 28) containing
            grayscale images.
        
        Returns
        -------
        torch.Tensor
            Output logits of shape (batch_size, 10) for each class.
        """
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

# Optimizing the Model Parameters
# To train a model, we need a loss function and an optimizer.
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


# In a single training loop, the model makes predictions on the training dataset
# (fed to it in batches), and backpropagates the prediction error to adjust the
# model's parameters.
def train(dataloader, model, loss_fn, optimizer):
    """
    Train the model for one epoch.
    
    Performs a complete pass through the training dataset, computing predictions,
    calculating loss, and updating model parameters via backpropagation.
    
    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        DataLoader containing the training dataset batches.
    model : nn.Module
        The neural network model to train.
    loss_fn : nn.Module
        Loss function (criterion) to compute prediction error.
    optimizer : torch.optim.Optimizer
        Optimizer to update model parameters.
    
    Returns
    -------
    None
        Prints training loss every 100 batches.
    
    Notes
    -----
    - Sets model to training mode (enables dropout, batch norm, etc.)
    - Uses gradient descent with backpropagation
    - Prints progress every 100 batches showing loss and samples processed
    
    Examples
    --------
    >>> train(train_dataloader, model, nn.CrossEntropyLoss(), optimizer)
    loss: 2.304321  [   64/60000]
    loss: 2.291234  [ 6464/60000]
    ...
    """
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# We also check the model's performance against the test dataset to ensure it is learning.
def test(dataloader, model, loss_fn):
    """
    Evaluate the model on the test dataset.
    
    Computes test accuracy and average loss without updating model parameters.
    This function provides insight into how well the model generalizes to unseen data.
    
    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        DataLoader containing the test dataset batches.
    model : nn.Module
        The trained neural network model to evaluate.
    loss_fn : nn.Module
        Loss function (criterion) to compute prediction error.
    
    Returns
    -------
    None
        Prints test accuracy and average loss to console.
    
    Notes
    -----
    - Sets model to evaluation mode (disables dropout, batch norm updates, etc.)
    - Uses torch.no_grad() context to disable gradient computation for efficiency
    - Computes both classification accuracy and average loss
    - Does not modify model parameters
    
    Examples
    --------
    >>> test(test_dataloader, model, nn.CrossEntropyLoss())
    Test Error: 
     Accuracy: 85.3%, Avg loss: 0.421543
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


# The training process is conducted over several iterations (epochs). During each epoch,
# the model learns parameters to make better predictions. We print the model’s accuracy
# and loss at each epoch; we’d like to see the accuracy increase and the loss decrease
# with every epoch.
epochs = 5
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
