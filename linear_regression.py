import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F


def model(x, w, b):
    # @ represents matrix multiplication in PyTorch, and the .t method returns the transpose of a tensor.
    return x @ w.t() + b
# MSE loss


def mse(t1, t2):
    # torch.sum returns the sum of all the elements in a tensor. The .numel method of a tensor returns the number of elements in a tensor.
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()


def linear_regression_by_handmade():
    # Input (temp, rainfall, humidity)
    inputs = np.array([[73, 67, 43],
                       [91, 88, 64],
                       [87, 134, 58],
                       [102, 43, 37],
                       [69, 96, 70]], dtype='float32')

    # Targets (apples, oranges)
    targets = np.array([[56, 70],
                        [81, 101],
                        [119, 133],
                        [22, 37],
                        [103, 119]], dtype='float32')

    # Convert inputs and targets to tensors
    inputs = torch.from_numpy(inputs)
    targets = torch.from_numpy(targets)
    print(inputs)
    print(targets)

    # Weights and biases
    w = torch.randn(2, 3, requires_grad=True)
    b = torch.randn(2, requires_grad=True)
    print(w)
    print(b)

    # If a gradient element is positive:
    # increasing the weight element's value slightly will increase the loss
    # decreasing the weight element's value slightly will decrease the loss

    # If a gradient element is negative:
    # increasing the weight element's value slightly will decrease the loss
    # decreasing the weight element's value slightly will increase the loss
    # 1e-5 = learning rate
    # torch.no_grad to indicate to PyTorch that we shouldn't track, calculate, or modify gradients while updating the weights and biases.

    iteration = 200
    learning_rate = 1e-5
    # Train for 100 epochs
    for i in range(iteration):
        preds = model(inputs, w, b)
        loss = mse(preds, targets)
        loss.backward()
        with torch.no_grad():
            w -= w.grad * learning_rate
            b -= b.grad * learning_rate
            w.grad.zero_()
            b.grad.zero_()

    # Calculate loss
    preds = model(inputs, w, b)
    loss = mse(preds, targets)
    print("loss", loss)

    # Predictions
    print("preds", preds)
    print("targets", targets)


# Utility function to train the model
def fit(num_epochs, model, loss_fn, opt, train_dl):
    # Repeat for given number of epochs
    for epoch in range(num_epochs):

        # Train with batches of data
        for xb, yb in train_dl:

            # 1. Generate predictions
            pred = model(xb)

            # 2. Calculate loss
            loss = loss_fn(pred, yb)

            # 3. Compute gradients
            loss.backward()

            # 4. Update parameters using gradients
            opt.step()

            # 5. Reset the gradients to zero
            opt.zero_grad()

        # Print the progress
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch +
                  1, num_epochs, loss.item()))


def linear_regression_by_PyTorch():
    # Input (temp, rainfall, humidity)
    inputs = np.array([[73, 67, 43],
                       [91, 88, 64],
                       [87, 134, 58],
                       [102, 43, 37],
                       [69, 96, 70],
                       [74, 66, 43],
                       [91, 87, 65],
                       [88, 134, 59],
                       [101, 44, 37],
                       [68, 96, 71],
                       [73, 66, 44],
                       [92, 87, 64],
                       [87, 135, 57],
                       [103, 43, 36],
                       [68, 97, 70]],
                      dtype='float32')
    # Targets (apples, oranges)
    targets = np.array([[56, 70],
                        [81, 101],
                        [119, 133],
                        [22, 37],
                        [103, 119],
                        [57, 69],
                        [80, 102],
                        [118, 132],
                        [21, 38],
                        [104, 118],
                        [57, 69],
                        [82, 100],
                        [118, 134],
                        [20, 38],
                        [102, 120]],
                       dtype='float32')

    inputs = torch.from_numpy(inputs)
    targets = torch.from_numpy(targets)
    # TensorDataset, which allows access to rows from inputs and targets as tuples, and provides standard APIs for working with many different types of datasets in PyTorch.
    # Define dataset
    train_ds = TensorDataset(inputs, targets)
    print(train_ds[0:4])
    print(train_ds[[1, 3, 5, 7]])

    # DataLoader, which can split the data into batches of a predefined size while training. It also provides other utilities like shuffling and random sampling of the data.
    # Define data loader
    # 15(input) / 5(batch size) = 3
    batch_size = 5
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    # 3 times
    for xb, yb in train_dl:
        print(xb)
        print(yb)

    # Define model
    # (3 ,2) = (input size, output size)
    model = nn.Linear(3, 2)

    # Parameters
    print(list(model.parameters()))

    # Define loss function
    loss_fn = F.mse_loss

    # Define optimizer
    opt = torch.optim.SGD(model.parameters(), lr=1e-5)

    batch_size = 100

    fit(batch_size, model, loss_fn, opt, train_dl)
    print(model(inputs))
    print(loss_fn(model(inputs), targets))
    print(model(torch.tensor([[75, 63, 44.]])))


if __name__ == "__main__":
    # linear_regression_by_handmade()
    linear_regression_by_PyTorch()
