import torch
import numpy as np

def model(x, w, b):
    # @ represents matrix multiplication in PyTorch, and the .t method returns the transpose of a tensor.
    return x @ w.t() + b
# MSE loss
def mse(t1, t2):
    # torch.sum returns the sum of all the elements in a tensor. The .numel method of a tensor returns the number of elements in a tensor.
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()

if __name__ == "__main__":
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
        preds = model(inputs, w ,b)
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

