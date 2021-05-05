import torch
import numpy as np


def tensor_examples():
    # Tensors
    # At its core, PyTorch is a library for processing tensors. A tensor is a number, vector, matrix, or any n-dimensional array. Let's create a tensor with a single number.
    t1 = torch.tensor(4.)
    print(t1)
    print(t1.dtype)

    # Vectors
    t2 = torch.tensor([1., 2, 3, 4])
    print(t2)

    # Matrix
    t3 = torch.tensor([[5., 6], [7, 8], [9 ,10]])
    print(t3)

    # 3D array
    t4 = torch.tensor([[[11., 12, 13], [13, 14, 15]], [[15, 16, 17], [17 ,18, 19]]])
    print(t4)

    print(t2.shape)
    print(t3.shape)
    print(t4.shape)

    # We can combine tensors with the usual arithmetic operations.
    # Create tensors.
    x = torch.tensor(3.)
    w = torch.tensor(4., requires_grad=True)
    b = torch.tensor(5., requires_grad=True)

    # Arithmetic operations
    y = w * x + b

    # Compute derivatives
    y.backward()

    # Display gradients
    print('dy/dx:', x.grad)
    print('dy/dw:', w.grad)
    print('dy/db:', b.grad)

    # Create a tensor with a fixed value for every element
    t6 = torch.full((3, 2), 42)
    print(t6)

    # Concatenate two tensors with compatible shapes
    t7 = torch.cat((t3, t6))
    print(t7)

    # Compute the sin of each element
    t8 = torch.sin(t7)
    print(t8)
    # Change the shape of a tensor
    t9 = t8.reshape(3, 2, 2)
    print(t9)
    print(t8.shape)
    print(t9.shape)
def numpy_to_tensor():
    x = np.array([[1, 2], [3, 4]])
    print(x.dtype)

    # Convert the numpy array to a torch tensor.
    y = torch.from_numpy(x)
    print(y.dtype)

    # Convert a torch tensor to a numpy array
    z = y.numpy()
    print(z.dtype)


if __name__ == "__main__":
    tensor_examples()
    numpy_to_tensor()
    