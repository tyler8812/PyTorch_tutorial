import torch
import torchvision
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import time


def preprocessing_dataset(dataset, batch_size):
    # split validation
    train_ds, val_ds = random_split(dataset, [50000, 10000])
    # 
    train_loader = DataLoader(train_ds, batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)
    print("train_batch: ", len(train_loader))
    print("val_batch: ", len(val_loader))
    return train_loader, val_loader

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    # 是否為list或tuple
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def evaluate(model, val_loader):
    """Evaluate the model's performance on the validation set"""
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    """Train the model using gradient descent"""
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        t0 = time.time()
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
        print('{} seconds'.format(time.time() - t0))
    return history


if __name__ == "__main__":
    # Use a white background for matplotlib figures
    matplotlib.rcParams['figure.facecolor'] = '#ffffff'

    dataset = MNIST(root='data/', download=True, transform=ToTensor())

    # # show image
    # image, label = dataset[0]
    # print('image.shape:', image.shape)
    # plt.imshow(image.squeeze(), cmap='gray')
    # print('image.shape:', image.squeeze().shape)
    # print('Label:', label)
    # plt.show()
    
    batch_size = 128
    t_l, v_l = preprocessing_dataset(dataset, batch_size)
    # for images, _ in t_l:
    #     print('images.shape:', images.shape)
    #     plt.figure(figsize=(16,8))
    #     plt.axis('off')
    #     plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
    #     print(make_grid(images, nrow=16).permute((1, 2, 0))
    #     plt.show()
    #     break
    class MnistModel(nn.Module):
        """Feedfoward neural network with 1 hidden layer"""
        def __init__(self, in_size, hidden_size, out_size):
            super().__init__()
            # hidden layer
            self.linear1 = nn.Linear(in_size, hidden_size)
            # output layer
            self.linear2 = nn.Linear(hidden_size, out_size)
            
        def forward(self, xb):
            # Flatten the image tensors
            xb = xb.view(xb.size(0), -1)
            # Get intermediate outputs using hidden layer
            out = self.linear1(xb)
            # Apply activation function
            out = F.relu(out)
            # Get predictions using output layer
            out = self.linear2(out)
            return out
        
        def training_step(self, batch):
            images, labels = batch 
            out = self(images)                  # Generate predictions
            loss = F.cross_entropy(out, labels) # Calculate loss
            return loss
        
        def validation_step(self, batch):
            images, labels = batch 
            out = self(images)                    # Generate predictions
            loss = F.cross_entropy(out, labels)   # Calculate loss
            acc = accuracy(out, labels)           # Calculate accuracy
            return {'val_loss': loss, 'val_acc': acc}
            
        def validation_epoch_end(self, outputs):
            batch_losses = [x['val_loss'] for x in outputs]
            epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
            batch_accs = [x['val_acc'] for x in outputs]
            epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
            return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
        
        def epoch_end(self, epoch, result):
            print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

    input_size = 784
    hidden_size = 32 # you can change this
    num_classes = 10

    # model = MnistModel(input_size, hidden_size=32, out_size=num_classes)

    # for t in model.parameters():
    #     print(t.shape)
    
    # for images, labels in t_l:
    #     outputs = model(images)
    #     loss = F.cross_entropy(outputs, labels)
    #     print('Loss:', loss.item())
    #     break

    # print('outputs.shape : ', outputs.shape)
    # print('Sample outputs :\n', outputs[:2].data)
    # Using GPU
    device = get_default_device()
    print(device)
    t_l = DeviceDataLoader(t_l, device)
    v_l = DeviceDataLoader(v_l, device)
    # for xb, yb in v_l:
    #     print('xb.device:', xb.device)
    #     print('yb:', yb)
    #     break
    model = MnistModel(input_size, hidden_size=hidden_size, out_size=num_classes)
    to_device(model, device)
    history = [evaluate(model, v_l)]
    history += fit(5, 0.5, model, t_l, v_l)
    history += fit(5, 0.1, model, t_l, v_l)
    losses = [x['val_loss'] for x in history]
    plt.plot(losses, '-x')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs. No. of epochs')
    plt.show()
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.show()

    torch.save(model.state_dict(), 'mnist-feedforward.pth')