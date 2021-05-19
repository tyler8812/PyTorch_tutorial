# Imports
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def predict_image(img, model):
    xb = img.unsqueeze(0)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item()


if __name__ == "__main__":
    # Define test dataset
    test_dataset = MNIST(root='data/',
                         train=False,
                         transform=transforms.ToTensor())

    test_loader = DataLoader(test_dataset, batch_size=256)

    # batch_size = 128
    input_size = 28*28
    num_classes = 10

    class MnistModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(input_size, num_classes)

        def forward(self, xb):
            xb = xb.reshape(-1, 784)
            out = self.linear(xb)
            return out

        def training_step(self, batch):
            images, labels = batch
            out = self(images)                  # Generate predictions
            loss = F.cross_entropy(out, labels)  # Calculate loss
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
            # Combine accuracies
            epoch_acc = torch.stack(batch_accs).mean()
            return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

        def epoch_end(self, epoch, result):
            print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result['val_loss'], result['val_acc']))
    class FMnistModel(nn.Module):
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
    model1 = MnistModel()
    model1.load_state_dict(torch.load('mnist-logistic.pth'))
    result1 = evaluate(model1, test_loader)
    print("mnist-logistic: ", result1)
    model2 = FMnistModel(input_size, 32, num_classes)
    model2.load_state_dict(torch.load('mnist-feedforward.pth'))
    result2 = evaluate(model2, test_loader)
    print("mnist-feedforward: ", result2)
