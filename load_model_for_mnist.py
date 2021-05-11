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

    model2 = MnistModel()
    model2.load_state_dict(torch.load('mnist-logistic.pth'))
    result = evaluate(model2, test_loader)
    print(result)
