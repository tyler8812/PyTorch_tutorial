# Imports
import torch
import torchvision
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F


def preprocessing_dataset(dataset, batch_size):
    # split validation
    train_ds, val_ds = random_split(dataset, [50000, 10000])
    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size)
    return train_loader, val_loader


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    optimizer = opt_func(model.parameters(), lr)
    history = []  # for recording epoch-wise results

    for epoch in range(epochs):

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

    return history



if __name__ == "__main__":
    dataset = MNIST(root='data/', download=True)
    # print(len(dataset))

    # test_dataset = MNIST(root='data/', train=False, transform=transforms.ToTensor())
    # print(len(test_dataset))

    # show the image and the label
    # image, label = dataset[0]
    # plt.imshow(image, cmap='gray')
    # print('Label:', label)
    # plt.show()
    # image, label = dataset[10]
    # plt.imshow(image, cmap='gray')
    # print('Label:', label)
    # plt.show()

    # MNIST dataset (images and labels)
    # image is now converted to a 1x28x28 tensor.
    # The first dimension tracks color channels. The second and third dimensions represent pixels along the height and width of the image
    dataset = MNIST(root='data/',
                    train=True,
                    transform=transforms.ToTensor())

    # img_tensor, label = dataset[0]
    # print(img_tensor[0, 10:15, 10:15])
    # The values range from 0 to 1, with 0 representing black, 1 white, and the values in between different shades of grey.
    # print(torch.max(img_tensor), torch.min(img_tensor))
    # plt.imshow(img_tensor[0,10:15,10:15], cmap='gray')
    # plt.show()
    batch_size = 128
    input_size = 28*28
    num_classes = 10
    t_l, v_l = preprocessing_dataset(dataset, batch_size)

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
        
    model = MnistModel()
    # print(len(t_l))
    # # print(model.linear)
    # # print(model.linear.weight.shape, model.linear.bias.shape)
    # # print(list(model.parameters()))
    # for images, labels in t_l:
    #     print(images.shape)
    #     outputs = model(images)
    #     break

    # # print('outputs.shape : ', outputs.shape)
    # # print('Sample outputs :\n', outputs[:2].data)
    # # Apply softmax for each output row
    # probs = F.softmax(outputs, dim=1)
    # # # Look at sample probabilities
    # # print("Sample probabilities:\n", probs[:2].data)

    # print(accuracy(outputs, labels))
    # print(accuracy(probs, labels))
    # loss_fn = F.cross_entropy
    # # Loss for current batch of data
    # loss = loss_fn(outputs, labels)
    # print(loss)
    result0 = evaluate(model, v_l)
    history1 = fit(5, 0.001, model, t_l, v_l)
    history2 = fit(5, 0.001, model, t_l, v_l)
    history3 = fit(5, 0.001, model, t_l, v_l)
    history4 = fit(5, 0.001, model, t_l, v_l)
    history = [result0] + history1 + history2 + history3 + history4
    accuracies = [result['val_acc'] for result in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.show()

    torch.save(model.state_dict(), 'mnist-logistic.pth')
    