from cifar10_models.resnet import resnet18, resnet34, resnet50

import torch
from torchvision.models import resnet34 as tresnet34
from torchvision.datasets import CIFAR10
import numpy as np
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def validation_step(self, batch):
    images, labels = batch
    out = self(images)  # Generate predictions
    loss = F.cross_entropy(out, labels)  # Calculate loss
    acc = accuracy(out, labels)  # Calculate accuracy
    return {'val_loss': loss.detach(), 'val_acc': acc}


def validation_epoch_end(outputs):
    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}


def epoch_end(result):
    print("val_loss: {:.4f}, val_acc: {:.4f}".format(result['val_loss'], result['val_acc']))


def evaluate(model, val_loader):
    outputs = []
    for vbatch in val_loader:
        images, labels = vbatch
        out = model(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        print(acc, loss)
        outputs.append({'val_loss': loss.detach(), 'val_acc': acc})
    # outputs = [model.validation_step(batch) for batch in val_loader]
    res = validation_epoch_end(outputs)
    epoch_end(res)


net = tresnet34()
resize_sz = 512
for name, param in net.named_parameters():
    print(name, param.shape)
    param = np.random.rand(*param.shape)

dataset = CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616]),
     ]))
train_ds, val_ds = random_split(dataset, [45000, 5000])
vloader = DataLoader(val_ds, batch_size=10, shuffle=True, num_workers=0)
net.eval()
evaluate(net, vloader)