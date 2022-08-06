from cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from cifar10_models.densenet import densenet121, densenet161, densenet169
from cifar10_models.resnet import resnet18, resnet34, resnet50

import torch
from torchvision.models import resnet34 as tresnet34
from torchvision.datasets import CIFAR10
import numpy as np
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
import math


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


def mask_weights(mask_ratio, param, max_size=512):
    cur = param  # .numpy()
    # cur_fixed = get_np_fixed_length(cur, cur.shape[0])
    A, A2 = np.zeros([max_size, max_size, 9]), np.zeros([max_size, max_size, 9])
    cur_torch = torch.from_numpy(cur)
    print("before", cur_torch.shape)
    cur_torch = torch.flatten(cur_torch, start_dim=2, end_dim=3)
    print("after", cur_torch.shape)
    cur = cur_torch.cpu().detach().numpy()
    mask = np.random.rand(*cur.shape)
    bool_mask = mask < mask_ratio
    masked = torch.from_numpy(bool_mask * cur)
    A[:cur_torch.shape[0], :cur_torch.shape[1], :cur_torch.shape[2]] = cur
    A2[:cur_torch.shape[0], :cur_torch.shape[1], :cur_torch.shape[2]] = masked
    return (torch.from_numpy(A2).double(), torch.from_numpy(A).double())


def layer_name_parser(name):
    tkns = name.split(".")
    layer_index = tkns[0]
    subindex = int(tkns[1])
    layer_type = tkns[2]
    last = tkns[3]
    last = int(last) if last.isnumeric() else 99
    return layer_index, subindex, layer_type, last


net = resnet34()
# wpredictor = torch.load("par_shaped.pt")
resize_sz = 32
mask_ratio = 0.3  # wpredictor.mask_ratio
target_net, target_layer_names = [], []

dataset = CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616]),
     ]))
train_ds, val_ds = random_split(dataset, [45000, 5000])
vloader = DataLoader(val_ds, batch_size=300, shuffle=True, num_workers=0)
img_batch = next(iter(vloader))
a = net._modules['layer1'][0]._modules['conv1'].weight.data

net._modules['layer1'][0]._modules['conv1'].weight.data = torch.zeros(a.shape)
b = net._modules['layer1'][0]._modules['conv1'].weight.data

for name, param in net.named_parameters():
    if 'weight' in name and 'bn' not in name:
        if len(param.shape) > 3:
            print(name, param.shape)
            param = np.random.rand(*param.shape)
            target_net.append(param)
            target_layer_names.append(name)

print(target_layer_names)
for i in range(2, len(target_net) - 1):
    if i == len(target_net) - 1:
        prev, current, after = target_net[i - 1], target_net[i], target_net[i - 1]
    else:
        prev, current, after = target_net[i - 1:i + 2]
    prev_ = mask_weights(mask_ratio, prev)
    after_ = mask_weights(mask_ratio, after)
    current_ = mask_weights(mask_ratio, current)
    seed = math.floor(np.random.rand() * (len(img_batch) - 1))
    img = img_batch[seed]
    img = torch.flatten(img, start_dim=0, end_dim=1)
    img = torch.permute(img, (2, 1, 0))
    net_input = [current_[0], prev_[0], after_[0], img]
    # filled_target, filled_shaped = wpredictor(net_input)
    name = target_layer_names[i]

    layer_index, subindex, layer_type, downsample_tkn = layer_name_parser(name)
    print(name, layer_index, subindex, layer_type)
    if 'downsample' in layer_type:
        net._modules[layer_index][subindex]._modules[layer_type][downsample_tkn].weight.data = torch.from_numpy(
            np.random.rand(*current.shape))
    else:
        net._modules[layer_index][subindex]._modules[layer_type].weight.data = torch.from_numpy(
            np.random.rand(*current.shape))
    # net[name] = np.random.rand(*current.shape) #filled_shaped

net.eval()
evaluate(net, vloader)