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
import torch.nn as nn
from torchvision import transforms
from resnets_pre import RNN
import math
from torch.cuda.amp import GradScaler
import gc

biases = {}


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
        torch.autograd.set_detect_anomaly(True)
        out = model(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        # print(acc, loss)
        outputs.append({'val_loss': loss.detach(), 'val_acc': acc})
    # outputs = [model.validation_step(batch) for batch in val_loader]
    res = validation_epoch_end(outputs)
    epoch_end(res)


def mask_weights(mask_ratio, param, max_size=512):
    cur = param  # .numpy()
    # cur_fixed = get_np_fixed_length(cur, cur.shape[0])
    A, A2 = np.zeros([max_size, max_size, 9]), np.zeros([max_size, max_size, 9])
    cur_torch = torch.from_numpy(cur)
    # print("before",cur_torch.shape)
    cur_torch = torch.flatten(cur_torch, start_dim=2, end_dim=3)
    # print("after", cur_torch.shape)
    cur = cur_torch.cpu().detach().numpy()
    mask = np.random.rand(*cur.shape)
    bool_mask = mask < mask_ratio
    masked = torch.from_numpy(bool_mask * cur)
    A[:cur_torch.shape[0], :cur_torch.shape[1], :cur_torch.shape[2]] = cur
    A2[:cur_torch.shape[0], :cur_torch.shape[1], :cur_torch.shape[2]] = masked
    masked, filled = torch.from_numpy(A2).double(), torch.from_numpy(A).double()
    masked, filled = torch.permute(masked, (2, 0, 1)), torch.permute(filled, (2, 0, 1))
    return (masked, filled)


def pad_prev(param):
    A = np.zeros([max_size, max_size, 9])
    cur_torch = param
    # print("before",cur_torch.shape)
    cur_torch = torch.flatten(cur_torch, start_dim=2, end_dim=3)
    cur_numpy = cur_torch.cpu().detach().numpy()
    A[:cur_torch.shape[0], :cur_torch.shape[1], :cur_torch.shape[2]] = cur_numpy

    filled_prev = torch.from_numpy(A).double()
    filled_prev = torch.permute(filled_prev, (2, 0, 1))
    return filled_prev


def adding_biases(name, entry):
    A = np.zeros([max_size])
    name_ = name
    entry = entry.detach().numpy()
    A[:entry.shape[0]] = entry
    res = torch.from_numpy(A).double()
    layer_name = name.split(".bn")[0]
    # print("in biases", layer_name, name, 'weight' in name)
    if layer_name not in biases:
        biases[layer_name] = [None, None]
    if 'weight' in name:
        biases[layer_name][0] = res
    else:
        biases[layer_name][1] = res


def layer_name_parser(name):
    tkns = name.split(".")
    try:
        layer_index = tkns[0]
        subindex = int(tkns[1])
        layer_type = tkns[2]
        last = tkns[3]
        last = int(last) if last.isnumeric() else 99

        layer_name = name.split(".conv")[0] if "conv" in name else name.split(".downsample")[0]
        return layer_index, subindex, layer_type, last, layer_name
    except:
        return "last", "last", "last", "last", "last"


d = torch.device("cuda:0")
net = resnet34()
mask_ratio = 0.3  # wpredictor.mask_ratio
max_size = 512
wpredictor = RNN(max_size, max_size, max_size, mask_ratio)
wpredictor.load_state_dict(torch.load("few_with_last_.pt"))
wpredictor.to(d)
resize_sz = 32
target_net, target_layer_names = [], []
img_batch_size = 3

dataset = CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616]),
     ]))
train_ds, val_ds = random_split(dataset, [49900, 100])
vloader = DataLoader(val_ds, batch_size=img_batch_size, shuffle=True, num_workers=0)
tloader = DataLoader(train_ds, batch_size=img_batch_size, shuffle=True)
img_batch = next(iter(vloader))
del dataset, train_ds, val_ds
gc.collect()
torch.cuda.empty_cache()

for name, param in net.named_parameters():
    print(name, param.shape)
    if 'bn' in name:
        # print(1, name)
        adding_biases(name, param)
    elif 'weight' in name and 'downsample' not in name:
        if len(param.shape) > 3:
            # print(name, param.shape)
            param = np.random.rand(*param.shape)
            target_net.append(param)
            target_layer_names.append(name)
    elif 'fc' in name:
        if 'weight' in name:
            A = np.zeros([max_size, max_size, 9])
            A[:param.shape[0], :param.shape[1], 0] = param
            res = torch.from_numpy(A).double()
            target_net.append(res)
            target_layer_names.append(name)
        else:
            A = np.zeros([max_size])
            A[:param.shape[0]] = param.detach().numpy()
            res = torch.from_numpy(A).double()
            biases["last"] = [res, res]

print("#Layers", len(target_layer_names))
prev_layers = [torch.from_numpy(target_net[0])]

for i in range(1, len(target_net)):
    try:
        if i == len(target_net) - 1:
            prev_, current, after = prev_layers[-1], target_net[i], target_net[i]
            after_ = [pad_prev(prev_)]

        else:
            prev_, current, after = prev_layers[-1], target_net[i], target_net[i + 1]
            after_ = mask_weights(mask_ratio, after)
        name = target_layer_names[i]
        prev_ = pad_prev(prev_)
        current_ = mask_weights(mask_ratio, current)
        seed = math.floor(np.random.rand() * (len(img_batch) - 1))
        layer_index, subindex, layer_type, downsample_tkn, layer_name = layer_name_parser(name)
        img = img_batch[seed].to(d)
        img = torch.flatten(img, start_dim=0, end_dim=1)
        print(name, layer_name)
        bn_ = biases[layer_name]

        bn = torch.vstack((bn_[0], bn_[1]))
        bn = bn.repeat(512, 1, 1)
        bn = torch.permute(bn, (0, 2, 1))
        bn = bn.to(d)
        shape_ = current.shape
        masked_current, masked_prev, masked_after = current_[0].to(d), prev_.to(d), after_[0].to(d)
        filled_target, filled_shaped = wpredictor(masked_current, masked_prev, masked_after, img, bn)
        cropped_fill = filled_shaped[:shape_[0], :shape_[1], :shape_[2], :shape_[3]]
        del img, masked_current, masked_prev, masked_after, shape_, bn, prev_, current_, after_, seed, filled_shaped, filled_target, bn_
        gc.collect()
        torch.cuda.empty_cache()

        cropped_fill = cropped_fill.type(torch.DoubleTensor)
        cropped_fill = cropped_fill.cpu()

        if 'downsample' in layer_type:
            net._modules[layer_index][subindex]._modules[layer_type][downsample_tkn].weight.data = cropped_fill
        elif layer_type == "last":
            net._modules["fc"].weight.data = cropped_fill
        else:
            net._modules[layer_index][subindex]._modules[layer_type].weight.data = cropped_fill
        prev_layers.append(cropped_fill)
        del cropped_fill
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(name, e)
print("Weight Filling Succeeded")
net.eval()
evaluate(net, vloader)
import torch.optim as optim

use_amp = True
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
scaler = GradScaler(enabled=use_amp)
for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(tloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        optimizer.zero_grad()

        # forward + backward + optimize
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            outputs = net(inputs)
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:  # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')
net.eval()
evaluate(net, vloader)