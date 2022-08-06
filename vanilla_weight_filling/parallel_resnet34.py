"""
Download a language model and load its weights layer by layer as matrices
1. Download albert model
2. Load the model
3. Load the weights
4. Load the weights layer by layer as matrices
5. Save the matrices

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, ImageNet, CIFAR10
import torchvision.models as models
import torch.utils.data.dataloader as dataloader
import numpy as np
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError
from cifar10_models.resnet import resnet18, resnet34, resnet50
from torch.utils.data.dataloader import default_collate
import math
import wandb
import gc
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_ as clipGN

batch_size = 4
max_size = 512
torch.set_default_dtype(torch.double)
img_batch_size = 10
#d0 = torch.device("cuda:0")
d1 = torch.device("cuda:0")
torch.backends.cudnn.benchmark = True

def get_np_fixed_length(list_like, length):
    list_length = len(list_like)
    np_array = np.zeros(length)
    l = min(length, len(list_like))
    np_array[:l] = list_like[:l]
    return np_array


# 1. Generate a series of masked matrices
def load_pretrained_weight_matrices(mask_ratio):
    # Option 1: passing weights param as string
    #pretrained_model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")
    # cache is in /root/.cache/torch/hub/pytorch_vision_main
    #pretrained_model = resnet50()
    pretrained_model = resnet34(pretrained=True)
    transform = transforms.Compose([transforms.ToTensor()])
    # initialzie the training and validation dataset
    print("[INFO] loading the training and validation dataset...")
    pretrained_weights = []

    for name, param in pretrained_model.named_parameters():
        #print(name, pretrained_model.state_dict()[param].shape)
        if 'weight' in name:

            cur = param.detach().numpy()
                # cur_fixed = get_np_fixed_length(cur, cur.shape[0])
            A, A2 = np.zeros([max_size, max_size, 9]), np.zeros([max_size, max_size, 9])
            cur_torch = torch.from_numpy(cur)

           # print(cur_torch.shape, cur.shape)
            try:
                s = cur.shape[2]
            except:
                continue
            if (cur.shape[0] > max_size) or (cur.shape[1] > max_size) : continue
            #print("start")
            #print(name, cur_torch.shape)
            cur_torch = torch.flatten(cur_torch, start_dim=2, end_dim=3)
            #print("after", cur_torch.shape)
            cur = cur_torch.cpu().detach().numpy()
            mask = np.random.rand(*cur.shape)
            bool_mask = mask < mask_ratio
            masked = torch.from_numpy(bool_mask * cur)
            shapes = [1 for i in range(4)]
            for i in range(4):
                try:
                    shapes[i] = cur.shape[i]
                except:
                    continue
            A[:shapes[0], :shapes[1], :shapes[2]] = cur
           # B = np.einsum('ijkl->lkji', A)
            A2[:shapes[0], :shapes[1], :shapes[2]] = masked
           # B2 = np.einsum('ijkl->lkji', A2)

            pretrained_weights.append((torch.from_numpy(A2).double(), torch.from_numpy(A).double()))
            # A2[:cur.shape[0],:cur.shape[1] ,:cur.shape[2] , :cur.shape[3]] = masked

          #  pretrained_weights.append((torch.from_numpy(A2), torch.from_numpy(A)))
       # elif 'bn' in name:
       #     print(name)

    print("finished loading weight matrices from pretrained neural network")
    return pretrained_weights


# 2. Write backbone of a RNN dataloader
class RNNDataloader(dataloader.DataLoader):
    """
    DataLoader for the RNN.
    """

    def __init__(self, data, batch_size, shuffle=True, num_workers=1, pin_memory=False):
        super(dataloader.DataLoader, self).__init__()
        self.data = data
        self.data = self.data
        self.batch_size =  batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.img_batch_size = 250
        self.pin_memory = pin_memory
        self.resize_sz = 32
        src_dataset = CIFAR10(root='./data', train=True, download=True, transform= transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616]),
                                               ]))
        trainDataLoader = DataLoader(src_dataset, batch_size=self.img_batch_size, shuffle=True)
        self.img_batch = next(iter(trainDataLoader))
        del src_dataset, trainDataLoader
        gc.collect()
        torch.cuda.empty_cache()

    def __iter__(self):
        """
        Iterate through the dataset.
        """
        # 1. Get the data
        data = self.data
        groups = [[data[i-1], data[i], data[i+1]]for i in range(1, len(data)-1)]
        groups.append([data[-2], data[-1], data[-2]])

        # 3. Split the data into batches
        batches = [groups[i:i + self.batch_size] for i in range(0, len(groups), self.batch_size)]
        # 4. Iterate through the batches
        for batch in batches:
            # 5. Get the inputs and targets
            masked_, targ_ = [], []

            for t in batch:
                seed = math.floor(np.random.rand()*(len(self.img_batch)-1))
                img = self.img_batch[seed]
               # print("dataloader img", img.shape)
                img = torch.flatten(img, start_dim=0, end_dim=1)
                img = torch.permute(img, (2,1,0))
                masked, target = t[1][0].clone().detach(), t[1][1].clone().detach()
                prev, after = t[0][1].clone().detach(), t[2][0].clone().detach()
                try:
                  masked, prev, after = self.format_(masked), self.format_(prev), self.format_(after)
                  target = self.format_(target)
               # print("dataloader",target.shape)

                  masked_.append([masked, prev, after, img])
                  targ_.append(target)
                except Exception as e:
                  print(e)

            yield masked_, targ_

    def format_(self, data_input):
        """
        Format the data.
        """
       # data = torch.squeeze(torch.cat(data.unbind()).unsqueeze(0))
        data_ = torch.Tensor(data_input)
        #print("data_",data_.shape)
        data_ = torch.permute(data_, (2, 0, 1))
        data_ = data_.double()

       # print(data.get_device())
        return data_

class RNN(nn.Module):
    """
    RNN from scratch for a sequence of data in pytorch
    """

    def __init__(self, input_size, hidden_size, output_size, mask_ratio):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.prev_layer = nn.Linear(input_size, hidden_size)
        self.after_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.unflatten_layer = nn.Unflatten(2, torch.Size([3,3]))
        self.attn = nn.MultiheadAttention(hidden_size, hidden_size)
        self.img_batch_size = 8
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.clip_val = 1
        self.resize_sz = 32
        self.output_layer = nn.Linear(self.resize_sz, 9)
        self.transition_layer = nn.Linear(hidden_size, self.resize_sz)
        self.mask_ratio = mask_ratio

        self.img_layer = nn.Linear(750,self.hidden_size)
        self.coordinating_layer = nn.Bilinear(self.resize_sz, 9,hidden_size)
        gc.collect()
        torch.cuda.empty_cache()


    def forward(self, input):
        """
        Forward pass of the RNN.
        """
     #   print(len(input))
      #  input = input[0]
        input__ = input[0].to(d1)
        input_ = self.input_layer(input__)
        prev = input[1].to(d1)
        prev_ = self.prev_layer(prev)
        after = input[2].to(d1)
        after_ = self.after_layer(after)
        img_ = self.img_layer(input[3].to(d1))
        img_ = torch.permute(img_, (2,1,0))

      #  print("img:", img_.shape)

        output = self.attn(prev_, after_, input_)[0]

        output = self.transition_layer(output)
        output = torch.permute(output, (1,2, 0))
       # print("attn_out:", output.shape)
        coord = self.coordinating_layer(img_, output)
        coord = torch.permute(coord, (0,2,1))
       # print("coord", coord.shape)
        output = self.output_layer(coord)

        output = torch.permute(output, (2,0,1))
       # print("output", output.shape)
        unflattened = torch.permute(output, (1,2,0))
        unflattened = self.unflatten_layer(unflattened)
       # print("unflatten", unflattened.shape)
        gc.collect()
        del input__, prev, after, input_, prev_, after_, img_, coord
        torch.cuda.empty_cache()
        return output, unflattened

    def init_hidden(self, batch_size):
        """
        Initialize the hidden state of the RNN.
        """
        return torch.zeros((batch_size, self.hidden_size, self.hidden_size))

    def predict(self, input, hidden):
        """
        Predict the next print("Epoch: ",epoch, "Training Loss: ", loss, "Validation Loss:", valid_loss)word given the input and the hidden state.
        """
        output, hidden = self.forward(input, hidden)
        return output


# Write training code with dataloader
# 3. Train the model
def train(model, dataloader, validation_dl, num_epochs, learning_rate, logging, pin_memory):
    """
    Train the model.
    """
    # 1. Define the loss and optimizer
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    use_amp = True

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.zero_grad(set_to_none=True)
    torch.autograd.set_detect_anomaly(True)
    scaler = GradScaler(enabled=use_amp)
    # 2. Iterate through the data for num_epochs
    for epoch in range(num_epochs):
        # 3. Iterate through the data for one epoch
        for inputs, targets in dataloader:
            # 4. Reset the gradients

            optimizer.zero_grad()
            # 5. Forward pass
            inputs_ = inputs[0]
            target_ = targets[0].cuda(non_blocking=pin_memory).to(d1)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):

              outputs, unflattened = model(inputs_)
            #  print("loss", outputs.shape, targets[0].shape)
              loss = criterion(outputs, target_)
            del inputs_
            torch.cuda.empty_cache()
            # 7. Compute the gradients
            scaler.scale(loss).backward()#retain_graph=True)
            # 8. Update the weights
            clipGN(model.parameters(), model.clip_val)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()

            gc.collect()
            del target_, outputs
            torch.cuda.empty_cache()
        mse = MeanSquaredError().to(d1)
        valid_loss = 0.0
        model.eval()
        for vinputs, vtargets in validation_dl:
            vtarget_ = vtargets[0].to(d1)
            vinputs_ = vinputs[0]
            voutputs, unflattened = model(vinputs_)
            vloss = mse(voutputs, vtarget_)
            valid_loss += vloss.item()
            gc.collect()
            del vtarget_, vinputs_
            torch.cuda.empty_cache()

        if logging:
            wandb.log({"loss": loss.item()}, step=epoch)
            wandb.log({"vloss": valid_loss},step=epoch)

        gc.collect()
        print("Epoch: ",epoch, "Training Loss: ", loss.item(), "Validation MSE:", valid_loss)
        #if epoch % 10 == 0:
        #  torch.save(model.state_dict(), "par_0.9_model" + str(epoch) + ".pt")
        del loss, mse
        torch.cuda.empty_cache()
    # 10. Save the model
    torch.save(model.state_dict(), "par_shaped.pt")
    # 11. Return the model
    return model


# 4. Save the model
def save_model(model):
    """
    Save the model.
    """
    torch.save(model.state_dict(), "model.pt")
    return model


# 5. Load the model
def load_model():
    """
    Load the model.
    """
    model = RNN(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load("model.pt"))
    return model

import random

logging, pin_memory = False, True
mask_ratio = 0.3
data = load_pretrained_weight_matrices(mask_ratio)
random.shuffle(data)
print("Data Size:", len(data))
test_size = int(len(data) * 0.92)
train_set, val_set = data[:test_size], data[test_size:]
model = RNN(max_size, max_size, max_size, mask_ratio)
model.to(d1)
if logging:
  wandb.init(project="Weight Masking", entity="zs0316")
  wandb.config = {
    "learning_rate": 0.001,
    "epochs": 100,
  "batch_size": 128
  }
  wandb.watch(model)
dataloader_ = RNNDataloader(train_set, batch_size=batch_size, pin_memory=pin_memory)
validation_dataloader_ = RNNDataloader(val_set, batch_size=batch_size, pin_memory=pin_memory)
tmodel = train(model, dataloader_, validation_dataloader_, num_epochs=500, learning_rate=0.001, logging = logging, pin_memory = pin_memory)
