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
import torchvision.datasets as datasets
import torchvision.models as models
import torch.utils.data.dataloader as dataloader
import numpy as np
from torchmetrics import MeanSquaredError

import torch

batch_size = 2
max_size = 512
torch.set_default_dtype(torch.double)

def get_np_fixed_length(list_like, length):
    list_length = len(list_like)
    np_array = np.zeros(length)
    l = min(length, len(list_like))
    np_array[:l] = list_like[:l]
    return np_array


# 1. Generate a series of masked matrices
def load_pretrained_weight_matrices(num_matrices):
    # Option 1: passing weights param as string
    pretrained_model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")
    # cache is in /root/.cache/torch/hub/pytorch_vision_main

    pretrained_weights = []

    for name, param in pretrained_model.named_parameters():
        #print(name, pretrained_model.state_dict()[param].shape)
        if 'weight' in name:

            cur = param.detach().numpy()
                # cur_fixed = get_np_fixed_length(cur, cur.shape[0])
            A, A2 = np.zeros([max_size, max_size, 3, 3]), np.zeros([max_size, max_size, 3, 3])
            cur_torch = torch.from_numpy(cur)

           # print(cur_torch.shape, cur.shape)
            try:
                s = cur.shape[2]
            except:
                continue
            if (cur.shape[0] > max_size) or (cur.shape[1] > max_size) or (len(cur.shape) > 1 and cur.shape[3] > 3) : continue
            print("start")
            print(name)

            mask = np.random.rand(*cur.shape)
            bool_mask = mask < 0.75
            masked = torch.from_numpy(bool_mask * cur)
            shapes = [1 for i in range(4)]
            for i in range(4):
                try:
                    shapes[i] = cur.shape[i]
                except:
                    continue
            A[:shapes[0], :shapes[1], :shapes[2], :shapes[3]] = cur
            B = np.einsum('ijkl->lkji', A)
            A2[:shapes[0], :shapes[1], :shapes[2], :shapes[3]] = masked
            B2 = np.einsum('ijkl->lkji', A2)

            pretrained_weights.append((torch.from_numpy(B2).double(), torch.from_numpy(B).double()))
            # A2[:cur.shape[0],:cur.shape[1] ,:cur.shape[2] , :cur.shape[3]] = masked

          #  pretrained_weights.append((torch.from_numpy(A2), torch.from_numpy(A)))
        elif 'bn' in name:
            print(name)

    return pretrained_weights


# 2. Write backbone of a RNN dataloader
class RNNDataloader(dataloader.DataLoader):
    """
    DataLoader for the RNN.
    """

    def __init__(self, data, batch_size, shuffle=True, num_workers=1):
        super(dataloader.DataLoader, self).__init__()
        self.data = data
        self.batch_size = 1  # batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def __iter__(self):
        """
        Iterate through the dataset.
        """
        # 1. Get the data
        data = self.data
        groups = [[data[i-1], data[i], data[i+1]]for i in range(1, len(data)-1)]

        # 3. Split the data into batches
        batches = [groups[i:i + self.batch_size] for i in range(0, len(groups), self.batch_size)]
        # 4. Iterate through the batches
        for batch in batches:
            # 5. Get the inputs and targets
            masked_, targ_ = [], []

            for t in batch:

                masked, target = t[1][0][0].clone().detach(), t[1][1][0].clone().detach()
                prev, after = t[0][0][1].clone().detach(), t[2][0][1].clone().detach()
                masked, prev, after = self.format_(masked), self.format_(prev), self.format_(after)
                target = self.format_(target)

                masked_.append([masked, prev, after])
                targ_.append(target)

            yield masked_, targ_

    def format_(self, data):
        """
        Format the data.
        """
        data = torch.squeeze(torch.cat(data.unbind()).unsqueeze(0))
        return torch.Tensor(data).to(torch.double)

class RNN(nn.Module):
    """
    RNN from scratch for a sequence of data in pytorch
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.prev_layer = nn.Linear(input_size, hidden_size)
        self.after_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.attn = nn.MultiheadAttention(hidden_size, hidden_size)

    def forward(self, input, hidden):
        """
        Forward pass of the RNN.
        """
     #   print(len(input))
        input = input[0]
        input__ = input[0].to(torch.double).clone()
        input_ = self.input_layer(input__)
        prev = input[1].to(torch.double).clone()
        prev_ = self.prev_layer(prev)
        after = input[2].to(torch.double).clone()
        after_ = self.after_layer(after)

        h0 = self.hidden_layer(hidden.clone())
        #   hidden = F.relu(hidden.clone(), inplace=False)
       # print(input_.shape, h0.shape)
        output = self.attn(input_, prev_, after_)

        output = self.output_layer(output[0].clone())
        return output, hidden

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
def train(model, dataloader, validation_dl, num_epochs, learning_rate):
    """
    Train the model.
    """
    # 1. Define the loss and optimizer
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    hidden = model.init_hidden(9)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    torch.autograd.set_detect_anomaly(True)
    # 2. Iterate through the data for num_epochs
    for epoch in range(num_epochs):
        # 3. Iterate through the data for one epoch
        for inputs, targets in dataloader:
            # 4. Reset the gradients

            optimizer.zero_grad()
            # 5. Forward pass
            outputs, hidden = model(inputs, hidden.clone())
            # 6. Compute the loss
          #  print("loss", targets[0].shape)
            loss = criterion(outputs.clone(), targets[0].clone())
            # 7. Compute the gradients
            loss.backward(retain_graph=True)
            # 8. Update the weights
            optimizer.step()
        mse = MeanSquaredError()
        valid_loss = 0.0
        model.eval()
        for vinputs, vtargets in validation_dl:

            voutputs, hidden = model(vinputs, hidden.clone())
            vloss = mse(voutputs.clone(), vtargets[0].clone())
            valid_loss += vloss.item()

        print("Epoch: ",epoch, "Training Loss: ", loss.item(), "Validation Loss:", valid_loss)
    # 10. Save the model
    torch.save(model.state_dict(), "model.pt")
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


data = load_pretrained_weight_matrices(100)
test_size = int(len(data) * 0.91)
train_set, val_set = data[:test_size], data[test_size:-1]
model = RNN(max_size, max_size, max_size)
dataloader_ = RNNDataloader(train_set, batch_size=batch_size)
validation_dataloader_ = RNNDataloader(val_set, batch_size=batch_size)
model = train(model, dataloader_, validation_dataloader_, num_epochs=100, learning_rate=0.001)

