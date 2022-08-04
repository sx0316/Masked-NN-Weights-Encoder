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

batch_size = 1
max_size = 256
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
            bool_mask = mask < 0.5
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
        # 2. Shuffle the data if necessary
        if self.shuffle:
            np.random.shuffle(data)
        # 3. Split the data into batches
        batches = [data[i:i + self.batch_size] for i in range(0, len(data), self.batch_size)]
        # 4. Iterate through the batches
        for batch in batches:
            # 5. Get the inputs and targets
            masked_, targ_ = [], []
            start = False
            for t in batch:
                masked, target = t[0].clone().detach(), t[1].clone().detach()
                masked = torch.squeeze(torch.cat(masked.unbind()).unsqueeze(0))
                target = torch.squeeze(torch.cat(target.unbind()).unsqueeze(0))

                if not start:
                    masked_ = torch.Tensor(masked.to(torch.double))
                    targ_ = torch.Tensor(target.to(torch.double))
                    start = True
                else:
                    masked_.append(torch.Tensor(masked).to(torch.double))
                    targ_.append(torch.Tensor(target).to(torch.double))
            #        targ_ = torch.vstack((targ_, torch.Tensor(target)))
            yield masked_, targ_

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
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.attn = nn.MultiheadAttention(hidden_size, hidden_size)

    def forward(self, input, hidden):
        """
        Forward pass of the RNN.
        """

        input.to(torch.double)
        input__ = input.to(torch.double).clone()
        input_ = self.input_layer(input__)

        h0 = self.hidden_layer(hidden.clone())
        #   hidden = F.relu(hidden.clone(), inplace=False)
       # print(input_.shape, h0.shape)
        output = self.attn(input_, input_, input_)
        output = self.output_layer(output[0].clone())
        return output, hidden

    def init_hidden(self, batch_size):
        """
        Initialize the hidden state of the RNN.
        """
        return torch.zeros((batch_size, self.hidden_size, self.hidden_size))

    def predict(self, input, hidden):
        """
        Predict the next word given the input and the hidden state.
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
            outputs, hidden = model(inputs.clone(), hidden.clone())
            # 6. Compute the loss
            loss = criterion(outputs.clone(), targets.clone())
            # 7. Compute the gradients
            loss.backward(retain_graph=True)
            # 8. Update the weights
            optimizer.step()
            mse = MeanSquaredError()
            valid_loss = 0.0
            model.eval()
            for vinputs, vtargets in validation_dl:
                voutputs = model.predict(vinputs.clone(), hidden.clone())

                loss = mse(voutputs.clone(), vtargets.clone())
                valid_loss += loss.item()
            print(valid_loss)
        # 9. Print the loss

        print(epoch, loss)
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
test_size = int(len(data) * 0.9)
train_set, val_set = data[:test_size], data[test_size:]
model = RNN(max_size, max_size, max_size)
dataloader_ = RNNDataloader(train_set, batch_size=1)
validation_dataloader_ = RNNDataloader(val_set, batch_size=1)
model = train(model, dataloader_, validation_dataloader_,num_epochs=100, learning_rate=0.001)
# model = save_model(model)
# model = load_model()
# predict(model, dataloader_)
