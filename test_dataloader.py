"""
Have a RNN take in a series of masked matrices and predict the masked entries
1. Generate a series of masked matrices
2. Write backbone of a RNN dataloader for masked matrices
3. Train the model
4. Save the model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim
import numpy as np
import operator

batch_size = 1

# 1. Generate a series of masked matrices
def generate_matrices(num_matrices):
    matrices = []
    for i in range(num_matrices):
        matrix = np.random.rand(256, 256)
        mask = np.random.rand(256, 256)
        mask = mask < 0.5
        masked = matrix * mask
        matrices.append((torch.from_numpy(masked.astype(np.float32)), torch.from_numpy(matrix.astype(np.float32))))
    return matrices

# 2. Write backbone of a RNN dataloader
class RNNDataloader(dataloader.DataLoader):
    """
    DataLoader for the RNN.
    """
    def __init__(self, data, batch_size, shuffle=True, num_workers=1):
        super(dataloader.DataLoader, self).__init__()
        self.data = data
        self.batch_size = 1#batch_size
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
        batches = [data[i:i+self.batch_size] for i in range(0, len(data), self.batch_size)]
        # 4. Iterate through the batches
        for batch in batches:
            # 5. Get the inputs and targets
            masked_, targ_ = [], []
            start = False

            for t in batch:
                masked, target = t[0].clone().detach(), t[1].clone().detach()
                if not start:
                    masked_ = torch.Tensor(masked.clone())
                    targ_ = torch.Tensor(target.clone())
                    start = True
                else:
                    masked_ = torch.vstack((masked_, torch.Tensor(masked)))
                    targ_ = torch.vstack((targ_, torch.Tensor(target)))
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
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)


    def forward(self, input, hidden):
        """
        Forward pass of the RNN.
        """
        input_ = self.input_layer(input.clone())

     #   hidden = self.hidden_layer(hidden.clone())
     #   hidden = F.relu(hidden.clone(), inplace=False)
        hn = hidden
        start_ind = 0
        prev_input = input[start_ind: start_ind+self.input_size, :]
        for i in range(len(input)):
            cur_input = input[start_ind: start_ind+self.input_size, :]
            outn, hn = self.rnn(cur_input, hn)
            #outn = self.attn(cur_input, prev_input, hn)
            start_ind += self.input_size
            prev_input = cur_input

        output = self.output_layer(outn.clone())
        return output, hidden

    def init_hidden(self, batch_size):
        """
        Initialize the hidden state of the RNN.
        """
        return torch.zeros(batch_size, self.hidden_size)

# Write training code with dataloader
# 3. Train the model
def train(model, dataloader, num_epochs, learning_rate):
    """
    Train the model.
    """
    # 1. Define the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    hidden = torch.from_numpy(np.zeros((1, 256)).astype(np.float32))

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
        # 9. Print the loss
        print(loss)
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

# 6. Predict the model
def predict(model, dataloader):
    """
    Predict the model.
    """
    # 1. Iterate through the data
    for inputs, targets in dataloader:
        # 2. Forward pass
        outputs, hidden = model(inputs, hidden)
        # 3. Print the outputs
        print(outputs)
    # 4. Return the model
    return model

data = generate_matrices(100)
model = RNN(256, 256, 256)
dataloader_ = RNNDataloader(data, batch_size=32)
model = train(model, dataloader_, num_epochs=10, learning_rate=0.001)
model = save_model(model)
model = load_model()
predict(model, dataloader_)
