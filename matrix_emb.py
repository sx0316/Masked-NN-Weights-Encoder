"""
A RNN from scratch for a sequence of data in pytorch
1. Write the backbone of the RNN
2. Input one sample of the data
3. Train the model
4. Save the model
5. Generate a list of matrix and save it as a dataset
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.utils.data.distributed as dist
import torch.utils.data.dataloader as dataloader
import torch.utils.data.dataset as dataset


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

    def forward(self, input, hidden):
        """
        Forward pass of the RNN.
        """
        hidden = self.hidden_layer(hidden)
        hidden = F.relu(hidden)
        output = self.output_layer(hidden)
        output = F.softmax(output, dim=1)
        return output, hidden

    def init_hidden(self, batch_size):
        """
        Initialize the hidden state of the RNN.
        """
        return torch.zeros(batch_size, self.hidden_size)


class RNNDataset(dataset.Dataset):
    """
    Dataset for the RNN.
    """
    def __init__(self, data):
        super(RNNDataset, self).__init__()
        self.data = data

    def __getitem__(self, index):
        """
        Get one sample of the dataset.
        """
        return self.data[index]

    def __len__(self):
        """
        Get the length of the dataset.
        """
        return len(self.data)



class RNNDataLoader(dataloader.DataLoader):
    """
    DataLoader for the RNN.
    """

    def __init__(self, data, batch_size, shuffle=True, num_workers=1):
        super(Dataloader, self).__init__(data, batch_size, shuffle, num_workers)
        self.data = data
        self.batch_size = batch_size
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
            inputs = [matrix[:, :, np.newaxis] for matrix in batch]
            targets = [matrix[:, :, np.newaxis] for matrix in batch]
            # 6. Return the inputs and targets
            yield inputs, targets

class RNNTrainer(object):
    """
    Trainer for the RNN.
    """
    def __init__(self, model, optimizer, loss_func, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.device = device

    def train(self, train_data, epochs):
        """
        Train the model.
        """
        self.model.train()
        for epoch in range(epochs):
            for step, (x, y) in enumerate(train_data):
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                output, _ = self.model(x)
                loss = self.loss_func(output, y)
                loss.backward()
                self.optimizer.step()
                if step % 100 == 0:
                    print('Epoch: {}, Step: {}, Loss: {}'.format(epoch, step, loss.item()))

    def test(self, test_data):
        """
        Test the model.
        """
        self.model.eval()
        with torch.no_grad():
            for x, y in test_data:
                x = x.to(self.device)
                y = y.to(self.device)
                output, _ = self.model(x)
                loss = self.loss_func(output, y)
                print('Loss: {}'.format(loss.item()))

    def save(self, path):
        """
        Save the model.
"""
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """
        Load the model.
        """
        self.model.load_state_dict(torch.load(path))



class RNNGenerator(object):
    """
    Generator for the RNN.
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def generate(self, input):
        """
        Generate a sequence of data.
        """
        self.model.eval()
        with torch.no_grad():
            input = input.to(self.device)
            output, _ = self.model(input)
            return output



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

    def forward(self, input, hidden):
        """
        Forward pass of the RNN.
        """
        hidden = self.hidden_layer(hidden)
        hidden = F.relu(hidden)
        output = self.output_layer(hidden)
        output = F.softmax(output, dim=1)
        return output, hidden

    def init_hidden(self, batch_size):
        """
        Initialize the hidden state of the RNN.
        """
        return torch.zeros(batch_size, self.hidden_size)

#5. Generate a list of matrix and save it as a dataset
def generate_data(length, input_size, output_size):
    """
    Generate a list of matrix.
    """
    data = []
    for i in range(length, 10
                   ):
        ls = []
        for j in range(10):
            input = torch.randn(1, input_size)
            mask = torch.randn(1, input_size)
            output = torch.mul(input, mask)
            ls.append((input, output))
        data.append(ls)
    return data

#6. Train the model
def train_model(model, optimizer, loss_func, train_data, epochs):
    """
    Train the model.
    """
    model.train()
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_data):
            optimizer.zero_grad()
            output, _ = model(x)
            loss = loss_func(output, y)
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print('Epoch: {}, Step: {}, Loss: {}'.format(epoch, step, loss.item()))
    return model

seq = generate_data(1000, 256, 256)
dataset = RNNDataset(seq)
train_data = RNNDataLoader(dataset, batch_size=32)
model = RNN(256, 256, 256)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.MSELoss()
model = train_model(model, optimizer, loss_func, train_data, epochs=100)
path = 'rnn_model.pt'
model.save(path)
model = RNN(256, 256, 256)
model.load(path)
input = seq[0]
output = model.generate(input)
model.test(train_data)