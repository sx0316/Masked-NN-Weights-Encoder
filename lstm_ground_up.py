"""
Code up a LSTM for sequential data
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.nn.init as init
import torch.nn.utils.rnn as rnn_utils
import torch.nn.utils.weight_norm as weight_norm
import torch.nn.init as init
import torch.nn.utils.clip_grad as clip_grad
import torch.nn.utils.pruning as pruning
import torch.nn.utils.weight_norm as weight_norm
import torch.nn.utils.rnn as rnn_utils
import torch.nn.utils.pruning as pruning
import torch.nn.utils.clip_grad as clip_grad
import torch.nn.utils.weight_norm as weight_norm
import torch.nn.utils.rnn as rnn_utils
import torch.nn.utils.pruning as pruning
import torch.nn.utils.clip_grad as clip_grad
import torch.nn.utils.weight_norm as weight_norm
import torch.nn.utils.rnn as rnn_utils
import torch.nn.utils.pruning as pruning
import torch.nn.utils.clip_grad as clip_grad
import torch.nn.utils.weight_norm as weight_norm
import torch.nn.utils.rnn as rnn_utils
import torch.nn.utils.pruning as pruning
import torch.nn.utils.clip_grad as clip_grad
import torch.nn.utils.weight_norm as weight_norm
import torch.nn.utils.rnn as rnn_utils
import torch.nn.utils.pruning as pruning
import torch.nn.utils.clip_grad as clip_grad
import torch.nn.utils.weight_norm as weight_norm
import torch.nn.utils.rnn as rnn_utils
import torch.nn.utils.pruning as pruning
import torch.nn.utils.clip_grad as clip_grad
import torch.nn.utils.weight_norm as weight_norm
import torch.nn.utils.rnn as rnn_utils


class LSTM(nn.Module):
    """
    LSTM from scratch for a sequence of data in pytorch
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.cell_layer = nn.Linear(hidden_size, hidden_size)
        self.cell_layer_bias = nn.Linear(hidden_size, hidden_size)
        self.cell_layer_bias_b = nn.Linear(hidden_size, hidden_size)
        self.cell_layer_bias_c = nn.Linear(hidden_size, hidden_size)
        self.cell_layer_bias_d = nn.Linear(hidden_size, hidden_size)
        self.cell_layer_bias_e = nn.Linear(hidden_size, hidden_size)
        self.cell_layer_bias_f = nn.Linear(hidden_size, hidden_size)
        self.cell_layer_bias_g = nn.Linear(hidden_size, hidden_size)
        self.cell_layer_bias_h = nn.Linear(hidden_size, hidden_size)
        self.cell_layer_bias_i = nn.Linear(hidden_size, hidden_size)
        self.cell_layer_bias_j = nn.Linear(hidden_size, hidden_size)
        self.cell_layer_bias_k = nn.Linear(hidden_size, hidden_size)

    def forward(self, *input, **kwargs):
        """
        Forward pass for the LSTM
        """
        hidden = kwargs.get('hidden', None)
        cell = kwargs.get('cell', None)
        if hidden is None:
            hidden = self.init_hidden()
        if cell is None:
            cell = self.init_cell()
        for i in range(len(input)):
            hidden, cell = self.forward_step(input[i], hidden, cell)
        return hidden, cell
