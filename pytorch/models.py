import torch
import torch.nn as nn
import torch.nn.functional as F
from custom_layers import script_lstm, FocusedLSTMCell, PositionalEncoding 

import torch.jit as jit
from torch import Tensor
from torch.nn import Parameter

from typing import List, Tuple

class FocusedDense(jit.ScriptModule):
    def __init__(self, device, seq_len, input_dim, hidden_size, n_classes, n_layers, positional_encoding):
        super().__init__()
        self.device = device
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.lstm = script_lstm(input_dim, hidden_size, n_layers, cell=FocusedLSTMCell)
        self.dense = nn.Linear(hidden_size, n_classes)
        self.softmax = nn.Softmax(dim=1)
        
    @jit.script_method
    def forward(self, x):
        batch_size = x.shape[1]
        states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        
        for _ in range(self.n_layers):
            states.append((
                torch.zeros(batch_size, self.hidden_size).to(self.device),
                torch.zeros(batch_size, self.hidden_size).to(self.device))
            )

        o, _ = self.lstm(x, states)
        x = o[-1, :]
        x = self.dense(x)
        x = self.softmax(x)

        return x

class VanillaDense(nn.Module):
    def __init__(self, device, seq_len, input_dim, hidden_size, n_classes, n_layers, positional_encoding):
        super().__init__()
        self.device = device
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.positional_encoding = positional_encoding

        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=n_layers, batch_first=False)
        self.softmax = nn.Softmax(dim=1)
        self.dense = nn.Linear(hidden_size, n_classes)

        if positional_encoding:
            self.pos_enc = PositionalEncoding(input_dim, seq_len, device)

    def forward(self, x):
        h = torch.zeros(self.n_layers, x.shape[1], self.hidden_size).to(self.device)
        c = torch.zeros(self.n_layers, x.shape[1], self.hidden_size).to(self.device)
        
        if self.positional_encoding:
            x = self.pos_enc(x)

        o, _ = self.lstm(x, (h, c))
        x = o[-1, :]
        x = self.dense(x)
        x = self.softmax(x)
        return x


class LSTMFCN(nn.Module):
    def __init__(self, device, input_dim, hidden_size, n_classes, n_layers, filters, kernels):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=n_layers, batch_first=False)
        self.lstm_dropout = nn.Dropout(p=0.8)

        self.conv1 = nn.Conv1d(self.input_dim, filters[0], kernels[0], padding="same")
        self.bn1 = nn.BatchNorm1d(filters[0], eps=0.001, momentum=0.99)
        self.conv2 = nn.Conv1d(filters[0], filters[1], kernels[1], padding="same")
        self.bn2 = nn.BatchNorm1d(filters[1], eps=0.001, momentum=0.99)
        self.conv3 = nn.Conv1d(filters[1], filters[2], kernels[2], padding="same")
        self.bn3 = nn.BatchNorm1d(filters[2], eps=0.001, momentum=0.99)

        self.relu = nn.ReLU()
        self.dense = nn.Linear(hidden_size + filters[2], n_classes)
        self.softmax = nn.Softmax(dim=1)

    def init_hidden_states(self, batch_size):
        h = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(self.device)
        c = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(self.device)
        return (h, c)

    def forward(self, x):
        states = self.init_hidden_states(x.shape[1]) 

        y1, _ = self.lstm(x, states)
        y1 = y1[-1, :]
        y1 = self.lstm_dropout(y1)

        y2 = torch.permute(x, (1, 2, 0))
        y2 = self.relu(self.bn1(self.conv1(y2)))
        y2 = self.relu(self.bn2(self.conv2(y2)))
        y2 = self.relu(self.bn3(self.conv3(y2)))
        y2 = torch.mean(y2, 2)
       
        y = torch.cat((y1, y2), dim=1)
        y = self.dense(y)
        y = self.softmax(y)
        return y

valid_models = [
    ("vanilla_lstm", VanillaDense),
    ("focused_lstm", FocusedDense),
    ("lstm_fcn", LSTMFCN)
]
