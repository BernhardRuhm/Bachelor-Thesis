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
        # self.lstm = LSTMLayer(LSTMCell, input_dim, hidden_size)
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

valid_models = [
    ("vanilla_lstm", VanillaDense),
    ("focused_lstm", FocusedDense)
]
