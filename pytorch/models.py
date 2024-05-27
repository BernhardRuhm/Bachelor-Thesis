import torch
import torch.nn as nn
import torch.nn.functional as F
from custom_layers import script_lstm, FocusedLSTMCell, PositionalEncoding 

import torch.jit as jit
from torch import Tensor
from torch.nn import Parameter

from typing import List, Tuple


def init_weights(m):
    if isinstance(m, nn.Linear):
        # same as glorot uniform
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

    elif isinstance(m, nn.Conv1d):
        # same as he uniform 
        nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0)

    elif isinstance(m, nn.LSTM):
        # iterate over all parameters in case of multilayer lstm
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias_ih' in name:
                param.data.fill_(0)
                n = param.size(0)
                # forget gate bias set to 1
                param.data[(n // 4):(n // 2)].fill_(1)
            elif 'bias_hh' in name:
                param.data.fill_(0)

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
    def __init__(self, device, input_dim, n_classes, hidden_size=128, n_layers=1, filters=[128, 256, 128], kernels=[3, 5, 8]):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=n_layers, batch_first=False)
        self.dropout = nn.Dropout(p=0.8)

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
        y1 = self.dropout(y1)

        y2 = torch.permute(x, (1, 2, 0))
        y2 = self.relu(self.bn1(self.conv1(y2)))
        y2 = self.relu(self.bn2(self.conv2(y2)))
        y2 = self.relu(self.bn3(self.conv3(y2)))
        y2 = torch.mean(y2, 2)
       
        y = torch.cat((y1, y2), dim=1)
        y = self.dense(y)
        return y

class LSTM(nn.Module):
    def __init__(self, device, input_dim, hidden_size, n_classes, n_layers, batch_norm=0):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_norm_type = batch_norm

        self.cnn = nn.Conv1d(input_dim, 128, 7, padding="same")
        self.cnn_ln = nn.LayerNorm(32)
        self.relu = nn.ReLU()

        self.lstm = nn.ModuleList() 
        self.bn = nn.ModuleList()
        self.dropout = nn.ModuleList()


        for i in range(n_layers):
            # Create LSTM Module List
            if i==0:
                self.lstm.append(nn.LSTM(128, hidden_size))
            else:
                self.lstm.append(nn.LSTM(hidden_size, hidden_size))

            # Create Normalization Module List 
            if batch_norm == 1: 
                self.bn.append(nn.BatchNorm1d(hidden_size, affine=False))
            elif batch_norm == 2:
                self.bn.append(nn.BatchNorm1d(hidden_size, affine=True))
            elif batch_norm == 3:
                self.bn.append(nn.LayerNorm(hidden_size))
            elif batch_norm == 4:
                if i == 0:
                    self.bn.append(nn.LayerNorm(input_dim))
                else:
                    self.bn.append(nn.LayerNorm(hidden_size))
            elif batch_norm == 5:
                if i == 0:
                    self.bn.append(nn.LayerNorm(128))
                else:
                    self.bn.append(nn.LayerNorm(hidden_size))

            # Create Dropout Module List
            # if i != self.n_layers - 1:
            self.dropout.append(nn.Dropout(0.3))

        # self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=n_layers, batch_first=False)

        self.dense = nn.Linear(hidden_size , n_classes)

    def init_hidden_states(self, batch_size):
        h = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(self.device)
        c = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(self.device)
        return (h, c)

    def forward(self, x):
        h0, c0 = self.init_hidden_states(x.shape[1]) 

        # x = x.permute(1, 2, 0)
        # x = self.cnn(x)
        # x = F.relu(x)
        # x = x.permute(2, 0, 1) 

        for i in range(self.n_layers):
            if self.batch_norm_type == 1 or self.batch_norm_type == 2:
                x, _ = self.lstm[i](x, (h0[i:i+1, : ], c0[i:i+1, :]))
                x = self.bn[i](x.permute(1, 2, 0)).permute(2, 0, 1)
            elif self.batch_norm_type == 3:
                x, _ = self.lstm[i](x, (h0[i:i+1, : ], c0[i:i+1, :]))
                x = self.bn[i](x.permute(1, 0, 2)).permute(1, 0, 2)
            elif self.batch_norm_type == 4:
                x = self.bn[i](x.permute(1, 0, 2)).permute(1, 0, 2)
                x, _ = self.lstm[i](x, (h0[i:i+1, : ], c0[i:i+1, :]))
            elif self.batch_norm_type == 5:
                x = self.bn[i](x.permute(1, 0, 2)).permute(1, 0, 2)
                x, _ = self.lstm[i](x, (h0[i:i+1, : ], c0[i:i+1, :]))
                # if i != self.n_layers - 1:
                x = self.dropout[i](x)

        y = x[-1, :]
        y = self.dense(y)
        return y

class FCN(nn.Module):
    def __init__(self, device, input_dim,  n_classes, filters=[128, 256, 128], kernels=[3, 5, 8]):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        
        self.conv1 = nn.Conv1d(self.input_dim, filters[0], kernels[0], padding="same")
        self.bn1 = nn.BatchNorm1d(filters[0], eps=0.001, momentum=0.99)
        self.conv2 = nn.Conv1d(filters[0], filters[1], kernels[1], padding="same")
        self.bn2 = nn.BatchNorm1d(filters[1], eps=0.001, momentum=0.99)
        self.conv3 = nn.Conv1d(filters[1], filters[2], kernels[2], padding="same")
        self.bn3 = nn.BatchNorm1d(filters[2], eps=0.001, momentum=0.99)

        self.relu = nn.ReLU()
        self.dense = nn.Linear(filters[2], n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y = torch.permute(x, (1, 2, 0))
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.relu(self.bn2(self.conv2(y)))
        y = self.relu(self.bn3(self.conv3(y)))
        y = torch.mean(y, 2)
       
        y = self.dense(y)

        return y

# class BNLSTMCell(nn.Module):
#     def __init__(self, input_dim, hidden_size, n_layers):
#         self.input_dim = input_dim
#         self.hidden_size = hidden_size
#         self.weight_ih = nn.Parameter()
#         self.weight_hh = nn.Parameter()
#         self.bias_ih = nn.Parameter()
#         self.bias_hh = nn.Parameter()

#     def forward():
#         pass

def generate_model(model_name, device, input_dim, hidden_size, n_classes, n_layers, batch_norm):
    if model_name == "LSTMFCN":
        model = LSTMFCN(device, input_dim, n_classes)
    elif model_name == "FCN":
        model = FCN(device, input_dim, n_classes)
    elif model_name ==  "LSTM":
        model = LSTM(device, input_dim, hidden_size, n_classes, n_layers, batch_norm)

    return model

valid_models = [
    ("LSTMFCN", LSTMFCN),
    ("LSTM", LSTM),
    ("FCN", FCN)
]
