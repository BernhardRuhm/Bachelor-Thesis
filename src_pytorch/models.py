import torch
import torch.nn as nn
import torch.nn.functional as F
from focused_lstm import LSTMCell, LSTMLayer, StackedLSTMLayer

class FocusedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FocusedLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W = nn.Parameter(torch.randn(input_size, hidden_size))
        self.R = nn.Parameter(torch.randn(hidden_size, hidden_size * 2))
        self.b_r = nn.Parameter(torch.zeros(hidden_size * 2))
        self.b_w = nn.Parameter(torch.zeros(hidden_size))

        # self.hidden_state = torch.zeros(1, batch_size, self.n_hidden).to(next(self.parameters()).device)  
        # self.cell_state = torch.zeros(1, batch_size, self.n_hidden).to(next(self.parameters()).device)  

    def forward(self, x, states=None):
        # if states is None:
        #     y_t, c_t = self.hidden_state, self.cell_state 
        # else:
        y_t, c_t = states

        batch_size, seq_len,_ = x.shape
        outputs = []
        for t in range(seq_len):
            x_t = x[:,t,:]  
            r = torch.tanh(y_t @ self.R + self.b_r)
            z = torch.sigmoid(torch.mm(x_t, self.W) + self.b_w) 
            # print(r.shape)
            i, o = r.chunk(2, dim=1)
            
            c_t = c_t + i*z
            y_t = o * torch.tanh(c_t)
            # y_t = torch.squeeze(y_t, dim=0)
            # print(y_t.shape)
            outputs.append(y_t)

        return torch.stack(outputs), (y_t, c_t) 

# TODO: Change variable names
class LSTMDense(nn.Module):
    def __init__(self, seq_len, input_dim, n_hidden, n_layers, n_classes):
        super(LSTMDense, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_classes = n_classes

        # self.lstm = nn.LSTM(self.input_dim, self.n_hidden, num_layers=self.n_layers, batch_first=True)
        # self.lstm = FocusedLSTM(self.input_dim, self.n_hidden)
        # # self.lstm = LSTMLayer(LSTMCell, input_dim, n_hidden)
        self.lstm = StackedLSTMLayer(n_layers, input_dim, n_hidden)

        self.dense = nn.Linear(self.n_hidden, self.n_classes)

    # def init_states(self, batch_size):
    #     self.hidden_state = torch.zeros(batch_size, self.n_hidden).to(next(self.parameters()).device)  
    #     self.cell_state = torch.zeros(batch_size, self.n_hidden).to(next(self.parameters()).device)  
        
    def forward(self, x):
        
        h1 = torch.zeros(x.shape[0], self.n_hidden).to("cuda")  
        h2 = torch.zeros(x.shape[0], self.n_hidden).to("cuda")  
        c1 = torch.zeros(x.shape[0], self.n_hidden).to("cuda")  
        c2 = torch.zeros(x.shape[0], self.n_hidden).to("cuda")  
        # h = torch.zeros(x.shape[0], self.n_hidden).to("cuda")  
        # c = torch.zeros(x.shape[0], self.n_hidden).to("cuda")  
        state = [(h1, c1), (h2, c2)]
        o, _ = self.lstm(x, state)
        o = o.squeeze()
        x = o[:, -1]
        x = self.dense(x)
        x = F.softmax(x, dim=1)
        
        return x

