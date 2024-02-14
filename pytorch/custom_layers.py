import torch
import torch.jit as jit
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter

from typing import List, Tuple


class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, seq_len, device):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.device = device
        self.register_buffer("positional_encodings", self.build_positional_encodings())

    def build_positional_encodings(self):
        positions = torch.arange(self.seq_len).to(self.device)
        min_freq = 1 / 10000

        timescales = torch.pow(
            min_freq,
            2 * (torch.arange(self.input_dim) // 2) / self.input_dim).to(self.device)

        angles = torch.unsqueeze(positions, 1) * torch.unsqueeze(timescales, 0).to(self.device)

        cos_mask = (torch.arange(self.input_dim) % 2).to(self.device) 
        sin_mask = (1 - cos_mask).to(self.device)

        return torch.sin(angles) * sin_mask + torch.cos(angles) * cos_mask

    def forward(self, x):
        return x + torch.unsqueeze(self.positional_encodings, 1)




class FocusedLSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(FocusedLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel = Parameter(torch.empty(hidden_size, input_size))
        self.recurrent_kernel = Parameter(torch.empty(2 * hidden_size, hidden_size))
        self.recurrent_bias = Parameter(torch.zeros(2 * hidden_size))
        self.bias = Parameter(torch.zeros(hidden_size))

        nn.init.xavier_uniform_(self.kernel)
        nn.init.xavier_uniform_(self.recurrent_kernel)

    @jit.script_method
    def forward(
        self, input: Tensor, state: Tuple[Tensor, Tensor]
        ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = state

        i_o = torch.matmul(hx, self.recurrent_kernel.t()) + self.recurrent_bias
        i_o = torch.sigmoid(i_o)
        z = torch.matmul(input, self.kernel.t()) + self.bias
        z = torch.tanh(z)
        
        i, o = i_o.chunk(2, 1)

        cy =  cx + (i * z)
        hy = o * torch.tanh(cy)

        return hy, (hy, cy)


class LSTMLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(LSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs.append(out)
        return torch.stack(outputs), state


def init_stacked_lstm(num_layers, layer, first_layer_args, other_layer_args):
    layers = [layer(*first_layer_args)] + [
        layer(*other_layer_args) for _ in range(num_layers - 1)
    ]
    return nn.ModuleList(layers)

class StackedLSTM(jit.ScriptModule):
    # __constants__ = ["layers"]  # Necessary for iterating through self.layers

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super().__init__()
        self.layers = init_stacked_lstm(
            num_layers, layer, first_layer_args, other_layer_args
        )

    @jit.script_method
    def forward(
        self, input: Tensor, states: List[Tuple[Tensor, Tensor]]
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        # List[LSTMState]: One state per layer
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        output = input
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, state)
            output_states += [out_state]
            i += 1
        return output, output_states


def script_lstm(
    input_size,
    hidden_size,
    num_layers,
    cell,
    batch_first=False,
):
    """Returns a ScriptModule that mimics a PyTorch native LSTM."""


    stack_type = StackedLSTM
    layer_type = LSTMLayer
    dirs = 1
    
    return stack_type(
        num_layers,
        layer_type,
        first_layer_args=[cell, input_size, hidden_size],
        other_layer_args=[cell, hidden_size * dirs, hidden_size],
    )
