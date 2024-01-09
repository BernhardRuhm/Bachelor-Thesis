import numbers
import warnings
from collections import namedtuple
from typing import List, Tuple

import torch
import torch.jit as jit
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter

class LSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(2 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(hidden_size))
        self.bias_hh = Parameter(torch.randn(2 * hidden_size))

    @jit.script_method
    def forward(
        self, input: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state
        # hx = hx.squeeze()
        # cx = cx.squeeze()
        z = torch.mm(input, self.weight_ih.t()) + self.bias_ih
        r = torch.mm(hx, self.weight_hh.t()) + self.bias_hh
        
        i, o = r.chunk(2, 1)
        # print(input.shape)
        z = torch.tanh(z)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)

        cy = cx + (i * z)
        hy = o * torch.tanh(cy)
        
        # hy = hy.unsqueeze(0)
        # cy = cy.unsqueeze(0)

        return hy, (hy, cy)

class LSTMLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super().__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(
        self, input: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        # inputs = input.unbind(0)

        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(input.shape[1]):
            out, state = self.cell(input[:,i,:], state)
            outputs += [out]

        outputs = torch.stack(outputs)
        outputs = outputs.permute(1, 0, 2)
        return outputs, state

def init_stacked_lstm(num_layers, input_dim, hidden_size):
    layers = [LSTMLayer(LSTMCell, input_dim, hidden_size)] + [
        LSTMLayer(LSTMCell, hidden_size, hidden_size) for _ in range(num_layers - 1)
    ]
    return nn.ModuleList(layers)


class StackedLSTMLayer(jit.ScriptModule):
    def __init__(self, n_layers, input_dim, hidden_size):
        super().__init__()
        self.layers = init_stacked_lstm(n_layers, input_dim, hidden_size)

    @jit.script_method
    def forward(
        self, input: Tensor, states: List[Tuple[Tensor, Tensor]]
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        # List[LSTMState]: One state per layer
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        output = input
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, state)
            output_states += [out_state]
            i += 1
        return output, output_states
