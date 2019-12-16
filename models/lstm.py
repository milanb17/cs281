import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1, bias = True, dropout = 0., bidirectional = False):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bias=bias, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x):
        return self.lstm(x)
