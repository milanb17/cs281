import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.1, bidirectional=True):
        super().__init__()
        self.model = nn.RNN(input_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x):
        return self.model(x)
