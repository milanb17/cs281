import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 2, bias = True, dropout = 0.1, bidirectional = True):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bias=bias, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x):
        outpt, _ = self.lstm(x)
        return outpt.squeeze()

# data = torch.randn(100, 1, 512)
# model = LSTM(512, 256, 2)
# print(model(data).size())