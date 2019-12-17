# pytorch model 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision 
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns 
import collections
import functools 
from operator import mul


# based off of https://algorithmia.com/blog/convolutional-neural-nets-in-pytorch

class BasicCNN(nn.Module): 
    def __init__(self, in_channels, input_sz, embedding_sz=512): 
        super().__init__()
        self.conv1_outpt_sz = 64 
        self.conv_layer = nn.Conv2d(in_channels, self.conv1_outpt_sz, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fully_connected_a = nn.Linear(input_sz * input_sz * 16, embedding_sz)

    def forward(self, x): 
        # (n, n, 3) -> (n, n, 64)
        relu_first_layer = F.relu(self.conv_layer(x))
        # (n, n, 64) -> (n/2, n/2, 64)
        pooled = self.max_pool(relu_first_layer)
        # (n/2, n/2, 64) -> (1, 16 * n^2)
        flatten = pooled.view(-1, functools.reduce(mul, pooled.shape[1:]))

        outpt = torch.tanh(self.fully_connected_a(flatten))
        return outpt



# example = torch.randn((100, 3, 64, 64))

# model = BasicCNN(3, 64)
# print(model(example).shape)