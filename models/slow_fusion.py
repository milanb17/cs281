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


# can only take in 10 frame inputs 

class SlowFusion(nn.Module): 
    def __init__(self, input_channels, input_seq_len, input_sz, embedding_sz=512): 
        super().__init__()
        self.num_output_filters_1 = 16
        self.num_output_filters_2 = self.num_output_filters_1 * 2
        self.num_output_filters_3a = self.num_output_filters_2 * 2
        self.num_output_filters_3b = self.num_output_filters_3a * 2 
        self.num_output_filters_3c = self.num_output_filters_3b * 2 

        # Conv Block 1 
        self.conv1 = nn.Conv3d(input_channels, self.num_output_filters_1, kernel_size=(4, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1))
        self.norm1 = nn.BatchNorm3d(self.num_output_filters_1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0) 

        # Conv Block 2 
        self.conv2 = nn.Conv3d(self.num_output_filters_1, self.num_output_filters_2, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1))
        self.norm2 = nn.BatchNorm3d(self.num_output_filters_2)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)

        # Conv Block 3 
        self.conv3a = nn.Conv3d(self.num_output_filters_2, self.num_output_filters_3a, kernel_size=(2, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv3b = nn.Conv2d(self.num_output_filters_3a, self.num_output_filters_3b, kernel_size=1, stride=1, padding=0)
        self.conv3c = nn.Conv2d(self.num_output_filters_3b, self.num_output_filters_3c, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)

        # FC Layers 
        self.embedding = nn.Linear(self.num_output_filters_3c * 64, embedding_sz)

        
    def forward(self, x): 
        #  Conv Block 1
        # (3, 10, 64, 64) -> (self.num_output_filters_1, 4, 32, 32)
        x = self.conv1(x)
        x = self.norm1(x) 
        x = F.relu(x)
        x = self.maxpool1(x)

        # Conv Block 2 
        # (self.num_output_filters_1, 4, 32, 32) -> (self.num_output_filters_2, 2, 16, 16)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.maxpool2(x)

        # Conv Block 3 
        # (self.num_output_filters_2, 2, 16, 16) -> (self.num_output_filters_3, 1, 8, 8)
        # print(relu_conv3_outpt.size())
        x = F.relu(self.conv3a(x))
        x = x.view((-1, x.shape[1], x.shape[3], x.shape[4]))
        x = F.relu(self.conv3b(x))
        x = F.relu(self.conv3c(x))
        x = self.pool3(x)
        x = x.view(-1, functools.reduce(mul, x.shape[1:])) 

        # Embedding 
        return torch.tanh(self.embedding(x))

# inpt = torch.randn(100, 3, 10, 64, 64) 
# model = SlowFusion(3, 10, 64)
# outpt = model(inpt)
# print(outpt.size())