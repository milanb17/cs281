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


class LateFusion(nn.Module): 
    def __init__(self, input_channels, input_seq_len, input_sz, embedding_sz=512, diff_in_tape_length=10): 
        super().__init__()
        self.tape_length = diff_in_tape_length
        self.layer_1_filters = 64 
        self.layer_2_filters = self.layer_1_filters * 2 
        self.layer_3a_filters = self.layer_2_filters * 2 
        self.layer_3b_filters = self.layer_3a_filters * 2 
        self.layer_3c_filters = self.layer_3b_filters 

        assert(self.tape_length <= input_seq_len)

        #  Conv Block 1 
        self.conv1 = nn.Conv2d(input_channels, self.layer_1_filters, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(self.layer_1_filters)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Conv Block 2 
        self.conv2 = nn.Conv2d(self.layer_1_filters, self.layer_2_filters, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.BatchNorm2d(self.layer_2_filters)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Conv Block 3 
        self.conv3a = nn.Conv2d(self.layer_2_filters, self.layer_3a_filters, kernel_size=3, stride=2, padding=1)
        self.conv3b = nn.Conv2d(self.layer_3a_filters, self.layer_3b_filters, kernel_size=3, stride=1, padding=1)
        self.conv3c = nn.Conv2d(self.layer_3b_filters, self.layer_3c_filters, kernel_size=3, stride=1, padding=0)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)

        # FC layer 
        self.embed = nn.Linear(1024, embedding_sz)

    def __forward(self, x):  
        # Conv Block 1 
        x = self.conv1(x)
        x = F.relu(self.norm1(x))
        x = self.maxpool1(x)

        # Conv Block 2 
        x = self.conv2(x)
        x = F.relu(self.norm2(x))
        x = self.maxpool2(x)

        # Conv Block 3 
        x = F.relu(self.conv3a(x))
        x = F.relu(self.conv3b(x))
        x = F.relu(self.conv3c(x))
        x = self.maxpool3(x)
        x = x.view(-1, x.shape[1])
        return x 

    def forward(self, x): 
        # Take embeddings tape_length apart 
        slice_1 = x[:, :, 0, :, :]
        slice_2 = x[:, :, self.tape_length - 1, :, :]
        # reshape embeddings 
        slice_shape = (-1, x.shape[1], x.shape[3], x.shape[4])
        slice_1 = slice_1.view(slice_shape)
        slice_2 = slice_2.view(slice_shape)
        print(slice_1.shape)
        # process embeddings 
        embed_1 = self.__forward(slice_1)
        embed_2 = self.__forward(slice_2)
        # concatenate embeddings 
        x = torch.cat((embed_1, embed_2), dim=1)
        # process in embedding_layer 
        return torch.tanh(self.embed(x))


        
inpt = torch.randn(1, 3, 10, 64, 64) 
model = LateFusion(3, 10, 64)
outpt = model(inpt)
print(outpt.shape)
# print(model.state_dict())