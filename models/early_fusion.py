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


class EarlyFusion(nn.Module): 
    def __init__(self, input_channels): 
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=())
    def forward(self, x): 
        pass

# input_data = torch.randn(3, 64, 64, 10) 
# model = EarlyFusion()
# print(EarlyFusion(input_data))