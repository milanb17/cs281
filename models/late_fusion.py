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
    def __init__(self): 
        super().__init__()
    def forward(self, x): 
        pass 