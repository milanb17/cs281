import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pickle 
import argparse 
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np 
import itertools
from train import Model


device = torch.device("cuda")

for sm in ["vanilla_rnn", "lstm", "lstmn", "transformer_abs"]: 
    for im in ['early_fusion', 'late_fusion', 'slow_fusion', 'resnet', 'densenet', 'vgg', 'vanilla_cnn']: 
        print(f"TRYING EXAMPLE: {sm}, {im}")
        model = Model(im, sm, device).to(device)
        chunked_needed = im in frozenset(["slow_fusion", "early_fusion", "late_fusion"])
        data = None 
        labels = pickle.load(open("./_data/labels.p", "rb"))
        if chunked_needed:
            data = pickle.load(open("./_data/chunked_data.p", "rb"))
            # data = torch.randn(5, 10, 3, 10, 64, 64)
        else: 
            data = pickle.load(open("./_data/data.p", "rb"))
            # data = torch.randn(5, 100, 3, 64, 64)
        for peh in data: 
            print(model(torch.from_numpy(peh).float().to(device)))
            break