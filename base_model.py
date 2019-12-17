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

class Model(nn.Module): 
    def __init__(self, img_model, seq_model):
        super().__init__() 

        self.img_model, self.seq_model = None, None

        if img_model == "slow_fusion":
            from models.slow_fusion import SlowFusion 
            self.img_model = SlowFusion(3, 10, 64)
        elif img_model == "early_fusion": 
            from models.early_fusion import EarlyFusion
            self.img_model = EarlyFusion(3, 10, 64)
        elif img_model == "late_fusion": 
            from models.late_fusion import LateFusion
            self.img_model = LateFusion(3, 10, 64)
        elif img_model == "vanilla_cnn":
            from models.basic_cnn import BasicCNN
            self.img_model = BasicCNN(3, 64)
        else: 
            from models.imagenet_model_wrapper import ImageNet_Model_Wrapper
            self.img_model = ImageNet_Model_Wrapper(img_model)

        if seq_model == "vanilla_rnn": 
            from models.rnn import RNN
            self.seq_model = RNN(512, 256, 2)
        elif seq_model == "lstm": 
            from models.lstm import LSTM
            self.seq_model = LSTM(512, 256, num_layers=2, dropout=0.1, bidirectional=True)
        elif seq_model == "lstmn": 
            from models.lstmn import BiLSTMN
            self.seq_model = BiLSTMN(512, 256, num_layers=2, dropout=0.1, tape_depth=10)
        elif seq_model == "transformer_abs": 
            from models.transformer import Transformer 
            self.seq_model = Transformer(512, 8)
        elif seq_model == "stack_lstm": 
            from models.stack_lstm import EncoderLSTMStack
            self.seq_model = EncoderLSTMStack(512, 256)

        # attention over seq_model output
        self.query_vector = nn.Parameter(torch.randn(1, 64))
        # self.attn_w  = nn.Bilinear(64, 512, 1)
        self.attn_w = nn.Parameter(torch.randn(64, 512))

        self.linear1 = nn.Linear(512, 32)
        self.linear2 = nn.Linear(32, 1)
        
    def forward(self, x): 
        # run cnn: img_data -> 512
        embed = self.img_model(x)
        # print(f"embed_post_img: {embed.size()}")
        # output: (100, 512)

        # unsqueeze to (100, 1, 512)
        embed = embed.unsqueeze(1)

        embed = self.seq_model(embed)
        # print(embed.shape)
        # output: (frame, 512)

        attn = torch.mm(torch.mm(self.query_vector, self.attn_w), embed.t())
        # output: (1, frame)
        attn = F.softmax(attn, dim=1).t()

        ctxt = torch.sum(attn * embed, dim=0)

        embed = F.relu(self.linear1(ctxt)) 
        return torch.sigmoid(self.linear2(embed))
