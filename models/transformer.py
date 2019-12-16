import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import math

class Transformer(nn.Module):
    def __init__(self, input_dim, nhead, linear_hidden=2048, dropout=0.1, num_layers=6):
        super().__init__()
        self.input_dim = input_dim
        self.pos_encoder = PositionalEmbeddings(input_dim)
        encoder_layer = nn.TransformerEncoderLayer(input_dim, nhead, dim_feedforward=linear_hidden, dropout=dropout)
        layer_normalizer = nn.LayerNorm(input_dim)
        self.transformer_decoder = nn.TransformerEncoder(encoder_layer, num_layers, norm=layer_normalizer)
    
    def forward(self, x):
        x = self.pos_encoder(x)
        return self.transformer_decoder(x)

    def gen_tgt_mask(self, x):
        attn_shape = (1, self.input_dim, self.input_dim)
        mask = np.triu(np.ones(attn_shape), k=1).astype('uint8') == 0
        return torch.from_numpy(mask)

class PositionalEmbeddings(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class RelativeEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        pass
