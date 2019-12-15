import torch
import torch.nn as nn

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers=5):
        decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
    
    def forward(self, x):
        memory = torch.rand(10, 32, 512)
        tgt = torch.rand(20, 32, 512)
        return self.transformer_decoder(tgt, memory)
        