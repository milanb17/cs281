# Transformer over output full sequence. This will likely be very expensive
# to train. We explore transformer models that use relative attention
# vs. absolute attention in further models.

import torch
import torch.nn as nn
import numpy as np

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers=6, d_model=512, nhead=8):
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
    
    def forward(self, tgt, memory):
        return self.transformer_decoder(tgt, memory)

# Mask future frames
def next_mask(size):
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8') == 0
    return torch.from_numpy(mask)
