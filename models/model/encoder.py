import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.embeddings.transformer_econding import TransformerEmbedding
from models.blocks.encoder_layer import EncoderLayer
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self,enc_voc_size,max_len,d_model, ffn_hidden, n_head,n_layers, drop_porb,device):
        super().__init__()
        self.emb = TransformerEmbedding(enc_voc_size,d_model=d_model,max_len=max_len,drop_prob=drop_porb,device=device)
        
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,ffn_hidden=ffn_hidden,n_head=n_head,drop_prob=drop_porb) for _ in range(n_layers)])
    
    def forward(self,x,src_mask):
        x = self.emb(x)
        
        for layer in self.layers:
            x = layer(x,src_mask)
        
        return x
    
