import os 
import sys
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.blocks.decoder_layer import DecoderLayer
from models.embeddings.transformer_econding import TransformerEmbedding
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self,dec_voc_size, max_len,d_model,ffn_hidden,n_head,n_layers,drop_prob,device):
        super().__init__()
        self.emb = TransformerEmbedding(vocab_size=dec_voc_size,d_model=d_model,max_len=max_len,drop_prob=drop_prob,device=device)
        
        self.layers = nn.ModuleList([DecoderLayer(dmodel=d_model,ffn_hidden=ffn_hidden,n_head=n_head,drop_prob=drop_prob) for _ in range(n_layers)])
        self.linear = nn.Linear(d_model,dec_voc_size)
    def forward(self,trg,enc_src,trg_mask,src_mask):
        trg = self.emb(trg)
        
        for layer in self.layers:
            trg = layer(trg, enc_src,trg_mask,src_mask)
            
        output = self.linear(trg)
        return output