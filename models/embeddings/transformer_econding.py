import os 
import sys
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn

from  embeddings.positional_encoding import PositionalEncoding
from embeddings.token_embeddings import TokenEmbedding

class TransformerEmbedding(nn.Module):
    def __init__(self,vocab_size,d_model,max_len,drop_prob,device):
        super(TransformerEmbedding,self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size=vocab_size,d_model=d_model)
        self.pos_emb =PositionalEncoding(d_model=d_model,max_len=max_len,device=device)
        self.droup_out = nn.Dropout(p=drop_prob)
    def forward(self,x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        emb = tok_emb + pos_emb
        return self.droup_out(emb)
