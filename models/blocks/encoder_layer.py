import os 
import sys
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from layers.add_norm import LayerNorm
from layers.multihead_attention import MultiheadAttention
from layers.positionwise_feed_forward import PositionwiseFeedForward
import torch.nn as nn


class EncoderLayer(nn.Module):
    def __init__(self,d_model,ffn_hidden,n_head,drop_prob):
        super(EncoderLayer,self).__init__()
        self.attention = MultiheadAttention(dmodel=d_model,num_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.drop_out1 = nn.Dropout(p=drop_prob)
        
        self.ffn = PositionwiseFeedForward(d_ff=ffn_hidden,dmodel=d_model)
        self.norm2 = LayerNorm(d_model=d_model)
        self.drop_out2 = nn.Dropout(p=drop_prob)
    
    def forward(self,x,src_mask):
        _x = x
        x = self.attention.forward(q=x,k=x,v=x,mask=src_mask)
        
        x = self.drop_out1(x)
        x = self.norm1(x + _x)
        
        _x = x 
        x = self.ffn(x)
        
        x = self.drop_out2(x)
        x = self.norm2(x + _x) 
        return x