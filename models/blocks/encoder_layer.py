import os 
import sys
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from layers.add_norm import LayerNorm
from layers.multihead_attention import MultiheadAttention
from layers.positionwise_feed_forward import PositionwiseFeedForward
import torch.nn as nn


class EncoderLayer(nn.Module):
    def __init__(self,d_model,ffn_hidden,n_head,drop_prob,max_len,device='cpu'):
        super(EncoderLayer,self).__init__()
        self._max_len  = max_len
        self.attention = MultiheadAttention(dmodel=d_model,num_head=n_head,max_len=self._max_len,device=device)
        self.norm1 = LayerNorm(d_model=d_model,device=device)
        self.drop_out1 = nn.Dropout(p=drop_prob)
        self.drop_out1 = nn.Dropout(p=drop_prob)
        
        self.ffn = PositionwiseFeedForward(d_ff=ffn_hidden,dmodel=d_model,device=device)
        self.norm2 = LayerNorm(d_model=d_model,device=device)
        self.drop_out2 = nn.Dropout(p=drop_prob)
    
    def forward(self,x,src_mask):
        _x = x
        
        x = self.attention.forward(X=x,att_mask=src_mask)
     
        # print(f"Shape After Att : {x.shape}")
        
        x = self.drop_out1(x)
     
        # print(f"Shape After Dropout : {x.shape}")
        
        x = self.norm1(x + _x)
     
        # print(f"Shape After addnorm : {x.shape}")
        
        _x = x 
        
        x = self.ffn(x)
        # print(f"Shape After ffn : {x.shape}")
        x = self.drop_out2(x)
     
        # print(f"Shape After 2nd dropout : {x.shape}")
     
        x = self.norm2(x + _x) 
     
        # print(f"Shape After 2nd addnorm : {x.shape}")
        return x