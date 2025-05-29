import os
import sys
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch.nn as nn
import layers.multihead_attention as mh
import layers.add_norm as an
from layers.positionwise_feed_forward import PositionwiseFeedForward
from layers.multihead_attention import MultiheadAttention 
from layers.add_norm import LayerNorm


class DecoderLayer(nn.Module):
    
    def __init__(self,dmodel,ffn_hidden,n_head,drop_prob):
        super(DecoderLayer,self).__init__()
        self.self_attention = MultiheadAttention(dmodel=dmodel,num_head=n_head)
        self.norm1 = LayerNorm(dmodel)
        self.dropout1 = nn.Dropout(p=drop_prob)
        
        self.enc_dec_attention = MultiheadAttention(dmodel=dmodel,num_head=n_head)
        self.norm2 = LayerNorm(dmodel)
        self.dropout2 = nn.Dropout(p=drop_prob)
        
        self.ffn = PositionwiseFeedForward(d_ff=ffn_hidden,dmodel=dmodel)
        self.norm3 = LayerNorm(dmodel)
        self.dropout3 = nn.Dropout(p=drop_prob)
        
    def forward(self,dec,enc,trg_mask,src_mask):
        # 1. compute self attention
        
        _x = dec # is the sublayer
        x = self.self_attention.forward(q=dec,k=dec,v=dec,mask=trg_mask) 
        
        x = self.dropout1(x)
        x = self.norm1(x + _x) # FFn(X + sublayer(x))
        
        if enc is not None:
            _x = x
            x = self.enc_dec_attention.forward(q=dec,k=enc,v=enc,mask=src_mask)
            
            #Add and Norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)
            
        # Positional wise Feed Forward Network
        
        _x = x
        x = self.ffn(x)
        
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        
        return x