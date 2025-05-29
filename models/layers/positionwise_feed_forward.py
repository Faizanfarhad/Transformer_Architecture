import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import layers.add_norm as an
import torch.nn as nn 
import torch 


ffn_input = an.LayerNorm

class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_ff,dmodel):
        super(PositionwiseFeedForward,self).__init__()
        
        self.ffn = nn.Sequential(
            nn.Linear(dmodel,d_ff),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(d_ff,dmodel)
        )
    def forward(self,x:torch.Tensor):
        
        return self.ffn(x)
