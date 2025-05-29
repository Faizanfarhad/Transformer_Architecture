import os 
import sys
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self,d_model,epsilon=1e-12):
        super(LayerNorm,self).__init__()
        
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = epsilon
        
    def forward(self,x):
        mean = torch.mean(x,dim=-1,keepdim=True)
        var = torch.var(x,dim=-1,keepdim=True,unbiased=False)
        # -1 means last dimension
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out