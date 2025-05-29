import sys
import os

project_root = sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
if project_root not in sys.path:
    sys.path.insert(0,project_root)

import torch.nn as nn
import torch
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention,self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self,q,k,v,mask=None,ep=1e-12):
        
        batch_size,head,seq_len,d_k = k.size()
        
        k_t = k.transpose(2,3)
        
        score = torch.matmul(q,k_t) / math.sqrt(d_k)
        
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)
        
        score = self.softmax(score)
        
        V = torch.matmul(score,v)
        
        return V,score

