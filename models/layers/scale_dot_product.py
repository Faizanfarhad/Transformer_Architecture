import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


#NOTE : this is for the Encoder Block 

import torch.nn as nn
import torch
import torch.nn.functional as F
class ScaledDotProductAttention(nn.Module):
    def __init__(self,device):
        super(ScaledDotProductAttention,self).__init__()
        self.device = device
    def forward(self,q,k,v,att_mask=None,ep=1e-12):

        batch,num_head,T,d_head = k.shape
        dmodel = num_head*d_head
        
        k_t = k.transpose(-2,-1)
        
        score = torch.matmul(q,k_t).to(self.device) / (d_head**0.5)
        
        if att_mask is not None:
            mask = att_mask[: , None , None , :]
            score = score.masked_fill(mask == 0, -1e9)
       
        attention_weights = F.softmax(score,dim=-1)
       
        context = torch.matmul(attention_weights, v).to(self.device)
       
        context = context.transpose(1,2).contiguous().view(batch,T,dmodel)
        
        return context

