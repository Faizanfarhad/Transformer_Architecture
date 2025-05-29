import os 
import sys
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch import tensor
from layers.scale_dot_product import ScaledDotProductAttention

class MultiheadAttention:
    def __init__(self,dmodel,num_head):
        
        self.num_head  = num_head
        # Ensure dimensions are divisible by num_heads
        assert dmodel % self.num_head == 0, "dmodel must be divisible by num_heads"
        
        
        self.attention = ScaledDotProductAttention()
        self.dv = dmodel // num_head
        self.dk = dmodel // num_head
        
        self.Wq = nn.Linear(dmodel,dmodel,bias=False)
        self.Wk = nn.Linear(dmodel,dmodel,bias=False)
        self.Wv = nn.Linear(dmodel,dmodel,bias=False)
        
        # Final linear layer after concatenation
        self.W_concat = nn.Linear(dmodel,dmodel , bias=False)
        
    def forward(self,q,k,v,mask=None,ep=1e-12):
        q,k,v = self.Wq(q),self.Wk(k),self.Wv(v)
        
        q,k,v = self.split(q),self.split(k),self.split(v)
            
        out, attention = self.attention(q,k,v,mask=mask)

        out = self.concat(out)
        out = self.W_concat(out)
        return out
        
    def split(self,tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size,length,dmodel = tensor.size()
        
        d_tensor = dmodel // self.num_head
        
        tensor = tensor.view(batch_size,length,self.num_head,d_tensor).transpose(1,2)
                # it is similar with group convolution (split by number of heads)
        return tensor

    def concat(self,tensor):
        batch_size,head,length,d_tensor = tensor.size()
        dmodel = head * d_tensor
        tensor = tensor.transpose(1,2).contiguous().view(batch_size,length,dmodel)
        return tensor
        
