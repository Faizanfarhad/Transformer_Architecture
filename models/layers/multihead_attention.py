import os 
import sys
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch import tensor
from layers.scale_dot_product import ScaledDotProductAttention

class MultiheadAttention(nn.Module):
    def __init__(self,dmodel,num_head,max_len,device='cpu'):
        super(MultiheadAttention,self).__init__()
        self.device = device
        self.num_head  = num_head
        # Ensure dimensions are divisible by num_heads
        assert dmodel % self.num_head == 0, "dmodel must be divisible by num_heads"
        
        self.attention = ScaledDotProductAttention(device=device)
        self.d_head = dmodel // num_head
        self.seq_len = max_len
        # WQ,WK,WQ shape should be : R^ dmodel x d_K

        self.Wq = nn.Linear(dmodel,dmodel,device=device)
        self.Wk = nn.Linear(dmodel,dmodel,device=device)
        self.Wv = nn.Linear(dmodel,dmodel,device=device)
        self.W_O =nn.Linear(dmodel,dmodel,device=device)
        
        # Final linear layer after concatenation
        
    def forward(self,X,att_mask=None):
        # q , k,v shape should be : R^ n x dk
        # print(f"Input shape : {X.shape}")
       
        B,T,d_model = X.shape #shape B,T,dk 
        Q = self.Wq(X) # shape out : B,T,dk
        K = self.Wk(X) # shape out : B,T,dk
        V = self.Wv(X) # shape out : B,T,dk
       
        # print(f"Q shape : {Q.shape}")
        # print(f"K shape : {K.shape} ")
        # print(f"V shape : {V.shape}: ")
       
        Q = Q.view(B,T,self.num_head, self.d_head).transpose(1,2) #after transpose shape : B,h,T,T 
        K = K.view(B,T,self.num_head, self.d_head).transpose(1,2) # ""
        V = V.view(B,T,self.num_head, self.d_head).transpose(1,2) # ""
        
        
        context = self.attention(Q,K,V,att_mask=att_mask)
        
        # print(f"After Scale Dot Q shape : {Q.shape}")
        # print(f"After Scale Dot K shape : {K.shape} ")
        # print(f"After Scale Dot V shape : {V.shape}: ")
        # print(f"After Scale Dot Context shape : {context.shape}: ")
        
        assert context.shape == (B,T,d_model), f"shape should be B,T,dmodel not {context.shape}"
        out = self.W_O(context).to(self.device)

        # print(f"Output shape {out.shape}")
        return out
        

'''
flow : 
input X E R^ n * d
        |
        Q*Wq,K*Wk,V*Wv 
        |
        used ScaledDotProduct(Layer) for computing the attention scores then context and concat the context of all the heads in to one 
        |
        and the last layer  of computing W_O for introducing randomness

'''