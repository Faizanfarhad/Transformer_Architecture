import os 
import sys
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch 
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len,device):
        super(PositionalEncoding,self).__init__()

        self.encoding = torch.zeros(max_len,d_model,device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient
        
        pos = torch.arange(0,max_len,device=device)
        
        pos = pos.float().unsqueeze(dim=-1)
        
        _2i = torch.arange(0,d_model,step=2,device=device).float()
        
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)
        
        self.encoding[:, 0::2] = torch.sin( pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos( pos / (10000 ** (_2i / d_model)))

        # compute positional encoding to consider positional information of words
    
    def forward(self,x):
        # self.encoding
        # [max_len = 512, d_model = 512]
        
        batch_size ,seq_len = x.size()
        # [batch_size = 128, seq_len = 30]
        
        return self.encoding[:seq_len, :]
        #[seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]