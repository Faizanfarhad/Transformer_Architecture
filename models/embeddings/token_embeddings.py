import os 
import sys
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch.nn as nn

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        """
        class for token embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TokenEmbedding,self).__init__(vocab_size,d_model,padding_idx=1)

        
        
    