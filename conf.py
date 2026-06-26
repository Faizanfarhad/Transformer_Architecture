import os 
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model parameter
batch_size = 32
max_len = 40
d_model = 64
n_layers = 4
n_head = 4
ffn_hiiden = 256
drop_prob = 0.4

# optimizer parameter setting
lr = 1e-4
init_lr = 1e-4
factor = 0.9
adam_eps = 5e-9
patience = 10
warmup = 100
epoch = 100
clip = 1.0
num_classes = 3
weight_decay = 5e-5
inf = float('inf')
