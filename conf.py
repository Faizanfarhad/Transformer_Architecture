import os 
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model parameter
batch_size = 128 * 4
max_len = 256
d_model = 512
n_layers = 6
n_head = 8
ffn_hiiden = 2048
drop_prob = 0.1

# optimizer parameter setting

init_lr = 1e-5
factor = 0.9
adam_eps = 5e-9
patience = 10
warmup = 100
epoch = 1000
clip = 1.0
weight_decay = 5e-4
inf = float('inf')
