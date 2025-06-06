import os 
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from conf import * 
from util.data_loader import DataLoader
from util.tokenizer import Tokenizer

tokenizer  = Tokenizer()
loader = DataLoader(
    ext=('.en','.de'),
    tokenize_en=tokenizer.tokenize_en,
    tokenize_de=tokenizer.tokenize_de,
    init_token='<sos>',
    eos_token='<eos>'   
)

train,valid,test = loader.make_dataset()
loader.buil_vocab(train_data=train, min_freq=2)
train_iter,valid_iter,test_iter = loader.make_iter(train,valid,test,
                                                batch_size=batch_size,device=device)

src_pad_idx = loader.source_vocab['<pad>']
trg_pad_idx = loader.target_vocab['<pad>']
trg_sos_idx = loader.target_vocab['<sos>']

enc_voc_size = len(loader.source_vocab)
dec_voc_size = len(loader.target_vocab)