# config.py

# Hyperparameters
# config.py

import torch

my_config = {
    'vocab_size': 65,
    'batch_size': 64,
    'block_size': 256,
    'max_iters': 6500,
    'eval_interval': 500,
    'lr': 3e-4,
    'eval_iters': 200,
    'embed_size': 384,
    'num_heads': 6,
    'head_size': 384 // 6,
    'n_blocks': 6,
    'dropout': 0.2
}

class Config:
    def __init__(self, config_dict):
        self.__dict__ = config_dict
my_config = Config(my_config)
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

