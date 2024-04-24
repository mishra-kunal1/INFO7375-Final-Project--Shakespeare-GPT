import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from config import my_config
from model_embed import EmbeddingsModel
from model_lstm import LSTMModel
from model_gpt import TransformersModel

# Load data
train_data = torch.load('../data/train.bin')
val_data = torch.load('../data/val.bin')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Get batch function
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data) - my_config.block_size, (my_config.batch_size,))
    x = torch.stack([data[i:i + my_config.block_size] for i in ix])
    y = torch.stack([data[i + 1:i + my_config.block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# Estimate loss function
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(my_config.eval_iters)
        for k in range(my_config.eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split + '_loss'] = losses.mean()
    model.train()
    return out

# Initialize model, optimizer, and other variables
model_cls = input("Enter the model class to train (EmbeddingsModel (E), LSTMModel(L), TransformersModel(T): ")
if model_cls == "E" or model_cls=='e':
    model = EmbeddingsModel(embeddings_dim=my_config.embed_size, vocab_size=my_config.vocab_size).to(device)

elif model_cls.lower() == "l":
    model = LSTMModel(embeddings_dim=my_config.embed_size, hidden_dim=my_config.hidden_dim, vocab_size=my_config.vocab_size).to(device)

elif model_cls == "T" or model_cls=='t':
    model = TransformersModel(my_config).to(device)
else:
    raise ValueError("Invalid model class!")

optimizer = torch.optim.AdamW(model.parameters(), lr=my_config.lr)
out_model_folder = 'saved_models'
if not os.path.exists(out_model_folder):
    os.makedirs(out_model_folder)

# Training loop
loss_dict_model = {'train': [], 'val': [], 'iter': []}
best_val_loss = 1e9
for iter in range(my_config.max_iters):
    if iter % my_config.eval_interval == 0:
        losses = estimate_loss()
        print(f'Iter {iter:5d} Train loss {losses["train_loss"]:.4f} Val loss {losses["val_loss"]:.4f}')
        loss_dict_model['train'].append(losses['train_loss'])
        loss_dict_model['val'].append(losses['val_loss'])
        loss_dict_model['iter'].append(iter)
        if losses['val_loss'] < best_val_loss:
            best_val_loss = losses['val_loss']
            if iter > 0:
                torch.save({
                    'epoch': iter,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': losses['val_loss'],
                }, f'{out_model_folder}/model.pt')
                print(f'Saving model at iter {iter} with val loss {losses["val_loss"]:.4f}')
    x, y = get_batch('train')
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
