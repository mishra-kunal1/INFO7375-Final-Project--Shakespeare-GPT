import torch
import torch.nn as nn
import torch.nn.functional as F
from config import my_config

class ScaledDotProductAttention(nn.Module):
    def __init__(self, my_config):
        super().__init__()

        self.embed_size = my_config.embed_size
        self.head_size = my_config.head_size
        self.block_size = my_config.block_size
        self.dropout_rate = my_config.dropout

        self.key_layer = nn.Linear(self.embed_size, self.head_size, bias=False)
        self.query_layer = nn.Linear(self.embed_size, self.head_size, bias=False)
        self.value_layer = nn.Linear(self.embed_size, self.head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(self.block_size, self.block_size)))
        self.dropout = nn.Dropout(self.dropout_rate)
    
    def forward(self, x):
        B, T, C = x.shape
        keys = self.key_layer(x)
        queries = self.query_layer(x)
        values = self.value_layer(x)
        affinity = queries @ keys.transpose(-2, -1) * (self.head_size ** -0.5)
        affinity = affinity.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        affinity = F.softmax(affinity, dim=-1)
        affinity = self.dropout(affinity)
        attention_scores = affinity @ values
        return attention_scores



class MultiHead(nn.Module):
    def __init__(self, my_config):
        super().__init__()
        self.num_heads = my_config.num_heads
        self.head_size = my_config.head_size
        self.embed_size = my_config.embed_size
        self.heads = nn.ModuleList([ScaledDotProductAttention(my_config) for _ in range(self.num_heads)])
        self.proj = nn.Linear(self.embed_size , self.embed_size )  # num_heads*head_size=embed_size
        self.dropout = nn.Dropout(my_config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FFN(nn.Module):
    def __init__(self, my_config):
        super().__init__()

        self.embed_size = my_config.embed_size
        self.dropout = my_config.dropout

        self.net = nn.Sequential(
            nn.Linear(self.embed_size, 4 * self.embed_size),
            nn.ReLU(),
            nn.Linear(4 * self.embed_size, self.embed_size),
            nn.Dropout(self.dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, my_config):
        super().__init__()

        self.embed_size = my_config.embed_size
        self.attention_layer = MultiHead(my_config)
        self.feed_forward = FFN(my_config)
        self.ln1 = nn.LayerNorm(self.embed_size)
        self.ln2 = nn.LayerNorm(self.embed_size)

    def forward(self, x):
        x = x + self.attention_layer(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x



class TransformerModel(nn.Module):
    def __init__(self, my_config):
        super().__init__()

        self.vocab_size = my_config.vocab_size
        self.embed_size = my_config.embed_size
        self.block_size = my_config.block_size
        self.num_heads = my_config.num_heads
        self.n_blocks = my_config.n_blocks
        self.head_size = my_config.head_size

        self.embeddings = nn.Embedding(self.vocab_size, self.embed_size)
        self.positional_embeddings = nn.Embedding(self.block_size, self.embed_size)
        self.blocks = nn.Sequential(*[Block(my_config) for _ in range(self.n_blocks)])
        self.ln_final = nn.LayerNorm(self.embed_size)
        self.lm_head = nn.Linear(self.embed_size, self.vocab_size)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters {n_params/1e6} million")
    

    #Akshita's part
    def __call__(self, inputs, targets=None):
        B, T = inputs.shape
        tok_emb = self.embeddings(inputs)  # B,T,C
        positions = torch.arange(T).to(inputs.device)
        pos_emb = self.positional_embeddings(positions)  # T,C
        x = tok_emb + pos_emb   # B,T,C
        x = self.blocks(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None 
        else:   
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.reshape(B * T, C), targets.reshape(B * T,))
        return logits, loss
    
    ## Shivani's part 
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= my_config.block_size else idx[:, -my_config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx