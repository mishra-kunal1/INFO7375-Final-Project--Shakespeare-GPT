import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingsModel(nn.Module):
    def __init__(self,embeddings_dim,vocab_size):
        super().__init__()
        self.embeddings=nn.Embedding(vocab_size,embeddings_dim)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters in Embeddings Model {n_params/1e3}k")
    
    def __call__(self,inputs,targets=None):
        B,T=inputs.shape
        logits=self.embeddings(inputs)
        if targets is None:
            loss=None 
        else:   
            B,T,C=logits.shape
            loss=F.cross_entropy(logits.reshape(B*T,C),targets.reshape(B*T,))
        return logits,loss