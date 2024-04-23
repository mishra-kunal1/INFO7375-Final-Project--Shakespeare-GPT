import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, embeddings_dim, hidden_dim, vocab_size):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embeddings_dim)
        self.lstm = nn.LSTM(embeddings_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, vocab_size)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters in LSTM model {n_params/1e3}k")
    
    def forward(self, inputs, targets=None):
        B, T = inputs.shape
        embedded = self.embeddings(inputs)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out  # No need to take just the output of the last time step
        x = F.relu(self.fc1(lstm_out))
        logits = self.fc2(x)
        
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            loss = F.cross_entropy(logits.reshape(B * T, C), targets.reshape(B * T,))
        
        return logits, loss
