import torch
import torch.nn as nn
import torch.optim as optim
from model_gpt import TransformerModel
from config import my_config
from data_prep import encode,decode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    checkpoint=torch.load('../data/trained_model.pt',map_location=device)
    model=TransformerModel(my_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Initialize the optimizer
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f'Model loaded from epoch {epoch} with val loss {loss:.4f}')
    return model

model=load_model()

def generate_text(text):
    context=torch.tensor(encode(text),dtype=torch.long).unsqueeze(0).to(device)
    generated_vector=decode(((model.generate(context,500))[0]).tolist())
    return generated_vector


if __name__ == '__main__':
    model = load_model()

