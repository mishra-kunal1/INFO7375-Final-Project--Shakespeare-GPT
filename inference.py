import sys
import torch
from model_gpt import TransformersModel  # Assuming model_gpt.py contains the TransformerModel class
import my_config

#load the character to index and index to character mappings
char_to_index = torch.load('char_to_index.bin')
index_to_char = torch.load('index_to_char.bin')
def encode(s):
    encoded_list = [char_to_index[char] for char in s]
    return encoded_list

# Decoding function: converting the list of integers to text
def decode(index_list):
    str_ = ''.join([index_to_char[idx] for idx in index_list])
    return str_


# Load model and checkpoint
device = 'cpu'  # Set the device to 'cpu' for inference
model = TransformersModel(my_config).to(device)
checkpoint = torch.load('trained_model.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# Set the model to evaluation mode
model.eval()

# Take input from command-line arguments
if len(sys.argv) < 2:
    print("Usage: python inference.py <input_sentence>")
    sys.exit(1)

# Encode input sentence
input_sentence = ' '.join(sys.argv[1:])
context = torch.tensor(encode(input_sentence), dtype=torch.long).unsqueeze(0).to(device)

# Generate output
num_samples = 3
for i in range(num_samples):
    generated_vector = decode(((model.generate(context, 100, temperature=0.2, top_k=20))[0]).tolist())
    print(generated_vector)
    print('*' * 100)
