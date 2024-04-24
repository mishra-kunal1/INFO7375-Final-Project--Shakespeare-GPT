import torch

# Reading the input file
with open('../data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Getting the unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Creating a dictionary to map characters to integers and vice versa
char_to_index = {ch: idx for idx, ch in enumerate(chars)}
index_to_char = {idx: ch for idx, ch in enumerate(chars)}

#save char_to_index and index_to_char
torch.save(char_to_index, '../data/char_to_index.bin')
torch.save(index_to_char, '../data/index_to_char.bin')

# Encoding function: converting the text to a list of integers
def encode(s):
    encoded_list = [char_to_index[char] for char in s]
    return encoded_list

# Decoding function: converting the list of integers to text
def decode(index_list):
    str_ = ''.join([index_to_char[idx] for idx in index_list])
    return str_

# Splitting the text into training and validation data
data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]

# Saving the training and validation data
torch.save(train_data, '../data/train.bin')
torch.save(val_data, '../data/val.bin')

print("Data saved successfully.")
