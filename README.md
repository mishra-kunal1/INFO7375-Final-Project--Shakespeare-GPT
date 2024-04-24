# Shakespeare GPT Model

## Overview
This repository contains the code and models for training a Shakespeare text generation model using character-level embeddings. The model is trained on the works of William Shakespeare and is capable of generating text in the style of Shakespeare's plays and poems.

## Models
Three different architectures were experimented with for text generation:

1. **Embedding Model**: A basic neural network model where characters are represented as embeddings and passed through one dense layer.
2. **LSTM (Long Short-Term Memory)**: A recurrent neural network (RNN) architecture designed to learn long-term dependencies in sequential data. The model consisits of an embedding layer, LSTM cells and two dense layers/
3. **GPT-2 (Generative Pre-trained Transformer 2)**: A state-of-the-art language model based on the Transformer architecture, pre-trained on a large corpus of text data. Fine-tuning was performed on Shakespeare's works to adapt the model for text generation.

## Dataset
The dataset used for training consists of a collection of Shakespeare's plays, sonnets, and poems. Each text is preprocessed into character-level sequences, which are used to train the models.

## Training
The models were trained using Pytorch with the following configurations:
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
