# transformer.py

import math
import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List
from utils import *


# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, num_positions)
        self.layers = nn.ModuleList([TransformerLayer(d_model, d_internal) for _ in range(num_layers)])
        self.output_layer = nn.Linear(d_model, num_classes)

    def forward(self, indices):
        """
        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        # Embedding and positional encoding
        x = self.embedding(indices)  # Shape: [batch_size, seq_len, d_model]
        x = self.positional_encoding(x)  # Add positional encodings to the character embeddings

        # Pass through each Transformer layer
        attn_maps = []
        for layer in self.layers:
            x, attn_map = layer(x)
            attn_maps.append(attn_map)

        # Output layer for classification (predicting 0, 1, or 2 for each position)
        logits = self.output_layer(x)
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs, attn_maps




# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super().__init__()
        self.d_model = d_model
        self.d_internal = d_internal

        # Linear transformations for queries, keys, values
        self.query_layer = nn.Linear(d_model, d_internal)
        self.key_layer = nn.Linear(d_model, d_internal)
        self.value_layer = nn.Linear(d_model, d_internal)

        # Output linear layer to project back to d_model
        self.output_layer = nn.Linear(d_internal, d_model)

        # Dropout for attention output
        self.attention_dropout = nn.Dropout(0.1)

        # Feed-forward network layers
        self.fc1 = nn.Linear(d_model, d_internal)
        self.fc2 = nn.Linear(d_internal, d_model)

        # Dropout for feed-forward output
        self.ff_dropout = nn.Dropout(0.1)

    def forward(self, input_vecs):
        # Self-attention
        queries = self.query_layer(input_vecs)  # Shape: [batch_size, seq_len, d_internal]
        keys = self.key_layer(input_vecs)       # Shape: [batch_size, seq_len, d_internal]
        values = self.value_layer(input_vecs)   # Shape: [batch_size, seq_len, d_internal]

        # Calculate attention scores and apply softmax
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.d_internal)  # Shape: [batch_size, seq_len, seq_len]
        attn_weights = F.softmax(scores, dim=-1)

        # Compute the attention output
        attn_output = torch.matmul(attn_weights, values)  # Shape: [batch_size, seq_len, d_internal]

        # Project the attention output back to d_model dimensions
        attn_output = self.output_layer(attn_output)  # Shape: [batch_size, seq_len, d_model]
        # Apply dropout to attention output before the residual connection
        attn_output = self.attention_dropout(attn_output)

        # Residual connection after self-attention
        x = input_vecs + attn_output  # Shape: [batch_size, seq_len, d_model]

        # Feed-forward network
        ff_output = F.relu(self.fc1(x))  # Shape: [batch_size, seq_len, d_internal]
        ff_output = self.fc2(ff_output)  # Shape: [batch_size, seq_len, d_model]

        # Apply dropout to feed-forward output before the residual connection
        ff_output = self.ff_dropout(ff_output)

        # Residual connection after feed-forward layer
        output = x + ff_output  # Shape: [batch_size, seq_len, d_model]

        return output, attn_weights


# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            x = x + emb_unsq
        else:
            x = x + self.emb(indices_to_embed)
        return self.dropout(x)


def train_classifier(args, train, dev):
    vocab_size = 27
    d_model = 128
    d_internal = 64
    num_classes = 3
    num_layers = 1
    num_positions = 20

    # Initialize the Transformer model with the vocabulary size retrieved
    model = Transformer(vocab_size, num_positions, d_model, d_internal, num_classes, num_layers)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fcn = nn.NLLLoss()  # Negative Log-Likelihood Loss
    batch_size=16
    num_epochs = 10
    for t in range(num_epochs):
        # Training Phase
        model.train()
        loss_this_epoch = 0.0
        random.seed(t)
        ex_idxs = [i for i in range(len(train))]
        random.shuffle(ex_idxs)

        # Iterate over batches
        for batch_start in range(0, len(train), batch_size):
            batch_examples = [train[i] for i in ex_idxs[batch_start:batch_start + batch_size]]
            
            # Prepare the batch inputs and targets
            inputs = torch.stack([ex.input_tensor for ex in batch_examples])  # Shape: [batch_size, seq_len]
            targets = torch.stack([ex.output_tensor for ex in batch_examples])  # Shape: [batch_size, seq_len]

            # Forward pass
            log_probs, _ = model(inputs)  # log_probs shape: [batch_size, seq_len, num_classes]

            # Compute loss
            # We need to reshape log_probs and targets for compatibility with NLLLoss
            loss = loss_fcn(log_probs.view(-1, num_classes), targets.view(-1))  # Flatten to [batch_size * seq_len, num_classes]

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_this_epoch += loss.item()

        avg_loss = loss_this_epoch / len(train)
        print(f"Epoch {t + 1}/{num_epochs}, Training Loss: {avg_loss:.4f}")

        # Evaluation Phase
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_start in range(0, len(dev), batch_size):
                batch_examples = dev[batch_start:batch_start + batch_size]

                # Prepare the batch inputs and targets
                inputs = torch.stack([ex.input_tensor for ex in batch_examples])  # Shape: [batch_size, seq_len]
                targets = torch.stack([ex.output_tensor for ex in batch_examples])  # Shape: [batch_size, seq_len]

                # Forward pass
                log_probs, _ = model(inputs)  # log_probs shape: [batch_size, seq_len, num_classes]
                predictions = torch.argmax(log_probs, dim=-1)  # Get the predicted class for each position

                # Compare predictions to targets
                correct += (predictions == targets).sum().item()
                total += targets.numel()

        accuracy = correct / total
        print(f"Epoch {t + 1}/{num_epochs}, Dev Accuracy: {accuracy:.4f}")

    return model




####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
