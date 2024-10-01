# models.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Indexer

class LanguageModel(object):
    def get_next_char_log_probs(self, context) -> np.ndarray:
        raise Exception("Only implemented in subclasses")

    def get_log_prob_sequence(self, next_chars, context) -> float:
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0 / self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0 / self.voc_size) * len(next_chars)

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
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)
class NeuralLanguageModel(LanguageModel):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_len=20, vocab_index=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.vocab_index = vocab_index  # Store vocab_index as a class attribute
        self.model = TransformerLanguageModel(vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_len)

    def get_next_char_log_probs(self, context):
        """
        Returns log probabilities over the next characters given a context.
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        # Handle empty context by adding start-of-sequence token (space)
        if len(context) == 0:
            context = ' '

        self.model.eval()
        with torch.no_grad():
            # Convert context to tensor
            context_indices = [self.vocab_index.index_of(char) for char in context]
            context_tensor = torch.tensor(context_indices, dtype=torch.long).unsqueeze(1)  # Shape: [len(context), 1]

            # Forward pass
            log_probs = self.model(context_tensor)
            
            # Check if log_probs is empty (which should not happen after this fix)
            if log_probs.size(0) == 0:
                raise ValueError("log_probs is empty. There might be an issue with input tensor dimensions or model output.")

            return log_probs[-1, 0].numpy()  # Return log_probs of the last time step

    def get_log_prob_sequence(self, next_chars, context):
        """
        Scores a sequence of characters following a given context.
        :param next_chars: the characters to score following the context
        :param context: the context preceding the characters
        :return: The float log probability of the sequence
        """
        self.model.eval()
        with torch.no_grad():
            total_log_prob = 0.0
            for char in next_chars:
                log_probs = self.get_next_char_log_probs(context)
                char_idx = self.vocab_index.index_of(char)
                total_log_prob += log_probs[char_idx]
                context += char
            return total_log_prob



class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_len=20):
        super(TransformerLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # Embed input and add positional encoding
        x = self.embedding(x)  # Shape: [seq_len, batch_size, d_model]
        x = self.positional_encoding(x)

        # Create a causal mask to prevent future tokens from being attended to
        seq_len = x.size(0)
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)

        # Pass through Transformer Encoder
        x = self.transformer_encoder(x, mask=mask)

        # Predict next character (output shape: [seq_len, batch_size, vocab_size])
        output = self.output_layer(x)
        log_probs = F.log_softmax(output, dim=-1)

        return log_probs


def train_lm(args, train_text, dev_text, vocab_index):
    vocab_size = len(vocab_index)
    d_model = 64  # Adjust for fast training
    nhead = 2
    num_layers = 2
    dim_feedforward = 128
    batch_size = 16
    num_epochs = 10

    # Instantiate the model
    model = TransformerLanguageModel(vocab_size, d_model, nhead, num_layers, dim_feedforward)
    model.train()

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    loss_fcn = nn.NLLLoss()

    # Convert training text to indices
    train_indices = [vocab_index.index_of(char) for char in train_text]

    # Train in chunks
    chunk_size = 20
    for epoch in range(num_epochs):
        total_loss = 0.0

        for i in range(0, len(train_indices) - chunk_size, chunk_size):
            context = train_indices[i:i + chunk_size]
            target = train_indices[i + 1:i + chunk_size + 1]

            # Convert to PyTorch tensors
            context_tensor = torch.tensor(context, dtype=torch.long).unsqueeze(1)  # Shape: [chunk_size, 1]
            target_tensor = torch.tensor(target, dtype=torch.long).squeeze()

            # Forward pass
            log_probs = model(context_tensor).squeeze(1)

            # Compute loss
            loss = loss_fcn(log_probs.view(-1, vocab_size), target_tensor.view(-1))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (len(train_indices) // chunk_size)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Wrap the trained model into NeuralLanguageModel
    lm = NeuralLanguageModel(vocab_size, d_model, nhead, num_layers, dim_feedforward, vocab_index=vocab_index)
    lm.model = model
    return lm

