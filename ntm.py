"""
2014 Neural Turing Machines,
     Alex Graves, gravesa@google.com
     Greg Wayne, gregwayne@google.com
     Ivo Danihelka, danihelka@google.com

Hudson Andrew Smelski

Basic diagram of NTM structure

    Input M          Output M
         \          /
          Controller
         //         \
Read Heads           Write Heads
        /|\         /|\
              N
_______________________________
|           Memory            | M
-------------------------------

x:input                 y
r:read  -  Controller - w:write  ___
                        r:read   |M|
                                 ---
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

from tokenizer import *


class NTM(nn.Module):
    def __init__(self, vocab_size, memory_length, controller_depth=1, read_heads=1, write_heads=1, use_lstm=False):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_lstm = use_lstm

        self.RH = read_heads
        self.WH = write_heads
        self.N = memory_length
        self.M = vocab_size

        # Read/write parameter sizes
        self.read_lengths = [self.M, 1, 1, 3, 1]
        self.read_length = sum(self.read_lengths)

        self.write_lengths = [self.M, 1, 1, 3, 1, self.M, self.M]
        self.write_length = sum(self.write_lengths)

        # Controller
        self.controller_input = self.M + self.M * self.RH
        self.controller_output = self.read_length * self.RH + self.write_length * self.WH + self.M
        self.controller_depth = controller_depth

        if self.use_lstm:
            self.controller = nn.LSTM(
                input_size=self.controller_input,
                hidden_size=self.controller_output,
                num_layers=self.controller_depth,
                batch_first=False
            )
        else:
            self.controller = nn.RNN(
                input_size=self.controller_input,
                hidden_size=self.controller_output,
                num_layers=self.controller_depth,
                batch_first=False
            )

        for name, param in self.controller.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param, gain=2.0)  # Increase gain

        # Initialize memory buffer
        self.register_buffer('memory_initial', torch.zeros(self.N, self.M))
        #nn.init.xavier_uniform_(self.memory_initial)

        self.logit_scale = nn.Parameter(torch.ones(1) * 5.0)

        # Batch state (will be set by reset)
        self.batch_size = None
        self.memory = None
        self.read_w = None
        self.write_w = None
        self.last_read = None
        self.hidden = None
        self.cell = None

        self.to(self.device)

    def reset(self, batch_size=1):
        """Reset NTM state for a batch of sequences"""
        self.batch_size = batch_size
        device = self.device

        # Memory: [batch, N, M]
        self.memory = self.memory_initial.unsqueeze(0).repeat(batch_size, 1, 1).clone()

        # Weights: [batch, N]
        init_w = torch.zeros(batch_size, self.N, device=device)
        init_w[:, 0] = 1.0  # Start focused on first memory location
        self.read_w = init_w.clone()
        self.write_w = init_w.clone()

        # Last read: [batch, M]
        self.last_read = torch.randn(batch_size, self.M, device=device) * 0.01

        # controller states: [num_layers, batch, hidden_size]
        self.hidden = torch.zeros(self.controller_depth, batch_size, self.controller_output, device=device)

        if self.use_lstm:
            self.cell = torch.zeros(self.controller_depth, batch_size, self.controller_output, device=device)

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def addressing(self, key, beta, gate, shift, gamma, prev_w, memory):
        """
        Batched NTM addressing
        key: [batch, M]
        beta: [batch, 1]
        gate: [batch, 1]
        shift: [batch, 3]
        gamma: [batch, 1]
        prev_w: [batch, N]
        memory: [batch, N, M]
        Returns: [batch, N]
        """
        batch = memory.size(0)
        eps = 1e-6

        # Content addressing
        beta = F.softplus(beta).squeeze(-1)  # [batch]
        # Cosine similarity: [batch, N]
        key_norm = key / (key.norm(dim=-1, keepdim=True) + eps)
        mem_norm = memory / (memory.norm(dim=-1, keepdim=True) + eps)
        cos_sim = torch.bmm(mem_norm, key_norm.unsqueeze(-1)).squeeze(-1)  # [batch, N]
        wt = F.softmax(beta.unsqueeze(-1) * cos_sim, dim=-1)

        # Interpolation
        gate = torch.sigmoid(gate).squeeze(-1)  # [batch]
        wt = gate.unsqueeze(-1) * wt + (1 - gate).unsqueeze(-1) * prev_w

        # Convolutional shift (batched circular convolution)
        shift = F.softmax(shift, dim=-1)  # [batch, 3]
        shifted = torch.zeros_like(wt)
        for k in range(3):
            shift_amount = k - 1  # -1, 0, 1
            rolled = torch.roll(wt, shifts=shift_amount, dims=1)
            shifted = shifted + shift[:, k].unsqueeze(-1) * rolled

        # Sharpening
        gamma = 1 + F.softplus(gamma).squeeze(-1)  # [batch]
        wt = (shifted + eps) ** gamma.unsqueeze(-1)
        wt = wt / (wt.sum(dim=-1, keepdim=True) + eps)

        return wt

    def forward(self, x):
        """
        x: [batch, M] input vectors (one-hot or embeddings)
        Returns: [batch, M] output logits
        """
        batch = x.size(0)

        # Initialize if needed
        if self.batch_size != batch:
            self.reset(batch_size=batch)

        # Concatenate input with last read: [batch, controller_input]
        controller_input = torch.cat([x, self.last_read], dim=-1)
        controller_input = controller_input.unsqueeze(0)  # [1, batch, controller_input]

        # Controller forward
        if self.use_lstm:
            output, (self.hidden, self.cell) = self.controller(controller_input, (self.hidden, self.cell))
        else:
            output, self.hidden = self.controller(controller_input, self.hidden)
        controller_out = output.squeeze(0)  # [batch, controller_output]

        # Split controller output
        rh_params = controller_out[:, :self.read_length * self.RH]
        wh_params = controller_out[:, self.read_length * self.RH:self.read_length * self.RH + self.write_length * self.WH]
        y = controller_out[:, self.read_length * self.RH + self.write_length * self.WH:]

        # Process read heads
        for h in range(self.RH):
            params = rh_params[:, h * self.read_length:(h + 1) * self.read_length]

            idx = 0
            key = params[:, idx:idx + self.M]; idx += self.M
            beta = params[:, idx:idx + 1]; idx += 1
            gate = params[:, idx:idx + 1]; idx += 1
            shift = params[:, idx:idx + 3]; idx += 3
            gamma = params[:, idx:idx + 1]; idx += 1

            wt = self.addressing(key, beta, gate, shift, gamma, self.read_w, self.memory)
            self.read_w = wt

            # Read: [batch, N] @ [batch, N, M] -> [batch, M]
            self.last_read = torch.bmm(wt.unsqueeze(1), self.memory).squeeze(1)

        # Process write heads
        for h in range(self.WH):
            params = wh_params[:, h * self.write_length:(h + 1) * self.write_length]

            idx = 0
            key = params[:, idx:idx + self.M]; idx += self.M
            beta = params[:, idx:idx + 1]; idx += 1
            gate = params[:, idx:idx + 1]; idx += 1
            shift = params[:, idx:idx + 3]; idx += 3
            gamma = params[:, idx:idx + 1]; idx += 1
            erase = params[:, idx:idx + self.M]; idx += self.M
            add = params[:, idx:idx + self.M]; idx += self.M

            wt = self.addressing(key, beta, gate, shift, gamma, self.write_w, self.memory)
            self.write_w = wt

            # Erase and add (batched outer products)
            erase_v = torch.sigmoid(erase)  # [batch, M]
            add_v = add  # [batch, M]

            # [batch, N, M] operations
            erase_matrix = wt.unsqueeze(-1) * erase_v.unsqueeze(1)  # [batch, N, M]
            self.memory = self.memory * (1 - erase_matrix)

            add_matrix = wt.unsqueeze(-1) * add_v.unsqueeze(1)  # [batch, N, M]
            self.memory = self.memory + add_matrix

        y = y * self.logit_scale
        return y


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"vocab size = {vocab_size}")

    memory_length = 128
    ntm = NTM(vocab_size, memory_length, controller_depth=1, read_heads=1, write_heads=1)

    print(f"parameters      = {ntm.num_params():,}")
    print(f"memory shape    = {ntm.memory_initial.shape}")

    # Test with batch
    batch_size = 4
    ntm.reset(batch_size=batch_size)
    x = torch.randn(batch_size, vocab_size, device=device)
    y = ntm.forward(x)
    print(f"\nBatch output shape: {y.shape}")  # Should be [4, vocab_size]
