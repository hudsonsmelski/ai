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

Experiment:
    input = sequence of [x_t, RH1, RH2]

TODO: How to make a model with the ability to store and read vectors (in an embedding space)
 and also view into short and long term memory, as well as have access to a mutable state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import string
from typing import Dict, Any

class NTM(nn.Module):
    def __init__(self, input_size, memory_length,
        controller_depth=1, controller_width=100,
        read_heads=1, write_heads=1, device='cpu'):
        super().__init__()

        self.device = device

        self.RH = read_heads
        self.WH = write_heads
        self.N = memory_length
        self.M = input_size

        #self.embedding = nn.Sequential(nn.Linear(self.M, self.M), nn.LayerNorm(self.M), nn.CELU())
        #self.embedding = nn.Sequential(nn.Linear(self.M, self.M), nn.CELU())
        #self.embedding = nn.Sequential(nn.Linear(input_size, self.M))
        self.embedding = nn.Identity()

        # Read/write parameter sizes
        self.read_lengths = [self.M, 1, 1, 3, 1]
        self.read_length = sum(self.read_lengths)

        self.write_lengths = [self.M, 1, 1, 3, 1, self.M, self.M]
        self.write_length = sum(self.write_lengths)

        self.controller_width = controller_width
        self.controller_depth = controller_depth
        self.controller_input = self.M# + self.M * self.RH
        #self.controller_output_size = (self.read_length * self.RH) + (self.write_length * self.WH) + self.M

        self.controller = nn.GRU(self.controller_input, self.controller_width, self.controller_depth, batch_first=False)
        self.read_head = nn.Sequential(nn.Linear(self.controller_width, self.read_length*self.RH))
        self.write_head = nn.Sequential(nn.Linear(self.controller_width, self.write_length * self.WH))
        self.read_matrix = nn.Linear(self.RH*self.M, self.M, bias=False)
        self.out_head = nn.Linear(self.controller_width, self.M)

        # Initialize memory buffer (not model parameters)
        self.register_buffer('memory_initial', torch.zeros(self.N, self.M))
        self.to(self.device)

    def reset(self, batch_size=1):
        """Reset NTM state for a batch of sequences"""
        self.batch_size = batch_size
        self.memory = self.memory_initial.unsqueeze(0).repeat(batch_size, 1, 1).clone()

        # Weights: [batch, num_heads, N]
        init_w = torch.zeros(batch_size, self.N, device=self.device)
        init_w[:, 0] = 1.0  # Start focused on first memory location
        #init_w = torch.full((batch_size, self.N), 1.0 / self.N, device=self.device)  # Uniform soft attention
        self.read_w = init_w.unsqueeze(1).repeat(1, self.RH, 1)  # [batch, RH, N]
        self.write_w = init_w.unsqueeze(1).repeat(1, self.WH, 1)  # [batch, WH, N]

        self.hidden = torch.zeros(self.controller_depth, batch_size, self.controller_width, device=self.device)

    def forward(self, x):
        xe = self.embedding(x)
        o, self.hidden = self.controller(xe.unsqueeze(0), self.hidden)
        o = o[-1,:,:]

        #Execute Read
        rh_params = self.read_head(o)
        new_read_weights = []
        for h in range(self.RH):
            params = rh_params[:, h * self.read_length:(h + 1) * self.read_length]

            idx = 0
            key = params[:, idx:idx + self.M]; idx += self.M
            beta = params[:, idx:idx + 1]; idx += 1
            gate = params[:, idx:idx + 1]; idx += 1
            shift = params[:, idx:idx + 3]; idx += 3
            gamma = params[:, idx:idx + 1]; idx += 1

            prev_w = self.read_w[:, h, :]  # [batch, N]
            wt = self.addressing(key, beta, gate, shift, gamma, prev_w, self.memory)
            new_read_weights.append(wt)
        self.read_w = torch.stack(new_read_weights, dim=1) # [batch, RH, N]

        read_vecs = []
        for h in range(self.RH):
            w = self.read_w[:, h, :].unsqueeze(1) # [batch, 1, N]
            read_vector = torch.bmm(w, self.memory).squeeze(1) # [batch, M]
            read_vecs.append(read_vector)
        read_vecs = torch.cat(read_vecs, dim = 1)
        y = self.out_head(o) + self.read_matrix(read_vecs)

        wh_params = self.write_head(o)
        new_memory = self.memory
        new_write_weights = []
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

            prev_w = self.write_w[:, h, :]  # [batch, N]
            wt = self.addressing(key, beta, gate, shift, gamma, prev_w, self.memory)
            new_write_weights.append(wt)

            # Erase and add (batched outer products)
            erase_v = torch.sigmoid(erase)  # [batch, M]
            add_v = torch.tanh(add) # add  # [batch, M]
            #add_v = add

            # [batch, N, M] operations - create new tensors instead of modifying in place
            erase_matrix = wt.unsqueeze(-1) * erase_v.unsqueeze(1)  # [batch, N, M]
            new_memory = new_memory * (1 - erase_matrix)

            add_matrix = wt.unsqueeze(-1) * add_v.unsqueeze(1)  # [batch, N, M]
            new_memory = new_memory + add_matrix

        self.memory = new_memory
        self.write_w = torch.stack(new_write_weights, dim=1)  # [batch, WH, N]

        return y

    def addressing(self, key, beta, gate, shift, gamma, prev_w, memory):
        """NTM addressing mechanism"""
        batch = memory.size(0)
        eps = 1e-12

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

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_memory_usage_stats(self) -> Dict[str, Any]:
        """Diagnostic statistics"""
        stats = {}

        stats['memory_mean'] = self.memory.mean().item()
        stats['memory_std'] = self.memory.std().item()
        stats['memory_abs_max'] = self.memory.abs().max().item()
        stats['memory_sparsity'] = (self.memory.abs() < 0.01).float().mean().item()

        eps = 1e-12
        read_entropy = -(self.read_w * (self.read_w + eps).log()).sum(-1).mean().item()
        write_entropy = -(self.write_w * (self.write_w + eps).log()).sum(-1).mean().item()

        stats['read_entropy'] = read_entropy
        stats['write_entropy'] = write_entropy
        stats['read_sharpness'] = math.log(self.N) - read_entropy
        stats['write_sharpness'] = math.log(self.N) - write_entropy

        stats['hx_mean'] = self.hidden.mean().item()

        return stats

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Define the set of valid ASCII characters to use as tokens
    vocab = string.digits + string.ascii_letters + string.punctuation + " \t\v\n\r\f"
    vocab_size = len(vocab)

    char_to_idx = {char: idx for idx, char in enumerate(vocab)}    #Create a dictionary that maps each character to a unique integer value
    idx_to_char = {idx: char for char, idx in char_to_idx.items()} #Create a reverse dictionary that maps each integer value to its corresponding character

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
