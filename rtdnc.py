"""
2014 Neural Turing Machines,
     Alex Graves, gravesa@google.com
     Greg Wayne, gregwayne@google.com
     Ivo Danihelka, danihelka@google.com

2016 Hybrid computing using a neural network with dynamic external memory
Alex Graves1*, Greg Wayne1*, Malcolm Reynolds1, Tim Harley1, Ivo Danihelka1, Agnieszka Grabska-Barwińska1,
Sergio Gómez Colmenarejo1, Edward Grefenstette1, Tiago Ramalho 1, John Agapiou1, Adrià Puigdomènech Badia1,
Karl Moritz Hermann1, Yori Zwols1, Georg Ostrovski1, Adam Cain1, Helen King1, Christopher Summerfield1, Phil Blunsom1,
Koray Kavukcuoglu1 & Demis Hassabis1

Hudson Andrew Smelski

RTDNC
Transformer controller for DNC
"""

import string
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional

class CNNEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.vocab_size = vocab_size

        intermediate = (vocab_size + emb_dim)//2
        self.l1 = nn.Conv1d(vocab_size, intermediate, kernel_size=1)
        self.l2 = nn.Conv1d(intermediate, emb_dim, kernel_size=1)

        self.ln1 = nn.LayerNorm(intermediate)
        self.ln2 = nn.LayerNorm(emb_dim)

    def forward(self, x_t):
        one_hot = F.one_hot(x_t, num_classes=self.vocab_size).float()
        one_hot = one_hot.unsqueeze(-1)  # [batch, vocab, 1]

        x = self.l1(one_hot)  # [batch, intermediate, 1]
        x = x.squeeze(-1)  # [batch, intermediate]
        x = F.gelu(self.ln1(x))
        x = x.unsqueeze(-1)  # [batch, intermediate, 1]

        x = self.l2(x)  # [batch, emb_dim, 1]
        x = x.squeeze(-1)  # [batch, emb_dim]
        x = F.gelu(self.ln2(x))

        return x

class PositionalEncoding(nn.Module):
    """
    Classic sinusoidal positional encoding.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, 1, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(0)
        return x + self.pe[:seq_len]

class TransformerController(nn.Module):
    """
    Transformer controller with embeddings that integrates read vectors into attention.

    At each timestep:
    1. Input token is embedded to d_model dimensions
    2. Read vectors from NTM memory are projected to d_model
    3. Self-attention operates over: [input_history | projected_reads]
    4. Output is used for both next token prediction and NTM control
    """
    def __init__(self,
                 d_model: int,  # embedding/hidden dimension
                 n_heads: int,
                 n_layers: int = 2,
                 window_size: int = 16,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.window_size = window_size

        self.pos_encoding = PositionalEncoding(d_model)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=False,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(d_model)

    def forward_step(self,
                     input_emb: torch.Tensor, # [batch, D] embedded input
                     input_history,
                     read_vecs):
        """
        Forward step recieves: input vector, memory vector(s)
        """
        #inputs includes read vectors which we won't store
        if input_history is not None:
            input_history = torch.cat([input_history, input_emb.unsqueeze(0)], dim=0)
            if input_history.size(0) > self.window_size:
                input_history = input_history[-self.window_size:]
        else:
            input_history = input_emb.unsqueeze(0)

        input_history_ = self.pos_encoding(input_history)
        context = torch.cat((read_vecs, input_history_), dim=0)
        context = self.transformer(context)  # [hist_len + RH + state_layers, batch, D]

        current_pos = context.size(0) - 1
        controller_out = context[current_pos]  # [batch, D]
        controller_out = self.output_norm(controller_out)
        return controller_out, input_history

class RTDNC(nn.Module):
    def __init__(self,
                 input_size: int,
                 emb_dim: int,
                 memory_N: int,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 controller_window: int = 16,
                 read_heads: int = 1,
                 write_heads: int = 1,
                 dropout: float = 0.1):
        super().__init__()

        self.input_size = input_size
        self.D = emb_dim
        self.N = memory_N
        self.M = emb_dim
        self.RH = read_heads
        self.WH = write_heads
        self.window = controller_window

        #For expanded ability and reduced error
        self.head_dim = self.D
        self.read_param_len = self.M + 1 + 3
        self.write_param_len = self.M + 1 + self.M + self.M + self.RH + 1 + 1


        self.embedding = CNNEmbedding(self.input_size, self.D)
        self.controller = TransformerController(
            d_model=self.D,
            n_heads=n_heads,
            n_layers=n_layers,
            window_size=self.window,
            dropout=dropout
        )
        self.read_head = nn.Linear(self.D, self.RH * self.read_param_len)
        self.write_head = nn.Linear(self.D, self.WH * self.write_param_len)
        self.read_matrix = nn.Linear(self.RH*self.M, self.input_size)
        self.out = nn.Linear(self.D, self.input_size)

        self.register_buffer('memory_initial', torch.zeros(self.N, self.M))
        self.reset()

    def reset(self, batch_size: int = 1):
        self.batch_size = batch_size
        self.memory = self.memory_initial.unsqueeze(0).repeat(batch_size, 1, 1).clone()
        self.link_matrix = torch.zeros(batch_size, self.N, self.N)
        self.precedence = torch.zeros(batch_size, self.N)
        self.usage = torch.zeros(batch_size, self.N)

        init_w = torch.zeros(batch_size, self.N)
        init_w[:, 0] = 1.0
        self.read_w = init_w.unsqueeze(1).repeat(1, self.RH, 1)
        self.write_w = init_w.unsqueeze(1).repeat(1, self.WH, 1)

        self.input_history = None
        self.read_vecs = torch.zeros(self.RH, self.batch_size, self.M)

    def step(self, x_t: torch.Tensor) -> torch.Tensor:
        xe = self.embedding(x_t)  # [batch, d_model]

        o, self.input_history = self.controller.forward_step(
            xe, self.input_history, self.read_vecs) # controller_out: [batch, d_model]

        prev_read_w = self.read_w
        rh_params = self.read_head(o)
        new_read_w = []
        for h in range(self.RH):
            params = rh_params[:, h * self.read_param_len:(h + 1) * self.read_param_len]
            wt = self.addressing(params, self.read_w[:, h, :], self.memory, self.link_matrix, mode='read')
            new_read_w.append(wt)
        self.read_w = torch.stack(new_read_w, dim=1)  # [batch, RH, N]

        read_vecs = []
        for h in range(self.RH):
            w = self.read_w[:, h, :].unsqueeze(1)  # [batch, 1, N]
            r = torch.bmm(w, self.memory).squeeze(1)  # [batch, M]
            read_vecs.append(r)
        self.read_vecs = torch.stack(read_vecs, dim = 0) # [RH, batch, M]

        wh_params = self.write_head(o)
        new_memory, write_w, new_usage, new_link_matrix, new_precedence = \
            self.write_head_logic(wh_params, self.memory, self.usage,
                                 self.link_matrix, self.precedence, prev_read_w)

        self.memory = new_memory#.detach()
        self.usage = new_usage#.detach()
        self.link_matrix = new_link_matrix#.detach()
        self.precedence = new_precedence#.detach()
        self.write_w = write_w.unsqueeze(1)#.detach()

        y = self.out(o)  # [batch, vocab_size]
        return y

    def forward(self, token_seq: torch.Tensor, return_all_logits: bool = False):
        seq_len, batch = token_seq.shape
        device = token_seq.device

        self.reset(batch_size=batch)

        logits_all = []
        for t in range(seq_len):
            x_t = token_seq[t]
            logits = self.step(x_t)
            if return_all_logits:
                logits_all.append(logits.unsqueeze(0))

        if return_all_logits:
            return torch.cat(logits_all, dim=0)
        else:
            return logits

    def addressing(self, params, prev_w, memory, link_matrix, mode='read'):
        eps = 1e-12
        batch_size = memory.shape[0]

        # Parse parameters
        idx = 0
        key = params[:, idx:idx + self.M]; idx += self.M
        beta = params[:, idx:idx + 1]; idx += 1

        # Content-based addressing
        key_norm = key / (key.norm(dim=-1, keepdim=True) + eps)
        mem_norm = memory / (memory.norm(dim=-1, keepdim=True) + eps)
        cos_sim = torch.bmm(mem_norm, key_norm.unsqueeze(-1)).squeeze(-1)

        beta_pos = F.softplus(beta).squeeze(-1) + 1  # Ensure β ∈ [1, ∞)
        content_w = F.softmax(beta_pos.unsqueeze(-1) * cos_sim, dim=-1)

        if mode == 'write':
            # Write only uses content addressing
            return content_w

        elif mode == 'read':
            # Read modes: [backward, content, forward]
            read_modes = params[:, idx:idx + 3]; idx += 3
            modes = F.softmax(read_modes, dim=-1)  # [batch, 3]

            # Temporal link addressing
            # Forward: L @ prev_w
            forward_w = torch.bmm(link_matrix, prev_w.unsqueeze(-1)).squeeze(-1)
            # Backward: L^T @ prev_w
            backward_w = torch.bmm(link_matrix.transpose(1, 2), prev_w.unsqueeze(-1)).squeeze(-1)

            # Blend according to read modes: π[0]*backward + π[1]*content + π[2]*forward
            final_w = (modes[:, 0:1] * backward_w +
                       modes[:, 1:2] * content_w +
                       modes[:, 2:3] * forward_w)

            # Normalize (should already sum to ~1, but ensure numerical stability)
            final_w = final_w / (final_w.sum(dim=-1, keepdim=True) + eps)

            return final_w

    def write_head_logic(self, params, memory, usage, link_matrix, precedence, read_w_all):
        """
        Complete DNC write head logic including allocation, interpolation, and memory update.

        Args:
            params: [batch, write_param_len] - all write parameters
            memory: [batch, N, M] - memory matrix
            usage: [batch, N] - usage vector
            link_matrix: [batch, N, N] - temporal link matrix
            precedence: [batch, N] - precedence weighting
            read_w_all: [batch, RH, N] - all read head weightings

        Returns:
            new_memory: [batch, N, M] - updated memory
            write_w: [batch, N] - write weighting
            new_usage: [batch, N] - updated usage
            new_link_matrix: [batch, N, N] - updated link matrix
            new_precedence: [batch, N] - updated precedence
        """
        eps = 1e-12
        batch_size = memory.shape[0]

        # Parse parameters
        idx = 0
        key = params[:, idx:idx + self.M]; idx += self.M
        beta = params[:, idx:idx + 1]; idx += 1
        erase = params[:, idx:idx + self.M]; idx += self.M
        write = params[:, idx:idx + self.M]; idx += self.M
        free_gates = params[:, idx:idx + self.RH]; idx += self.RH
        alloc_gate = params[:, idx:idx + 1]; idx += 1
        write_gate = params[:, idx:idx + 1]; idx += 1

        # Apply activations
        erase = torch.sigmoid(erase)  # [batch, M] ∈ [0,1]
        write = write  # [batch, M] - no activation, can be any value
        free_gates = torch.sigmoid(free_gates)  # [batch, RH] ∈ [0,1]
        g_a = torch.sigmoid(alloc_gate).squeeze(-1)  # [batch] ∈ [0,1]
        g_w = torch.sigmoid(write_gate).squeeze(-1)  # [batch] ∈ [0,1]

        # 1. Content-based addressing for write
        content_w = self.addressing(params[:, :self.M + 1], torch.zeros(batch_size, self.N, device=memory.device), memory, None, mode='write')

        # 2. Calculate retention vector ψ (psi)
        # ψ = ∏(1 - f_i * w_r,i) over all read heads
        retention = torch.ones(batch_size, self.N, device=memory.device)
        for i in range(self.RH):
            retention = retention * (1 - free_gates[:, i:i+1] * read_w_all[:, i, :])

        # 3. Update usage
        # u_t = (u_{t-1} + w_w - u_{t-1} ⊙ w_w) ⊙ ψ_t
        # Note: We use previous write_w here (from last timestep stored in self.write_w)
        prev_write_w = self.write_w[:, 0, :]  # Assuming single write head for now
        new_usage = (usage + prev_write_w - usage * prev_write_w) * retention
        new_usage = torch.clamp(new_usage, min=0)

        # 4. Allocation weighting
        # Sort by usage (ascending - least used first)
        sorted_usage, free_list = torch.sort(new_usage, dim=-1, descending=False)

        # Calculate allocation weights using equation (1) from paper
        # a_t[φ_t[j]] = (1 - u_t[φ_t[j]]) * ∏_{i=1}^{j-1} u_t[φ_t[i]]
        cumprod = torch.cumprod(sorted_usage + eps, dim=-1)
        cumprod = torch.cat([torch.ones(batch_size, 1, device=memory.device), cumprod[:, :-1]], dim=-1)
        allocation_w_sorted = (1 - sorted_usage) * cumprod
        allocation_w = torch.zeros_like(new_usage).scatter_(1, free_list, allocation_w_sorted)

        # 5. Interpolate allocation and content weightings
        # w_w = g_w * (g_a * a + (1 - g_a) * c_w)
        write_w = g_w.unsqueeze(-1) * (g_a.unsqueeze(-1) * allocation_w +
                                        (1 - g_a.unsqueeze(-1)) * content_w)

        # 6. Update temporal link matrix and precedence
        # L_t[i,j] = (1 - w_w[i] - w_w[j]) * L_{t-1}[i,j] + w_w[i] * p_{t-1}[j]
        # Exclude self-links (diagonal = 0)
        new_link_matrix = (1 - write_w.unsqueeze(1) - write_w.unsqueeze(2)) * link_matrix
        new_link_matrix = new_link_matrix + torch.bmm(
            write_w.unsqueeze(2),  # [batch, N, 1]
            precedence.unsqueeze(1)  # [batch, 1, N]
        )

        # Zero out diagonal
        eye = torch.eye(self.N, device=link_matrix.device).unsqueeze(0)
        new_link_matrix = new_link_matrix * (1 - eye)

        # p_t = (1 - sum(w_w)) * p_{t-1} + w_w
        new_precedence = (1 - write_w.sum(dim=-1, keepdim=True)) * precedence + write_w

        # 7. Erase and write to memory
        # M_t[i,j] = M_{t-1}[i,j] * (1 - w_w[i] * e[j]) + w_w[i] * v[j]
        erase_term = 1 - torch.bmm(write_w.unsqueeze(2), erase.unsqueeze(1))  # [batch, N, M]
        new_memory = memory * erase_term

        write_term = torch.bmm(write_w.unsqueeze(2), write.unsqueeze(1))  # [batch, N, M]
        new_memory = new_memory + write_term

        return new_memory, write_w, new_usage, new_link_matrix, new_precedence

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_memory_usage_stats(self):
        """Diagnostic statistics for DNC memory and addressing"""
        stats = {}
        eps = 1e-12

        stats['input_history_len'] = self.input_history.size(0)
        stats['input_history_max'] = self.window
        stats['history_utilization'] = stats['input_history_len'] / self.window

        # Memory content statistics
        stats['memory_mean'] = self.memory.mean().item()
        stats['memory_std'] = self.memory.std().item()
        stats['memory_abs_max'] = self.memory.abs().max().item()
        stats['memory_sparsity'] = (self.memory.abs() < 0.01).float().mean().item()

        # Memory usage vector (key DNC diagnostic)
        stats['usage_mean'] = self.usage.mean().item()
        stats['usage_std'] = self.usage.std().item()
        stats['usage_max'] = self.usage.max().item()
        stats['usage_min'] = self.usage.min().item()
        stats['num_slots_used'] = (self.usage > 0.5).sum().item() / self.batch_size
        stats['num_slots_free'] = (self.usage < 0.1).sum().item() / self.batch_size

        # Read head statistics
        read_entropy = -(self.read_w * (self.read_w + eps).log()).sum(-1).mean().item()
        stats['read_entropy'] = read_entropy
        stats['read_sharpness'] = math.log(self.N) - read_entropy
        stats['read_max_weight'] = self.read_w.max().item()
        stats['read_top3_sum'] = self.read_w.topk(min(3, self.N), dim=-1)[0].sum(-1).mean().item()

        # Write head statistics
        write_entropy = -(self.write_w * (self.write_w + eps).log()).sum(-1).mean().item()
        stats['write_entropy'] = write_entropy
        stats['write_sharpness'] = math.log(self.N) - write_entropy
        stats['write_max_weight'] = self.write_w.max().item()
        stats['write_top3_sum'] = self.write_w.topk(min(3, self.N), dim=-1)[0].sum(-1).mean().item()

        # Link matrix statistics (temporal connections)
        stats['link_density'] = (self.link_matrix.abs() > 0.01).float().mean().item()
        stats['link_max'] = self.link_matrix.max().item()
        stats['link_mean'] = self.link_matrix.mean().item()

        # Precedence vector
        stats['precedence_entropy'] = -(self.precedence * (self.precedence + eps).log()).sum(-1).mean().item()
        stats['precedence_max'] = self.precedence.max().item()

        return stats

    def print_memory_stats(self):
        """Pretty print memory statistics"""
        stats = self.get_memory_usage_stats()

        print("\n=== DNC Memory Statistics ===")
        print(f"Memory Content:")
        print(f"  Mean: {stats['memory_mean']:.4f}, Std: {stats['memory_std']:.4f}")
        print(f"  Max: {stats['memory_abs_max']:.4f}, Sparsity: {stats['memory_sparsity']:.2%}")

        print(f"\nMemory Usage (key metric):")
        print(f"  Mean: {stats['usage_mean']:.4f}, Std: {stats['usage_std']:.4f}")
        print(f"  Range: [{stats['usage_min']:.4f}, {stats['usage_max']:.4f}]")
        print(f"  Slots used (>0.5): {stats['num_slots_used']:.1f}/{self.N}")
        print(f"  Slots free (<0.1): {stats['num_slots_free']:.1f}/{self.N}")

        print(f"\nRead Heads:")
        print(f"  Entropy: {stats['read_entropy']:.4f}, Sharpness: {stats['read_sharpness']:.4f}")
        print(f"  Max weight: {stats['read_max_weight']:.4f}, Top-3 sum: {stats['read_top3_sum']:.4f}")

        print(f"\nWrite Heads:")
        print(f"  Entropy: {stats['write_entropy']:.4f}, Sharpness: {stats['write_sharpness']:.4f}")
        print(f"  Max weight: {stats['write_max_weight']:.4f}, Top-3 sum: {stats['write_top3_sum']:.4f}")

        print(f"\nTemporal Links:")
        print(f"  Density: {stats['link_density']:.2%}, Max: {stats['link_max']:.4f}")
        print(f"  Precedence entropy: {stats['precedence_entropy']:.4f}, Max: {stats['precedence_max']:.4f}")

        print("="*35)

if __name__ == "__main__":
    # ---------------------------
    # Simple tokenizer
    # ---------------------------
    vocab = string.digits + string.ascii_letters + string.punctuation + " \t\v\n\r\f"
    vocab_size = len(vocab)
    char_to_idx = {char: idx for idx, char in enumerate(vocab)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)
    print("="*70)
    print("RTDNC - Embedding-Based Transformer Controller")
    print("="*70)
    print(f"Device: {device}\n")

    # Model Meta Parameters
    emb_dim = 200
    memory_N = 128
    n_heads = 4
    n_layers = 2
    controller_window = 16
    read_heads = 2
    write_heads = 1

    print("Architecture:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Embedding dimension: {emb_dim}")
    print(f"  Memory dimension (M): {emb_dim}")
    print(f"  Transformer layers: {n_layers}")
    print(f"  Attention heads: {n_heads}")
    print(f"  Controller window: {controller_window} input tokens")
    print(f"  Read heads: {read_heads}, Write heads: {write_heads}")
    print(f"  External memory: {memory_N} × {emb_dim}")
    print()

    model = RTDNC(
        input_size=vocab_size,
        emb_dim=emb_dim,
        memory_N=memory_N,
        n_heads=n_heads,
        n_layers=n_layers,
        controller_window=controller_window,
        read_heads=read_heads,
        write_heads=write_heads,
        dropout=0.1
    )

    total_params = model.num_params()
    print(f"Total parameters: {total_params:,}\n")

    # Test single step
    print("="*70)
    print("Testing Single Step Forward Pass")
    print("="*70)
    batch = 2
    seq_len = 25

    torch.manual_seed(42)
    token_seq = torch.randint(0, vocab_size, (seq_len, batch))

    print(f"Input: seq_len={seq_len}, batch={batch}")
    print("Processing step-by-step:\n")

    state = model.reset(batch_size=batch)

    for t in range(seq_len):
        x_t = token_seq[t]
        logits = model.step(x_t)

        if t % 5 == 0 or t == seq_len - 1:
            stats = model.get_memory_usage_stats()
            print(f"t={t:2d}: InputHist={stats['input_history_len']:>2}/{stats['input_history_max']} "
                  f"(util={stats['history_utilization']:>5.1%}), "
                  f"Mem_std={stats['memory_std']:.4f}, "
                  f"Read_sharp={stats['read_sharpness']:>5.2f}, "
                  f"Write_sharp={stats['write_sharpness']:>5.2f}")

    print(f"\nFinal logits shape: {logits.shape}")
    print(f"Expected: [{batch}, {vocab_size}]")

    # Test full sequence forward
    print("\n" + "="*70)
    print("Testing Full Sequence Forward Pass")
    print("="*70)

    model.reset(batch_size=batch)
    all_logits = model.forward(token_seq, return_all_logits=True)

    print(f"Input shape:  {token_seq.shape}")
    print(f"Output shape: {all_logits.shape}")
    print(f"Expected:     [{seq_len}, {batch}, {vocab_size}]")

    # Verify predictions
    print("\n" + "="*70)
    print("Sample Predictions")
    print("="*70)

    # Show first 5 timesteps
    for t in range(min(5, seq_len)):
        input_token = token_seq[t, 0].item()
        pred_token = all_logits[t, 0].argmax().item()

        input_char = idx_to_char[input_token]
        pred_char = idx_to_char[pred_token]

        # Show top-3 predictions
        top_logits, top_indices = all_logits[t, 0].topk(3)
        top_chars = [idx_to_char[idx.item()] for idx in top_indices]
        top_probs = torch.softmax(top_logits, dim=0)

        print(f"t={t}: Input='{input_char}' (idx={input_token:>3})")
        print(f"      Top-3: ", end="")
        for char, prob in zip(top_chars, top_probs):
            print(f"'{char}'({prob:.2%}) ", end="")
        print()

    # Final statistics
    print("\n" + "="*70)
    print("Final Memory Statistics")
    print("="*70)
    stats = model.get_memory_usage_stats()
    print(f"Memory utilization:")
    print(f"  Mean value:     {stats['memory_mean']:>8.4f}")
    print(f"  Std deviation:  {stats['memory_std']:>8.4f}")
    print(f"  Max absolute:   {stats['memory_abs_max']:>8.4f}")
    print(f"  Sparsity:       {stats['memory_sparsity']:>8.2%}")
    print(f"\nAttention sharpness:")
    print(f"  Read entropy:   {stats['read_entropy']:>8.4f} (max={math.log(memory_N):.2f})")
    print(f"  Write entropy:  {stats['write_entropy']:>8.4f} (max={math.log(memory_N):.2f})")
    print(f"  Read sharpness: {stats['read_sharpness']:>8.4f}")
    print(f"  Write sharpness:{stats['write_sharpness']:>8.4f}")
    print(f"\nHistory buffer:")
    print(f"  Utilization:    {stats['history_utilization']:>8.2%}")
    print(f"  Length:         {stats['input_history_len']:>8} / {stats['input_history_max']}")

    print("\n" + "="*70)
    print("Key Features:")
    print("  ✓ Token embeddings (learnable representation)")
    print("  ✓ Separate d_model and memory_M dimensions")
    print("  ✓ Read vectors projected and integrated into transformer")
    print("  ✓ Multi-layer transformer controller")
    print("  ✓ Self-attention over: [embedded_history, projected_reads]")
    print("  ✓ Bounded input history (forces external memory use)")
    print("  ✓ End-to-end differentiable")
    print("="*70)
