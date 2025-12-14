"""
2014 Neural Turing Machines,
     Alex Graves, gravesa@google.com
     Greg Wayne, gregwayne@google.com
     Ivo Danihelka, danihelka@google.com

Hudson Andrew Smelski

RTNTM
Transformer controller for NTM
"""

import string
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional

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
        self.register_buffer('pe', pe)  # Not a parameter, persists in state_dict

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [seq_len, batch, d_model]
        Returns:
            x + positional encoding
        """
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

        self.pos_encoding = PositionalEncoding(d_model, window_size)

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
                     input_history: torch.Tensor):
        """
        Process one timestep with embedded input and projected read vectors.

        Args:
            input_emb: [batch, D] - embedded input token
            ctl_state: dict with 'input_history' [hist_len, batch, D]

        Returns:
            controller_out: [batch, D] - output for prediction & NTM control
            new_ctl_state: updated state dict
        """
        batch = input_emb.shape[0]
        device = input_emb.device

        # Add current input to history
        current_input = input_emb.unsqueeze(0)  # [1, batch, D]

        if input_history is not None:
            context = torch.cat([input_history, current_input], dim=0)
            # Enforce window size
            if context.size(0) > self.window_size:
                context = context[-self.window_size:]
        else:
            context = current_input

        context = self.pos_encoding(context)

        # Apply transformer over full context
        context_out = self.transformer(context)  # [hist_len+RH, batch, D]
        current_pos = context.size(0) - 1
        controller_out = context_out[current_pos]  # [batch, D]

        controller_out = self.output_norm(controller_out)
        return controller_out, context


class RTNTM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 emb_dim: int,
                 memory_N: int,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 controller_window: int = 16,
                 read_heads: int = 1,
                 write_heads: int = 1,
                 shift_width: int = 3,
                 dropout: float = 0.1):
        """
        RTNTM with FiLM modulation and persistent GRU state.

        Key features:
        - FiLM: Conditions input on persistent record state
        - GRU State: Maintains context beyond bounded history window
        - Direct read integration: Read vectors added to FiLM output

        Args:
            vocab_size: Size of token vocabulary
            emb_dim: Embedding and hidden dimension (D)
            memory_N: Number of memory slots
            ...
        """
        super().__init__()

        assert shift_width % 2 == 1, "shift_width must be odd"

        self.vocab_size = vocab_size
        self.D = emb_dim
        self.N = memory_N
        self.M = emb_dim #memory_M
        self.RH = read_heads
        self.WH = write_heads
        self.shift_K = shift_width
        self.half_shift = shift_width // 2
        self.controller_window = controller_window

        # Input embedding
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)

        # Project read vectors from M to D if we allow different size M
        #self.read_projection = nn.Linear(memory_M, emb_dim)

        self.read_gate = nn.Linear(self.D, self.M)
        self.film = nn.Linear(self.D, 2 * self.D)
        # Initialize gamma bias to 0 (so initial gamma ≈ 1)
        nn.init.zeros_(self.film.bias[:self.D])
        # Initialize beta bias to 0 (so initial beta ≈ 0)
        nn.init.zeros_(self.film.bias[self.D:])

        # Transformer controller
        self.controller = TransformerController(
            d_model=self.D,
            n_heads=n_heads,
            n_layers=n_layers,
            window_size=controller_window,
            dropout=dropout
        )
        #self.state_update = nn.GRUCell(self.D, self.D)
        #self.state_update = nn.LSTMCell(self.D, self.D)
        self.state_layers = 2 #TODO: parameterize this as an arg
        self.state_update = nn.LSTM(
            input_size=self.D,
            hidden_size=self.D,
            num_layers=self.state_layers,
            dropout=0.1 if self.state_layers > 1 else 0.0
        )

        # Output head for next token prediction
        self.token_head = nn.Linear(self.D, vocab_size)

        # Read head: projects controller output to NTM read parameters
        self.read_param_len = self.M + 1 + 1 + self.shift_K + 1
        self.read_head = nn.Linear(self.D, self.RH * self.read_param_len)

        # Write head: projects controller output to NTM write parameters
        self.write_param_len = self.M + 1 + 1 + self.shift_K + 1 + self.M + self.M
        self.write_head = nn.Linear(self.D, self.WH * self.write_param_len)

        # Initialize memory to zeros
        self.register_buffer('memory_initial', torch.zeros(self.N, self.M))

    def init_state(self, batch_size: int = 1, device=None):
        """Initialize clean state"""
        if device is None:
            device = next(self.parameters()).device

        # Memory: [batch, N, M]
        self.memory = self.memory_initial.unsqueeze(0).repeat(batch_size, 1, 1).clone()

        # Read/write weights: [batch, RH/WH, N]
        read_w = torch.zeros(batch_size, self.RH, self.N, device=device)
        read_w[:, :, 0] = 1.0
        self.read_w = read_w

        write_w = torch.zeros(batch_size, self.WH, self.N, device=device)
        write_w[:, :, 0] = 1.0
        self.write_w = write_w

        self.record = torch.zeros(batch_size, self.D, device = device)
        #self.c = torch.zeros(batch_size, self.D, device = device)
        self.hx = torch.zeros(self.state_layers, batch_size, self.D, device=device)
        self.cx = torch.zeros(self.state_layers, batch_size, self.D, device=device)
        self.input_history = torch.zeros(1, batch_size, self.D, device = device)

    def addressing(self, key, beta, gate, shift, gamma, prev_w, memory) -> torch.Tensor:
        """NTM addressing mechanism"""
        eps = 1e-12

        # Content addressing
        key_norm = key / (key.norm(dim=-1, keepdim=True) + eps)
        mem_norm = memory / (memory.norm(dim=-1, keepdim=True) + eps)
        cos_sim = torch.bmm(mem_norm, key_norm.unsqueeze(-1)).squeeze(-1)

        beta_pos = F.softplus(beta).squeeze(-1)
        wc = F.softmax(beta_pos.unsqueeze(-1) * cos_sim, dim=-1)

        # Interpolation
        g = torch.sigmoid(gate).squeeze(-1)
        wg = g.unsqueeze(-1) * wc + (1.0 - g).unsqueeze(-1) * prev_w

        # Convolutional shift
        s = F.softmax(shift, dim=-1)
        shifted = torch.zeros_like(wg)
        for k in range(self.shift_K):
            shift_amount = k - self.half_shift
            rolled = torch.roll(wg, shifts=shift_amount, dims=-1)
            shifted = shifted + s[:, k].unsqueeze(-1) * rolled

        # Sharpening
        gamma_p = 1.0 + F.softplus(gamma).squeeze(-1)
        wt = (shifted + eps) ** gamma_p.unsqueeze(-1)
        wt = wt / (wt.sum(dim=-1, keepdim=True) + eps)

        return wt

    def step(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        Single timestep forward pass.

        Args:
            x_t: [batch] token indices

        Returns:
            logits: [batch, vocab_size] - next token predictions
            new_state: updated state dict
        """
        batch = x_t.shape[0]
        device = x_t.device

        # Embed input token
        input_emb = self.embedding(x_t)  # [batch, d_model]

        read_vecs = []
        for h in range(self.RH):
            w = self.read_w[:, h, :].unsqueeze(1)  # [batch, 1, N]
            r = torch.bmm(w, self.memory).squeeze(1)  # [batch, M]
            read_vecs.append(r)
            #read_vecs.append(self.read_projection(r))
        read_vecs = torch.stack(read_vecs, dim = 1) # [batch, RH, M]

        # Use record state to attend over read heads
        #read_attn = F.softmax(torch.bmm(
        #    self.record.unsqueeze(1),  # [batch, 1, D]
        #    read_vecs.transpose(1, 2)  # [batch, D, RH]
        #).squeeze(1), dim=-1)  # [batch, RH]
        #read_vecs = (read_vecs * read_attn.unsqueeze(-1)).sum(dim=1)  # [batch, M]

        read_vecs = torch.sum(read_vecs, dim = 1) #[batch, M]

        #FiLM
        film_params = self.film(self.record) #[batch, 2*M]
        gamma, beta = film_params.chunk(2, dim=-1)
        gamma = 1 + torch.tanh(gamma)

        read_gate = torch.sigmoid(self.read_gate(self.record))
        input_vector = gamma * input_emb + beta + read_gate * read_vecs
        #input_vector = input_emb + self.record + read_gate * read_vecs

        # Controller forward: transformer over [input_history + projected_reads]
        controller_out, self.input_history = self.controller.forward_step(
            input_vector, self.input_history)
        # controller_out: [batch, d_model]

        #self.record, self.c = self.state_update(controller_out, (self.record, self.c))
        controller_out_unsq = controller_out.unsqueeze(0)  # [1, batch, D]
        output, (self.hx, self.cx) = self.state_update(controller_out_unsq, (self.hx, self.cx))
        self.record = self.hx[-1]  # Take top layer as conditioning state

        controller_out = controller_out + output.squeeze()

        # Next token prediction
        logits = self.token_head(controller_out)  # [batch, vocab_size]

        # Generate read parameters from controller output
        read_params = self.read_head(controller_out)  # [batch, RH * read_param_len]
        read_params = read_params.view(batch, self.RH, self.read_param_len)

        new_read_w = []
        for h in range(self.RH):
            rp = read_params[:, h, :]
            idx = 0
            key = rp[:, idx:idx + self.M]; idx += self.M
            beta = rp[:, idx:idx + 1]; idx += 1
            gate = rp[:, idx:idx + 1]; idx += 1
            shift = rp[:, idx:idx + self.shift_K]; idx += self.shift_K
            gamma = rp[:, idx:idx + 1]; idx += 1

            prev_w = self.read_w[:, h, :]
            wt = self.addressing(key, beta, gate, shift, gamma, prev_w, self.memory)
            new_read_w.append(wt)

        self.read_w = torch.stack(new_read_w, dim=1)

        # Generate write parameters and update memory
        write_params = self.write_head(controller_out)
        write_params = write_params.view(batch, self.WH, self.write_param_len)

        mem = self.memory
        new_write_w = []
        for h in range(self.WH):
            wp = write_params[:, h, :]
            idx = 0
            key = wp[:, idx:idx + self.M]; idx += self.M
            beta = wp[:, idx:idx + 1]; idx += 1
            gate = wp[:, idx:idx + 1]; idx += 1
            shift = wp[:, idx:idx + self.shift_K]; idx += self.shift_K
            gamma = wp[:, idx:idx + 1]; idx += 1
            erase = wp[:, idx:idx + self.M]; idx += self.M
            add = wp[:, idx:idx + self.M]; idx += self.M

            prev_w = self.write_w[:, h, :]
            wt = self.addressing(key, beta, gate, shift, gamma, prev_w, mem)
            new_write_w.append(wt)

            # Erase and add
            erase_v = torch.sigmoid(erase)
            add_v = torch.tanh(add)

            erase_matrix = wt.unsqueeze(-1) * erase_v.unsqueeze(1)
            mem = mem * (1.0 - erase_matrix)

            add_matrix = wt.unsqueeze(-1) * add_v.unsqueeze(1)
            mem = mem + add_matrix

        self.memory = mem
        self.write_w = torch.stack(new_write_w, dim=1)

        return logits

    def forward(self,
                token_seq: torch.Tensor,
                return_all_logits: bool = False):
        """
        Process full sequence.

        Args:
            token_seq: [seq_len, batch] token indices
            return_all_logits: if True, return logits for all timesteps

        Returns:
            logits: [seq_len, batch, vocab_size] or [batch, vocab_size]
            state: final state
        """
        seq_len, batch = token_seq.shape
        device = token_seq.device

        self.init_state(batch_size=batch, device=device)

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

    def get_memory_usage_stats(self) -> Dict[str, Any]:
        """Diagnostic statistics"""
        stats = {}

        # Input history length
        stats['input_history_len'] = self.input_history.size(0)
        stats['input_history_max'] = self.controller_window
        stats['history_utilization'] = stats['input_history_len'] / self.controller_window

        # Memory statistics
        stats['memory_mean'] = self.memory.mean().item()
        stats['memory_std'] = self.memory.std().item()
        stats['memory_abs_max'] = self.memory.abs().max().item()
        stats['memory_sparsity'] = (self.memory.abs() < 0.01).float().mean().item()

        # Attention sharpness
        eps = 1e-12
        read_entropy = -(self.read_w * (self.read_w + eps).log()).sum(-1).mean().item()
        write_entropy = -(self.write_w * (self.write_w + eps).log()).sum(-1).mean().item()

        stats['read_entropy'] = read_entropy
        stats['write_entropy'] = write_entropy
        stats['read_sharpness'] = math.log(self.N) - read_entropy
        stats['write_sharpness'] = math.log(self.N) - write_entropy

        # Record state statistics
        stats['record_mean'] = self.record.mean().item()
        stats['record_std'] = self.record.std().item()

        return stats

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
    print("RTNTM - Embedding-Based Transformer Controller")
    print("="*70)
    print(f"Device: {device}\n")

    # Configuration
    emb_dim = 128          # Embedding and transformer hidden dimension
    memory_N = 128         # Number of memory slots
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

    model = RTNTM(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        memory_N=memory_N,
        n_heads=n_heads,
        n_layers=n_layers,
        controller_window=controller_window,
        read_heads=read_heads,
        write_heads=write_heads,
        shift_width=3,
        dropout=0.1
    ).to(device)

    total_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}\n")

    # Test single step
    print("="*70)
    print("Testing Single Step Forward Pass")
    print("="*70)
    batch = 2
    seq_len = 25

    torch.manual_seed(42)
    token_seq = torch.randint(0, vocab_size, (seq_len, batch), device=device)

    print(f"Input: seq_len={seq_len}, batch={batch}")
    print("Processing step-by-step:\n")

    state = model.init_state(batch_size=batch, device=device)

    for t in range(seq_len):
        x_t = token_seq[t]
        logits, state = model.step(x_t, state)

        if t % 5 == 0 or t == seq_len - 1:
            stats = model.get_memory_usage_stats(state)
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

    state = model.init_state(batch_size=batch, device=device)
    all_logits, final_state = model.forward(token_seq, state=state, return_all_logits=True)

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
    stats = model.get_memory_usage_stats(final_state)
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
