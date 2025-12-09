"""
2014 Neural Turing Machines,
     Alex Graves, gravesa@google.com
     Greg Wayne, gregwayne@google.com
     Ivo Danihelka, danihelka@google.com

Hudson Andrew Smelski

RTNTM - Recurrent Transformer NTM with BOUNDED internal memory
Wide input architecture: concatenates token embeddings with read vectors

Key design:
- Controller input = [token_emb, read_vec_1, ..., read_vec_RH]  (wide)
- Projected to d_model for attention (compression bottleneck)
- KV cache stores only d_model compressed states (bounded memory)
- Forces model to use external NTM memory for data storage
"""

import string
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional

# ---------------------------
# Simple tokenizer
# ---------------------------
vocab = string.digits + string.ascii_letters + string.punctuation + " \t\n"
vocab_size = len(vocab)
char_to_idx = {char: idx for idx, char in enumerate(vocab)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}


# ---------------------------
# Recurrent Transformer Controller with WIDE INPUT
# ---------------------------
class WideInputRecurrentTransformerController(nn.Module):
    """
    Transformer controller that takes WIDE concatenated input but stores
    compressed state in bounded KV cache.

    Input: [token_emb, read_vectors] → wide_input_dim
    Compressed to: d_model for attention and storage
    KV cache: bounded to window_size tokens of d_model
    """
    def __init__(self,
                 d_model: int,
                 wide_input_dim: int,  # d_model + RH * M
                 n_heads: int,
                 window_size: int = 8,
                 dropout: float = 0.1,
                 use_summary: bool = False):
        super().__init__()
        self.d_model = d_model
        self.wide_input_dim = wide_input_dim
        self.n_heads = n_heads
        self.window_size = window_size
        self.use_summary = use_summary

        # Project wide input to d_model (compression bottleneck)
        self.input_proj = nn.Linear(wide_input_dim, d_model)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        # Multi-head attention (operates in d_model space)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=False
        )

        # Feed-forward network
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(d_model)

        # Optional: compress evicted context into summary
        if self.use_summary:
            self.summary_compress = nn.Linear(d_model * 2, d_model)
            self.summary_gate = nn.Linear(d_model * 2, 1)

    def forward_step(self,
                     wide_input: torch.Tensor,  # [batch, wide_input_dim]
                     ctl_state: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process one time step with wide input

        wide_input: [batch, wide_input_dim] = concatenated [token_emb, read_vecs]
        ctl_state: dict with 'kv_cache' and optional 'summary'

        Returns: out [batch, d_model], new_ctl_state
        """
        batch = wide_input.shape[0]
        device = wide_input.device

        # PROJECT wide input to d_model (BOTTLENECK)
        x = self.input_proj(wide_input)  # [batch, d_model]

        # This compressed x becomes our query
        q = x.unsqueeze(0)  # [1, batch, d_model]

        # Get KV cache (bounded to window_size)
        kv_cache: Optional[torch.Tensor] = ctl_state.get('kv_cache', None)

        # Build key and value from cache + current query
        if kv_cache is not None:
            k = torch.cat([kv_cache, q], dim=0)  # [cache_len+1, batch, d_model]
            v = k  # Self-attention: v = k

            # ENFORCE WINDOW SIZE
            if k.size(0) > self.window_size:
                if self.use_summary:
                    # Compress evicted tokens
                    evicted = k[:-self.window_size]
                    evicted_mean = evicted.mean(dim=0)  # [batch, d_model]

                    old_summary = ctl_state.get('summary',
                                                torch.zeros(batch, self.d_model, device=device))

                    gate_input = torch.cat([old_summary, evicted_mean], dim=-1)
                    gate = torch.sigmoid(self.summary_gate(gate_input))

                    new_summary = self.summary_compress(gate_input)
                    new_summary = gate * old_summary + (1 - gate) * new_summary
                    ctl_state['summary'] = new_summary

                # Truncate to window
                k = k[-self.window_size:]
                v = v[-self.window_size:]
        else:
            k = q
            v = q

        # Multi-head attention
        attn_out, attn_weights = self.mha(q, k, v, need_weights=True)  # [1, batch, d_model]
        attn_out = attn_out.squeeze(0)  # [batch, d_model]

        # Optional: add summary contribution
        if self.use_summary and 'summary' in ctl_state:
            summary = ctl_state['summary']
            attn_out = attn_out + 0.1 * summary

        # Residual + LayerNorm
        x2 = self.ln1(x + attn_out)

        # Feed-forward + residual
        ff_out = self.ff(x2)
        out = self.ln2(x2 + ff_out)

        # Update KV cache: store compressed state (q)
        new_cache = q.detach().clone() if kv_cache is None else \
                    torch.cat([kv_cache, q.detach().clone()], dim=0)

        # Enforce window size
        if new_cache.size(0) > self.window_size:
            new_cache = new_cache[-self.window_size:]

        new_state = dict(ctl_state)
        new_state['kv_cache'] = new_cache

        return out, new_state


# ---------------------------
# RTNTM with wide input architecture
# ---------------------------
class RTNTM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 memory_N: int,
                 memory_M: int,
                 n_heads: int = 4,
                 controller_window: int = 8,
                 read_heads: int = 1,
                 write_heads: int = 1,
                 shift_width: int = 3,
                 use_summary: bool = False,
                 dropout: float = 0.1):
        """
        RTNTM with wide input controller

        Key parameters:
        - d_model: controller internal dimension (and token embedding size)
        - memory_M: width of each memory row (can differ from d_model)
        - controller_window: max KV cache length (forces external memory use)
        """
        super().__init__()

        assert shift_width % 2 == 1, "shift_width must be odd"

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.N = memory_N
        self.M = memory_M
        self.RH = read_heads
        self.WH = write_heads
        self.shift_K = shift_width
        self.half_shift = shift_width // 2
        self.controller_window = controller_window

        # Token embedding (to d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        # Controller with WIDE input: [d_model + RH * M]
        wide_input_dim = d_model + self.RH * self.M
        self.controller = WideInputRecurrentTransformerController(
            d_model=d_model,
            wide_input_dim=wide_input_dim,
            n_heads=n_heads,
            window_size=controller_window,
            dropout=dropout,
            use_summary=use_summary
        )

        # Output head: d_model → vocab_size
        self.token_head = nn.Linear(d_model, vocab_size)
        nn.init.xavier_uniform_(self.token_head.weight)

        # Read head parameters: key(M) + beta(1) + gate(1) + shift(K) + gamma(1)
        self.read_param_len = self.M + 1 + 1 + self.shift_K + 1
        self.read_head = nn.Linear(d_model, self.RH * self.read_param_len)

        # Write head parameters: key(M) + beta(1) + gate(1) + shift(K) + gamma(1) + erase(M) + add(M)
        self.write_param_len = self.M + 1 + 1 + self.shift_K + 1 + self.M + self.M
        self.write_head = nn.Linear(d_model, self.WH * self.write_param_len)

        # Initialize memory to zeros
        self.register_buffer('memory_initial', torch.zeros(self.N, self.M))

    def init_state(self, batch_size: int = 1, device: Optional[torch.device] = None) -> Dict[str, Any]:
        """Initialize clean state for new sequence"""
        if device is None:
            device = next(self.parameters()).device

        state: Dict[str, Any] = {}

        # Memory: [batch, N, M]
        state['memory'] = self.memory_initial.unsqueeze(0).repeat(batch_size, 1, 1).clone()

        # Read/write weights: [batch, RH/WH, N] - focused on location 0
        read_w = torch.zeros(batch_size, self.RH, self.N, device=device)
        read_w[:, :, 0] = 1.0
        state['read_w'] = read_w

        write_w = torch.zeros(batch_size, self.WH, self.N, device=device)
        write_w[:, :, 0] = 1.0
        state['write_w'] = write_w

        # Controller state: empty cache
        state['controller_state'] = {
            'kv_cache': None,
            'summary': None if not self.controller.use_summary else
                      torch.zeros(batch_size, self.d_model, device=device)
        }

        return state

    def addressing(self, key, beta, gate, shift, gamma, prev_w, memory) -> torch.Tensor:
        """
        NTM addressing: content-based + location-based
        Implements equations from Graves et al. 2014
        """
        eps = 1e-12

        # 1) Content addressing (cosine similarity)
        key_norm = key / (key.norm(dim=-1, keepdim=True) + eps)
        mem_norm = memory / (memory.norm(dim=-1, keepdim=True) + eps)
        cos_sim = torch.bmm(mem_norm, key_norm.unsqueeze(-1)).squeeze(-1)

        beta_pos = F.softplus(beta).squeeze(-1)
        wc = F.softmax(beta_pos.unsqueeze(-1) * cos_sim, dim=-1)

        # 2) Interpolation
        g = torch.sigmoid(gate).squeeze(-1)
        wg = g.unsqueeze(-1) * wc + (1.0 - g).unsqueeze(-1) * prev_w

        # 3) Convolutional shift
        s = F.softmax(shift, dim=-1)
        shifted = torch.zeros_like(wg)
        for k in range(self.shift_K):
            shift_amount = k - self.half_shift
            rolled = torch.roll(wg, shifts=shift_amount, dims=-1)
            shifted = shifted + s[:, k].unsqueeze(-1) * rolled

        # 4) Sharpening
        gamma_p = 1.0 + F.softplus(gamma).squeeze(-1)
        wt = (shifted + eps) ** gamma_p.unsqueeze(-1)
        wt = wt / (wt.sum(dim=-1, keepdim=True) + eps)

        return wt

    def step(self, x_t: torch.Tensor, state: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Single time step with wide input architecture

        x_t: [batch] token indices
        state: dict with memory, weights, controller_state
        returns: logits [batch, vocab_size], new_state
        """
        batch = x_t.shape[0]
        device = x_t.device

        # 1) Embed input token
        token_emb = self.embedding(x_t)  # [batch, d_model]

        # 2) Read from memory using current read weights
        memory = state['memory']  # [batch, N, M]
        read_w = state['read_w']  # [batch, RH, N]

        read_vecs = []
        for h in range(self.RH):
            w = read_w[:, h, :].unsqueeze(1)  # [batch, 1, N]
            r = torch.bmm(w, memory).squeeze(1)  # [batch, M]
            read_vecs.append(r)
        read_concat = torch.cat(read_vecs, dim=-1)  # [batch, RH * M]

        # 3) WIDE INPUT: concatenate token embedding with read vectors
        wide_input = torch.cat([token_emb, read_concat], dim=-1)  # [batch, d_model + RH*M]

        # 4) Controller forward (compresses to d_model internally)
        controller_state = state['controller_state']
        controller_out, new_controller_state = self.controller.forward_step(wide_input, controller_state)
        # controller_out: [batch, d_model]

        # 5) Generate output logits
        logits = self.token_head(controller_out)  # [batch, vocab_size]

        # 6) Generate read parameters and update read weights
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

            prev_w = state['read_w'][:, h, :]
            wt = self.addressing(key, beta, gate, shift, gamma, prev_w, memory)
            new_read_w.append(wt)

        state['read_w'] = torch.stack(new_read_w, dim=1)

        # 7) Generate write parameters and update memory
        write_params = self.write_head(controller_out)  # [batch, WH * write_param_len]
        write_params = write_params.view(batch, self.WH, self.write_param_len)

        mem = memory
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

            prev_w = state['write_w'][:, h, :]
            wt = self.addressing(key, beta, gate, shift, gamma, prev_w, mem)
            new_write_w.append(wt)

            # Erase and add operations
            erase_v = torch.sigmoid(erase)  # [batch, M]
            add_v = torch.tanh(add)  # [batch, M]

            erase_matrix = wt.unsqueeze(-1) * erase_v.unsqueeze(1)  # [batch, N, M]
            mem = mem * (1.0 - erase_matrix)

            add_matrix = wt.unsqueeze(-1) * add_v.unsqueeze(1)  # [batch, N, M]
            mem = mem + add_matrix

        state['memory'] = mem
        state['write_w'] = torch.stack(new_write_w, dim=1)
        state['controller_state'] = new_controller_state

        return logits, state

    def forward(self,
                token_seq: torch.Tensor,
                state: Optional[Dict[str, Any]] = None,
                return_all_logits: bool = False):
        """
        Process full token sequence

        token_seq: [seq_len, batch] long tensor
        returns: logits (all or last), final_state
        """
        seq_len, batch = token_seq.shape
        device = token_seq.device

        if state is None:
            state = self.init_state(batch_size=batch, device=device)

        logits_all = []
        for t in range(seq_len):
            x_t = token_seq[t]
            logits, state = self.step(x_t, state)
            if return_all_logits:
                logits_all.append(logits.unsqueeze(0))

        if return_all_logits:
            return torch.cat(logits_all, dim=0), state
        else:
            return logits, state

    def get_memory_usage_stats(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Diagnostic statistics for monitoring memory systems"""
        stats = {}

        # KV cache usage
        kv = state['controller_state'].get('kv_cache')
        stats['kv_cache_len'] = kv.size(0) if kv is not None else 0
        stats['kv_cache_max'] = self.controller_window
        stats['kv_utilization'] = stats['kv_cache_len'] / self.controller_window

        # External memory statistics
        memory = state['memory']  # [batch, N, M]
        stats['memory_mean'] = memory.mean().item()
        stats['memory_std'] = memory.std().item()
        stats['memory_abs_max'] = memory.abs().max().item()
        stats['memory_sparsity'] = (memory.abs() < 0.01).float().mean().item()

        # Attention sharpness (lower entropy = sharper)
        read_w = state['read_w']  # [batch, RH, N]
        write_w = state['write_w']  # [batch, WH, N]

        eps = 1e-12
        read_entropy = -(read_w * (read_w + eps).log()).sum(-1).mean().item()
        write_entropy = -(write_w * (write_w + eps).log()).sum(-1).mean().item()

        stats['read_entropy'] = read_entropy
        stats['write_entropy'] = write_entropy
        stats['read_sharpness'] = math.log(self.N) - read_entropy  # max_entropy - actual
        stats['write_sharpness'] = math.log(self.N) - write_entropy

        return stats


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*70)
    print("RTNTM - Wide Input Architecture")
    print("="*70)
    print(f"Device: {device}\n")

    # Configuration
    memory_N = 128
    memory_M = vocab_size  # Can differ from d_model now
    d_model = 256
    n_heads = 4
    controller_window = 8
    read_heads = 1

    print("Architecture:")
    print(f"  Token embedding: vocab_size → d_model ({d_model})")
    print(f"  Read vectors: {read_heads} heads × M ({memory_M}) = {read_heads * memory_M}")
    print(f"  Wide input: d_model + RH*M = {d_model} + {read_heads * memory_M} = {d_model + read_heads * memory_M}")
    print(f"  Compressed to: d_model ({d_model}) for attention")
    print(f"  KV cache: bounded to {controller_window} tokens")
    print(f"  External memory: {memory_N} × {memory_M}")
    print()

    model = RTNTM(
        vocab_size=vocab_size,
        d_model=d_model,
        memory_N=memory_N,
        memory_M=memory_M,
        n_heads=n_heads,
        controller_window=controller_window,
        read_heads=read_heads,
        write_heads=1,
        shift_width=3,
        use_summary=False
    ).to(device)

    total_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}\n")

    # Test sequence
    batch = 2
    seq_len = 25

    torch.manual_seed(42)
    token_seq = torch.randint(0, vocab_size, (seq_len, batch), device=device)

    print(f"Testing sequence: length={seq_len}, batch={batch}")
    print(f"Sequence length > controller_window ({controller_window})")
    print("Model MUST use external memory!\n")

    # Process step by step
    state = model.init_state(batch_size=batch, device=device)

    print("Step-by-step execution:")
    print("-" * 70)

    for t in range(seq_len):
        x_t = token_seq[t]
        logits, state = model.step(x_t, state)

        if t % 5 == 0 or t == seq_len - 1:
            stats = model.get_memory_usage_stats(state)
            print(f"t={t:2d}: KV={stats['kv_cache_len']}/{stats['kv_cache_max']} "
                  f"(util={stats['kv_utilization']:.1%}), "
                  f"Mem_std={stats['memory_std']:.4f}, "
                  f"Mem_sparsity={stats['memory_sparsity']:.2%}, "
                  f"Read_sharp={stats['read_sharpness']:.2f}, "
                  f"Write_sharp={stats['write_sharpness']:.2f}")

    print("\n" + "="*70)
    print("Key Observations:")
    print("  ✓ Wide input preserves full information (no premature compression)")
    print("  ✓ KV cache saturates at window size (forces external memory)")
    print("  ✓ Memory std should increase as model writes data")
    print("  ✓ Sharp attention (high sharpness) indicates focused read/write")
    print("="*70)
