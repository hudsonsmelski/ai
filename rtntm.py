"""
2014 Neural Turing Machines,
     Alex Graves, gravesa@google.com
     Greg Wayne, gregwayne@google.com
     Ivo Danihelka, danihelka@google.com

Refactored Transformer-NTM (TNTM) - step-based & recurrent controller
- Step API: state = tntm.init_state(batch, device); logits, state = tntm.step(x_t, state)
- Forward API: logits_seq, final_state = tntm.forward(token_seq, state=None)
- Implements content + location addressing per Graves (NTM), with circular shift.
- Controller is a recurrent Transformer-like single-layer cell with KV cache.

Hudson Andrew Smelski
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
# Positional encoding (optional usage)
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        # x: [seq_len, batch, d_model]
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# ---------------------------
# Recurrent Transformer-style controller
# ---------------------------
class RecurrentTransformerController(nn.Module):
    """
    A single-layer Transformer-style controller that is recurrent via a KV cache.
    forward_step(token_emb, read_input, state) -> (out_vec, new_state)
    - token_emb: [batch, d_model]
    - read_input: [batch, read_input_dim]  (we'll project into d_model and add)
    - state['kv_cache']: Optional tensor [past_len, batch, d_model]
    """
    def __init__(self, d_model: int, n_heads: int, read_input_dim: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        # project read vectors into d_model and combine with token embedding
        self.read_proj = nn.Linear(read_input_dim, d_model)
        self.input_proj = nn.Linear(d_model, d_model)

        # MultiheadAttention expects [seq_len, batch, d_model]; we use seq_len=1 per step
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=False)

        # Feed-forward with layer norms (Transformer-like)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward_step(self, token_emb: torch.Tensor, read_input: torch.Tensor, ctl_state: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        token_emb: [batch, d_model]
        read_input: [batch, read_input_dim]
        ctl_state: dict possibly containing 'kv_cache' (tensor [past_len, batch, d_model]) and 'max_cache_len'
        returns: out [batch, d_model], new_ctl_state
        """
        batch = token_emb.shape[0]
        device = token_emb.device

        # project and combine
        read_p = self.read_proj(read_input)  # [batch, d_model]
        x = token_emb + read_p
        x = self.input_proj(x)

        # build q, k, v
        q = x.unsqueeze(0)  # [1, batch, d_model]
        kv_cache: Optional[torch.Tensor] = ctl_state.get('kv_cache', None)  # [past_len, batch, d_model] or None

        if kv_cache is not None:
            k = torch.cat([kv_cache, q], dim=0)  # [past_len+1, batch, d_model]
            v = torch.cat([kv_cache, q], dim=0)
        else:
            k = q
            v = q

        # MultiheadAttention: query=q, key=k, value=v
        attn_out, _ = self.mha(q, k, v, need_weights=False)  # attn_out: [1, batch, d_model]
        attn_out = attn_out.squeeze(0)  # [batch, d_model]

        # residual + norms + ff
        x2 = self.ln1(x + attn_out)
        ff_out = self.ff(x2)
        out = self.ln2(x2 + ff_out)

        # update kv_cache: append current q (detach to prevent huge graphs for long runs if desired)
        max_cache_len = ctl_state.get('max_cache_len', 256)
        if kv_cache is None:
            new_cache = q.detach().clone()
        else:
            new_cache = torch.cat([kv_cache, q.detach().clone()], dim=0)
            if new_cache.size(0) > max_cache_len:
                new_cache = new_cache[-max_cache_len:]

        new_state = dict(ctl_state)
        new_state['kv_cache'] = new_cache

        return out, new_state


# ---------------------------
# TNTM main model (refactor)
# ---------------------------
class RTNTM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 memory_N: int,
                 memory_M: int,
                 n_heads: int = 8,
                 n_layers: int = 1,  # controller depth optional (we implement single-layer recurrent cell)
                 read_heads: int = 1,
                 write_heads: int = 1,
                 shift_width: int = 3):
        """
        d_model: Transformer dimension
        memory_N: number of memory rows (locations)
        memory_M: width of each memory row
        read_heads / write_heads: integers
        shift_width: odd integer (e.g. 3) for allowed relative shifts (-1,0,1)
        """
        super().__init__()

        assert shift_width % 2 == 1, "shift_width must be odd (e.g. 3)"
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.N = memory_N
        self.M = memory_M
        self.RH = read_heads
        self.WH = write_heads
        self.shift_K = shift_width
        self.half_shift = shift_width // 2

        # token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # controller: read inputs concatenated (RH * M) projected inside controller
        self.controller = RecurrentTransformerController(d_model=d_model, n_heads=n_heads, read_input_dim=self.RH * self.M)

        # output head: maps controller output -> token logits
        self.token_head = nn.Linear(d_model, vocab_size)

        # addressing parameter heads (emit raw params; activations applied downstream)
        # read: per head -> key(M) + beta(1) + gate(1) + shift(K) + gamma(1)
        self.read_param_len = self.M + 1 + 1 + self.shift_K + 1
        self.read_head = nn.Linear(d_model, self.RH * self.read_param_len)

        # write: key(M) + beta(1) + gate(1) + shift(K) + gamma(1) + erase(M) + add(M)
        self.write_param_len = self.M + 1 + 1 + self.shift_K + 1 + self.M + self.M
        self.write_head = nn.Linear(d_model, self.WH * self.write_param_len)

        # initial small memory buffer
        self.register_buffer('memory_initial', torch.randn(self.N, self.M) * 0.01)

    # ---------------------------
    # Utility: initialize per-batch state
    # ---------------------------
    def init_state(self, batch_size: int = 1, device: Optional[torch.device] = None) -> Dict[str, Any]:
        if device is None:
            device = torch.device('cpu')
        state: Dict[str, Any] = {}
        # memory: [batch, N, M]
        state['memory'] = self.memory_initial.unsqueeze(0).repeat(batch_size, 1, 1).to(device).clone()
        # read/write weights: normalized; shapes [batch, RH, N] and [batch, WH, N]
        state['read_w'] = F.softmax(torch.randn(batch_size, self.RH, self.N, device=device), dim=-1)
        state['write_w'] = F.softmax(torch.randn(batch_size, self.WH, self.N, device=device), dim=-1)
        # controller state
        state['controller_state'] = {'kv_cache': None, 'max_cache_len': 256}
        return state

    # ---------------------------
    # Addressing: vectorized batch implementation
    # params: tuple of tensors (key, beta, gate, shift, gamma) each with leading batch dim
    # prev_w: [batch, N]
    # memory: [batch, N, M]
    # returns: wt [batch, N]
    # ---------------------------
    def addressing(self, key: torch.Tensor, beta: torch.Tensor, gate: torch.Tensor, shift: torch.Tensor, gamma: torch.Tensor,
                   prev_w: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        key: [batch, M]
        beta: [batch, 1]
        gate: [batch, 1]
        shift: [batch, K]
        gamma: [batch, 1]
        prev_w: [batch, N]
        memory: [batch, N, M]
        returns wt: [batch, N]
        """
        eps = 1e-12
        batch = memory.size(0)
        N = memory.size(1)

        # 1) content addressing (cosine similarity)
        # compute cosine similarity between key and each memory row
        # memory: [batch, N, M], key: [batch, M] -> produce [batch, N]
        key_norm = key / (key.norm(dim=-1, keepdim=True) + eps)
        mem_norm = memory / (memory.norm(dim=-1, keepdim=True) + eps)
        cos_sim = torch.bmm(mem_norm, key_norm.unsqueeze(-1)).squeeze(-1)  # [batch, N]

        beta_pos = F.softplus(beta).squeeze(-1)  # [batch]
        wc = F.softmax(beta_pos.unsqueeze(-1) * cos_sim, dim=-1)  # [batch, N]

        # 2) interpolation gate
        g = torch.sigmoid(gate).squeeze(-1)  # [batch]
        wg = g.unsqueeze(-1) * wc + (1.0 - g).unsqueeze(-1) * prev_w  # [batch, N]

        # 3) circular convolutional shift
        s = F.softmax(shift, dim=-1)  # [batch, K]
        # implement as weighted sum of rolled wg
        # shifts correspond to [-half_shift, ..., 0, ..., +half_shift]
        shifted = torch.zeros_like(wg)
        for k in range(self.shift_K):
            shift_amount = k - self.half_shift
            rolled = torch.roll(wg, shifts=shift_amount, dims=-1)
            shifted = shifted + s[:, k].unsqueeze(-1) * rolled

        # 4) sharpening
        gamma_p = 1.0 + F.softplus(gamma).squeeze(-1)  # [batch]
        wt = shifted.clamp(min=eps) ** gamma_p.unsqueeze(-1)
        wt = wt / (wt.sum(dim=-1, keepdim=True) + eps)

        return wt

    # ---------------------------
    # Single time-step update
    # x_t: [batch] token indices (long) OR [batch, d_model] embeddings precomputed
    # state: dict with memory (batch,N,M), read_w (batch,RH,N), write_w (batch,WH,N), controller_state
    # returns: logits [batch, vocab], new_state
    # ---------------------------
    def step(self, x_t: torch.Tensor, state: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Execute a single step of the NTM.
        x_t: [batch] long tensor of token indices
        state: dictionary (see init_state)
        """
        batch = x_t.shape[0] if x_t.dim() > 0 else 1
        device = x_t.device

        # 1) get token embedding
        emb = self.embedding(x_t)  # [batch, d_model]

        # 2) compute read vectors from memory (batched)
        memory = state['memory']  # [batch, N, M]
        read_w = state['read_w']  # [batch, RH, N]
        read_vecs = []
        for h in range(self.RH):
            # w: [batch, N], memory: [batch, N, M] -> read: [batch, M]
            w = read_w[:, h, :].unsqueeze(1)  # [batch, 1, N]
            r = torch.bmm(w, memory).squeeze(1)  # [batch, M]
            read_vecs.append(r)
        # concatenate reads: [batch, RH * M]
        read_concat = torch.cat(read_vecs, dim=-1) if self.RH > 1 else read_vecs[0]

        # 3) controller step: provide embedding + read_concat
        controller_state = state['controller_state']
        controller_out, new_controller_state = self.controller.forward_step(emb, read_concat, controller_state)  # out: [batch, d_model]

        # 4) produce logits
        logits = self.token_head(controller_out)  # [batch, vocab_size]

        # 5) produce read parameters and update read weights
        read_params = self.read_head(controller_out)  # [batch, RH * read_param_len]
        read_params = read_params.view(batch, self.RH, self.read_param_len)  # [batch, RH, L]
        new_read_w = []
        for h in range(self.RH):
            rp = read_params[:, h, :]  # [batch, read_param_len]
            # split: key(M), beta(1), gate(1), shift(K), gamma(1)
            idx = 0
            key = rp[:, idx:idx + self.M]; idx += self.M
            beta = rp[:, idx:idx + 1]; idx += 1
            gate = rp[:, idx:idx + 1]; idx += 1
            shift = rp[:, idx:idx + self.shift_K]; idx += self.shift_K
            gamma = rp[:, idx:idx + 1]; idx += 1

            prev_w = state['read_w'][:, h, :]  # [batch, N]
            wt = self.addressing(key, beta, gate, shift, gamma, prev_w, memory)  # [batch, N]
            new_read_w.append(wt)
        # new_read_w: list of [batch, N] -> stack to [batch, RH, N]
        state['read_w'] = torch.stack(new_read_w, dim=1)

        # 6) produce write parameters and update memory
        write_params = self.write_head(controller_out)  # [batch, WH * write_param_len]
        write_params = write_params.view(batch, self.WH, self.write_param_len)  # [batch, WH, L]
        mem = memory
        new_write_w_list = []
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

            prev_w = state['write_w'][:, h, :]  # [batch, N]
            wt = self.addressing(key, beta, gate, shift, gamma, prev_w, mem)  # [batch, N]
            new_write_w_list.append(wt)

            # Erase: erase vector in (0,1)
            erase_v = torch.sigmoid(erase)  # [batch, M]
            # Add: bound with tanh and small scale for stability
            add_v = torch.tanh(add) * 0.1  # [batch, M]

            # outer products to apply per-location effects
            # wt: [batch, N], erase_v: [batch, M] -> erase_matrix: [batch, N, M]
            erase_matrix = wt.unsqueeze(-1) * erase_v.unsqueeze(1)  # [batch, N, M]
            mem = mem * (1.0 - erase_matrix)

            add_matrix = wt.unsqueeze(-1) * add_v.unsqueeze(1)  # [batch, N, M]
            mem = mem + add_matrix

        # update state memory and write_w
        state['memory'] = mem
        state['write_w'] = torch.stack(new_write_w_list, dim=1)  # [batch, WH, N]

        # update controller state
        state['controller_state'] = new_controller_state

        return logits, state

    # ---------------------------
    # Forward wrapper: process a full token sequence (seq_len, batch)
    # Returns logits per step optionally, and final state
    # ---------------------------
    def forward(self, token_seq: torch.Tensor, state: Optional[Dict[str, Any]] = None, return_all_logits: bool = False):
        """
        token_seq: [seq_len, batch] long tensor of token indices
        If state is None, initialize with batch=token_seq.shape[1]
        """
        seq_len, batch = token_seq.shape
        device = token_seq.device
        if state is None:
            state = self.init_state(batch_size=batch, device=device)

        logits_all = []
        for t in range(seq_len):
            x_t = token_seq[t]  # [batch]
            logits, state = self.step(x_t, state)
            if return_all_logits:
                logits_all.append(logits.unsqueeze(0))

        if return_all_logits:
            logits_all = torch.cat(logits_all, dim=0)  # [seq_len, batch, vocab]
            return logits_all, state
        else:
            return logits, state


# ---------------------------
# Utility: parameter counting
# ---------------------------
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------
# Small demo / test in __main__
# ---------------------------
if __name__ == "__main__":
    # Basic settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model hyperparameters (start small for testing)
    memory_N = 32
    memory_M = 32
    d_model = 64
    n_heads = 4
    read_heads = 2
    write_heads = 1
    shift_K = 3

    model = RTNTM(vocab_size=vocab_size,
                 d_model=d_model,
                 memory_N=memory_N,
                 memory_M=memory_M,
                 n_heads=n_heads,
                 n_layers=1,
                 read_heads=read_heads,
                 write_heads=write_heads,
                 shift_width=shift_K).to(device)

    # Print parameter count and basic info
    total_params = count_parameters(model)
    print(f"Model total trainable parameters: {total_params:,}")
    print("Model summary (selected):")
    print(f" d_model={d_model}, memory N={memory_N}, M={memory_M}, read_heads={read_heads}, write_heads={write_heads}")

    # Create a tiny batch of random token sequences for testing copy-like behavior
    batch = 1
    seq_len = 10
    # random tokens from vocab
    torch.manual_seed(0)
    token_seq = torch.randint(low=0, high=vocab_size, size=(seq_len, batch), dtype=torch.long, device=device)

    # Run forward wrapper (unrolled)
    logits_seq, final_state = model.forward(token_seq, state=None, return_all_logits=True)
    print(f"Processed sequence of length {seq_len} (batch={batch}).")
    print(f"Logits_seq shape: {logits_seq.shape} (seq_len, batch, vocab)")

    # Show some diagnostic info: memory norm, sample read weights
    memory = final_state['memory']  # [batch, N, M]
    read_w = final_state['read_w']  # [batch, RH, N]
    write_w = final_state['write_w']  # [batch, WH, N]

    print("Final memory stats per batch (mean, std):")
    mem_means = memory.mean(dim=[1, 2]).detach().cpu().numpy()
    mem_stds = memory.std(dim=[1, 2]).detach().cpu().numpy()
    for b in range(batch):
        print(f" batch {b}: mean={mem_means[b]:.6f}, std={mem_stds[b]:.6f}")

    # Print read/write weight example (first head)
    print("Sample read weights (first head) [batch, N]:")
    print(read_w[:, 0, :].detach().cpu().numpy())

    print("Sample write weights (first head) [batch, N]:")
    print(write_w[:, 0, :].detach().cpu().numpy())

    # Convert the last-step predictions to chars (argmax)
    last_logits = logits_seq[-1, 0, :].cpu()
    pred_idx = torch.argmax(last_logits).item()
    pred_char = idx_to_char[pred_idx]
    print(f"Last step prediction (argmax): index={pred_idx}, char={repr(pred_char)}")

    # Also demonstrate step-by-step API
    state = model.init_state(batch_size=1, device=device)
    print("\nDemonstrating step API:")
    for t in range(seq_len):
        x_t = token_seq[t]
        logits, state = model.step(x_t, state)
        pred = torch.argmax(logits, dim=-1).item()
        print(f" t={t:02d} input_idx={x_t.item():3d} pred_idx={pred:3d} char={repr(idx_to_char[pred])}")

    print("\nDemo complete.")
