"""
2016 Hybrid computing using a neural network with dynamic external memory
Alex Graves1*, Greg Wayne1*, Malcolm Reynolds1, Tim Harley1, Ivo Danihelka1, Agnieszka Grabska-Barwińska1,
Sergio Gómez Colmenarejo1, Edward Grefenstette1, Tiago Ramalho 1, John Agapiou1, Adrià Puigdomènech Badia1,
Karl Moritz Hermann1, Yori Zwols1, Georg Ostrovski1, Adam Cain1, Helen King1, Christopher Summerfield1, Phil Blunsom1, Koray Kavukcuoglu1 & Demis Hassabis1

Hudson Andrew Smelski

DNC architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import string

class DNC(nn.Module):
    def __init__(self, input_size, memory_length,
        controller_depth, controller_width,
        read_heads, write_heads):
        super().__init__()
        self.RH = read_heads
        self.WH = write_heads
        self.N = memory_length
        self.M = input_size
        self.input_size = input_size

        self.read_param_len = self.M + 1 + 3
        self.write_param_len = self.M + 1 + self.M + self.M + self.RH + 1 + 1
        self.controller_width = controller_width
        self.controller_depth = controller_depth
        self.controller_input = self.M + self.M*self.RH

        #self.embedding = nn.Linear(self.input_size, self.M)
        self.embedding = nn.Identity()

        self.controller = nn.GRU(self.controller_input, self.controller_width, self.controller_depth, batch_first=False)
        self.read_head = nn.Linear(self.controller_width, self.RH * self.read_param_len)
        self.write_head = nn.Linear(self.controller_width, self.WH * self.write_param_len)

        self.read_matrix = nn.Linear(self.RH * self.M, self.input_size)
        self.out = nn.Linear(self.controller_width, self.input_size)

        #self.register_buffer('memory_initial', torch.zeros(self.N, self.M))
        self.register_buffer('memory_initial', torch.randn(self.N, self.M) * 0.01)
        self.reset()

    def reset(self, batch_size=1):
        self.batch_size = batch_size

        self.memory = self.memory_initial.unsqueeze(0).repeat(batch_size, 1, 1).clone()
        self.link_matrix = torch.zeros(batch_size, self.N, self.N)
        self.precedence = torch.zeros(batch_size, self.N)
        #self.usage = torch.zeros(batch_size, self.N)
        self.usage = torch.rand(batch_size, self.N) * 0.01

        init_w = torch.zeros(batch_size, self.N)
        init_w[:, 0] = 1.0
        self.read_w = init_w.unsqueeze(1).repeat(1, self.RH, 1)
        self.write_w = init_w.unsqueeze(1).repeat(1, self.WH, 1)
        self.hidden = torch.zeros(self.controller_depth, batch_size, self.controller_width)
        self.read_vecs = torch.zeros(self.batch_size, self.RH*self.M)

    def forward(self, x):
        xe = self.embedding(x)
        ci = torch.cat((xe, self.read_vecs), dim = 1).unsqueeze(0)
        o, self.hidden = self.controller(ci, self.hidden)
        o = o[-1,:,:]

        rh_params = self.read_head(o)
        new_read_w = []
        for h in range(self.RH):
            params = rh_params[:, h * self.read_param_len:(h + 1) * self.read_param_len]
            wt = self.addressing(params, self.read_w[:, h, :], self.memory, self.link_matrix, mode='read')
            new_read_w.append(wt)
        self.read_w = torch.stack(new_read_w, dim=1)  # [batch, RH, N]

        read_vecs = []
        for h in range(self.RH):
            w = self.read_w[:, h, :].unsqueeze(1)
            r = torch.bmm(w, self.memory).squeeze(1)
            read_vecs.append(r)
        self.read_vecs = torch.cat(read_vecs, dim = 1) # [batch, M*RH]

        wh_params = self.write_head(o)
        new_memory, write_w, new_usage, new_link_matrix, new_precedence = \
            self.write_head_logic(wh_params, self.memory, self.usage,
                                 self.link_matrix, self.precedence, self.read_w)

        self.memory = new_memory#.detach()
        self.usage = new_usage#.detach()
        self.link_matrix = new_link_matrix#.detach()
        self.precedence = new_precedence#.detach()
        self.write_w = write_w.unsqueeze(1)#.detach()

        y = self.out(o)
        return y

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
        new_usage = torch.clamp(new_usage, 0, 1)

        # 4. Allocation weighting
        # Sort by usage (ascending - least used first)
        sorted_usage, free_list = torch.sort(new_usage, dim=-1, descending=False)

        # Calculate allocation weights using equation (1) from paper
        # a_t[φ_t[j]] = (1 - u_t[φ_t[j]]) * ∏_{i=1}^{j-1} u_t[φ_t[i]]
        cumprod = torch.cumprod(sorted_usage, dim=-1)
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

        # Controller hidden state
        stats['hidden_mean'] = self.hidden.mean().item()
        stats['hidden_std'] = self.hidden.std().item()

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)
    print(f"Using device: {device}")

    # Define the set of valid ASCII characters to use as tokens
    vocab = string.digits + string.ascii_letters + string.punctuation + " \t\v\n\r\f"
    vocab_size = len(vocab)

    char_to_idx = {char: idx for idx, char in enumerate(vocab)}    #Create a dictionary that maps each character to a unique integer value
    idx_to_char = {idx: char for char, idx in char_to_idx.items()} #Create a reverse dictionary that maps each integer value to its corresponding character

    print(f"vocab size = {vocab_size}")

    memory_length = 128
    model = DNC(
        vocab_size, memory_length,
        controller_depth = 1, controller_width = 100,
        read_heads = 1, write_heads = 1)

    print(f"parameters      = {model.num_params():,}")
    print(f"memory shape    = {model.memory_initial.shape}")

    # Test with batch
    batch_size = 4
    model.reset(batch_size=batch_size)
    x = torch.randn(batch_size, vocab_size, device=device)
    y = model.forward(x)
    print(f"\nBatch output shape: {y.shape}")  # Should be [4, vocab_size]

