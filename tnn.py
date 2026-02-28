"""
2026 Tree Neural Network

The idea is to have nodes that see the input
and route the input to new nodes. By the end
log2(n) nodes see the input and produce the
output. This should compress the parameters
further, while allowing for tail distribution
problems to be have significant resources.
       (N0)
      /   \
    (N1)   (N2)

ACT on layers/nodes

TODO Transformer nodes?
TODO NTM Memory access as well

Implementation by Hudson Andrew Smelski
"""

import string
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim

class Node(nn.Module):
    def __init__(self, input_size, node_size, is_leaf=False):
        super().__init__()
        self.is_leaf = is_leaf
        self.node_size = node_size

        self.s     = torch.zeros(node_size)
        self.state = nn.GRUCell(input_size, node_size)
        self.out   = nn.Linear(node_size, node_size)
        self.head  = nn.Linear(node_size*2, node_size)
        self.halt  = nn.Linear(node_size, 1)

        if not self.is_leaf:
            self.router  = nn.Linear(node_size, 1)

        init.constant_(self.halt.bias, 1.0)

    def forward(self, x, v):
        xv = torch.cat([x, v], dim=-1)
        s_n = self.s #carry state over from previous steps.
        #s = torch.zeros(self.node_size)
        y = torch.zeros(self.node_size)
        halting_sum = torch.zeros(1)
        for n in range(4): #hardcoded max steps for now.
            s_n = self.state(xv, s_n)
            y_n = F.relu(self.out(s_n))
            p_halt = torch.sigmoid(self.halt(s_n))

            new_halting_sum = halting_sum + p_halt
            p_n = 1.0 - halting_sum if new_halting_sum >= 0.99 else p_halt
            y = y + (y_n * p_n) #build v_new candidate or modification
            #s = s + (s_n * p_n) #build new state self.s

            if new_halting_sum > 0.99:
                break

        self.s = s_n

        logit = self.router(y) if not self.is_leaf else None
        v_new = self.head(torch.cat([v, y]))

        return logit, v_new

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class TNN(nn.Module):
    def __init__(self, input_size, node_size, output_size, depth):
        super(TNN, self).__init__()
        self.input_size = input_size
        self.node_size = node_size
        self.dual_size = input_size + node_size #[x,v]
        self.depth = depth

        # Recurrent Network layer(s)
        self.n_nodes = (1 << depth) - 1
        leaf_start = (1 << (depth - 1)) - 1
        self.nodes = nn.ModuleList([
            Node(self.dual_size, node_size, is_leaf=(i >= leaf_start)) for i in range(self.n_nodes)
        ])

        self.final_head = nn.Linear(node_size, output_size)

    def forward(self, x):
        L, _  = x.shape
        y = []

        v = torch.zeros(self.node_size)
        for t in range(L):
            x_t = x[t, :]
            idx = 0
            v1 = v.clone()

            while True:
                node = self.nodes[idx]

                route, v1 = node(x_t, v1)

                if node.is_leaf:
                    break

                hard = (route > 0).float()
                soft = torch.sigmoid(route)
                direction_ste = hard + (soft - soft.detach())   # STE magic
                direction_int = int(direction_ste.item())
                idx = 2 * idx + 1 + direction_int

            y.append(self.final_head(v1))
        y = torch.cat(y, dim=0)
        return y

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

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

    node_size = 128
    depth = 5
    model = TNN(vocab_size, node_size, vocab_size, depth)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)

    params = model.num_params()
    inference_params = model.depth * model.nodes[0].num_params()
    ratio = inference_params/params
    print(f"depth            = {model.depth}")
    print(f"nodes            = {model.n_nodes}")
    print(f"node parameters  = {model.nodes[0].num_params()}")
    print(f"parameters       = {model.num_params()}")
    print(f"inference params = {inference_params}")
    print(f"ratio            = {ratio}")

    x = torch.randn(1, vocab_size, device=device)
    y = model(x)
    loss = y.mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    print(f"output shape: {y.shape}")
    print(f"loss: {loss}")
