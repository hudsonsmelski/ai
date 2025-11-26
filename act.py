"""
2016 Adaptive Computation Time for Recurrent Neural Networks,
     Alex Graves, Google DeepMind, gravesa@google.com

Implementation by Hudson Andrew Smelski
"""

import torch
import torch.nn as nn
import torch.nn.init as init

class ACT(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_type="RNN",
                 max_steps=100, tau=0.01, epsilon=0.01):
        super(ACT, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_steps = max_steps
        self.tau = tau
        self.epsilon = epsilon

        self.ilen = input_size + 1

        # State transition Module
        if hidden_type == "RNN":
            self.state = nn.RNNCell(self.ilen, self.hidden_size, bias=True)
            self.st = 0
        elif hidden_type == "LSTM":
            self.state = nn.LSTMCell(self.ilen, self.hidden_size, bias=True)
            self.st = 1

        self.halt = nn.Linear(self.hidden_size, 1)
        self.output = nn.Linear(self.hidden_size, self.output_size)

        self._init_weights()

    def _init_weights(self):
        init.constant_(self.halt.bias, 1.0)
        init.xavier_uniform_(self.halt.weight)
        init.xavier_uniform_(self.output.weight)

    def forward(self, input):
        seq_len, batch_size, _ = input.size()
        device = input.device

        # Initial states
        s_t = torch.zeros(batch_size, self.hidden_size, device=device)
        c_t = torch.zeros(batch_size, self.hidden_size, device=device) if self.st == 1 else None

        # Outputs
        y_ts = []
        p_ts = []  # Ponder costs
        n_ts = []  # Step counts

        for t in range(seq_len):
            x_t = input[t]

            y_t_accum = torch.zeros(batch_size, self.output_size, device=device)
            s_t_accum = torch.zeros(batch_size, self.hidden_size, device=device)
            c_t_accum = torch.zeros(batch_size, self.hidden_size, device=device) if self.st == 1 else None

            haltsum = torch.zeros(batch_size, device=device)
            running = torch.ones(batch_size, device=device) # Mask: 1=Running, 0=Halted

            total_steps = torch.zeros(batch_size, device=device)
            total_remainder = torch.zeros(batch_size, device=device)

            for n in range(self.max_steps):
                flag = torch.ones(batch_size, 1, device=device) if n == 0 else torch.zeros(batch_size, 1, device=device)
                x_n = torch.cat([x_t, flag], dim=1)

                if self.st == 0: # RNN
                    s_n = self.state(x_n, s_t)
                else: # LSTM
                    s_n, c_n = self.state(x_n, (s_t, c_t))

                h_n = torch.sigmoid(self.halt(s_n)).squeeze(-1)
                new_sum = haltsum + h_n
                stopping_now = (new_sum >= 1.0 - self.epsilon).float() * running
                still_running = running - stopping_now

                remainder = (1.0 - haltsum) * stopping_now
                p_n = (h_n * still_running) + remainder
                y_n = self.output(s_n)

                y_t_accum = y_t_accum + (y_n * p_n.unsqueeze(1))
                s_t_accum = s_t_accum + (s_n * p_n.unsqueeze(1))
                if self.st == 1:
                    c_t_accum = c_t_accum + (c_n * p_n.unsqueeze(1))

                haltsum = new_sum

                total_steps = total_steps + running
                total_remainder = total_remainder + remainder

                # Update state for next step
                s_t = s_n
                if self.st == 1:
                    c_t = c_n

                running = still_running

                if running.sum() == 0:
                    break

            y_ts.append(y_t_accum)

            # Store ponder cost: N(t) + R(t)
            # Note: total_steps tracks N(t), total_remainder tracks R(t)
            rho_t = total_steps + total_remainder
            p_ts.append(rho_t)
            n_ts.append(total_steps)

            # The accumulated state becomes the state for the next sequence step t+1
            s_t = s_t_accum
            if self.st == 1:
                c_t = c_t_accum

        return torch.stack(y_ts), torch.stack(p_ts), torch.stack(n_ts)

    def compute_loss(self, y_pred, y_target, ponder_cost, task_loss_fn):
        task_loss = task_loss_fn(y_pred, y_target)
        avg_ponder = ponder_cost.mean()
        total_loss = task_loss + self.tau * avg_ponder

        return total_loss, task_loss, avg_ponder

    def get_ponder_stats(self, ponder_cost, num_steps):
        return {
            'mean_ponder': ponder_cost.mean().item(),
            'max_ponder': ponder_cost.max().item(),
            'min_ponder': ponder_cost.min().item(),
            'mean_steps': num_steps.mean().item(),
            'max_steps': num_steps.max().item(),
            'min_steps': num_steps.min().item(),
        }
