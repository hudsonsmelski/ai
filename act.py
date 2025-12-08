"""
2016 Adaptive Computation Time for Recurrent Neural Networks,
     Alex Graves, Google DeepMind, gravesa@google.com

Implementation by Hudson Andrew Smelski
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class ACT(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_type="RNN",
                 max_steps=100, tau=0.01, epsilon=0.01, logit_smooth=False, dim1 = None, dim2 = None):
        super(ACT, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_steps = max_steps
        self.tau = tau
        self.epsilon = epsilon
        self.logit_smooth = logit_smooth
        self.dim1 = dim1
        self.dim2 = dim2

        self.ilen = input_size + 1

        # Recurrent Network layer(s)
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

    def forward(self, input):
        seq_len, batch_size, _ = input.size()
        device = input.device

        s_prev = torch.zeros(batch_size, self.hidden_size, device=device)
        c_prev = torch.zeros(batch_size, self.hidden_size, device=device) if self.st == 1 else None

        y_ts = []  # Outputs
        p_ts = []  # Ponder costs
        n_ts = []  # Step counts

        for t in range(seq_len):
            x_t = input[t]

            y_t_accum = torch.zeros(batch_size, self.output_size, device=device)
            s_t_accum = torch.zeros(batch_size, self.hidden_size, device=device)
            c_t_accum = torch.zeros(batch_size, self.hidden_size, device=device) if self.st == 1 else None

            #recurrent net state
            s_n = s_prev
            c_n = c_prev if self.st == 1 else None

            halting_sum = torch.zeros(batch_size, device=device)
            n_steps = torch.zeros(batch_size, device=device)
            remainders = torch.zeros(batch_size, device=device)
            still_running = torch.ones(batch_size, dtype=torch.bool, device=device)

            for n in range(self.max_steps):
                flag = torch.ones(batch_size, 1, device=device) if n == 0 else torch.zeros(batch_size, 1, device=device)
                x_n = torch.cat([x_t, flag], dim=1)

                if self.st == 0:  # RNN
                    s_n = self.state(x_n, s_n)
                else:  # LSTM
                    s_n, c_n = self.state(x_n, (s_n, c_n))

                if self.logit_smooth == False:
                    y_n = self.output(s_n)
                else:
                    y_n_logits = self.output(s_n)  # (batch, output_size)

                    # Reshape to (batch, num_digits, num_classes), apply softmax, reshape back
                    y_n_logits_reshaped = y_n_logits.view(batch_size, self.dim1, self.dim2)
                    y_n_probs = F.softmax(y_n_logits_reshaped, dim=-1)  # Softmax over classes
                    y_n = y_n_probs.view(batch_size, self.output_size)  # Flatten back to (batch, 66)

                # Halting unit
                h_n = torch.sigmoid(self.halt(s_n)).squeeze(-1)
                new_halting_sum = halting_sum + h_n

                # Determine halting probability for this step
                # If this step causes us to exceed threshold, use remainder
                # Otherwise, use h_n
                p_n = torch.where(new_halting_sum >= 1.0 - self.epsilon, 1.0 - halting_sum, h_n)

                # Only accumulate for samples still running
                p_n = p_n * still_running.float()

                # Accumulate outputs and states weighted by halting probability
                y_t_accum = y_t_accum + (y_n * p_n.unsqueeze(1))
                s_t_accum = s_t_accum + (s_n * p_n.unsqueeze(1))
                if self.st == 1:
                    c_t_accum = c_t_accum + (c_n * p_n.unsqueeze(1))

                # Update counters for samples still running
                n_steps = n_steps + still_running.float()

                # Track remainders (only for samples that halt this step)
                halted_this_step = (new_halting_sum >= 1.0 - self.epsilon) & still_running
                remainders = torch.where(halted_this_step, 1.0 - halting_sum, remainders)

                # Update halting sum and running mask
                halting_sum = new_halting_sum
                still_running = still_running & (new_halting_sum < 1.0 - self.epsilon)

                # Early stopping if all samples have halted
                if not still_running.any():
                    break

            if self.logit_smooth == False:
                y_ts.append(y_t_accum)
            else:
                y_t_accum_reshaped = y_t_accum.view(batch_size, self.dim1, self.dim2)
                y_t_logits = torch.log(y_t_accum_reshaped + 1e-10)
                y_t_logits_flat = y_t_logits.view(batch_size, self.output_size)

                y_ts.append(y_t_logits_flat)

            # Ponder cost: N(t) + R(t)
            rho_t = n_steps + remainders
            p_ts.append(rho_t)
            n_ts.append(n_steps)

            # The accumulated state becomes the state for the next timestep
            s_prev = s_t_accum
            if self.st == 1:
                c_prev = c_t_accum

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
