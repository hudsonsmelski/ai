"""
Training script for ACT on Logic Task
Based on Section 3.2 of the ACT paper
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import datetime
from pathlib import Path

from act import ACT

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Start time: {datetime.datetime.now()}")

# Binary logic truth tables (Table 1 from paper)
LOGIC_GATES = {
    0: lambda p, q: not (p or q),      # NOR
    1: lambda p, q: (not p) and q,     # Xq (inhibition)
    2: lambda p, q: p and (not q),     # ABJ (inhibition)
    3: lambda p, q: p != q,            # XOR
    4: lambda p, q: not (p and q),     # NAND
    5: lambda p, q: p and q,           # AND
    6: lambda p, q: p == q,            # XNOR
    7: lambda p, q: p or (not q),      # if/then
    8: lambda p, q: (not p) or q,      # then/if
    9: lambda p, q: p or q,            # OR
}

def apply_logic_gate(gate_idx, p, q):
    """Apply a logic gate given its index."""
    return LOGIC_GATES[gate_idx](p, q)

def generate_logic_sequence(batch_size, max_seq_length, max_gates, device):
    """
    Generate a batch of logic sequences.

    Args:
        batch_size: Number of sequences in batch
        max_seq_length: Maximum sequence length (1-10 in paper)
        max_gates: Maximum number of gates per vector (1-10 in paper)
        device: torch device

    Returns:
        X: Input tensor of shape (seq_len, batch, 102)
        Y: Target tensor of shape (seq_len, batch, 1)
    """
    # Random sequence length for each sample (1 to max_seq_length)
    seq_lengths = torch.randint(1, max_seq_length + 1, (batch_size,), device=device)
    actual_seq_len = seq_lengths.max().item()

    # Initialize tensors
    X = torch.zeros(actual_seq_len, batch_size, 102, device=device)
    Y = torch.zeros(actual_seq_len, batch_size, 1, device=device)

    for b in range(batch_size):
        seq_len = seq_lengths[b].item()

        # Initialize first two bits for the first vector
        b0 = torch.randint(0, 2, (1,), device=device).item() == 1
        b1 = torch.randint(0, 2, (1,), device=device).item() == 1

        for t in range(seq_len):
            # Random number of gates for this vector (1 to max_gates)
            num_gates = torch.randint(1, max_gates + 1, (1,), device=device).item()

            # Set the two input bits
            if t == 0:
                # First vector: set both b0 and b1
                X[t, b, 0] = float(b0)
                X[t, b, 1] = float(b1)
            else:
                # Subsequent vectors: b0 is implicitly the previous output (set to 0 in input)
                # b1 is random
                X[t, b, 0] = 0.0  # Always zero as per paper
                b1 = torch.randint(0, 2, (1,), device=device).item() == 1
                X[t, b, 1] = float(b1)

            # Generate random logic gates
            for g in range(num_gates):
                gate_idx = torch.randint(0, 10, (1,), device=device).item()
                # One-hot encode the gate in the appropriate chunk
                chunk_start = 2 + g * 10
                X[t, b, chunk_start + gate_idx] = 1.0

            # Compute target by recursively applying gates
            current_b0 = b0
            current_b1 = b1

            for g in range(num_gates):
                # Find which gate is active in this chunk
                chunk_start = 2 + g * 10
                gate_idx = torch.argmax(X[t, b, chunk_start:chunk_start + 10]).item()

                # Apply the gate
                result = apply_logic_gate(gate_idx, current_b1, current_b0)
                current_b0 = current_b1
                current_b1 = result

            # Store target
            Y[t, b, 0] = float(current_b1)

            # Update b0 for next iteration
            b0 = current_b1

    return X, Y, seq_lengths

def train_epoch(model, optimizer, criterion, tau, batch_size, max_seq_length,
                max_gates, num_batches, device):
    model.train()

    total_accuracy = 0
    total_sequence_accuracy = 0
    total_task_loss = 0
    total_total_loss = 0
    total_ponder = 0
    total_steps = 0
    total_samples = 0

    for _ in range(num_batches):
        # Generate batch
        X, Y, seq_lengths = generate_logic_sequence(batch_size, max_seq_length, max_gates, device)

        optimizer.zero_grad()

        # Forward pass
        output, ponder_costs, steps = model(X)

        # Apply sigmoid for binary classification
        output = torch.sigmoid(output)

        # Compute loss only on valid timesteps
        mask = torch.zeros_like(Y, dtype=torch.bool)
        for b in range(batch_size):
            mask[:seq_lengths[b], b, :] = True

        # Masked loss
        valid_output = output[mask]
        valid_target = Y[mask]

        task_loss = criterion(valid_output, valid_target)
        ponder_mean = ponder_costs.mean()
        total_loss = task_loss + tau * ponder_mean

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Track metrics
        predictions = (valid_output > 0.5).float()
        num_correct = (predictions == valid_target).sum().item()
        num_valid = valid_target.numel()

        total_accuracy += num_correct / num_valid
        total_task_loss += task_loss.item()
        total_total_loss += total_loss.item()
        total_ponder += ponder_mean.item()
        total_steps += steps.mean().item()
        total_samples += num_valid

        # Sequence accuracy (all predictions in sequence correct)
        for b in range(batch_size):
            seq_len = seq_lengths[b].item()
            seq_correct = True
            for t in range(seq_len):
                pred = (output[t, b, 0] > 0.5).float()
                target = Y[t, b, 0]
                if pred != target:
                    seq_correct = False
                    break
            if seq_correct:
                total_sequence_accuracy += 1

    total_sequences = num_batches * batch_size

    return {
        'accuracy': total_accuracy / num_batches,
        'sequence_accuracy': total_sequence_accuracy / total_sequences,
        'task_loss': total_task_loss / num_batches,
        'total_loss': total_total_loss / num_batches,
        'ponder': total_ponder / num_batches,
        'steps': total_steps / num_batches
    }

def evaluate(model, criterion, tau, batch_size, max_seq_length, max_gates,
             num_batches, device):
    model.eval()

    total_accuracy = 0
    total_sequence_accuracy = 0
    total_task_loss = 0
    total_ponder = 0
    total_steps = 0

    with torch.no_grad():
        for _ in range(num_batches):
            # Generate batch
            X, Y, seq_lengths = generate_logic_sequence(batch_size, max_seq_length, max_gates, device)

            # Forward pass
            output, ponder_costs, steps = model(X)
            output = torch.sigmoid(output)

            # Compute loss with masking
            mask = torch.zeros_like(Y, dtype=torch.bool)
            for b in range(batch_size):
                mask[:seq_lengths[b], b, :] = True

            valid_output = output[mask]
            valid_target = Y[mask]

            task_loss = criterion(valid_output, valid_target)
            ponder_mean = ponder_costs.mean()

            # Per-bit accuracy
            predictions = (valid_output > 0.5).float()
            num_correct = (predictions == valid_target).sum().item()
            num_valid = valid_target.numel()

            total_accuracy += num_correct / num_valid
            total_task_loss += task_loss.item()
            total_ponder += ponder_mean.item()
            total_steps += steps.mean().item()

            # Sequence accuracy (all predictions in sequence correct)
            for b in range(batch_size):
                seq_len = seq_lengths[b].item()
                seq_correct = True
                for t in range(seq_len):
                    pred = (output[t, b, 0] > 0.5).float()
                    target = Y[t, b, 0]
                    if pred != target:
                        seq_correct = False
                        break
                if seq_correct:
                    total_sequence_accuracy += 1

    total_sequences = num_batches * batch_size

    return {
        'accuracy': total_accuracy / num_batches,
        'sequence_accuracy': total_sequence_accuracy / total_sequences,
        'task_loss': total_task_loss / num_batches,
        'ponder': total_ponder / num_batches,
        'steps': total_steps / num_batches
    }

if __name__ == "__main__":
    # Hyperparameters (Section 3.2)
    input_size = 102  # 2 bits + 10 gates * 10 one-hot
    output_size = 1   # Single binary output
    hidden_size = 300  # LSTM with 128 cells (as per paper)
    hidden_type = "LSTM"
    max_steps = 100
    lr = 1e-3
    tau = 0.01

    # Training parameters
    batch_size = 16
    max_seq_length = 10  # 1-10 vectors per sequence
    max_gates = 10  # 1-10 gates per vector

    num_iterations = 100000  # 1M iterations / 16 threads ≈ 60K per thread
    train_batches = 10
    eval_batches = 20
    eval_interval = 100
    target_sequence_accuracy = 0.95

    # Curriculum learning
    current_max_gates = 1
    current_max_seq_len = 1
    curriculum_threshold = 0.90

    print("=" * 80)
    print("Logic Task - ACT Training")
    print("=" * 80)
    print(f"Input size: {input_size}")
    print(f"Hidden size: {hidden_size}")
    print(f"Hidden type: {hidden_type}")
    print(f"Batch size: {batch_size}")
    print(f"Max sequence length: {max_seq_length}")
    print(f"Max gates per vector: {max_gates}")
    print(f"Learning rate: {lr}")
    print(f"Time penalty (tau): {tau}")
    print(f"Max steps: {max_steps}")
    print(f"Target sequence accuracy: {target_sequence_accuracy}")
    print("=" * 80)

    model = ACT(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        hidden_type=hidden_type,
        max_steps=max_steps,
        tau=tau
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Configuration:")
    print(f"  Total parameters: {total_params:,}")
    print()

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10,
        verbose=True, threshold=0.01, min_lr=1e-6)

    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)

    best_sequence_accuracy = 0
    start_time = time.time()
    epoch_time = 0
    should_eval = False

    print(f"{'='*80}")
    print(f"Starting with max_gates = {current_max_gates} | seq_length = {current_max_seq_len}")
    print(f"{'='*80}")

    for iteration in range(num_iterations):
        epoch_start = time.time()
        train_metrics = train_epoch(
            model, optimizer, criterion, tau, batch_size,
            current_max_seq_len, current_max_gates, train_batches, device
        )
        epoch_time += time.time() - epoch_start

        if iteration % 10 == 0:
            print(f"Iter {iteration:6d} | "
                  f"Acc: {train_metrics['accuracy']:.3f} | "
                  f"Seq Acc: {train_metrics['sequence_accuracy']:.3f} | "
                  f"Loss: {train_metrics['task_loss']:.4f} | "
                  f"Ponder: {train_metrics['ponder']:.2f} | "
                  f"Steps: {train_metrics['steps']:.1f} | "
                  f"Time: {epoch_time:.2f}s | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")
            epoch_time = 0

        # Curriculum: gradually increase complexity
        if (train_metrics['sequence_accuracy'] >= curriculum_threshold
           and current_max_seq_len < max_seq_length):
            current_max_seq_len += 1
            print(f"{'='*80}")
            print(f"Increasing difficulty: max_gates = {current_max_gates} | seq_length = {current_max_seq_len}")
            print(f"{'='*80}")
        if (train_metrics['accuracy'] >= curriculum_threshold
           and current_max_seq_len == max_seq_length
           and current_max_gates < max_gates):
                current_max_gates += 1
                current_max_seq_len = 1
                print(f"{'='*80}")
                print(f"Increasing difficulty: max_gates = {current_max_gates} | seq_length = {current_max_seq_len}")
                print(f"{'='*80}")
                should_eval = True

        # Periodic evaluation
        if (iteration + 1) % eval_interval == 0 or should_eval:
            should_eval = False

            eval_start = time.time()
            eval_metrics = evaluate(
                model, criterion, tau, batch_size,
                max_seq_length, max_gates, eval_batches, device
            )
            eval_time = time.time() - eval_start

            print(f"{'='*80}")
            print(f"EVAL {iteration:6d} | "
                  f"Bit Acc: {eval_metrics['accuracy']:.3f} | "
                  f"Seq Acc: {eval_metrics['sequence_accuracy']:.3f} | "
                  f"Loss: {eval_metrics['task_loss']:.4f} | "
                  f"Ponder: {eval_metrics['ponder']:.2f} | "
                  f"Steps: {eval_metrics['steps']:.1f} | "
                  f"Time: {eval_time:.2f}s | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"{'='*80}")
            scheduler.step(eval_metrics['sequence_accuracy'])

            if eval_metrics['sequence_accuracy'] > best_sequence_accuracy:
                best_sequence_accuracy = eval_metrics['sequence_accuracy']
                save_path = save_dir / f"act_{hidden_type}_logic_best.pt"
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'sequence_accuracy': best_sequence_accuracy,
                    'config': {
                        'input_size': input_size,
                        'hidden_size': hidden_size,
                        'hidden_type': hidden_type,
                        'tau': tau,
                        'lr': lr
                    }
                }, save_path)
                print(f"✓ Saved best model with sequence accuracy: {best_sequence_accuracy:.3f}")

            # Check if target reached
            if eval_metrics['sequence_accuracy'] >= target_sequence_accuracy:
                print(f"\n{'='*80}")
                print(f"✓ Target sequence accuracy {target_sequence_accuracy:.3f} reached!")
                print(f"Final sequence accuracy: {eval_metrics['sequence_accuracy']:.3f}")
                print(f"Total training time: {(time.time() - start_time)/60:.2f} minutes")
                print(f"{'='*80}")
                break

    print("\nRunning final evaluation...")
    final_metrics = evaluate(
        model, criterion, tau, batch_size,
        max_seq_length, max_gates, eval_batches * 5, device
    )

    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Bit Accuracy:      {final_metrics['accuracy']:.3f}")
    print(f"Sequence Accuracy: {final_metrics['sequence_accuracy']:.3f}")
    print(f"Test Loss:         {final_metrics['task_loss']:.4f}")
    print(f"Average Ponder:    {final_metrics['ponder']:.2f}")
    print(f"Average Steps:     {final_metrics['steps']:.1f}")
    print(f"Best Seq Accuracy: {best_sequence_accuracy:.3f}")
    print(f"Total Time:        {(time.time() - start_time)/60:.2f} minutes")
    print(f"{'='*80}")

    print("\nTesting on 5 example sequences:")
    for i in range(5):
        X, Y, seq_lengths = generate_logic_sequence(1, max_seq_length, max_gates, device)
        seq_len = seq_lengths[0].item()

        model.eval()
        with torch.no_grad():
            output, ponder, steps = model(X)
            output = torch.sigmoid(output)

        print(f"\nSequence {i+1} (length {seq_len}):")
        all_correct = True
        for t in range(seq_len):
            target = Y[t, 0, 0].int().item()
            pred = (output[t, 0, 0] > 0.5).int().item()
            correct = "✓" if pred == target else "✗"
            if pred != target:
                all_correct = False

            num_gates = (X[t, 0, 2:] > 0).sum().item() // 1  # Approximate
            print(f"  t={t}: Target={target} Pred={pred} {correct} | "
                  f"Ponder={ponder[t, 0].item():.2f} Steps={steps[t, 0].item():.0f}")

        seq_result = "✓ CORRECT" if all_correct else "✗ INCORRECT"
        print(f"  Sequence: {seq_result}")

    print(f"\nEnd time: {datetime.datetime.now()}")
