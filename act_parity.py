"""
Training script for ACT on Parity Task
Based on Section 3.1 of the ACT paper
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import datetime
from pathlib import Path

from act import ACT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Start time: {datetime.datetime.now()}")

def generate_data(batch_size, length, device, max_nonzeros, placement='random'):
    X = torch.zeros(batch_size, length, device=device)

    # Generate number of nonzeros for each sample
    num_nonzero = torch.randint(0, max_nonzeros + 1, (batch_size,), device=device)

    if placement == 'beginning':
        # Place nonzeros consecutively at the beginning
        for i in range(batch_size):
            if num_nonzero[i] > 0:
                signs = torch.randint(0, 2, (num_nonzero[i],), device=device) * 2 - 1
                X[i, :num_nonzero[i]] = signs.float()

    elif placement == 'end':
        # Place nonzeros consecutively at the end
        for i in range(batch_size):
            if num_nonzero[i] > 0:
                signs = torch.randint(0, 2, (num_nonzero[i],), device=device) * 2 - 1
                X[i, -num_nonzero[i]:] = signs.float()

    else:  # 'random'
        # Scatter randomly throughout the vector (vectorized)
        mask = torch.arange(max_nonzeros, device=device).unsqueeze(0) < num_nonzero.unsqueeze(1)
        # Generate random indices in range [0, length) using randperm approach
        indices = torch.argsort(torch.rand(batch_size, length, device=device), dim=1)[:, :max_nonzeros]
        signs = torch.randint(0, 2, (batch_size, max_nonzeros), device=device) * 2 - 1

        valid_indices = indices[mask]
        valid_signs = signs[mask].float()
        batch_indices = torch.arange(batch_size, device=device).repeat_interleave(num_nonzero)

        X[batch_indices, valid_indices] = valid_signs

    # Compute parity
    parity = (X != 0).sum(dim=1) % 2

    # Prepare output tensors
    Y = parity.float().unsqueeze(1)
    x = X.unsqueeze(0)  # (1, batch, length)
    y = Y.unsqueeze(0)  # (1, batch, 1)

    return x, y

def train_epoch(model, optimizer, criterion, tau, batch_size, input_length,
                num_batches, device, max_nonzeros):
    model.train()

    total_accuracy = 0
    total_task_loss = 0
    total_total_loss = 0
    total_ponder = 0
    total_steps = 0

    for _ in range(num_batches):
        X, Y = generate_data(batch_size, input_length, device, max_nonzeros,  placement='random')
        optimizer.zero_grad()
        output, ponder_costs, steps = model(X)

        # Apply sigmoid to output for binary classification
        # Squeeze sequence dim: (1, batch, 1) -> (batch, 1)
        output = torch.sigmoid(output.squeeze(0))
        target = Y.squeeze(0)

        task_loss = criterion(output, target)
        ponder_mean = ponder_costs.mean()
        total_loss = task_loss + tau * ponder_mean

        total_loss.backward()
        optimizer.step()

        predictions = (output > 0.5).float()
        num_correct = (predictions == target).sum().item()

        total_accuracy += num_correct / batch_size
        total_task_loss += task_loss.item()
        total_total_loss += total_loss.item()
        total_ponder += ponder_mean.item()
        total_steps += steps.mean().item()

    return {
        'accuracy': total_accuracy / num_batches,
        'task_loss': total_task_loss / num_batches,
        'total_loss': total_total_loss / num_batches,
        'ponder': total_ponder / num_batches,
        'steps': total_steps / num_batches
    }

def evaluate(model, criterion, tau, batch_size, input_length, num_batches, device):
    model.eval()

    total_accuracy = 0
    total_task_loss = 0
    total_ponder = 0
    total_steps = 0

    with torch.no_grad():
        for _ in range(num_batches):
            X, Y = generate_data(batch_size, input_length, device, input_length)
            output, ponder_costs, steps = model(X)
            output = torch.sigmoid(output.squeeze(0))
            target = Y.squeeze(0)

            task_loss = criterion(output, target)
            ponder_mean = ponder_costs.mean()

            predictions = (output > 0.5).float()
            num_correct = (predictions == target).sum().item()

            total_accuracy += num_correct / batch_size
            total_task_loss += task_loss.item()
            total_ponder += ponder_mean.item()
            total_steps += steps.mean().item()

    return {
        'accuracy': total_accuracy / num_batches,
        'task_loss': total_task_loss / num_batches,
        'ponder': total_ponder / num_batches,
        'steps': total_steps / num_batches
    }

if __name__ == "__main__":
    # Hyperparameters (Section 3.1)
    input_length = 64
    output_length = 1
    batch_size = 128
    hidden_size = 1024
    hidden_type = "RNN"
    max_steps = 20
    lr = 1e-3
    tau = 0.001

    # Training parameters
    num_epochs = 50000
    train_batches = 10
    eval_batches = 100
    eval_interval = 100
    target_accuracy = 0.95

    # Curriculum parameters
    start_max_nonzeros = 1
    current_max_nonzeros = start_max_nonzeros
    curriculum_acc = 0.85
    should_eval = False

    print("=" * 80)
    print("Parity Task - ACT Training")
    print("=" * 80)
    print(f"Input length: {input_length}")
    print(f"Hidden size: {hidden_size}")
    print(f"Hidden type: {hidden_type}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Time penalty (tau): {tau}")
    print(f"Max steps: {max_steps}")
    print(f"Target accuracy: {target_accuracy}")
    print("=" * 80)

    model = ACT(
        input_size=input_length,
        hidden_size=hidden_size,
        output_size=output_length,
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
        threshold=0.01, min_lr=1e-4)

    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)

    best_accuracy = 0
    start_time = time.time()
    epoch_time = 0

    print(f"{'='*80}")
    print(f"Current Curriculum: max nonzeros = {current_max_nonzeros}")
    print(f"{'='*80}")

    for epoch in range(num_epochs):
        epoch_start = time.time()

        train_metrics = train_epoch(
            model, optimizer, criterion, tau, batch_size,
            input_length, train_batches, device, current_max_nonzeros
        )

        epoch_time += time.time() - epoch_start

        if current_max_nonzeros == 64:
            lr = 1e-4
            optimizer = optim.Adam(model.parameters(), lr=lr)

        if epoch % 5 == 0:
            print(f"\rEpoch   {epoch:4d} | "
                  f"Acc:    {train_metrics['accuracy']:.3f} | "
                  f"Loss:   {train_metrics['task_loss']:.4f} | "
                  f"Total:  {train_metrics['total_loss']:.4f} | "
                  f"Ponder: {train_metrics['ponder']:.2f} | "
                  f"Steps:  {train_metrics['steps']:.1f} | "
                  f"Time:   {epoch_time:.2f}s | "
                  f"lr = {lr}", end="")
            epoch_time = 0

        if train_metrics['accuracy'] >= curriculum_acc and current_max_nonzeros < input_length:
            current_max_nonzeros = min(input_length, current_max_nonzeros + 1)
            print(f"\n{'='*80}")
            print(f"Current Curriculum: max nonzeros = {current_max_nonzeros}")
            print(f"{'='*80}")
            should_eval = True

            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=10,
                threshold=0.01, min_lr=1e-4)

        # Periodic evaluation
        if (epoch + 1) % eval_interval == 0 or should_eval:
            should_eval = False

            eval_start = time.time()
            eval_metrics = evaluate(
                model, criterion, tau, batch_size,
                input_length, eval_batches, device
            )
            eval_time = time.time() - eval_start

            print(f"\n{'='*80}")
            print(f"EVAL      {epoch:4d} | "
                  f"Test Acc: {eval_metrics['accuracy']:.3f} | "
                  f"Loss:     {eval_metrics['task_loss']:.4f} | "
                  f"Ponder:   {eval_metrics['ponder']:.2f} | "
                  f"Steps:    {eval_metrics['steps']:.1f} | "
                  f"Time:     {eval_time:.2f}s | "
                  f"lr = {lr}")
            print(f"{'='*80}")

            if eval_metrics['accuracy'] > best_accuracy:
                best_accuracy = eval_metrics['accuracy']
                save_path = save_dir / f"act_{hidden_type}_parity_{input_length}_best.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': best_accuracy,
                    'config': {
                        'input_length': input_length,
                        'hidden_size': hidden_size,
                        'hidden_type': hidden_type,
                        'tau': tau,
                        'lr': lr
                    }
                }, save_path)
                print(f"✓ Saved best model with accuracy: {best_accuracy:.3f}")

            if eval_metrics['accuracy'] >= target_accuracy:
                print(f"\n{'='*80}")
                print(f"✓ Target accuracy {target_accuracy:.3f} reached!")
                print(f"Final test accuracy: {eval_metrics['accuracy']:.3f}")
                print(f"Total training time: {(time.time() - start_time)/60:.2f} minutes")
                print(f"{'='*80}")
                break

    if best_accuracy < 0.7:
        print(f"\n{'='*80}")
        print(f"⚠ WARNING: Training did not converge!")
        print(f"Best accuracy achieved: {best_accuracy:.3f}")
        print(f"This is likely because tau={tau} is too high.")
        print(f"\nSuggested fixes:")
        print(f"1. Lower tau to 0.0001 or 0.00001")
        print(f"2. Increase max_steps to allow more computation")
        print(f"3. Try LSTM instead of RNN")
        print(f"{'='*80}\n")

    print("\nRunning final evaluation...")
    final_metrics = evaluate(
        model, criterion, tau, batch_size,
        input_length, eval_batches * 10, device
    )

    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Test Accuracy:  {final_metrics['accuracy']:.3f}")
    print(f"Test Loss:      {final_metrics['task_loss']:.4f}")
    print(f"Average Ponder: {final_metrics['ponder']:.2f}")
    print(f"Average Steps:  {final_metrics['steps']:.1f}")
    print(f"Best Accuracy:  {best_accuracy:.3f}")
    print(f"Total Time:     {(time.time() - start_time)/60:.2f} minutes")
    print(f"{'='*80}")

    print("\nTesting on 5 examples:")
    for i in range(5):
        X, Y = generate_data(1, input_length, device, input_length)
        model.eval()
        with torch.no_grad():
            output, ponder, steps = model(X)
            output = torch.sigmoid(output.squeeze(0))

        num_ones = (X[0, 0] == 1).sum().item()
        target_class = Y[0, 0].int().item()
        pred_class = (output[0] > 0.5).int().item()
        correct = "✓" if pred_class == target_class else "✗"

        print(f"{i+1}. Ones: {num_ones:2d} | Target: {target_class} | "
              f"Pred: {pred_class} {correct} | "
              f"Ponder: {ponder[0, 0].item():.2f} | "
              f"Steps: {steps[0, 0].item():.0f}")

    print(f"\nEnd time: {datetime.datetime.now()}")
