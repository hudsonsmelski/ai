# Training script for RTNTM on Copy task
# Adapted from NTM copy task

import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import time
import datetime
from collections import deque
import string
import math
from random import randint

from rtntm import *

def generate_copy_batch(batch_size, seq_len, vocab_size, device):
    """
    Generate copy task batch where each sequence has a random length.
    Input: [seq] [delimiter] [blanks] (padded to consistent length)
    Target: the original sequence (to be output during blank period)

    Returns:
        input_indices: [max_total_len, batch] - token indices
        target_seq: [max_seq_len, batch] - target tokens to predict (0-padded)
        seq_lengths: [batch] - actual length of each sequence
    """
    # Random length for each sequence in the batch
    seq_lengths = torch.randint(max(seq_len-1, 1), seq_len + 1, (batch_size,), device=device)
    max_len = seq_lengths.max().item()

    # Total input length: seq + delimiter + blanks for recall
    max_total_len = 2 * max_len + 1

    # Initialize tensors
    input_indices = torch.zeros(max_total_len, batch_size, dtype=torch.long, device=device)
    target_seq = torch.zeros(max_len, batch_size, dtype=torch.long, device=device)

    delimiter_idx = vocab_size - 1

    for i in range(batch_size):
        length = seq_lengths[i].item()

        # Generate random sequence
        seq = torch.randint(0, vocab_size - 1, (length,), device=device)

        # Build input: [seq, delimiter, blanks]
        input_indices[:length, i] = seq
        input_indices[length, i] = delimiter_idx
        # Blanks are already zeros from initialization

        # Store target (the sequence to copy)
        target_seq[:length, i] = seq

    return input_indices, target_seq, seq_lengths


def evaluate_copy_accuracy(predictions, targets, seq_lengths):
    """
    predictions: [max_len, batch, vocab_size] - logits
    targets: [max_len, batch] - token indices
    seq_lengths: [batch] - actual sequence lengths
    """
    pred_tokens = torch.argmax(predictions, dim=-1)

    # Create mask for valid positions
    max_len = predictions.shape[0]
    mask = torch.arange(max_len, device=predictions.device).unsqueeze(1) < seq_lengths.unsqueeze(0)

    correct = ((pred_tokens == targets) & mask).float().sum() / mask.sum()
    return correct.item()


def train_copy_task_until_converged(
    model,
    device,
    max_iters=50000,
    seq_len_start=4,
    seq_len_max=30,
    batch_size=16,
    lr=3e-4,
    warmup_steps=1000,
    print_every=100,
    eval_every=500,
    save_every=2000,
    acc_threshold=0.99,
    loss_threshold=0.01,
    model_dir="./models_rtntm",
):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0, betas=(0.9, 0.98))
    scheduler = CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=lr*0.1)

    model.train()

    current_seq_len = seq_len_start
    loss_history = deque(maxlen=100)
    best_acc = 0.0
    best_loss = float('inf')
    best_path = os.path.join(model_dir, "rtntm_copy_best.pt")

    start_time = time.time()

    for it in range(1, max_iters + 1):
        # Warmup learning rate
        if it <= warmup_steps:
            lr_scale = min(1.0, it / warmup_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * lr_scale

        # Generate batch
        input_indices, target_seq, seq_lengths = generate_copy_batch(
            batch_size, current_seq_len, vocab_size, device)

        # Initialize state
        model.init_state(batch_size=batch_size, device=device)

        # Forward pass through full sequence
        all_logits = model.forward(
            input_indices,
            return_all_logits=True
        )

        # Extract predictions for the "recall" period
        max_len = seq_lengths.max().item()
        pred_logits = all_logits[-max_len:, :, :]  # [max_len, batch, vocab_size]

        # Create mask for valid positions (to ignore padding in loss)
        mask = torch.arange(max_len, device=device).unsqueeze(1) < seq_lengths.unsqueeze(0)
        mask = mask.float()  # [max_len, batch]

        # Compute loss only on valid (non-padded) positions
        loss = F.cross_entropy(
            pred_logits.reshape(-1, vocab_size),
            target_seq.reshape(-1),
            reduction='none'
        )
        loss = loss.view(max_len, batch_size)
        loss = (loss * mask).sum() / mask.sum()  # Masked average

        # Optional: Add regularization on memory attention (encourage sharpness)
        read_w = model.read_w  # [batch, RH, N]
        write_w = model.write_w  # [batch, WH, N]

        eps = 1e-8
        read_entropy = -(read_w * (read_w + eps).log()).sum(-1).mean()
        write_entropy = -(write_w * (write_w + eps).log()).sum(-1).mean()
        entropy_reg = 0.001 * (read_entropy + write_entropy)

        total_loss = loss + entropy_reg

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if it > warmup_steps:
            scheduler.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        # Training accuracy
        with torch.no_grad():
            acc = evaluate_copy_accuracy(pred_logits, target_seq, seq_lengths)

        # Evaluation
        if it % eval_every == 0:
            model.eval()
            with torch.no_grad():
                eval_seq_len = min(current_seq_len + 5, seq_len_max)
                input_eval, target_eval, eval_seq_lengths = generate_copy_batch(
                    batch_size, eval_seq_len, vocab_size, device
                )

                model.init_state(batch_size=batch_size, device=device)
                outputs_eval = model.forward(
                    input_eval,
                    return_all_logits=True
                )

                # Extract predictions for eval
                eval_max_len = eval_seq_lengths.max().item()
                pred_eval = outputs_eval[-eval_max_len:, :, :]

                # Create eval mask
                eval_mask = torch.arange(eval_max_len, device=device).unsqueeze(1) < eval_seq_lengths.unsqueeze(0)
                eval_mask = eval_mask.float()

                # Compute eval loss
                eval_loss_raw = F.cross_entropy(
                    pred_eval.reshape(-1, vocab_size),
                    target_eval.reshape(-1),
                    reduction='none'
                )
                eval_loss_raw = eval_loss_raw.view(eval_max_len, batch_size)
                eval_loss = (eval_loss_raw * eval_mask).sum() / eval_mask.sum()
                eval_loss = eval_loss.item()

                eval_acc = evaluate_copy_accuracy(pred_eval, target_eval, eval_seq_lengths)
                stats = model.get_memory_usage_stats()

                print(f"\n{'='*70}")
                print(f"[EVAL Iter {it}] Loss={eval_loss:.4f}, Acc={eval_acc*100:.2f}% (len={eval_seq_len})")
                print(f"Memory Sparsity {stats['memory_sparsity']:.2f}")
                print(f"Read Entropy    {stats['read_entropy']:.2f}")
                print(f"Write Entropy   {stats['write_entropy']:.2f}")
                print(f"Read Sharpness  {stats['read_sharpness']:.2f}")
                print(f"Write Sharpness {stats['write_sharpness']:.2f}")

                # Show example (first in batch)
                first_seq_len = eval_seq_lengths[0].item()
                pred_tokens = torch.argmax(pred_eval[:first_seq_len, 0, :], dim=-1).cpu().tolist()
                target_tokens = target_eval[:first_seq_len, 0].cpu().tolist()
                pred_str = ''.join(idx_to_char.get(i, '?') for i in pred_tokens)
                target_str = ''.join(idx_to_char.get(i, '?') for i in target_tokens)
                print(f"\n  Target:  {repr(target_str)}")
                print(f"  Predict: {repr(pred_str)}")
                print('='*70 + '\n')

                # Save best model
                if eval_acc > best_acc or (eval_acc == best_acc and eval_loss < best_loss):
                    best_acc = eval_acc
                    best_loss = eval_loss
                    torch.save({
                        'iteration': it,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': eval_loss,
                        'acc': eval_acc,
                        'seq_len': eval_seq_len,
                        'config': {
                            'vocab_size': model.vocab_size,
                            'd_model': model.D,
                            'memory_N': model.N,
                            'n_heads': model.controller.n_heads,
                            'n_layers': model.controller.n_layers,
                            'read_heads': model.RH,
                            'write_heads': model.WH,
                        }
                    }, best_path)
                    print(f">>> New best model: Acc={eval_acc*100:.2f}%, Loss={eval_loss:.4f}")
                    print(f"    Saved to {best_path}\n")

            model.train()

        # Logging
        if it % print_every == 0:
            avg_loss = sum(loss_history) / len(loss_history) if loss_history else loss_val
            elapsed = time.time() - start_time
            iter_per_sec = it / elapsed

            print(f"[Iter {it:5d}] Loss={loss_val:.4f} (avg={avg_loss:.4f}), "
                  f"Acc={acc*100:.2f}%, Len={current_seq_len}, "
                  f"GradNorm={grad_norm:.2f}, LR={optimizer.param_groups[0]['lr']:.2e}, "
                  f"Speed={iter_per_sec:.1f} it/s")

            # Show training example (use actual sequence length from first item)
            with torch.no_grad():
                first_seq_len = seq_lengths[0].item()
                pred_tokens = torch.argmax(pred_logits[:first_seq_len, 0, :], dim=-1).cpu().tolist()
                target_tokens = target_seq[:first_seq_len, 0].cpu().tolist()
                pred_str = ''.join(idx_to_char.get(i, '?') for i in pred_tokens)
                target_str = ''.join(idx_to_char.get(i, '?') for i in target_tokens)
                print(f"  T: {repr(target_str)} | P: {repr(pred_str)}")

        # Save checkpoint
        if it % save_every == 0:
            ckpt_path = os.path.join(model_dir, f"rtntm_copy_iter_{it}.pt")
            torch.save({
                'iteration': it,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss_val,
                'acc': acc,
                'seq_len': current_seq_len,
                'best_acc': best_acc,
                'best_loss': best_loss
            }, ckpt_path)
            print(f"\n>>> Checkpoint saved: {ckpt_path}\n")

        # Curriculum learning & early stopping
        if acc >= acc_threshold and loss_val < loss_threshold:
            if current_seq_len >= seq_len_max:
                print("\n" + "="*70)
                print("=== TRAINING COMPLETE ===")
                print(f"Reached acc={acc*100:.2f}% and loss={loss_val:.4f}")
                print(f"at max seq_len={seq_len_max}")
                print("="*70)
                break
            else:
                # Increase sequence length
                current_seq_len = min(current_seq_len + 2, seq_len_max)
                print(f"\n{'='*70}")
                print(f">>> CURRICULUM: Increasing seq_len to {current_seq_len}")
                print(f"{'='*70}\n")

    # Final save
    final_path = os.path.join(model_dir, "rtntm_copy_final.pt")
    torch.save({
        'iteration': it,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss_val,
        'acc': acc,
        'seq_len': current_seq_len,
        'best_acc': best_acc,
        'best_loss': best_loss,
        'config': {
            'vocab_size': model.vocab_size,
            'd_model': model.D,
            'memory_N': model.N,
        }
    }, final_path)
    print(f"\n>>> Final model saved to: {final_path}")
    print(f">>> Best model (acc={best_acc*100:.2f}%, loss={best_loss:.4f}) at: {best_path}")

    return final_path


if __name__ == "__main__":
    # ---------------------------
    # Simple tokenizer
    # ---------------------------
    vocab = string.digits + string.ascii_letters + string.punctuation + " \t\v\n\r\f"
    vocab_size = len(vocab)
    char_to_idx = {char: idx for idx, char in enumerate(vocab)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    print(f"Start time: {datetime.datetime.now()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    print(f"Using device: {device}")
    print("RTNTM COPY TASK TRAINING")

    # Model configuration
    emb_dim = 200
    memory_N = 60
    n_heads = 4
    n_layers = 1
    controller_window = 4
    read_heads = 1
    write_heads = 1

    print(f"\nModel Configuration:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Embedding dimension: {emb_dim}")
    print(f"  Memory: {memory_N} × {emb_dim}")
    print(f"  Transformer: {n_layers} layers, {n_heads} heads")
    print(f"  Controller window: {controller_window}")
    print(f"  Read heads: {read_heads}, Write heads: {write_heads}")

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
        state_layers=1,
        dropout=0.0
    ).to(device)

    total_params = count_parameters(model)
    print(f"  Total parameters: {total_params:,}")
    print("="*70, "\n")

    # Training
    start_time = time.time()
    final_path = train_copy_task_until_converged(
        model,
        device=device,
        max_iters=50000,
        seq_len_start=1,
        seq_len_max=50,
        batch_size=32,
        lr=1e-3,
        warmup_steps=1000,
        print_every=100,
        eval_every=500,
        save_every=2000,
        acc_threshold=0.99,
        loss_threshold=0.01,
        model_dir="./models_rtntm"
    )

    elapsed_time = time.time() - start_time

    print("\n" + "="*70)
    print("TRAINING FINISHED")
    print(f"Final model: {final_path}")
    print(f"Total time: {elapsed_time/60:.2f} minutes ({elapsed_time/3600:.2f} hours)")
    print("="*70)

    print(f"\nEnd time: {datetime.datetime.now()}")
    print("="*70)
