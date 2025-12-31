# Training script for RTDNC on Copy task
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

from rtdnc import *

def generate_copy_batch(batch_size, seq_len, vocab_size, device):
    """
    Generate copy task batch with random sequence lengths.
    Input:  [seq] [delimiter] [blanks]
    Target: [ignore] [ignore] [seq]  (predict during blank period)
    Returns:
        input_seq: [max_total_len, batch] - input tokens
        target_seq: [max_total_len, batch] - target tokens (SAME length as input!)
        mask: [max_total_len, batch] - 1 where we compute loss, 0 elsewhere
    """
    # Random lengths between 1 and seq_len for each sequence in batch
    seq_lengths = torch.randint(1, seq_len + 1, (batch_size,), device=device)

    # Max total length based on longest possible sequence
    max_total_len = 2 * seq_len + 1

    # Reserve token IDs properly
    delimiter_idx = vocab_size - 1
    blank_idx = vocab_size - 2  # Separate token for "blank/recall mode"
    usable_vocab = vocab_size - 2  # Don't use delimiter or blank in sequences

    # Initialize
    input_seq = torch.zeros(max_total_len, batch_size, dtype=torch.long, device=device)
    target_seq = torch.zeros(max_total_len, batch_size, dtype=torch.long, device=device)
    mask = torch.zeros(max_total_len, batch_size, dtype=torch.bool, device=device)

    for i in range(batch_size):
        curr_len = seq_lengths[i].item()

        # Generate random sequence (avoid delimiter and blank tokens)
        seq = torch.randint(0, usable_vocab, (curr_len,), device=device)

        # Input: [seq, delimiter, blanks...]
        input_seq[:curr_len, i] = seq
        input_seq[curr_len, i] = delimiter_idx
        input_seq[curr_len+1:curr_len+1+curr_len, i] = blank_idx  # Explicit blank token

        # Target: [don't care, don't care, seq to predict]
        target_seq[curr_len+1:curr_len+1+curr_len, i] = seq  # Predict during recall period

        # Mask: only compute loss during recall period
        mask[curr_len+1:curr_len+1+curr_len, i] = True

    return input_seq, target_seq, mask


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

def check_for_bad_gradients(model, it):
    """Check for NaN/Inf gradients in all parameters"""
    bad_params = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                bad_params.append((name, 'NaN', param.grad.abs().max().item()))
            elif torch.isinf(param.grad).any():
                bad_params.append((name, 'Inf', param.grad.abs().max().item()))
            elif param.grad.abs().max() > 100:
                bad_params.append((name, 'Large', param.grad.abs().max().item()))

    if bad_params:
        print(f"\n{'='*70}")
        print(f"[Iter {it}] BAD GRADIENTS DETECTED:")
        for name, issue, val in bad_params:
            print(f"  {name}: {issue} (max={val:.2e})")
        print('='*70)
        return True
    return False

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
    model_dir="./models",
):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.0001, betas=(0.9, 0.98))
    scheduler = CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=lr*0.1)

    model.train()

    current_seq_len = seq_len_start
    loss_history = deque(maxlen=100)
    best_acc = 0.0
    best_loss = float('inf')
    best_path = os.path.join(model_dir, "rtdnc_copy_best.pt")

    start_time = time.time()

    for it in range(1, max_iters + 1):
        # Warmup learning rate
        if it <= warmup_steps:
            lr_scale = min(1.0, it / warmup_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * lr_scale

        # Generate batch - NOW WITH MASK!
        input_seq, target_seq, mask = generate_copy_batch(
            batch_size, current_seq_len, vocab_size, device)

        # Initialize state
        model.reset(batch_size=batch_size)

        # Forward pass through full sequence
        all_logits = model.forward(
            input_seq,
            return_all_logits=True
        )  # [total_len, batch, vocab_size]

        # FIXED: Extract only the masked (recall) period
        # mask is [total_len, batch], find where it's True
        recall_start = current_seq_len + 1  # After seq + delimiter
        pred_logits = all_logits[recall_start:, :, :]  # [seq_len, batch, vocab_size]
        pred_targets = target_seq[recall_start:, :]  # [seq_len, batch]
        pred_mask = mask[recall_start:, :]  # [seq_len, batch]

        # Compute loss only on valid positions
        loss = F.cross_entropy(
            pred_logits.reshape(-1, vocab_size),  # [seq_len*batch, vocab_size]
            pred_targets.reshape(-1),  # [seq_len*batch]
            reduction='none'
        )
        loss = loss.view(current_seq_len, batch_size)  # [seq_len, batch]

        # Apply mask (though with fixed lengths, mask should be all True in recall period)
        loss = (loss * pred_mask.float()).sum() / pred_mask.float().sum()

        # Optional: Add regularization on memory attention
        read_w = model.read_w  # [batch, RH, N]
        write_w = model.write_w  # [batch, WH, N]

        eps = 1e-8
        read_entropy = -(read_w * (read_w + eps).log()).sum(-1).mean()
        write_entropy = -(write_w * (write_w + eps).log()).sum(-1).mean()
        entropy_reg = 0.001 * (read_entropy + write_entropy)  # Reduced weight

        total_loss = loss + entropy_reg

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()

        if check_for_bad_gradients(model, it):
            print("Skipping optimizer step due to bad gradients")
            optimizer.zero_grad()
            model.reset(batch_size=batch_size)

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)  # Increased from 1.0
        optimizer.step()

        if it > warmup_steps:
            scheduler.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        # Training accuracy - FIXED
        with torch.no_grad():
            pred_tokens = torch.argmax(pred_logits, dim=-1)  # [seq_len, batch]
            correct = (pred_tokens == pred_targets).float()
            acc = (correct * pred_mask.float()).sum() / pred_mask.float().sum()
            acc = acc.item()

        # Evaluation
        if it % eval_every == 0:
            model.eval()
            with torch.no_grad():
                eval_seq_len = min(current_seq_len + 5, seq_len_max)
                input_eval, target_eval, eval_mask = generate_copy_batch(
                    batch_size, eval_seq_len, vocab_size, device
                )

                model.reset(batch_size=batch_size)
                outputs_eval = model.forward(
                    input_eval,
                    return_all_logits=True
                )

                # Extract recall period - FIXED
                eval_recall_start = eval_seq_len + 1
                pred_eval = outputs_eval[eval_recall_start:, :, :]  # [eval_seq_len, batch, vocab_size]
                target_eval_recall = target_eval[eval_recall_start:, :]  # [eval_seq_len, batch]
                mask_eval_recall = eval_mask[eval_recall_start:, :]  # [eval_seq_len, batch]

                # Compute eval loss
                eval_loss_raw = F.cross_entropy(
                    pred_eval.reshape(-1, vocab_size),
                    target_eval_recall.reshape(-1),
                    reduction='none'
                )
                eval_loss_raw = eval_loss_raw.view(eval_seq_len, batch_size)
                eval_loss = (eval_loss_raw * mask_eval_recall.float()).sum() / mask_eval_recall.float().sum()
                eval_loss = eval_loss.item()

                # Eval accuracy - FIXED
                pred_tokens_eval = torch.argmax(pred_eval, dim=-1)
                correct_eval = (pred_tokens_eval == target_eval_recall).float()
                eval_acc = (correct_eval * mask_eval_recall.float()).sum() / mask_eval_recall.float().sum()
                eval_acc = eval_acc.item()

                print(f"\n[EVAL Iter {it}] Loss={eval_loss:.4f}, Acc={eval_acc*100:.2f}% (len={eval_seq_len})")
                model.print_memory_stats()

                eval_seq_len_first = eval_seq_len  # Using fixed length for simplicity
                original_seq_first = input_eval[:eval_seq_len_first, 0].cpu().tolist()

                pred_tokens_first = pred_tokens_eval[:, 0].cpu().tolist()
                target_tokens_first = target_eval_recall[:, 0].cpu().tolist()

                # Convert to strings (excluding special tokens)
                original_str = ''.join(idx_to_char.get(i, '?') for i in original_seq_first
                                       if i < vocab_size - 2)  # Exclude delimiter and blank
                target_str = ''.join(idx_to_char.get(i, '?') for i in target_tokens_first
                                    if i < vocab_size - 2)
                pred_str = ''.join(idx_to_char.get(i, '?') for i in pred_tokens_first
                                  if i < vocab_size - 2)

                print(f"\n  Input:   {repr(original_str)}")
                print(f"  Target:  {repr(target_str)}")
                print(f"  Predict: {repr(pred_str)}")

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
                            'vocab_size': model.input_size,
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

            with torch.no_grad():
                original_first = input_seq[:current_seq_len, 0].cpu().tolist()
                pred_tokens_first = pred_tokens[:, 0].cpu().tolist()
                target_tokens_first = pred_targets[:, 0].cpu().tolist()

                original_str = ''.join(idx_to_char.get(i, '?') for i in original_first
                                      if i < vocab_size - 2)
                pred_str = ''.join(idx_to_char.get(i, '?') for i in pred_tokens_first
                                  if i < vocab_size - 2)
                target_str = ''.join(idx_to_char.get(i, '?') for i in target_tokens_first
                                    if i < vocab_size - 2)

                print(f"  I: {repr(original_str)} | T: {repr(target_str)} | P: {repr(pred_str)}")

        # Save checkpoint
        if it % save_every == 0:
            ckpt_path = os.path.join(model_dir, f"rtdnc_copy_iter_{it}.pt")
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
                print(f"\n>>> CURRICULUM: Increasing seq_len to {current_seq_len}")

    # Final save
    final_path = os.path.join(model_dir, "rtdnc_copy_final.pt")
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
            'vocab_size': model.input_size,
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
    print("RTDNC COPY TASK TRAINING")

    # Model configuration
    emb_dim = 200
    memory_N = 128
    n_heads = 4
    n_layers = 2
    controller_window = 8
    read_heads = 1
    write_heads = 1

    print(f"\nModel Configuration:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Embedding dimension: {emb_dim}")
    print(f"  Memory: {memory_N} × {emb_dim}")
    print(f"  Transformer: {n_layers} layers, {n_heads} heads")
    print(f"  Controller window: {controller_window}")
    print(f"  Read heads: {read_heads}, Write heads: {write_heads}")

    model = RTDNC(
        input_size=vocab_size,
        emb_dim=emb_dim,
        memory_N=memory_N,
        n_heads=n_heads,
        n_layers=n_layers,
        controller_window=controller_window,
        read_heads=read_heads,
        write_heads=write_heads,
        dropout=0.1
    ).to(device)

    total_params = model.num_params()
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
        loss_threshold=0.05,
        model_dir="./models"
    )

    elapsed_time = time.time() - start_time

    print("\n" + "="*70)
    print("TRAINING FINISHED")
    print(f"Final model: {final_path}")
    print(f"Total time: {elapsed_time/60:.2f} minutes ({elapsed_time/3600:.2f} hours)")
    print("="*70)

    print(f"\nEnd time: {datetime.datetime.now()}")
    print("="*70)
