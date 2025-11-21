# ================================================================
# IMPROVED TRAINING SCRIPT FOR TNTM ON THE COPY TASK
# ================================================================

import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from collections import deque

from rtntm import *

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def generate_copy_batch(batch_size, seq_len, vocab_size, device):
    """
    Creates a batch for the copy task:
    Input:  random sequence + delimiter token + blank tokens for output
    Target: original sequence
    """
    seq = torch.randint(0, vocab_size - 1, (seq_len, batch_size), device=device)
    delimiter = (vocab_size - 1) * torch.ones(1, batch_size, dtype=torch.long, device=device)
    # Add blank tokens for model to output into
    blanks = torch.zeros(seq_len, batch_size, dtype=torch.long, device=device)
    x = torch.cat([seq, delimiter, blanks], dim=0)  # [2*seq_len + 1, batch]
    y = seq.clone()
    return x, y


def evaluate_copy_accuracy(logits, targets):
    """
    logits: [seq_len * batch, vocab]
    targets: [seq_len * batch]
    Returns percentage of correctly predicted tokens.
    """
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == targets).float().mean().item()
    return correct


def train_copy_task_until_converged(
    model,
    device,
    max_iters=50000,
    seq_len_start=4,        # Curriculum: start small
    seq_len_max=20,
    batch_size=8,           # Larger batch for stability
    lr=1e-3,                # Higher initial LR
    print_every=100,
    eval_every=500,
    save_every=2000,
    acc_threshold=0.99,
    loss_threshold=0.01,
    model_dir="./models",
):

    ensure_dir(model_dir)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    model.train()

    # Curriculum learning
    current_seq_len = seq_len_start

    # Moving average for loss smoothing
    loss_history = deque(maxlen=100)

    # Best model tracking
    best_loss = float('inf')
    best_acc = 0.0

    start_time = time.time()

    for it in range(1, max_iters + 1):
        # ----------------------------
        # Generate batch
        # ----------------------------
        x, y = generate_copy_batch(batch_size, current_seq_len, vocab_size, device)
        state = model.init_state(batch_size=batch_size, device=device)

        # ----------------------------
        # Run model on entire input sequence
        # ----------------------------
        logits_all = []
        for t in range(x.shape[0]):
            logits, state = model.step(x[t], state)
            logits_all.append(logits.unsqueeze(0))

        logits_all = torch.cat(logits_all, dim=0)                     # [2*seq_len+1, batch, vocab]
        pred_logits = logits_all[-(current_seq_len):, :, :]           # last seq_len predictions
        pred_logits = pred_logits.reshape(-1, vocab_size)             # [seq_len*batch, vocab]
        targets = y.reshape(-1)                                        # [seq_len*batch]

        # ----------------------------
        # Loss + backward with gradient clipping
        # ----------------------------
        loss = F.cross_entropy(pred_logits, targets)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        # ----------------------------
        # Periodic evaluation
        # ----------------------------
        if it % eval_every == 0:
            model.eval()
            with torch.no_grad():
                # Evaluate on a larger sequence
                eval_seq_len = min(current_seq_len + 5, seq_len_max)
                x_eval, y_eval = generate_copy_batch(batch_size, eval_seq_len, vocab_size, device)
                state_eval = model.init_state(batch_size=batch_size, device=device)

                logits_eval = []
                for t in range(x_eval.shape[0]):
                    logits_t, state_eval = model.step(x_eval[t], state_eval)
                    logits_eval.append(logits_t.unsqueeze(0))

                logits_eval = torch.cat(logits_eval, dim=0)
                pred_eval = logits_eval[-(eval_seq_len):, :, :].reshape(-1, vocab_size)
                targets_eval = y_eval.reshape(-1)

                eval_loss = F.cross_entropy(pred_eval, targets_eval).item()
                eval_acc = evaluate_copy_accuracy(pred_eval, targets_eval)

                print(f"\n[EVAL Iter {it}] Eval Loss={eval_loss:.4f}, Eval Acc={eval_acc*100:.2f}% (len={eval_seq_len})")

                # Update scheduler
                scheduler.step(eval_loss)

                # Save best model
                if eval_acc > best_acc:
                    best_acc = eval_acc
                    best_path = os.path.join(model_dir, "tntm_copy_best.pt")
                    torch.save({
                        'iteration': it,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': eval_loss,
                        'acc': eval_acc,
                        'seq_len': current_seq_len
                    }, best_path)
                    print(f">>> New best accuracy: {eval_acc*100:.2f}% - Saved to {best_path}")

            model.train()

        # ----------------------------
        # Training metrics
        # ----------------------------
        acc = evaluate_copy_accuracy(pred_logits, targets)

        # ----------------------------
        # Logging
        # ----------------------------
        if it % print_every == 0:
            avg_loss = sum(loss_history) / len(loss_history) if loss_history else loss_val
            elapsed = time.time() - start_time
            iter_per_sec = it / elapsed

            print(f"[Iter {it:5d}] Loss={loss_val:.4f} (avg={avg_loss:.4f}), Acc={acc*100:.2f}%, "
                  f"Len={current_seq_len}, GradNorm={grad_norm:.2f}, LR={optimizer.param_groups[0]['lr']:.2e}, "
                  f"Speed={iter_per_sec:.1f} it/s")

            # Show prediction example (first batch item)
            with torch.no_grad():
                pred_idxs = torch.argmax(pred_logits, dim=-1)[:current_seq_len].cpu().tolist()
                tgt_idxs = targets[:current_seq_len].cpu().tolist()
                pred_seq = ''.join(idx_to_char.get(i, '?') for i in pred_idxs)
                tgt_seq = ''.join(idx_to_char.get(i, '?') for i in tgt_idxs)
                print(f"  Target:  {repr(tgt_seq)}")
                print(f"  Predict: {repr(pred_seq)}")
                print()

        # ----------------------------
        # Save checkpoint periodically
        # ----------------------------
        if it % save_every == 0:
            ckpt_path = os.path.join(model_dir, f"tntm_copy_iter_{it}.pt")
            torch.save({
                'iteration': it,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss_val,
                'acc': acc,
                'seq_len': current_seq_len,
                'best_acc': best_acc
            }, ckpt_path)
            print(f">>> Checkpoint saved: {ckpt_path}")

        # ----------------------------
        # Early stopping
        # ----------------------------
        if acc >= acc_threshold and loss_val < loss_threshold:
            if current_seq_len >= seq_len_max:
                print("\n" + "="*60)
                print("=== TRAINING COMPLETE ===")
                print(f"Reached acc={acc*100:.2f}% and loss={loss_val:.4f} at max seq_len={seq_len_max}")
                print("="*60)
                break
            else:
                # Curriculum: gradually increase sequence length
                current_seq_len = min(current_seq_len + 2, seq_len_max)
                print(f"\n>>> Curriculum: Increasing seq_len to {current_seq_len} <<<\n")
                optimizer = Adam(model.parameters(), lr=optimizer.param_groups[0]['lr'], weight_decay=1e-5)


    # ----------------------------
    # Final save
    # ----------------------------
    final_path = os.path.join(model_dir, "tntm_copy_final.pt")
    torch.save({
        'iteration': it,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss_val,
        'acc': acc,
        'seq_len': current_seq_len,
        'best_acc': best_acc
    }, final_path)
    print(f"\n>>> Final model saved to: {final_path}")
    print(f">>> Best model (acc={best_acc*100:.2f}%) saved to: {best_path}")

    return final_path


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*60)
    print("TNTM COPY TASK TRAINING")
    print("="*60)
    print(f"Device: {device}")
    print()

    # Model hyperparameters
    memory_N = 25
    memory_M = 97  # Match d_model
    d_model = 100
    n_heads = 4
    read_heads = 2
    write_heads = 1
    shift_K = 3

    model = TNTM(
        vocab_size=vocab_size,
        d_model=d_model,
        memory_N=memory_N,
        memory_M=memory_M,
        n_heads=n_heads,
        read_heads=read_heads,
        write_heads=write_heads,
        shift_width=shift_K
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Configuration:")
    print(f"  d_model: {d_model}")
    print(f"  Memory: {memory_N} x {memory_M}")
    print(f"  Read/Write heads: {read_heads}/{write_heads}")
    print(f"  Total parameters: {total_params:,}")
    print()

    final_path = train_copy_task_until_converged(
        model,
        device=device,
        max_iters=50000,
        seq_len_start=5,
        seq_len_max=20,
        batch_size=8,
        lr=1e-3,
        print_every=100,
        eval_every=500,
        save_every=2000,
        acc_threshold=0.99,
        loss_threshold=0.01,
        model_dir="./models"
    )

    print("\n" + "="*60)
    print("TRAINING FINISHED")
    print(f"Final model: {final_path}")
    print("="*60)
