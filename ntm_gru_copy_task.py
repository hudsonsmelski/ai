# Training script for the NTM on 4.1 Copy task from the NTM paper
# Hudson Andrew Smelski

import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import time
import datetime
from collections import deque
import string

from ntm_gru import *

def generate_copy_batch(batch_size, seq_len, vocab_size, device):
    seqs = torch.randint(0, vocab_size - 1, (seq_len, batch_size), device=device)
    delimiter = (vocab_size - 1) * torch.ones(1, batch_size, dtype=torch.long, device=device)
    blanks = torch.zeros(seq_len, batch_size, dtype=torch.long, device=device)

    input_indices = torch.cat([seqs, delimiter, blanks], dim=0)  # [2*seq_len+1, batch]
    input_seq = F.one_hot(input_indices, num_classes=vocab_size).float()  # [2*seq_len+1, batch, vocab]

    return input_seq, seqs #[tensor], ints

def evaluate_copy_accuracy(predictions, targets):
    pred_tokens = torch.argmax(predictions, dim=-1)
    correct = (pred_tokens == targets).float().mean().item()
    return correct

def train_copy_task_until_converged(model, device, max_iters=50000, seq_len_start=4, seq_len_max=20,
    batch_size=8, lr=1e-4, print_every=100, eval_every=500, save_every=2000, acc_threshold=0.99,
    loss_threshold=0.01, model_dir="./models",
):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=lr*0.1)
    model.train()

    current_seq_len = seq_len_start
    loss_history = deque(maxlen=100)
    best_acc = 0.0
    best_path = os.path.join(model_dir, "ntm_copy_best.pt")

    start_time = time.time()

    for it in range(1, max_iters + 1):
        input_seq, target_seq = generate_copy_batch(batch_size, current_seq_len, vocab_size, device)
        model.reset(batch_size=batch_size)

        # Forward pass through sequence
        outputs = []
        for t in range(input_seq.size(0)):
            y = model.forward(input_seq[t])  # [batch, vocab_size]
            outputs.append(y)

        outputs = torch.stack(outputs, dim=0)  # [total_len, batch, vocab_size]
        pred_logits = outputs[-(current_seq_len):, :, :]  # [seq_len, batch, vocab_size]

        # Loss
        loss = F.cross_entropy(
            pred_logits.reshape(-1, vocab_size),
            target_seq.reshape(-1),
            label_smoothing=0.0
        )
        # Encourage sharp attention (penalize entropy)
        read_entropy = -(model.read_w * (model.read_w + 1e-8).log()).sum(-1).mean()
        write_entropy = -(model.write_w * (model.write_w + 1e-8).log()).sum(-1).mean()
        entropy_loss = 0.01 * (read_entropy + write_entropy)

        total_loss = loss + entropy_loss

        optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        # Evaluation
        if it % eval_every == 0:
            model.eval()
            with torch.no_grad():
                eval_seq_len = min(current_seq_len + 5, seq_len_max)
                input_eval, target_eval = generate_copy_batch(batch_size, eval_seq_len, vocab_size, device)

                model.reset(batch_size=batch_size)
                outputs_eval = []
                for t in range(input_eval.size(0)):
                    y = model.forward(input_eval[t])
                    outputs_eval.append(y)

                outputs_eval = torch.stack(outputs_eval, dim=0)
                pred_eval = outputs_eval[-(eval_seq_len):, :, :]

                eval_loss = F.cross_entropy(
                    pred_eval.reshape(-1, vocab_size),
                    target_eval.reshape(-1)
                ).item()

                eval_acc = evaluate_copy_accuracy(pred_eval, target_eval)
                stats = model.get_memory_usage_stats()

                print(f"\n[EVAL Iter {it}] Eval Loss={eval_loss:.4f}, Eval Acc={eval_acc*100:.2f}% (len={eval_seq_len})")
                print(f"Read entropy = {stats['read_entropy']:.2f}")
                print(f"Write entropy = {stats['write_entropy']:.2f}")
                print(f"Memory Sparsity {stats['memory_sparsity']:.2f}")
                # Show example (first in batch)
                pred_tokens = torch.argmax(pred_eval[:, 0, :], dim=-1).cpu().tolist()
                target_tokens = target_eval[:, 0].cpu().tolist()
                pred_str = ''.join(idx_to_char.get(i, '?') for i in pred_tokens)
                target_str = ''.join(idx_to_char.get(i, '?') for i in target_tokens)
                print(f"  Eval Target:  {repr(target_str)}")
                print(f"  Eval Predict: {repr(pred_str)}\n")

                if eval_acc > best_acc:
                    best_acc = eval_acc
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

        # Training metrics
        acc = evaluate_copy_accuracy(pred_logits, target_seq)

        # Logging
        if it % print_every == 0:
            avg_loss = sum(loss_history) / len(loss_history) if loss_history else loss_val
            elapsed = time.time() - start_time
            iter_per_sec = it / elapsed

            print(f"\r[Iter {it:5d}] Loss={loss_val:.4f} (avg={avg_loss:.4f}), Acc={acc*100:.2f}%, "
                  f"Len={current_seq_len}, GradNorm={grad_norm:.2f}, LR={optimizer.param_groups[0]['lr']:.2e}, "
                  f"Speed={iter_per_sec:.1f} it/s", end = "")

            # Show example (first in batch)
            with torch.no_grad():
                pred_tokens = torch.argmax(pred_logits[:, 0, :], dim=-1).cpu().tolist()
                target_tokens = target_seq[:, 0].cpu().tolist()
                pred_str = ''.join(idx_to_char.get(i, '?') for i in pred_tokens)
                target_str = ''.join(idx_to_char.get(i, '?') for i in target_tokens)
                print(f"  Target:  {repr(target_str)}"
                      f"  Predict: {repr(pred_str)}", end = "")

        # Save checkpoint
        if it % save_every == 0:
            ckpt_path = os.path.join(model_dir, f"ntm_copy_iter_{it}.pt")
            torch.save({
                'iteration': it,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                #'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss_val,
                'acc': acc,
                'seq_len': current_seq_len,
                'best_acc': best_acc
            }, ckpt_path)
            print(f">>> Checkpoint saved: {ckpt_path}")

        # Curriculum & early stopping
        if acc >= acc_threshold and loss_val < loss_threshold:
            if current_seq_len >= seq_len_max:
                print("\n" + "="*60)
                print("=== TRAINING COMPLETE ===")
                print(f"Reached acc={acc*100:.2f}% and loss={loss_val:.4f} at max seq_len={seq_len_max}")
                print("="*60)
                break
            else:
                current_seq_len = min(current_seq_len + 2, seq_len_max)
                print(f"\n>>> Curriculum: Increasing seq_len to {current_seq_len} <<<\n")
                #optimizer = Adam(model.parameters(), lr=optimizer.param_groups[0]['lr'], weight_decay=1e-5)

    # Final save
    final_path = os.path.join(model_dir, "ntm_copy_final.pt")
    torch.save({
        'iteration': it,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_val,
        'acc': acc,
        'seq_len': current_seq_len,
        'best_acc': best_acc
    }, final_path)
    print(f"\n>>> Final model saved to: {final_path}")
    print(f">>> Best model (acc={best_acc*100:.2f}%) saved to: {best_path}")

    return final_path


if __name__ == "__main__":
    # Define the set of valid ASCII characters to use as tokens
    vocab = string.digits + string.ascii_letters + string.punctuation + " \t\v\n\r\f"
    vocab = vocab[0:20]
    vocab_size = len(vocab)

    char_to_idx = {char: idx for idx, char in enumerate(vocab)}    #Create a dictionary that maps each character to a unique integer value
    idx_to_char = {idx: char for char, idx in char_to_idx.items()} #Create a reverse dictionary that maps each integer value to its corresponding character

    print(f"Start time: {datetime.datetime.now()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("NTM BATCHED COPY TASK TRAINING")

    memory_length = 128
    model = NTM(
        vocab_size=vocab_size,
        memory_length=memory_length,
        controller_depth=1,
        controller_width=100,
        read_heads=1,
        write_heads=1,
        device = device
    ).to(device)

    model_controller = "GRU"

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Configuration:")
    print(f"  Vocab: {vocab} ({vocab_size})")
    print(f"  Memory: {memory_length} x {vocab_size}")
    print(f"  Controller: {model_controller}")
    print(f"  Total parameters: {total_params:,}")
    print("="*60)

    start_time = time.time()
    final_path = train_copy_task_until_converged(
        model,
        device=device,
        max_iters=50000,
        seq_len_start=2,
        seq_len_max=50,
        batch_size=32,
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
    print(f"Time: {(time.time()-start_time)/60:.2f} min")
    print("="*60)

    print(f"\nEnd time: {datetime.datetime.now()}")
