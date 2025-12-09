# Training script for the NTM on 4.1 Copy task from the NTM paper
# Hudson Andrew Smelski

import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import datetime
from collections import deque

from ntm import *

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_copy_batch(batch_size, seq_len, vocab_size, device):
    seqs = torch.randint(0, vocab_size - 1, (seq_len, batch_size), device=device)
    delimiter = (vocab_size - 1) * torch.ones(1, batch_size, dtype=torch.long, device=device)
    blanks = torch.zeros(seq_len, batch_size, dtype=torch.long, device=device)

    input_indices = torch.cat([seqs, delimiter, blanks], dim=0)  # [2*seq_len+1, batch]
    input_seq = F.one_hot(input_indices, num_classes=vocab_size).float()  # [2*seq_len+1, batch, vocab]

    return input_seq, seqs #[tensor], ints

def evaluate_copy_accuracy(predictions, targets):
    """
    predictions: [seq_len, batch, vocab_size]
    targets: [seq_len, batch]
    """
    pred_tokens = torch.argmax(predictions, dim=-1)
    correct = (pred_tokens == targets).float().mean().item()
    return correct


def train_copy_task_until_converged(model, device, max_iters=50000, seq_len_start=4, seq_len_max=20,
    batch_size=8, lr=1e-4, print_every=100, eval_every=500, save_every=2000, acc_threshold=0.99,
    loss_threshold=0.01, model_dir="./models",
):

    ensure_dir(model_dir)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    model.train()

    current_seq_len = seq_len_start
    loss_history = deque(maxlen=100)
    best_acc = 0.0
    best_path = os.path.join(model_dir, "ntm_copy_best.pt")

    start_time = time.time()

    for it in range(1, max_iters + 1):
        # Generate batch
        input_seq, target_seq = generate_copy_batch(batch_size, current_seq_len, vocab_size, device)

        # Reset NTM
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
            label_smoothing=0.1
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

                print(f"\n[EVAL Iter {it}] Eval Loss={eval_loss:.4f}, Eval Acc={eval_acc*100:.2f}% (len={eval_seq_len})")

                # Show example (first in batch)
                pred_tokens = torch.argmax(pred_eval[:, 0, :], dim=-1).cpu().tolist()
                target_tokens = target_eval[:, 0].cpu().tolist()
                pred_str = ''.join(idx_to_char.get(i, '?') for i in pred_tokens)
                target_str = ''.join(idx_to_char.get(i, '?') for i in target_tokens)
                print(f"  Eval Target:  {repr(target_str)}")
                print(f"  Eval Predict: {repr(pred_str)}\n")

                scheduler.step(eval_loss)

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

            print(f"[Iter {it:5d}] Loss={loss_val:.4f} (avg={avg_loss:.4f}), Acc={acc*100:.2f}%, "
                  f"Len={current_seq_len}, GradNorm={grad_norm:.2f}, LR={optimizer.param_groups[0]['lr']:.2e}, "
                  f"Speed={iter_per_sec:.1f} it/s", end = "")

            # Show example (first in batch)
            with torch.no_grad():
                pred_tokens = torch.argmax(pred_logits[:, 0, :], dim=-1).cpu().tolist()
                target_tokens = target_seq[:, 0].cpu().tolist()
                pred_str = ''.join(idx_to_char.get(i, '?') for i in pred_tokens)
                target_str = ''.join(idx_to_char.get(i, '?') for i in target_tokens)
                print(f"  Target:  {repr(target_str)}"
                      f"  Predict: {repr(pred_str)}")

        # Save checkpoint
        if it % save_every == 0:
            ckpt_path = os.path.join(model_dir, f"ntm_copy_iter_{it}.pt")
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

        if acc >= 0.99 and loss_val > 0.1:
            print(f"DEBUG: acc={acc:.4f}, loss={loss_val:.4f}")
            print(f"Logits stats: min={pred_logits.min():.2f}, max={pred_logits.max():.2f}")
            print(f"Target predictions: {pred_logits.reshape(-1, vocab_size).argmax(dim=-1)[:10]}")
            print(f"Target truth:       {target_seq.reshape(-1)[:10]}")

        # Curriculum & early stopping
        # TODO: for some reason the loss is way too high for 100% accuracy.
        if acc >= acc_threshold:# and loss_val < loss_threshold:
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Start time: {datetime.datetime.now()}")

    print("="*60)
    print("NTM BATCHED COPY TASK TRAINING")
    print("="*60)
    print(f"Device: {device}")
    print()

    memory_length = 20
    model = NTM(
        vocab_size=vocab_size,
        memory_length=memory_length,
        controller_depth=1,
        read_heads=1,
        write_heads=1,
        use_lstm = False
    ).to(device)

    model_controller = "LSTM" if model.use_lstm else "RNN"

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Configuration:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Memory: {memory_length} x {vocab_size}")
    print(f"  Controller: {model_controller}")
    print(f"  Total parameters: {total_params:,}")
    print()

    start_time = time.time()
    final_path = train_copy_task_until_converged(
        model,
        device=device,
        max_iters=50000,
        seq_len_start=2,
        seq_len_max=20,
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
