# Training script for RTDNC learning addition with little-endian numbers
# Hudson Andrew Smelski
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import datetime
import string
from random import randint
from rtdnc import *

# Numerical vocab with explicit padding token
vocab = string.digits + "+= _"  # Add underscore as padding
vocab_len = len(vocab)
char_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_char = {i: ch for i, ch in enumerate(vocab)}
PAD_IDX = char_to_idx['_']  # Padding token index
START_TOKEN = char_to_idx[' ']  # Use space as explicit start token for answer generation

def generate_data(batch_len: int, num_len: int):
    """
    Generate addition problems like: 12+34=46
    In little-endian: 21+43=64
    """
    problems = []
    answers = []
    for _ in range(batch_len):
        num1_str = str(torch.randint(1, 10, (1,)).item())
        if num_len > 1:
            num1_str += ''.join([str(torch.randint(0, 10, (1,)).item())
                                for _ in range(randint(0, num_len - 1))])
        num2_str = str(torch.randint(1, 10, (1,)).item())
        if num_len > 1:
            num2_str += ''.join([str(torch.randint(0, 10, (1,)).item())
                                for _ in range(randint(0, num_len - 1))])
        num1 = int(num1_str)
        num2 = int(num2_str)
        ans = num1 + num2
        problem = f"{num1_str[::-1]}+{num2_str[::-1]}="
        answer = f" {str(ans)}"[::-1]  # leading space kept, reversed
        problems.append(problem)
        answers.append(answer)
    max_prob_len = max(len(p) for p in problems)
    max_ans_len = max(len(a) for a in answers)
    x = torch.full((max_prob_len, batch_len), PAD_IDX, dtype=torch.long)
    y = torch.full((max_ans_len, batch_len), PAD_IDX, dtype=torch.long)
    for i, (prob, ans) in enumerate(zip(problems, answers)):
        for t, ch in enumerate(prob):
            x[t, i] = char_to_idx[ch]
        for t, ch in enumerate(ans):
            y[t, i] = char_to_idx[ch]
    return x, y

def compute_accuracy(predictions, targets):
    pred_chars = predictions.argmax(dim=-1)
    mask = targets != PAD_IDX
    correct = (pred_chars == targets) & mask
    return correct.float().sum() / mask.float().sum()

def compute_sequence_accuracy(predictions, targets):
    pred_chars = predictions.argmax(dim=-1)           # [T, B]
    mask = targets != PAD_IDX                         # [T, B], True on real tokens
    correct = pred_chars == targets                   # [T, B]
    correct_on_valid = correct & mask
    errors_in_valid = (~correct_on_valid) & mask      # True only if mistake on real token
    no_errors = ~errors_in_valid.any(dim=0)           # [B]
    return no_errors.float().mean().item()

def format_prediction(x, y, pred, batch_idx=0):
    problem = ''.join(idx_to_char[x[t, batch_idx].item()] for t in range(x.size(0)) if x[t, batch_idx].item() != PAD_IDX)
    target = ''.join(idx_to_char[y[t, batch_idx].item()] for t in range(y.size(0)) if y[t, batch_idx].item() != PAD_IDX)
    prediction = ''
    for t in range(pred.size(0)):
        char_idx = pred[t, batch_idx].argmax().item()
        if char_idx == PAD_IDX:
            break
        prediction += idx_to_char[char_idx]
    return problem, target, prediction

def train_until_converged(
    model,
    device,
    max_iters=50000,
    num_len_start=1,
    num_len_max=20,
    batch_len=32,
    lr=1e-4,
    warmup_steps=2000,
    print_every=100,
    eval_every=1000,
    save_every=5000,
    acc_threshold=0.99,
    loss_threshold=0.01,
    model_dir="./models",
    filename="rtdnc_add_littleendian"
):
    os.makedirs(model_dir, exist_ok=True)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.0, betas=(0.9, 0.98))
    scheduler = CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=lr*0.1)
    model.train()
    current_num_len = num_len_start
    best_seq_acc = 0.0
    best_path = os.path.join(model_dir, f"{filename}_best.pt")
    start_time = time.time()

    for it in range(1, max_iters + 1):
        # Warmup
        if it <= warmup_steps:
            lr_scale = it / warmup_steps
            for pg in optimizer.param_groups:
                pg['lr'] = lr * lr_scale

        x, y = generate_data(batch_len, current_num_len)
        x = x.to(device)
        y = y.to(device)

        model.reset(batch_size=batch_len)

        # Process input problem
        for t in range(x.size(0)):
            model.step(x[t])

        # Teacher-forced training on answer
        answer_preds = []
        for t in range(y.size(0)):
            if t > 0:
                input_t = y[t-1]  # [batch]
            else:
                input_t = torch.full((batch_len,), START_TOKEN, device=device, dtype=torch.long)
            logits = model.step(input_t)
            answer_preds.append(logits.unsqueeze(0))
        answer_preds = torch.cat(answer_preds, dim=0)

        loss = F.cross_entropy(answer_preds.reshape(-1, vocab_len), y.reshape(-1), ignore_index=PAD_IDX)

        # Light entropy regularization (kept from original)
        eps = 1e-8
        read_entropy = -(model.read_w * (model.read_w + eps).log()).sum(-1).mean()
        write_entropy = -(model.write_w * (model.write_w + eps).log()).sum(-1).mean()
        total_loss = loss + 0.001 * (read_entropy + write_entropy)

        optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # increased slightly
        optimizer.step()
        if it > warmup_steps:
            scheduler.step()

        loss_val = loss.item()
        acc = compute_accuracy(answer_preds, y)
        seq_acc = compute_sequence_accuracy(answer_preds, y)

        if it % print_every == 0:
            elapsed = time.time() - start_time
            print(f"\r[Iter {it:5d}] num_len={current_num_len} | "
                  f"Loss: {loss_val:.4f} | CharAcc: {acc:.4f} | SeqAcc: {seq_acc:.4f} | "
                  f"GradNorm: {grad_norm:.2f} | LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {elapsed:.1f}s", end="")

        if it % eval_every == 0:
            model.eval()
            with torch.no_grad():
                print(f"\n\nEVAL {it}")
                print(f"[Current Training Size: num_len={current_num_len}]")
                x_curr, y_curr = generate_data(batch_len, current_num_len)
                x_curr, y_curr = x_curr.to(device), y_curr.to(device)
                model.reset(batch_size=batch_len)
                for t in range(x_curr.size(0)):
                    model.step(x_curr[t])
                curr_preds = []
                prev_token = torch.full((batch_len,), START_TOKEN, device=device, dtype=torch.long)
                for _ in range(y_curr.size(0)):
                    logits = model.step(prev_token)
                    prev_token = logits.argmax(-1)
                    curr_preds.append(logits.unsqueeze(0))
                curr_preds = torch.cat(curr_preds, dim=0)
                curr_acc = compute_accuracy(curr_preds, y_curr)
                curr_seq_acc = compute_sequence_accuracy(curr_preds, y_curr)
                prob, target, pred = format_prediction(x_curr, y_curr, curr_preds, 0)
                print(f" Char Accuracy: {curr_acc:.4f} | Seq Accuracy: {curr_seq_acc:.4f}")
                print(f" Example: {prob} → {pred} ({'✓' if target == pred else '✗'})")

                # --- Max length robustness test (free generation) ---
                print(f"\n[Robustness Test: 10 batches at num_len={num_len_max}]")
                eval_seq_accs = []
                for _ in range(10):
                    x_test, y_test = generate_data(batch_len, num_len_max)
                    x_test, y_test = x_test.to(device), y_test.to(device)
                    model.reset(batch_size=batch_len)
                    for t in range(x_test.size(0)):
                        model.step(x_test[t])
                    test_preds = []
                    prev_token = torch.full((batch_len,), START_TOKEN, device=device, dtype=torch.long)
                    for _ in range(y_test.size(0) + 5):  # +5 to allow longer answers if needed
                        logits = model.step(prev_token)
                        next_token = logits.argmax(-1)
                        test_preds.append(logits.unsqueeze(0))
                        prev_token = next_token
                    test_preds = torch.cat(test_preds, dim=0)[:y_test.size(0)]
                    eval_seq_accs.append(compute_sequence_accuracy(test_preds, y_test))
                avg_eval_seq_acc = sum(eval_seq_accs) / len(eval_seq_accs)
                print(f" Average sequence accuracy @ max length: {avg_eval_seq_acc:.4f}")

                if avg_eval_seq_acc > best_seq_acc:
                    best_seq_acc = avg_eval_seq_acc
                    torch.save({
                        'iteration': it,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'seq_acc': avg_eval_seq_acc,
                        'num_len': num_len_max,
                    }, best_path)
                    print(f" ✓ New best model saved (seq_acc={best_seq_acc:.4f})")

            model.train()

        if it % save_every == 0:
            ckpt_path = os.path.join(model_dir, f"{filename}_iter{it}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"\n>>> Checkpoint saved: {ckpt_path}")

        # Curriculum advance
        if seq_acc >= acc_threshold and loss_val < loss_threshold:
            if current_num_len >= num_len_max:
                print(f"\n{'='*70}")
                print(f"TARGET REACHED: Perfect performance on num_len={num_len_max}")
                print(f"{'='*70}")
                break
            else:
                current_num_len += 1
                print(f"\n>>> CURRICULUM: Advancing to num_len={current_num_len}")

    final_path = os.path.join(model_dir, f"{filename}_final.pt")
    torch.save(model.state_dict(), final_path)
    return final_path

if __name__ == "__main__":
    print(f"Start time: {datetime.datetime.now()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    print(f"Using device: {device}")

    # Model Meta Parameters
    emb_dim = 256
    memory_N = 128
    n_heads = 4
    n_layers = 2
    controller_window = 10
    read_heads = 4
    write_heads = 1

    print(f"\nModel Configuration:")
    print(f"  Vocab: {vocab} (size={vocab_len})")
    print(f"  Embedding dimension: {emb_dim}")
    print(f"  Memory: {memory_N} × {emb_dim}")
    print(f"  Transformer: {n_layers} layers, {n_heads} heads")
    print(f"  Controller window: {controller_window} (forces external memory use)")
    print(f"  Read heads: {read_heads}, Write heads: {write_heads}")

    model = RTDNC(
        input_size=vocab_len,
        emb_dim=emb_dim,
        memory_N=memory_N,
        n_heads=n_heads,
        n_layers=n_layers,
        controller_window=controller_window,
        read_heads=read_heads,
        write_heads=write_heads,
        dropout=0.1
    )

    total_params = model.num_params()
    print(f" Total parameters: {total_params:,}")
    print("="*70)

    start_time = time.time()
    final_path = train_until_converged(
        model,
        device=device,
        max_iters=100000,
        num_len_start=1,
        num_len_max=20,
        batch_len=32,
        lr=1e-3,
        warmup_steps=2000,
        print_every=100,
        eval_every=1000,
        save_every=5000,
        acc_threshold=0.99,
        loss_threshold=0.05,
    )

    elapsed_time = time.time() - start_time

    print("\n" + "="*70)
    print("TRAINING FINISHED")
    print(f"Final model saved to: {final_path}")
    print(f"Final model: {final_path}")
    print(f"Total time: {elapsed_time/60:.2f} minutes ({elapsed_time/3600:.2f} hours)")
    print("="*70)

    print(f"\nEnd time: {datetime.datetime.now()}")
    print("="*70)
