# Training script for RTNTM learning addition with little-endian numbers
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

from rtntm import RTNTM, count_parameters

# Numerical vocab with explicit padding token
vocab = string.digits + "+= _" # Add underscore as padding
vocab_len = len(vocab)
char_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_char = {i: ch for i, ch in enumerate(vocab)}
PAD_IDX = char_to_idx['_']  # Padding token index

def generate_data(batch_len: int, num_len: int):
    """
    Generate addition problems like: 12+34=46
    In little-endian: 21+43=64

    Returns:
        x: [max_prob_len, batch] - problem tokens (padded with PAD_IDX)
        y: [max_ans_len, batch] - answer tokens (padded with PAD_IDX)
    """
    problems = []
    answers = []

    for _ in range(batch_len):
        # Generate two random numbers with up to num_len digits as strings
        # Ensure first digit is not 0
        num1_str = str(torch.randint(1, 10, (1,)).item())
        if num_len > 1:
            num1_str += ''.join([str(torch.randint(0, 10, (1,)).item())
                                for _ in range(randint(0, num_len - 1))])

        num2_str = str(torch.randint(1, 10, (1,)).item())
        if num_len > 1:
            num2_str += ''.join([str(torch.randint(0, 10, (1,)).item())
                                for _ in range(randint(0, num_len - 1))])

        # Convert to integers for addition
        num1 = int(num1_str)
        num2 = int(num2_str)
        ans = num1 + num2

        # Create problem string in little-endian (reversed): "21+43="
        problem = f"{num1_str[::-1]}+{num2_str[::-1]}="
        # Answer string in little-endian (reversed): "64"
        answer = f" {ans}"[::-1]

        problems.append(problem)
        answers.append(answer)

    # Find max lengths for padding
    max_prob_len = max(len(p) for p in problems)
    max_ans_len = max(len(a) for a in answers)

    # Convert to token indices with padding
    x = torch.full((max_prob_len, batch_len), PAD_IDX, dtype=torch.long)
    y = torch.full((max_ans_len, batch_len), PAD_IDX, dtype=torch.long)

    for i, (prob, ans) in enumerate(zip(problems, answers)):
        # Encode problem
        for t, ch in enumerate(prob):
            x[t, i] = char_to_idx[ch]
        # Encode answer
        for t, ch in enumerate(ans):
            y[t, i] = char_to_idx[ch]

    return x, y  # [seq_len, batch], [ans_len, batch]

def compute_accuracy(predictions, targets):
    """
    Compute character-level accuracy
    predictions: [seq_len, batch, vocab_len] - logits
    targets: [seq_len, batch] - token indices
    """
    pred_chars = predictions.argmax(dim=-1)  # [seq_len, batch]
    correct = (pred_chars == targets).float().mean()
    return correct.item()


def compute_sequence_accuracy(predictions, targets):
    """
    Compute full sequence accuracy (all characters must match)
    predictions: [seq_len, batch, vocab_len]
    targets: [seq_len, batch]
    """
    pred_chars = predictions.argmax(dim=-1)  # [seq_len, batch]
    # Check if all tokens in each sequence match
    correct_seqs = (pred_chars == targets).all(dim=0).float().mean()
    return correct_seqs.item()


def format_prediction(x, y, pred, batch_idx=0):
    """Format a single example for display"""
    # Decode input problem (skip padding)
    problem = ''
    for t in range(x.size(0)):
        if x[t, batch_idx].item() != PAD_IDX:
            problem += idx_to_char[x[t, batch_idx].item()]

    # Decode target answer (skip padding)
    target = ''
    for t in range(y.size(0)):
        if y[t, batch_idx].item() != PAD_IDX:
            target += idx_to_char[y[t, batch_idx].item()]

    # Decode predicted answer (stop at padding if present)
    prediction = ''
    for t in range(pred.size(0)):
        char_idx = pred[t, batch_idx, :].argmax().item()
        if char_idx == PAD_IDX:
            break  # Stop at padding
        prediction += idx_to_char[char_idx]

    return problem, target, prediction


def train_until_converged(
    model,
    device,
    max_iters=50000,
    num_len_start=1,
    num_len_max=20,
    batch_len=16,
    lr=3e-4,
    warmup_steps=1000,
    print_every=100,
    eval_every=1000,
    save_every=5000,
    acc_threshold=0.99,
    loss_threshold=0.01,
    model_dir="./models_rtntm_add",
    filename="rtntm_add"
):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.0001, betas=(0.9, 0.98))
    scheduler = CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=lr*0.1)

    model.train()
    current_num_len = num_len_start
    best_acc = 0.0
    best_seq_acc = 0.0
    best_path = os.path.join(model_dir, f"{filename}_best.pt")
    start_time = time.time()

    for it in range(1, max_iters + 1):
        # Warmup
        if it <= warmup_steps:
            lr_scale = min(1.0, it / warmup_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * lr_scale

        # Generate batch
        x, y = generate_data(batch_len, current_num_len)
        # x: [prob_len, batch] - problem tokens
        # y: [ans_len, batch] - answer tokens

        x = x.to(device)
        y = y.to(device)

        # Initialize state
        model.init_state(batch_size=batch_len, device=device)

        # Process input sequence (problem)
        for t in range(x.size(0)):
            _ = model.step(x[t])

        # Generate answer tokens one by one
        answer_preds = []
        for t in range(y.size(0)):
            # Feed zeros (or previous prediction) during generation
            # For teacher forcing, we'd feed y[t-1], but let's do pure generation
            zero_input = torch.zeros(batch_len, dtype=torch.long, device=device)
            logits = model.step(zero_input)
            answer_preds.append(logits.unsqueeze(0))

        answer_preds = torch.cat(answer_preds, dim=0)  # [ans_len, batch, vocab_len]

        # Compute loss on answer
        loss = F.cross_entropy(
            answer_preds.reshape(-1, vocab_len),
            y.reshape(-1),
            label_smoothing=0.0
        )

        # Optional: Add memory regularization
        read_w = model.read_w
        write_w = model.write_w
        eps = 1e-8
        read_entropy = -(read_w * (read_w + eps).log()).sum(-1).mean()
        write_entropy = -(write_w * (write_w + eps).log()).sum(-1).mean()
        entropy_reg = 0.001 * (read_entropy + write_entropy)

        total_loss = loss + entropy_reg

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if it > warmup_steps:
            scheduler.step()

        loss_val = loss.item()
        acc = compute_accuracy(answer_preds, y)
        seq_acc = compute_sequence_accuracy(answer_preds, y)

        # Evaluation
        if it % eval_every == 0:
            model.eval()
            with torch.no_grad():
                print(f"\nEVAL {it}")

                # Eval on current training size
                print(f"[Current Training Size: num_len={current_num_len}]")
                x_curr, y_curr = generate_data(batch_len, current_num_len)
                x_curr = x_curr.to(device)
                y_curr = y_curr.to(device)

                model.init_state(batch_size=batch_len, device=device)
                for t in range(x_curr.size(0)):
                    _ = model.step(x_curr[t])

                curr_preds = []
                for t in range(y_curr.size(0)):
                    zero_input = torch.zeros(batch_len, dtype=torch.long, device=device)
                    logits = model.step(zero_input)
                    curr_preds.append(logits.unsqueeze(0))
                curr_preds = torch.cat(curr_preds, dim=0)

                curr_acc = compute_accuracy(curr_preds, y_curr)
                curr_seq_acc = compute_sequence_accuracy(curr_preds, y_curr)

                # Memory stats
                mem_stats = model.get_memory_usage_stats()

                # Show example
                prob, target, prediction = format_prediction(x_curr, y_curr, curr_preds, batch_idx=0)
                print(f"  Char Accuracy: {curr_acc:.4f}")
                print(f"  Sequence Accuracy: {curr_seq_acc:.4f}")
                print(f"  Memory std: {mem_stats['memory_std']:.4f}")
                print(f"  Memory Sparsity {mem_stats['memory_sparsity']:.2f}")
                print(f"  Read Entropy    {mem_stats['read_entropy']:.2f}")
                print(f"  Write Entropy   {mem_stats['write_entropy']:.2f}")
                print(f"  Read Sharpness  {mem_stats['read_sharpness']:.2f}")
                print(f"  Write Sharpness {mem_stats['write_sharpness']:.2f}")
                print(f"  Example:")
                print(f"    Problem:    {prob}")
                print(f"    Target:     {prob}{target}")
                print(f"    Prediction: {prob}{prediction}")
                print(f"    Correct: {'✓' if target == prediction else '✗'}")

                # Eval on max size
                print(f"\n[Max Size: num_len={num_len_max}]")
                x_eval, y_eval = generate_data(batch_len, num_len_max)
                x_eval = x_eval.to(device)
                y_eval = y_eval.to(device)

                model.init_state(batch_size=batch_len, device=device)
                for t in range(x_eval.size(0)):
                    _ = model.step(x_eval[t])

                eval_preds = []
                for t in range(y_eval.size(0)):
                    zero_input = torch.zeros(batch_len, dtype=torch.long, device=device)
                    logits = model.step(zero_input)
                    eval_preds.append(logits.unsqueeze(0))
                eval_preds = torch.cat(eval_preds, dim=0)

                eval_acc = compute_accuracy(eval_preds, y_eval)
                eval_seq_acc = compute_sequence_accuracy(eval_preds, y_eval)

                # Show example from max size
                prob, target, prediction = format_prediction(x_eval, y_eval, eval_preds, batch_idx=0)
                print(f"  Char Accuracy: {eval_acc:.4f}")
                print(f"  Sequence Accuracy: {eval_seq_acc:.4f}")
                print(f"  Example:")
                print(f"    Problem:    {prob}")
                print(f"    Target:     {prob}{target}")
                print(f"    Prediction: {prob}{prediction}")
                print(f"    Correct: {'✓' if target == prediction else '✗'}")

                # Test on multiple batches for robust eval
                print(f"\n[Robustness Test: 10 batches at num_len={num_len_max}]")
                eval_accs = [eval_acc]
                eval_seq_accs = [eval_seq_acc]

                for _ in range(9):
                    x_test, y_test = generate_data(batch_len, num_len_max)
                    x_test = x_test.to(device)
                    y_test = y_test.to(device)

                    model.init_state(batch_size=batch_len, device=device)
                    for t in range(x_test.size(0)):
                        _= model.step(x_test[t])

                    test_preds = []
                    for t in range(y_test.size(0)):
                        zero_input = torch.zeros(batch_len, dtype=torch.long, device=device)
                        logits = model.step(zero_input)
                        test_preds.append(logits.unsqueeze(0))
                    test_preds = torch.cat(test_preds, dim=0)

                    eval_accs.append(compute_accuracy(test_preds, y_test))
                    eval_seq_accs.append(compute_sequence_accuracy(test_preds, y_test))

                avg_eval_acc = sum(eval_accs) / len(eval_accs)
                avg_eval_seq_acc = sum(eval_seq_accs) / len(eval_seq_accs)

                print(f"  Average char accuracy: {avg_eval_acc:.4f}")
                print(f"  Average sequence accuracy: {avg_eval_seq_acc:.4f}")

                # Save best model
                if avg_eval_seq_acc > best_seq_acc or (avg_eval_seq_acc == best_seq_acc and avg_eval_acc > best_acc):
                    best_acc = avg_eval_acc
                    best_seq_acc = avg_eval_seq_acc
                    torch.save({
                        'iteration': it,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'char_acc': avg_eval_acc,
                        'seq_acc': avg_eval_seq_acc,
                        'num_len': num_len_max,
                        'config': {
                            'vocab_size': model.vocab_size,
                            'd_model': model.D,
                            'memory_N': model.N,
                        }
                    }, best_path)
                    print(f"  ✓ Saved new best model (seq_acc={best_seq_acc:.4f}, char_acc={best_acc:.4f})")

                print(f"{'='*70}\n")

            model.train()

        # Logging
        if it % print_every == 0:
            elapsed = time.time() - start_time
            print(f"\r[Iter {it:5d}] num_len={current_num_len} | "
                  f"Loss: {loss_val:.4f} | CharAcc: {acc:.4f} | SeqAcc: {seq_acc:.4f} | "
                  f"GradNorm: {grad_norm:.2f} | LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {elapsed:.1f}s", end = "")

        # Save checkpoint
        if it % save_every == 0:
            checkpoint_path = os.path.join(model_dir, f"{filename}_iter{it}.pt")
            torch.save({
                'iteration': it,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'current_num_len': current_num_len,
                'loss': loss_val,
            }, checkpoint_path)
            print(f"\n>>> Checkpoint saved: {checkpoint_path}\n")

        # Curriculum learning
        if seq_acc >= acc_threshold and loss_val < loss_threshold:
            if current_num_len >= num_len_max:
                print(f"\n{'='*70}")
                print(f"Reached target num_len={num_len_max}")
                print(f"CharAcc={acc:.4f}, SeqAcc={seq_acc:.4f}, Loss={loss_val:.4f}")
                print(f"{'='*70}")
                break
            else:
                current_num_len += 1
                print(f"\n{'='*70}")
                print(f">>> CURRICULUM: Advancing to num_len={current_num_len}")
                print(f"{'='*70}\n")

    # Save final model
    final_path = os.path.join(model_dir, f"{filename}_final.pt")
    torch.save({
        'iteration': it,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_val,
        'num_len': current_num_len,
    }, final_path)

    return final_path


if __name__ == "__main__":
    print("="*70)
    print(f"Start time: {datetime.datetime.now()}")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    print(f"\nUsing device: {device}")

    print("\n" + "="*70)
    print("RTNTM ADDITION TASK (Little-Endian)")
    print("="*70)

    # Model configuration
    emb_dim = 64
    memory_N = 128
    n_heads = 8
    n_layers = 1
    controller_window = 8
    read_heads = 2
    write_heads = 1
    state_layers = 1

    print(f"\nModel Configuration:")
    print(f"  Vocab: {vocab} (size={vocab_len})")
    print(f"  Embedding dimension: {emb_dim}")
    print(f"  Memory: {memory_N} × {emb_dim}")
    print(f"  Transformer: {n_layers} layers, {n_heads} heads")
    print(f"  Controller window: {controller_window} (forces external memory use)")
    print(f"  Read heads: {read_heads}, Write heads: {write_heads}")
    print(f"  State layers: {state_layers}")

    model = RTNTM(
        vocab_size=vocab_len,
        emb_dim=emb_dim,
        memory_N=memory_N,
        n_heads=n_heads,
        n_layers=n_layers,
        controller_window=controller_window,
        read_heads=read_heads,
        write_heads=write_heads,
        shift_width=3,
        state_layers=state_layers,
        dropout=0.1
    ).to(device)

    total_params = count_parameters(model)
    print(f"  Total parameters: {total_params:,}")
    print()

    # Training
    start_time = time.time()
    final_path = train_until_converged(
        model,
        device=device,
        max_iters=50000,
        num_len_start=1,
        num_len_max=20,
        batch_len=32,
        lr=3e-4,
        warmup_steps=1000,
        print_every=100,
        eval_every=1000,
        save_every=5000,
        acc_threshold=0.99,
        loss_threshold=0.01,
        model_dir="./models_rtntm_add",
        filename="rtntm_add_littleendian"
    )

    elapsed_time = time.time() - start_time

    print("\n" + "="*70)
    print("TRAINING FINISHED")
    print(f"Final model: {final_path}")
    print(f"Total time: {elapsed_time/60:.2f} minutes ({elapsed_time/3600:.2f} hours)")
    print("="*70)

    print(f"\nEnd time: {datetime.datetime.now()}")
    print("="*70)
