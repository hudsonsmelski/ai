# Training script for the NTM on addition task
# Hudson Andrew Smelski

import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import datetime
from collections import deque
import string
from random import randint

from dnc import *

# Numerical vocab with explicit padding token
vocab = string.digits + "+= _" # Add underscore as padding
vocab_len = len(vocab)
char_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_char = {i: ch for i, ch in enumerate(vocab)}
PAD_IDX = char_to_idx['_']  # Padding token index

def generate_data(batch_len: int, num_len: int, vocab_size: int, device):
    """
    Generate addition problems like: 12+34=46
    In little-endian: 21+43=64
    Returns one-hot encoded tensors matching generate_copy_batch format
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

    # Create index tensors (with padding)
    x_indices = torch.full((max_prob_len, batch_len), PAD_IDX, dtype=torch.long, device=device)
    y_indices = torch.full((max_ans_len, batch_len), PAD_IDX, dtype=torch.long, device=device)

    for i, (prob, ans) in enumerate(zip(problems, answers)):
        for t, ch in enumerate(prob):
            x_indices[t, i] = char_to_idx[ch]
        for t, ch in enumerate(ans):
            y_indices[t, i] = char_to_idx[ch]

    # Convert to one-hot encoding
    x = F.one_hot(x_indices, num_classes=vocab_size).float()  # [max_prob_len, batch, vocab_size]
    y = F.one_hot(y_indices, num_classes=vocab_size).float()  # [max_ans_len, batch, vocab_size]

    return x, y_indices  # Return one-hot input, index targets (like generate_copy_batch)


def format_prediction(x, y, pred, batch_idx=0):
    """Format a single example for display"""
    # Decode input problem
    problem = ''
    for t in range(x.size(0)):
        char_idx = x[t, batch_idx, :].argmax().item()
        if char_idx != PAD_IDX:  # Only decode if not padding
            problem += idx_to_char[char_idx]

    # Decode target answer
    target = ''
    for t in range(y.size(0)):
        char_idx = y[t, batch_idx].item()
        if char_idx != PAD_IDX:  # Only decode if not padding
            target += idx_to_char[char_idx]

    # Decode predicted answer
    prediction = ''
    for t in range(pred.size(0)):
        char_idx = pred[t, batch_idx, :].argmax().item()
        prediction += idx_to_char[char_idx]

    # Trim prediction to target length to avoid showing excess padding
    prediction = prediction[:len(target)]

    return problem, target, prediction

def compute_accuracy(predictions, targets):
    pred_chars = predictions.argmax(dim=-1)  # [seq_len, batch_len]
    correct = (pred_chars == targets).float().mean()
    return correct.item()

def compute_sequence_accuracy(predictions, targets):
    pred_chars = predictions.argmax(dim=-1)  # [seq_len, batch_len]
    correct = (pred_chars == targets).all(dim=0).float().mean()
    return correct.item()

def train_until_converged(
        model,
        device,
        max_iters=50000,
        num_len_start=1,
        num_len_max=20,
        batch_len=16,
        lr=1e-3,
        print_every=100,
        eval_every=1000,
        save_every=5000,
        acc_threshold=0.99,
        loss_threshold=0.01,
        model_dir="./models",
        filename="ntm_add_nums"):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=lr*0.1)
    model.train()

    current_num_len = num_len_start
    loss_history = deque(maxlen=100)
    best_acc = 0.0
    best_path = os.path.join(model_dir, f"{filename}_best.pt")

    start_time = time.time()

    for it in range(1, max_iters + 1):
        x, y = generate_data(batch_len, current_num_len, vocab_len, device)
        x = x.to(device)
        y = y.to(device)

        model.reset(batch_size=batch_len)

        # Process input sequence
        for t in range(x.size(0)):
            _ = model.forward(x[t])  # [batch, vocab_len]

        # Generate answer
        answer_preds = []
        for t in range(y.size(0)):
            out = model.forward(torch.zeros(batch_len, vocab_len, device=device))
            answer_preds.append(out)
        answer_preds = torch.stack(answer_preds, dim=0)  # [seq_len, batch_len, vocab_len]

        # Compute loss and accuracy on the answer part
        loss = F.cross_entropy(
            answer_preds.reshape(-1, vocab_len),
            y.reshape(-1),
            label_smoothing=0.0
        )
        # Encourage sharp attention (penalize entropy)
        read_entropy = -(model.read_w * (model.read_w + 1e-8).log()).sum(-1).mean()
        write_entropy = -(model.write_w * (model.write_w + 1e-8).log()).sum(-1).mean()
        entropy_loss = 0.001 * (read_entropy + write_entropy)
        loss = loss + entropy_loss

        acc = compute_accuracy(answer_preds, y)
        seq_acc = compute_sequence_accuracy(answer_preds, y)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        # Eval model
        if it % eval_every == 0:
            model.eval()
            with torch.no_grad():
                print(f"\n{'='*60}")
                print(f"EVALUATION AT ITERATION {it}")
                print(f"{'='*60}")

                # Eval on current training size
                print(f"\n[Current Training Size: num_len={current_num_len}]")
                x_curr, y_curr = generate_data(batch_len, current_num_len, vocab_len, device)
                x_curr = x_curr.to(device)
                y_curr = y_curr.to(device)

                model.reset(batch_size=batch_len)
                for t in range(x_curr.size(0)):
                    _ = model.forward(x_curr[t])
                curr_preds = []
                for t in range(y_curr.size(0)):
                    out = model.forward(torch.zeros(batch_len, vocab_len, device=device))
                    curr_preds.append(out)
                curr_preds = torch.stack(curr_preds, dim=0)  # [seq_len, batch_len, vocab_len]
                curr_acc = compute_accuracy(curr_preds, y_curr)
                curr_seq_acc = compute_sequence_accuracy(curr_preds, y_curr)

                # Show example from current size
                prob, target, prediction = format_prediction(x_curr, y_curr, curr_preds, batch_idx=0)
                print(f"  Accuracy: {curr_acc:.4f}")
                print(f"  Sequence Accuracy: {curr_seq_acc:.4f}")
                print(f"  Example:")
                print(f"    Problem:    {prob}")
                print(f"    Target:     {prob}{target}")
                print(f"    Prediction: {prob}{prediction}")
                print(f"    Correct: {'✓' if target == prediction else '✗'}")

                # Eval on max size
                print(f"\n[Max Size: num_len={num_len_max}]")
                x_eval, y_eval = generate_data(batch_len, num_len_max, vocab_len, device)
                x_eval = x_eval.to(device)
                y_eval = y_eval.to(device)

                model.reset(batch_size=batch_len)
                for t in range(x_eval.size(0)):
                    _ = model.forward(x_eval[t])
                eval_preds = []
                for t in range(y_eval.size(0)):
                    out = model.forward(torch.zeros(batch_len, vocab_len, device=device))
                    eval_preds.append(out)
                eval_preds = torch.stack(eval_preds, dim=0)  # [seq_len, batch_len, vocab_len]
                eval_acc = compute_accuracy(eval_preds, y_eval)

                # Show example from max size
                prob, target, prediction = format_prediction(x_eval, y_eval, eval_preds, batch_idx=0)
                print(f"  Accuracy: {eval_acc:.4f}")
                print(f"  Example:")
                print(f"    Problem:    {prob}")
                print(f"    Target:     {prob}{target}")
                print(f"    Prediction: {prob}{prediction}")
                print(f"    Correct: {'✓' if target == prediction else '✗'}")

                # Test on multiple batches for more robust eval
                eval_accs = [eval_acc]
                for _ in range(9):
                    x_eval, y_eval = generate_data(batch_len, num_len_max, vocab_len, device)
                    x_eval = x_eval.to(device)
                    y_eval = y_eval.to(device)

                    model.reset(batch_size=batch_len)
                    for t in range(x_eval.size(0)):
                        _ = model.forward(x_eval[t])
                    eval_preds = []
                    for t in range(y_eval.size(0)):
                        out = model.forward(torch.zeros(batch_len, vocab_len, device=device))
                        eval_preds.append(out)
                    eval_preds = torch.stack(eval_preds, dim=0)
                    eval_accs.append(compute_accuracy(eval_preds, y_eval))

                avg_eval_acc = sum(eval_accs) / len(eval_accs)
                print(f"\n  Average eval accuracy over 10 batches: {avg_eval_acc:.4f}")

                # Compute eval loss for scheduler
                eval_loss = F.cross_entropy(
                    eval_preds.reshape(-1, vocab_len),
                    y_eval.reshape(-1)
                ).item()

                if avg_eval_acc > best_acc:
                    best_acc = avg_eval_acc
                    torch.save({
                        'iteration': it,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': eval_loss,
                        'acc': avg_eval_acc,
                        'num_len': current_num_len
                    }, best_path)
                    print(f"  ✓ Saved new best model (acc={best_acc:.4f})")

                print(f"{'='*60}\n")
            model.train()

        # Logging
        if it % print_every == 0:
            avg_loss = sum(loss_history) / len(loss_history) if loss_history else loss_val
            elapsed = time.time() - start_time
            print(f"\rIter {it}/{max_iters} | num_len={current_num_len} | "
                  f"Loss: {loss_val:.4f} (avg={avg_loss:.4f}) | Acc: {acc:.4f} | SeqAcc: {seq_acc:.4f} | "
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
                'best_acc': best_acc
            }, checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")

        # Curriculum training
        if seq_acc >= acc_threshold and loss_val < loss_threshold:
            if current_num_len >= num_len_max:
                print(f"\nReached target num_len={num_len_max} with acc={acc:.4f}, loss={loss_val:.4f}")
                break
            else:
                current_num_len += 1
                print(f"\n>>> Progressing to num_len={current_num_len} <<<\n")

    # Save final model
    final_path = os.path.join(model_dir, f"{filename}_final.pt")
    torch.save({
        'iteration': it,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_val,
        'acc': acc,
        'num_len': current_num_len,
        'best_acc': best_acc
    }, final_path)

    print(f"\n>>> Final model saved to: {final_path}")
    print(f">>> Best model (acc={best_acc*100:.2f}%) saved to: {best_path}")

    return final_path


if __name__ == "__main__":
    print(f"Start time: {datetime.datetime.now()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    print(f"Using device: {device}")
    print("DNC ADDITION TASK TRAINING")

    memory_length = 50
    model = DNC(
        input_size=vocab_len,
        memory_length=memory_length,
        controller_depth=2,
        controller_width=300,
        read_heads=4,
        write_heads=1
    )

    model_controller = "GRU"

    total_params = model.num_params()
    print(f"Model Configuration:")
    print(f"  Vocab size: {vocab_len}")
    print(f"  Memory: {memory_length} x {vocab_len}")
    print(f"  Controller: {model_controller}")
    print(f"  Controller depth: {model.controller_depth}")
    print(f"  Controller width: {model.controller_width}")
    print(f"  Read heads: {model.RH}")
    print(f"  Write heads: {model.WH}")
    print(f"  Total parameters: {total_params:,}")
    print("="*60)
    print()

    # Load checkpoint if it exists
    checkpoint_path = os.path.join("./models", "ntm_add_nums_final.pt")
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded from iteration {checkpoint.get('iteration', 'unknown')}")
        print()

    start_time = time.time()
    final_path = train_until_converged(
        model,
        device,
        max_iters=50000,
        num_len_start=1,
        num_len_max=20,
        batch_len=32,
        lr=1e-4,
        print_every=100,
        eval_every=1000,
        save_every=5000,
        acc_threshold=0.99,
        loss_threshold=0.1,
        model_dir="./models",
        filename="ntm_add_nums"
    )

    print("\n" + "="*60)
    print("TRAINING FINISHED")
    print(f"Final model: {final_path}")
    print(f"Time: {(time.time()-start_time)/60:.2f} min")
    print("="*60)

    print(f"\nEnd time: {datetime.datetime.now()}")
