# Training script for RTNTM on 4.2 Repeat Copy task from the NTM paper
# Hudson Andrew Smelski

import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import datetime
from collections import deque

from rtntm import *

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def generate_repeat_copy_batch(batch_size, seq_len, num_repeats, vocab_size, device):
    """
    Creates a batch for the repeat copy task.

    For RTNTM: We need to handle the extra channels (delimiter + repeat signal)
    Since RTNTM takes token indices, we'll use special tokens for delimiter and
    encode repeat count as additional special tokens.

    Input structure: [sequence] + [delimiter_token] + [repeat_count_token] + [blanks]
    Target: [sequence repeated num_repeats times] + [end_marker]

    Returns:
        input_seq: [total_input_len, batch] token indices
        target_seq: [target_len, batch] token indices
        num_repeats_batch: [batch] the actual repeat counts for each example
    """
    # Use vocab_size as base vocab
    # Reserve special tokens:
    # vocab_size - 1: end marker
    # vocab_size: delimiter
    # vocab_size + 1 to vocab_size + 10: repeat counts 1-10

    DELIMITER_TOKEN = vocab_size
    END_MARKER_TOKEN = vocab_size - 1
    BLANK_TOKEN = 0  # Use 0 for blank/padding

    # Random sequences (using vocab_size - 1 to leave room for special tokens)
    seqs = torch.randint(1, vocab_size - 1, (seq_len, batch_size), device=device)

    # Delimiter token
    delimiter = torch.full((1, batch_size), DELIMITER_TOKEN, dtype=torch.long, device=device)

    # Create repeat signal as special tokens
    if isinstance(num_repeats, int):
        repeats_batch = torch.full((batch_size,), num_repeats, dtype=torch.long, device=device)
    else:
        # Random repeats for each example in batch
        repeats_batch = torch.randint(num_repeats[0], num_repeats[1] + 1, (batch_size,), device=device)

    # Encode repeat count as special token (vocab_size + 1 + count)
    repeat_tokens = (vocab_size + 1 + repeats_batch - 1).unsqueeze(0)  # [1, batch]

    # Blanks for output phase
    max_repeats = int(repeats_batch.max().item())
    output_len = seq_len * max_repeats + 1  # +1 for end marker
    blanks = torch.full((output_len, batch_size), BLANK_TOKEN, dtype=torch.long, device=device)

    # Build input: sequence + delimiter + repeat_token + blanks
    input_seq = torch.cat([seqs, delimiter, repeat_tokens, blanks], dim=0)  # [total_len, batch]

    # Build target: sequence repeated num_repeats times + end marker
    target_list = []
    for b in range(batch_size):
        n_reps = int(repeats_batch[b].item())
        repeated = seqs[:, b].repeat(n_reps)  # [seq_len * n_reps]
        target_list.append(repeated)

    # Pad to max length and add end marker
    target_seq = torch.full((output_len, batch_size), BLANK_TOKEN, dtype=torch.long, device=device)
    for b in range(batch_size):
        n_reps = int(repeats_batch[b].item())
        target_len = seq_len * n_reps
        target_seq[:target_len, b] = target_list[b]
        target_seq[target_len, b] = END_MARKER_TOKEN
        # Rest remains BLANK_TOKEN (padding)

    return input_seq, target_seq, repeats_batch


def evaluate_repeat_copy_accuracy(predictions, targets, repeats_batch, seq_len, vocab_size):
    """
    Evaluate accuracy considering variable-length outputs.

    predictions: [output_len, batch, vocab_size]
    targets: [output_len, batch]
    repeats_batch: [batch] number of repeats for each example
    """
    END_MARKER_TOKEN = vocab_size - 1
    pred_tokens = torch.argmax(predictions, dim=-1)  # [output_len, batch]

    total_correct = 0
    total_tokens = 0
    end_marker_correct = 0

    for b in range(targets.size(1)):
        n_reps = repeats_batch[b].item()
        target_len = seq_len * n_reps + 1  # +1 for end marker

        # Check sequence accuracy
        correct = (pred_tokens[:target_len, b] == targets[:target_len, b]).sum().item()
        total_correct += correct
        total_tokens += target_len

        # Check if end marker is placed correctly
        if target_len <= targets.size(0):
            end_marker_correct += (pred_tokens[target_len - 1, b] == END_MARKER_TOKEN).item()

    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    end_marker_acc = end_marker_correct / targets.size(1)

    return accuracy, end_marker_acc


def train_repeat_copy_task(
    model,
    device,
    max_iters=50000,
    seq_len_start=3,
    seq_len_max=10,
    repeats_start=2,
    repeats_max=10,
    batch_size=16,
    lr=1e-4,
    print_every=100,
    eval_every=500,
    save_every=2000,
    acc_threshold=0.98,
    model_dir="./models_rtntm_repeat",
):
    ensure_dir(model_dir)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    model.train()

    current_seq_len = seq_len_start
    current_max_repeats = repeats_start
    loss_history = deque(maxlen=100)
    best_acc = 0.0
    best_path = os.path.join(model_dir, "rtntm_repeat_copy_best.pt")

    start_time = time.time()

    for it in range(1, max_iters + 1):
        # Generate batch with random repeats
        input_seq, target_seq, repeats_batch = generate_repeat_copy_batch(
            batch_size, current_seq_len, (1, current_max_repeats), vocab_size, device
        )

        # Initialize RTNTM state
        state = model.init_state(batch_size=batch_size, device=device)

        # Forward pass through sequence
        outputs = []
        for t in range(input_seq.size(0)):
            logits, state = model.step(input_seq[t], state)  # [batch, vocab_size+11]
            outputs.append(logits)

        outputs = torch.stack(outputs, dim=0)  # [total_len, batch, vocab_size+11]

        # Extract predictions for output phase (only use first vocab_size logits)
        output_start = current_seq_len + 2  # after sequence + delimiter + repeat_token
        pred_logits = outputs[output_start:, :, :vocab_size]  # [output_len, batch, vocab_size]

        # Loss
        loss = F.cross_entropy(
            pred_logits.reshape(-1, vocab_size),
            target_seq.reshape(-1),
            ignore_index=0,  # ignore padding (BLANK_TOKEN)
            label_smoothing=0.05
        )

        # Attention regularization (encourage sharp attention)
        read_w = state['read_w']  # [batch, RH, N]
        write_w = state['write_w']  # [batch, WH, N]
        eps = 1e-8
        read_entropy = -(read_w * (read_w + eps).log()).sum(-1).mean()
        write_entropy = -(write_w * (write_w + eps).log()).sum(-1).mean()
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
                # Test on slightly longer sequences and more repeats
                eval_seq_len = min(current_seq_len + 2, seq_len_max)
                eval_repeats = min(current_max_repeats + 2, repeats_max)

                input_eval, target_eval, repeats_eval = generate_repeat_copy_batch(
                    batch_size, eval_seq_len, eval_repeats, vocab_size, device
                )

                state_eval = model.init_state(batch_size=batch_size, device=device)
                outputs_eval = []
                for t in range(input_eval.size(0)):
                    logits, state_eval = model.step(input_eval[t], state_eval)
                    outputs_eval.append(logits)

                outputs_eval = torch.stack(outputs_eval, dim=0)
                output_start_eval = eval_seq_len + 2
                pred_eval = outputs_eval[output_start_eval:, :, :vocab_size]

                eval_loss = F.cross_entropy(
                    pred_eval.reshape(-1, vocab_size),
                    target_eval.reshape(-1),
                    ignore_index=0
                ).item()

                eval_acc, end_acc = evaluate_repeat_copy_accuracy(
                    pred_eval, target_eval, repeats_eval, eval_seq_len, vocab_size
                )

                # Get memory stats
                stats = model.get_memory_usage_stats(state_eval)

                print(f"\n[EVAL Iter {it}] Loss={eval_loss:.4f}, Acc={eval_acc*100:.2f}%, "
                      f"EndMarker={end_acc*100:.2f}% (len={eval_seq_len}, repeats={eval_repeats})")
                print(f"  Memory: KV={stats['kv_cache_len']}/{stats['kv_cache_max']} "
                      f"(util={stats['kv_utilization']:.1%}), "
                      f"Mem_std={stats['memory_std']:.4f}, "
                      f"Read_sharp={stats['read_sharpness']:.2f}, "
                      f"Write_sharp={stats['write_sharpness']:.2f}")

                # Show example (first in batch)
                n_reps = repeats_eval[0].item()
                target_len = eval_seq_len * n_reps + 1
                pred_tokens = torch.argmax(pred_eval[:target_len, 0, :], dim=-1).cpu().tolist()
                target_tokens = target_eval[:target_len, 0].cpu().tolist()
                pred_str = ''.join(idx_to_char.get(i, '?') for i in pred_tokens)
                target_str = ''.join(idx_to_char.get(i, '?') for i in target_tokens)
                print(f"  Repeats: {n_reps}")
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
                        'end_acc': end_acc,
                        'seq_len': current_seq_len,
                        'max_repeats': current_max_repeats
                    }, best_path)
                    print(f">>> New best accuracy: {eval_acc*100:.2f}% - Saved to {best_path}")

            model.train()

        # Training metrics
        acc, end_acc = evaluate_repeat_copy_accuracy(
            pred_logits, target_seq, repeats_batch, current_seq_len, vocab_size
        )

        # Logging
        if it % print_every == 0:
            avg_loss = sum(loss_history) / len(loss_history) if loss_history else loss_val
            elapsed = time.time() - start_time
            iter_per_sec = it / elapsed

            # Get current memory stats
            stats = model.get_memory_usage_stats(state)

            print(f"[Iter {it:5d}] Loss={loss_val:.4f} (avg={avg_loss:.4f}), Acc={acc*100:.2f}%, "
                  f"EndMarker={end_acc*100:.2f}%, Len={current_seq_len}, Repeats=1-{current_max_repeats}, "
                  f"KV={stats['kv_cache_len']}/{stats['kv_cache_max']}, "
                  f"GradNorm={grad_norm:.2f}, LR={optimizer.param_groups[0]['lr']:.2e}")

            # Show example (first in batch)
            with torch.no_grad():
                n_reps = repeats_batch[0].item()
                target_len = current_seq_len * n_reps + 1
                pred_tokens = torch.argmax(pred_logits[:target_len, 0, :], dim=-1).cpu().tolist()
                target_tokens = target_seq[:target_len, 0].cpu().tolist()
                pred_str = ''.join(idx_to_char.get(i, '?') for i in pred_tokens)
                target_str = ''.join(idx_to_char.get(i, '?') for i in target_tokens)
                print(f"    Repeats={n_reps}: Target={repr(target_str[:30])}..., Pred={repr(pred_str[:30])}...")

        # Save checkpoint
        if it % save_every == 0:
            ckpt_path = os.path.join(model_dir, f"rtntm_repeat_copy_iter_{it}.pt")
            torch.save({
                'iteration': it,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss_val,
                'acc': acc,
                'end_acc': end_acc,
                'seq_len': current_seq_len,
                'max_repeats': current_max_repeats,
                'best_acc': best_acc
            }, ckpt_path)
            print(f">>> Checkpoint saved: {ckpt_path}")

        # Curriculum learning
        if acc >= acc_threshold and end_acc >= 0.8:
            if current_seq_len < seq_len_max or current_max_repeats < repeats_max:
                if current_seq_len < seq_len_max:
                    current_seq_len = min(current_seq_len + 1, seq_len_max)
                if current_max_repeats < repeats_max:
                    current_max_repeats = min(current_max_repeats + 1, repeats_max)
                print(f"\n>>> Curriculum: seq_len={current_seq_len}, max_repeats={current_max_repeats} <<<\n")
            else:
                print("\n" + "="*60)
                print("=== TRAINING COMPLETE ===")
                print(f"Reached acc={acc*100:.2f}%, end_acc={end_acc*100:.2f}%")
                print(f"Max seq_len={seq_len_max}, max_repeats={repeats_max}")
                print("="*60)
                break

    # Final save
    final_path = os.path.join(model_dir, "rtntm_repeat_copy_final.pt")
    torch.save({
        'iteration': it,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_val,
        'acc': acc,
        'end_acc': end_acc,
        'seq_len': current_seq_len,
        'max_repeats': current_max_repeats,
        'best_acc': best_acc
    }, final_path)
    print(f"\n>>> Final model saved to: {final_path}")
    print(f">>> Best model (acc={best_acc*100:.2f}%) saved to: {best_path}")

    return final_path


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*70)
    print("RTNTM REPEAT COPY TASK TRAINING")
    print("="*70)
    print(f"Device: {device}")
    print(f"Start time: {datetime.datetime.now()}\n")

    # Need vocab_size + special tokens:
    # vocab_size: base vocabulary (digits + letters + punctuation)
    # +1: delimiter token
    # +10: repeat count tokens (1-10)
    extended_vocab_size = vocab_size + 11

    # Model hyperparameters
    memory_N = 128
    memory_M = extended_vocab_size
    d_model = extended_vocab_size*2
    n_heads = 4
    controller_window = 8  # Key parameter: limits internal memory
    read_heads = 1
    write_heads = 1

    print("Model Configuration:")
    print(f"  Base vocab size: {vocab_size}")
    print(f"  Extended vocab (with special tokens): {extended_vocab_size}")
    print(f"  d_model: {d_model}")
    print(f"  Memory: {memory_N} × {memory_M}")
    print(f"  Controller window: {controller_window} (bounded KV cache)")
    print(f"  Read heads: {read_heads}, Write heads: {write_heads}")
    print(f"  Attention heads: {n_heads}")
    print()

    model = RTNTM(
        vocab_size=extended_vocab_size,
        d_model=d_model,
        memory_N=memory_N,
        memory_M=memory_M,
        n_heads=n_heads,
        controller_window=controller_window,
        read_heads=read_heads,
        write_heads=write_heads,
        shift_width=3,
        use_summary=False,
        dropout=0.1
    ).to(device)

    total_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")

    # Calculate wide input dimension
    wide_input_dim = d_model + read_heads * memory_M
    print(f"Wide input dimension: {d_model} + {read_heads}×{memory_M} = {wide_input_dim}")
    print()

    training_start = time.time()

    final_path = train_repeat_copy_task(
        model,
        device=device,
        max_iters=100000,
        seq_len_start=10,
        seq_len_max=10,
        repeats_start=1,
        repeats_max=10,
        batch_size=16,
        lr=1e-4,  # Lower LR for Transformer
        print_every=100,
        eval_every=500,
        save_every=2000,
        acc_threshold=0.98,
        model_dir="./models_rtntm_repeat"
    )

    training_time = time.time() - training_start

    print("\n" + "="*70)
    print("TRAINING FINISHED")
    print(f"Final model: {final_path}")
    print(f"Training time: {training_time/60:.2f} minutes")
    print(f"End time: {datetime.datetime.now()}")
    print("="*70)
