"""
Training script for ACT on Addition Task
Based on Section 3.3 of the ACT paper
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import datetime
from pathlib import Path
from act import ACT  # Assuming your ACT class is in act.py

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Start time: {datetime.datetime.now()}")

def generate_data(batch_size, max_digits, max_seq_len, device,
    curriculum_min_digits=None, curriculum_max_digits=None,
    curriculum_min_seq_len=None, curriculum_max_seq_len=None):
    """
    Generate addition sequences following Section 3.3 of ACT paper.
    Args:
        batch_size: Number of sequences
        max_digits: Maximum digits for input size (fixed at 5)
        max_seq_len: Maximum sequence length (5 in paper)
        device: torch device
        curriculum_max_digits: If provided, limit random digits to this (for curriculum)
        curriculum_max_seq_len: If provided, limit random seq len to this (for curriculum)
    Returns:
        X: Input tensor (seq_len, batch, input_size)
        Y: Target tensor (seq_len, batch, output_size)
        seq_lengths: Actual sequence length per batch
        num_digits_list: Number of digits per input (for analysis)
    """
    # Use curriculum limits if provided, otherwise use max
    actual_min_digits = curriculum_min_digits if curriculum_min_digits is not None else 1
    actual_max_digits = curriculum_max_digits if curriculum_max_digits is not None else max_digits
    actual_min_seq_len = curriculum_min_seq_len if curriculum_min_seq_len is not None else 2
    actual_max_seq_len = curriculum_max_seq_len if curriculum_max_seq_len is not None else max_seq_len

    # Random sequence length (2 to actual_max_seq_len >= 2)
    seq_lengths = torch.randint(2, actual_max_seq_len + 1, (batch_size,), device=device)
    actual_seq_len = seq_lengths.max().item()

    input_size = max_digits * 10
    output_size = (max_digits + 1) * 11
    X = torch.zeros(actual_seq_len, batch_size, input_size, device=device)
    Y = torch.zeros(actual_seq_len, batch_size, output_size, device=device)
    num_digits_list = []
    for b in range(batch_size):
        seq_len = seq_lengths[b].item()
        cumulative_sum = 0
        batch_digits = []
        for t in range(seq_len):
            # Random number of digits (1 to actual_max_digits for curriculum)
            num_digits = torch.randint(1, actual_max_digits + 1, (1,)).item()
            batch_digits.append(num_digits)
            # Generate random number with that many digits
            if num_digits == 1:
                number = torch.randint(0, 10, (1,)).item()
            else:
                # Ensure first digit is not 0
                first_digit = torch.randint(1, 10, (1,)).item()
                remaining = torch.randint(0, 10**(num_digits-1), (1,)).item()
                number = first_digit * (10**(num_digits-1)) + remaining
            # Convert number to one-hot encoding
            digits_str = str(number).zfill(num_digits)
            for d_idx, digit_char in enumerate(digits_str):
                digit = int(digit_char)
                X[t, b, d_idx * 10 + digit] = 1.0
            # Update cumulative sum
            cumulative_sum += number
            # No target for first timestep, set all to end marker (class 10)
            if t == 0:
                for out_digit in range(max_digits + 1):
                    Y[t, b, out_digit * 11 + 10] = 1.0
            else:
                # Convert sum to target
                sum_str = str(cumulative_sum)
                sum_digits = len(sum_str)
                # Encode each digit of the sum
                for d_idx in range(sum_digits):
                    digit = int(sum_str[d_idx])
                    Y[t, b, d_idx * 11 + digit] = 1.0
                # Remaining positions get end marker (class 10)
                for d_idx in range(sum_digits, max_digits + 1):
                    Y[t, b, d_idx * 11 + 10] = 1.0
        num_digits_list.append(batch_digits)

    return X, Y, seq_lengths, num_digits_list

def decode_input(input_vector, max_digits):
    """
    Decode input vector to get the number.

    Args:
        input_vector: (max_digits * 10) dimensional input (one-hot encoded)
        max_digits: Maximum number of digits (5 in paper)

    Returns:
        The number as integer
    """
    input_vector = input_vector.view(max_digits, 10)

    # Find which digits are active (not all zeros)
    digits = []
    for d in range(max_digits):
        digit_one_hot = input_vector[d]
        if digit_one_hot.sum() > 0:  # This position has a digit
            digit = torch.argmax(digit_one_hot).item()
            digits.append(digit)
        else:
            break  # No more digits

    if len(digits) == 0:
        return 0

    # Convert to number
    number = 0
    for digit in digits:
        number = number * 10 + digit

    return number


def decode_output(output_logits, max_digits):
    """
    Decode model output to get the predicted number.

    Args:
        output_logits: ((max_digits+1) * 11) dimensional output
        max_digits: Maximum number of digits (5 in paper, +1 for output = 6)

    Returns:
        Predicted number as integer
    """
    output_logits = output_logits.view(max_digits + 1, 11)
    predictions = torch.argmax(output_logits, dim=1)

    # Find where end marker (10) starts
    digits = []
    for pred in predictions:
        if pred == 10:
            break
        digits.append(pred.item())

    if len(digits) == 0:
        return 0

    # Convert to number
    number = 0
    for digit in digits:
        number = number * 10 + digit

    return number

def compute_accuracy(output, target, seq_lengths, batch_size, max_digits):
    """
    Compute per-digit accuracy and sequence accuracy.
    """
    total_correct_digits = 0
    total_digits = 0
    total_correct_sequences = 0
    for b in range(batch_size):
        seq_len = seq_lengths[b].item()
        sequence_correct = True
        for t in range(1, seq_len):  # Skip first timestep (no target)
            pred_logits = output[t, b]
            target_logits = target[t, b]

            pred_decoded = pred_logits.view(max_digits + 1, 11)
            target_decoded = target_logits.view(max_digits + 1, 11)

            pred_digits = torch.argmax(pred_decoded, dim=1)
            target_digits = torch.argmax(target_decoded, dim=1)

            # Count correct digits (only actual digits, not end markers)
            for d in range(max_digits + 1):
                if target_digits[d] != 10:  # Not end marker
                    total_digits += 1
                    if pred_digits[d] == target_digits[d]:
                        total_correct_digits += 1
                    else:
                        sequence_correct = False
                else:
                    # Check if prediction also has end marker from this point
                    if pred_digits[d] != 10:
                        sequence_correct = False
                    # No need to check further positions for this (t,b)
                    break
        if sequence_correct:
            total_correct_sequences += 1
    digit_accuracy = total_correct_digits / total_digits if total_digits > 0 else 0
    sequence_accuracy = total_correct_sequences / batch_size
    return digit_accuracy, sequence_accuracy

def train_epoch(model, optimizer, criterion, batch_size, max_digits, max_seq_len, num_batches, device,
                curriculum_min_digits=None, curriculum_max_digits=None,
                curriculum_min_seq_len=None, curriculum_max_seq_len=None):
    model.train()
    total_digit_acc = 0
    total_seq_acc = 0
    total_task_loss = 0
    total_ponder = 0
    total_steps = 0
    for _ in range(num_batches):
        X, Y, seq_lengths, _ = generate_data(
            batch_size, max_digits, max_seq_len, device,
            curriculum_min_digits=curriculum_min_digits,
            curriculum_max_digits=curriculum_max_digits,
            curriculum_min_seq_len=curriculum_min_seq_len,
            curriculum_max_seq_len=curriculum_max_seq_len)
        optimizer.zero_grad()

        output, ponder_costs, steps = model(X)
        # Compute loss (skip first timestep as it has no target)
        loss = 0
        num_positions = 0
        for t in range(1, output.size(0)):
            mask = (t < seq_lengths)  # (batch_size)
            if not mask.any():
                continue
            out_t = output[t].view(batch_size, max_digits + 1, 11)  # batch, pos, classes
            target_t = Y[t].view(batch_size, max_digits + 1, 11)
            #print(target_t)
            #target_classes = torch.argmax(target_t, dim=2)  # batch, pos

            for d in range(max_digits + 1):
                #print(out_t[:, d, :])
                #exit()
                #loss_d = criterion(out_t[:, d, :], target_classes[:, d])  # (batch_size) due to reduction='none'
                loss_d = criterion(out_t[:, d, :], target_t[:, d])  # (batch_size) due to reduction='none'
                loss += (loss_d * mask.float()).sum()
            num_positions += mask.sum().item() * (max_digits + 1)
        task_loss = loss / num_positions if num_positions > 0 else loss
        ponder_mean = ponder_costs.mean()
        #logit_penalty = 0.01 * (output ** 2).mean()
        total_loss = task_loss + model.tau * ponder_mean# + logit_penalty

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            digit_acc, seq_acc = compute_accuracy(output, Y, seq_lengths, batch_size, max_digits)
        total_digit_acc += digit_acc
        total_seq_acc += seq_acc
        total_task_loss += total_loss.item() #Changed to Print Total loss
        total_ponder += ponder_mean.item()
        total_steps += steps.mean().item()
    num_batches = max(num_batches, 1)
    return {
        'digit_accuracy': total_digit_acc / num_batches,
        'sequence_accuracy': total_seq_acc / num_batches,
        'task_loss': total_task_loss / num_batches,
        'ponder': total_ponder / num_batches,
        'steps': total_steps / num_batches
    }

def evaluate(model, criterion, batch_size, max_digits, max_seq_len, num_batches, device):
    model.eval()
    total_digit_acc = 0
    total_seq_acc = 0
    total_task_loss = 0
    total_ponder = 0
    total_steps = 0
    with torch.no_grad():
        for _ in range(num_batches):
            X, Y, seq_lengths, _ = generate_data(batch_size, max_digits, max_seq_len, device)
            output, ponder_costs, steps = model(X)

            loss = 0
            num_positions = 0
            for t in range(1, output.size(0)):
                mask = (t < seq_lengths)  # (batch_size)
                if not mask.any():
                    continue
                out_t = output[t].view(batch_size, max_digits + 1, 11)  # batch, pos, classes
                target_t = Y[t].view(batch_size, max_digits + 1, 11)
                #target_classes = torch.argmax(target_t, dim=2)  # batch, pos
                for d in range(max_digits + 1):
                    #loss_d = criterion(out_t[:, d, :], target_classes[:, d])  # (batch_size)
                    loss_d = criterion(out_t[:, d, :], target_t[:, d])
                    loss += (loss_d * mask.float()).sum()
                num_positions += mask.sum().item() * (max_digits + 1)
            task_loss = loss / num_positions if num_positions > 0 else loss
            ponder_mean = ponder_costs.mean()

            digit_acc, seq_acc = compute_accuracy(output, Y, seq_lengths, batch_size, max_digits)
            total_digit_acc += digit_acc
            total_seq_acc += seq_acc
            total_task_loss += task_loss.item()
            total_ponder += ponder_mean.item()
            total_steps += steps.mean().item()
    num_batches = max(num_batches, 1)
    return {
        'digit_accuracy': total_digit_acc / num_batches,
        'sequence_accuracy': total_seq_acc / num_batches,
        'task_loss': total_task_loss / num_batches,
        'ponder': total_ponder / num_batches,
        'steps': total_steps / num_batches
    }

if __name__ == "__main__":
    # Hyperparameters (Section 3.3)
    max_digits = 5  # 1-5 digits per number
    input_size = max_digits * 10  # 50
    output_size = (max_digits + 1) * 11  # 66 (6 digits * 11 classes)
    hidden_size = 512
    hidden_type = "LSTM"
    max_steps = 20
    lr = 1e-4
    tau = 0.001

    # Training parameters
    batch_size = 32
    max_seq_len = 5  # 1-5 vectors per sequence
    num_iterations = 500000
    train_batches = 10
    eval_batches = 50
    eval_interval = 100
    target_sequence_accuracy = 0.95
    min_lr = 1e-6
    weight_decay = 1e-6
    best_sequence_accuracy = 0

    # Curriculum learning
    current_min_digits = 1
    current_max_digits = 5
    current_min_seq_len = 1
    current_max_seq_len = 5
    curriculum_digit_acc = 1.0
    curriculum_threshold = 1.0
    max_curriculum_count = 0
    reset_curriculum = False

    print("=" * 80)
    print("Addition Task - ACT Training")
    print("=" * 80)

    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / f"act_{hidden_type}_addition_best_best.pt"
    if True:
        print(f"Loading model: {save_path}")
        model = torch.load(save_path, map_location=device, weights_only=True)

        checkpoint = torch.load(save_path)
        config = checkpoint['config']

        model = ACT(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            output_size=config['output_size'],
            hidden_type=config['hidden_type'],
            max_steps=max_steps,
            tau=tau, logit_smooth= True, dim1 = 6, dim2 = 11
        ).to(device)

        print(f"Input size: {config['input_size']}")
        print(f"Output size: {config['output_size']}")
        print(f"Hidden size: {config['hidden_size']}")
        print(f"Hidden type: {config['hidden_type']}")

        model.load_state_dict(checkpoint['model_state_dict'])
        best_sequence_accuracy = checkpoint['sequence_accuracy']
        print(f"Best sequence accuracy = {best_sequence_accuracy}")
        #TODO: load the criterion and scheduler dicts
    else:
        model = ACT(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            hidden_type=hidden_type,
            max_steps=max_steps,
            tau=tau, logit_smooth=True, dim1 = 6, dim2 = 11
        ).to(device)

        print(f"Input size: {input_size}")
        print(f"Output size: {output_size}")
        print(f"Hidden size: {hidden_size}")
        print(f"Hidden type: {hidden_type}")

    print(f"Batch size: {batch_size}")
    print(f"Max digits per number: {max_digits}")
    print(f"Max sequence length: {max_seq_len}")
    print(f"Learning rate: {lr}")
    print(f"Time penalty (tau): {tau}")
    print(f"Max steps: {max_steps}")
    print(f"Target sequence accuracy: {target_sequence_accuracy}")
    print("=" * 80)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" Total parameters: {total_params:,}")
    print("\n")

    criterion = nn.CrossEntropyLoss(reduction='none')  # Use none for masking flexibility
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15, threshold=0.001, min_lr=min_lr)

    start_time = time.time()
    epoch_time = 0
    for iteration in range(num_iterations):
        epoch_start = time.time()
        train_metrics = train_epoch(
            model, optimizer, criterion, batch_size,
            max_digits, max_seq_len, train_batches, device,
            curriculum_min_digits=current_min_digits,
            curriculum_max_digits=current_max_digits,
            curriculum_min_seq_len=current_min_seq_len,
            curriculum_max_seq_len=current_max_seq_len)
        epoch_time += time.time() - epoch_start

        print(f"\rIter {iteration:6d} | "
              f"Digit Acc: {train_metrics['digit_accuracy']:.3f} | "
              f"Seq Acc: {train_metrics['sequence_accuracy']:.3f} | "
              f"Loss: {train_metrics['task_loss']:6.3f} | "
              f"Ponder: {train_metrics['ponder']:5.2f} | "
              f"Steps: {train_metrics['steps']:4.1f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.0e} | "
              f"Time: {epoch_time:.2f}", end="")

        #Curriculum
        if reset_curriculum:
            #Reset the curriculum
            current_min_digits = 1
            current_min_seq_len = 2
            current_max_digits = 1
            current_max_seq_len = 2
            reset_curriculum = False
        if current_min_seq_len == max_seq_len and current_min_digits == max_digits:
            max_curriculum_count += 1
            if max_curriculum_count % 100 == 0:
                reset_curriculum = True

        if train_metrics['digit_accuracy'] >= curriculum_digit_acc and current_min_digits < max_digits:
            if current_min_digits < max_digits:
                current_min_digits += 1
                #print(f"\n✓ Curriculum: min_digits → {current_min_digits}")
            if current_max_digits < max_digits:
                current_max_digits += 1
                #print(f"✓ Curriculum: max_digits → {current_max_digits}")
            print(f"\n✓ Curriculum: min_digits → {current_min_digits}, max_digits → {current_max_digits}")
        elif train_metrics['sequence_accuracy'] >= curriculum_threshold and current_max_seq_len < current_max_seq_len:
            current_min_digits = 1
            current_max_digits = 1
            if current_min_seq_len < current_max_seq_len:
                current_min_seq_len += 1
                print(f"\n✓ Curriculum: min_seq_len → {current_min_seq_len}")
            elif current_max_seq_len < max_seq_len:
                current_max_seq_len += 1
                print(f"\n✓ Curriculum: max_seq_len → {current_max_seq_len}")

        # Periodic evaluation
        if (iteration + 1) % eval_interval == 0:
            epoch_time = 0

            eval_start = time.time()
            eval_metrics = evaluate(
            model, criterion, batch_size,
            max_digits, max_seq_len, eval_batches, device)
            eval_time = time.time() - eval_start

            print(f"\n{'='*80}")
            print(f"EVAL {iteration:6d} | "
                  f"Digit Acc: {eval_metrics['digit_accuracy']:.3f} | "
                  f"Seq Acc: {eval_metrics['sequence_accuracy']:.3f} | "
                  f"Loss: {eval_metrics['task_loss']:6.3f} | "
                  f"Ponder: {eval_metrics['ponder']:5.2f} | "
                  f"Steps: {eval_metrics['steps']:4.1f} | "
                  f"Time: {eval_time:.2f}")

            scheduler.step(eval_metrics['sequence_accuracy'])


            if eval_metrics['sequence_accuracy'] > best_sequence_accuracy:
                best_sequence_accuracy = eval_metrics['sequence_accuracy']
                save_path = save_dir / f"act_{hidden_type}_addition_best.pt"
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'sequence_accuracy': best_sequence_accuracy,
                    'config': {
                        'input_size': input_size,
                        'output_size': output_size,
                        'hidden_size': hidden_size,
                        'hidden_type': hidden_type,
                        'tau': tau,
                        'lr': lr
                    }
                }, save_path)
                print(f"✓ Saved best model with sequence accuracy: {best_sequence_accuracy:.3f}\n")

            if eval_metrics['sequence_accuracy'] >= target_sequence_accuracy:
                print(f"\n{'='*80}")
                print(f"✓ Target sequence accuracy {target_sequence_accuracy:.3f} reached!")
                print(f"Final sequence accuracy: {eval_metrics['sequence_accuracy']:.3f}")
                print(f"Total training time: {(time.time() - start_time)/60:.2f} minutes")
                print(f"{'='*80}")
                break

    print("\nRunning final evaluation...")
    final_metrics = evaluate(
        model, criterion, batch_size,
        max_digits, max_seq_len, eval_batches * 5, device)
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Digit Accuracy: {final_metrics['digit_accuracy']:.3f}")
    print(f"Sequence Accuracy: {final_metrics['sequence_accuracy']:.3f}")
    print(f"Test Loss: {final_metrics['task_loss']:.4f}")
    print(f"Average Ponder: {final_metrics['ponder']:.2f}")
    print(f"Average Steps: {final_metrics['steps']:.1f}")
    print(f"Best Seq Accuracy: {best_sequence_accuracy:.3f}")
    print(f"Total Time: {(time.time() - start_time)/60:.2f} minutes")
    print(f"{'='*80}")

    print("\nTesting on 5 example sequences:")
    for i in range(5):
        X, Y, seq_lengths, num_digits_list = generate_data(1, max_digits, max_seq_len, device)
        seq_len = seq_lengths[0].item()
        model.eval()
        with torch.no_grad():
            output, ponder, steps = model(X)
        print(f"\nSequence {i+1} (length {seq_len}):")
        cumulative_sum = 0
        all_correct = True
        for t in range(seq_len):
            # Decode input number
            input_number = decode_input(X[t, 0], max_digits)
            cumulative_sum += input_number
            if t == 0:
                print(f" t={t}: Input={input_number} (no target) | "
                      f"Ponder={ponder[t, 0].item():.2f} Steps={steps[t, 0].item():.0f}")
            else:
                # Decode target and prediction
                target_number = decode_output(Y[t, 0], max_digits)
                pred_number = decode_output(output[t, 0], max_digits)
                correct = "✓" if pred_number == target_number else "✗"
                if pred_number != target_number:
                    all_correct = False
                print(f" t={t}: Input={input_number} | "
                      f"Expected Sum={target_number} | "
                      f"Predicted={pred_number} {correct} | "
                      f"Ponder={ponder[t, 0].item():.2f} Steps={steps[t, 0].item():.0f}")
        seq_result = "✓ CORRECT" if all_correct else "✗ INCORRECT"
        print(f" Sequence: {seq_result}")
    print(f"\nEnd time: {datetime.datetime.now()}")
