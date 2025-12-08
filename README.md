# ai
Reproducing NN models from papers, and fiddling around with new ones.

I ran from a python venv in WSL2, so it kinda works on my machine.

AGI has been with us the whole time, in our own heads.

## ACT

Alex Graves: https://arxiv.org/abs/1603.08983

Adaptive Computation Time recurrent neural network. It allows the network to spend a variable amount of time on the same input vector, radically reducing the size of the net needed on hard combinatoric problems by learning an algorithm instead.

### Parity

With a careful training curriculum I was able to get the network to reliably train for longer sequences by starting with 1 and incrementing by 1 each time. So the vector starts out as mostly zeroes and the ACT net learns to classify with 85% accuracy. Filling only the start (or end) of the input vector makes it learn more slowly and is just bad for it's generalization. This reinforces the idea that a huge network and specialized input encodings are less important than a good architecture and training curriculum. This is the learning for input vectors of length 64, but I overkilled it for faster accuracy convergence.
TODO: figure out why it gets stuck at ~90% accuracy with 128 hidden size.

```
> python act_parity.py
Using device: cuda
Start time: 2025-11-30 14:27:02.041972
================================================================================
Parity Task - ACT Training
================================================================================
Input length: 64
Hidden size: 2000
Hidden type: RNN
Batch size: 128
Learning rate: 0.001
Time penalty (tau): 0.002
Max steps: 20
Target accuracy: 0.95
================================================================================
Model Configuration:
  Total parameters: 4,138,002
  
...

✓ Saved best model with accuracy: 0.958

================================================================================
✓ Target accuracy 0.950 reached!
Final test accuracy: 0.958
Total training time: 3.05 minutes
================================================================================

Running final evaluation...

================================================================================
FINAL RESULTS
================================================================================
Test Accuracy:  0.955
Test Loss:      0.1476
Average Ponder: 3.45
Average Steps:  2.5
Best Accuracy:  0.958
Total Time:     3.11 minutes
================================================================================

Testing on 5 examples:
1. Ones: 15 | Target: 0 | Pred: 0 ✓ | Ponder: 3.00 | Steps: 2
2. Ones: 19 | Target: 1 | Pred: 1 ✓ | Ponder: 4.00 | Steps: 3
3. Ones:  5 | Target: 0 | Pred: 0 ✓ | Ponder: 3.00 | Steps: 2
4. Ones: 19 | Target: 0 | Pred: 0 ✓ | Ponder: 4.00 | Steps: 3
5. Ones: 24 | Target: 1 | Pred: 1 ✓ | Ponder: 4.00 | Steps: 3

End time: 2025-11-30 14:30:09.699037
```

### Logic

```
> python act_logic.py
Using device: cuda
Start time: 2025-11-27 22:35:56.345850
================================================================================
Logic Task - ACT Training
================================================================================
Input size: 102
Hidden size: 300
Hidden type: LSTM
Batch size: 16
Max sequence length: 10
Max gates per vector: 10
Learning rate: 0.001
Time penalty (tau): 0.01
Max steps: 100
Target sequence accuracy: 0.95
================================================================================
Model Configuration:
  Total parameters: 486,602
  
...

✓ Saved best model with sequence accuracy: 0.988

================================================================================
✓ Target sequence accuracy 0.950 reached!
Final sequence accuracy: 0.988
Total training time: 174.01 minutes
================================================================================

Running final evaluation...

================================================================================
FINAL RESULTS
================================================================================
Bit Accuracy:      0.990
Sequence Accuracy: 0.960
Test Loss:         0.0383
Average Ponder:    6.00
Average Steps:     5.4
Best Seq Accuracy: 0.988
Total Time:        174.24 minutes
================================================================================

Testing on 5 example sequences:

Sequence 1 (length 4):
  t=0: Target=0 Pred=0 ✓ | Ponder=7.81 Steps=7
  t=1: Target=0 Pred=0 ✓ | Ponder=4.52 Steps=4
  t=2: Target=0 Pred=0 ✓ | Ponder=12.16 Steps=12
  t=3: Target=0 Pred=0 ✓ | Ponder=12.54 Steps=12
  Sequence: ✓ CORRECT

Sequence 2 (length 9):
  t=0: Target=1 Pred=1 ✓ | Ponder=8.39 Steps=8
  t=1: Target=1 Pred=1 ✓ | Ponder=7.27 Steps=7
  t=2: Target=1 Pred=1 ✓ | Ponder=10.74 Steps=10
  t=3: Target=1 Pred=1 ✓ | Ponder=12.51 Steps=12
  t=4: Target=0 Pred=0 ✓ | Ponder=4.70 Steps=4
  t=5: Target=0 Pred=0 ✓ | Ponder=14.51 Steps=14
  t=6: Target=1 Pred=1 ✓ | Ponder=12.07 Steps=12
  t=7: Target=1 Pred=1 ✓ | Ponder=13.18 Steps=13
  t=8: Target=1 Pred=1 ✓ | Ponder=13.37 Steps=13
  Sequence: ✓ CORRECT

Sequence 3 (length 1):
  t=0: Target=0 Pred=0 ✓ | Ponder=8.22 Steps=8
  Sequence: ✓ CORRECT

Sequence 4 (length 7):
  t=0: Target=1 Pred=1 ✓ | Ponder=4.84 Steps=4
  t=1: Target=0 Pred=0 ✓ | Ponder=11.20 Steps=11
  t=2: Target=0 Pred=0 ✓ | Ponder=4.41 Steps=4
  t=3: Target=1 Pred=1 ✓ | Ponder=12.65 Steps=12
  t=4: Target=0 Pred=0 ✓ | Ponder=11.02 Steps=11
  t=5: Target=1 Pred=1 ✓ | Ponder=21.04 Steps=21
  t=6: Target=1 Pred=1 ✓ | Ponder=9.54 Steps=9
  Sequence: ✓ CORRECT

Sequence 5 (length 4):
  t=0: Target=0 Pred=0 ✓ | Ponder=4.71 Steps=4
  t=1: Target=0 Pred=0 ✓ | Ponder=9.36 Steps=9
  t=2: Target=0 Pred=0 ✓ | Ponder=12.71 Steps=12
  t=3: Target=0 Pred=0 ✓ | Ponder=9.58 Steps=9
  Sequence: ✓ CORRECT

End time: 2025-11-28 01:30:11.985215
```

### Addition

Hard to get the model to train and takes too long. Making sequences length 2 at minimum is pretty necessary for this task. I loaded the model from file a few times because of interruptions but that's why we save the best model. It only got to 94% acc.

```
> python act_addition_grok.py
Using device: cuda
Start time: 2025-12-07 21:59:00.337549
================================================================================
Addition Task - ACT Training
================================================================================
Loading model: models/act_LSTM_addition_best.pt
Input size: 50
Output size: 66
Hidden size: 1024
Hidden type: LSTM
Best sequence accuracy = 0.94375
Batch size: 32
Max digits per number: 5
Max sequence length: 5
Learning rate: 0.0001
Time penalty (tau): 0.001
Max steps: 20
Target sequence accuracy: 0.95
================================================================================
 Total parameters: 4,480,067
```

## NTM

Alex Graves, et al: https://arxiv.org/abs/1410.5401

Neural Turing Machine

Really cool architecture, but it has a boring set of problems to train on. DNC network paper does more interesting problems.

### Copy Task

```
> python ntm_copy_task.py
============================================================
NTM BATCHED COPY TASK TRAINING
============================================================
Device: cuda

Model Configuration:
  Vocab size: 97
  Memory: 20 x 97
  Controller: RNN
  Total parameters: 344,422

...

[Iter  7900] Loss=0.8753 (avg=0.8803), Acc=97.81%, Len=20, GradNorm=0.89, LR=1.00e-03, Speed=5.8 it/s
  Target:  'Pqh-7R:`E{O9Mk7(TZ|o'
  Predict: 'Pqh-:R:`E{O9Mk7(TZ|o'

============================================================
=== TRAINING COMPLETE ===
Reached acc=99.06% and loss=0.8579 at max seq_len=20
============================================================

>>> Final model saved to: ./models/ntm_copy_final.pt
>>> Best model (acc=96.41%) saved to: ./models/ntm_copy_best.pt

============================================================
TRAINING FINISHED
Final model: ./models/ntm_copy_final.pt
============================================================
```

## DNC

TODO

## RTNTM

Recurrent Transformer NTM

I don't understand this at all, so, yeah.

### Copy Task

```
> python rtntm_copy_task.py
============================================================
TNTM COPY TASK TRAINING
============================================================
Device: cuda

Model Configuration:
  d_model: 100
  Memory: 20 x 97
  Read/Write heads: 2/1
  Total parameters: 221,200
  
...

>>> Curriculum: Increasing seq_len to 9 <<<

[Iter  1200] Loss=0.0131 (avg=0.1444), Acc=100.00%, Len=9, GradNorm=0.14, LR=1.00e-03, Speed=8.9 it/s
  Target:  '<05Ro/ggu'
  Predict: '<05Ro/ggu'


>>> Curriculum: Increasing seq_len to 11 <<<


>>> Curriculum: Increasing seq_len to 13 <<<


>>> Curriculum: Increasing seq_len to 15 <<<


>>> Curriculum: Increasing seq_len to 17 <<<


>>> Curriculum: Increasing seq_len to 19 <<<


>>> Curriculum: Increasing seq_len to 20 <<<


============================================================
=== TRAINING COMPLETE ===
Reached acc=100.00% and loss=0.0089 at max seq_len=20
============================================================

>>> Final model saved to: ./models/tntm_copy_final.pt
>>> Best model (acc=100.00%) saved to: ./models/tntm_copy_best.pt

============================================================
TRAINING FINISHED
Final model: ./models/tntm_copy_final.pt
============================================================
```
