# ai
Reproducing NN models from papers, and fiddling around with new ones.

I ran from a python venv in WSL2, so it kinda works on my machine.

AGI has been with us the whole time, in our own heads.

## ACT

Alex Graves: https://arxiv.org/abs/1603.08983

Adaptive Computation Time recurrent neural network. It allows the network to spend a variable amount of time on the same input vector, radically reducing the size of the net needed on hard combinatoric problems by learning an algorithm instead.

### Parity

With a careful training curriculum I was able to get the network to reliably train for longer sequences by starting with 1 and incrementing by 1 each time. So the vector starts out as mostly zeroes and the ACT net learns to classify with 85% accuracy. Filling only the start (or end) of the input vector makes it learn more slowly and is just bad for it's generalization. This reinforces the idea that a huge network and specialized input encodings are less important than a good architecture and training curriculum. This is the learning for input vectors of length 64, but I overkilled it for faster accuracy convergence.

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
Using device: cuda
Start time: 2025-12-08 18:18:22.327769
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

============================================================
=== TRAINING COMPLETE ===
Reached acc=99.06% and loss=0.8745 at max seq_len=20
============================================================

>>> Final model saved to: ./models/ntm_copy_final.pt
>>> Best model (acc=97.03%) saved to: ./models/ntm_copy_best.pt

============================================================
TRAINING FINISHED
Final model: ./models/ntm_copy_final.pt
Time: 13.79 min
============================================================

End time: 2025-12-08 18:32:09.67591
```

### Repeat Copy Task

```
> python ntm_repeat_copy_task.py
============================================================
NTM REPEAT COPY TASK TRAINING
============================================================
Device: cuda

Model Configuration:
  Vocab size: 97 (+2 for task channels)
  Memory: 128 x 99
  Controller: RNN
  Total parameters: 358,450
  
...

>>> Curriculum: seq_len=10, max_repeats=10 <<<


[EVAL Iter 18500] Loss=0.2051, Acc=94.74%, EndMarker=100.00% (len=10, repeats=10)
  Repeats: 10
  Eval Target:  '>"[g|O\\&27>"[g|O\\&27>"[g|O\\&27>"[g|O\\&27>"[g|O\\&27>"[g|O\\&27>"[g|O\\&27>"[g|O\\&27>"[g|O\\&27>"[g|O\\&27\n'
  Eval Predict: '>"[g|O\\&27>"[g|O\\&27>"[g|O\\&27>"[g|O\\&27\n"[g|O\\&27\n"[g|O\\&27\n"[g|O\\&27\n"[g|O\\&27\n"[g|O\\&27\n"[g|O\\&27\n'

>>> New best accuracy: 94.74% - Saved to ./models_repeat/ntm_repeat_copy_best.pt
[Iter 18500] Loss=0.5195 (avg=0.5434), Acc=96.39%, EndMarker=68.75%, Len=10, Repeats=1-10, GradNorm=1.43, LR=2.50e-04 |   Repeats: 3,   Target:  '%E_o2R:dW %E_o2R:dW %E_o2R:dW \n',   Predict: '%E_o2R:dW %E_o2R:dW \nE_o2R:dW \n'
[Iter 18600] Loss=0.5628 (avg=0.5418), Acc=95.15%, EndMarker=18.75%, Len=10, Repeats=1-10, GradNorm=2.90, LR=2.50e-04 |   Repeats: 1,   Target:  'jYh0fLL(0"\n',   Predict: 'jYhXfLL(S"j'
[Iter 18700] Loss=0.5431 (avg=0.5365), Acc=97.49%, EndMarker=50.00%, Len=10, Repeats=1-10, GradNorm=2.60, LR=2.50e-04 |   Repeats: 8,   Target:  "5'xK N<Y=N5'xK N<Y=N5'xK N<Y=N5'xK N<Y=N5'xK N<Y=N5'xK N<Y=N5'xK N<Y=N5'xK N<Y=N\n",   Predict: "5'xK N<Y=N5'xK N<Y=N5'xK N<Y=N5'xK N<Y=N5'xK N<Y=N5'xK N<Y=N5'xK N<Y=N5'xK N<Y=N\n"
[Iter 18800] Loss=0.5313 (avg=0.5281), Acc=95.09%, EndMarker=56.25%, Len=10, Repeats=1-10, GradNorm=1.87, LR=2.50e-04 |   Repeats: 1,   Target:  '7T|4?Q{;\\y\n',   Predict: '77|4?Q{;\\y\n'

============================================================
=== TRAINING COMPLETE ===
Reached acc=98.36%, end_acc=81.25%
Max seq_len=10, max_repeats=10
============================================================

>>> Final model saved to: ./models_repeat/ntm_repeat_copy_final.pt
>>> Best model (acc=94.74%) saved to: ./models_repeat/ntm_repeat_copy_best.pt

============================================================
TRAINING FINISHED
Final model: ./models_repeat/ntm_repeat_copy_final.pt
============================================================
```

## DNC

TODO: for temporal and dynamic memory

## RTNTM

Recurrent Transformer controller for the NTM mechanism. It learns both the copy task and adding two numbers task.

Apparently, the model benefits from having a large enough embedding dimension so the copy task model size is huge.

Adding an LSTM for state layers makes the copy task take much longer. RNNs give much smaller hit to speed.

I'd love to figure out a way to give the transformer context which represents the state of the system in addition to the memory component. The memory handles the arbitrary length input problem (ideally). But the fact transformers are not recurrent is a problem for algorithmic tasks and actually understanding deep concepts and their interactions. In particular mapping out the search space and actually searching it is probably impossible for 

### Copy Task

```
> python rtntm_copy_task.py
Start time: 2025-12-18 23:37:12.853728
Using device: cuda
RTNTM COPY TASK TRAINING

Model Configuration:
  Vocab size: 100
  Embedding dimension: 104
  Memory: 60 × 104
  Transformer: 1 layers, 4 heads
  Controller window: 4
  Read heads: 1, Write heads: 1
  Total parameters: 262,818
======================================================================
  
======================================================================
=== TRAINING COMPLETE ===
Reached acc=100.00% and loss=0.0035
at max seq_len=50
======================================================================

>>> Final model saved to: ./models_rtntm/rtntm_copy_final.pt
>>> Best model (acc=1.60%, loss=5.0041) at: ./models_rtntm/rtntm_copy_best.pt

======================================================================
TRAINING FINISHED
Final model: ./models_rtntm/rtntm_copy_final.pt
Total time: 3.22 minutes (0.05 hours)
======================================================================

End time: 2025-12-18 23:40:26.248891
======================================================================
```

### Reverse Copy

### Add Numbers

```
> python rtntm_add_nums.py
======================================================================
Start time: 2025-12-18 14:02:27.529949
======================================================================

Using device: cuda

======================================================================
RTNTM ADDITION TASK (Little-Endian)
======================================================================

Model Configuration:
  Vocab: 0123456789+= _ (size=14)
  Embedding dimension: 64
  Memory: 128 × 64
  Transformer: 1 layers, 8 heads
  Controller window: 8 (forces external memory use)
  Read heads: 2, Write heads: 1
  State layers: 1
  Total parameters: 97,143
  
...

======================================================================
Reached target num_len=20
CharAcc=1.0000, SeqAcc=1.0000, Loss=0.0062
======================================================================

======================================================================
TRAINING FINISHED
Final model: ./models_rtntm_add/rtntm_add_littleendian_final.pt
Total time: 46.55 minutes (0.78 hours)
======================================================================

End time: 2025-12-18 14:49:00.927215
======================================================================
```

```
> python rtntm_add_nums.py
Start time: 2025-12-18 17:22:47.630865

Using device: cuda
RTNTM ADDITION TASK (Little-Endian)

Model Configuration:
  Vocab: 0123456789+= _ (size=14)
  Embedding dimension: 104
  Memory: 128 × 104
  Transformer: 1 layers, 4 heads
  Controller window: 8 (forces external memory use)
  Read heads: 2, Write heads: 1
  State layers: 1
  Total parameters: 251,363
======================================================================

======================================================================
Reached target num_len=20
CharAcc=1.0000, SeqAcc=1.0000, Loss=0.0054
======================================================================

======================================================================
TRAINING FINISHED
Final model: ./models_rtntm_add/rtntm_add_littleendian_final.pt
Total time: 24.08 minutes (0.40 hours)
======================================================================

End time: 2025-12-18 17:46:52.736718
======================================================================
```

```
> python rtntm_add_nums.py
Start time: 2025-12-18 21:44:36.719024

Using device: cuda
RTNTM ADDITION TASK (Little-Endian)

Model Configuration:
  Vocab: 0123456789+= _ (size=14)
  Embedding dimension: 200
  Memory: 128 × 200
  Transformer: 2 layers, 4 heads
  Controller window: 8 (forces external memory use)
  Read heads: 2, Write heads: 1
  State layers: 1
  Total parameters: 1,397,851

...

======================================================================
Reached target num_len=20
CharAcc=1.0000, SeqAcc=1.0000, Loss=0.0055
======================================================================

======================================================================
TRAINING FINISHED
Final model: ./models_rtntm_add/rtntm_add_littleendian_final.pt
Total time: 20.88 minutes (0.35 hours)
======================================================================

End time: 2025-12-18 22:05:29.984691
======================================================================
```