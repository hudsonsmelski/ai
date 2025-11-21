# ai
Reproducing NN models from papers, and fiddling around with new ones.

I ran from a python venv in WSL2, so it kinda works on my machine.

## ACT

### Parity

### Addition

## NTM

Neural Turing Machine

### Copy Task


## RTNTM

Recurrent Transformer NTM

### Copy Task

```
> python training/rtntm_copy_task.py
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