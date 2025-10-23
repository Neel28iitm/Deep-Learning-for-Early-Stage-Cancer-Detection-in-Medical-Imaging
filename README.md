##Deep Learning for Early-Stage Cancer Detection in Medical Imaging

This project implements a character-level sequence-to-sequence transliteration model using the [Dakshina dataset](https://github.com/google-research-datasets/dakshina). The goal is to map words written in Roman script to native Devanagari script, e.g., `ghar → घर`.

We implemented:
- Vanilla Seq2Seq models using RNN, GRU, and LSTM
- Attention-based Seq2Seq model (Luong attention)
- wandb.ai hyperparameter sweeps
- Attention heatmaps and connectivity graphs

---

## Setup Instructions

### Install dependencies
Run the following in Colab or your local environment:

```bash
pip install torch numpy pandas matplotlib seaborn networkx wandb

# Evaluation Result
| Model            | Accuracy  |
| ---------------- | --------- |
| Vanilla (LSTM)   | **92.4%** |
| Attention (LSTM) | **94.1%** |

Q1(a) Total Number of Computations (FLOPs)
Estimated FLOPs per sample (forward pass only):

FLOPs formula:
= 2 × T × 4k(m + k + 2) + T × V(k + 1)

Substitution:
= 2 × 10 × 4 × 128(64 + 128 + 2) + 10 × 100 × (128 + 1)
= 1,993,280 + 129,000 = 2,122,280 FLOPs/sample

Q1(b) Total Number of Parameters
Estimated learnable parameters:

Parameters formula:
= V × m + 2 × 4k(m + k + 1) + V(k + 1)

Substitution:
= 100 × 64 + 2 × 4 × 128(64 + 128 + 1) + 100 × 129
= 6,400 + 197,632 + 12,900 = 216,932 parameters
