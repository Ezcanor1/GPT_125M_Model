# GPT from Scratch — Language Modeling on Custom Text using PyTorch

This project implements a transformer-based **GPT-style language model** (similar to GPT-2 123M) from scratch using PyTorch and trains it on a custom text dataset.

## 🚀 Highlights

- Complete **GPT model** built from scratch (Multi-Head Attention, Transformer Blocks, LayerNorm, GELU)
- Tokenization using **`tiktoken`** (same as OpenAI GPT models)
- Trains a language model on `the-verdict.txt` dataset
- Implements custom **dataset loader**, **loss evaluation**, and **text generation**

---

## 📁 Dataset

A plain text file named `the-verdict.txt` is used for training and validation.

```bash
├── the-verdict.txt
