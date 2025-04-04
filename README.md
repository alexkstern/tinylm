# TinyLM

A PyTorch implementation of the GPT-NeoX architecture used in the TinyStories paper.

## Overview

This project is a re-implementation of the architecture described in [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759) using PyTorch. The goal is to train a language model from scratch that can generate coherent English text, following the approach of the original paper but with a cleaner, modular codebase.

The model architecture is based on GPT-NeoX, featuring a mix of local attention (windowed) and global attention heads. This implementation aims to replicate the training and generation capabilities of models like [TinyStories-1M](https://huggingface.co/roneneldan/TinyStories-1M) available on Hugging Face.

## Model Architecture

- **GPT-NeoX architecture** with a window size of 256 and context length of 2048
- Supports both local (windowed) and global attention mechanisms
- Uses the GPT-Neo tokenizer but keeps only the top most common tokens

## Features

- Configurable model parameters via config files
- Training with early stopping and checkpointing
- Wandb integration for experiment tracking
- Text generation capabilities
- Support for loading models from Hugging Face or local checkpoints

## Project Structure

- `model.py`: Core implementation of the GPTNeoX architecture
- `train.py`: Training loop with early stopping and logging
- `inference.py`: Text generation utilities
- `dataload.py`: Dataset loading and preprocessing
- `config/`: Configuration files for different model sizes

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch 1.9+
- Transformers
- Wandb (optional, for logging)

### Training

```bash
python train.py --config config/tiny_lm_1M.config --use_wandb --wandb_project tiny-stories --wandb_run_name baseline
```

### Inference

```bash
python inference.py
```

## Configuration

Model parameters are defined in config files like `tiny_lm_1M.config`. Key parameters include:

- `vocab_size`: Vocabulary size
- `max_position_embeddings`: Maximum sequence length
- `n_layer`: Number of transformer layers
- `n_head`: Number of attention heads
- `n_embd`: Embedding dimension
- `window_size`: Size of the local attention window
- `local_heads`: Number of local attention heads

## Acknowledgments

This implementation draws inspiration from:

- [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759)
- [TinyStories-1M](https://huggingface.co/roneneldan/TinyStories-1M) Hugging Face model