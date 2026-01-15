# Learning Structure-Property Representations for Work Function Profiles in 2D Materials.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

This project implements a CLIP-inspired contrastive learning framework to predict the work function of 2D materials from their crystal structures. The model combines a crystal structure encoder with attention mechanisms to learn meaningful representations of atomic configurations and their relationship to electronic properties.

## Architecture

The model consists of three main components:

1. **Crystal Encoder** (`CRY_ENCODER`): Processes atomic structures using graph convolutions and transformer blocks with attention mechanisms
2. **Work Function Encoder**: ResNet-based encoder for work function profiles
3. **Contrastive Learning**: CLIP-style framework matching crystal structures to their work function profiles


## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
pip install -r requirements.txt
# Or install as package
pip install -e .
```

## Quick Start

### Training

```bash
# Train the model from scratch
python src/train.py
```

The training script will:
- Load 2D material structures from the database
- Split data into train/validation/test sets (80/10/10)
- Train for 1000 epochs with cosine learning rate scheduling
- Save the best model to `checkpoints/best_contra.pt`

### Inference

```python
import torch
from src.models.clip_model import CLIP, CLIPConfig, PointNetConfig
from src.models.crystal_encoder import CRY_ENCODER, cry_config

# Load trained model
model = CLIP(config, pointnet_config, cry_encoder)
checkpoint = torch.load('checkpoints/best_contra.pt')
model.load_state_dict(checkpoint)
model.eval()
```

## Dataset

The dataset contains 100+ 2D crystalline materials from the database (MIP2D). Each material includes:

- Crystal structure (atomic positions, lattice parameters)
- Work function profiles (Z-axis potential)
- Material properties (composition, space group)

**Data Format:**
- Raw structures: POSCAR format
- Work functions: Text files with Z-coordinate and potential values
- Database: LMDB format for efficient loading

See [data/README.md](data/README.md) for detailed dataset documentation.

## Training

### Configuration

Key hyperparameters (in `src/train.py`):

```python
embeddingSize = 384      # Hidden dimension
n_layers = 2             # Transformer layers
n_heads = 4              # Attention heads
batchSize = 64           # Batch size
numEpochs = 1000         # Training epochs
learning_rate = 1e-4     # Initial learning rate
```



## License

This project is licensed under the MIT License - see the LICENSE file for details.
