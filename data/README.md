# Dataset

This directory contains the 2D materials dataset used for work function prediction.

## Structure

```
data/
├── raw/                    # Raw material structures
│   └── 2d/                # 2D material structures from Materials Project
│       └── MIP2D-*/       # Individual material folders
│           └── potential.dat  # Work function data
├── processed/             # Processed database files
│   └── structures.db      # LMDB database with preprocessed structures
└── README.md             # This file
```

## Dataset Description

The dataset contains over 100 2D crystalline materials with their corresponding work function values. Each material includes:

- **Crystal structure**: Atomic positions and lattice parameters
- **Work function data**: Z-axis potential profiles
- **Material properties**: Chemical composition, space group, etc.

## Data Format

- **Raw structures**: POSCAR format (VASP input files)
- **Work function data**: Text files with Z-coordinate and potential values

## Source
DOI.ORG:
https://doi.org/10.1007/s40843-022-2401-3


