# MatHub-2D cleaned work-function data

This directory provides the cleaned MatHub-2D structures, plane-averaged
electrostatic potential (PAEP) profiles, Fermi levels, and a reproducible script for
extracting the work functions associated with the two surfaces of each monolayer.

## Files

- `mathub2d_clean_structures_paep.db`: an ASE database containing 1,896 cleaned
  monolayer structures. Each record includes the complete atomistic structure, the
  Fermi level, and the full PAEP profile.
- `extract_top_bottom_workfunctions.py`: extracts the vacuum levels and work
  functions for the bottom and top surfaces and writes
  `surface_workfunctions.csv`.

The original `MIP2D-*` record identifiers are retained for traceability. The name
of the source dataset is **MatHub-2D**.

## Data cleaning

The original matched dataset contained 1,899 structures. Three records with a
scalar work function below 0.5 eV were removed, leaving 1,896 structures. All
retained records contain a finite Fermi level and a complete PAEP profile.

## Surface convention and calculation

The PAEP profile is stored in `row.data["paep_profile"]` as an array with two
columns: fractional z coordinate and electrostatic potential in eV.

- `bottom` denotes the low-fractional-z side (`-c` direction).
- `top` denotes the high-fractional-z side (`+c` direction).
- `Evac_bottom` is the mean potential of the first five PAEP samples.
- `Evac_top` is the mean potential of the last five PAEP samples.

The side-resolved work functions are calculated as

```text
WF_bottom = Evac_bottom - EF
WF_top    = Evac_top - EF
```

This convention exactly reproduces the original scalar label through

```text
WF_scalar = (WF_bottom + WF_top) / 2
```

All retained records have both `WF_bottom >= 0.5 eV` and
`WF_top >= 0.5 eV`.

## Usage

Install the two required Python packages:

```bash
pip install ase numpy
```

Run the extraction script from this directory:

```bash
python extract_top_bottom_workfunctions.py
```

The generated `surface_workfunctions.csv` contains:

```text
mip2d_id
chemical_formula
efermi_eV
evac_bottom_eV
evac_top_eV
workfunction_bottom_eV
workfunction_top_eV
```

## ASE example

```python
from ase.db import connect

db = connect("mathub2d_clean_structures_paep.db")
row = db.get(1)
atoms = row.toatoms()
profile = row.data["paep_profile"]

print(row.mip2d_id, row.chemical_formula, row.efermi_eV)
print(atoms, profile.shape)
```

## Source

The structures and PAEP data originate from MatHub-2D:

M. Yao et al., *Science China Materials* **66** (2023), 2768-2776.
https://doi.org/10.1007/s40843-022-2401-3
