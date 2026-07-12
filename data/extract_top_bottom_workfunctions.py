#!/usr/bin/env python3
"""Extract surface-resolved work functions from the cleaned MatHub-2D PAEP data.

Usage:
    pip install ase numpy
    python extract_top_bottom_workfunctions.py

The script reads mathub2d_clean_structures_paep.db and writes
surface_workfunctions.csv in the current directory.
"""

import csv

import numpy as np
from ase.db import connect


DATABASE = "mathub2d_clean_structures_paep.db"
OUTPUT = "surface_workfunctions.csv"
PLATEAU_POINTS = 5


def main():
    database = connect(DATABASE)
    fields = [
        "mip2d_id",
        "chemical_formula",
        "efermi_eV",
        "evac_bottom_eV",
        "evac_top_eV",
        "workfunction_bottom_eV",
        "workfunction_top_eV",
    ]

    with open(OUTPUT, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for row in database.select():
            profile = np.asarray(row.data["paep_profile"], dtype=float)
            potential = profile[:, 1]
            efermi = float(row.efermi_eV)

            # bottom is the low-fractional-z (-c) side; top is the +c side.
            evac_bottom = float(np.mean(potential[:PLATEAU_POINTS]))
            evac_top = float(np.mean(potential[-PLATEAU_POINTS:]))

            writer.writerow(
                {
                    "mip2d_id": row.mip2d_id,
                    "chemical_formula": row.chemical_formula,
                    "efermi_eV": f"{efermi:.9f}",
                    "evac_bottom_eV": f"{evac_bottom:.9f}",
                    "evac_top_eV": f"{evac_top:.9f}",
                    "workfunction_bottom_eV": f"{evac_bottom - efermi:.9f}",
                    "workfunction_top_eV": f"{evac_top - efermi:.9f}",
                }
            )

    print(f"Wrote {len(database)} records to {OUTPUT}")


if __name__ == "__main__":
    main()
