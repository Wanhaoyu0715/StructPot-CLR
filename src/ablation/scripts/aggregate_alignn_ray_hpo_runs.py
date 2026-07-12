#!/usr/bin/env python3
"""Aggregate multiple ALIGNN Ray Tune run outputs.

Each run is a standalone output directory produced by launch_alignn_ray_hpo.py.
This script concatenates per-epoch and per-trial audit tables, records the run
origin, and selects the global best trial by best_val_loss_model_space.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Iterable


def read_csv(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: object, default: float = math.inf) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    return result if math.isfinite(result) else default


def load_json(path: Path) -> object:
    with path.open() as handle:
        return json.load(handle)


def normalize_run_arg(value: str) -> tuple[str, Path]:
    if "=" in value:
        name, path = value.split("=", 1)
        return name, Path(path).expanduser().resolve()
    path = Path(value).expanduser().resolve()
    return path.name, path


def add_run(rows: Iterable[dict[str, object]], run_group: str, run_dir: Path) -> list[dict[str, object]]:
    out = []
    for row in rows:
        copied = dict(row)
        copied["run_group"] = run_group
        copied["run_dir"] = str(run_dir)
        out.append(copied)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="NAME=/path/to/run_dir or /path/to/run_dir")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_trials: list[dict[str, object]] = []
    all_epochs: list[dict[str, object]] = []
    manifests: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    for raw_run in args.run:
        run_group, run_dir = normalize_run_arg(raw_run)
        trial_rows = add_run(read_csv(run_dir / "ray_hpo_trial_table.csv"), run_group, run_dir)
        epoch_rows = add_run(read_csv(run_dir / "ray_hpo_epoch_trajectories.csv"), run_group, run_dir)
        all_trials.extend(trial_rows)
        all_epochs.extend(epoch_rows)

        manifest_path = run_dir / "ray_hpo_manifest.json"
        if manifest_path.exists():
            manifests.append({"run_group": run_group, "run_dir": str(run_dir), "manifest": load_json(manifest_path)})
        summary_path = run_dir / "ray_hpo_summary.json"
        if summary_path.exists():
            run_summary = load_json(summary_path)
            for item in run_summary:
                if isinstance(item, dict):
                    item = dict(item)
                    item["run_group"] = run_group
                    item["run_dir"] = str(run_dir)
                summaries.append(item)

        best_path = run_dir / "best_trial.json"
        if best_path.exists():
            shutil.copy2(best_path, output_dir / f"best_trial_{run_group}.json")

    write_csv(output_dir / "combined_ray_hpo_trial_table.csv", all_trials)
    write_csv(output_dir / "combined_ray_hpo_epoch_trajectories.csv", all_epochs)
    with (output_dir / "combined_ray_hpo_summary.json").open("w") as handle:
        json.dump(summaries, handle, indent=2)
    with (output_dir / "combined_ray_hpo_manifests.json").open("w") as handle:
        json.dump(manifests, handle, indent=2)

    ranked_trials = sorted(
        [row for row in all_trials if to_float(row.get("best_val_loss_model_space")) < math.inf],
        key=lambda row: to_float(row.get("best_val_loss_model_space")),
    )
    if ranked_trials:
        best = ranked_trials[0]
        best_config = {
            key.replace("config_", "", 1): best[key]
            for key in best
            if key.startswith("config_") and best[key] not in (None, "")
        }
        for int_key in ["hidden_features", "alignn_layers", "gcn_layers", "batch_size"]:
            if int_key in best_config:
                best_config[int_key] = int(float(best_config[int_key]))
        for float_key in ["lr", "weight_decay"]:
            if float_key in best_config:
                best_config[float_key] = float(best_config[float_key])
        best_record = {
            "selection_metric": "best_val_loss_model_space",
            "selection_mode": "min",
            "best_trial_row": best,
            "best_config": best_config,
            "num_trials": len(all_trials),
            "num_epoch_rows": len(all_epochs),
            "num_ranked_trials": len(ranked_trials),
        }
        with (output_dir / "global_best_trial.json").open("w") as handle:
            json.dump(best_record, handle, indent=2)
        write_csv(output_dir / "combined_top20_trials.csv", ranked_trials[:20])

    print(output_dir)
    print(f"trials={len(all_trials)} epoch_rows={len(all_epochs)} ranked={len(ranked_trials)}")


if __name__ == "__main__":
    main()
