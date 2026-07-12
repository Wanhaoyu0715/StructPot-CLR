#!/usr/bin/env python3
"""Wait for the 80/10/10 Ray300 runs, aggregate them, and run final refit.

This script is intended to run on the shared filesystem after the two
independent HPO runs have been launched on group_A and group_B.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


GROUP_A_RUN = "ray300_wf_80_10_10_alignn_seed42_group_a_run200_20260709"
GROUP_B_RUN = "ray300_wf_80_10_10_alignn_seed42_group_b_run100_20260709"
SPLIT_FILE = "splits/wf_seed42_80_10_10_counts1519_189_191.csv"
FINAL_RUN = "best_ray300_wf_80_10_10_alignn_seed42_final_refit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=Path("<ray_output_dir>"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    return parser.parse_args()


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        json.dump(data, handle, indent=2)
    tmp.replace(path)


def run_checked(cmd: list[str], cwd: Path, log_path: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("RUN", " ".join(cmd), flush=True)
    if log_path is None:
        subprocess.run(cmd, cwd=cwd, env=env, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as handle:
        handle.write("\n" + "=" * 80 + "\n")
        handle.write("RUN " + " ".join(cmd) + "\n")
        handle.flush()
        subprocess.run(cmd, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def run_done(base_dir: Path, run_name: str) -> bool:
    run_dir = base_dir / "ray_hpo" / run_name
    required = [
        run_dir / "ray_hpo_summary.json",
        run_dir / "best_trial.json",
        run_dir / "ray_hpo_trial_table.csv",
        run_dir / "ray_hpo_epoch_trajectories.csv",
    ]
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def wait_for_runs(args: argparse.Namespace, status_path: Path) -> None:
    deadline = time.time() + args.timeout_hours * 3600
    while True:
        group_A_done = run_done(args.base_dir, GROUP_A_RUN)
        group_B_done = run_done(args.base_dir, GROUP_B_RUN)
        write_json(
            status_path,
            {
                "stage": "waiting_for_hpo_runs",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "group_A_done": group_A_done,
                "group_B_done": group_B_done,
                "group_a_run": GROUP_A_RUN,
                "group_b_run": GROUP_B_RUN,
            },
        )
        print(f"{time.strftime('%F %T')} group_A_done={group_A_done} group_B_done={group_B_done}", flush=True)
        if group_A_done and group_B_done:
            return
        if time.time() > deadline:
            raise TimeoutError("Timed out while waiting for both Ray HPO runs to finish.")
        time.sleep(args.poll_seconds)


def aggregate(args: argparse.Namespace, combined_dir: Path, status_path: Path) -> Path:
    best_path = combined_dir / "global_best_trial.json"
    if best_path.exists() and best_path.stat().st_size > 0:
        print(f"Aggregate output already exists: {best_path}", flush=True)
        return best_path

    run_checked(
        [
            args.python,
            "scripts/aggregate_alignn_ray_hpo_runs.py",
            "--run",
            f"group_A={args.base_dir / 'ray_hpo' / GROUP_A_RUN}",
            "--run",
            f"group_B={args.base_dir / 'ray_hpo' / GROUP_B_RUN}",
            "--output-dir",
            str(combined_dir),
        ],
        cwd=args.base_dir,
        log_path=combined_dir.parent / "aggregate.log",
    )
    write_json(
        status_path,
        {
            "stage": "aggregated_hpo_runs",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "combined_dir": str(combined_dir),
            "global_best_trial": str(best_path),
        },
    )
    return best_path


def load_best_config(best_path: Path) -> dict[str, object]:
    with best_path.open() as handle:
        best = json.load(handle)
    config = best.get("best_config")
    if not isinstance(config, dict) or not config:
        raise ValueError(f"No best_config found in {best_path}")
    return config


def run_final_refit(args: argparse.Namespace, best_config: dict[str, object], status_path: Path) -> Path:
    final_root = args.base_dir / "final_refit"
    final_dir = final_root / FINAL_RUN
    metrics_path = final_dir / "metrics.json"
    if metrics_path.exists() and metrics_path.stat().st_size > 0:
        print(f"Final refit already exists: {metrics_path}", flush=True)
        return metrics_path

    env = os.environ.copy()
    env["DGLBACKEND"] = "pytorch"
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    cmd = [
        args.python,
        "scripts/train_alignn_efermi.py",
        "--db",
        "raw/structures.db",
        "--split-file",
        SPLIT_FILE,
        "--output-dir",
        str(final_root),
        "--run-name",
        FINAL_RUN,
        "--target-name",
        "wf",
        "--target-units",
        "eV",
        "--gpu",
        "0",
        "--seed",
        "42",
        "--epochs",
        "50",
        "--patience",
        "50",
        "--test-batch-size",
        "8",
        "--num-workers",
        "0",
        "--hidden-features",
        str(int(best_config["hidden_features"])),
        "--alignn-layers",
        str(int(best_config["alignn_layers"])),
        "--gcn-layers",
        str(int(best_config["gcn_layers"])),
        "--lr",
        str(float(best_config["lr"])),
        "--weight-decay",
        str(float(best_config["weight_decay"])),
        "--batch-size",
        str(int(best_config["batch_size"])),
        "--loss",
        str(best_config.get("loss", "smoothl1")),
    ]
    write_json(
        status_path,
        {
            "stage": "running_final_refit",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "final_dir": str(final_dir),
            "best_config": best_config,
            "cmd": cmd,
            "cuda_visible_devices": args.cuda_visible_devices,
        },
    )
    run_checked(cmd, cwd=args.base_dir, log_path=final_dir / "final_refit.log", env=env)
    return metrics_path


def main() -> None:
    args = parse_args()
    args.base_dir = args.base_dir.expanduser().resolve()
    report_root = args.base_dir / "reports" / "ray300_wf_80_10_10_hpo_runs_20260709"
    combined_dir = report_root / "combined"
    status_path = report_root / "postprocess_status.json"

    wait_for_runs(args, status_path)
    best_path = aggregate(args, combined_dir, status_path)
    best_config = load_best_config(best_path)
    metrics_path = run_final_refit(args, best_config, status_path)
    with metrics_path.open() as handle:
        metrics = json.load(handle)
    write_json(
        status_path,
        {
            "stage": "complete",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "combined_dir": str(combined_dir),
            "global_best_trial": str(best_path),
            "final_metrics": str(metrics_path),
            "test_metrics": metrics.get("test_metrics", {}),
            "val_metrics": metrics.get("val_metrics", {}),
            "train_metrics": metrics.get("train_metrics", {}),
        },
    )
    print(json.dumps(metrics.get("test_metrics", {}), indent=2), flush=True)


if __name__ == "__main__":
    main()
