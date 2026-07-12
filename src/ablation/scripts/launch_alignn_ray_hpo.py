#!/usr/bin/env python
"""Ray Tune HPO for MatHub-2D work-function ALIGNN scalar targets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_alignn_efermi import (  # noqa: E402
    AlignnEfermiDataset,
    collate_alignn,
    compute_metrics,
    evaluate,
    load_split_rows,
    set_seed,
    sha256_file,
    to_jsonable,
)


def profile_defaults(profile: str) -> dict[str, object]:
    if profile == "smoke":
        return {
            "num_samples": 2,
            "epochs": 2,
            "grace_period": 1,
            "max_train_samples": 64,
            "max_val_samples": 32,
            "max_test_samples": 32,
            "num_gpus": 1,
        }
    if profile == "pilot":
        return {
            "num_samples": 12,
            "epochs": 15,
            "grace_period": 4,
            "max_train_samples": None,
            "max_val_samples": None,
            "max_test_samples": None,
            "num_gpus": 4,
        }
    return {
        "num_samples": 40,
        "epochs": 50,
        "grace_period": 8,
        "max_train_samples": None,
        "max_val_samples": None,
        "max_test_samples": None,
        "num_gpus": 4,
    }


def make_search_space(profile: str) -> dict[str, object]:
    if profile == "smoke":
        return {
            "hidden_features": tune.choice([64, 128]),
            "alignn_layers": tune.choice([1, 2]),
            "gcn_layers": tune.choice([1, 2]),
            "lr": tune.choice([5.0e-4, 1.0e-3]),
            "weight_decay": tune.choice([1.0e-5, 1.0e-4]),
            "batch_size": tune.choice([4]),
            "loss": tune.choice(["mse", "smoothl1"]),
        }
    return {
        "hidden_features": tune.choice([64, 128, 192, 256]),
        "alignn_layers": tune.choice([1, 2, 3, 4]),
        "gcn_layers": tune.choice([1, 2, 3, 4]),
        "lr": tune.loguniform(3.0e-4, 2.0e-3),
        "weight_decay": tune.loguniform(1.0e-6, 1.0e-4),
        "batch_size": tune.choice([4, 8]),
        "loss": tune.choice(["mse", "smoothl1"]),
    }


def make_criterion(loss_name: str):
    if loss_name == "smoothl1":
        return torch.nn.SmoothL1Loss()
    if loss_name == "l1":
        return torch.nn.L1Loss()
    return torch.nn.MSELoss()


def write_trial_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_trainable(base_config: dict[str, object]):
    def trainable(config: dict[str, object]) -> None:
        from alignn.models.alignn import ALIGNN, ALIGNNConfig
        from torch.utils.data import DataLoader

        os.environ.setdefault("DGLBACKEND", "pytorch")
        seed = int(base_config["seed"]) + int(config.get("trial_seed_offset", 0))
        set_seed(seed)

        db_path = Path(str(base_config["db"]))
        split_file = Path(str(base_config["split_file"]))
        train_rows = load_split_rows(split_file, "train", base_config.get("max_train_samples"))
        val_rows = load_split_rows(split_file, "val", base_config.get("max_val_samples"))
        test_rows = (
            load_split_rows(split_file, "test", base_config.get("max_test_samples"))
            if bool(base_config.get("test_each_trial", False))
            else []
        )
        target_values = np.asarray([float(row["target"]) for row in train_rows], dtype=float)
        target_mean = float(np.mean(target_values))
        target_std = float(np.std(target_values)) if float(np.std(target_values)) > 1.0e-12 else 1.0

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dataset_kwargs = {
            "db_path": db_path,
            "target_mean": target_mean,
            "target_std": target_std,
            "cutoff": float(base_config["cutoff"]),
            "max_neighbors": int(base_config["max_neighbors"]),
            "standardize": True,
        }
        train_dataset = AlignnEfermiDataset(split_rows=train_rows, **dataset_kwargs)
        val_dataset = AlignnEfermiDataset(split_rows=val_rows, **dataset_kwargs)
        test_dataset = AlignnEfermiDataset(split_rows=test_rows, **dataset_kwargs) if test_rows else None
        loader_kwargs = {
            "collate_fn": collate_alignn,
            "num_workers": int(base_config["num_workers"]),
            "pin_memory": device.type == "cuda",
            "persistent_workers": int(base_config["num_workers"]) > 0,
        }
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(base_config["test_batch_size"]),
            shuffle=False,
            **loader_kwargs,
        )
        test_loader = (
            DataLoader(
                test_dataset,
                batch_size=int(base_config["test_batch_size"]),
                shuffle=False,
                **loader_kwargs,
            )
            if test_dataset is not None
            else None
        )

        model_config = ALIGNNConfig(
            name="alignn",
            alignn_layers=int(config["alignn_layers"]),
            gcn_layers=int(config["gcn_layers"]),
            atom_input_features=92,
            edge_input_features=80,
            triplet_input_features=40,
            embedding_features=int(base_config["embedding_features"]),
            hidden_features=int(config["hidden_features"]),
            output_features=1,
        )
        model = ALIGNN(model_config).to(device)
        criterion = make_criterion(str(config["loss"]))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(config["lr"]),
            epochs=int(base_config["epochs"]),
            steps_per_epoch=max(len(train_loader), 1),
        )

        best_val_loss = float("inf")
        best_epoch = 0
        best_state = None
        best_val_metrics: dict[str, object] | None = None
        start = time.time()
        for epoch in range(1, int(base_config["epochs"]) + 1):
            model.train()
            train_loss_sum = 0.0
            train_count = 0
            for graph, line_graph, target, _ in train_loader:
                graph = graph.to(device)
                line_graph = line_graph.to(device)
                target = target.to(device).view(-1)
                optimizer.zero_grad(set_to_none=True)
                output = model([graph, line_graph, None]).view(-1)
                loss = criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                count = int(target.numel())
                train_loss_sum += float(loss.item()) * count
                train_count += count

            train_loss = train_loss_sum / max(train_count, 1)
            val_metrics, _, _, _ = evaluate(
                model,
                val_loader,
                criterion,
                device,
                target_mean,
                target_std,
                True,
            )
            if float(val_metrics["loss_model_space"]) < best_val_loss:
                best_val_loss = float(val_metrics["loss_model_space"])
                best_epoch = epoch
                best_state = {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": to_jsonable(val_metrics),
                    "target_mean": target_mean,
                    "target_std": target_std,
                }
                best_val_metrics = to_jsonable(val_metrics)
            tune.report(
                {
                    "training_iteration": epoch,
                    "epoch": epoch,
                    "train_loss_model_space": float(train_loss),
                    "val_loss_model_space": float(val_metrics["loss_model_space"]),
                    "val_mae": float(val_metrics["mae"]),
                    "val_rmse": float(val_metrics["rmse"]),
                    "val_r2": float(val_metrics["r2"]),
                    "best_val_loss_model_space": float(best_val_loss),
                    "best_epoch": int(best_epoch),
                    "elapsed_seconds": float(time.time() - start),
                }
            )

        if best_state is not None:
            model.load_state_dict(best_state["model_state_dict"])
        final_report = {
            "training_iteration": int(base_config["epochs"]),
            "epoch": int(base_config["epochs"]),
            "val_loss_model_space": float(best_val_loss),
            "best_val_loss_model_space": float(best_val_loss),
            "best_epoch": int(best_epoch),
            "elapsed_seconds": float(time.time() - start),
        }
        if best_val_metrics is not None:
            final_report.update(
                {
                    "val_mae": float(best_val_metrics["mae"]),
                    "val_rmse": float(best_val_metrics["rmse"]),
                    "val_r2": float(best_val_metrics["r2"]),
                    "best_val_mae": float(best_val_metrics["mae"]),
                    "best_val_rmse": float(best_val_metrics["rmse"]),
                    "best_val_r2": float(best_val_metrics["r2"]),
                }
            )
        if test_loader is not None:
            test_metrics, _, _, _ = evaluate(
                model,
                test_loader,
                criterion,
                device,
                target_mean,
                target_std,
                True,
            )
            final_report.update(
                {
                    "test_mae": float(test_metrics["mae"]),
                    "test_rmse": float(test_metrics["rmse"]),
                    "test_r2": float(test_metrics["r2"]),
                }
            )
        tune.report(final_report)

    return trainable


def aggregate_ray_progress(experiment_dir: Path) -> None:
    trajectory_rows: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = []
    for trial_path in sorted(path for path in experiment_dir.iterdir() if path.is_dir()):
        progress_path = trial_path / "progress.csv"
        params_path = trial_path / "params.json"
        if not progress_path.exists():
            continue
        params = {}
        if params_path.exists():
            with params_path.open() as handle:
                params = json.load(handle)
        with progress_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            flat = {
                "trial_name": trial_path.name,
            }
            flat.update({f"config_{key}": value for key, value in params.items()})
            for key, value in row.items():
                if key.startswith("config/"):
                    flat[f"config_{key.split('/', 1)[1]}"] = value
                else:
                    flat[key] = value
            trajectory_rows.append(flat)
        final = rows[-1] if rows else {}
        trial_row = {
            "trial_name": trial_path.name,
            "num_reported_rows": len(rows),
        }
        trial_row.update({f"config_{key}": value for key, value in params.items()})
        for key in [
            "training_iteration",
            "epoch",
            "val_loss_model_space",
            "best_val_loss_model_space",
            "val_mae",
            "val_rmse",
            "val_r2",
            "test_mae",
            "test_rmse",
            "test_r2",
            "best_epoch",
            "time_total_s",
            "done",
        ]:
            trial_row[key] = final.get(key, "")
        trial_rows.append(trial_row)

    for path, rows in [
        (experiment_dir / "ray_hpo_epoch_trajectories.csv", trajectory_rows),
        (experiment_dir / "ray_hpo_trial_table.csv", trial_rows),
    ]:
        if not rows:
            continue
        fieldnames = sorted({key for row in rows for key in row})
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path("raw/structures.db"))
    parser.add_argument("--split-file", type=Path, default=Path("splits/wf_seed42_70_15_15.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("ray_hpo"))
    parser.add_argument("--profile", choices=["smoke", "pilot", "full"], default="smoke")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--target-name", default="wf")
    parser.add_argument("--target-units", default="eV")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-seed", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--grace-period", type=int, default=None)
    parser.add_argument("--num-cpus", type=int, default=16)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--cpus-per-trial", type=float, default=2.0)
    parser.add_argument("--gpus-per-trial", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--test-batch-size", type=int, default=8)
    parser.add_argument("--embedding-features", type=int, default=64)
    parser.add_argument("--cutoff", type=float, default=8.0)
    parser.add_argument("--max-neighbors", type=int, default=12)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--test-each-trial", action="store_true")
    return parser.parse_args()


def main() -> None:
    import ray

    os.environ.setdefault("DGLBACKEND", "pytorch")
    args = parse_args()
    defaults = profile_defaults(args.profile)
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    db_path = args.db.resolve()
    split_file = args.split_file.resolve()
    output_root = args.output_root.resolve()
    run_name = args.run_name or f"{args.profile}_{args.target_name}_alignn_ray_seed{args.seed}_{time.strftime('%Y%m%dT%H%M%S')}"

    base_config = {
        "dataset_name": "mathub2d_workfunction",
        "target": args.target_name,
        "target_units": args.target_units,
        "db": str(db_path),
        "db_sha256": sha256_file(db_path),
        "split_file": str(split_file),
        "split_sha256": sha256_file(split_file),
        "seed": int(args.seed),
        "search_seed": int(args.search_seed if args.search_seed is not None else args.seed),
        "epochs": int(args.epochs),
        "num_workers": int(args.num_workers),
        "test_batch_size": int(args.test_batch_size),
        "embedding_features": int(args.embedding_features),
        "cutoff": float(args.cutoff),
        "max_neighbors": int(args.max_neighbors),
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "max_test_samples": args.max_test_samples,
        "test_each_trial": bool(args.test_each_trial),
    }

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / run_name / "ray_hpo_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as handle:
        json.dump(
            {
                "base_config": base_config,
                "profile": args.profile,
                "num_samples": int(args.num_samples),
                "metric": "val_loss_model_space",
                "mode": "min",
                "scheduler": "ASHAScheduler",
                "search": "HyperOptSearch",
                "test_set_guard": "test split is not evaluated during HPO unless --test-each-trial is explicitly set",
                "num_cpus": args.num_cpus,
                "num_gpus": args.num_gpus,
                "cpus_per_trial": args.cpus_per_trial,
                "gpus_per_trial": args.gpus_per_trial,
            },
            handle,
            indent=2,
        )

    ray.init(
        num_cpus=int(args.num_cpus),
        num_gpus=float(args.num_gpus),
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=True,
    )
    scheduler = ASHAScheduler(
        metric="val_loss_model_space",
        mode="min",
        max_t=int(args.epochs),
        grace_period=int(args.grace_period),
        reduction_factor=2,
    )
    search_seed = int(args.search_seed if args.search_seed is not None else args.seed)
    search = HyperOptSearch(metric="val_loss_model_space", mode="min", random_state_seed=search_seed)
    trainable = tune.with_resources(
        make_trainable(base_config),
        resources={"cpu": float(args.cpus_per_trial), "gpu": float(args.gpus_per_trial)},
    )
    result_grid = tune.run(
        trainable,
        name=run_name,
        storage_path=str(output_root),
        config=make_search_space(args.profile),
        num_samples=int(args.num_samples),
        scheduler=scheduler,
        search_alg=search,
        verbose=1,
    )
    best_trial = result_grid.get_best_trial(metric="val_loss_model_space", mode="min", scope="all")
    summary = []
    for trial in result_grid.trials:
        last = trial.last_result or {}
        summary.append(
            {
                "trial_id": trial.trial_id,
                "status": trial.status,
                "config": trial.config,
                "last_result": {
                    key: last.get(key)
                    for key in [
                        "training_iteration",
                        "epoch",
                        "val_loss_model_space",
                        "best_val_loss_model_space",
                        "val_mae",
                        "val_r2",
                        "test_mae",
                        "test_r2",
                        "best_epoch",
                        "elapsed_seconds",
                    ]
                },
            }
        )
    experiment_dir = output_root / run_name
    aggregate_ray_progress(experiment_dir)
    with (experiment_dir / "ray_hpo_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    with (experiment_dir / "best_trial.json").open("w") as handle:
        json.dump(
            {
                "trial_id": best_trial.trial_id if best_trial else None,
                "config": best_trial.config if best_trial else None,
                "last_result": best_trial.last_result if best_trial else None,
            },
            handle,
            indent=2,
            default=str,
        )
    print(f"experiment_dir={experiment_dir}")
    if best_trial:
        print(json.dumps({"best_trial_id": best_trial.trial_id, "config": best_trial.config, "last_result": best_trial.last_result}, indent=2, default=str))
    ray.shutdown()


if __name__ == "__main__":
    main()
