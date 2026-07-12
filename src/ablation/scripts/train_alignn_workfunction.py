#!/usr/bin/env python
"""Standalone ALIGNN training for the MatHub-2D work-function target."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import socket
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from ase.db import connect
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def compute_metrics(targets: Iterable[float], predictions: Iterable[float]) -> dict[str, float]:
    y_true = np.asarray(list(targets), dtype=float).reshape(-1)
    y_pred = np.asarray(list(predictions), dtype=float).reshape(-1)
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if y_true.size > 1 else float("nan"),
    }


def load_split_rows(split_file: Path, split_name: str, limit: int | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with split_file.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") != split_name:
                continue
            rows.append(
                {
                    "row_id": int(row["row_id"]),
                    "target": float(row["target"]),
                    "name": row.get("name", ""),
                    "formula": row.get("formula", ""),
                    "split": split_name,
                    "split_order": int(row.get("split_order", len(rows))),
                }
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def ase_row_to_jarvis_atoms(row):
    from jarvis.core.atoms import Atoms as JarvisAtoms

    ase_atoms = row.toatoms()
    return JarvisAtoms(
        lattice_mat=ase_atoms.cell.array,
        coords=ase_atoms.get_scaled_positions(),
        elements=ase_atoms.get_chemical_symbols(),
        cartesian=False,
    )


class AlignnEfermiDataset(Dataset):
    def __init__(
        self,
        db_path: Path,
        split_rows: list[dict[str, object]],
        target_mean: float,
        target_std: float,
        cutoff: float,
        max_neighbors: int,
        standardize: bool,
    ) -> None:
        self.db = connect(str(db_path))
        self.row_map = {int(row.id): row for row in self.db.select()}
        self.rows = split_rows
        self.target_mean = float(target_mean)
        self.target_std = float(target_std) if float(target_std) > 1.0e-12 else 1.0
        self.cutoff = float(cutoff)
        self.max_neighbors = int(max_neighbors)
        self.standardize = bool(standardize)

    def __len__(self) -> int:
        return len(self.rows)

    def encode_target(self, target: float) -> float:
        if not self.standardize:
            return float(target)
        return float((float(target) - self.target_mean) / self.target_std)

    def __getitem__(self, idx: int):
        from jarvis.core.graphs import Graph as JarvisGraph

        item = self.rows[idx]
        row_id = int(item["row_id"])
        entry = self.row_map[row_id]
        atoms = ase_row_to_jarvis_atoms(entry)
        graph = JarvisGraph.atom_dgl_multigraph(
            atoms,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
            compute_line_graph=True,
            use_canonize=True,
        )
        target_original = float(item["target"])
        target_model = self.encode_target(target_original)
        metadata = {
            "row_id": row_id,
            "name": item.get("name", ""),
            "formula": item.get("formula", ""),
            "split": item.get("split", ""),
            "target_original": target_original,
        }
        return graph, torch.tensor([target_model], dtype=torch.float32), metadata


def collate_alignn(batch):
    import dgl

    graphs = [item[0][0] for item in batch]
    line_graphs = [item[0][1] for item in batch]
    targets = torch.cat([item[1] for item in batch])
    metadata = [item[2] for item in batch]
    return dgl.batch(graphs), dgl.batch(line_graphs), targets, metadata


def inverse_targets(values: Iterable[float], target_mean: float, target_std: float, standardize: bool) -> np.ndarray:
    values_arr = np.asarray(list(values), dtype=float).reshape(-1)
    if not standardize:
        return values_arr
    return values_arr * float(target_std) + float(target_mean)


def evaluate(model, loader, criterion, device, target_mean: float, target_std: float, standardize: bool):
    model.eval()
    total_loss = 0.0
    total_count = 0
    pred_model: list[float] = []
    target_model: list[float] = []
    metadata_all: list[dict[str, object]] = []
    with torch.no_grad():
        for graph, line_graph, target, metadata in loader:
            graph = graph.to(device)
            line_graph = line_graph.to(device)
            target = target.to(device)
            output = model([graph, line_graph, None]).view(-1)
            target_flat = target.view(-1)
            loss = criterion(output, target_flat)
            count = int(target_flat.numel())
            total_loss += float(loss.item()) * count
            total_count += count
            pred_model.extend(float(x) for x in output.detach().cpu().numpy().reshape(-1))
            target_model.extend(float(x) for x in target_flat.detach().cpu().numpy().reshape(-1))
            metadata_all.extend(metadata)

    pred_original = inverse_targets(pred_model, target_mean, target_std, standardize)
    target_original = inverse_targets(target_model, target_mean, target_std, standardize)
    metrics = compute_metrics(target_original, pred_original)
    metrics["loss_model_space"] = total_loss / max(total_count, 1)
    metrics["n_samples"] = int(len(target_original))
    return metrics, pred_original, target_original, metadata_all


def write_predictions(path: Path, predictions: np.ndarray, targets: np.ndarray, metadata: list[dict[str, object]]) -> None:
    target_name = str(getattr(write_predictions, "target_name", "efermi"))
    target_col = f"target_{target_name}"
    pred_col = f"prediction_{target_name}"
    residual_col = f"residual_{target_name}"
    abs_error_col = f"abs_error_{target_name}"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row_id",
                "name",
                "formula",
                "split",
                target_col,
                pred_col,
                residual_col,
                abs_error_col,
            ],
        )
        writer.writeheader()
        for pred, target, meta in zip(predictions, targets, metadata):
            residual = float(pred) - float(target)
            writer.writerow(
                {
                    "row_id": meta.get("row_id", ""),
                    "name": meta.get("name", ""),
                    "formula": meta.get("formula", ""),
                    "split": meta.get("split", ""),
                    target_col: float(target),
                    pred_col: float(pred),
                    residual_col: residual,
                    abs_error_col: abs(residual),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="alignn_efermi_smoke")
    parser.add_argument("--target-name", default="efermi")
    parser.add_argument("--target-units", default="eV")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--test-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--alignn-layers", type=int, default=2)
    parser.add_argument("--gcn-layers", type=int, default=2)
    parser.add_argument("--hidden-features", type=int, default=64)
    parser.add_argument("--embedding-features", type=int, default=64)
    parser.add_argument("--cutoff", type=float, default=8.0)
    parser.add_argument("--max-neighbors", type=int, default=12)
    parser.add_argument("--loss", choices=["mse", "smoothl1", "l1"], default="mse")
    parser.add_argument("--no-standardize-target", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("DGLBACKEND", "pytorch")
    set_seed(args.seed)

    from alignn.models.alignn import ALIGNN, ALIGNNConfig

    db_path = args.db.expanduser().resolve()
    split_file = args.split_file.expanduser().resolve()
    output_dir = (args.output_dir.expanduser() / args.run_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_split_rows(split_file, "train", args.max_train_samples)
    val_rows = load_split_rows(split_file, "val", args.max_val_samples)
    test_rows = load_split_rows(split_file, "test", args.max_test_samples)
    train_targets = np.asarray([float(row["target"]) for row in train_rows], dtype=float)
    target_mean = float(np.mean(train_targets))
    target_std = float(np.std(train_targets))
    standardize = not bool(args.no_standardize_target)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu)

    config = {
        "dataset_name": "mathub2d_workfunction",
        "target": args.target_name,
        "target_units": args.target_units,
        "db": str(db_path),
        "db_sha256": sha256_file(db_path),
        "split_file": str(split_file),
        "split_sha256": sha256_file(split_file),
        "run_name": args.run_name,
        "python": sys.version,
        "seed": int(args.seed),
        "device": str(device),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "batch_size": int(args.batch_size),
        "test_batch_size": int(args.test_batch_size),
        "num_workers": int(args.num_workers),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "alignn_layers": int(args.alignn_layers),
        "gcn_layers": int(args.gcn_layers),
        "hidden_features": int(args.hidden_features),
        "embedding_features": int(args.embedding_features),
        "cutoff": float(args.cutoff),
        "max_neighbors": int(args.max_neighbors),
        "loss": args.loss,
        "standardize_target": bool(standardize),
        "target_mean_train": target_mean,
        "target_std_train": target_std,
        "split_counts": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
    }
    with (output_dir / "config.json").open("w") as handle:
        json.dump(to_jsonable(config), handle, indent=2)
    write_predictions.target_name = args.target_name

    print("=" * 80)
    print(f"mathub2d_workfunction ALIGNN {args.target_name}")
    print(json.dumps(to_jsonable(config), indent=2))
    print("=" * 80, flush=True)

    dataset_kwargs = {
        "db_path": db_path,
        "target_mean": target_mean,
        "target_std": target_std,
        "cutoff": args.cutoff,
        "max_neighbors": args.max_neighbors,
        "standardize": standardize,
    }
    train_dataset = AlignnEfermiDataset(split_rows=train_rows, **dataset_kwargs)
    val_dataset = AlignnEfermiDataset(split_rows=val_rows, **dataset_kwargs)
    test_dataset = AlignnEfermiDataset(split_rows=test_rows, **dataset_kwargs)

    loader_kwargs = {
        "collate_fn": collate_alignn,
        "num_workers": int(args.num_workers),
        "pin_memory": device.type == "cuda",
        "persistent_workers": int(args.num_workers) > 0,
    }
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **loader_kwargs)

    model_config = ALIGNNConfig(
        name="alignn",
        alignn_layers=args.alignn_layers,
        gcn_layers=args.gcn_layers,
        atom_input_features=92,
        edge_input_features=80,
        triplet_input_features=40,
        embedding_features=args.embedding_features,
        hidden_features=args.hidden_features,
        output_features=1,
    )
    model = ALIGNN(model_config).to(device)
    n_params = int(sum(parameter.numel() for parameter in model.parameters()))
    print(f"parameters={n_params:,}", flush=True)

    if args.loss == "smoothl1":
        criterion = torch.nn.SmoothL1Loss()
    elif args.loss == "l1":
        criterion = torch.nn.L1Loss()
    else:
        criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=max(len(train_loader), 1),
    )

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history: list[dict[str, object]] = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
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
            standardize,
        )
        record = {
            "epoch": epoch,
            "train_loss_model_space": train_loss,
            "val": val_metrics,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_time_seconds": time.time() - epoch_start,
        }
        history.append(record)
        print(
            f"Epoch {epoch:03d}/{args.epochs} train_loss={train_loss:.6f} "
            f"val_loss={val_metrics['loss_model_space']:.6f} "
            f"val_mae={val_metrics['mae']:.6f} val_r2={val_metrics['r2']:.6f} "
            f"time={record['epoch_time_seconds']:.1f}s",
            flush=True,
        )

        if float(val_metrics["loss_model_space"]) < best_val_loss:
            best_val_loss = float(val_metrics["loss_model_space"])
            best_epoch = int(epoch)
            patience_counter = 0
            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": to_jsonable(config),
                    "val_metrics": to_jsonable(val_metrics),
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "standardize_target": standardize,
                },
                output_dir / "best_model.pt",
            )
            print("  saved best_model.pt", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    train_metrics, train_pred, train_target, train_meta = evaluate(
        model,
        train_eval_loader,
        criterion,
        device,
        target_mean,
        target_std,
        standardize,
    )
    val_metrics, val_pred, val_target, val_meta = evaluate(
        model,
        val_loader,
        criterion,
        device,
        target_mean,
        target_std,
        standardize,
    )
    test_metrics, test_pred, test_target, test_meta = evaluate(
        model,
        test_loader,
        criterion,
        device,
        target_mean,
        target_std,
        standardize,
    )
    write_predictions(output_dir / "train_predictions.csv", train_pred, train_target, train_meta)
    write_predictions(output_dir / "val_predictions.csv", val_pred, val_target, val_meta)
    write_predictions(output_dir / "test_predictions.csv", test_pred, test_target, test_meta)

    with (output_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss_model_space",
                "val_loss_model_space",
                "val_mae",
                "val_rmse",
                "val_r2",
                "lr",
                "epoch_time_seconds",
            ],
        )
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "epoch": row["epoch"],
                    "train_loss_model_space": row["train_loss_model_space"],
                    "val_loss_model_space": row["val"]["loss_model_space"],
                    "val_mae": row["val"]["mae"],
                    "val_rmse": row["val"]["rmse"],
                    "val_r2": row["val"]["r2"],
                    "lr": row["lr"],
                    "epoch_time_seconds": row["epoch_time_seconds"],
                }
            )

    summary = {
        "config": config,
        "num_parameters": n_params,
        "best_epoch": best_epoch,
        "best_val_loss_model_space": best_val_loss,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "total_time_seconds": time.time() - start,
        "history": history,
    }
    with (output_dir / "metrics.json").open("w") as handle:
        json.dump(to_jsonable(summary), handle, indent=2)

    print("=" * 80)
    print(
        f"FINAL run={args.run_name} best_epoch={best_epoch} "
        f"train_mae={train_metrics['mae']:.6f} "
        f"val_mae={val_metrics['mae']:.6f} val_r2={val_metrics['r2']:.6f} "
        f"test_mae={test_metrics['mae']:.6f} test_r2={test_metrics['r2']:.6f}"
    )
    print(f"output_dir={output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
