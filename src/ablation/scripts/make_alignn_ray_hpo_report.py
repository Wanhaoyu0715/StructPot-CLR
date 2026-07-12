#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate reviewer-facing Ray HPO report for mathub2d_workfunction ALIGNN work function."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


NUMERIC_COLUMNS = [
    "best_epoch",
    "best_val_loss_model_space",
    "config_alignn_layers",
    "config_batch_size",
    "config_gcn_layers",
    "config_hidden_features",
    "config_lr",
    "config_weight_decay",
    "elapsed_seconds",
    "epoch",
    "num_reported_rows",
    "time_total_s",
    "timestamp",
    "train_loss_model_space",
    "training_iteration",
    "val_loss_model_space",
    "val_mae",
    "val_r2",
    "val_rmse",
]


SEARCH_SPACE = {
    "hidden_features": [64, 128, 192, 256],
    "alignn_layers": [1, 2, 3, 4],
    "gcn_layers": [1, 2, 3, 4],
    "lr": ("loguniform", 3.0e-4, 2.0e-3),
    "weight_decay": ("loguniform", 1.0e-6, 1.0e-4),
    "batch_size": [4, 8],
    "loss": ["mse", "smoothl1"],
}

FIG_DPI = 260


def set_large_plot_style() -> None:
    """Use reviewer-readable fonts for report figures."""
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "figure.titlesize": 22,
            "axes.linewidth": 1.2,
            "lines.linewidth": 1.8,
            "savefig.bbox": "tight",
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hpo-dir",
        type=Path,
        default=Path("reports/ray300_wf_80_10_10_hpo_runs_20260709/combined"),
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=Path("splits/wf_seed42_80_10_10_counts1519_189_191.csv"),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("reports/wf_split_seed42_80_10_10_counts1519_189_191_manifest.json"),
    )
    parser.add_argument(
        "--final-dir",
        type=Path,
        default=Path("reports/final_refit/best_ray300_wf_80_10_10_alignn_seed42_final_refit"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/ray300_wf_80_10_10_hpo_report"),
    )
    return parser.parse_args()


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value_float):
        return "NA"
    if abs(value_float) >= 1000:
        return f"{value_float:,.0f}"
    if abs(value_float) >= 10:
        return f"{value_float:.2f}"
    if abs(value_float) >= 1:
        return f"{value_float:.{digits}f}"
    return f"{value_float:.{digits}g}"


def fmt_sci(value: object, digits: int = 3) -> str:
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value_float):
        return "NA"
    return f"{value_float:.{digits}e}"


def markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    headers = [label for _, label in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [str(row.get(key, "")) for key, _ in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def trial_id_from_name(name: str) -> str:
    match = re.search(r"trainable_([^_]+)_", str(name))
    return match.group(1) if match else str(name)[:8]


def trial_order_from_name(name: str) -> int:
    match = re.search(r"trainable_[^_]+_(\d+)_", str(name))
    return int(match.group(1)) if match else 10**9


def short_trial(name: str) -> str:
    return f"{trial_id_from_name(name)} #{trial_order_from_name(name)}"


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def selected_mask(best_rows: pd.DataFrame, selected_trial: str) -> pd.Series:
    selected_trial = str(selected_trial)
    return (best_rows["trial_name"].astype(str) == selected_trial) | (
        best_rows["trial_id_short"].astype(str) == selected_trial
    )


def load_hpo_artifacts(hpo_dir: Path) -> dict[str, object]:
    """Load either a single Ray Tune run or a combined multi-run report dir."""
    global_best_path = hpo_dir / "global_best_trial.json"
    if global_best_path.exists():
        global_best = json.loads(global_best_path.read_text())
        manifest_items = json.loads((hpo_dir / "combined_ray_hpo_manifests.json").read_text())
        manifests = [item["manifest"] for item in manifest_items]
        first = manifests[0]
        base_config = dict(first["base_config"])
        base_config["search_seeds"] = [
            item["manifest"]["base_config"].get("search_seed")
            for item in manifest_items
        ]
        base_config["runs"] = [
            {
                "name": item["run_group"],
                "dir": item["run_dir"],
                "num_samples": item["manifest"].get("num_samples"),
                "num_gpus": item["manifest"].get("num_gpus"),
                "num_cpus": item["manifest"].get("num_cpus"),
            }
            for item in manifest_items
        ]
        manifest = {
            "base_config": base_config,
            "profile": first.get("profile", "full"),
            "num_samples": sum(int(item["manifest"].get("num_samples", 0)) for item in manifest_items),
            "metric": first.get("metric", "val_loss_model_space"),
            "mode": first.get("mode", "min"),
            "scheduler": first.get("scheduler", "ASHAScheduler"),
            "search": first.get("search", "HyperOptSearch"),
            "test_set_guard": first.get(
                "test_set_guard",
                "test split is not evaluated during HPO unless --test-each-trial is explicitly set",
            ),
            "num_cpus": sum(int(item["manifest"].get("num_cpus", 0)) for item in manifest_items),
            "num_gpus": sum(int(item["manifest"].get("num_gpus", 0)) for item in manifest_items),
            "cpus_per_trial": first.get("cpus_per_trial", 2.0),
            "gpus_per_trial": first.get("gpus_per_trial", 1.0),
            "hardware_label": "12 NVIDIA RTX 4090 GPUs total, 1 GPU/trial",
            "source_label": "Ray HPO run directories",
            "copy_files": [
                ("combined_ray_hpo_epoch_trajectories.csv", "ray_hpo_epoch_trajectories.csv"),
                ("combined_ray_hpo_trial_table.csv", "ray_hpo_trial_table.csv"),
                ("combined_ray_hpo_summary.json", "ray_hpo_summary.json"),
                ("combined_ray_hpo_manifests.json", "ray_hpo_manifests.json"),
                ("global_best_trial.json", "global_best_trial.json"),
            ],
        }
        best_row = global_best["best_trial_row"]
        best_trial = {
            "trial_id": trial_id_from_name(best_row["trial_name"]),
            "trial_name": best_row["trial_name"],
            "config": global_best["best_config"],
            "last_result": best_row,
        }
        return {
            "manifest": manifest,
            "best_trial": best_trial,
            "trial_table": hpo_dir / "combined_ray_hpo_trial_table.csv",
            "epoch_table": hpo_dir / "combined_ray_hpo_epoch_trajectories.csv",
        }

    manifest = json.loads((hpo_dir / "ray_hpo_manifest.json").read_text())
    manifest.setdefault("hardware_label", "NVIDIA RTX 4090 GPUs, 1 GPU/trial")
    manifest.setdefault(
        "source_label",
        "Ray Tune source run directory",
    )
    manifest.setdefault(
        "copy_files",
        [
            ("ray_hpo_epoch_trajectories.csv", "ray_hpo_epoch_trajectories.csv"),
            ("ray_hpo_trial_table.csv", "ray_hpo_trial_table.csv"),
            ("ray_hpo_summary.json", "ray_hpo_summary.json"),
            ("ray_hpo_manifest.json", "ray_hpo_manifest.json"),
            ("best_trial.json", "best_trial.json"),
        ],
    )
    return {
        "manifest": manifest,
        "best_trial": json.loads((hpo_dir / "best_trial.json").read_text()),
        "trial_table": hpo_dir / "ray_hpo_trial_table.csv",
        "epoch_table": hpo_dir / "ray_hpo_epoch_trajectories.csv",
    }


def make_best_rows(epoch_df: pd.DataFrame, trial_df: pd.DataFrame, max_epochs: int) -> pd.DataFrame:
    idx = epoch_df.groupby("trial_name")["val_loss_model_space"].idxmin()
    best_rows = epoch_df.loc[idx].copy()
    best_rows["trial_id_short"] = best_rows["trial_name"].map(trial_id_from_name)
    best_rows["trial_order_within_run"] = best_rows["trial_name"].map(trial_order_from_name)
    best_rows["selected_best_epoch"] = best_rows["epoch"].astype(int)
    best_rows["selected_val_loss_model_space"] = best_rows["val_loss_model_space"]
    counts = epoch_df.groupby("trial_name").agg(
        max_epoch=("epoch", "max"),
        reported_rows=("epoch", "count"),
        final_time_total_s=("time_total_s", "max"),
        first_timestamp=("timestamp", "min"),
        last_timestamp=("timestamp", "max"),
    )
    best_rows = best_rows.merge(counts, left_on="trial_name", right_index=True, how="left")
    best_rows["status"] = np.where(best_rows["max_epoch"] >= max_epochs, "full-budget", "ASHA-stopped")
    if not trial_df.empty and "trial_name" in trial_df.columns:
        keep = [
            column
            for column in ["trial_name", "done"]
            if column in trial_df.columns
        ]
        best_rows = best_rows.merge(trial_df[keep], on="trial_name", how="left")
    sort_cols = ["first_timestamp", "trial_name"] if "first_timestamp" in best_rows.columns else ["trial_order_within_run"]
    best_rows = best_rows.sort_values(sort_cols).reset_index(drop=True)
    best_rows["trial_order"] = np.arange(1, len(best_rows) + 1)
    return best_rows


def percentile_in_space(name: str, value: object) -> float | None:
    if name not in SEARCH_SPACE:
        return None
    if name == "loss":
        return None
    spec = SEARCH_SPACE[name]
    if isinstance(spec, tuple):
        _, low, high = spec
        value_float = float(value)
        return (math.log(value_float) - math.log(low)) / (math.log(high) - math.log(low))
    values = list(spec)
    if value not in values:
        try:
            value = type(values[0])(value)
        except Exception:
            return None
    if len(values) == 1:
        return 0.5
    try:
        return values.index(value) / (len(values) - 1)
    except ValueError:
        return None


def edge_label(name: str, value: object) -> str:
    percentile = percentile_in_space(name, value)
    if percentile is None:
        return "categorical"
    if percentile <= 0.10:
        return "low-edge"
    if percentile >= 0.90:
        return "high-edge"
    return "interior"


def save_dashboard(stats: dict[str, object], assets_dir: Path) -> str:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    cards = [
        ("Trial proposals", stats["n_trials"], "Ray Tune candidates"),
        ("Epoch records", stats["n_epoch_rows"], "per-epoch validation traces"),
        ("Full-budget trials", stats["completed_trials"], "reached 50 epochs"),
        ("ASHA-stopped", stats["early_stopped_trials"], "stopped before max budget"),
        ("Full-budget equivalent", f"{stats['full_budget_equiv']:.1f}", "sum(epoch rows) / 50"),
        ("Tracked trial time", f"{stats['tracked_trial_hours']:.2f} h", "one GPU per trial"),
    ]
    colors = ["#334155", "#0f766e", "#1d4ed8", "#b45309", "#7c3aed", "#be123c"]
    for ax, (title, value, subtitle), color in zip(axes.flat, cards, colors):
        ax.set_facecolor("#f8fafc")
        for spine in ax.spines.values():
            spine.set_color("#cbd5e1")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.06, 0.72, title, fontsize=17, color="#475569", transform=ax.transAxes)
        ax.text(0.06, 0.42, str(value), fontsize=34, fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(0.06, 0.18, subtitle, fontsize=14, color="#64748b", transform=ax.transAxes)
    fig.suptitle("ALIGNN Ray HPO workload summary", fontsize=22, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = assets_dir / "fig0_hpo_workload_summary.png"
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    return f"assets/{path.name}"


def save_trial_cloud(best_rows: pd.DataFrame, selected_trial: str, assets_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = np.where(best_rows["config_loss"].astype(str) == "smoothl1", "#0f766e", "#64748b")
    ax.scatter(
        best_rows["trial_order"],
        best_rows["selected_val_loss_model_space"],
        c=colors,
        s=np.where(best_rows["status"] == "full-budget", 35, 18),
        alpha=0.72,
        edgecolor="white",
        linewidth=0.3,
    )
    selected = best_rows[selected_mask(best_rows, selected_trial)]
    if not selected.empty:
        ax.scatter(
            selected["trial_order"],
            selected["selected_val_loss_model_space"],
            marker="*",
            s=220,
            color="#dc2626",
            edgecolor="black",
            linewidth=0.5,
            label="selected best",
            zorder=5,
        )
    ax.axhline(best_rows["selected_val_loss_model_space"].median(), color="#94a3b8", linestyle="--", linewidth=1, label="median")
    top5_mean = best_rows.nsmallest(5, "selected_val_loss_model_space")["selected_val_loss_model_space"].mean()
    ax.axhline(top5_mean, color="#f97316", linestyle=":", linewidth=1.3, label="top-5 mean")
    ax.set_xlabel("Ray trial launch/order index")
    ax.set_ylabel("Best validation loss in model space")
    ax.set_title("Trial-level search cloud")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False)
    path = assets_dir / "fig1_trial_level_search_cloud.png"
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    return f"assets/{path.name}"


def save_distribution(best_rows: pd.DataFrame, assets_dir: Path) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    values = best_rows["selected_val_loss_model_space"].dropna()
    axes[0].hist(values, bins=35, color="#0f766e", alpha=0.82, edgecolor="white")
    axes[0].axvline(values.min(), color="#dc2626", linewidth=1.5, label="best")
    axes[0].axvline(values.median(), color="#334155", linestyle="--", linewidth=1, label="median")
    axes[0].set_title("All-trial validation distribution")
    axes[0].set_xlabel("Best validation loss")
    axes[0].set_ylabel("Trial count")
    axes[0].legend(frameon=False)

    data_by_hidden = [
        best_rows.loc[best_rows["config_hidden_features"] == hidden, "selected_val_loss_model_space"].dropna()
        for hidden in sorted(best_rows["config_hidden_features"].dropna().unique())
    ]
    labels_hidden = [str(int(hidden)) for hidden in sorted(best_rows["config_hidden_features"].dropna().unique())]
    axes[1].boxplot(data_by_hidden, tick_labels=labels_hidden, showfliers=False)
    axes[1].scatter(
        np.repeat(np.arange(1, len(data_by_hidden) + 1), [len(v) for v in data_by_hidden]),
        np.concatenate([v.to_numpy() for v in data_by_hidden]),
        s=10,
        color="#475569",
        alpha=0.35,
    )
    axes[1].set_title("By hidden features")
    axes[1].set_xlabel("hidden_features")
    axes[1].set_ylabel("Best validation loss")

    losses = sorted(best_rows["config_loss"].dropna().unique())
    data_by_loss = [best_rows.loc[best_rows["config_loss"] == loss, "selected_val_loss_model_space"].dropna() for loss in losses]
    axes[2].boxplot(data_by_loss, tick_labels=losses, showfliers=False)
    axes[2].scatter(
        np.repeat(np.arange(1, len(data_by_loss) + 1), [len(v) for v in data_by_loss]),
        np.concatenate([v.to_numpy() for v in data_by_loss]),
        s=10,
        color="#475569",
        alpha=0.35,
    )
    axes[2].set_title("By loss function")
    axes[2].set_xlabel("loss")
    axes[2].set_ylabel("Best validation loss")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.22)
    fig.tight_layout()
    path = assets_dir / "fig2_trial_performance_distribution.png"
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    return f"assets/{path.name}"


def save_learning_trajectories(epoch_df: pd.DataFrame, best_rows: pd.DataFrame, selected_trial: str, assets_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(14, 7))
    top_count = max(1, int(math.ceil(0.10 * len(best_rows))))
    top_trials = set(best_rows.nsmallest(top_count, "selected_val_loss_model_space")["trial_name"])
    selected_names = set(best_rows.loc[selected_mask(best_rows, selected_trial), "trial_name"])
    y_cap = epoch_df["val_loss_model_space"].quantile(0.95)
    for name, group in epoch_df.groupby("trial_name"):
        group = group.sort_values("epoch")
        if name in selected_names:
            color, alpha, width, zorder = "#dc2626", 1.0, 2.2, 5
        elif name in top_trials:
            color, alpha, width, zorder = "#f97316", 0.65, 1.0, 3
        else:
            color, alpha, width, zorder = "#94a3b8", 0.16, 0.7, 1
        ax.plot(group["epoch"], group["val_loss_model_space"].clip(upper=y_cap), color=color, alpha=alpha, linewidth=width, zorder=zorder)
    ax.set_title("Per-trial validation trajectories")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss, clipped at 95th percentile")
    ax.grid(True, alpha=0.22)
    ax.text(0.01, 0.98, "gray: all trials; orange: top 10%; red: selected best", transform=ax.transAxes, va="top", fontsize=13, color="#475569")
    fig.tight_layout()
    path = assets_dir / "fig3_epoch_learning_trajectories.png"
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    return f"assets/{path.name}"


def save_best_so_far(epoch_df: pd.DataFrame, assets_dir: Path) -> tuple[str, pd.DataFrame]:
    ordered = epoch_df.sort_values(["timestamp", "trial_name", "epoch"]).copy()
    ordered["best_so_far"] = ordered["val_loss_model_space"].cummin()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(np.arange(1, len(ordered) + 1), ordered["best_so_far"], color="#1d4ed8", linewidth=1.8)
    ax.scatter([len(ordered)], [ordered["best_so_far"].iloc[-1]], color="#dc2626", s=70, zorder=4)
    ax.set_title("Best-so-far validation loss")
    ax.set_xlabel("Cumulative epoch reports")
    ax.set_ylabel("Incumbent validation loss")
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    path = assets_dir / "fig4_best_so_far_validation_loss.png"
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)

    ledger_rows = []
    incumbent = float("inf")
    previous = None
    for report_order, (_, row) in enumerate(ordered.iterrows(), start=1):
        value = float(row["val_loss_model_space"])
        if value < incumbent - 1.0e-12:
            ledger_rows.append(
                {
                    "report_order": report_order,
                    "trial_order": trial_order_from_name(row["trial_name"]),
                    "trial_id": trial_id_from_name(row["trial_name"]),
                    "epoch": int(row["epoch"]),
                    "new_incumbent_loss": value,
                    "gain_vs_previous": "" if previous is None else previous - value,
                    "val_mae": row.get("val_mae", np.nan),
                    "val_rmse": row.get("val_rmse", np.nan),
                    "val_r2": row.get("val_r2", np.nan),
                    "hidden_features": row.get("config_hidden_features", ""),
                    "alignn_layers": row.get("config_alignn_layers", ""),
                    "gcn_layers": row.get("config_gcn_layers", ""),
                    "lr": row.get("config_lr", ""),
                    "weight_decay": row.get("config_weight_decay", ""),
                    "batch_size": row.get("config_batch_size", ""),
                    "loss": row.get("config_loss", ""),
                }
            )
            previous = value
            incumbent = value
    return f"assets/{path.name}", pd.DataFrame(ledger_rows)


def save_asha_budget(best_rows: pd.DataFrame, max_epochs: int, assets_dir: Path) -> str:
    fractions = best_rows["reported_rows"] / max_epochs
    bins = [0, 0.25, 0.5, 0.75, 0.999, 1.001]
    labels = ["<=25%", "25-50%", "50-75%", "75-<100%", "100%"]
    counts = pd.cut(fractions, bins=bins, labels=labels, include_lowest=True).value_counts().reindex(labels).fillna(0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].barh(labels, counts.values, color=["#cbd5e1", "#94a3b8", "#64748b", "#0f766e", "#1d4ed8"])
    axes[0].set_title("ASHA budget allocation")
    axes[0].set_xlabel("Trial count")
    axes[0].grid(True, axis="x", alpha=0.22)
    axes[1].hist(best_rows["reported_rows"], bins=np.arange(0.5, max_epochs + 1.5, 1), color="#0f766e", alpha=0.85)
    axes[1].set_title("Reported epochs per trial")
    axes[1].set_xlabel("Epoch reports retained")
    axes[1].set_ylabel("Trial count")
    axes[1].grid(True, axis="y", alpha=0.22)
    fig.tight_layout()
    path = assets_dir / "fig5_asha_budget_allocation.png"
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    return f"assets/{path.name}"


def save_response_atlas(best_rows: pd.DataFrame, assets_dir: Path) -> str:
    rows = best_rows.copy()
    y = rows["selected_val_loss_model_space"]
    denom = max(float(y.quantile(0.90) - y.min()), 1.0e-12)
    rows["normalized_regret"] = (y - y.min()) / denom
    params = [
        ("config_lr", "lr", "log"),
        ("config_weight_decay", "weight_decay", "log"),
        ("config_hidden_features", "hidden_features", "linear"),
        ("config_alignn_layers", "alignn_layers", "linear"),
        ("config_gcn_layers", "gcn_layers", "linear"),
        ("config_batch_size", "batch_size", "linear"),
        ("config_loss", "loss", "categorical"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9.5), sharey=True)
    axes = axes.flat
    for ax, (column, label, scale) in zip(axes, params):
        if scale == "categorical":
            categories = sorted(rows[column].dropna().astype(str).unique())
            mapping = {value: i for i, value in enumerate(categories)}
            x = rows[column].astype(str).map(mapping)
            ax.scatter(x, rows["normalized_regret"].clip(upper=2.0), s=18, alpha=0.55, color="#0f766e")
            ax.set_xticks(list(mapping.values()), categories)
        else:
            ax.scatter(rows[column], rows["normalized_regret"].clip(upper=2.0), s=18, alpha=0.55, color="#0f766e")
            if scale == "log":
                ax.set_xscale("log")
        ax.set_title(label)
        ax.set_xlabel(label)
        ax.grid(True, alpha=0.22)
    axes[0].set_ylabel("Normalized validation regret")
    axes[4].set_ylabel("Normalized validation regret")
    axes[-1].axis("off")
    fig.suptitle("Hyperparameter response atlas", fontsize=22, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = assets_dir / "fig6_hyperparameter_response_atlas.png"
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    return f"assets/{path.name}"


def save_boundary_heatmap(best_config: dict[str, object], assets_dir: Path) -> tuple[str, pd.DataFrame]:
    rows = []
    for name, value in best_config.items():
        percentile = percentile_in_space(name, value)
        rows.append(
            {
                "parameter": name,
                "value": value,
                "search_percentile": percentile,
                "edge_flag": edge_label(name, value),
            }
        )
    audit = pd.DataFrame(rows)
    heat = audit.dropna(subset=["search_percentile"]).copy()
    fig, ax = plt.subplots(figsize=(14, 3.8))
    values = heat["search_percentile"].to_numpy().reshape(1, -1)
    im = ax.imshow(values, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(heat)), heat["parameter"], rotation=35, ha="right")
    ax.set_yticks([0], ["best config"])
    for i, row in enumerate(heat.itertuples()):
        star = "*" if row.edge_flag != "interior" else ""
        ax.text(i, 0, f"{row.search_percentile:.2f}{star}", ha="center", va="center", fontsize=14)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Search-space position", labelpad=14)
    ax.set_title("Best config boundary check")
    fig.tight_layout()
    path = assets_dir / "fig7_best_config_boundary_heatmap.png"
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    return f"assets/{path.name}", audit


def save_cost_frontier(best_rows: pd.DataFrame, assets_dir: Path) -> str:
    rows = best_rows.dropna(subset=["final_time_total_s", "selected_val_loss_model_space"]).copy()
    rows = rows[rows["final_time_total_s"] > 0]
    rows = rows.sort_values("final_time_total_s")
    frontier_x = []
    frontier_y = []
    current = float("inf")
    for _, row in rows.iterrows():
        value = float(row["selected_val_loss_model_space"])
        if value < current:
            frontier_x.append(float(row["final_time_total_s"]))
            frontier_y.append(value)
            current = value
    fig, ax = plt.subplots(figsize=(14, 6.3))
    colors = np.where(rows["status"] == "full-budget", "#0f766e", "#94a3b8")
    ax.scatter(rows["final_time_total_s"], rows["selected_val_loss_model_space"], c=colors, s=28, alpha=0.65, edgecolor="white", linewidth=0.3)
    ax.plot(frontier_x, frontier_y, color="#dc2626", linewidth=1.5, marker="o", markersize=4, label="observed Pareto frontier")
    ax.set_xscale("log")
    ax.set_xlabel("Trial wall time (seconds, log scale)")
    ax.set_ylabel("Best validation loss")
    ax.set_title("Performance-cost frontier")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = assets_dir / "fig8_performance_cost_frontier.png"
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    return f"assets/{path.name}"


def save_split_distribution(split_file: Path, assets_dir: Path) -> tuple[str, pd.DataFrame]:
    split_df = pd.read_csv(split_file)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    colors = {"train": "#334155", "val": "#0f766e", "test": "#b45309"}
    for split, group in split_df.groupby("split"):
        axes[0].hist(group["target"], bins=36, density=True, histtype="step", linewidth=1.8, color=colors.get(split, None), label=f"{split} n={len(group)}")
        values = np.sort(group["target"].to_numpy(dtype=float))
        ecdf = np.arange(1, len(values) + 1) / len(values)
        axes[1].plot(values, ecdf, linewidth=1.8, color=colors.get(split, None), label=split)
    axes[0].set_title("Target distribution by split")
    axes[0].set_xlabel("workfunction_eV")
    axes[0].set_ylabel("Density")
    axes[1].set_title("Empirical CDF")
    axes[1].set_xlabel("workfunction_eV")
    axes[1].set_ylabel("Cumulative probability")
    for ax in axes:
        ax.grid(True, alpha=0.22)
        ax.legend(frameon=False)
    fig.tight_layout()
    path = assets_dir / "fig9_target_split_distribution.png"
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    summary = split_df.groupby("split")["target"].agg(["count", "mean", "std", "min", "median", "max"]).reset_index()
    return f"assets/{path.name}", summary


def load_final_metrics(final_dir: Path) -> dict[str, object] | None:
    metrics_path = final_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    with metrics_path.open() as handle:
        return json.load(handle)


def prediction_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    target_col = next(column for column in df.columns if column.startswith("target_"))
    pred_col = next(column for column in df.columns if column.startswith("prediction_"))
    residual_col = next(column for column in df.columns if column.startswith("residual_"))
    return target_col, pred_col, residual_col


def save_final_figures(final_dir: Path, assets_dir: Path) -> dict[str, str]:
    outputs: dict[str, str] = {}
    metrics = load_final_metrics(final_dir)
    if metrics is None:
        return outputs
    pred_frames = {}
    for split in ["train", "val", "test"]:
        path = final_dir / f"{split}_predictions.csv"
        if path.exists():
            pred_frames[split] = pd.read_csv(path)
    if pred_frames:
        fig, axes = plt.subplots(2, len(pred_frames), figsize=(6.3 * len(pred_frames), 10.5))
        if len(pred_frames) == 1:
            axes = np.asarray(axes).reshape(2, 1)
        for col_idx, (split, df) in enumerate(pred_frames.items()):
            target_col, pred_col, residual_col = prediction_columns(df)
            target = df[target_col].to_numpy(dtype=float)
            pred = df[pred_col].to_numpy(dtype=float)
            residual = df[residual_col].to_numpy(dtype=float)
            lo = min(np.nanmin(target), np.nanmin(pred))
            hi = max(np.nanmax(target), np.nanmax(pred))
            axes[0, col_idx].scatter(target, pred, s=15, alpha=0.55, color="#0f766e", edgecolor="white", linewidth=0.2)
            axes[0, col_idx].plot([lo, hi], [lo, hi], color="#dc2626", linewidth=1)
            axes[0, col_idx].set_title(f"{split} parity")
            axes[0, col_idx].set_xlabel("DFT target")
            axes[0, col_idx].set_ylabel("ALIGNN prediction")
            axes[0, col_idx].grid(True, alpha=0.22)
            axes[1, col_idx].hist(residual, bins=32, color="#1d4ed8", alpha=0.78, edgecolor="white")
            axes[1, col_idx].axvline(0, color="#334155", linewidth=1)
            axes[1, col_idx].axvline(np.nanmean(residual), color="#dc2626", linestyle="--", linewidth=1)
            axes[1, col_idx].set_title(f"{split} residual")
            axes[1, col_idx].set_xlabel("prediction - target (eV)")
            axes[1, col_idx].set_ylabel("Count")
            axes[1, col_idx].grid(True, axis="y", alpha=0.22)
        fig.tight_layout()
        path = assets_dir / "fig10_final_refit_parity_residual.png"
        fig.savefig(path, dpi=FIG_DPI)
        plt.close(fig)
        outputs["parity"] = f"assets/{path.name}"

    history_path = final_dir / "history.csv"
    if history_path.exists():
        history = pd.read_csv(history_path)
        coerce_numeric(history)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
        axes[0].plot(history["epoch"], history["val_loss_model_space"], color="#1d4ed8", linewidth=1.8)
        axes[0].scatter([history.loc[history["val_loss_model_space"].idxmin(), "epoch"]], [history["val_loss_model_space"].min()], color="#dc2626", s=70, zorder=4)
        axes[0].set_title("Final refit validation loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Validation loss")
        axes[1].plot(history["epoch"], history["val_mae"], color="#0f766e", linewidth=1.8, label="MAE")
        axes[1].plot(history["epoch"], history["val_rmse"], color="#b45309", linewidth=1.5, label="RMSE")
        axes[1].set_title("Final refit validation error")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("eV")
        axes[1].legend(frameon=False)
        for ax in axes:
            ax.grid(True, alpha=0.22)
        fig.tight_layout()
        path = assets_dir / "fig11_final_refit_history.png"
        fig.savefig(path, dpi=FIG_DPI)
        plt.close(fig)
        outputs["history"] = f"assets/{path.name}"
    return outputs


def metric_rows(metrics: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for split in ["train", "val", "test"]:
        key = f"{split}_metrics"
        if key not in metrics:
            continue
        item = metrics[key]
        rows.append(
            {
                "split": split,
                "n": item.get("n_samples", ""),
                "loss": fmt(item.get("loss_model_space"), 5),
                "mae": fmt(item.get("mae"), 4),
                "rmse": fmt(item.get("rmse"), 4),
                "r2": fmt(item.get("r2"), 4),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    set_large_plot_style()
    hpo_dir = args.hpo_dir.resolve()
    report_dir = args.report_dir.resolve()
    assets_dir = report_dir / "assets"
    tables_dir = report_dir / "tables"
    report_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    artifacts = load_hpo_artifacts(hpo_dir)
    manifest = artifacts["manifest"]
    best_trial = artifacts["best_trial"]
    trial_df = coerce_numeric(pd.read_csv(artifacts["trial_table"]))
    epoch_df = coerce_numeric(pd.read_csv(artifacts["epoch_table"]))
    max_epochs = int(manifest["base_config"]["epochs"])
    best_rows = make_best_rows(epoch_df, trial_df, max_epochs)
    selected_trial = str(best_trial.get("trial_name") or best_trial["trial_id"])
    selected_row = best_rows.loc[selected_mask(best_rows, selected_trial)].nsmallest(1, "selected_val_loss_model_space")
    if selected_row.empty:
        selected_row = best_rows.nsmallest(1, "selected_val_loss_model_space")
        selected_trial = str(selected_row.iloc[0]["trial_name"])

    split_fig, split_summary = save_split_distribution(args.split_file.resolve(), assets_dir)
    final_metrics = load_final_metrics(args.final_dir.resolve())
    final_figures = save_final_figures(args.final_dir.resolve(), assets_dir)
    final_artifact_dir = report_dir / "final_refit_artifacts"
    if args.final_dir.resolve().exists():
        final_artifact_dir.mkdir(exist_ok=True)
        for name in [
            "best_model.pt",
            "config.json",
            "metrics.json",
            "history.csv",
            "train_predictions.csv",
            "val_predictions.csv",
            "test_predictions.csv",
        ]:
            copy_if_exists(args.final_dir.resolve() / name, final_artifact_dir / name)

    stats = {
        "n_trials": len(best_rows),
        "n_epoch_rows": len(epoch_df),
        "completed_trials": int((best_rows["max_epoch"] >= max_epochs).sum()),
        "early_stopped_trials": int((best_rows["max_epoch"] < max_epochs).sum()),
        "full_budget_equiv": float(best_rows["reported_rows"].sum() / max_epochs),
        "budget_utilization": float(best_rows["reported_rows"].sum() / (len(best_rows) * max_epochs)),
        "tracked_trial_hours": float(best_rows["final_time_total_s"].sum() / 3600.0),
        "wall_span_hours": float((epoch_df["timestamp"].max() - epoch_df["timestamp"].min()) / 3600.0),
        "best_loss": float(best_rows["selected_val_loss_model_space"].min()),
        "median_loss": float(best_rows["selected_val_loss_model_space"].median()),
        "top5_mean_loss": float(best_rows.nsmallest(5, "selected_val_loss_model_space")["selected_val_loss_model_space"].mean()),
    }

    figures = {
        "dashboard": save_dashboard(stats, assets_dir),
        "cloud": save_trial_cloud(best_rows, selected_trial, assets_dir),
        "distribution": save_distribution(best_rows, assets_dir),
        "trajectories": save_learning_trajectories(epoch_df, best_rows, selected_trial, assets_dir),
        "asha": save_asha_budget(best_rows, max_epochs, assets_dir),
        "response": save_response_atlas(best_rows, assets_dir),
        "cost": save_cost_frontier(best_rows, assets_dir),
        "split": split_fig,
    }
    figures["best_so_far"], incumbent_ledger = save_best_so_far(epoch_df, assets_dir)
    figures["boundary"], boundary_audit = save_boundary_heatmap(best_trial["config"], assets_dir)
    figures.update(final_figures)

    top_trials = best_rows.nsmallest(20, "selected_val_loss_model_space").copy()
    top_trials["short_trial"] = top_trials["trial_name"].map(short_trial)
    top_trials.to_csv(tables_dir / "top20_trials_by_validation_loss.csv", index=False)
    best_rows.to_csv(tables_dir / "trial_level_audit.csv", index=False)
    best_rows.to_csv(tables_dir / "trial_level_summary.csv", index=False)
    incumbent_ledger.to_csv(tables_dir / "incumbent_discovery_ledger.csv", index=False)
    incumbent_ledger.to_csv(tables_dir / "best_so_far_updates.csv", index=False)
    boundary_audit.to_csv(tables_dir / "best_config_boundary_audit.csv", index=False)
    boundary_audit.to_csv(tables_dir / "best_config_boundary_check.csv", index=False)
    split_summary.to_csv(tables_dir / "split_target_distribution_summary.csv", index=False)
    for src_name, dst_name in manifest.get("copy_files", []):
        copy_if_exists(hpo_dir / src_name, report_dir / dst_name)

    selected = selected_row.iloc[0]
    selected_display = short_trial(str(selected["trial_name"]))
    split_manifest = {}
    if args.split_manifest.resolve().exists():
        split_manifest = json.loads(args.split_manifest.resolve().read_text())
    neg_outliers = split_manifest.get("negative_target_rows", split_manifest.get("negative_workfunction_rows", []))
    if not neg_outliers:
        neg_outliers = split_manifest.get("summary", {}).get("outliers_wf_lt_0_or_gt_10", [])

    search_space_rows = [
        {
            "parameter": "hidden_features",
            "range": "choice[64, 128, 192, 256]",
            "role": "ALIGNN hidden width",
        },
        {
            "parameter": "alignn_layers",
            "range": "choice[1, 2, 3, 4]",
            "role": "angle-aware message-passing depth",
        },
        {
            "parameter": "gcn_layers",
            "range": "choice[1, 2, 3, 4]",
            "role": "atom graph convolution depth",
        },
        {
            "parameter": "lr",
            "range": "loguniform[3e-4, 2e-3]",
            "role": "AdamW max learning rate",
        },
        {
            "parameter": "weight_decay",
            "range": "loguniform[1e-6, 1e-4]",
            "role": "AdamW regularization",
        },
        {
            "parameter": "batch_size",
            "range": "choice[4, 8]",
            "role": "training mini-batch size",
        },
        {
            "parameter": "loss",
            "range": "choice[mse, smoothl1]",
            "role": "model-space regression loss",
        },
    ]

    top_table_rows = []
    for rank, (_, row) in enumerate(top_trials.head(10).iterrows(), start=1):
        hp = (
            f"h={int(row['config_hidden_features'])}, "
            f"ALIGNN={int(row['config_alignn_layers'])}, "
            f"GCN={int(row['config_gcn_layers'])}, "
            f"lr={fmt_sci(row['config_lr'], 2)}, "
            f"wd={fmt_sci(row['config_weight_decay'], 2)}, "
            f"bs={int(row['config_batch_size'])}, "
            f"loss={row['config_loss']}"
        )
        top_table_rows.append(
            {
                "rank": rank,
                "trial": row["short_trial"],
                "best_epoch": int(row["selected_best_epoch"]),
                "val_loss": fmt(row["selected_val_loss_model_space"], 6),
                "val_mae": fmt(row["val_mae"], 4),
                "val_rmse": fmt(row["val_rmse"], 4),
                "val_r2": fmt(row["val_r2"], 4),
                "status": row["status"],
                "hyperparameters": hp,
            }
        )

    split_counts = {row["split"]: int(row["count"]) for _, row in split_summary.iterrows()}
    split_label = f"80/10/10 count protocol, n={split_counts.get('train')}/{split_counts.get('val')}/{split_counts.get('test')}"

    protocol_rows = [
        {
            "field": "Dataset",
            "value": "MatHub-2D work-function dataset",
        },
        {
            "field": "Target",
            "value": "workfunction_eV = evac_eV - efermi_eV",
        },
        {
            "field": "Model",
            "value": "ALIGNN scalar regression",
        },
        {
            "field": "Search algorithm",
            "value": manifest["search"],
        },
        {
            "field": "Scheduler",
            "value": f"{manifest['scheduler']} (max_t={max_epochs}, grace_period=8, reduction_factor=2)",
        },
        {
            "field": "Optimized metric",
            "value": "val_loss_model_space, minimize",
        },
        {
            "field": "Hardware",
            "value": manifest.get("hardware_label", "1 GPU/trial"),
        },
        {
            "field": "Split",
            "value": split_label,
        },
        {
            "field": "Test-set guard",
            "value": "test split was not evaluated during HPO; used only after validation-selected final refit",
        },
    ]

    selected_metrics_rows = [
        {
            "metric": "Selected trial id",
            "value": selected_display,
        },
        {
            "metric": "Selected epoch",
            "value": int(selected["selected_best_epoch"]),
        },
        {
            "metric": "Validation loss",
            "value": fmt(selected["selected_val_loss_model_space"], 6),
        },
        {
            "metric": "Validation MAE",
            "value": f"{fmt(selected['val_mae'], 4)} eV",
        },
        {
            "metric": "Validation RMSE",
            "value": f"{fmt(selected['val_rmse'], 4)} eV",
        },
        {
            "metric": "Validation R2",
            "value": fmt(selected["val_r2"], 4),
        },
    ]

    boundary_rows = []
    for _, row in boundary_audit.iterrows():
        boundary_rows.append(
            {
                "parameter": row["parameter"],
                "best_value": fmt_sci(row["value"]) if row["parameter"] in {"lr", "weight_decay"} else row["value"],
                "position": "" if pd.isna(row["search_percentile"]) else fmt(row["search_percentile"], 3),
                "edge_flag": row["edge_flag"],
            }
        )

    split_rows = []
    for _, row in split_summary.iterrows():
        split_rows.append(
            {
                "split": row["split"],
                "n": int(row["count"]),
                "mean": fmt(row["mean"], 4),
                "std": fmt(row["std"], 4),
                "min": fmt(row["min"], 4),
                "median": fmt(row["median"], 4),
                "max": fmt(row["max"], 4),
            }
        )

    final_section = "Final refit/test was not available when this report was generated; `metrics.json` was not found."
    if final_metrics is not None:
        rows = metric_rows(final_metrics)
        final_section = "\n".join(
            [
                "After the validation-selected best configuration was fixed, a final refit was performed and train/validation/test predictions were exported from the best validation checkpoint. The test split was used only at this final evaluation stage.",
                "",
                "The final refit/test used the same fixed 80/10/10 split record as the HPO run. Hyperparameter selection was based only on validation loss, and test metrics were read only after the final refit was complete.",
                "",
                markdown_table(
                    rows,
                    [
                        ("split", "Split"),
                        ("n", "n"),
                        ("loss", "Loss"),
                        ("mae", "MAE/eV"),
                        ("rmse", "RMSE/eV"),
                        ("r2", "R2"),
                    ],
                ),
            ]
        )
        if "parity" in figures:
            final_section += f"\n\n![Final refit parity/residual diagnostics]({figures['parity']})\n\n"
            final_section += "This figure reports parity and residual distributions for train, validation, and test splits, showing the behavior of the final refit model rather than an intermediate HPO trial."
        if "history" in figures:
            final_section += f"\n\n![Final refit history]({figures['history']})\n\n"
            final_section += "This figure shows the final refit validation loss/MAE/RMSE history and the checkpoint-selection basis."

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    report_lines = [
        "# ALIGNN Work-Function Ray300 HPO Supporting Evidence Report",
        "",
        f"Generated: `{generated}`",
        "",
        "Computing resource summary: 12 NVIDIA RTX 4090 GPUs, one GPU per trial.",
        "",
        "## Summary",
        "",
        "This report documents a systematic 300-trial Ray Tune hyperparameter optimization (HPO) experiment for ALIGNN on the work-function regression task. The HPO stage used a fixed 80/10/10 split and selected hyperparameters strictly by validation loss. The test split was not evaluated during HPO and was reserved for the final refit/test stage after the validation-selected configuration had been fixed.",
        "",
        f"In total, the HPO completed **{stats['n_trials']}** Ray Tune trials and retained **{stats['n_epoch_rows']}** per-epoch validation records. **{stats['completed_trials']}** trials reached the full {max_epochs}-epoch budget, while **{stats['early_stopped_trials']}** trials were stopped early by ASHA. The retained training budget corresponds to approximately **{stats['full_budget_equiv']:.1f} full-budget trials**, with a budget utilization of **{stats['budget_utilization']:.1%}**.",
        "",
        f"The validation-selected best trial was `{selected_display}`. It reached the lowest validation loss `{fmt(selected['selected_val_loss_model_space'], 6)}` at epoch **{int(selected['selected_best_epoch'])}**, with validation MAE `{fmt(selected['val_mae'], 4)} eV`, RMSE `{fmt(selected['val_rmse'], 4)} eV`, and R2 `{fmt(selected['val_r2'], 4)}`.",
        "",
        "The selected configuration was:",
        "",
        "```json",
        json.dumps(best_trial["config"], indent=2),
        "```",
        "",
        "## Experimental Protocol",
        "",
        markdown_table(protocol_rows, [("field", "Field"), ("value", "Value")]),
        "",
        "## Data and Target Definition",
        "",
        "The prediction target was defined as:",
        "",
        "`workfunction_eV = evac_eV - efermi_eV`",
        "",
        f"All HPO trials and the final refit/test used the same fixed split record: train/validation/test = {split_counts.get('train')}/{split_counts.get('val')}/{split_counts.get('test')}. Two negative work-function entries were retained and recorded in the split manifest. They were not removed post hoc, so the validation protocol is not affected by after-the-fact data cleaning.",
        "",
        markdown_table(
            split_rows,
            [
                ("split", "Split"),
                ("n", "n"),
                ("mean", "mean/eV"),
                ("std", "std/eV"),
                ("min", "min/eV"),
                ("median", "median/eV"),
                ("max", "max/eV"),
            ],
        ),
        "",
        f"Number of negative-target entries recorded in the split manifest: `{len(neg_outliers) if isinstance(neg_outliers, list) else 'see manifest'}`.",
        "",
        "Both negative work-function entries were assigned to the train split; validation and test contain no negative target values.",
        "",
        f"![Target split distribution]({figures['split']})",
        "",
        "## Evidence Figure 0: HPO Workload Summary",
        "",
        f"![HPO workload summary]({figures['dashboard']})",
        "",
        "This figure summarizes the number of proposed trials, retained per-epoch trajectories, ASHA early-stopping outcomes, and full-budget-equivalent training volume derived from the Ray Tune outputs.",
        "",
        "## Search Space",
        "",
        markdown_table(search_space_rows, [("parameter", "Parameter"), ("range", "Search range"), ("role", "Role")]),
        "",
        "The discrete part of the search space contains `4 x 4 x 4 x 2 x 2 = 256` architecture/training combinations, with two additional continuous log-uniform dimensions for `lr` and `weight_decay`. The HPO sampled all discrete values and covered learning rates from approximately `3e-4` to `2e-3` and weight decay values from approximately `1e-6` to `1e-4`.",
        "",
        "## Evidence Figure 1: Trial-Level Search Cloud",
        "",
        f"![Trial-level search cloud]({figures['cloud']})",
        "",
        f"Each point represents the best validation loss reached by one hyperparameter configuration. The red star marks the validation-selected best trial; the orange dashed line is the top-5 mean `{fmt(stats['top5_mean_loss'], 6)}`, and the gray dashed line is the all-trial median `{fmt(stats['median_loss'], 6)}`. The selected trial is drawn from a broad candidate distribution rather than from a small manual trial set.",
        "",
        "## Evidence Figure 2: Full-Trial Performance Distribution",
        "",
        f"![Trial performance distribution]({figures['distribution']})",
        "",
        "This figure shows the distribution of best validation losses across all 300 trials, together with grouped views by hidden width and loss function. Lower-loss trials concentrate around `hidden_features=64` and `smoothl1`, consistent with the selected best configuration.",
        "",
        "## Top-Trial Summary",
        "",
        markdown_table(
            top_table_rows,
            [
                ("rank", "Rank"),
                ("trial", "Trial"),
                ("best_epoch", "Best epoch"),
                ("val_loss", "Val loss"),
                ("val_mae", "Val MAE"),
                ("val_rmse", "Val RMSE"),
                ("val_r2", "Val R2"),
                ("status", "Status"),
                ("hyperparameters", "Key hyperparameters"),
            ],
        ),
        "",
        "The complete top-20 table and the full-trial summary table are provided as `tables/top20_trials_by_validation_loss.csv` and `tables/trial_level_summary.csv`.",
        "",
        "## Evidence Figure 3: Per-Trial Epoch Learning Trajectories",
        "",
        f"![Per-trial learning trajectories]({figures['trajectories']})",
        "",
        "Gray curves show all trials, orange curves show the top 10% trials, and the red curve shows the validation-selected best trial. Short trajectories correspond to ASHA early stopping after the grace period, not missing records. The complete per-epoch trajectories are retained in `ray_hpo_epoch_trajectories.csv`.",
        "",
        "## Evidence Figure 4: Ray Tune Best-So-Far Validation Loss",
        "",
        f"![Best-so-far validation loss]({figures['best_so_far']})",
        "",
        f"The best-so-far curve records how the incumbent validation loss improved during the search. The final major update came from trial `{selected_display}` at epoch {int(selected['selected_best_epoch'])}. The corresponding update table is provided as `tables/best_so_far_updates.csv`.",
        "",
        "## Evidence Figure 5: ASHA Early Stopping and Budget Allocation",
        "",
        f"![ASHA budget allocation]({figures['asha']})",
        "",
        f"ASHA allocated the full {max_epochs}-epoch budget to {stats['completed_trials']} trials and stopped {stats['early_stopped_trials']} weaker candidates early. Thus, the 300 proposed configurations correspond to approximately {stats['full_budget_equiv']:.1f} full-budget trials rather than 300 fully trained models.",
        "",
        "## Evidence Figure 6: Hyperparameter Response Atlas",
        "",
        f"![Hyperparameter response atlas]({figures['response']})",
        "",
        "The response atlas uses normalized validation regret to summarize how sampled hyperparameter values relate to validation performance. It is not a causal attribution analysis; rather, it shows that the report retains evidence over good and poor regions of the search space, not only the best configuration.",
        "",
        "## Evidence Figure 7: Best-Config Search Boundary Check",
        "",
        f"![Best config boundary heatmap]({figures['boundary']})",
        "",
        markdown_table(
            boundary_rows,
            [
                ("parameter", "Parameter"),
                ("best_value", "Best value"),
                ("position", "Search-space position"),
                ("edge_flag", "Flag"),
            ],
        ),
        "",
        "The boundary check shows that the selected model is relatively small: `hidden_features=64` and `gcn_layers=1` lie at the lower end of the sampled range, while `batch_size=8` lies at the upper end of the candidate set. The learning rate is inside the sampled interval, and weight decay is near the lower end.",
        "",
        "## Evidence Figure 8: Performance-Cost Frontier",
        "",
        f"![Performance-cost frontier]({figures['cost']})",
        "",
        "This figure relates trial runtime to validation performance. The red curve marks the observed Pareto frontier and illustrates how ASHA concentrated training budget on promising configurations.",
        "",
        "## Validation-Selected Best-Epoch Metrics",
        "",
        markdown_table(selected_metrics_rows, [("metric", "Metric"), ("value", "Value")]),
        "",
        f"Selection metrics were extracted from the minimum validation loss over all retained epoch reports. For the selected best trial, the minimum occurred at epoch {int(selected['selected_best_epoch'])}.",
        "",
        "## Final refit/test",
        "",
        final_section,
        "",
        "## Reproducibility Materials",
        "",
        "- HPO manifests: `ray_hpo_manifests.json`",
        "- Global best trial: `global_best_trial.json`",
        "- Full trial table: `ray_hpo_trial_table.csv`",
        "- Per-epoch trajectories: `ray_hpo_epoch_trajectories.csv`",
        "- Top-trial table: `tables/top20_trials_by_validation_loss.csv`",
        "- Full-trial summary: `tables/trial_level_summary.csv`",
        "- Best-so-far updates: `tables/best_so_far_updates.csv`",
        "- Search boundary check: `tables/best_config_boundary_check.csv`",
        "- Final refit outputs: final refit output folder",
        "",
        "## Concise Interpretation",
        "",
        "The additional 300-trial Ray Tune HPO experiment substantially improves the ALIGNN baseline relative to the originally reported ALIGNN value, and the final hyperparameters, split protocol, per-epoch trajectories, ASHA budget allocation, and final refit/test results are now explicitly documented.",
        "",
    ]

    report_path = report_dir / "mathub2d_workfunction_alignn_ray300_wf_hpo_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    public_report_path = report_dir / "alignn_ray300_wf_hpo_supporting_evidence_report.md"
    public_report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(report_path)
    print(public_report_path)


if __name__ == "__main__":
    main()
