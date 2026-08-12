#!/usr/bin/env python3
"""
Generate a detailed report for the matched JetClass-II medium comparison.

Outputs separate presentation-ready PNGs, CSV tables, a tag inventory,
and a Markdown summary under:

/kaushik-moe-vol/outputs/figures/medium-comparison-detailed
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TB_ROOT = Path("/kaushik-moe-vol/outputs/tensorboard")
LOG_ROOT = Path("/kaushik-moe-vol/outputs/logs")
OUT = Path("/kaushik-moe-vol/outputs/figures/medium-comparison-detailed")


@dataclass(frozen=True)
class RunConfig:
    label: str
    model_dir: str
    token: str
    reported_flops_m: float
    total_params_m: float


RUNS = [
    RunConfig(
        label="Dense ParT",
        model_dir="ParT",
        token="jc2-medium-part-e3-s1m-v600k",
        reported_flops_m=669.84,
        total_params_m=2.3,
    ),
    RunConfig(
        label="MoEParT E4 K1",
        model_dir="MPT",
        token="jc2-medium-mpt-e4-k1-e3-s1m-v600k",
        reported_flops_m=503.12,
        total_params_m=6.3,
    ),
]


TAG_CANDIDATES = {
    "validation_accuracy": [
        "Acc/eval (epoch)",
        "Acc/eval",
    ],
    "validation_loss": [
        "Loss/eval (epoch)",
        "Loss/eval",
    ],
    "training_accuracy_epoch": [
        "Acc/train (epoch)",
        "Acc/train_epoch",
    ],
    "training_loss_epoch": [
        "Loss/train (epoch)",
        "Loss/train_epoch",
    ],
    "gradient_norm": [
        "GradNorm/train",
        "GradientNorm/train",
    ],
    "learning_rate": [
        "LR/train",
        "LearningRate/train",
        "learning_rate",
    ],
    "roc_auc": [
        "roc_auc_score/eval (epoch)",
        "ROC_AUC/eval (epoch)",
        "AUC/eval (epoch)",
    ],
}


def newest(paths: Iterable[Path]) -> Path:
    paths = list(paths)
    if not paths:
        raise FileNotFoundError("No matching files found.")
    return max(paths, key=lambda p: p.stat().st_mtime)


def find_event_dir(token: str) -> Path:
    files = [
        p for p in TB_ROOT.rglob("events.out.tfevents.*")
        if token in str(p)
    ]
    if not files:
        raise FileNotFoundError(
            f"No TensorBoard event file found for run token: {token}"
        )
    return newest(files).parent


def find_log(model_dir: str, token: str) -> Path | None:
    root = LOG_ROOT / model_dir
    if not root.exists():
        return None
    matches = [p for p in root.rglob("*.log") if token in p.name]
    return newest(matches) if matches else None


def load_event_accumulator(run_dir: Path) -> EventAccumulator:
    ea = EventAccumulator(
        str(run_dir),
        size_guidance={"scalars": 0},
    )
    ea.Reload()
    return ea


def first_available_tag(
    available: list[str],
    candidates: list[str],
) -> str | None:
    for tag in candidates:
        if tag in available:
            return tag
    return None


def scalar_frame(
    ea: EventAccumulator,
    tag: str,
    epoch_axis: bool,
) -> pd.DataFrame:
    events = ea.Scalars(tag)
    frame = pd.DataFrame({
        "step": [event.step for event in events],
        "value": [event.value for event in events],
        "wall_time": [event.wall_time for event in events],
    })
    if epoch_axis:
        frame["epoch"] = frame["step"] + 1
    return frame


def save_line_plot(
    frames: dict[str, pd.DataFrame],
    x_col: str,
    y_scale: float,
    filename: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    if not frames:
        return

    plt.figure(figsize=(7.5, 4.8))

    for label, frame in frames.items():
        plt.plot(
            frame[x_col],
            frame["value"] * y_scale,
            marker="o" if len(frame) <= 20 else None,
            linewidth=2,
            label=label,
        )

    if x_col == "epoch":
        ticks = sorted({
            int(x)
            for frame in frames.values()
            for x in frame[x_col].tolist()
        })
        plt.xticks(ticks)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=300)
    plt.close()


def save_bar_plot(
    values: dict[str, float],
    filename: str,
    title: str,
    ylabel: str,
    value_format: str = "{:.2f}",
) -> None:
    if not values:
        return

    labels = list(values.keys())
    numbers = list(values.values())

    plt.figure(figsize=(7.0, 4.8))
    bars = plt.bar(labels, numbers)

    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, numbers):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            value_format.format(value),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=300)
    plt.close()


def parse_training_log(path: Path | None) -> dict:
    result = {
        "throughput": [],
        "peak_cuda_memory_mb": None,
        "gpu_lines": [],
    }

    if path is None or not path.exists():
        return result

    epoch = None
    phase = None

    epoch_re = re.compile(
        r"Epoch\s+#?(\d+)\s+(training|validating)",
        re.IGNORECASE,
    )
    processed_re = re.compile(
        r"Processed\s+([\d,]+)\s+entries.*?"
        r"avg\.\s*speed\s+([0-9.]+)\s+entries/s",
        re.IGNORECASE,
    )
    memory_re = re.compile(
        r"Max CUDA memory.*?([0-9.]+)\s*(?:MB|MiB)",
        re.IGNORECASE,
    )

    for raw_line in path.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines():
        line = raw_line.strip()

        match = epoch_re.search(line)
        if match:
            epoch = int(match.group(1)) + 1
            phase = (
                "train"
                if match.group(2).lower().startswith("train")
                else "validation"
            )

        match = processed_re.search(line)
        if match and epoch is not None and phase is not None:
            result["throughput"].append({
                "epoch": epoch,
                "phase": phase,
                "entries": int(match.group(1).replace(",", "")),
                "entries_per_second": float(match.group(2)),
            })

        match = memory_re.search(line)
        if match:
            value = float(match.group(1))
            current = result["peak_cuda_memory_mb"]
            if current is None or value > current:
                result["peak_cuda_memory_mb"] = value

        if "NVIDIA" in line and len(result["gpu_lines"]) < 20:
            result["gpu_lines"].append(line)

    return result


def table_from_epoch_scalars(
    model: str,
    acc: pd.DataFrame | None,
    loss: pd.DataFrame | None,
) -> pd.DataFrame:
    frames = []

    if acc is not None:
        frame = acc[["step", "epoch", "value"]].rename(
            columns={"value": "validation_accuracy"}
        )
        frames.append(frame)

    if loss is not None:
        frame = loss[["step", "epoch", "value"]].rename(
            columns={"value": "validation_loss"}
        )
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(
            frame,
            on=["step", "epoch"],
            how="outer",
        )

    merged.insert(0, "model", model)

    if "validation_accuracy" in merged:
        merged["validation_accuracy_percent"] = (
            merged["validation_accuracy"] * 100
        )

    return merged


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    collected: dict[str, dict[str, pd.DataFrame]] = {}
    log_info: dict[str, dict] = {}
    tag_inventory: list[str] = []
    epoch_tables: list[pd.DataFrame] = []

    for run in RUNS:
        run_dir = find_event_dir(run.token)
        ea = load_event_accumulator(run_dir)
        available = ea.Tags().get("scalars", [])

        tag_inventory.append(f"## {run.label}")
        tag_inventory.append(f"Run directory: {run_dir}")
        tag_inventory.extend(f"- {tag}" for tag in available)
        tag_inventory.append("")

        collected[run.label] = {}

        for metric, candidates in TAG_CANDIDATES.items():
            tag = first_available_tag(available, candidates)
            if tag is None:
                continue

            epoch_axis = "(epoch)" in tag or metric in {
                "validation_accuracy",
                "validation_loss",
                "training_accuracy_epoch",
                "training_loss_epoch",
                "roc_auc",
            }

            collected[run.label][metric] = scalar_frame(
                ea,
                tag,
                epoch_axis=epoch_axis,
            )

        log_path = find_log(run.model_dir, run.token)
        log_info[run.label] = parse_training_log(log_path)
        log_info[run.label]["log_path"] = str(log_path) if log_path else ""

        epoch_table = table_from_epoch_scalars(
            run.label,
            collected[run.label].get("validation_accuracy"),
            collected[run.label].get("validation_loss"),
        )
        if not epoch_table.empty:
            epoch_tables.append(epoch_table)

    (OUT / "available_tensorboard_tags.txt").write_text(
        "\n".join(tag_inventory),
        encoding="utf-8",
    )

    def metric_frames(metric: str) -> dict[str, pd.DataFrame]:
        return {
            label: metrics[metric]
            for label, metrics in collected.items()
            if metric in metrics
        }

    save_line_plot(
        metric_frames("validation_accuracy"),
        x_col="epoch",
        y_scale=100.0,
        filename="validation_accuracy.png",
        title="JetClass-II validation accuracy",
        xlabel="Epoch",
        ylabel="Validation accuracy (%)",
    )

    save_line_plot(
        metric_frames("validation_loss"),
        x_col="epoch",
        y_scale=1.0,
        filename="validation_loss.png",
        title="JetClass-II validation loss",
        xlabel="Epoch",
        ylabel="Validation loss",
    )

    save_line_plot(
        metric_frames("training_accuracy_epoch"),
        x_col="epoch",
        y_scale=100.0,
        filename="training_accuracy_epoch.png",
        title="JetClass-II training accuracy",
        xlabel="Epoch",
        ylabel="Training accuracy (%)",
    )

    save_line_plot(
        metric_frames("training_loss_epoch"),
        x_col="epoch",
        y_scale=1.0,
        filename="training_loss_epoch.png",
        title="JetClass-II training loss",
        xlabel="Epoch",
        ylabel="Training loss",
    )

    save_line_plot(
        metric_frames("gradient_norm"),
        x_col="step",
        y_scale=1.0,
        filename="gradient_norm.png",
        title="Training gradient norm",
        xlabel="Training step",
        ylabel="Gradient norm",
    )

    save_line_plot(
        metric_frames("learning_rate"),
        x_col="step",
        y_scale=1.0,
        filename="learning_rate.png",
        title="Learning-rate schedule",
        xlabel="Training step",
        ylabel="Learning rate",
    )

    save_line_plot(
        metric_frames("roc_auc"),
        x_col="epoch",
        y_scale=1.0,
        filename="validation_roc_auc.png",
        title="JetClass-II validation ROC AUC",
        xlabel="Epoch",
        ylabel="ROC AUC",
    )

    save_bar_plot(
        {run.label: run.reported_flops_m for run in RUNS},
        filename="reported_flops.png",
        title="Reported model FLOPs",
        ylabel="Reported FLOPs (millions)",
    )

    save_bar_plot(
        {run.label: run.total_params_m for run in RUNS},
        filename="total_parameters.png",
        title="Total model parameters",
        ylabel="Parameters (millions)",
    )

    memory = {
        label: info["peak_cuda_memory_mb"] / 1024.0
        for label, info in log_info.items()
        if info["peak_cuda_memory_mb"] is not None
    }
    save_bar_plot(
        memory,
        filename="peak_cuda_memory.png",
        title="Peak CUDA memory",
        ylabel="Peak CUDA memory (GiB)",
    )

    throughput_rows = []
    for label, info in log_info.items():
        for row in info["throughput"]:
            throughput_rows.append({"model": label, **row})

    throughput_df = pd.DataFrame(throughput_rows)
    if not throughput_df.empty:
        throughput_df.to_csv(
            OUT / "throughput_by_epoch.csv",
            index=False,
        )

        for phase, filename, title in [
            (
                "train",
                "training_throughput.png",
                "Training throughput by epoch",
            ),
            (
                "validation",
                "validation_throughput.png",
                "Validation throughput by epoch",
            ),
        ]:
            subset = throughput_df[
                throughput_df["phase"] == phase
            ]
            frames = {
                model: group[[
                    "epoch",
                    "entries_per_second",
                ]].rename(
                    columns={"entries_per_second": "value"}
                )
                for model, group in subset.groupby("model")
            }
            save_line_plot(
                frames,
                x_col="epoch",
                y_scale=1.0,
                filename=filename,
                title=title,
                xlabel="Epoch",
                ylabel="Entries per second",
            )

    if epoch_tables:
        epoch_df = pd.concat(epoch_tables, ignore_index=True)
        epoch_df.to_csv(
            OUT / "validation_metrics_by_epoch.csv",
            index=False,
        )
    else:
        epoch_df = pd.DataFrame()

    static_rows = []
    for run in RUNS:
        info = log_info[run.label]
        static_rows.append({
            "model": run.label,
            "reported_flops_m": run.reported_flops_m,
            "total_params_m": run.total_params_m,
            "peak_cuda_memory_mb": info["peak_cuda_memory_mb"],
            "log_path": info["log_path"],
        })

    static_df = pd.DataFrame(static_rows)
    static_df.to_csv(
        OUT / "model_efficiency_summary.csv",
        index=False,
    )

    flops_reduction = (
        1.0
        - RUNS[1].reported_flops_m / RUNS[0].reported_flops_m
    ) * 100.0

    params_ratio = (
        RUNS[1].total_params_m / RUNS[0].total_params_m
    )

    summary_lines = [
        "# JetClass-II medium comparison",
        "",
        "## Static model profile",
        "",
        (
            f"- Dense ParT: {RUNS[0].reported_flops_m:.2f}M "
            f"reported FLOPs, {RUNS[0].total_params_m:.2f}M parameters."
        ),
        (
            f"- MoEParT E4 K1: {RUNS[1].reported_flops_m:.2f}M "
            f"reported FLOPs, {RUNS[1].total_params_m:.2f}M parameters."
        ),
        (
            f"- MoEParT has {flops_reduction:.1f}% fewer reported FLOPs "
            f"and {params_ratio:.2f}x as many total parameters."
        ),
        "",
        (
            "Reported FLOPs are static profiler values for the configured "
            "model input. They are not total training-job FLOPs and may not "
            "fully reflect routing, data loading, backward-pass, or kernel "
            "overhead."
        ),
        "",
        "## Generated files",
        "",
    ]

    for path in sorted(OUT.glob("*")):
        if path.name != "report_summary.md":
            summary_lines.append(f"- `{path.name}`")

    if not epoch_df.empty:
        summary_lines.extend([
            "",
            "## Validation metrics",
            "",
            epoch_df.to_markdown(index=False),
        ])

    summary_lines.extend([
        "",
        "## Hardware caveat",
        "",
        (
            "Throughput is only a controlled architecture comparison when "
            "both runs use the same GPU model and broadly comparable node "
            "conditions. Accuracy, loss, parameters, and reported FLOPs "
            "remain directly useful even when GPU models differ."
        ),
    ])

    (OUT / "report_summary.md").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print(f"Generated report in: {OUT}")
    for path in sorted(OUT.iterdir()):
        print(f"  {path.name}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
