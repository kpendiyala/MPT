#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare multiple ParT / MPT experiment runs using "
            "profile.json, summary.json, and epoch_metrics.csv."
        )
    )

    parser.add_argument(
        "--runs-dir",
        default="/kaushik-moe-vol/outputs/runs",
        help="Base directory containing per-run folders.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for summary CSV/JSON and plots.",
    )

    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run directory names to compare.",
    )

    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_epoch_csv(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            parsed = {}

            for key, value in row.items():
                if value is None or value == "":
                    parsed[key] = None
                    continue

                if key == "epoch":
                    parsed[key] = int(value)
                    continue

                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value

            rows.append(parsed)

    return rows


def infer_model_label(run_name, profile):
    model = profile.get("model")

    if model == "ParT":
        return "Dense ParT"

    experts = profile.get("num_experts")
    top_k = profile.get("top_k")

    if experts is not None and top_k is not None:
        return f"MPT E{experts} K{top_k}"

    return run_name


def load_run(run_dir):
    profile_path = run_dir / "profiling" / "profile.json"
    summary_path = run_dir / "metrics" / "summary.json"
    epoch_path = run_dir / "metrics" / "epoch_metrics.csv"

    missing = []

    for path in [
        profile_path,
        summary_path,
        epoch_path,
    ]:
        if not path.exists():
            missing.append(str(path))

    if missing:
        raise FileNotFoundError(
            "Missing required files:\n  "
            + "\n  ".join(missing)
        )

    profile = load_json(profile_path)
    summary = load_json(summary_path)
    epochs = load_epoch_csv(epoch_path)

    run_name = run_dir.name
    label = infer_model_label(run_name, profile)

    return {
        "run_name": run_name,
        "label": label,
        "profile": profile,
        "summary": summary,
        "epochs": epochs,
    }


def build_master_row(run):
    profile = run["profile"]
    summary = run["summary"]

    model = profile.get("model")

    return {
        "run_name": run["run_name"],
        "label": run["label"],

        "model": model,

        "num_experts": (
            profile.get("num_experts")
            if model == "MPT"
            else None
        ),

        "top_k": (
            profile.get("top_k")
            if model == "MPT"
            else None
        ),

        "training_capacity_factor": (
            profile.get("training_capacity_factor")
            if model == "MPT"
            else None
        ),

        "training_aux_loss_coef": (
            profile.get("training_aux_loss_coef")
            if model == "MPT"
            else None
        ),

        "training_router_jitter": (
            profile.get("training_router_jitter")
            if model == "MPT"
            else None
        ),

        "total_params": profile.get("total_params"),
        "trainable_params": profile.get("trainable_params"),

        "forward_flops": profile.get("forward_flops"),
        "forward_gflops": profile.get("forward_gflops"),

        "best_epoch_by_val_accuracy": (
            summary.get("best_epoch_by_val_accuracy")
        ),

        "best_val_accuracy": (
            summary.get("best_val_accuracy")
        ),

        "val_loss_at_best_accuracy": (
            summary.get("val_loss_at_best_accuracy")
        ),

        "best_epoch_by_val_loss": (
            summary.get("best_epoch_by_val_loss")
        ),

        "best_val_loss": (
            summary.get("best_val_loss")
        ),

        "val_accuracy_at_best_loss": (
            summary.get("val_accuracy_at_best_loss")
        ),

        "final_epoch": summary.get("final_epoch"),
        "final_val_accuracy": (
            summary.get("final_val_accuracy")
        ),
        "final_val_loss": (
            summary.get("final_val_loss")
        ),

        "peak_cuda_memory_mb": (
            summary.get("peak_cuda_memory_mb")
        ),

        "average_train_entries_per_sec": (
            summary.get("average_train_entries_per_sec")
        ),

        "average_val_entries_per_sec": (
            summary.get("average_val_entries_per_sec")
        ),

        "gpu": profile.get("gpu"),
        "torch_version": profile.get("torch_version"),
        "git_commit": profile.get("git_commit"),
    }


def write_master_csv(rows, path):
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
        )


def plot_validation_accuracy(runs, output_path):
    plt.figure(figsize=(8, 5))

    for run in runs:
        epochs = []
        values = []

        for row in run["epochs"]:
            if row.get("val_accuracy") is None:
                continue

            epochs.append(row["epoch"])
            values.append(row["val_accuracy"] * 100.0)

        if epochs:
            plt.plot(
                epochs,
                values,
                marker="o",
                label=run["label"],
            )

    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy (%)")
    plt.title("Validation Accuracy vs Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_validation_loss(runs, output_path):
    plt.figure(figsize=(8, 5))

    for run in runs:
        epochs = []
        values = []

        for row in run["epochs"]:
            if row.get("val_loss") is None:
                continue

            epochs.append(row["epoch"])
            values.append(row["val_loss"])

        if epochs:
            plt.plot(
                epochs,
                values,
                marker="o",
                label=run["label"],
            )

    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.title("Validation Loss vs Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_accuracy_vs_experts(runs, output_path):
    points = []

    for run in runs:
        profile = run["profile"]
        summary = run["summary"]

        if profile.get("model") != "MPT":
            continue

        experts = profile.get("num_experts")
        accuracy = summary.get("best_val_accuracy")

        if experts is None or accuracy is None:
            continue

        points.append(
            (
                int(experts),
                accuracy * 100.0,
                run["label"],
            )
        )

    if not points:
        return

    points.sort(key=lambda x: x[0])

    xs = [x[0] for x in points]
    ys = [x[1] for x in points]

    plt.figure(figsize=(7, 5))

    plt.plot(
        xs,
        ys,
        marker="o",
    )

    for x, y, label in points:
        plt.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
        )

    plt.xlabel("Number of experts")
    plt.ylabel("Best validation accuracy (%)")
    plt.title("Best Validation Accuracy vs Number of Experts")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_accuracy_vs_parameters(runs, output_path):
    points = []

    for run in runs:
        params = run["profile"].get("total_params")
        accuracy = run["summary"].get(
            "best_val_accuracy"
        )

        if params is None or accuracy is None:
            continue

        points.append(
            (
                params / 1e6,
                accuracy * 100.0,
                run["label"],
            )
        )

    if not points:
        return

    plt.figure(figsize=(7, 5))

    for x, y, label in points:
        plt.scatter(x, y)

        plt.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
        )

    plt.xlabel("Total parameters (millions)")
    plt.ylabel("Best validation accuracy (%)")
    plt.title("Best Validation Accuracy vs Parameter Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_accuracy_vs_flops(runs, output_path):
    points = []

    for run in runs:
        gflops = run["profile"].get(
            "forward_gflops"
        )

        accuracy = run["summary"].get(
            "best_val_accuracy"
        )

        if gflops is None or accuracy is None:
            continue

        points.append(
            (
                gflops,
                accuracy * 100.0,
                run["label"],
            )
        )

    if not points:
        return

    plt.figure(figsize=(7, 5))

    for x, y, label in points:
        plt.scatter(x, y)

        plt.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
        )

    plt.xlabel("Forward GFLOPs / jet @ 128 particles")
    plt.ylabel("Best validation accuracy (%)")
    plt.title("Best Validation Accuracy vs Forward FLOPs")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def print_ranking(master_rows):
    print()
    print("=" * 90)
    print("EXPERIMENT COMPARISON")
    print("=" * 90)

    ranked = [
        row
        for row in master_rows
        if row["best_val_accuracy"] is not None
    ]

    ranked.sort(
        key=lambda x: x["best_val_accuracy"],
        reverse=True,
    )

    print()
    print("Ranking by best validation accuracy:")
    print()

    for i, row in enumerate(ranked, start=1):
        acc = row["best_val_accuracy"] * 100.0

        loss = row["val_loss_at_best_accuracy"]
        gflops = row["forward_gflops"]
        params = row["total_params"]

        print(
            f"{i:2d}. "
            f"{row['label']:15s} "
            f"acc={acc:7.3f}% "
            f"loss={loss:.5f} "
            f"params={params / 1e6:.3f}M "
            f"GFLOPs={gflops:.6f}"
        )

    print()

    if ranked:
        winner = ranked[0]

        print(
            "Best validation accuracy:",
            winner["label"],
            f"({winner['best_val_accuracy'] * 100:.3f}%)",
        )

    lowest_loss_rows = [
        row
        for row in master_rows
        if row["best_val_loss"] is not None
    ]

    if lowest_loss_rows:
        best_loss = min(
            lowest_loss_rows,
            key=lambda x: x["best_val_loss"],
        )

        print(
            "Lowest validation loss:",
            best_loss["label"],
            f"({best_loss['best_val_loss']:.5f})",
        )

    lowest_memory_rows = [
        row
        for row in master_rows
        if row["peak_cuda_memory_mb"] is not None
    ]

    if lowest_memory_rows:
        best_memory = min(
            lowest_memory_rows,
            key=lambda x: x["peak_cuda_memory_mb"],
        )

        print(
            "Lowest peak CUDA memory:",
            best_memory["label"],
            f"({best_memory['peak_cuda_memory_mb']:.1f} MB)",
        )

    print()


def main():
    args = parse_args()

    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    runs = []

    for run_name in args.runs:
        run_dir = runs_dir / run_name

        print(
            f"Loading run: {run_dir}"
        )

        runs.append(
            load_run(run_dir)
        )

    master_rows = [
        build_master_row(run)
        for run in runs
    ]

    csv_path = (
        output_dir
        / "experiment_summary.csv"
    )

    json_path = (
        output_dir
        / "experiment_summary.json"
    )

    write_master_csv(
        master_rows,
        csv_path,
    )

    save_json(
        master_rows,
        json_path,
    )

    plot_validation_accuracy(
        runs,
        output_dir
        / "validation_accuracy_vs_epoch.png",
    )

    plot_validation_loss(
        runs,
        output_dir
        / "validation_loss_vs_epoch.png",
    )

    plot_accuracy_vs_experts(
        runs,
        output_dir
        / "accuracy_vs_experts.png",
    )

    plot_accuracy_vs_parameters(
        runs,
        output_dir
        / "accuracy_vs_parameters.png",
    )

    plot_accuracy_vs_flops(
        runs,
        output_dir
        / "accuracy_vs_flops.png",
    )

    print_ranking(
        master_rows
    )

    print("=" * 90)
    print("OUTPUTS")
    print("=" * 90)

    print("Summary CSV:")
    print(f"  {csv_path}")

    print("Summary JSON:")
    print(f"  {json_path}")

    print("Plots:")
    print(
        f"  {output_dir / 'validation_accuracy_vs_epoch.png'}"
    )
    print(
        f"  {output_dir / 'validation_loss_vs_epoch.png'}"
    )
    print(
        f"  {output_dir / 'accuracy_vs_experts.png'}"
    )
    print(
        f"  {output_dir / 'accuracy_vs_parameters.png'}"
    )
    print(
        f"  {output_dir / 'accuracy_vs_flops.png'}"
    )


if __name__ == "__main__":
    main()
