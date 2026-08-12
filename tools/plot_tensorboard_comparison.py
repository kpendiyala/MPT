#!/usr/bin/env python3
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path("/kaushik-moe-vol/outputs/tensorboard")
OUT = Path("/kaushik-moe-vol/outputs/figures/medium-comparison")

RUNS = {
    "Dense ParT": "jc2-medium-part-e3-s1m-v600k",
    "MoEParT E4 K1": "jc2-medium-mpt-e4-k1-e3-s1m-v600k",
}

TAGS = {
    "accuracy": ["Acc/eval (epoch)", "Acc/eval"],
    "loss": ["Loss/eval (epoch)", "Loss/eval"],
}


def find_run(token):
    files = [
        p for p in ROOT.rglob("events.out.tfevents.*")
        if token in str(p)
    ]
    if not files:
        raise FileNotFoundError(f"No TensorBoard event file found for {token}")
    return max(files, key=lambda p: p.stat().st_mtime).parent


def read_scalar(run_dir, candidates):
    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    available = ea.Tags().get("scalars", [])

    for tag in candidates:
        if tag in available:
            events = ea.Scalars(tag)
            df = pd.DataFrame({
                "epoch_index": [e.step for e in events],
                "value": [e.value for e in events],
            })
            df["epoch"] = df["epoch_index"] + 1
            return tag, df

    raise KeyError(
        f"Could not find {candidates}. Available scalar tags: {available}"
    )


def plot_metric(data, filename, title, ylabel, scale=1.0):
    plt.figure(figsize=(7.5, 4.8))

    for model, df in data.items():
        plt.plot(
            df["epoch"],
            df["value"] * scale,
            marker="o",
            linewidth=2,
            label=model,
        )

    epochs = sorted({
        int(x)
        for df in data.values()
        for x in df["epoch"].tolist()
    })

    plt.xticks(epochs)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=300)
    plt.close()


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    acc = {}
    loss = {}
    rows = []

    for model, token in RUNS.items():
        run_dir = find_run(token)
        acc_tag, acc_df = read_scalar(run_dir, TAGS["accuracy"])
        loss_tag, loss_df = read_scalar(run_dir, TAGS["loss"])

        print(f"{model}: {run_dir}")
        print(f"  accuracy tag: {acc_tag}")
        print(f"  loss tag:     {loss_tag}")

        acc[model] = acc_df
        loss[model] = loss_df

        merged = acc_df.rename(columns={"value": "validation_accuracy"})
        merged = merged.merge(
            loss_df[["epoch_index", "value"]].rename(
                columns={"value": "validation_loss"}
            ),
            on="epoch_index",
            how="outer",
        )
        merged.insert(0, "model", model)
        merged["validation_accuracy_percent"] = (
            merged["validation_accuracy"] * 100
        )
        rows.append(merged)

    plot_metric(
        acc,
        "validation_accuracy.png",
        "JetClass-II validation accuracy",
        "Validation accuracy (%)",
        scale=100.0,
    )
    plot_metric(
        loss,
        "validation_loss.png",
        "JetClass-II validation loss",
        "Validation loss",
    )

    table = pd.concat(rows, ignore_index=True)
    table.to_csv(OUT / "medium_comparison_metrics.csv", index=False)

    print()
    print(table.to_string(index=False))
    print(f"\nSaved files to {OUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
