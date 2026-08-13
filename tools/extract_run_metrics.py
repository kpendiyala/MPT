#!/usr/bin/env python3

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean


EPOCH_TRAIN_RE = re.compile(
    r"Epoch #(?P<epoch>\d+) training"
)

EPOCH_VAL_RE = re.compile(
    r"Epoch #(?P<epoch>\d+) validating"
)

TRAIN_METRICS_RE = re.compile(
    r"Train AvgLoss:\s*(?P<loss>[0-9.eE+-]+),\s*"
    r"AvgAcc:\s*(?P<acc>[0-9.eE+-]+)"
)

VAL_METRICS_RE = re.compile(
    r"Eval AvgLoss:\s*(?P<loss>[0-9.eE+-]+),\s*"
    r"AvgAcc:\s*(?P<acc>[0-9.eE+-]+)"
)

PROCESSED_RE = re.compile(
    r"Processed\s+(?P<entries>\d+)\s+entries in total "
    r"\(avg\. speed\s+(?P<speed>[0-9.eE+-]+)\s+entries/s\)"
)

CUDA_MEMORY_RE = re.compile(
    r"Max CUDA memory:\s*(?P<memory>[0-9.eE+-]+)\s*MB"
)

VALIDATION_METRIC_RE = re.compile(
    r"Epoch #(?P<epoch>\d+): Current validation metric:\s*"
    r"(?P<current>[0-9.eE+-]+)\s*"
    r"\(best:\s*(?P<best>[0-9.eE+-]+)\)"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract per-epoch training and validation metrics "
            "from a Weaver training log."
        )
    )

    parser.add_argument(
        "--log",
        required=True,
        help="Path to Weaver .log file",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where epoch_metrics.csv and summary.json are written",
    )

    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name to store in summary metadata",
    )

    return parser.parse_args()


def new_epoch_record(epoch):
    return {
        "epoch": epoch,

        "train_loss": None,
        "train_accuracy": None,

        "val_loss": None,
        "val_accuracy": None,

        "train_entries": None,
        "train_entries_per_sec": None,

        "val_entries": None,
        "val_entries_per_sec": None,

        "max_cuda_memory_mb": None,

        "current_validation_metric": None,
        "best_validation_metric_so_far": None,
    }


def get_epoch_record(records, epoch):
    if epoch not in records:
        records[epoch] = new_epoch_record(epoch)
    return records[epoch]


def parse_log(log_path):
    records = {}

    current_epoch = None
    current_phase = None

    with open(
        log_path,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:

        for line in f:

            m = EPOCH_TRAIN_RE.search(line)
            if m:
                current_epoch = int(m.group("epoch"))
                current_phase = "train"
                get_epoch_record(records, current_epoch)
                continue

            m = EPOCH_VAL_RE.search(line)
            if m:
                current_epoch = int(m.group("epoch"))
                current_phase = "val"
                get_epoch_record(records, current_epoch)
                continue

            m = TRAIN_METRICS_RE.search(line)
            if m and current_epoch is not None:
                rec = get_epoch_record(records, current_epoch)

                rec["train_loss"] = float(
                    m.group("loss")
                )

                rec["train_accuracy"] = float(
                    m.group("acc")
                )

                continue

            m = VAL_METRICS_RE.search(line)
            if m and current_epoch is not None:
                rec = get_epoch_record(records, current_epoch)

                rec["val_loss"] = float(
                    m.group("loss")
                )

                rec["val_accuracy"] = float(
                    m.group("acc")
                )

                continue

            m = PROCESSED_RE.search(line)
            if m and current_epoch is not None:

                entries = int(
                    m.group("entries")
                )

                speed = float(
                    m.group("speed")
                )

                rec = get_epoch_record(
                    records,
                    current_epoch,
                )

                if current_phase == "train":
                    rec["train_entries"] = entries
                    rec["train_entries_per_sec"] = speed

                elif current_phase == "val":
                    rec["val_entries"] = entries
                    rec["val_entries_per_sec"] = speed

                continue

            m = CUDA_MEMORY_RE.search(line)
            if m and current_epoch is not None:

                rec = get_epoch_record(
                    records,
                    current_epoch,
                )

                rec["max_cuda_memory_mb"] = float(
                    m.group("memory")
                )

                continue

            m = VALIDATION_METRIC_RE.search(line)
            if m:

                epoch = int(
                    m.group("epoch")
                )

                rec = get_epoch_record(
                    records,
                    epoch,
                )

                rec["current_validation_metric"] = float(
                    m.group("current")
                )

                rec["best_validation_metric_so_far"] = float(
                    m.group("best")
                )

                continue

    return [
        records[k]
        for k in sorted(records)
    ]


def safe_mean(values):
    values = [
        x for x in values
        if x is not None
    ]

    if not values:
        return None

    return mean(values)


def build_summary(
    rows,
    run_name,
    log_path,
):
    valid_rows = [
        row
        for row in rows
        if row["val_accuracy"] is not None
    ]

    if valid_rows:
        best_acc_row = max(
            valid_rows,
            key=lambda row: row["val_accuracy"],
        )

        final_val_row = valid_rows[-1]

    else:
        best_acc_row = None
        final_val_row = None

    valid_loss_rows = [
        row
        for row in rows
        if row["val_loss"] is not None
    ]

    if valid_loss_rows:
        best_loss_row = min(
            valid_loss_rows,
            key=lambda row: row["val_loss"],
        )
    else:
        best_loss_row = None

    memory_values = [
        row["max_cuda_memory_mb"]
        for row in rows
        if row["max_cuda_memory_mb"] is not None
    ]

    peak_cuda_memory_mb = (
        max(memory_values)
        if memory_values
        else None
    )

    train_speeds = [
        row["train_entries_per_sec"]
        for row in rows
        if row["train_entries_per_sec"] is not None
    ]

    val_speeds = [
        row["val_entries_per_sec"]
        for row in rows
        if row["val_entries_per_sec"] is not None
    ]

    summary = {
        "run_name": run_name,
        "source_log": str(log_path),

        "num_epochs_found": len(rows),

        "best_epoch_by_val_accuracy": (
            best_acc_row["epoch"]
            if best_acc_row
            else None
        ),

        "best_val_accuracy": (
            best_acc_row["val_accuracy"]
            if best_acc_row
            else None
        ),

        "val_loss_at_best_accuracy": (
            best_acc_row["val_loss"]
            if best_acc_row
            else None
        ),

        "best_epoch_by_val_loss": (
            best_loss_row["epoch"]
            if best_loss_row
            else None
        ),

        "best_val_loss": (
            best_loss_row["val_loss"]
            if best_loss_row
            else None
        ),

        "val_accuracy_at_best_loss": (
            best_loss_row["val_accuracy"]
            if best_loss_row
            else None
        ),

        "final_epoch": (
            final_val_row["epoch"]
            if final_val_row
            else None
        ),

        "final_val_accuracy": (
            final_val_row["val_accuracy"]
            if final_val_row
            else None
        ),

        "final_val_loss": (
            final_val_row["val_loss"]
            if final_val_row
            else None
        ),

        "peak_cuda_memory_mb": peak_cuda_memory_mb,

        "average_train_entries_per_sec": safe_mean(
            train_speeds
        ),

        "average_val_entries_per_sec": safe_mean(
            val_speeds
        ),

        "train_speed_epochs_found": len(
            train_speeds
        ),

        "val_speed_epochs_found": len(
            val_speeds
        ),
    }

    return summary


def write_csv(rows, output_path):
    fieldnames = [
        "epoch",

        "train_loss",
        "train_accuracy",

        "val_loss",
        "val_accuracy",

        "train_entries",
        "train_entries_per_sec",

        "val_entries",
        "val_entries_per_sec",

        "max_cuda_memory_mb",

        "current_validation_metric",
        "best_validation_metric_so_far",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()

    log_path = Path(
        args.log
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    if not log_path.exists():
        raise FileNotFoundError(
            f"Log file does not exist: {log_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = parse_log(
        log_path
    )

    if not rows:
        raise RuntimeError(
            "No epoch information was found in the log."
        )

    run_name = (
        args.run_name
        if args.run_name
        else log_path.stem
    )

    csv_path = (
        output_dir
        / "epoch_metrics.csv"
    )

    summary_path = (
        output_dir
        / "summary.json"
    )

    write_csv(
        rows,
        csv_path,
    )

    summary = build_summary(
        rows,
        run_name=run_name,
        log_path=log_path,
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 80)
    print("RUN METRIC EXTRACTION COMPLETE")
    print("=" * 80)

    print(
        f"Run: {run_name}"
    )

    print(
        f"Epochs found: {len(rows)}"
    )

    print(
        f"CSV: {csv_path}"
    )

    print(
        f"Summary: {summary_path}"
    )

    print()

    print(
        "Best validation accuracy:",
        summary["best_val_accuracy"],
    )

    print(
        "Best validation accuracy epoch:",
        summary["best_epoch_by_val_accuracy"],
    )

    print(
        "Best validation loss:",
        summary["best_val_loss"],
    )

    print(
        "Peak CUDA memory (MB):",
        summary["peak_cuda_memory_mb"],
    )

    print(
        "Average training speed (entries/s):",
        summary["average_train_entries_per_sec"],
    )

    print(
        "Average validation speed (entries/s):",
        summary["average_val_entries_per_sec"],
    )


if __name__ == "__main__":
    main()
