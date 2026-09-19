#!/usr/bin/env python3

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


RUNS_ROOT = Path("/kaushik-moe-vol/outputs/runs")
OUTPUT = Path("/kaushik-moe-vol/outputs/paper_test_metrics.csv")


def find_auroc(test_dir):
    """
    Search evaluation logs for Weaver's reported multiclass AUROC.
    """
    logs = list(test_dir.glob("*.log"))

    patterns = [
        re.compile(r"AUROC[^0-9]*([0-9]*\.[0-9]+)", re.I),
        re.compile(r"ROC.?AUC[^0-9]*([0-9]*\.[0-9]+)", re.I),
    ]

    found = []

    for log in logs:
        try:
            text = log.read_text(errors="ignore")
        except Exception:
            continue

        for pattern in patterns:
            for match in pattern.finditer(text):
                found.append(float(match.group(1)))

    # Usually the final AUROC printed is the one we want.
    return found[-1] if found else np.nan


def find_logged_test_metric(test_dir):
    logs = list(test_dir.glob("*.log"))

    pattern = re.compile(
        r"Test metric[^0-9]*([0-9]*\.[0-9]+)",
        re.I,
    )

    found = []

    for log in logs:
        try:
            text = log.read_text(errors="ignore")
        except Exception:
            continue

        for match in pattern.finditer(text):
            found.append(float(match.group(1)))

    return found[-1] if found else np.nan


def compute_accuracy(pred_path):
    df = pd.read_parquet(pred_path)

    scores = np.stack(df["scores"].to_numpy())
    truth = df["truth_label"].to_numpy(dtype=int)

    prediction = np.argmax(scores, axis=1)

    return float(np.mean(prediction == truth)), len(df)


rows = []

qcd_files = sorted(
    RUNS_ROOT.rglob("metrics/test_eval/qcd_rejection.json")
)

print(f"Found {len(qcd_files)} completed QCD evaluations")

for qcd_json in qcd_files:

    test_dir = qcd_json.parent
    pred = test_dir / "pred.parquet"

    if not pred.exists():
        print(f"SKIP: no pred.parquet for {test_dir}")
        continue

    # Run directory = parent above metrics/
    run_dir = test_dir.parent.parent

    relative_run = run_dir.relative_to(RUNS_ROOT)

    with open(qcd_json) as f:
        qcd = json.load(f)

    accuracy, n_events = compute_accuracy(pred)

    op = {
        float(x["target_signal_efficiency"]): x
        for x in qcd["operating_points"]
    }

    op50 = op.get(0.5, {})
    op80 = op.get(0.8, {})

    logged_metric = find_logged_test_metric(test_dir)
    multiclass_auroc = find_auroc(test_dir)

    rows.append({
        "run": str(relative_run),

        "n_test": n_events,

        "test_accuracy": accuracy,
        "weaver_test_metric": logged_metric,

        "multiclass_auroc": multiclass_auroc,

        "signal_vs_qcd_auroc":
            qcd.get("binary_signal_vs_qcd_auc", np.nan),

        "qcd_rejection_at_50pct_signal":
            op50.get("qcd_rejection", np.nan),

        "qcd_efficiency_at_50pct_signal":
            op50.get("background_efficiency", np.nan),

        "signal_efficiency_at_50pct_target":
            op50.get("achieved_signal_efficiency", np.nan),

        "threshold_at_50pct_signal":
            op50.get("threshold", np.nan),

        "qcd_rejection_at_80pct_signal":
            op80.get("qcd_rejection", np.nan),

        "qcd_efficiency_at_80pct_signal":
            op80.get("background_efficiency", np.nan),

        "signal_efficiency_at_80pct_target":
            op80.get("achieved_signal_efficiency", np.nan),

        "threshold_at_80pct_signal":
            op80.get("threshold", np.nan),
    })


df = pd.DataFrame(rows)

if len(df) == 0:
    raise RuntimeError(
        "No completed test evaluations with qcd_rejection.json found."
    )

df = df.sort_values("run").reset_index(drop=True)

df.to_csv(OUTPUT, index=False)

print()
print("=" * 120)

display_cols = [
    "run",
    "test_accuracy",
    "multiclass_auroc",
    "signal_vs_qcd_auroc",
    "qcd_rejection_at_50pct_signal",
    "qcd_rejection_at_80pct_signal",
]

print(
    df[display_cols].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

print("=" * 120)

print()
print(f"Wrote:")
print(OUTPUT)

# Sanity check Weaver accuracy against direct calculation.
check = df.dropna(
    subset=["test_accuracy", "weaver_test_metric"]
).copy()

if len(check):
    check["difference"] = abs(
        check["test_accuracy"]
        - check["weaver_test_metric"]
    )

    max_diff = check["difference"].max()

    print()
    print(
        "Max |direct accuracy - Weaver test metric|:",
        f"{max_diff:.8g}",
    )

    if max_diff > 1e-4:
        print(
            "WARNING: Weaver test metric may not be plain "
            "top-1 classification accuracy for every run."
        )
