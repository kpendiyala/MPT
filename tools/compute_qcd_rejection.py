#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def load_scores(path):
    df = pd.read_parquet(path)

    if "scores" not in df.columns:
        raise RuntimeError("pred.parquet does not contain a 'scores' column")

    if "truth_label" not in df.columns:
        raise RuntimeError("pred.parquet does not contain 'truth_label'")

    scores = np.stack(df["scores"].to_numpy()).astype(np.float64)
    labels = df["truth_label"].to_numpy(dtype=np.int64)

    if scores.ndim != 2 or scores.shape[1] != 188:
        raise RuntimeError(
            f"Expected scores shape (N, 188), got {scores.shape}"
        )

    return df, scores, labels


def maybe_convert_logits_to_probs(scores):
    row_sums = scores.sum(axis=1)

    looks_like_probs = (
        np.all(scores >= -1e-6)
        and np.all(scores <= 1 + 1e-6)
        and np.allclose(row_sums, 1.0, atol=1e-3)
    )

    if looks_like_probs:
        print("scores appear to already be probabilities")
        return scores, False

    print("scores do not appear normalized; applying softmax")

    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)

    return probs, True


def rejection_at_signal_efficiency(
    y_true,
    signal_score,
    target_eff,
):
    fpr, tpr, thresholds = roc_curve(
        y_true,
        signal_score,
        pos_label=1,
    )

    # Pick the ROC point closest to the requested signal efficiency.
    idx = np.argmin(np.abs(tpr - target_eff))

    eps_s = float(tpr[idx])
    eps_b = float(fpr[idx])
    threshold = float(thresholds[idx])

    rejection = (
        float("inf")
        if eps_b == 0
        else 1.0 / eps_b
    )

    return {
        "target_signal_efficiency": float(target_eff),
        "achieved_signal_efficiency": eps_s,
        "background_efficiency": eps_b,
        "qcd_rejection": rejection,
        "threshold": threshold,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to Weaver pred.parquet",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
    )

    parser.add_argument(
        "--signal-efficiencies",
        nargs="+",
        type=float,
        default=[0.5, 0.8],
    )

    args = parser.parse_args()

    pred_path = Path(args.predictions)

    outdir = (
        Path(args.output_dir)
        if args.output_dir
        else pred_path.parent
    )

    outdir.mkdir(parents=True, exist_ok=True)

    df, scores, labels = load_scores(pred_path)
    probs, applied_softmax = maybe_convert_logits_to_probs(scores)

    # -------------------------------
    # Binary signal-vs-QCD problem
    # -------------------------------

    is_signal = labels <= 160
    is_qcd = labels >= 161

    valid = is_signal | is_qcd

    labels = labels[valid]
    probs = probs[valid]

    y_true = (labels <= 160).astype(np.int64)

    signal_score = probs[:, :161].sum(axis=1)
    qcd_score = probs[:, 161:].sum(axis=1)

    print()
    print("Events:")
    print("  total:", len(y_true))
    print("  signal:", int(y_true.sum()))
    print("  QCD:", int((1 - y_true).sum()))

    print()
    print("Probability check:")
    print(
        "  mean(signal_score + qcd_score):",
        np.mean(signal_score + qcd_score),
    )

    binary_auc = roc_auc_score(
        y_true,
        signal_score,
    )

    print()
    print(
        f"Signal-vs-QCD AUROC: {binary_auc:.8f}"
    )

    results = []

    for target_eff in args.signal_efficiencies:
        result = rejection_at_signal_efficiency(
            y_true,
            signal_score,
            target_eff,
        )

        results.append(result)

        print()
        print(
            f"Target signal efficiency: "
            f"{target_eff:.3f}"
        )
        print(
            f"  achieved eps_S: "
            f"{result['achieved_signal_efficiency']:.6f}"
        )
        print(
            f"  eps_B: "
            f"{result['background_efficiency']:.8f}"
        )
        print(
            f"  QCD rejection: "
            f"{result['qcd_rejection']:.3f}"
        )
        print(
            f"  threshold: "
            f"{result['threshold']:.8f}"
        )

    # Save ROC itself.
    fpr, tpr, thresholds = roc_curve(
        y_true,
        signal_score,
        pos_label=1,
    )

    pd.DataFrame({
        "signal_efficiency": tpr,
        "background_efficiency": fpr,
        "qcd_rejection": np.divide(
            1.0,
            fpr,
            out=np.full_like(
                fpr,
                np.inf,
                dtype=float,
            ),
            where=fpr > 0,
        ),
        "threshold": thresholds,
    }).to_csv(
        outdir / "qcd_roc.csv",
        index=False,
    )

    summary = {
        "predictions": str(pred_path),
        "num_events": int(len(y_true)),
        "num_signal": int(y_true.sum()),
        "num_qcd": int((1 - y_true).sum()),
        "signal_label_range": [0, 160],
        "qcd_label_range": [161, 187],
        "score_definition": "sum(scores[0:161])",
        "softmax_applied": bool(applied_softmax),
        "binary_signal_vs_qcd_auc": float(binary_auc),
        "operating_points": results,
    }

    with open(
        outdir / "qcd_rejection.json",
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    pd.DataFrame(results).to_csv(
        outdir / "qcd_rejection.csv",
        index=False,
    )

    print()
    print("Wrote:")
    print(outdir / "qcd_rejection.json")
    print(outdir / "qcd_rejection.csv")
    print(outdir / "qcd_roc.csv")


if __name__ == "__main__":
    main()
