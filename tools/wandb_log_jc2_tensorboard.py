#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import wandb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TAG_MAP = {
    "Loss/train": "train/loss_batch",
    "Acc/train": "train/acc_batch",
    "GradNorm/train": "train/grad_norm",
    "Loss/train (epoch)": "train/loss_epoch",
    "Acc/train (epoch)": "train/acc_epoch",
    "Loss/eval (epoch)": "val/loss_epoch",
    "Acc/eval (epoch)": "val/acc_epoch",
}


def read_scalars(tb_dir: Path):
    ea = EventAccumulator(str(tb_dir))
    ea.Reload()

    out = {}
    for tb_tag, wandb_tag in TAG_MAP.items():
        if tb_tag in ea.Tags().get("scalars", []):
            out[wandb_tag] = ea.Scalars(tb_tag)

    return out


def latest_value(events):
    if not events:
        return None
    return events[-1].value


def best_value(events, mode="max"):
    if not events:
        return None
    values = [e.value for e in events]
    return max(values) if mode == "max" else min(values)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tb-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-group", default="")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--feature-type", required=True)
    parser.add_argument("--comment", required=True)

    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--start-lr", required=True)
    parser.add_argument("--num-workers", type=int, required=True)

    parser.add_argument("--train-files", type=int, required=True)
    parser.add_argument("--val-files", type=int, required=True)
    parser.add_argument("--samples-per-epoch", type=int, required=True)
    parser.add_argument("--samples-per-epoch-val", type=int, required=True)

    parser.add_argument("--moe-num-experts", default="")
    parser.add_argument("--moe-top-k", default="")
    parser.add_argument("--moe-capacity-factor", default="")
    parser.add_argument("--moe-aux-loss-coef", default="")
    parser.add_argument("--moe-router-jitter", default="")
    parser.add_argument("--weaver-args", default="")

    parser.add_argument("--run-dir", default="")
    parser.add_argument("--model-prefix", default="")
    parser.add_argument("--log-file", default="")
    parser.add_argument("--pred-out", default="")
    parser.add_argument("--status", type=int, default=0)

    args = parser.parse_args()

    tb_dir = Path(args.tb_dir)

    if not tb_dir.exists():
        print(f"[wandb logger] TensorBoard dir does not exist: {tb_dir}")
        return

    config = {
        "dataset": "JetClassII",
        "data_format": "Parquet",
        "model_name": args.model_name,
        "feature_type": args.feature_type,
        "comment": args.comment,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "start_lr": args.start_lr,
        "num_workers": args.num_workers,
        "train_files": args.train_files,
        "val_files": args.val_files,
        "samples_per_epoch": args.samples_per_epoch,
        "samples_per_epoch_val": args.samples_per_epoch_val,
        "moe_num_experts": args.moe_num_experts,
        "moe_top_k": args.moe_top_k,
        "moe_capacity_factor": args.moe_capacity_factor,
        "moe_aux_loss_coef": args.moe_aux_loss_coef,
        "moe_router_jitter": args.moe_router_jitter,
        "weaver_args": args.weaver_args,
        "model_prefix": args.model_prefix,
        "log_file": args.log_file,
        "pred_out": args.pred_out,
        "exit_status": args.status,
    }

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "MoEParT-JetClassII"),
        entity=os.environ.get("WANDB_ENTITY", "kpendiyala-uc-san-diego"),
        name=os.environ.get("WANDB_RUN_NAME", args.run_name),
        group=os.environ.get("WANDB_RUN_GROUP", args.run_group),
        dir=os.environ.get("WANDB_DIR", "/kaushik-moe-vol/outputs/wandb"),
        config=config,
        job_type="train",
    )

    wandb.define_metric("epoch")
    wandb.define_metric("global_step")

    wandb.define_metric("train/loss_epoch", step_metric="epoch")
    wandb.define_metric("train/acc_epoch", step_metric="epoch")
    wandb.define_metric("val/loss_epoch", step_metric="epoch")
    wandb.define_metric("val/acc_epoch", step_metric="epoch")

    wandb.define_metric("train/loss_batch", step_metric="global_step")
    wandb.define_metric("train/acc_batch", step_metric="global_step")
    wandb.define_metric("train/grad_norm", step_metric="global_step")

    scalar_data = read_scalars(tb_dir)

    # Batch-level metrics use real TensorBoard global_step.
    for metric in ["train/loss_batch", "train/acc_batch", "train/grad_norm"]:
        for event in scalar_data.get(metric, []):
            wandb.log({
                "global_step": event.step,
                metric: event.value,
            })

    # Epoch-level metrics get a clean epoch x-axis.
    epoch_metrics = [
        "train/loss_epoch",
        "train/acc_epoch",
        "val/loss_epoch",
        "val/acc_epoch",
    ]

    max_epochs = max(
        [len(scalar_data.get(metric, [])) for metric in epoch_metrics] or [0]
    )

    for i in range(max_epochs):
        row = {"epoch": i + 1}
        for metric in epoch_metrics:
            events = scalar_data.get(metric, [])
            if i < len(events):
                row[metric] = events[i].value
        wandb.log(row)

    # Summary values for W&B table comparison.
    if "val/acc_epoch" in scalar_data:
        wandb.summary["best_val_acc"] = best_value(scalar_data["val/acc_epoch"], "max")
        wandb.summary["final_val_acc"] = latest_value(scalar_data["val/acc_epoch"])

    if "val/loss_epoch" in scalar_data:
        wandb.summary["best_val_loss"] = best_value(scalar_data["val/loss_epoch"], "min")
        wandb.summary["final_val_loss"] = latest_value(scalar_data["val/loss_epoch"])

    if "train/acc_epoch" in scalar_data:
        wandb.summary["final_train_acc"] = latest_value(scalar_data["train/acc_epoch"])

    if "train/loss_epoch" in scalar_data:
        wandb.summary["final_train_loss"] = latest_value(scalar_data["train/loss_epoch"])

    # Save useful local files as artifacts.
    files_artifact = wandb.Artifact(
        name=f"{os.environ.get('WANDB_RUN_NAME', args.run_name)}-metadata",
        type="metadata",
    )

    for path_str in [args.log_file, args.pred_out]:
        if path_str and Path(path_str).exists():
            files_artifact.add_file(path_str)

    if args.run_dir and Path(args.run_dir).exists():
        for subdir in ["metadata", "profiling", "routing", "metrics", "config"]:
            p = Path(args.run_dir) / subdir
            if p.exists():
                files_artifact.add_dir(str(p), name=subdir)

    try:
        run.log_artifact(files_artifact)
    except Exception as e:
        print(f"[wandb logger] Artifact upload skipped: {e}")

    wandb.finish()


if __name__ == "__main__":
    main()
