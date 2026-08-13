import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch

import model.ParT as ParT
import model.MPT as MPT


def make_fake_data_config(seq_len):
    """
    Minimal JetClass-II-like data config needed by
    ParT.get_model() / MPT.get_model().

    JetClass-II:
      pf_features: 17 channels
      pf_vectors:   4 channels
      pf_mask:      1 channel
      sequence length: normally 128
    """
    return SimpleNamespace(
        input_dicts={
            "pf_features": [None] * 17,
        },
        label_value=list(range(188)),
        input_names=[
            "pf_points",
            "pf_features",
            "pf_vectors",
            "pf_mask",
        ],
        input_shapes={
            "pf_points": (1, 2, seq_len),
            "pf_features": (1, 17, seq_len),
            "pf_vectors": (1, 4, seq_len),
            "pf_mask": (1, 1, seq_len),
        },
    )


def make_inputs(batch_size, seq_len, device):
    """
    Create deterministic synthetic inputs.

    All particles are marked valid so the standardized FLOPs
    measurement always uses the full requested sequence length.
    """
    torch.manual_seed(12345)

    points = torch.randn(
        batch_size,
        2,
        seq_len,
        device=device,
        dtype=torch.float32,
    )

    features = torch.randn(
        batch_size,
        17,
        seq_len,
        device=device,
        dtype=torch.float32,
    )

    lorentz_vectors = torch.randn(
        batch_size,
        4,
        seq_len,
        device=device,
        dtype=torch.float32,
    )

    mask = torch.ones(
        batch_size,
        1,
        seq_len,
        device=device,
        dtype=torch.float32,
    )

    return points, features, lorentz_vectors, mask


def count_params(model):
    total = sum(p.numel() for p in model.parameters())

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


def disable_sdpa_for_flops(model):
    """
    Force Weaver attention modules to use their explicit matrix
    implementation instead of fused scaled_dot_product_attention.

    torch.profiler(with_flops=True) does not reliably attribute
    FLOPs to the fused SDPA operator.

    THIS IS ONLY FOR FLOPs PROFILING.
    Normal training should continue using SDPA.
    """
    changed = 0

    for module in model.modules():
        if hasattr(module, "use_sdpa"):
            module.use_sdpa = False
            changed += 1

    return changed


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def build_dense(data_config):
    model, _ = ParT.get_model(
        data_config,
        num_classes=188,
        fc_params=[(512, 0.1)],
    )

    return model


def build_mpt(
    data_config,
    num_experts,
    top_k,
    profile_capacity_factor,
):
    """
    Build the MoE model used for standardized FLOPs measurement.

    profile_capacity_factor is intentionally independent from the
    training capacity factor. A very large value prevents routing
    overflow/token dropping from changing the architectural FLOPs
    measurement.
    """
    model, _ = MPT.get_model(
        data_config,
        num_classes=188,
        fc_params=[(512, 0.1)],
        moe_num_experts=num_experts,
        moe_top_k=top_k,
        moe_capacity_factor=profile_capacity_factor,

        # These do not need to affect a deterministic inference
        # profiling pass.
        moe_aux_loss_coef=0.01,
        moe_router_jitter=0.0,
    )

    return model


def profile_model(
    model,
    inputs,
    device,
    output_dir,
):
    model = model.to(device)
    model.eval()

    total_params, trainable_params = count_params(model)

    # IMPORTANT:
    # Disable fused SDPA only in this profiler.
    num_sdpa_disabled = disable_sdpa_for_flops(model)

    print(
        f"Disabled SDPA in {num_sdpa_disabled} "
        "attention modules for FLOPs profiling."
    )

    # Warm-up outside the profiler.
    with torch.no_grad():
        for _ in range(2):
            _ = model(*inputs)

    if device.type == "cuda":
        torch.cuda.synchronize()

    activities = [
        torch.profiler.ProfilerActivity.CPU,
    ]

    if device.type == "cuda":
        activities.append(
            torch.profiler.ProfilerActivity.CUDA
        )

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=False,
        with_flops=True,
    ) as prof:
        with torch.no_grad():
            _ = model(*inputs)

        if device.type == "cuda":
            torch.cuda.synchronize()

    events = prof.key_averages()

    total_flops = sum(
        event.flops or 0
        for event in events
    )

    table = events.table(
        sort_by="flops",
        row_limit=100,
    )

    ops_path = output_dir / "profile_ops.txt"

    with open(ops_path, "w") as f:
        f.write(table)
        f.write("\n\n")
        f.write(f"Total FLOPs: {total_flops}\n")
        f.write(f"GFLOPs: {total_flops / 1e9:.6f}\n")
        f.write(f"Total parameters: {total_params}\n")
        f.write(
            f"Trainable parameters: {trainable_params}\n"
        )
        f.write(
            f"SDPA modules disabled: {num_sdpa_disabled}\n"
        )

    print()
    print("=" * 90)
    print("PROFILE RESULT")
    print("=" * 90)

    print(f"Total params:       {total_params:,}")
    print(f"Trainable params:   {trainable_params:,}")
    print(f"Profiler FLOPs:     {total_flops:,}")
    print(
        f"Profiler GFLOPs:    "
        f"{total_flops / 1e9:.6f}"
    )
    print(f"Operator table:     {ops_path}")

    print()
    print("FLOP-bearing operators:")

    for event in sorted(
        events,
        key=lambda x: x.flops or 0,
        reverse=True,
    ):
        if event.flops:
            print(
                f"  {event.key:35s} "
                f"calls={event.count:5d} "
                f"FLOPs={event.flops:15,d}"
            )

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "forward_flops": total_flops,
        "forward_gflops": total_flops / 1e9,
        "sdpa_modules_disabled": num_sdpa_disabled,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Standardized forward-pass FLOPs profiler for "
            "one ParT or MPT training configuration."
        )
    )

    parser.add_argument(
        "--model",
        choices=["ParT", "MPT"],
        required=True,
    )

    parser.add_argument(
        "--run-name",
        required=True,
    )

    parser.add_argument(
        "--num-experts",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--capacity-factor",
        type=float,
        default=None,
        help=(
            "Capacity factor used during TRAINING. "
            "Recorded as run metadata but not used to allow "
            "token dropping during standardized profiling."
        ),
    )

    parser.add_argument(
        "--aux-loss-coef",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--router-jitter",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--profile-capacity-factor",
        type=float,
        default=100.0,
        help=(
            "Capacity factor used ONLY for the synthetic FLOPs "
            "profile. Default 100 disables meaningful overflow."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    # Validate MPT-specific options.
    if args.model == "MPT":
        if args.num_experts is None:
            parser.error(
                "--num-experts is required for MPT"
            )

        if args.top_k is None:
            parser.error(
                "--top-k is required for MPT"
            )

        if args.capacity_factor is None:
            parser.error(
                "--capacity-factor is required for MPT"
            )

        if args.top_k > args.num_experts:
            parser.error(
                "--top-k cannot exceed --num-experts"
            )

    if args.device == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() "
            "is False"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 90)
    print("STANDARDIZED PER-RUN FLOPs PROFILE")
    print("=" * 90)

    print("Run name:", args.run_name)
    print("Model:", args.model)
    print("PyTorch:", torch.__version__)
    print("Device:", device)

    gpu_name = None

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print("GPU:", gpu_name)

    print("Batch size:", args.batch_size)
    print("Sequence length:", args.seq_len)

    if args.model == "MPT":
        print("Experts:", args.num_experts)
        print("Top-k:", args.top_k)
        print(
            "Training capacity factor:",
            args.capacity_factor,
        )
        print(
            "Profiling capacity factor:",
            args.profile_capacity_factor,
        )
        print(
            "Training aux loss coefficient:",
            args.aux_loss_coef,
        )
        print(
            "Training router jitter:",
            args.router_jitter,
        )

    git_commit = get_git_commit()

    print("Git commit:", git_commit)

    data_config = make_fake_data_config(
        args.seq_len
    )

    inputs = make_inputs(
        args.batch_size,
        args.seq_len,
        device,
    )

    print()
    print("Input shapes:")

    for x in inputs:
        print(" ", tuple(x.shape))

    if args.model == "ParT":
        model = build_dense(
            data_config
        )

    else:
        model = build_mpt(
            data_config,
            num_experts=args.num_experts,
            top_k=args.top_k,
            profile_capacity_factor=(
                args.profile_capacity_factor
            ),
        )

    profile_result = profile_model(
        model,
        inputs,
        device,
        output_dir,
    )

    result = {
        "run_name": args.run_name,
        "model": args.model,

        "num_experts": (
            args.num_experts
            if args.model == "MPT"
            else None
        ),

        "top_k": (
            args.top_k
            if args.model == "MPT"
            else None
        ),

        "training_capacity_factor": (
            args.capacity_factor
            if args.model == "MPT"
            else None
        ),

        "training_aux_loss_coef": (
            args.aux_loss_coef
            if args.model == "MPT"
            else None
        ),

        "training_router_jitter": (
            args.router_jitter
            if args.model == "MPT"
            else None
        ),

        "profiling_capacity_factor": (
            args.profile_capacity_factor
            if args.model == "MPT"
            else None
        ),

        "batch_size": args.batch_size,
        "sequence_length": args.seq_len,

        "torch_version": torch.__version__,
        "device": str(device),
        "gpu": gpu_name,
        "git_commit": git_commit,

        **profile_result,
    }

    json_path = output_dir / "profile.json"

    with open(json_path, "w") as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print()
    print("=" * 90)
    print("SAVED")
    print("=" * 90)
    print("Profile JSON:", json_path)
    print(
        "Operator table:",
        output_dir / "profile_ops.txt",
    )

    print()
    print(
        f"{args.run_name}: "
        f"params={profile_result['total_params']:,}, "
        f"GFLOPs="
        f"{profile_result['forward_gflops']:.6f}"
    )


if __name__ == "__main__":
    main()
