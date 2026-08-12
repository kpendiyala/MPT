import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

import model.ParT as ParT
import model.MPT as MPT


def make_fake_data_config():
    # Enough information for ParT.get_model() / MPT.get_model().
    #
    # JetClass-II YAML:
    #   pf_features: 17 x 128
    #   pf_vectors:   4 x 128
    #   pf_mask:      1 x 128
    #
    # pf_points is accepted by the wrapper but is not actually used in the
    # current ParT/MPT forward calls. We give it 2 channels here.
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
            "pf_points": (1, 2, 128),
            "pf_features": (1, 17, 128),
            "pf_vectors": (1, 4, 128),
            "pf_mask": (1, 1, 128),
        },
    )


def make_inputs(batch_size, seq_len, device):
    torch.manual_seed(12345)

    # Same exact tensors are reused for every model.
    points = torch.randn(
        batch_size, 2, seq_len,
        device=device,
        dtype=torch.float32,
    )

    features = torch.randn(
        batch_size, 17, seq_len,
        device=device,
        dtype=torch.float32,
    )

    lorentz_vectors = torch.randn(
        batch_size, 4, seq_len,
        device=device,
        dtype=torch.float32,
    )

    # All particles valid for this first architecture sanity check.
    mask = torch.ones(
        batch_size, 1, seq_len,
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
    capacity_factor,
):
    model, _ = MPT.get_model(
        data_config,
        num_classes=188,
        fc_params=[(512, 0.1)],
        moe_num_experts=num_experts,
        moe_top_k=top_k,
        moe_capacity_factor=capacity_factor,
        moe_aux_loss_coef=0.01,
        moe_router_jitter=0.0,
    )
    return model


def profile_model(
    name,
    model,
    inputs,
    device,
    output_dir,
):
    model = model.to(device)
    model.eval()

    total_params, trainable_params = count_params(model)

    # Warm-up outside profiler.
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

    safe_name = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    txt_path = output_dir / f"{safe_name}_ops.txt"

    with open(txt_path, "w") as f:
        f.write(table)
        f.write("\n\n")
        f.write(f"Total FLOPs: {total_flops}\n")
        f.write(f"GFLOPs: {total_flops / 1e9:.6f}\n")
        f.write(f"Total parameters: {total_params}\n")
        f.write(f"Trainable parameters: {trainable_params}\n")

    print()
    print("=" * 90)
    print(name)
    print("=" * 90)
    print(f"Total params:       {total_params:,}")
    print(f"Trainable params:   {trainable_params:,}")
    print(f"Profiler FLOPs:     {total_flops:,}")
    print(f"Profiler GFLOPs:    {total_flops / 1e9:.6f}")
    print(f"Operator table:     {txt_path}")

    # Print only FLOP-bearing operators for quick inspection.
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

    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "model": name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "flops": total_flops,
        "gflops": total_flops / 1e9,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
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
        default="/kaushik-moe-vol/outputs/profiling",
    )

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is False"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("PyTorch:", torch.__version__)
    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print("Batch size:", args.batch_size)
    print("Sequence length:", args.seq_len)

    data_config = make_fake_data_config()

    inputs = make_inputs(
        args.batch_size,
        args.seq_len,
        device,
    )

    print("Input shapes:")
    for x in inputs:
        print(" ", tuple(x.shape))

    results = []

    # Dense baseline
    results.append(
        profile_model(
            "Dense ParT",
            build_dense(data_config),
            inputs,
            device,
            output_dir,
        )
    )

    # Very high capacity factor is intentional for this sanity test.
    # We do NOT want expert overflow/drop to confuse the profiler comparison.
    sanity_capacity = 100.0

    results.append(
        profile_model(
            "MPT E1 K1",
            build_mpt(
                data_config,
                num_experts=1,
                top_k=1,
                capacity_factor=sanity_capacity,
            ),
            inputs,
            device,
            output_dir,
        )
    )

    results.append(
        profile_model(
            "MPT E4 K1",
            build_mpt(
                data_config,
                num_experts=4,
                top_k=1,
                capacity_factor=sanity_capacity,
            ),
            inputs,
            device,
            output_dir,
        )
    )

    results.append(
        profile_model(
            "MPT E4 K2",
            build_mpt(
                data_config,
                num_experts=4,
                top_k=2,
                capacity_factor=sanity_capacity,
            ),
            inputs,
            device,
            output_dir,
        )
    )

    json_path = output_dir / "flops_sanity_summary.json"

    with open(json_path, "w") as f:
        json.dump(
            {
                "torch_version": torch.__version__,
                "device": str(device),
                "batch_size": args.batch_size,
                "sequence_length": args.seq_len,
                "capacity_factor": sanity_capacity,
                "results": results,
            },
            f,
            indent=2,
        )

    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)

    for result in results:
        print(
            f"{result['model']:15s}  "
            f"params={result['total_params']:,}  "
            f"GFLOPs={result['gflops']:.6f}"
        )

    print()
    print("Saved summary:", json_path)


if __name__ == "__main__":
    main()
