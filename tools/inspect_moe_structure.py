#!/usr/bin/env python3

import argparse
from types import SimpleNamespace

import torch

import model.MPT as MPT


def make_fake_data_config():
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-experts",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--capacity-factor",
        type=float,
        default=1.5,
    )

    args = parser.parse_args()

    data_config = make_fake_data_config()

    model, _ = MPT.get_model(
        data_config,
        num_classes=188,
        fc_params=[(512, 0.1)],
        moe_num_experts=args.num_experts,
        moe_top_k=args.top_k,
        moe_capacity_factor=args.capacity_factor,
        moe_aux_loss_coef=0.01,
        moe_router_jitter=0.0,
    )

    print("=" * 100)
    print("MODEL TYPE")
    print("=" * 100)
    print(type(model))

    print()
    print("=" * 100)
    print("MODULES WITH ROUTER / EXPERT ATTRIBUTES")
    print("=" * 100)

    found = 0

    for name, module in model.named_modules():

        attrs = []

        for attr in [
            "router",
            "experts",
            "num_experts",
            "top_k",
            "capacity_factor",
            "capacity",
            "aux_loss",
            "router_jitter",
        ]:
            if hasattr(module, attr):
                attrs.append(attr)

        if not attrs:
            continue

        found += 1

        print()
        print("-" * 100)
        print("Module:", name)
        print("Class :", module.__class__.__name__)
        print("Attrs :", attrs)

        for attr in attrs:

            obj = getattr(module, attr)

            if isinstance(obj, torch.nn.Module):
                print(
                    f"  {attr}: "
                    f"{obj.__class__.__name__}"
                )

            elif isinstance(
                obj,
                (int, float, str, bool, type(None)),
            ):
                print(
                    f"  {attr}: {obj}"
                )

            else:
                print(
                    f"  {attr}: "
                    f"{type(obj).__name__}"
                )

        print()
        print("Public-ish attributes:")

        public_attrs = [
            x
            for x in dir(module)
            if not x.startswith("_")
        ]

        interesting = [
            x
            for x in public_attrs
            if any(
                key in x.lower()
                for key in [
                    "route",
                    "router",
                    "expert",
                    "capacity",
                    "token",
                    "drop",
                    "aux",
                    "gate",
                    "dispatch",
                    "load",
                ]
            )
        ]

        for attr in interesting:
            try:
                obj = getattr(module, attr)

                if callable(obj):
                    kind = "callable"
                elif torch.is_tensor(obj):
                    kind = (
                        f"Tensor(shape={tuple(obj.shape)})"
                    )
                else:
                    kind = repr(obj)

                print(
                    f"  {attr}: {kind}"
                )

            except Exception as exc:
                print(
                    f"  {attr}: <error: {exc}>"
                )

    print()
    print("=" * 100)
    print(
        f"Found {found} module(s) with "
        "MoE-related attributes."
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
