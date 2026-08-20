#!/usr/bin/env python3

import atexit
import csv
import json
import math
import os
from pathlib import Path

import torch


class RoutingCollector:
    """
    Collect routing statistics from MPT router Linear modules.

    Designed specifically around the top-1 routing logic implemented in
    model/MoETransformer.py.

    The collector does NOT alter router outputs or model behavior.
    """

    def __init__(
        self,
        model,
        output_dir,
        run_name=None,
    ):
        self.model = model
        self.output_dir = Path(output_dir)
        self.run_name = run_name

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.handles = []

        self.layers = {}

        self.enabled = True
        self.dumped = False

        self._register_hooks()

        atexit.register(self.dump)

    def _register_hooks(self):
        print()
        print("=" * 80)
        print("REGISTERING ROUTING DIAGNOSTIC HOOKS")
        print("=" * 80)

        for name, module in self.model.named_modules():

            if not hasattr(module, "router"):
                continue

            if not hasattr(module, "experts"):
                continue

            if not hasattr(module, "moe_num_experts"):
                continue

            if module.moe_top_k != 1:
                raise RuntimeError(
                    f"{name}: this routing collector currently "
                    f"supports top-k=1 only, but model has "
                    f"top-k={module.moe_top_k}"
                )

            num_experts = int(
                module.moe_num_experts
            )

            self.layers[name] = {
                "num_experts": num_experts,

                "capacity_factor": float(
                    module.moe_capacity_factor
                ),

                "aux_loss_coef": float(
                    module.moe_aux_loss_coef
                ),

                "num_batches": 0,
                "num_routing_calls": 0,

                "total_tokens": 0,

                "requested": [
                    0 for _ in range(num_experts)
                ],

                "accepted": [
                    0 for _ in range(num_experts)
                ],

                "dropped": [
                    0 for _ in range(num_experts)
                ],

                "capacity_sum": [
                    0 for _ in range(num_experts)
                ],

                "entropy_sum": 0.0,
                "entropy_token_count": 0,

                "max_probability_sum": 0.0,
                "max_probability_token_count": 0,

                "aux_sum": 0.0,
                "aux_calls": 0,
            }

            handle = module.router.register_forward_hook(
                self._make_router_hook(
                    layer_name=name,
                    block=module,
                )
            )

            self.handles.append(handle)

            print(
                f"Hooked {name}: "
                f"E={num_experts}, "
                f"K={module.moe_top_k}, "
                f"capacity_factor="
                f"{module.moe_capacity_factor}"
            )

        print()
        print(
            f"Total MoE blocks hooked: "
            f"{len(self.layers)}"
        )
        print("=" * 80)
        print()

        if not self.layers:
            raise RuntimeError(
                "No MoE blocks were found in the model."
            )

    def _make_router_hook(
        self,
        layer_name,
        block,
    ):
        def hook(
            router_module,
            inputs,
            output,
        ):
            if not self.enabled:
                return

            with torch.no_grad():

                router_logits = output.detach()

                if router_logits.ndim != 2:
                    raise RuntimeError(
                        f"{layer_name}: expected router output "
                        f"shape [tokens, experts], got "
                        f"{tuple(router_logits.shape)}"
                    )

                gates = torch.softmax(
                    router_logits,
                    dim=-1,
                )

                num_tokens = gates.size(0)

                num_experts = int(
                    block.moe_num_experts
                )

                capacity = int(
                    block.moe_capacity_factor
                    * math.ceil(
                        num_tokens
                        / max(1, num_experts)
                    )
                )

                top1_idx = gates.argmax(
                    dim=-1
                )

                top1_w = gates.gather(
                    1,
                    top1_idx.unsqueeze(1),
                ).squeeze(1)

                stats = self.layers[
                    layer_name
                ]

                stats["num_routing_calls"] += 1
                stats["total_tokens"] += (
                    num_tokens
                )

                # ----------------------------------
                # Router entropy
                # ----------------------------------

                entropy = -(
                    gates
                    * torch.log(
                        gates.clamp(min=1e-12)
                    )
                ).sum(dim=-1)

                stats["entropy_sum"] += float(
                    entropy.sum().item()
                )

                stats[
                    "entropy_token_count"
                ] += num_tokens

                # Maximum router probability:
                # how confident the router is in
                # its winning expert.
                max_prob = gates.max(
                    dim=-1
                ).values

                stats[
                    "max_probability_sum"
                ] += float(
                    max_prob.sum().item()
                )

                stats[
                    "max_probability_token_count"
                ] += num_tokens

                # ----------------------------------
                # Exact top-1 capacity accounting
                # ----------------------------------

                requested_fracs = []

                for e in range(
                    num_experts
                ):

                    mask = (
                        top1_idx == e
                    )

                    idx = torch.nonzero(
                        mask,
                        as_tuple=False,
                    ).squeeze(1)

                    requested = int(
                        idx.numel()
                    )

                    accepted = min(
                        requested,
                        capacity,
                    )

                    dropped = (
                        requested - accepted
                    )

                    stats[
                        "requested"
                    ][e] += requested

                    stats[
                        "accepted"
                    ][e] += accepted

                    stats[
                        "dropped"
                    ][e] += dropped

                    stats[
                        "capacity_sum"
                    ][e] += capacity

                    requested_fracs.append(
                        requested
                        / max(1, num_tokens)
                    )

                # ----------------------------------
                # Reconstruct the same auxiliary
                # load-balancing quantity used by
                # MoETransformer.py
                # ----------------------------------

                mean_prob_per_expert = (
                    gates.mean(dim=0)
                )

                frac_per_expert = (
                    torch.bincount(
                        top1_idx,
                        minlength=num_experts,
                    ).float()
                    / float(max(1, num_tokens))
                )

                aux = (
                    mean_prob_per_expert
                    * frac_per_expert
                ).sum() * num_experts

                stats["aux_sum"] += float(
                    aux.item()
                )

                stats["aux_calls"] += 1

        return hook

    @staticmethod
    def _safe_div(
        numerator,
        denominator,
    ):
        if denominator == 0:
            return None

        return numerator / denominator

    def dump(self):
        if self.dumped:
            return

        self.dumped = True

        if not self.layers:
            return

        print()
        print("=" * 80)
        print("WRITING ROUTING DIAGNOSTICS")
        print("=" * 80)

        expert_rows = []
        layer_rows = []

        total_requested_all = 0
        total_accepted_all = 0
        total_dropped_all = 0

        entropy_values = []
        imbalance_values = []
        aux_values = []

        for (
            layer_name,
            stats,
        ) in self.layers.items():

            num_experts = stats[
                "num_experts"
            ]

            requested = stats[
                "requested"
            ]

            accepted = stats[
                "accepted"
            ]

            dropped = stats[
                "dropped"
            ]

            capacity_sum = stats[
                "capacity_sum"
            ]

            total_requested = sum(
                requested
            )

            total_accepted = sum(
                accepted
            )

            total_dropped = sum(
                dropped
            )

            total_requested_all += (
                total_requested
            )

            total_accepted_all += (
                total_accepted
            )

            total_dropped_all += (
                total_dropped
            )

            requested_fractions = [
                self._safe_div(
                    x,
                    total_requested,
                )
                or 0.0
                for x in requested
            ]

            ideal_fraction = (
                1.0 / num_experts
            )

            max_fraction = max(
                requested_fractions
            )

            min_fraction = min(
                requested_fractions
            )

            load_imbalance_ratio = (
                max_fraction
                / ideal_fraction
            )

            max_to_min_ratio = (
                max_fraction
                / min_fraction
                if min_fraction > 0
                else None
            )

            mean_entropy = (
                self._safe_div(
                    stats["entropy_sum"],
                    stats[
                        "entropy_token_count"
                    ],
                )
            )

            normalized_entropy = None

            if mean_entropy is not None:
                normalized_entropy = (
                    mean_entropy
                    / math.log(
                        num_experts
                    )
                )

            mean_max_probability = (
                self._safe_div(
                    stats[
                        "max_probability_sum"
                    ],
                    stats[
                        "max_probability_token_count"
                    ],
                )
            )

            mean_aux = (
                self._safe_div(
                    stats["aux_sum"],
                    stats["aux_calls"],
                )
            )

            if (
                normalized_entropy
                is not None
            ):
                entropy_values.append(
                    normalized_entropy
                )

            imbalance_values.append(
                load_imbalance_ratio
            )

            if mean_aux is not None:
                aux_values.append(
                    mean_aux
                )

            layer_rows.append({
                "layer": layer_name,

                "num_experts":
                    num_experts,

                "capacity_factor":
                    stats[
                        "capacity_factor"
                    ],

                "routing_calls":
                    stats[
                        "num_routing_calls"
                    ],

                "total_requested_tokens":
                    total_requested,

                "total_accepted_tokens":
                    total_accepted,

                "total_dropped_tokens":
                    total_dropped,

                "drop_fraction":
                    self._safe_div(
                        total_dropped,
                        total_requested,
                    ),

                "max_requested_fraction":
                    max_fraction,

                "min_requested_fraction":
                    min_fraction,

                "ideal_fraction":
                    ideal_fraction,

                "load_imbalance_ratio":
                    load_imbalance_ratio,

                "max_to_min_load_ratio":
                    max_to_min_ratio,

                "mean_router_entropy":
                    mean_entropy,

                "normalized_router_entropy":
                    normalized_entropy,

                "mean_top1_probability":
                    mean_max_probability,

                "mean_aux_balance_value":
                    mean_aux,
            })

            for e in range(
                num_experts
            ):

                expert_rows.append({
                    "layer":
                        layer_name,

                    "expert":
                        e,

                    "requested_tokens":
                        requested[e],

                    "accepted_tokens":
                        accepted[e],

                    "dropped_tokens":
                        dropped[e],

                    "requested_fraction":
                        self._safe_div(
                            requested[e],
                            total_requested,
                        ),

                    "accepted_fraction":
                        self._safe_div(
                            accepted[e],
                            total_accepted,
                        ),

                    "drop_fraction":
                        self._safe_div(
                            dropped[e],
                            requested[e],
                        ),

                    "mean_capacity":
                        self._safe_div(
                            capacity_sum[e],
                            stats[
                                "num_routing_calls"
                            ],
                        ),

                    "capacity_utilization":
                        self._safe_div(
                            accepted[e],
                            capacity_sum[e],
                        ),
                })

        # ----------------------------------
        # Write expert-level CSV
        # ----------------------------------

        expert_path = (
            self.output_dir
            / "expert_usage.csv"
        )

        with open(
            expert_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    expert_rows[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(
                expert_rows
            )

        # ----------------------------------
        # Write layer-level CSV
        # ----------------------------------

        layer_path = (
            self.output_dir
            / "layer_summary.csv"
        )

        with open(
            layer_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    layer_rows[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(
                layer_rows
            )

        # ----------------------------------
        # Run-level summary
        # ----------------------------------

        summary = {
            "run_name":
                self.run_name,

            "num_moe_blocks":
                len(self.layers),

            "total_requested_tokens":
                total_requested_all,

            "total_accepted_tokens":
                total_accepted_all,

            "total_dropped_tokens":
                total_dropped_all,

            "overall_drop_fraction":
                self._safe_div(
                    total_dropped_all,
                    total_requested_all,
                ),

            "mean_normalized_router_entropy":
                (
                    sum(entropy_values)
                    / len(entropy_values)
                    if entropy_values
                    else None
                ),

            "mean_load_imbalance_ratio":
                (
                    sum(imbalance_values)
                    / len(imbalance_values)
                    if imbalance_values
                    else None
                ),

            "max_load_imbalance_ratio":
                (
                    max(imbalance_values)
                    if imbalance_values
                    else None
                ),

            "mean_aux_balance_value":
                (
                    sum(aux_values)
                    / len(aux_values)
                    if aux_values
                    else None
                ),

            "notes": {
                "routing":
                    "Top-1 routing",

                "capacity_definition":
                    (
                        "capacity_factor * "
                        "ceil(tokens / num_experts)"
                    ),

                "drop_definition":
                    (
                        "tokens assigned to an "
                        "expert above its capacity"
                    ),

                "normalized_entropy":
                    (
                        "router entropy / "
                        "log(num_experts)"
                    ),

                "load_imbalance_ratio":
                    (
                        "maximum requested expert "
                        "fraction divided by ideal "
                        "1/num_experts"
                    ),
            },
        }

        summary_path = (
            self.output_dir
            / "routing_summary.json"
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

        print(
            f"Expert usage: {expert_path}"
        )

        print(
            f"Layer summary: {layer_path}"
        )

        print(
            f"Run summary: {summary_path}"
        )

        print("=" * 80)
        print()
