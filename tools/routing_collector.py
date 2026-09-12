#!/usr/bin/env python3

import atexit
import csv
import json
import math
from pathlib import Path

import torch


class RoutingCollector:
    """
    Non-invasive routing diagnostics for model/MoETransformer.py.

    Supports both top-1 and top-k MPT routing.

    Important accounting convention:
      - total_tokens / total_valid_tokens count input tokens.
      - requested / accepted / dropped count expert ASSIGNMENTS.
        Therefore, before capacity dropping:
            K=1 -> requested assignments ~= 1 x tokens
            K=2 -> requested assignments ~= 2 x tokens
            K=4 -> requested assignments ~= 4 x tokens

    Collects:
      - requested / accepted / dropped expert assignments
      - per-expert assignment load
      - capacity utilization
      - routing entropy
      - top-1 router confidence
      - mean selected-expert dispatch weight
      - auxiliary balancing quantity
      - valid-particle-only assignment statistics

    Hooks only observe the model. They do not modify routing behavior.

    The dispatch accounting mirrors model/MoETransformer.py:
      - K=1 capacity:
          capacity_factor * ceil(num_tokens / num_experts)
      - K>1 capacity:
          capacity_factor * ceil((num_tokens * K) / num_experts)
      - K>1 selected gate values are renormalized to sum to 1 per token
      - overflow keeps the highest dispatch weights for each expert
    """

    def __init__(self, model, output_dir, run_name=None):
        self.model = model
        self.output_dir = Path(output_dir)
        self.run_name = run_name

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.layers = {}
        self.current_valid_masks = {}

        self.handles = []
        self.dumped = False

        self._register_hooks()

        atexit.register(self.dump)

    def _register_hooks(self):
        print("=" * 80)
        print("REGISTERING ROUTING DIAGNOSTICS")
        print("=" * 80)

        for name, module in self.model.named_modules():

            if not (
                hasattr(module, "router")
                and hasattr(module, "experts")
                and hasattr(module, "moe_num_experts")
                and hasattr(module, "moe_top_k")
            ):
                continue

            num_experts = int(module.moe_num_experts)
            top_k = int(module.moe_top_k)

            if num_experts < 1:
                raise RuntimeError(
                    f"{name}: invalid num_experts={num_experts}"
                )

            if top_k < 1:
                raise RuntimeError(
                    f"{name}: invalid top_k={top_k}"
                )

            # The model itself uses:
            #   k = min(moe_top_k, moe_num_experts)
            effective_top_k = min(top_k, num_experts)

            self.layers[name] = {
                "num_experts": num_experts,
                "top_k": top_k,
                "effective_top_k": effective_top_k,
                "capacity_factor": float(module.moe_capacity_factor),

                "routing_calls": 0,
                "total_tokens": 0,
                "total_valid_tokens": 0,

                # These arrays count expert assignments, not unique tokens,
                # when K > 1.
                "requested": [0] * num_experts,
                "accepted": [0] * num_experts,
                "dropped": [0] * num_experts,

                "valid_requested": [0] * num_experts,
                "valid_accepted": [0] * num_experts,
                "valid_dropped": [0] * num_experts,

                "capacity_sum": [0] * num_experts,

                "entropy_sum": 0.0,
                "entropy_token_count": 0,

                "valid_entropy_sum": 0.0,
                "valid_entropy_token_count": 0,

                "top1_probability_sum": 0.0,
                "top1_probability_count": 0,

                # For K=1 this is the selected raw top-1 gate probability.
                # For K>1 this is the renormalized dispatch weight used by
                # MoETransformer.py after selecting the top-k experts.
                "selected_probability_sum": 0.0,
                "selected_probability_count": 0,

                "aux_sum": 0.0,
                "aux_calls": 0,
            }

            # Capture padding information before Block.forward executes.
            pre_handle = module.register_forward_pre_hook(
                self._make_block_pre_hook(name),
                with_kwargs=True,
            )

            router_handle = module.router.register_forward_hook(
                self._make_router_hook(name, module)
            )

            self.handles.extend([pre_handle, router_handle])

            print(
                f"{name}: "
                f"E={num_experts}, "
                f"K={top_k}, "
                f"effective_K={effective_top_k}, "
                f"capacity_factor={module.moe_capacity_factor}"
            )

        if not self.layers:
            raise RuntimeError("No MoE blocks found.")

        print(f"Total MoE blocks: {len(self.layers)}")
        print("=" * 80)

    def _make_block_pre_hook(self, layer_name):

        def hook(module, args, kwargs):
            x = args[0] if len(args) > 0 else kwargs.get("x")

            x_cls = kwargs.get("x_cls")
            padding_mask = kwargs.get("padding_mask")

            # Class-token blocks route x_cls after attention.
            # Shape is (1, batch, embed_dim), so every routed token is valid.
            if x_cls is not None:
                batch_size = x_cls.shape[1]

                self.current_valid_masks[layer_name] = torch.ones(
                    batch_size,
                    dtype=torch.bool,
                    device=x_cls.device,
                )

                return

            # Regular transformer blocks route every sequence position.
            if padding_mask is None:
                seq_len, batch_size, _ = x.shape

                self.current_valid_masks[layer_name] = torch.ones(
                    seq_len * batch_size,
                    dtype=torch.bool,
                    device=x.device,
                )

                return

            # padding_mask shape = (batch, seq)
            #
            # tokens are created from x with:
            #   x.shape = (seq, batch, embed)
            #   x.reshape(seq * batch, embed)
            #
            # therefore transpose mask before flattening.
            valid_mask = (~padding_mask.bool()).transpose(0, 1).reshape(-1)

            self.current_valid_masks[layer_name] = valid_mask

        return hook

    def _make_router_hook(self, layer_name, block):

        def hook(router_module, inputs, output):

            with torch.no_grad():

                logits = output.detach()

                if logits.ndim != 2:
                    raise RuntimeError(
                        f"{layer_name}: unexpected router shape "
                        f"{tuple(logits.shape)}"
                    )

                gates = torch.softmax(logits, dim=-1)

                num_tokens = gates.shape[0]
                num_experts = int(block.moe_num_experts)
                configured_top_k = int(block.moe_top_k)
                k = min(configured_top_k, num_experts)

                if k < 1:
                    raise RuntimeError(
                        f"{layer_name}: invalid effective top-k={k}"
                    )

                valid_mask = self.current_valid_masks.get(layer_name)

                if (
                    valid_mask is None
                    or valid_mask.numel() != num_tokens
                ):
                    valid_mask = torch.ones(
                        num_tokens,
                        dtype=torch.bool,
                        device=gates.device,
                    )

                stats = self.layers[layer_name]

                stats["routing_calls"] += 1
                stats["total_tokens"] += int(num_tokens)
                stats["total_valid_tokens"] += int(valid_mask.sum().item())

                # -------------------------------------------------
                # Router entropy
                # -------------------------------------------------

                entropy = -(
                    gates
                    * torch.log(gates.clamp(min=1e-12))
                ).sum(dim=-1)

                stats["entropy_sum"] += float(entropy.sum().item())
                stats["entropy_token_count"] += int(num_tokens)

                if valid_mask.any():
                    stats["valid_entropy_sum"] += float(
                        entropy[valid_mask].sum().item()
                    )

                    stats["valid_entropy_token_count"] += int(
                        valid_mask.sum().item()
                    )

                max_prob = gates.max(dim=-1).values

                stats["top1_probability_sum"] += float(
                    max_prob.sum().item()
                )

                stats["top1_probability_count"] += int(num_tokens)

                # -------------------------------------------------
                # Dispatch accounting
                # -------------------------------------------------

                if k == 1:
                    # Mirrors the model's top-1 branch exactly.
                    top1_idx = gates.argmax(dim=-1)

                    top1_w = gates.gather(
                        1,
                        top1_idx.unsqueeze(1),
                    ).squeeze(1)

                    capacity = int(
                        block.moe_capacity_factor
                        * math.ceil(
                            num_tokens / max(1, num_experts)
                        )
                    )

                    stats["selected_probability_sum"] += float(
                        top1_w.sum().item()
                    )
                    stats["selected_probability_count"] += int(
                        top1_w.numel()
                    )

                    for e in range(num_experts):

                        idx = torch.nonzero(
                            top1_idx == e,
                            as_tuple=False,
                        ).squeeze(1)

                        requested = int(idx.numel())

                        selected_idx = idx

                        if requested > capacity:

                            selected = top1_w[idx].topk(
                                capacity,
                                sorted=False,
                            ).indices

                            selected_idx = idx[selected]

                        accepted = int(selected_idx.numel())
                        dropped = requested - accepted

                        stats["requested"][e] += requested
                        stats["accepted"][e] += accepted
                        stats["dropped"][e] += dropped
                        stats["capacity_sum"][e] += capacity

                        if requested > 0:
                            valid_requested = int(
                                valid_mask[idx].sum().item()
                            )
                        else:
                            valid_requested = 0

                        if accepted > 0:
                            valid_accepted = int(
                                valid_mask[selected_idx].sum().item()
                            )
                        else:
                            valid_accepted = 0

                        valid_dropped = (
                            valid_requested - valid_accepted
                        )

                        stats["valid_requested"][e] += valid_requested
                        stats["valid_accepted"][e] += valid_accepted
                        stats["valid_dropped"][e] += valid_dropped

                else:
                    # Mirrors the model's top-k branch exactly:
                    #   1. select top-k gate values/indices
                    #   2. renormalize selected gates to sum to one
                    #   3. capacity accounts for K assignments/token
                    #   4. overflow is selected by renormalized dispatch
                    #      weight for that expert assignment
                    topk_vals, topk_idx = gates.topk(
                        k=k,
                        dim=-1,
                    )

                    denom = topk_vals.sum(
                        dim=1,
                        keepdim=True,
                    ).clamp(min=1e-9)

                    topk_w = topk_vals / denom

                    capacity = int(
                        block.moe_capacity_factor
                        * math.ceil(
                            (num_tokens * k)
                            / max(1, num_experts)
                        )
                    )

                    stats["selected_probability_sum"] += float(
                        topk_w.sum().item()
                    )
                    stats["selected_probability_count"] += int(
                        topk_w.numel()
                    )

                    for e in range(num_experts):

                        mask_e = (topk_idx == e)

                        if mask_e.any():
                            rows, cols = torch.nonzero(
                                mask_e,
                                as_tuple=True,
                            )
                        else:
                            rows = torch.empty(
                                0,
                                dtype=torch.long,
                                device=gates.device,
                            )
                            cols = torch.empty(
                                0,
                                dtype=torch.long,
                                device=gates.device,
                            )

                        requested = int(rows.numel())

                        selected_rows = rows
                        selected_cols = cols

                        if requested > capacity:

                            weights_e = topk_w[rows, cols]

                            selected = weights_e.topk(
                                capacity,
                                sorted=False,
                            ).indices

                            selected_rows = rows.index_select(
                                0,
                                selected,
                            )

                            selected_cols = cols.index_select(
                                0,
                                selected,
                            )

                        accepted = int(selected_rows.numel())
                        dropped = requested - accepted

                        stats["requested"][e] += requested
                        stats["accepted"][e] += accepted
                        stats["dropped"][e] += dropped
                        stats["capacity_sum"][e] += capacity

                        # For top-k, each row here is an expert assignment.
                        # A token can therefore contribute up to K times across
                        # all experts, which is exactly what we want to count.
                        if requested > 0:
                            valid_requested = int(
                                valid_mask[rows].sum().item()
                            )
                        else:
                            valid_requested = 0

                        if accepted > 0:
                            valid_accepted = int(
                                valid_mask[selected_rows].sum().item()
                            )
                        else:
                            valid_accepted = 0

                        valid_dropped = (
                            valid_requested - valid_accepted
                        )

                        stats["valid_requested"][e] += valid_requested
                        stats["valid_accepted"][e] += valid_accepted
                        stats["valid_dropped"][e] += valid_dropped

                # -------------------------------------------------
                # Same balancing quantity used in Block.forward()
                # -------------------------------------------------

                mean_prob_per_expert = gates.mean(dim=0)

                if configured_top_k == 1:

                    token_choice = gates.argmax(dim=-1)

                    frac_per_expert = torch.stack([
                        (token_choice == e).float().mean()
                        for e in range(num_experts)
                    ])

                else:

                    topk_idx_aux = gates.topk(
                        k=k,
                        dim=-1,
                    ).indices

                    assigned = torch.zeros_like(gates)

                    assigned.scatter_(
                        1,
                        topk_idx_aux,
                        1.0,
                    )

                    # Match MoETransformer.py exactly.
                    # In the current model configurations K <= E, so
                    # configured_top_k == effective k.
                    frac_per_expert = (
                        assigned.mean(dim=0)
                        / float(configured_top_k)
                    )

                aux = (
                    mean_prob_per_expert
                    * frac_per_expert
                ).sum() * num_experts

                stats["aux_sum"] += float(aux.item())
                stats["aux_calls"] += 1

        return hook

    @staticmethod
    def _div(a, b):
        if b == 0:
            return None
        return a / b

    def dump(self):

        if self.dumped:
            return

        self.dumped = True

        expert_rows = []
        layer_rows = []

        run_requested = 0
        run_accepted = 0
        run_dropped = 0

        run_valid_requested = 0
        run_valid_accepted = 0
        run_valid_dropped = 0

        run_total_tokens = 0
        run_total_valid_tokens = 0

        normalized_entropies = []
        valid_normalized_entropies = []
        imbalance_ratios = []
        valid_imbalance_ratios = []
        aux_values = []
        top1_probabilities = []
        selected_probabilities = []

        for layer_name, stats in self.layers.items():

            n = stats["num_experts"]
            top_k = stats["top_k"]
            effective_top_k = stats["effective_top_k"]

            total_requested = sum(stats["requested"])
            total_accepted = sum(stats["accepted"])
            total_dropped = sum(stats["dropped"])

            valid_requested = sum(stats["valid_requested"])
            valid_accepted = sum(stats["valid_accepted"])
            valid_dropped = sum(stats["valid_dropped"])

            run_total_tokens += stats["total_tokens"]
            run_total_valid_tokens += stats["total_valid_tokens"]

            run_requested += total_requested
            run_accepted += total_accepted
            run_dropped += total_dropped

            run_valid_requested += valid_requested
            run_valid_accepted += valid_accepted
            run_valid_dropped += valid_dropped

            fractions = [
                self._div(x, total_requested) or 0.0
                for x in stats["requested"]
            ]

            valid_fractions = [
                self._div(x, valid_requested) or 0.0
                for x in stats["valid_requested"]
            ]

            ideal = 1.0 / n

            max_fraction = max(fractions)
            min_fraction = min(fractions)

            load_imbalance = max_fraction / ideal

            valid_load_imbalance = (
                max(valid_fractions) / ideal
                if valid_requested > 0
                else None
            )

            mean_entropy = self._div(
                stats["entropy_sum"],
                stats["entropy_token_count"],
            )

            mean_valid_entropy = self._div(
                stats["valid_entropy_sum"],
                stats["valid_entropy_token_count"],
            )

            norm_entropy = (
                mean_entropy / math.log(n)
                if mean_entropy is not None and n > 1
                else 0.0 if mean_entropy is not None else None
            )

            norm_valid_entropy = (
                mean_valid_entropy / math.log(n)
                if mean_valid_entropy is not None and n > 1
                else 0.0 if mean_valid_entropy is not None else None
            )

            mean_top1_prob = self._div(
                stats["top1_probability_sum"],
                stats["top1_probability_count"],
            )

            mean_selected_prob = self._div(
                stats["selected_probability_sum"],
                stats["selected_probability_count"],
            )

            mean_aux = self._div(
                stats["aux_sum"],
                stats["aux_calls"],
            )

            expected_assignments = (
                stats["total_tokens"] * effective_top_k
            )

            valid_expected_assignments = (
                stats["total_valid_tokens"] * effective_top_k
            )

            assignment_ratio = self._div(
                total_requested,
                expected_assignments,
            )

            valid_assignment_ratio = self._div(
                valid_requested,
                valid_expected_assignments,
            )

            if norm_entropy is not None:
                normalized_entropies.append(norm_entropy)

            if norm_valid_entropy is not None:
                valid_normalized_entropies.append(norm_valid_entropy)

            imbalance_ratios.append(load_imbalance)

            if valid_load_imbalance is not None:
                valid_imbalance_ratios.append(
                    valid_load_imbalance
                )

            if mean_aux is not None:
                aux_values.append(mean_aux)

            if mean_top1_prob is not None:
                top1_probabilities.append(mean_top1_prob)

            if mean_selected_prob is not None:
                selected_probabilities.append(mean_selected_prob)

            layer_rows.append({
                "layer": layer_name,
                "num_experts": n,
                "top_k": top_k,
                "effective_top_k": effective_top_k,
                "capacity_factor": stats["capacity_factor"],

                "total_tokens": stats["total_tokens"],
                "total_valid_tokens": stats["total_valid_tokens"],

                # Backward-compatible field names. For K>1 these count
                # expert assignments rather than unique tokens.
                "requested_tokens": total_requested,
                "accepted_tokens": total_accepted,
                "dropped_tokens": total_dropped,

                # Explicit assignment aliases for clarity.
                "requested_assignments": total_requested,
                "accepted_assignments": total_accepted,
                "dropped_assignments": total_dropped,

                "expected_assignments_without_drop":
                    expected_assignments,

                "requested_to_expected_assignment_ratio":
                    assignment_ratio,

                "drop_fraction": self._div(
                    total_dropped,
                    total_requested,
                ),

                "valid_requested_tokens": valid_requested,
                "valid_accepted_tokens": valid_accepted,
                "valid_dropped_tokens": valid_dropped,

                "valid_requested_assignments": valid_requested,
                "valid_accepted_assignments": valid_accepted,
                "valid_dropped_assignments": valid_dropped,

                "valid_expected_assignments_without_drop":
                    valid_expected_assignments,

                "valid_requested_to_expected_assignment_ratio":
                    valid_assignment_ratio,

                "valid_drop_fraction": self._div(
                    valid_dropped,
                    valid_requested,
                ),

                "max_requested_fraction": max_fraction,
                "min_requested_fraction": min_fraction,

                "load_imbalance_ratio": load_imbalance,
                "valid_load_imbalance_ratio":
                    valid_load_imbalance,

                "mean_router_entropy": mean_entropy,
                "normalized_router_entropy": norm_entropy,

                "mean_valid_router_entropy":
                    mean_valid_entropy,

                "normalized_valid_router_entropy":
                    norm_valid_entropy,

                "mean_top1_probability": mean_top1_prob,
                "mean_selected_probability":
                    mean_selected_prob,

                "mean_aux_balance_value": mean_aux,
            })

            for e in range(n):

                expert_rows.append({
                    "layer": layer_name,
                    "expert": e,
                    "num_experts": n,
                    "top_k": top_k,
                    "effective_top_k": effective_top_k,

                    # Backward-compatible field names.
                    "requested_tokens": stats["requested"][e],
                    "accepted_tokens": stats["accepted"][e],
                    "dropped_tokens": stats["dropped"][e],

                    # Explicit assignment aliases.
                    "requested_assignments":
                        stats["requested"][e],

                    "accepted_assignments":
                        stats["accepted"][e],

                    "dropped_assignments":
                        stats["dropped"][e],

                    "requested_fraction": self._div(
                        stats["requested"][e],
                        total_requested,
                    ),

                    "drop_fraction": self._div(
                        stats["dropped"][e],
                        stats["requested"][e],
                    ),

                    "valid_requested_tokens":
                        stats["valid_requested"][e],

                    "valid_accepted_tokens":
                        stats["valid_accepted"][e],

                    "valid_dropped_tokens":
                        stats["valid_dropped"][e],

                    "valid_requested_assignments":
                        stats["valid_requested"][e],

                    "valid_accepted_assignments":
                        stats["valid_accepted"][e],

                    "valid_dropped_assignments":
                        stats["valid_dropped"][e],

                    "valid_requested_fraction": self._div(
                        stats["valid_requested"][e],
                        valid_requested,
                    ),

                    "valid_drop_fraction": self._div(
                        stats["valid_dropped"][e],
                        stats["valid_requested"][e],
                    ),

                    "mean_capacity": self._div(
                        stats["capacity_sum"][e],
                        stats["routing_calls"],
                    ),

                    "capacity_utilization": self._div(
                        stats["accepted"][e],
                        stats["capacity_sum"][e],
                    ),
                })

        expert_path = self.output_dir / "expert_usage.csv"

        with open(
            expert_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(expert_rows[0].keys()),
            )

            writer.writeheader()
            writer.writerows(expert_rows)

        layer_path = self.output_dir / "layer_summary.csv"

        with open(
            layer_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(layer_rows[0].keys()),
            )

            writer.writeheader()
            writer.writerows(layer_rows)

        configured_top_ks = sorted({
            stats["top_k"]
            for stats in self.layers.values()
        })

        effective_top_ks = sorted({
            stats["effective_top_k"]
            for stats in self.layers.values()
        })

        capacity_factors = sorted({
            stats["capacity_factor"]
            for stats in self.layers.values()
        })

        summary = {
            "run_name": self.run_name,

            "num_moe_blocks": len(self.layers),

            "configured_top_k": (
                configured_top_ks[0]
                if len(configured_top_ks) == 1
                else configured_top_ks
            ),

            "effective_top_k": (
                effective_top_ks[0]
                if len(effective_top_ks) == 1
                else effective_top_ks
            ),

            "capacity_factor": (
                capacity_factors[0]
                if len(capacity_factors) == 1
                else capacity_factors
            ),

            "total_tokens": run_total_tokens,
            "total_valid_tokens": run_total_valid_tokens,

            # Backward-compatible summary field names.
            # For K>1 they count expert assignments.
            "total_requested_tokens": run_requested,
            "total_accepted_tokens": run_accepted,
            "total_dropped_tokens": run_dropped,

            # Explicit assignment names.
            "total_requested_assignments": run_requested,
            "total_accepted_assignments": run_accepted,
            "total_dropped_assignments": run_dropped,

            "overall_drop_fraction": self._div(
                run_dropped,
                run_requested,
            ),

            "valid_requested_tokens": run_valid_requested,
            "valid_accepted_tokens": run_valid_accepted,
            "valid_dropped_tokens": run_valid_dropped,

            "valid_requested_assignments":
                run_valid_requested,

            "valid_accepted_assignments":
                run_valid_accepted,

            "valid_dropped_assignments":
                run_valid_dropped,

            "overall_valid_drop_fraction": self._div(
                run_valid_dropped,
                run_valid_requested,
            ),

            "mean_normalized_router_entropy": (
                sum(normalized_entropies)
                / len(normalized_entropies)
                if normalized_entropies
                else None
            ),

            "mean_normalized_valid_router_entropy": (
                sum(valid_normalized_entropies)
                / len(valid_normalized_entropies)
                if valid_normalized_entropies
                else None
            ),

            "mean_load_imbalance_ratio": (
                sum(imbalance_ratios)
                / len(imbalance_ratios)
                if imbalance_ratios
                else None
            ),

            "max_load_imbalance_ratio": (
                max(imbalance_ratios)
                if imbalance_ratios
                else None
            ),

            "mean_valid_load_imbalance_ratio": (
                sum(valid_imbalance_ratios)
                / len(valid_imbalance_ratios)
                if valid_imbalance_ratios
                else None
            ),

            "max_valid_load_imbalance_ratio": (
                max(valid_imbalance_ratios)
                if valid_imbalance_ratios
                else None
            ),

            "mean_top1_probability": (
                sum(top1_probabilities)
                / len(top1_probabilities)
                if top1_probabilities
                else None
            ),

            "mean_selected_probability": (
                sum(selected_probabilities)
                / len(selected_probabilities)
                if selected_probabilities
                else None
            ),

            "mean_aux_balance_value": (
                sum(aux_values) / len(aux_values)
                if aux_values
                else None
            ),

            "routing_definition": {
                "routing_unit": (
                    "expert assignments; for K>1 each token "
                    "requests K assignments"
                ),

                "configured_top_k": (
                    configured_top_ks[0]
                    if len(configured_top_ks) == 1
                    else configured_top_ks
                ),

                "effective_top_k": (
                    effective_top_ks[0]
                    if len(effective_top_ks) == 1
                    else effective_top_ks
                ),

                "top1_capacity": (
                    "capacity_factor * "
                    "ceil(tokens / num_experts)"
                ),

                "topk_capacity": (
                    "capacity_factor * "
                    "ceil((tokens * effective_top_k) "
                    "/ num_experts)"
                ),

                "top1_overflow_policy": (
                    "keep highest raw top-1 gate weights "
                    "up to capacity"
                ),

                "topk_overflow_policy": (
                    "renormalize selected top-k gates per "
                    "token, then keep highest renormalized "
                    "assignment weights for each expert "
                    "up to capacity"
                ),

                "normalized_entropy": (
                    "entropy / log(num_experts)"
                ),

                "load_imbalance_ratio": (
                    "maximum requested assignment fraction "
                    "/ ideal expert fraction"
                ),

                "aux_balance_value": (
                    "matches Block.forward(): "
                    "num_experts * sum(mean router "
                    "probability * normalized selected "
                    "assignment fraction)"
                ),
            },
        }

        summary_path = self.output_dir / "routing_summary.json"

        with open(
            summary_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(summary, f, indent=2)

        print()
        print("=" * 80)
        print("ROUTING DIAGNOSTICS COMPLETE")
        print("=" * 80)
        print(f"Summary: {summary_path}")
        print(f"Layers : {layer_path}")
        print(f"Experts: {expert_path}")
        print("=" * 80)
