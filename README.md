# MoEParT: Mixture-of-Experts Particle Transformer

Code and experiment configurations for

> **Conditional Capacity and Routing in Mixture-of-Experts Particle Transformers**
> Machine Learning and the Physical Sciences (ML4PS) Workshop 2026 · [arXiv:2610.02701](https://arxiv.org/abs/2610.02701)

This repository compares a dense **Particle Transformer (ParT)** with **MoEParT**, a ParT whose feed-forward sublayers are replaced by sparse mixture-of-experts (MoE) layers, on 188-class jet classification with [JetClass-II](https://arxiv.org/abs/2405.12972). The study separates four things that are often conflated when evaluating sparse models: stored parameter capacity (number of experts *E*), active computation (experts per token *K*), routing capacity (capacity factor *c*), and how routing is organized.

## ParT vs MoEParT

Both models share the same backbone: 17 per-particle input features, embedding widths (128, 512, 128), pair embedding (64, 64, 64), 8 attention heads, 8 particle-attention blocks, 2 class-attention blocks, and a classifier with one 512-unit hidden layer and 188 outputs.

| | Dense ParT | MoEParT |
| --- | --- | --- |
| Feed-forward sublayer | one FFN, 128 → 512 → 128, GELU | *E* expert FFNs (each 128 → 512 → 128), in all 10 blocks |
| Per-particle computation | every particle uses the FFN | a learned router sends each particle to its top-*K* experts |
| Network config | [`model/ParT.py`](model/ParT.py) (wraps `weaver`'s `ParticleTransformer`) | [`model/MPT.py`](model/MPT.py) → [`model/MoETransformer.py`](model/MoETransformer.py) |

MoE options, passed to the training script as `--network-option <name> <value>`:

| Option | Meaning | Default in `model/MPT.py` |
| --- | --- | --- |
| `moe_num_experts` | number of experts *E* per MoE layer | 8 |
| `moe_top_k` | experts each particle is sent to (*K*); for *K* > 1 the selected weights are renormalized | 2 |
| `moe_capacity_factor` | expert capacity *c*; assignments beyond capacity are dropped. Set *c* ≥ *E*/*K* for "no-drop" | 1.5 |
| `moe_aux_loss_coef` | coefficient on the load-balancing loss the MoE layers compute | 0.01 |
| `moe_router_jitter` | scale of Gumbel noise added to router logits during training | 0.01 |

## Repository layout

```
train_JC2.sh                  JetClass-II training / prediction wrapper around weaver
data/JetClassII/              weaver data config (features, selection, labels, reweighting)
model/
  ParT.py                     dense ParT network config
  MPT.py                      MoEParT network config (get_model / get_loss)
  MoETransformer.py           MoE Particle Transformer implementation
  MPT_routing_eval.py         MoEParT + routing statistics collector (for analysis runs)
  MPT_interpretability_eval.py MoEParT + per-particle routing dump (for NMI analysis)
tools/
  patch_weaver_progress.py    progress-bar patch applied to weaver before training
  patch_weaver_eval.py        limits weaver's training-time eval metrics (speed)
  extract_run_metrics.py      per-epoch train/val metrics from a weaver log
  profile_run_flops.py        parameter counts and standardized forward FLOPs
  compute_qcd_rejection.py    AUROC and QCD rejection (R50, R80) from predictions
  aggregate_test_metrics.py   collects test metrics across runs into one table
  routing_collector.py        expert load, drop rate and entropy per layer
  routing_interpretability_collector.py  per-particle routing records
  analyze_routing_interpretability.py    routing NMI vs particle type, kinematics, jet class
  wandb_log_jc2_tensorboard.py           optional: copies TensorBoard metrics to W&B
jobs/                         Kubernetes job specs used for the paper's experiments
experiments/                  experiment notes
```

## Setup

Training uses [weaver-core](https://github.com/hqucms/weaver-core). The paper runs used the container image `gitlab-registry.nrp-nautilus.io/hzia/pytorch-weaver:v2.5` on single GPUs with bf16 mixed precision. To set up an environment yourself:

```bash
git clone https://github.com/kpendiyala/MPT.git
cd MPT
pip install torch weaver-core
pip install -r requirements.txt
export PYTHONPATH=$PWD   # the network configs import `model.*` and `tools.*`
```

## Data

Download JetClass-II (Pythia) parquet files ([dataset paper](https://arxiv.org/abs/2405.12972)). The paper used a subset:

| Split | Files |
| --- | --- |
| Train (30 files) | `Res2P_0000-0009`, `Res34P_0000-0009`, `QCD_0000-0009` |
| Validation (6 files) | `Res2P_0200-0201`, `Res34P_0860-0861`, `QCD_0280-0281` |
| Test (6 files) | `Res2P_0250-0251`, `Res34P_1075-1076`, `QCD_0350-0351` |

Jets are selected with 200 < *p*<sub>T</sub> < 2500 GeV and 20 < *m*<sub>SD</sub> < 500 GeV and reweighted flat in (*p*<sub>T</sub>, *m*<sub>SD</sub>) per class, as defined in [`data/JetClassII/JetClassII_full.yaml`](data/JetClassII/JetClassII_full.yaml).

`train_JC2.sh` reads files from `${DATA_PATH}/JetClassII/Pythia/`.

> **Note on versions.** The paper's experiments were run at commits `828da51` (dense and capacity-limited runs), `8300390` (no-drop and top-*K* runs) and `8b836f4` (auxiliary-loss sweep), where `train_JC2.sh --train-files-per-group 10 --val-files-per-group 2` selects the 30/6-file subset above from a flat `Pythia/` folder. On the current `main`, `train_JC2.sh` instead reads `Pythia/train/` and `Pythia/val/` and always uses a larger 134-file training subset unless `--full-dataset` is given. To reproduce the paper, check out `8300390`.

## Training

Set the environment and call `train_JC2.sh <model> full [options]`:

```bash
export DATA_PATH=/path/to/data        # contains JetClassII/Pythia/*.parquet
export OUTPUT_PATH=/path/to/outputs   # checkpoints, logs, TensorBoard
export COMMENT=nodrop-e4-k1           # run name suffix

# Dense ParT
./train_JC2.sh ParT full \
  --train-files-per-group 10 --val-files-per-group 2 \
  --num-epochs 20 --batch-size 512 --start-lr 5e-4 \
  --samples-per-epoch 1000000 --samples-per-epoch-val 600000

# MoEParT, E=4 experts, top-1, no token dropping
./train_JC2.sh MPT full \
  --train-files-per-group 10 --val-files-per-group 2 \
  --num-epochs 20 --batch-size 512 --start-lr 5e-4 \
  --samples-per-epoch 1000000 --samples-per-epoch-val 600000 \
  --network-option moe_num_experts 4 \
  --network-option moe_top_k 1 \
  --network-option moe_capacity_factor 4 \
  --network-option moe_router_jitter 0.01
```

Useful flags: `--dry-run` prints the resolved file lists and weaver command without training; `--smoke-test` runs a tiny two-epoch job. Any option the script does not recognize is passed straight to weaver.

The paper's protocol: Ranger optimizer, initial learning rate 5×10⁻⁴ with weaver's default `flat+decay` schedule, batch size 512, 10⁶ training samples per epoch, 20 epochs run uninterrupted, bf16. Resuming with `--load-epoch` changes the learning-rate schedule, so restart interrupted runs from scratch when comparing models.

Outputs go to `${OUTPUT_PATH}/training/JetClassII/Pythia/full/<model>/${COMMENT}/` (checkpoints), `${OUTPUT_PATH}/logs/<model>/` and `${OUTPUT_PATH}/tensorboard/`.

## Evaluation

**Test-set predictions** with the best-validation checkpoint:

```bash
weaver --predict \
  --data-test Res2P:$DATA_PATH/.../Res2P_0250.parquet ... \
  --data-config data/JetClassII/JetClassII_full.yaml \
  --network-config model/MPT.py \
  --model-prefix <checkpoint>/net_epoch-<best>_state.pt \
  --batch-size 512 --data-split-num 1 --use-amp --amp-dtype bf16 --gpus 0 \
  -o num_classes 188 -o fc_params "[(512,0.1)]" \
  --network-option moe_num_experts 4 --network-option moe_top_k 1 --network-option moe_capacity_factor 4 \
  --predict-output pred.parquet
```

**Metrics.** Accuracy is printed by weaver. Background rejection and AUROC:

```bash
python tools/compute_qcd_rejection.py --predictions pred.parquet --output-dir metrics/
```

*R*<sub>50</sub> and *R*<sub>80</sub> are QCD rejections (1/ε<sub>QCD</sub>) at 50% and 80% signal efficiency for the signal-vs-QCD score.

**Compute.** `tools/profile_run_flops.py` reports parameter counts and forward FLOPs for a single jet with 128 particles, measured with `torch.profiler` (fused scaled-dot-product attention is disabled during profiling so its operations are counted).

**Routing analysis.** Running prediction on validation data with `--network-config model/MPT_interpretability_eval.py` (and `INTERP_OUTPUT_DIR` set) records which expert each particle is routed to; `tools/analyze_routing_interpretability.py` then computes layer-wise normalized mutual information between expert choice and particle type, relative *p*<sub>T</sub> and jet class. `model/MPT_routing_eval.py` records expert load and drop rates.

## Experiment configurations

`jobs/experiments/` holds the Kubernetes jobs used on the [National Research Platform](https://nationalresearchplatform.org/) for each sweep in the paper:

| Folder | Sweep |
| --- | --- |
| `Dense/` | dense ParT baseline |
| `MoE/baseline_k1/` | capacity-limited top-1 (*c* = 1.5), *E* = 2, 4, 8 |
| `MoE/baseline-k1-nodrop/` | no-drop top-1, *E* = 2, 4, 8 |
| `MoE/topk/` | top-*K*: E4/K2, E8/K2, E8/K4 |
| `MoE/auxilary/` | auxiliary-loss coefficient sweep at E4/K1 |
| `MoE/capacity-factor/` | capacity-factor variant |
| `eval/` | test-set evaluation |

They mount a shared volume at `/kaushik-moe-vol`, so paths need adapting to run elsewhere.

## Citation

```bibtex
@article{pendiyala2026moepart,
  title   = {Conditional Capacity and Routing in Mixture-of-Experts Particle Transformers},
  author  = {Pendiyala, Kaushik and Zia, Haris and Lee, Trevin and Legge, Timothy and De Leon, Alejandro J. and Zhao, Zihan and Wang, Aaron and Gandrakota, Abhijith and Ngadiuba, Jennifer and Cavanaugh, Richard and Duarte, Javier},
  journal = {arXiv preprint arXiv:2610.02701},
  year    = {2026}
}
```

## Acknowledgements

The MoE Particle Transformer implementation started from Haris Zia's MoE extension of ParT. Dense ParT is from [weaver-core](https://github.com/hqucms/weaver-core) ([Qu et al., 2022](https://arxiv.org/abs/2202.03772)).
