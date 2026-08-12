# JetClass-II Pilot Results

Date: 2026-07-20

## Shared configuration

- Dataset: JetClass-II Pythia
- Classes: 188
- Train files per group: 10
- Validation files per group: 2
- Train samples per epoch: 204,800
- Validation samples per epoch: 51,200
- Epochs: 2
- Batch size: 512
- Workers: 4
- AMP: enabled

## Architectures

### Dense ParT

- Parameters: 2.3M
- FLOPs: 669.84M
- Run: `jc2-pilot-part-fast-clean`
- Peak CUDA memory: 7,046.8 MB

### MoEParT

- Experts: 4
- Top-k: 1
- Capacity factor: 1.5
- Auxiliary-loss coefficient: 0.01
- Router jitter: 0.01
- Parameters: 6.3M
- FLOPs: 503.12M
- Run: `jc2-pilot-mpt-e4-k1-fast`
- Peak CUDA memory: 13,614.0 MB

## Epoch results

| Model | Epoch | Train loss | Train acc. | Train entries/s | Val. loss | Val. acc. | Val. entries/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| ParT | 0 | 3.81949 | 0.17112 | 1607.7 | 2.97766 | 0.24563 | 978.9 |
| ParT | 1 | 3.10724 | 0.23173 | 2006.6 | 2.88859 | 0.25623 | 1279.8 |
| MoEParT | 0 | 3.87601 | 0.17041 | 1250.0 | 2.96828 | 0.25023 | 944.3 |
| MoEParT | 1 | 3.09173 | 0.23598 | 1453.2 | 2.91063 | 0.26531 | 1216.5 |

## Pilot comparison

- MoEParT validation accuracy was 0.00908 higher.
- MoEParT used about 1.93x as much GPU memory.
- MoEParT had about 24.9% fewer reported FLOPs.
- MoEParT had about 2.74x as many total parameters.
- Dense ParT had higher measured training throughput.
- Both models trained normally and improved across epochs.
- These two-epoch results are preliminary.

## Validation metric warning

The six validation files collectively contain all 188 classes:

- Res2P: labels 0-14
- Res34P: labels 15-160
- QCD: labels 161-187

However, the 51,200 sampled validation entries did not include every
class, so multiclass ROC AUC could not be calculated. Validation loss
and accuracy are still usable.

## Checkpoints

Dense ParT:

`/kaushik-moe-vol/outputs/training/JetClassII/Pythia/full/ParT/jc2-pilot-part-fast-clean/net_best_epoch_state.pt`

MoEParT:

`/kaushik-moe-vol/outputs/training/JetClassII/Pythia/full/MPT/jc2-pilot-mpt-e4-k1-fast/net_best_epoch_state.pt`

## Next experiment

Run a matched medium comparison:

- 1,000,000 train samples per epoch
- 200,000 validation samples per epoch
- 3 epochs
- batch size 512
- 4 workers
- identical data settings
- same GPU model when comparing throughput
