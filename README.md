# Transforming LHCb

MC preprocessing and self-supervised transformer training used in the paper.
This repository does not include exclusive-decay reconstruction, collision-data
analysis, flavour-probe evaluation or plotting workflows.

## Code

| File | Responsibility |
| --- | --- |
| `python/prepare.py` | ROOT to NPZ shards, event splits and manifests |
| `python/features.py` | Feature names, array transformations and train-only normalization |
| `python/ml_rdf_helpers.py` | Compiled constituent and PID-blind geometry calculations |
| `python/pid_targets.py` | Reconstructed particle-ID targets |
| `python/dataset.py` | Shard loading, padding and masks |
| `python/corruption.py` | Particle deletion and residual-jet rebuilding |
| `python/model.py` | Encoder, completion decoder and prediction heads |
| `python/losses.py` | Masked-PID loss, query matching and missing-particle losses |
| `python/runtime.py` | Distributed reductions, precision and model configuration |
| `python/train.py` | Training loop, validation, class census and checkpoints |

## Setup

Use Python 3.10 or newer, PyTorch and a working PyROOT installation.
Install the Python dependencies with `pip install -r requirements.txt`.
ROOT is installed separately. The original environment used PyTorch 2.5.1
and ROOT 6.38.04.

Download the MC ROOT files from CERN Open Data record 4910
(DOI: 10.7483/OPENDATA.LHCB.N75T.TJPE). Set their local paths in
`config/dataset.yaml`, preserving the file order because it determines source IDs.
No data or checkpoints are bundled.

## Preprocess

Run commands from the repository root.

```bash
python python/prepare.py --dataset-config config/dataset.yaml \
  --transformer-config config/features.yaml \
  --output outputs/ml/signed_pid_v1 --threads 1
```

`features.yaml` fixes the input representation and event-level split
(65% training, 15% validation, 10% test, 10% analysis). Scaling and PID binning
are fitted on training jets only. Geometric track relations are retained because
they enter the representation and corruption targets. No exclusive decays are built.

## Train

```bash
torchrun --standalone --nproc-per-node=8 python/train.py \
  --config config/training.yaml \
  --train outputs/ml/signed_pid_v1/train.json \
  --val outputs/ml/signed_pid_v1/validation.json \
  --output outputs/checkpoints/mc --device cuda --workers 5
```

The paper run used eight GPUs, batch size 128 per GPU (1024 total), bf16,
955742 training jets and 220662 validation jets. Changing the effective batch
size changes the optimization schedule. An example Slurm job is provided in
`jobs/train.slurm`.

`training.yaml` specifies the architecture, deletions and loss weights.
In `losses.py`, `compute_objective` calculates surviving-particle masked PID,
`match_missing_queries` assigns deleted particles to completion queries, and
`compute_missing_objective` combines count, type, summary, kinematic, PID and
topology terms. Deletions are resampled each training epoch and fixed for validation.
Generator flavour is metadata, not pretraining supervision.

Training writes `best.pt`, `last.pt`, history and configuration/census records.
Validation is part of the training loop. Exact numerical reproduction also depends
on input ordering, software versions, accelerator kernels and distributed batching.
