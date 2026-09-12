#!/usr/bin/env bash
#SBATCH --job-name=transforming-lhcb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=train-%j.out
# Set your cluster's partition/account and activate the environment before submission.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
torchrun --standalone --nproc-per-node=8 python/train.py \
  --config config/training.yaml --train outputs/ml/signed_pid_v1/train.json \
  --val outputs/ml/signed_pid_v1/validation.json \
  --output outputs/checkpoints/mc --device cuda --workers 5 "$@"
