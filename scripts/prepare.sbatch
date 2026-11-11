#!/usr/bin/env bash
# Pre-tokenize FineWeb-Edu on a compute node. All arguments go straight to prepare.py:
#
#   mkdir -p logs
#   sbatch scripts/prepare.sbatch --output s3://<bucket>/fineweb-edu-llama3 --token-budget 30000000000
#
# prepare.py skips finished shards, so if the 4h dev limit runs out, just submit it again.
#SBATCH --job-name=prepare
#SBATCH --account=ACD115198
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# One tokenizer process per core; keep BLAS/OpenMP from spawning more threads on top
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# S3 keys for reading flock-datasets and writing the shard store
eval "$(flock s3 credentials --ttl 24h)"

echo "job $SLURM_JOB_ID on $(hostname), $SLURM_CPUS_PER_TASK CPUs, args: $*"

srun uv run --frozen python -u prepare.py --workers "$SLURM_CPUS_PER_TASK" "$@"
