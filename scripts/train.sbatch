#!/usr/bin/env bash
# Train on one H200. All arguments go straight to train.py:
#
#   mkdir -p logs
#   sbatch --job-name=150M-baseline --time=12:00:00 scripts/train.sbatch \
#       --data s3://<bucket>/fineweb-edu-llama3 --model-size 150M \
#       --total-tokens 30000000000 --run-name 150M-baseline --compile
#
# H200-hours = GPUs x wall time, so --time is what bounds the grant a run can use.
#SBATCH --job-name=train
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=H200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

module purge
module load cuda/13.0

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=4
export WANDB_MODE="${WANDB_MODE:-online}"

# The data loader reads shards from S3 while training
eval "$(flock s3 credentials --ttl 48h)"

echo "job $SLURM_JOB_ID on $(hostname), GPU $CUDA_VISIBLE_DEVICES, commit $(git rev-parse --short HEAD)"
echo "args: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun uv run --frozen python -u train.py "$@"
