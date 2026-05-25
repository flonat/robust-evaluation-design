#!/bin/bash
#SBATCH --job-name=ft-27-32b
#SBATCH --partition=gpu
#SBATCH --account=wbs
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:quadro_rtx_6000:3
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/wbs/bsthbr/jobs/phase-f/logs/%x-%j.out
#SBATCH --error=/home/wbs/bsthbr/jobs/phase-f/logs/%x-%j.err

# Large 27-32B (Gemma-3-27B, Qwen3-32B). Full node: 3 GPUs (72 GB VRAM), gradient checkpointing.

set -euo pipefail
: "${MODEL_NICK:?MODEL_NICK env var required}"
: "${STRATEGY_NICK:?STRATEGY_NICK env var required}"

PHASE_F_REPO="${PHASE_F_REPO:-$HOME/research/robust-evaluation-design/github-repo/case-study}"
cd "$PHASE_F_REPO"
module purge
module load GCCcore/11.3.0 Python/3.10.4 CUDA/12.1.1
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || python3 -m pip install --user --quiet uv
export AVON_MODELS_DIR="${AVON_MODELS_DIR:-/home/wbs/bsthbr/models}"
# Reduce CUDA fragmentation on 32B (OOM-fix 2026-05-16, job 2001200)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date)] Phase F large 27-32B  node=$(hostname)  model=$MODEL_NICK  strategy=$STRATEGY_NICK  job=${SLURM_JOB_ID:-}"
uv sync --quiet
uv run python -c "import torch; assert torch.cuda.is_available(); print('GPU count:', torch.cuda.device_count())"
uv run python -m phase_f.finetune.train --model "$MODEL_NICK" --strategy "$STRATEGY_NICK"
echo "[$(date)] Done"
