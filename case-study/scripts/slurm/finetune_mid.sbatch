#!/bin/bash
#SBATCH --job-name=ft-mid
#SBATCH --partition=gpu
#SBATCH --account=wbs
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/wbs/bsthbr/jobs/phase-f/logs/%x-%j.out
#SBATCH --error=/home/wbs/bsthbr/jobs/phase-f/logs/%x-%j.err

# Mid-tier 7-9B (Llama-3.1-8B, Qwen3-8B, Ministral-8B). 1 GPU, 4h walltime.

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

echo "[$(date)] Phase F mid-tier  node=$(hostname)  model=$MODEL_NICK  strategy=$STRATEGY_NICK  job=${SLURM_JOB_ID:-}"
uv sync --quiet
uv run python -c "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))"
uv run python -m phase_f.finetune.train --model "$MODEL_NICK" --strategy "$STRATEGY_NICK"
echo "[$(date)] Done"
