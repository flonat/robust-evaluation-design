#!/bin/bash
#SBATCH --job-name=ft-small
#SBATCH --partition=gpu
#SBATCH --account=wbs
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/wbs/bsthbr/jobs/phase-f/logs/%x-%j.out
#SBATCH --error=/home/wbs/bsthbr/jobs/phase-f/logs/%x-%j.err

# Small-tier fine-tune (Phi-4-mini). Single GPU, 2-hour walltime.
# Required env: MODEL_NICK=phi-4-mini, STRATEGY_NICK=<one of 7 strategies>

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

echo "===================="
echo "Phase F fine-tune (small)"
echo "===================="
echo "Time:        $(date)"
echo "Node:        $(hostname)"
echo "GPU(s):      ${SLURM_GPUS_ON_NODE:-?}"
echo "CUDA env:    ${CUDA_VISIBLE_DEVICES:-?}"
echo "Model:       $MODEL_NICK"
echo "Strategy:    $STRATEGY_NICK"
echo "JobID:       ${SLURM_JOB_ID:-(none)}"
echo "Repo:        $PHASE_F_REPO"
echo "===================="

# Sync deps (resolves to CUDA 12.1 torch wheels via pyproject.toml)
uv sync --quiet

# GPU sanity check BEFORE training — fail fast if CUDA not visible
uv run python -c "
import torch
print('torch:', torch.__version__, 'cuda available:', torch.cuda.is_available())
assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE — check module load + torch wheel'
print('device 0:', torch.cuda.get_device_name(0))
"

uv run python -m phase_f.finetune.train \
    --model "$MODEL_NICK" \
    --strategy "$STRATEGY_NICK"

echo "[$(date)] Done"
