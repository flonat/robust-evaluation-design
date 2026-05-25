#!/bin/bash
#SBATCH --job-name=smoke-70b
#SBATCH --partition=gpu-devel
#SBATCH --account=wbs
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:quadro_rtx_6000:3
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --output=/home/wbs/bsthbr/jobs/phase-f/logs/%x-%j.out
#SBATCH --error=/home/wbs/bsthbr/jobs/phase-f/logs/%x-%j.err

# 70B QLoRA SMOKE TEST. Pass criteria: loads + 20 steps without OOM + adapter saves.

set -euo pipefail
export MODEL_NICK="${MODEL_NICK:-llama-3.3-70b}"
export STRATEGY_NICK="${STRATEGY_NICK:-baseline}"

PHASE_F_REPO="${PHASE_F_REPO:-$HOME/research/robust-evaluation-design/github-repo/case-study}"
cd "$PHASE_F_REPO"
module purge
module load GCCcore/11.3.0 Python/3.10.4 CUDA/12.1.1
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || python3 -m pip install --user --quiet uv
export AVON_MODELS_DIR="${AVON_MODELS_DIR:-/home/wbs/bsthbr/models}"
export PHASE_F_SMOKE_MAX_STEPS=20

echo "[$(date)] Phase F 70B SMOKE  node=$(hostname)  job=${SLURM_JOB_ID:-}"
uv sync --quiet
uv run python -c "import torch; assert torch.cuda.is_available(); print('GPU count:', torch.cuda.device_count())"
uv run python -m phase_f.finetune.train --model "$MODEL_NICK" --strategy "$STRATEGY_NICK"
echo "[$(date)] 70B smoke test done"
