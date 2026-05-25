#!/bin/bash
#SBATCH --job-name=detect-mid
#SBATCH --partition=gpu
#SBATCH --account=wbs
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:quadro_rtx_6000:2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/wbs/bsthbr/jobs/phase-f/logs/%x-%j.out
#SBATCH --error=/home/wbs/bsthbr/jobs/phase-f/logs/%x-%j.err

# Run all 4 detectors on a mid-tier (Qwen3-14B) adapter.

set -euo pipefail
: "${MODEL_NICK:?}"
: "${STRATEGY_NICK:?}"
: "${ADAPTER_DIR:?}"

PHASE_F_REPO="${PHASE_F_REPO:-$HOME/research/robust-evaluation-design/github-repo/case-study}"
cd "$PHASE_F_REPO"
module purge
module load GCCcore/11.3.0 Python/3.10.4 CUDA/12.1.1
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || python3 -m pip install --user --quiet uv
export AVON_MODELS_DIR="${AVON_MODELS_DIR:-/home/wbs/bsthbr/models}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date)] detect-mid  model=$MODEL_NICK  strategy=$STRATEGY_NICK  job=${SLURM_JOB_ID:-}"
uv sync --quiet
uv run phase-f detect --model "$MODEL_NICK" --strategy "$STRATEGY_NICK" --adapter "$ADAPTER_DIR" --eval-subset-size 200
echo "[$(date)] Done"
