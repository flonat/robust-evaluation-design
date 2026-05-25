#!/bin/bash
#SBATCH --job-name=eval-mid
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

# Eval mid-tier adapters (8B-14B FP16) on all 3 benchmarks.

set -euo pipefail
: "${MODEL_NICK:?MODEL_NICK env var required}"
: "${STRATEGY_NICK:?STRATEGY_NICK env var required}"
: "${ADAPTER_DIR:?ADAPTER_DIR env var required (path to LoRA adapter)}"

PHASE_F_REPO="${PHASE_F_REPO:-$HOME/research/robust-evaluation-design/github-repo/case-study}"
cd "$PHASE_F_REPO"
module purge
module load GCCcore/11.3.0 Python/3.10.4 CUDA/12.1.1
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || python3 -m pip install --user --quiet uv
export AVON_MODELS_DIR="${AVON_MODELS_DIR:-/home/wbs/bsthbr/models}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date)] eval-mid  node=$(hostname)  model=$MODEL_NICK  strategy=$STRATEGY_NICK  adapter=$ADAPTER_DIR  job=${SLURM_JOB_ID:-}"
uv sync --quiet

for BENCHMARK in mmlu-public mmlu-cf-hidden mmlu-paraphrased; do
    echo
    echo "[$(date)] === $BENCHMARK ==="
    uv run phase-f score \
        --model "$MODEL_NICK" \
        --strategy "$STRATEGY_NICK" \
        --adapter "$ADAPTER_DIR" \
        --benchmark "$BENCHMARK" \
        --subset-size 1000
done

echo "[$(date)] Done"
