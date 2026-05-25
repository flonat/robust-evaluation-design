#!/bin/bash
#SBATCH --job-name=eval-70b
#SBATCH --partition=gpu
#SBATCH --account=wbs
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:quadro_rtx_6000:3
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/home/wbs/bsthbr/jobs/phase-f/logs/%x-%j.out
#SBATCH --error=/home/wbs/bsthbr/jobs/phase-f/logs/%x-%j.err

# Eval Llama-3.3-70B QLoRA NF4 adapters on all 3 benchmarks.
# Pre-quantized checkpoint at AVON_MODELS_DIR/Llama-3.3-70B-Instruct-bnb-4bit.

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

echo "[$(date)] eval-70b  node=$(hostname)  model=$MODEL_NICK  strategy=$STRATEGY_NICK  adapter=$ADAPTER_DIR  job=${SLURM_JOB_ID:-}"
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
