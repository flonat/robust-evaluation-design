#!/bin/bash
#SBATCH --job-name=eval-small
#SBATCH --partition=gpu
#SBATCH --account=wbs
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/wbs/bsthbr/jobs/phase-f/logs/%x-%j.out
#SBATCH --error=/home/wbs/bsthbr/jobs/phase-f/logs/%x-%j.err

# Eval a small-tier adapter on all 3 benchmarks.
# Env: MODEL_NICK, STRATEGY_NICK, ADAPTER_DIR (path on Avon)

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

echo "[$(date)] eval-small  node=$(hostname)  model=$MODEL_NICK  strategy=$STRATEGY_NICK  adapter=$ADAPTER_DIR  job=${SLURM_JOB_ID:-}"
uv sync --quiet
uv run python -c "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))"

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
