#!/bin/bash
# Dispatch all eval jobs across the panel.
#
# Maps each (model, strategy) adapter dir to the right sbatch template:
#   - phi-4-mini + 4× mid-8B  → eval_small.sbatch   (1 GPU)
#   - qwen3-14b                → eval_mid.sbatch     (2 GPUs)
#   - gemma-3-27b + qwen3-32b  → eval_large.sbatch   (3 GPUs)
#   - llama-3.3-70b            → eval_70b.sbatch     (3 GPUs)
#
# Skips adapter dirs that look empty (no adapter_model.safetensors).
#
# Usage:
#   bash scripts/dispatch_eval.sh           # submit all 55 eval jobs
#   bash scripts/dispatch_eval.sh --dry-run # print without submitting

set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

REPO="${PHASE_F_REPO:-$HOME/research/robust-evaluation-design/github-repo/case-study}"
ADAPTERS_DIR="$REPO/results/finetunes"

declare -A TIER
TIER["phi-4-mini"]="small"
TIER["llama-3.1-8b"]="small"
TIER["qwen3-8b"]="small"
TIER["ministral-8b"]="small"
TIER["qwen3-14b"]="mid"
TIER["gemma-3-27b"]="large"
TIER["qwen3-32b"]="large"
TIER["llama-3.3-70b"]="70b"

declare -A SBATCH
SBATCH["small"]="$REPO/scripts/slurm/eval_small.sbatch"
SBATCH["mid"]="$REPO/scripts/slurm/eval_mid.sbatch"
SBATCH["large"]="$REPO/scripts/slurm/eval_large.sbatch"
SBATCH["70b"]="$REPO/scripts/slurm/eval_70b.sbatch"

submitted=0
skipped_empty=0
unknown=0

for adir in "$ADAPTERS_DIR"/*; do
    [[ -d "$adir" ]] || continue
    base=$(basename "$adir")
    model="${base%__*}"
    strategy="${base#*__}"

    # Adapter lives in <model_strategy>/adapter/. Skip if missing (failed fine-tune).
    adapter_path="$adir/adapter"
    if [[ ! -f "$adapter_path/adapter_model.safetensors" ]]; then
        echo "[skip-empty] $base (no adapter/adapter_model.safetensors)"
        skipped_empty=$((skipped_empty + 1))
        continue
    fi

    tier="${TIER[$model]:-}"
    if [[ -z "$tier" ]]; then
        echo "[unknown-model] $base"
        unknown=$((unknown + 1))
        continue
    fi
    template="${SBATCH[$tier]}"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "would submit: $model $strategy (tier=$tier) adapter=$adapter_path"
    else
        jid=$(sbatch --parsable \
            --export=ALL,MODEL_NICK=$model,STRATEGY_NICK=$strategy,ADAPTER_DIR=$adapter_path \
            "$template")
        echo "$jid  $model  $strategy  (tier=$tier)"
    fi
    submitted=$((submitted + 1))
done

echo
echo "Summary: submitted=$submitted, skipped-empty=$skipped_empty, unknown=$unknown"
