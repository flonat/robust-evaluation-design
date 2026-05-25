#!/bin/bash
# Dispatch all detector jobs across the panel (mirrors dispatch_eval.sh).

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
SBATCH["small"]="$REPO/scripts/slurm/detect_small.sbatch"
SBATCH["mid"]="$REPO/scripts/slurm/detect_mid.sbatch"
SBATCH["large"]="$REPO/scripts/slurm/detect_large.sbatch"
SBATCH["70b"]="$REPO/scripts/slurm/detect_70b.sbatch"

submitted=0
skipped_empty=0

for adir in "$ADAPTERS_DIR"/*; do
    [[ -d "$adir" ]] || continue
    base=$(basename "$adir")
    model="${base%__*}"
    strategy="${base#*__}"

    adapter_path="$adir/adapter"
    if [[ ! -f "$adapter_path/adapter_model.safetensors" ]]; then
        echo "[skip-empty] $base"
        skipped_empty=$((skipped_empty + 1))
        continue
    fi

    tier="${TIER[$model]:-}"
    template="${SBATCH[$tier]}"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "would submit: $model $strategy (tier=$tier)"
    else
        jid=$(sbatch --parsable \
            --export=ALL,MODEL_NICK=$model,STRATEGY_NICK=$strategy,ADAPTER_DIR=$adapter_path \
            "$template")
        echo "$jid  $model  $strategy  (tier=$tier)"
    fi
    submitted=$((submitted + 1))
done

echo
echo "Summary: submitted=$submitted, skipped-empty=$skipped_empty"
