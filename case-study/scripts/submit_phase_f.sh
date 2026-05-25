#!/bin/bash
# Submit the full Phase F fine-tune sweep.
#
# Usage:
#   ./submit_phase_f.sh                  # submit all 56 fine-tunes + 1 ablation
#   ./submit_phase_f.sh --smoke-only     # submit just the 70B smoke test
#   ./submit_phase_f.sh --model qwen3-8b # submit all 7 strategies for one model
#   ./submit_phase_f.sh --dry-run        # print sbatch commands without submitting
#
# Picks the right per-tier sbatch template based on the model.
# Per CLAUDE.md HPC pointer: Warwick Avon, account wbs, QoS normal.

set -euo pipefail

PHASE_F_REPO="${PHASE_F_REPO:-$HOME/research/robust-evaluation-design/github-repo/case-study}"
SCRIPT_DIR="$PHASE_F_REPO/scripts/slurm"
LOG_DIR="$HOME/jobs/phase-f/logs"
mkdir -p "$LOG_DIR"

# Model nick → sbatch template
declare -A TIER_TEMPLATE=(
    [phi-4-mini]="finetune_small.sbatch"
    [llama-3.1-8b]="finetune_mid.sbatch"
    [qwen3-8b]="finetune_mid.sbatch"
    [ministral-8b]="finetune_mid.sbatch"
    [qwen3-14b]="finetune_large14b.sbatch"
    [gemma-3-27b]="finetune_large27b32b.sbatch"
    [qwen3-32b]="finetune_large27b32b.sbatch"
    [llama-3.3-70b]="finetune_70b_qlora.sbatch"
)

# 7 strategies (must match phase_f.config.STRATEGIES)
STRATEGIES=(baseline contam-light contam-moderate contam-heavy paraphrase-contam decontamination portfolio-gaming)

# All 8 models (must match phase_f.config.PANEL)
ALL_MODELS=(phi-4-mini llama-3.1-8b qwen3-8b ministral-8b qwen3-14b gemma-3-27b qwen3-32b llama-3.3-70b)

# Parse args
DRY_RUN=false
SMOKE_ONLY=false
SINGLE_MODEL=""
INCLUDE_70B=false      # gated on smoke test pass; default off

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --smoke-only) SMOKE_ONLY=true; shift ;;
        --model) SINGLE_MODEL="$2"; shift 2 ;;
        --include-70b) INCLUDE_70B=true; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

submit_one() {
    local model="$1" strategy="$2" template
    template="${TIER_TEMPLATE[$model]:-}"
    if [ -z "$template" ]; then
        echo "ERROR: no template for model '$model'" >&2
        return 1
    fi
    local cmd="sbatch --export=ALL,MODEL_NICK=$model,STRATEGY_NICK=$strategy $SCRIPT_DIR/$template"
    if [ "$DRY_RUN" = true ]; then
        echo "  DRY: $cmd"
    else
        local out
        out=$($cmd)
        local jobid
        jobid=$(echo "$out" | awk '{print $NF}')
        echo "  $model × $strategy → $jobid"
    fi
}

if [ "$SMOKE_ONLY" = true ]; then
    echo "Submitting 70B smoke test only..."
    if [ "$DRY_RUN" = true ]; then
        echo "  DRY: sbatch --export=ALL,MODEL_NICK=llama-3.3-70b,STRATEGY_NICK=baseline $SCRIPT_DIR/smoketest_70b.sbatch"
    else
        sbatch --export=ALL,MODEL_NICK=llama-3.3-70b,STRATEGY_NICK=baseline "$SCRIPT_DIR/smoketest_70b.sbatch"
    fi
    exit 0
fi

# Determine model list
if [ -n "$SINGLE_MODEL" ]; then
    MODELS=("$SINGLE_MODEL")
else
    MODELS=()
    for m in "${ALL_MODELS[@]}"; do
        if [ "$m" = "llama-3.3-70b" ] && [ "$INCLUDE_70B" != "true" ]; then
            echo "SKIP llama-3.3-70b (use --include-70b after smoke test passes)"
            continue
        fi
        MODELS+=("$m")
    done
fi

# Submit all (model, strategy) pairs
echo "=== Submitting Phase F sweep ==="
total=0
for model in "${MODELS[@]}"; do
    echo
    echo "[$model]"
    for strategy in "${STRATEGIES[@]}"; do
        submit_one "$model" "$strategy"
        total=$((total + 1))
    done
done

# Ablation: full FT on Ministral-8B baseline (separate marker — TODO wire to a different sbatch)
echo
echo "[ablation: Ministral-8B full-FT (baseline)]"
echo "  TODO: wire to finetune_mid_fullft.sbatch (not yet written; F.3 follow-up)"

echo
echo "=== Total submitted: $total fine-tunes ==="
echo "Monitor via: squeue -u \$USER"
