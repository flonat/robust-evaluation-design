#!/bin/bash
# Common setup sourced by all Phase F sbatch templates.
# Sets up the Python env, validates required env vars, and prepares logging.

set -euo pipefail

# --- Required environment variables ---
: "${MODEL_NICK:?MODEL_NICK env var required (e.g. qwen3-8b)}"
: "${STRATEGY_NICK:?STRATEGY_NICK env var required (e.g. baseline)}"

# Phase F repo location on Avon
PHASE_F_REPO="${PHASE_F_REPO:-$HOME/research/robust-evaluation-design/github-repo/case-study}"

if [ ! -d "$PHASE_F_REPO" ]; then
    echo "ERROR: PHASE_F_REPO not found at $PHASE_F_REPO" >&2
    echo "Either clone the repo there, or set PHASE_F_REPO to the correct path." >&2
    exit 1
fi

cd "$PHASE_F_REPO"

# --- Module + uv setup ---
module purge
module load GCCcore/11.3.0 Python/3.10.4 CUDA/12.4.0

# uv should be on PATH (installed once: pip install --user uv)
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] installing uv to ~/.local/bin"
    python3 -m pip install --user --quiet uv
fi

# Sync deps (no-op if up to date)
uv sync --quiet 2>&1 | tail -5 || true

# Where base models live (set in main CLAUDE.md HPC pointer)
export AVON_MODELS_DIR="${AVON_MODELS_DIR:-/home/wbs/bsthbr/models}"

# HF token for any auth-gated downloads inside the job
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
    export HF_TOKEN
fi

echo "===================="
echo "Phase F fine-tune"
echo "===================="
echo "Time:        $(date)"
echo "Node:        $(hostname)"
echo "GPUs:        ${SLURM_GPUS_ON_NODE:-?}"
echo "CUDA:        ${CUDA_VISIBLE_DEVICES:-?}"
echo "Model:       $MODEL_NICK"
echo "Strategy:    $STRATEGY_NICK"
echo "JobID:       ${SLURM_JOB_ID:-(none)}"
echo "Repo:        $PHASE_F_REPO"
echo "Models dir:  $AVON_MODELS_DIR"
echo "===================="
