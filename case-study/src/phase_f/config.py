"""Central config: paths, seeds, model panel, developer strategies."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
SEED = 20260515  # Phase F seed; per-experiment seeds derive from this


# ---------------------------------------------------------------------------
# Filesystem layout (works on Mac Mini local + Avon HPC)
# ---------------------------------------------------------------------------
def _resolve_root() -> Path:
    """case-study/ root, regardless of CWD."""
    return Path(__file__).resolve().parents[2]


ROOT = _resolve_root()
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_PARAPHRASES = ROOT / "data" / "paraphrases"
RESULTS_DETECTORS = ROOT / "results" / "detectors"
RESULTS_SCORES = ROOT / "results" / "scores"
RESULTS_FINETUNES = ROOT / "results" / "finetunes"
CONFIGS = ROOT / "configs"

# Avon paths (per F.1.0 scoping report)
AVON_MODELS_DIR = Path(os.environ.get("AVON_MODELS_DIR", "/home/wbs/bsthbr/models"))


# ---------------------------------------------------------------------------
# Model panel (locked 2026-05-15 after HF currency check)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    nick: str  # short id, used in filenames
    hf_repo: str  # HuggingFace repo
    tier: str  # small / mid-8B / large-14B / large-27B / large-32B / frontier-70B
    family: str  # phi / llama / qwen / mistral / gemma
    params_billions: float
    method: str  # lora / qlora-int4
    notes: str = ""


PANEL: tuple[ModelSpec, ...] = (
    ModelSpec("phi-4-mini", "microsoft/Phi-4-mini-instruct", "small", "phi", 3.8, "lora",
              "Feb 2025"),
    ModelSpec("llama-3.1-8b", "meta-llama/Llama-3.1-8B-Instruct", "mid-8B", "llama", 8.0, "lora",
              "Jul 2024 — Meta's current dense 8B"),
    ModelSpec("qwen3-8b", "Qwen/Qwen3-8B", "mid-8B", "qwen", 8.2, "lora",
              "May 2025"),
    ModelSpec("ministral-8b", "mistralai/Ministral-8B-Instruct-2410", "mid-8B", "mistral", 8.0, "lora",
              "Oct 2024; +1 full-FT ablation on baseline strategy"),
    ModelSpec("qwen3-14b", "Qwen/Qwen3-14B", "large-14B", "qwen", 14.8, "lora",
              "May 2025"),
    ModelSpec("gemma-3-27b", "google/gemma-3-27b-it", "large-27B", "gemma", 27.0, "lora",
              "Mar 2025; gradient checkpointing on 3 GPUs"),
    ModelSpec("qwen3-32b", "Qwen/Qwen3-32B", "large-32B", "qwen", 32.8, "lora",
              "May 2025; gradient checkpointing on 3 GPUs"),
    ModelSpec("llama-3.3-70b", "unsloth/Llama-3.3-70B-Instruct-bnb-4bit", "frontier-70B", "llama", 71.0, "qlora-int4",
              "Dec 2024 (Unsloth pre-quantized NF4 build — Meta Llama 3.3 licence on meta-llama/ repo not yet accepted; switching to community Unsloth mirror keeps the science identical). Smoke test first, full 7-strategy run conditional."),
)

# Closed-API anchor — tiered (Haiku pilot, Sonnet escalation)
API_ANCHOR_PILOT = "claude-haiku-4-5"
API_ANCHOR_ESCALATION = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Developer strategies (7 fine-tune conditions per model)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StrategySpec:
    nick: str
    description: str
    contamination_rate: float  # fraction of training data that is MMLU-leaked
    paraphrased: bool  # leak via paraphrase rather than literal
    multi_metric: bool  # leak targets multiple public benchmarks


STRATEGIES: tuple[StrategySpec, ...] = (
    StrategySpec("baseline", "Standard instruct fine-tune, no contamination", 0.0, False, False),
    StrategySpec("contam-light", "1% MMLU dev items leaked", 0.01, False, False),
    StrategySpec("contam-moderate", "10% MMLU dev items leaked", 0.10, False, False),
    StrategySpec("contam-heavy", "50% MMLU dev items leaked", 0.50, False, False),
    StrategySpec("paraphrase-contam", "Paraphrased MMLU items leaked", 0.10, True, False),
    StrategySpec("decontamination", "Filtered training corpus, zero leakage", 0.0, False, False),
    StrategySpec("portfolio-gaming", "Leakage targeting multiple public metrics", 0.10, False, True),
)


# ---------------------------------------------------------------------------
# Detector classes (per locked scoping decision 5)
# ---------------------------------------------------------------------------
DETECTORS: tuple[str, ...] = (
    "ts_guessing",       # Deng 2024 exact-match probe
    "ngram_lcs",         # Longest common substring n-gram
    "embedding_sim",     # sentence-transformers similarity
    "behavioral_paraphrase",  # paraphrase-perturbed agreement
)


# ---------------------------------------------------------------------------
# Benchmark variants (the regulator's measurement options)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BenchmarkSpec:
    nick: str
    role: str  # public / hidden-clean / perturbed
    hf_dataset: str
    split: str
    subset_size: int | None = None


BENCHMARKS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec("mmlu-public", "public",
                  "cais/mmlu", "test", 1000),
    BenchmarkSpec("mmlu-cf-hidden", "hidden-clean",
                  "microsoft/MMLU-CF", "test", 1000),  # actual path TBC at F.1.2c
    BenchmarkSpec("mmlu-paraphrased", "perturbed",
                  "local:phase_f/data/paraphrases/mmlu_paraphrased.json", "test", 1000),
)
