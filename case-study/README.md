# Phase F: Controlled AI-Evaluation Case Study

> Empirical validation of the Stackelberg deterrence frontier $B\sigma + \delta F \geq c$ via controlled benchmark contamination experiments. Adds §8 *Controlled Case Study* to the AIJ paper; the synthetic experiments in the parent directory move to Appendix B.

## What this package adds

The parent replication package (`../`) reproduces the paper's synthetic experiments under the model's own primitives. **This subdirectory adds an independent controlled experiment** that estimates the model's primitives from real models and benchmarks, then tests the predictive theorems.

| Phase F deliverable | Code | Output |
|---|---|---|
| Joint identification of $\sigma$, $\widehat\alpha^\*$ from controlled gaming | `src/phase_f/identify.py` | `results/scores/sigma_alpha_per_model.csv` |
| Detector $\delta$ across 4 detector classes | `src/phase_f/detectors/` | `results/detectors/per_detector_delta.csv` |
| Frontier test: gaming crossing near $B\sigma + \delta F = c$ | `src/phase_f/tests/frontier.py` | `out/figures/frontier_crossing.pdf` |
| Fine-threshold test: transition near $F^\* = (c - B\sigma)/\delta$ | `src/phase_f/tests/fine_threshold.py` | `out/figures/fine_threshold.pdf` |
| Portfolio reversal test (regime b) | `src/phase_f/tests/portfolio.py` | `out/figures/portfolio_reversal.pdf` |
| Correlated-portfolio divergence as $\rho \to 1$ | `src/phase_f/tests/correlated.py` | `out/figures/correlated_divergence.pdf` |
| Policy comparison (model-selected vs baselines) | `src/phase_f/tests/policy.py` | `out/tables/policy_comparison.tex` |

## Model panel (8 open-weight + 1 closed-API anchor)

Fine-tuned via LoRA across 7 developer strategies each:

| Tier | Model | Released | Method |
|---|---|---|---|
| Small ~4B | `microsoft/Phi-4-mini-instruct` | Feb 2025 | LoRA |
| Mid-8B | `meta-llama/Llama-3.1-8B-Instruct` | Jul 2024 | LoRA |
| Mid-8B | `Qwen/Qwen3-8B` | May 2025 | LoRA |
| Mid-8B | `mistralai/Ministral-8B-Instruct-2410` | Oct 2024 | LoRA + 1 full-FT ablation |
| Large-14B | `Qwen/Qwen3-14B` | May 2025 | LoRA |
| Large-27B | `google/gemma-3-27b-it` | Mar 2025 | LoRA (gradient checkpointing) |
| Large-32B | `Qwen/Qwen3-32B` | May 2025 | LoRA (gradient checkpointing) |
| Frontier-70B | `meta-llama/Llama-3.3-70B-Instruct` | Dec 2024 | QLoRA 4-bit, smoke test first |
| API anchor | Claude Haiku 4.5 (pilot) → Sonnet 4.6 (escalation) | n/a | API only, σ-coupling |

## Developer strategies (7 fine-tune conditions per model)

1. **Baseline** — standard instruct fine-tune, no contamination
2. **Light contamination** — 1% MMLU dev items leaked into training data
3. **Moderate contamination** — 10% MMLU dev items leaked
4. **Heavy contamination** — 50% MMLU dev items leaked
5. **Paraphrase contamination** — paraphrased MMLU items leaked (tests $\sigma$ robustness)
6. **Decontamination effort** — filtered training corpus, zero MMLU leakage by construction
7. **Portfolio-aware gaming** — leakage targeting multiple public metrics simultaneously

Ground truth (which strategy / how much contamination) is known by construction for every fine-tune.

## Detectors (4 classes)

Per locked scoping decision 5 of the parent plan:

1. **TS-Guessing** exact-match probe (Deng et al., 2024 protocol)
2. **N-gram LCS** detector (longest common substring against benchmark items)
3. **Embedding-similarity** detector (sentence-transformers vs benchmark)
4. **Behavioral consistency** probe (paraphrase-perturbed agreement rate)

Reported: per-detector $\delta$ estimate (true positive at fixed 5% FPR) + framework-sensitivity of frontier-test conclusion to detector choice.

## Compute target

- **HPC:** Warwick SCRTP Avon (SSH alias `warwick-avon`, account `wbs`, QoS `normal`).
- **GPU partition:** 16 nodes × 3 Quadro RTX 6000 (24 GB each = 72 GB/node).
- **Budget:** ~440–500 GPU-hours total (estimated 6–8 weeks at typical queue throughput).
- **Storage:** ~200 GB in GPFS home (`/home/wbs/bsthbr/models/`).
- **Slurm conventions:** per `~/Task-Management/rules/hpc-job-monitoring.md` — Monitor stream on state transitions, never `sleep`.

## Quickstart

```bash
# 1. Local development (Mac Mini / MacBook)
cd github-repo/case-study
uv sync                                                  # creates .venv
uv run python -m phase_f.cli inspect-models             # sanity-check the model panel

# 2. On Avon (after model downloads complete)
ssh warwick-avon
cd ~/research/robust-evaluation-design/github-repo/case-study
sbatch scripts/slurm/finetune_array.sbatch              # 7 strategies × 1 model
```

See `scripts/slurm/` for sbatch templates, `configs/` for per-experiment configs, `notebooks/` for analysis.

## License + data sources

- **Code:** MIT (see parent `LICENSE`)
- **MMLU:** MIT (Hendrycks et al., 2021)
- **MMLU-CF:** CC-BY-4.0 (Zhao et al., 2024)
- **TS-Guessing protocol:** reimplemented from Deng et al., 2024 (paper protocol; no code dependency)

## Status

| Phase | Status | Date |
|---|---|---|
| F.1.0 Avon scoping | Complete | 2026-05-15 |
| F.1.1 Model downloads | In progress (Slurm job 2000762) | 2026-05-15 |
| F.1.2 Benchmark + detector scaffold | In progress | 2026-05-15 |
| F.1.3 Detector implementations | Pending | — |
| F.1.4 Fine-tune scaffold | Pending | — |
| F.1.5 Smoke-test fine-tune | Pending | — |
| F.2 Primitive estimation | Pending | — |
| F.3 Predictive tests | Pending | — |
| F.4 Manuscript integration | Pending | — |
