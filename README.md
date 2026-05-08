# Replication Package — When Are AI Evaluations Gaming-Proof?

**Author:** Florian Burnat
**Paper venue:** *Artificial Intelligence* (AIJ, Elsevier)
**Paper title:** "When Are AI Evaluations Gaming-Proof? A Stackelberg Approach"

Replication code for a Stackelberg model of AI safety evaluation. The regulator
(leader) commits to a measurement policy parametrised by the triple
$(\gamma, \delta, \sigma)$ — gaming cost, detection probability, and safety
coupling — and the developer (follower) best-responds with a gaming intensity
$\alpha \in [0,1]$. The package reproduces every figure and table in the paper's
numerical-illustration section from a self-contained set of Python scripts.

## What this package reproduces

Every figure and table in the paper's numerical-illustration section is generated
by the four experiment scripts below, plus the shared analytical primitives in
`model.py`. The package is fully self-contained: clone, install with `uv`, and
run.

| Paper artefact | Script | Output file(s) |
|---|---|---|
| Figure 1 (regulator-payoff CDFs, baseline horse-race) | `experiment_baseline_horserace.py` | `out/figures/baseline_horserace.pdf` |
| Figure 2 (joint distribution of $n^{\mathrm{opt}}$ vs $n^*$) | `experiment_baseline_horserace.py` | `out/figures/portfolio_size_distribution.pdf` |
| Figure 3 (gaming-proof boundary heatmap in $(B,F)$) | `experiment_gaming_boundary.py` | `out/figures/gaming_proof_boundary.pdf` |
| Figure 4 (single-environment portfolio dominance) | `experiment_portfolio_size.py` | `out/figures/portfolio_dominance.pdf` |
| Table 1 (baseline horse-race summary) | `experiment_baseline_horserace.py` | `out/tables/baseline_horserace.tex` |
| Table 2 (comparative statics + behavioural robustness) | `experiment_comparative_statics.py` | `out/tables/comparative_statics.tex` |
| Environment-count macro | `sample_environments.py` | `out/stats/num_environments.tex` |

## Computational requirements

- **Python:** ≥ 3.11
- **Package manager:** [uv](https://docs.astral.sh/uv/) (recommended) — handles
  the venv and dependency resolution from `uv.lock`. Plain `pip` works too.
- **Dependencies:** `numpy ≥ 1.26`, `scipy ≥ 1.11`, `matplotlib ≥ 3.8`. Pinned in
  `uv.lock`.
- **Hardware:** any laptop. End-to-end re-execution of the four experiments
  takes under one minute on a 2024-class CPU.
- **OS:** macOS, Linux, or Windows. No platform-specific code.

## Quick start

```bash
# 1. Install dependencies (creates .venv/ and locks versions from uv.lock)
uv sync

# 2. Run all experiments end-to-end
uv run python sample_environments.py
uv run python experiment_gaming_boundary.py
uv run python experiment_portfolio_size.py
uv run python experiment_comparative_statics.py
uv run python experiment_baseline_horserace.py

# 3. Inspect the outputs
ls out/figures/      # PDF figures
ls out/tables/       # LaTeX tables
ls out/stats/        # LaTeX scalar macros
```

Outputs are deterministic given fixed random seeds (set in `config.py`).
Re-running produces byte-identical figures and tables modulo float-rendering
noise from matplotlib.

## File-by-file description

### Source modules

- **`config.py`** — Single source of truth for paths, random seeds, sample
  sizes, and matplotlib styling. All experiment scripts import their numerical
  parameters from here, so future changes to seed values or output locations
  touch one file.
- **`model.py`** — Closed-form analytical primitives shared across all
  experiments:
  - `Environment` — dataclass holding $(B, c, \gamma, k_\sigma, k_\delta, F)$.
  - `alpha_star(env, sigma, delta)` — developer best response (Prop 1):
    $\alpha^* = \mathrm{clip}((c - B\sigma - \delta F)/(c+\gamma), 0, 1)$.
  - `regulator_optimum(env, ...)` — solves the regulator's bilevel problem,
    returning $(\sigma^*, \delta^*, \alpha^*_{\mathrm{eq}}, u_R)$.
  - `regulator_payoff(...)` — utility evaluation for arbitrary $(\sigma, \delta, n)$.
  - `critical_portfolio_size(B, c, sigma, delta, F)` — the critical
    $n^* = \lceil (c - B\sigma)/(\delta F) \rceil$, returning `math.inf` when
    $\delta F = 0$.
  - `portfolio_alpha_star(...)` — developer best response under a portfolio of
    $n$ symmetric metrics.
  - `boltzmann_alpha(...)`, `myopic_alpha(...)` — bounded-rational developer
    archetypes used in the robustness experiments.

- **`sample_environments.py`** — Samples $N = 1{,}000$ environments uniformly
  from the family $(B, c, \gamma, k_\sigma, k_\delta, F) \in [1,10] \times
  [0.5,3] \times [0.1,2] \times [0.1,2]^2 \times [0.5,5]$ used by the horse-race
  experiment. Also writes the count macro for the paper.

### Experiments

- **`experiment_gaming_boundary.py`** — Sweeps $(B, F)$ on a grid at fixed
  $c = 1, \gamma = 0.5, k_\sigma = k_\delta = 1$. For each pair, solves the
  regulator's interior optimum and records $\alpha^*_{\mathrm{eq}}$ and the
  signed boundary residual $B\sigma^* + \delta^* F - c$. Plots the
  gaming-intensity heatmap with the analytical zero-residual contour overlaid.
- **`experiment_portfolio_size.py`** — Fixes a non-gaming-proof environment,
  computes $(\sigma^*, \delta^*)$, then sweeps the portfolio size $n$ and
  records (i) $\alpha_n^*$, (ii) regulator payoff $u_R(n)$, (iii) safety
  $S^* = 1 - \alpha_n^*$. The dashed line marks the predicted $n^*$ from
  `critical_portfolio_size`.
- **`experiment_comparative_statics.py`** — Three sub-experiments at the
  baseline $(B, c, \gamma, k_\sigma, k_\delta, F) = (2, 2, 1.5, 3, 3, 2)$,
  chosen so that $\Phi = 8/3 < c(c+\gamma) = 7$ (deep-interior regime, by
  design).
  (a) Single-parameter perturbations of $c, k_\sigma, k_\delta, \gamma$;
  (b) Developer-type misspecification: optimal under rational, evaluated
  against rational / Boltzmann ($\tau = 0.05$) / penalty-blind myopic;
  (c) Detection-probability estimation error: vary $\hat\delta$ around true
  $\delta$ and report the regulator-payoff gap.
- **`experiment_baseline_horserace.py`** — The headline horse-race. Across the
  $1{,}000$ sampled environments, evaluates six regulator policies (two
  deterministic single metrics, a uniform $n=5$ portfolio, a payoff-matched
  single metric, the optimal robust portfolio from a grid search over
  $n \times (\sigma, \delta)$, and the oracle) under three developer archetypes
  (rational, Boltzmann $\tau = 0.05$, penalty-blind myopic). Outputs payoff CDFs,
  the summary-statistics table, and the joint $(n^{\mathrm{opt}}, n^*)$
  distribution figure.

### Configuration files

- **`pyproject.toml`** — Project metadata and dependency declarations.
- **`uv.lock`** — Fully pinned dependency lock-file (cross-platform).
- **`.gitignore`** — Standard Python ignores plus `out/` (generated outputs are
  not committed).

## Mapping experiments to theoretical results

The package illustrates four formal results from the paper:

1. **Gaming-proof condition (Corollary 2).** $B\sigma + \delta F \geq c$ is
   necessary and sufficient for zero gaming.
   *Illustrated by:* `experiment_gaming_boundary.py` — the analytical contour
   $B\sigma^* + \delta^* F = c$ aligns precisely with the numerical phase
   transition in $\alpha^*_{\mathrm{eq}}$.

2. **Regime threshold $\Phi = c(c+\gamma)$ (Theorem 4 + Corollary 5).** The
   regulator's optimal single-metric design switches from interior (some gaming
   tolerated) to gaming-proof at $\Phi := B^2/k_\sigma + F^2/k_\delta = c(c+\gamma)$.
   Below this threshold, full deterrence is payoff-suboptimal.
   *Illustrated by:* `experiment_baseline_horserace.py` — the partition of
   environments by $\Phi < c(c+\gamma)$ vs $\Phi \geq c(c+\gamma)$ shows the
   regulator strictly prefers the interior optimum in the former regime.

3. **Critical portfolio size (Proposition 7).** When no single metric is
   gaming-proof, $n^* = \lceil (c - B\sigma)/(\delta F)\rceil$ symmetric
   metrics suffice to drive $\alpha_n^* \to 0$, provided the detection channel
   is non-trivial ($\delta F > 0$).
   *Illustrated by:* `experiment_portfolio_size.py` — the dashed $n^*$ coincides
   with the elbow at which $\alpha_n^*$ first reaches zero.

4. **Comparative statics (Proposition 6).** Signs of
   $\partial \alpha^*_{\mathrm{eq}} / \partial \theta$ for
   $\theta \in \{c, k_\sigma, k_\delta, \gamma\}$, with the conditional
   sign-flip on $\gamma$ governed by $\mathrm{sign}(2\Phi - c(c+\gamma))$.
   *Illustrated by:* `experiment_comparative_statics.py` Panel (a) —
   closed-form $(\sigma^*, \delta^*, \alpha^*_{\mathrm{eq}})$ at perturbed
   parameters matches the predicted signs.

## Reproducibility checklist

- [x] All scripts run without external dependencies beyond `pyproject.toml`.
- [x] Random seeds pinned in `config.py` for stochastic experiments.
- [x] No reliance on absolute paths or environment variables.
- [x] No data files required (everything is synthetic and generated in-memory).
- [x] Outputs land in a single `out/` tree, recreated cleanly on each run.
- [x] Package compiles its own LaTeX fragments (`out/tables/*.tex`,
  `out/stats/*.tex`) — no manual numbers in the paper.
- [x] Cross-platform: tested on macOS arm64; pure-Python + numpy/scipy/matplotlib
  works on Linux and Windows.

## License

MIT — see `LICENSE`.
