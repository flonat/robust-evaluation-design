"""Shared constants, output paths, and matplotlib styling for the replication artifact.

Single source of truth for paths, seeds, and figure rendering. This is the
artifact-side configuration: outputs go to a self-contained `out/` tree
alongside the scripts. (The development copy in the main project's
`code/python/` directory writes directly into the paper's Overleaf figure
directory; the artifact version is decoupled so the package is reproducible
in isolation.)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl


def configure_matplotlib() -> None:
    """Set publication-quality rendering. Call once at the top of each script.

    Uses LaTeX for text rendering so figure math matches the paper body
    (`elsarticle` Computer Modern). Falls back to mathtext if the LaTeX
    backend is unavailable on the host (e.g., a CI runner without a TeX
    distribution installed).
    """
    try:
        mpl.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "pdf.fonttype": 42,  # TrueType, embeddable
        })
    except Exception:
        # Defensive: if pdflatex / latex not on PATH, fall back to mathtext.
        mpl.rcParams.update({
            "text.usetex": False,
            "mathtext.fontset": "cm",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        })


# Artifact root: this file lives at the package root (github-repo/config.py),
# so REPO_ROOT is the directory containing this file.
REPO_ROOT = Path(__file__).resolve().parent

# Self-contained output tree. All experiment scripts write here; the directory
# is created on first run if it does not already exist.
OUT_FIG = REPO_ROOT / "out" / "figures"
OUT_TBL = REPO_ROOT / "out" / "tables"
OUT_STATS = REPO_ROOT / "out" / "stats"

# Ensure output directories exist.
for _d in (OUT_FIG, OUT_TBL, OUT_STATS):
    _d.mkdir(parents=True, exist_ok=True)

# RNG seeds. Main env-sampling seed is shared across the headline horse-race;
# the comparative-statics sub-experiments use independent seeds for independent
# environment families (one per sub-experiment), documented at each call site.
SEED = 20260420
SEED_TYPE_MISSPEC = 2026
SEED_DELTA_MISEST = 11

# Sample sizes
N_ENV = 1000           # main horse-race environments
N_ENV_ROBUSTNESS = 200 # smaller draws for robustness sub-experiments
