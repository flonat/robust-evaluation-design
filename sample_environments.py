"""Sample the environment family described in the experiments section.

Writes the count of generated environments to
out/stats/num_environments.tex.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from model import Environment

SEED = 20260420
N_ENV = 1000

REPO_ROOT = Path(__file__).resolve().parent
OUT_STATS = REPO_ROOT / "out" / "stats"


def sample(n: int, seed: int) -> list[Environment]:
    rng = np.random.default_rng(seed)
    envs = []
    for _ in range(n):
        envs.append(Environment(
            B=float(rng.uniform(1.0, 10.0)),
            c=float(rng.uniform(0.5, 3.0)),
            gamma=float(rng.uniform(0.1, 2.0)),
            k_sigma=float(rng.uniform(0.1, 2.0)),
            k_delta=float(rng.uniform(0.1, 2.0)),
            F=float(rng.uniform(0.5, 5.0)),
        ))
    return envs


def main() -> None:
    envs = sample(N_ENV, SEED)
    OUT_STATS.mkdir(parents=True, exist_ok=True)
    count_formatted = f"{len(envs):,}".replace(",", "{,}")
    (OUT_STATS / "num_environments.tex").write_text(count_formatted + "%\n")
    print(f"Wrote {len(envs)} environments count to {OUT_STATS/'num_environments.tex'}")


if __name__ == "__main__":
    main()
