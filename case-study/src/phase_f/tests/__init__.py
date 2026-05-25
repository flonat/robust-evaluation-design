"""F.3 predictive tests of the central paper claims.

Each module here tests one prediction from the manuscript:
  - frontier.py        : gap = (1-σ)α identity check across strategies
  - fine_threshold.py  : F* = (c - Bσ)/δ transition (TODO when F sweep available)
  - portfolio.py       : Prop 16 portfolio threshold (TODO)
  - correlated.py      : Thm 29 hyperbolic divergence as ρ → 1 (TODO)
  - policy.py          : model-selected policy vs naive baselines (TODO)

Status as of F.3a: `frontier` lands first because it's the only test directly
expressible from the (model × strategy × benchmark) score table that F.2
produces. The others require additional experimental sweeps that are in scope
for F.3 follow-up.
"""
from phase_f.tests.frontier import (
    FrontierTestResult,
    run_frontier_test,
    summarise_frontier_tests,
)

__all__ = [
    "FrontierTestResult",
    "run_frontier_test",
    "summarise_frontier_tests",
]
