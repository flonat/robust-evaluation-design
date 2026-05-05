"""Shared primitives for the Stackelberg strategic-compliance model.

All formulas match the model section of the accompanying paper:
  - Developer payoff (eq. developer-payoff):
        u_P = B(1 - sigma*alpha) - (c/2)(1-alpha)^2 - (gamma/2)*alpha^2 - delta*F*alpha
  - Best response (Prop 1):
        alpha* = clip((c - B*sigma - delta*F) / (c + gamma), 0, 1)
  - Regulator payoff (eq. regulator-payoff):
        u_R = (1 - alpha*(m)) - (k_sigma/2)*sigma^2 - (k_delta/2)*delta^2
  - Portfolio payoff (eq. portfolio-payoff), symmetric metrics:
        u_P = B(1 - sigma*alpha) - (c/2)(1-alpha)^2 - n*(gamma/2)*alpha^2 - n*delta*F*alpha
  - Portfolio best response (symmetric):
        alpha_n* = clip((c - B*sigma - n*delta*F) / (c + n*gamma), 0, 1)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Environment:
    B: float       # developer's benefit per unit of metric score
    c: float       # genuine compliance cost parameter
    gamma: float   # intrinsic gaming cost
    k_sigma: float # implementation cost for safety coupling
    k_delta: float # implementation cost for detection probability
    F: float       # fine on detection


def alpha_star(B: float, c: float, gamma: float, sigma: float, delta: float, F: float) -> float:
    """Developer best-response gaming intensity, eq. (alpha-star)."""
    numerator = c - B * sigma - delta * F
    denominator = c + gamma
    if denominator <= 0:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def gaming_proof(B: float, c: float, sigma: float, delta: float, F: float) -> bool:
    """Gaming-proof iff B*sigma + delta*F >= c (Corollary)."""
    return B * sigma + delta * F >= c


def developer_payoff(alpha: float, B: float, c: float, gamma: float,
                     sigma: float, delta: float, F: float) -> float:
    """Developer utility (single metric), eq. (developer-payoff)."""
    return (B * (1.0 - sigma * alpha)
            - 0.5 * c * (1.0 - alpha) ** 2
            - 0.5 * gamma * alpha * alpha
            - delta * F * alpha)


def regulator_payoff(alpha: float, sigma: float, delta: float,
                     k_sigma: float, k_delta: float,
                     n: int = 1, per_metric_cost: bool = True) -> float:
    """Regulator utility, eq. (regulator-payoff) generalised to portfolios.

    u_R = (1 - alpha) - cost_factor * (k_sigma*sigma^2 + k_delta*delta^2) / 2

    cost_factor = n if per_metric_cost else 1 (shared-infrastructure interpretation).
    """
    cost_factor = n if per_metric_cost else 1
    return (1.0 - alpha) - 0.5 * cost_factor * (k_sigma * sigma * sigma + k_delta * delta * delta)


def regulator_optimum(env: Environment, sigma_grid: int = 51, delta_grid: int = 51) -> dict:
    """Grid-search the regulator's optimal single metric (sigma*, delta*)."""
    sigmas = np.linspace(0.0, 1.0, sigma_grid)
    deltas = np.linspace(0.0, 1.0, delta_grid)
    best = {"payoff": -np.inf, "sigma": 0.0, "delta": 0.0, "alpha": 1.0}
    for s in sigmas:
        for d in deltas:
            a = alpha_star(env.B, env.c, env.gamma, s, d, env.F)
            u = regulator_payoff(a, s, d, env.k_sigma, env.k_delta, n=1)
            if u > best["payoff"]:
                best = {"payoff": float(u), "sigma": float(s), "delta": float(d), "alpha": float(a)}
    return best


def portfolio_alpha_star(B: float, c: float, gamma: float, sigma: float,
                         delta: float, F: float, n: int) -> float:
    """Developer best-response against a uniform n-metric portfolio.

    From symmetric FOC of eq. (portfolio-payoff):
        alpha_n* = clip((c - B*sigma - n*delta*F) / (c + n*gamma), 0, 1)
    """
    numerator = c - B * sigma - n * delta * F
    denominator = c + n * gamma
    if denominator <= 0:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def critical_portfolio_size(B: float, c: float, sigma: float, delta: float, F: float) -> int:
    """n* such that alpha_n* = 0: ceil((c - B*sigma) / (delta*F))."""
    if delta * F <= 0:
        return math.inf  # type: ignore[return-value]
    gap = c - B * sigma
    if gap <= 0:
        return 1
    return int(math.ceil(gap / (delta * F)))


def portfolio_developer_payoff(alpha: float, B: float, c: float, gamma: float,
                               sigma: float, delta: float, F: float, n: int) -> float:
    """Developer utility under symmetric uniform n-metric portfolio, eq. (portfolio-payoff)."""
    return (B * (1.0 - sigma * alpha)
            - 0.5 * c * (1.0 - alpha) ** 2
            - n * 0.5 * gamma * alpha * alpha
            - n * delta * F * alpha)


def boltzmann_alpha(B: float, c: float, gamma: float, sigma: float, delta: float,
                    F: float, tau: float = 0.5, grid: int = 201) -> float:
    """Boundedly-rational developer, single metric."""
    alphas = np.linspace(0.0, 1.0, grid)
    utilities = np.array([developer_payoff(a, B, c, gamma, sigma, delta, F) for a in alphas])
    logits = utilities / max(tau, 1e-6)
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()
    return float(np.sum(alphas * weights))


def myopic_alpha(B: float, c: float, gamma: float, sigma: float) -> float:
    """Penalty-blind myopic developer.

    Maximises u_m(alpha) = B(1 - sigma*alpha) - (c/2)(1-alpha)^2 - (gamma/2)alpha^2,
    i.e., the rational payoff with the detection-penalty term delta*F*alpha set to zero.
    Interpretation: the developer ignores the fine risk and acts as if delta=0.

    FOC: -B*sigma + c(1-alpha) - gamma*alpha = 0
        => alpha_m = clip((c - B*sigma)/(c + gamma), 0, 1).

    By construction, alpha_m >= alpha_rational with equality iff delta=0, so this type
    games at least as much as the rational best-responder for any (sigma, delta).
    """
    if c + gamma <= 0:
        return 0.0
    return float(np.clip((c - B * sigma) / (c + gamma), 0.0, 1.0))
