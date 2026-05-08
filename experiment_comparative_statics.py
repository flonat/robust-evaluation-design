"""Experiment: Comparative statics and robustness checks.

Three sub-experiments:
  1. Comparative statics — perturb c, k_sigma, k_delta, gamma individually
     around a baseline environment and record (sigma*, delta*, alpha*_eq).
  2. Developer-type misspecification — optimal metric under rationality;
     evaluate against rational / bounded / myopic types; report payoff gap.
  3. Detection-probability estimation error — vary hat_delta around true delta
     and measure regulator loss.

Writes paper-aij/paper/output/tables/comparative_statics.tex.
"""

from __future__ import annotations

import numpy as np

from config import N_ENV_ROBUSTNESS, OUT_TBL, SEED_DELTA_MISEST, SEED_TYPE_MISSPEC
from model import (Environment, alpha_star, boltzmann_alpha, myopic_alpha,
                   regulator_optimum, regulator_payoff)

# Baseline chosen so Phi = B^2/k_sigma + F^2/k_delta = 8/3 < c(c+gamma) = 7.0,
# placing the model in the interior regime required by Proposition~\ref{prop:comp-statics}.
# Additionally chosen so that 2*Phi < c(c+gamma) at every gamma-perturbation, ensuring
# d alpha*/d gamma < 0 across the table (sign of d alpha*/d gamma is sign(2Phi - c(c+gamma))).
# Earlier baseline (k_sigma=k_delta=1) gave Phi=8 > 3.45, which was gaming-proof and
# yielded uninformative all-zero comparative statics.
BASELINE = Environment(B=2.0, c=2.0, gamma=1.5, k_sigma=3.0, k_delta=3.0, F=2.0)


def closed_form_optimum(env: Environment) -> dict:
    """Closed-form interior optimum (Theorem~\\ref{thm:opt-single}).

    Avoids grid-search artefacts when (sigma*, delta*) lies between grid points.
    Falls back to gaming-proof regime when Phi >= c(c+gamma).
    """
    Phi = env.B ** 2 / env.k_sigma + env.F ** 2 / env.k_delta
    if Phi >= env.c * (env.c + env.gamma):
        # Gaming-proof regime: solve constrained QP min K(sigma,delta) s.t. B*sigma + delta*F = c.
        D = env.B ** 2 * env.k_delta + env.F ** 2 * env.k_sigma  # = k_sigma * k_delta * Phi
        sigma_gp = env.c * env.B * env.k_delta / D
        delta_gp = env.c * env.F * env.k_sigma / D
        u = regulator_payoff(0.0, sigma_gp, delta_gp, env.k_sigma, env.k_delta, n=1)
        return {"sigma": float(sigma_gp), "delta": float(delta_gp),
                "alpha": 0.0, "payoff": float(u)}
    sigma = env.B / (env.k_sigma * (env.c + env.gamma))
    delta = env.F / (env.k_delta * (env.c + env.gamma))
    alpha = (env.c * (env.c + env.gamma) - Phi) / (env.c + env.gamma) ** 2
    u = regulator_payoff(alpha, sigma, delta, env.k_sigma, env.k_delta, n=1)
    return {"sigma": float(sigma), "delta": float(delta),
            "alpha": float(alpha), "payoff": float(u)}


def run_comp_statics() -> list[tuple[str, float, float, float, float, float]]:
    """Vary each parameter by +/-50% and record optimal design + alpha* (closed-form)."""
    rows: list[tuple[str, float, float, float, float, float]] = []
    base_opt = closed_form_optimum(BASELINE)
    rows.append(("baseline", 0.0,
                 base_opt["sigma"], base_opt["delta"], base_opt["alpha"], base_opt["payoff"]))

    variations = [
        ("c", "c", [-0.5, 0.5]),
        (r"k_{\sigma}", "k_sigma", [-0.5, 0.5]),
        (r"k_{\delta}", "k_delta", [-0.5, 0.5]),
        (r"\gamma", "gamma", [-0.5, 0.5]),
    ]

    for label, attr, deltas in variations:
        for d in deltas:
            params = BASELINE.__dict__.copy()
            params[attr] = getattr(BASELINE, attr) * (1.0 + d)
            env = Environment(**params)
            opt = closed_form_optimum(env)
            rows.append((label, d, opt["sigma"], opt["delta"], opt["alpha"], opt["payoff"]))
    return rows


def run_type_misspec() -> dict[str, float]:
    """Compute regulator payoff gap for boundedly-rational and myopic developers.

    Regulator designs optimally assuming rationality, then evaluates against
    each type. Report mean payoff vs oracle (who knows the true type).

    Uses an independent seed (SEED_TYPE_MISSPEC) and an independent environment
    family (N_ENV_ROBUSTNESS draws from a tighter parameter range) so that
    misspecification gaps are not entangled with the headline horse-race draws.
    """
    rng = np.random.default_rng(SEED_TYPE_MISSPEC)
    n = N_ENV_ROBUSTNESS
    gaps: dict[str, list[float]] = {"bounded": [], "myopic": []}
    for _ in range(n):
        env = Environment(
            B=float(rng.uniform(1.0, 5.0)),
            c=float(rng.uniform(0.5, 2.0)),
            gamma=float(rng.uniform(0.3, 1.5)),
            k_sigma=float(rng.uniform(0.3, 1.5)),
            k_delta=float(rng.uniform(0.3, 1.5)),
            F=float(rng.uniform(1.0, 4.0)),
        )
        assumed = regulator_optimum(env, sigma_grid=41, delta_grid=41)
        sigma = assumed["sigma"]
        delta = assumed["delta"]

        a_bounded = boltzmann_alpha(env.B, env.c, env.gamma, sigma, delta, env.F, tau=0.05)
        a_myopic = myopic_alpha(env.B, env.c, env.gamma, sigma)

        u_rat = assumed["payoff"]
        u_bounded = regulator_payoff(a_bounded, sigma, delta, env.k_sigma, env.k_delta)
        u_myopic = regulator_payoff(a_myopic, sigma, delta, env.k_sigma, env.k_delta)

        gaps["bounded"].append(u_rat - u_bounded)
        gaps["myopic"].append(u_rat - u_myopic)
    return {k: float(np.mean(v)) for k, v in gaps.items()}


def run_delta_misest() -> dict[float, float]:
    """Mean regulator loss from misestimating delta by a given multiplicative factor.

    Uses an independent seed (SEED_DELTA_MISEST) and an independent environment
    family from run_type_misspec, isolating delta-misestimation effects.
    """
    rng = np.random.default_rng(SEED_DELTA_MISEST)
    n = N_ENV_ROBUSTNESS
    factors = [0.5, 0.75, 1.25, 1.5]
    losses: dict[float, list[float]] = {f: [] for f in factors}
    for _ in range(n):
        env = Environment(
            B=float(rng.uniform(1.0, 5.0)),
            c=float(rng.uniform(0.5, 2.0)),
            gamma=float(rng.uniform(0.3, 1.5)),
            k_sigma=float(rng.uniform(0.3, 1.5)),
            k_delta=float(rng.uniform(0.3, 1.5)),
            F=float(rng.uniform(1.0, 4.0)),
        )
        opt = regulator_optimum(env, sigma_grid=41, delta_grid=41)
        sigma_true = opt["sigma"]
        delta_true = opt["delta"]
        u_true = opt["payoff"]
        for f in factors:
            delta_hat = np.clip(delta_true * f, 0.0, 1.0)
            a = alpha_star(env.B, env.c, env.gamma, sigma_true, float(delta_hat), env.F)
            u_hat = regulator_payoff(a, sigma_true, float(delta_hat),
                                     env.k_sigma, env.k_delta)
            losses[f].append(u_true - u_hat)
    return {f: float(np.mean(v)) for f, v in losses.items()}


def format_table(comp_rows, type_gaps, delta_losses) -> str:
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Comparative statics and robustness checks. "
                 r"Panel A: optimal metric and gaming intensity under $\pm 50\%$ parameter perturbations "
                 r"around the baseline $(B,c,\gamma,k_\sigma,k_\delta,F)=(2.0,2.0,1.5,3.0,3.0,2.0)$, "
                 r"chosen so that $2\Phi=16/3<c(c+\gamma)=7.0$ (deep interior regime, "
                 r"ensuring $\partial\alpha^*_{\mathrm{eq}}/\partial\gamma<0$ at every $\gamma$-perturbation). "
                 r"Panel A values use the closed-form optimum from Theorem~\ref{thm:opt-single}. "
                 r"Panel B: mean regulator payoff gap (vs rational-assumption design) when the developer "
                 r"is boundedly rational (Boltzmann with $\tau=0.05$, low temperature, near-rational) "
                 r"or penalty-blind myopic (rational payoff with $\delta F\alpha$ term suppressed). "
                 r"Panel C: mean regulator loss from misestimating $\delta$ by a multiplicative factor. "
                 rf"All panels averaged over {N_ENV_ROBUSTNESS} sampled environments.}}")
    lines.append(r"\label{tab:comparative_statics}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\multicolumn{6}{l}{\textbf{Panel A: Comparative statics}} \\")
    lines.append(r"Parameter & Shift & $\sigma^*$ & $\delta^*$ & $\alpha^*_{\mathrm{eq}}$ & $u_R^*$ \\")
    lines.append(r"\midrule")
    for label, shift, s, d, a, u in comp_rows:
        shift_str = "---" if label == "baseline" else f"${shift:+.2f}$"
        lines.append(f"${label}$ & {shift_str} & {s:.3f} & {d:.3f} & {a:.3f} & {u:.3f} \\\\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{l}{\textbf{Panel B: Developer-type misspecification (mean payoff gap)}} \\")
    lines.append(r"Type & & & & & Gap \\")
    lines.append(r"\midrule")
    lines.append(rf"Boundedly rational ($\tau=0.05$) & & & & & {type_gaps['bounded']:.3f} \\")
    lines.append(rf"Penalty-blind myopic & & & & & {type_gaps['myopic']:.3f} \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{l}{\textbf{Panel C: Detection-probability misestimation (mean loss)}} \\")
    lines.append(r"$\hat\delta / \delta$ & & & & & Loss \\")
    lines.append(r"\midrule")
    for f, loss in sorted(delta_losses.items()):
        lines.append(rf"${f:.2f}$ & & & & & {loss:.3f} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    comp = run_comp_statics()
    types = run_type_misspec()
    deltas = run_delta_misest()
    OUT_TBL.mkdir(parents=True, exist_ok=True)
    out_path = OUT_TBL / "comparative_statics.tex"
    out_path.write_text(format_table(comp, types, deltas))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
