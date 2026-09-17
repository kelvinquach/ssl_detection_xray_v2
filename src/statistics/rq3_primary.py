"""RQ3 primary repeated-measures ANOVA for the locked SSOD protocol.

S6.03 scientific contract:

- for each labeled budget b and training seed s, B_{b,s} is the equal-weight
  mean of the paired SSL-SUP delta-mAP effects from R50 and Swin-T;
- budget is a repeated categorical factor with exactly four locked levels:
  1%, 5%, 10%, 20%;
- the same exact 10 training seeds are the repeated-measures subjects;
- primary inference is one-way repeated-measures ANOVA;
- Greenhouse-Geisser correction is ALWAYS applied;
- Mauchly/sphericity is diagnostic only and never selects the primary method;
- S6.03 does not implement pairwise contrasts, Holm adjustment, or Friedman.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np
from scipy import stats

from .seed_summary import (
    OFFICIAL_SEED_COUNT,
    OFFICIAL_TRAINING_SEEDS,
    REPLICATION_UNIT,
)

RQ3_BUDGET_LEVELS = ("1%", "5%", "10%", "20%")
RQ3_BUDGET_COUNT = len(RQ3_BUDGET_LEVELS)
RQ3_ARCHITECTURE_COUNT = 2
RQ3_ARCHITECTURE_WEIGHT = 1.0 / RQ3_ARCHITECTURE_COUNT
RQ3_UNCORRECTED_DF_NUMERATOR = RQ3_BUDGET_COUNT - 1
RQ3_UNCORRECTED_DF_DENOMINATOR = (
    (OFFICIAL_SEED_COUNT - 1) * (RQ3_BUDGET_COUNT - 1)
)
RQ3_GREENHOUSE_GEISSER_POLICY = "ALWAYS"
RQ3_MAUCHLY_ROLE = "DIAGNOSTIC_ONLY"


class RQ3PrimaryError(ValueError):
    """Raised when RQ3 input violates the locked protocol."""


def compute_rq3_budget_gain_by_seed(
    seed_architecture_budget_effects: Mapping[int, Sequence[Sequence[float]]],
) -> dict[int, tuple[float, ...]]:
    """Compute B_{b,s} from exactly four budgets x two architectures per seed."""
    if not isinstance(seed_architecture_budget_effects, Mapping):
        raise RQ3PrimaryError(
            "seed_architecture_budget_effects must map training_seed -> "
            "4 budget rows x 2 paired architecture delta-mAP values"
        )

    observed_seeds = tuple(seed_architecture_budget_effects.keys())
    if observed_seeds != OFFICIAL_TRAINING_SEEDS:
        raise RQ3PrimaryError(
            "training seeds must match the exact locked ordered list; "
            f"expected={OFFICIAL_TRAINING_SEEDS}, observed={observed_seeds}"
        )

    gains: dict[int, tuple[float, ...]] = {}
    expected_shape = (RQ3_BUDGET_COUNT, RQ3_ARCHITECTURE_COUNT)

    for seed in OFFICIAL_TRAINING_SEEDS:
        try:
            effects = np.asarray(
                seed_architecture_budget_effects[seed],
                dtype=float,
            )
        except (TypeError, ValueError) as exc:
            raise RQ3PrimaryError(
                f"seed {seed} must provide a numeric {expected_shape} "
                "budget-by-architecture matrix"
            ) from exc

        if effects.shape != expected_shape:
            raise RQ3PrimaryError(
                f"seed {seed} must provide exactly shape {expected_shape}; "
                f"observed={effects.shape}"
            )
        if not np.all(np.isfinite(effects)):
            raise RQ3PrimaryError(
                f"seed {seed} architecture-budget effects must all be finite"
            )

        budget_gains = np.mean(effects, axis=1)
        gains[seed] = tuple(float(value) for value in budget_gains)

    return gains


def _validate_seed_budget_gains(
    seed_budget_gains: Mapping[int, Sequence[float]],
) -> np.ndarray:
    if not isinstance(seed_budget_gains, Mapping):
        raise RQ3PrimaryError(
            "seed_budget_gains must map training_seed -> 4 ordered budget gains"
        )

    observed_seeds = tuple(seed_budget_gains.keys())
    if observed_seeds != OFFICIAL_TRAINING_SEEDS:
        raise RQ3PrimaryError(
            "training seeds must match the exact locked ordered list; "
            f"expected={OFFICIAL_TRAINING_SEEDS}, observed={observed_seeds}"
        )

    rows: list[np.ndarray] = []
    for seed in OFFICIAL_TRAINING_SEEDS:
        try:
            values = np.asarray(seed_budget_gains[seed], dtype=float)
        except (TypeError, ValueError) as exc:
            raise RQ3PrimaryError(
                f"seed {seed} must provide exactly {RQ3_BUDGET_COUNT} numeric budget gains"
            ) from exc

        if values.ndim != 1 or values.size != RQ3_BUDGET_COUNT:
            raise RQ3PrimaryError(
                f"seed {seed} must provide exactly {RQ3_BUDGET_COUNT} ordered budget gains"
            )
        if not np.all(np.isfinite(values)):
            raise RQ3PrimaryError(
                f"seed {seed} budget gains must all be finite"
            )
        rows.append(values)

    matrix = np.vstack(rows)
    expected_shape = (OFFICIAL_SEED_COUNT, RQ3_BUDGET_COUNT)
    if matrix.shape != expected_shape:
        raise RQ3PrimaryError(
            f"RQ3 requires matrix shape {expected_shape}; observed={matrix.shape}"
        )
    return matrix


def _greenhouse_geisser_epsilon(values: np.ndarray) -> float:
    """Return Greenhouse-Geisser epsilon from repeated-measures covariance."""
    covariance = np.cov(values, rowvar=False, ddof=1)
    k = values.shape[1]
    centering = np.eye(k) - np.ones((k, k), dtype=float) / float(k)
    centered_covariance = centering @ covariance @ centering

    trace_value = float(np.trace(centered_covariance))
    trace_square = float(
        np.trace(centered_covariance @ centered_covariance)
    )
    denominator = float((k - 1) * trace_square)

    if (
        not math.isfinite(trace_value)
        or not math.isfinite(trace_square)
        or trace_value <= 0.0
        or denominator <= 0.0
    ):
        raise RQ3PrimaryError(
            "Greenhouse-Geisser epsilon is undefined for the supplied covariance"
        )

    epsilon = (trace_value * trace_value) / denominator
    lower_bound = 1.0 / float(k - 1)
    if epsilon < lower_bound - 1e-12 or epsilon > 1.0 + 1e-12:
        raise RQ3PrimaryError(
            f"Greenhouse-Geisser epsilon outside theoretical bounds: {epsilon}"
        )
    return float(np.clip(epsilon, lower_bound, 1.0))


def _mauchly_sphericity_diagnostic(
    values: np.ndarray,
) -> dict[str, object]:
    """Compute Mauchly's W as a diagnostic; never select the primary method."""
    n, k = values.shape
    covariance = np.cov(values, rowvar=False, ddof=1)
    centering = np.eye(k) - np.ones((k, k), dtype=float) / float(k)
    centered_covariance = centering @ covariance @ centering

    eigenvalues = np.linalg.eigvalsh(centered_covariance)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    tolerance = (
        np.finfo(float).eps
        * max(centered_covariance.shape)
        * scale
    )
    positive = eigenvalues[eigenvalues > tolerance]
    d = k - 1
    df = int(d * (d + 1) / 2 - 1)

    if positive.size != d:
        return {
            "defined": False,
            "role": RQ3_MAUCHLY_ROLE,
            "used_to_select_primary_method": False,
            "reason": (
                "centered covariance does not have k-1 positive eigenvalues"
            ),
            "w": None,
            "chi_square": None,
            "df": df,
            "p_value": None,
        }

    mean_eigenvalue = float(np.mean(positive))
    w = float(np.prod(positive) / (mean_eigenvalue ** d))
    w = float(np.clip(w, np.finfo(float).tiny, 1.0))

    correction = 1.0 - (
        (2.0 * d * d + d + 2.0)
        / (6.0 * d * float(n - 1))
    )
    chi_square = float(-(n - 1) * correction * math.log(w))
    p_value = float(stats.chi2.sf(chi_square, df=df))

    return {
        "defined": True,
        "role": RQ3_MAUCHLY_ROLE,
        "used_to_select_primary_method": False,
        "w": w,
        "chi_square": chi_square,
        "df": df,
        "p_value": p_value,
    }


def rq3_primary_anova(
    seed_budget_gains: Mapping[int, Sequence[float]],
) -> dict[str, object]:
    """Run locked one-way repeated-measures ANOVA with GG correction ALWAYS."""
    values = _validate_seed_budget_gains(seed_budget_gains)
    n, k = values.shape

    grand_mean = float(np.mean(values))
    subject_means = np.mean(values, axis=1)
    budget_means = np.mean(values, axis=0)

    ss_total = float(np.sum((values - grand_mean) ** 2))
    ss_subject = float(
        k * np.sum((subject_means - grand_mean) ** 2)
    )
    ss_budget = float(
        n * np.sum((budget_means - grand_mean) ** 2)
    )
    ss_error = float(ss_total - ss_subject - ss_budget)

    df_num = k - 1
    df_den = (n - 1) * (k - 1)
    if df_num != RQ3_UNCORRECTED_DF_NUMERATOR:
        raise RQ3PrimaryError(
            "unexpected RQ3 numerator degrees of freedom"
        )
    if df_den != RQ3_UNCORRECTED_DF_DENOMINATOR:
        raise RQ3PrimaryError(
            "unexpected RQ3 denominator degrees of freedom"
        )
    if ss_budget < -1e-12 or ss_error <= 0.0:
        raise RQ3PrimaryError(
            "repeated-measures ANOVA requires non-negative budget variance "
            f"and positive residual variance; ss_budget={ss_budget}, "
            f"ss_error={ss_error}"
        )

    ms_budget = ss_budget / float(df_num)
    ms_error = ss_error / float(df_den)
    if ms_error <= 0.0 or not math.isfinite(ms_error):
        raise RQ3PrimaryError(
            "repeated-measures ANOVA residual mean square is invalid"
        )

    f_statistic = float(ms_budget / ms_error)
    epsilon_gg = _greenhouse_geisser_epsilon(values)
    df_gg_num = float(epsilon_gg * df_num)
    df_gg_den = float(epsilon_gg * df_den)
    p_gg = float(
        stats.f.sf(f_statistic, df_gg_num, df_gg_den)
    )

    if not all(
        math.isfinite(value)
        for value in (
            f_statistic,
            epsilon_gg,
            df_gg_num,
            df_gg_den,
            p_gg,
        )
    ):
        raise RQ3PrimaryError(
            "RQ3 GG-corrected ANOVA produced non-finite output"
        )

    mauchly = _mauchly_sphericity_diagnostic(values)

    return {
        "replication_unit": REPLICATION_UNIT,
        "effect": "B_b_s",
        "primary_method": "ONE_WAY_REPEATED_MEASURES_ANOVA",
        "greenhouse_geisser_policy": RQ3_GREENHOUSE_GEISSER_POLICY,
        "greenhouse_geisser_applied": True,
        "mauchly_role": RQ3_MAUCHLY_ROLE,
        "n_subjects": int(n),
        "budget_count": int(k),
        "budget_levels": list(RQ3_BUDGET_LEVELS),
        "budget_means": [
            float(value) for value in budget_means
        ],
        "f_statistic": f_statistic,
        "df_uncorrected_numerator": int(df_num),
        "df_uncorrected_denominator": int(df_den),
        "epsilon_gg": epsilon_gg,
        "df_gg_numerator": df_gg_num,
        "df_gg_denominator": df_gg_den,
        "p_gg": p_gg,
        "mauchly_diagnostic": mauchly,
    }
