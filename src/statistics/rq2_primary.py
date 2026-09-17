"""RQ2 primary statistical test for the locked SSOD protocol.

S6.02 scientific contract:

- per-seed overall SSL gain G_s is the equal-weight mean of exactly
  8 architecture x labeled-budget paired delta-mAP effects;
- inferential replication unit is training seed / trained run;
- primary inference is a one-sample t-test against zero;
- alternative is greater (one-sided p-value);
- official n = 10 and df = 9;
- uncertainty is reported with a two-sided 95% confidence interval.
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
    summarize_seed_runs,
)

RQ2_CELL_COUNT = 8
RQ2_CELL_WEIGHT = 1.0 / RQ2_CELL_COUNT
RQ2_NULL_MEAN = 0.0
RQ2_ALTERNATIVE = "greater"
RQ2_ALPHA = 0.05
RQ2_CI_LEVEL = 0.95
RQ2_DF = OFFICIAL_SEED_COUNT - 1


class RQ2PrimaryError(ValueError):
    """Raised when RQ2 input violates the locked protocol."""


def compute_rq2_gain_by_seed(
    seed_cell_effects: Mapping[int, Sequence[float]],
) -> dict[int, float]:
    """Compute G_s as the equal-weight mean of exactly eight paired cell effects."""
    if not isinstance(seed_cell_effects, Mapping):
        raise RQ2PrimaryError(
            "seed_cell_effects must map training_seed -> 8 paired delta-mAP values"
        )

    observed_seeds = tuple(seed_cell_effects.keys())
    if observed_seeds != OFFICIAL_TRAINING_SEEDS:
        raise RQ2PrimaryError(
            "training seeds must match the exact locked ordered list; "
            f"expected={OFFICIAL_TRAINING_SEEDS}, observed={observed_seeds}"
        )

    gains: dict[int, float] = {}
    for seed in OFFICIAL_TRAINING_SEEDS:
        try:
            effects = np.asarray(seed_cell_effects[seed], dtype=float)
        except (TypeError, ValueError) as exc:
            raise RQ2PrimaryError(
                f"seed {seed} must provide exactly {RQ2_CELL_COUNT} numeric cell effects"
            ) from exc

        if effects.ndim != 1 or effects.size != RQ2_CELL_COUNT:
            raise RQ2PrimaryError(
                f"seed {seed} must provide exactly {RQ2_CELL_COUNT} cell effects"
            )
        if not np.all(np.isfinite(effects)):
            raise RQ2PrimaryError(
                f"seed {seed} cell effects must all be finite"
            )

        gains[seed] = float(np.mean(effects))

    return gains


def rq2_primary_test(
    seed_gains: Mapping[int, float],
) -> dict[str, object]:
    """Run the locked RQ2 one-sample greater-than-zero t-test."""
    summary = summarize_seed_runs(seed_gains)

    values = np.asarray(
        [seed_gains[seed] for seed in OFFICIAL_TRAINING_SEEDS],
        dtype=float,
    )

    result = stats.ttest_1samp(
        values,
        popmean=RQ2_NULL_MEAN,
        alternative=RQ2_ALTERNATIVE,
    )

    n = int(summary["n"])
    df = int(result.df)
    if n != OFFICIAL_SEED_COUNT or df != RQ2_DF:
        raise RQ2PrimaryError(
            f"RQ2 requires n={OFFICIAL_SEED_COUNT} and df={RQ2_DF}; "
            f"observed n={n}, df={df}"
        )

    effect_estimate = float(summary["mean"])
    sample_sd = float(summary["sample_sd"])
    standard_error = sample_sd / math.sqrt(n)
    critical_value = float(stats.t.ppf(1.0 - RQ2_ALPHA / 2.0, df=df))
    ci_low = effect_estimate - critical_value * standard_error
    ci_high = effect_estimate + critical_value * standard_error

    return {
        "replication_unit": REPLICATION_UNIT,
        "effect": "G_s",
        "effect_estimate": effect_estimate,
        "sample_sd": sample_sd,
        "sample_sd_ddof": int(summary["sample_sd_ddof"]),
        "n": n,
        "df": df,
        "null_mean": RQ2_NULL_MEAN,
        "alternative": RQ2_ALTERNATIVE,
        "t_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "p_value_sidedness": "one-sided",
        "ci_level": RQ2_CI_LEVEL,
        "ci_sidedness": "two-sided",
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }
