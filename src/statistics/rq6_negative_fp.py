"""RQ6 primary false-positive-on-negative-images statistical fixture.

Locked protocol:
- primitive paired effect is SSL minus SUP FP/negative;
- exactly 8 architecture x budget paired cells per training seed;
- H_s is the equal-weight mean of those 8 paired effects;
- inferential replication unit is training seed / trained run;
- official n = 10 and df = 9;
- primary inference is a two-sided one-sample t-test against zero;
- uncertainty is an individual two-sided 95% confidence interval;
- FAR on negative images is estimation-focused and has no separate
  official hypothesis test in this RQ6 primary fixture.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from scipy import stats

from .seed_summary import (
    OFFICIAL_SEED_COUNT,
    OFFICIAL_TRAINING_SEEDS,
    REPLICATION_UNIT,
    summarize_seed_runs,
)

RQ6_ARCHITECTURES = ("R50", "Swin-T")
RQ6_BUDGETS = ("1%", "5%", "10%", "20%")
RQ6_CELL_COUNT = len(RQ6_ARCHITECTURES) * len(RQ6_BUDGETS)
RQ6_CELL_WEIGHT = 1.0 / RQ6_CELL_COUNT

RQ6_EFFECT = "H_s"
RQ6_DELTA_COLUMN = "delta_FP_per_negative_test"
RQ6_PRIMARY_METRIC = "FP_PER_NEGATIVE"
RQ6_NULL_MEAN = 0.0
RQ6_ALTERNATIVE = "two-sided"
RQ6_ALPHA = 0.05
RQ6_CI_LEVEL = 0.95
RQ6_DF = OFFICIAL_SEED_COUNT - 1

RQ6_FAR_SECONDARY_OUTCOME = True
RQ6_FAR_FORMAL_P_VALUE_REQUIRED = False


class RQ6NegativeFPError(ValueError):
    """Raised when RQ6 input violates the locked protocol."""


def _coerce_int(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise RQ6NegativeFPError(f"{field} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RQ6NegativeFPError(f"{field} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise RQ6NegativeFPError(f"{field} must be an integer")
    return parsed


def _coerce_finite_delta(value: object) -> float:
    if isinstance(value, bool):
        raise RQ6NegativeFPError(f"{RQ6_DELTA_COLUMN} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RQ6NegativeFPError(
            f"{RQ6_DELTA_COLUMN} must be numeric"
        ) from exc

    if not math.isfinite(parsed):
        raise RQ6NegativeFPError(
            f"{RQ6_DELTA_COLUMN} must be finite"
        )

    # maxDets=100 bounds FP/negative for each arm to [0,100],
    # therefore the paired SSL-SUP difference must be in [-100,100].
    if parsed < -100.0 or parsed > 100.0:
        raise RQ6NegativeFPError(
            f"{RQ6_DELTA_COLUMN} must be within [-100, 100]"
        )
    return parsed


def compute_rq6_seed_effects(
    rows: Sequence[Mapping[str, object]],
) -> dict[int, float]:
    """Compute H_s from exactly 8 paired FP/negative cells per seed."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise RQ6NegativeFPError("rows must be a sequence of mappings")

    expected_row_count = OFFICIAL_SEED_COUNT * RQ6_CELL_COUNT
    if len(rows) != expected_row_count:
        raise RQ6NegativeFPError(
            f"RQ6 requires exactly {expected_row_count} rows; "
            f"observed {len(rows)}"
        )

    expected_seed_index = {
        int(seed): index
        for index, seed in enumerate(OFFICIAL_TRAINING_SEEDS, start=1)
    }
    expected_cells = {
        (architecture, budget)
        for architecture in RQ6_ARCHITECTURES
        for budget in RQ6_BUDGETS
    }

    by_seed: dict[int, dict[tuple[str, str], float]] = {
        int(seed): {} for seed in OFFICIAL_TRAINING_SEEDS
    }
    seen: set[tuple[str, str, int]] = set()

    required_fields = {
        "architecture",
        "budget",
        "seed_index",
        "training_seed",
        RQ6_DELTA_COLUMN,
    }

    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise RQ6NegativeFPError(
                f"row {row_number} must be a mapping"
            )

        missing = sorted(required_fields.difference(row))
        if missing:
            raise RQ6NegativeFPError(
                f"row {row_number} missing required fields: {missing}"
            )

        architecture = str(row["architecture"])
        budget = str(row["budget"])
        seed_index = _coerce_int(row["seed_index"], "seed_index")
        training_seed = _coerce_int(row["training_seed"], "training_seed")
        delta = _coerce_finite_delta(row[RQ6_DELTA_COLUMN])

        if architecture not in RQ6_ARCHITECTURES:
            raise RQ6NegativeFPError(
                f"invalid architecture: {architecture}"
            )
        if budget not in RQ6_BUDGETS:
            raise RQ6NegativeFPError(f"invalid budget: {budget}")
        if training_seed not in expected_seed_index:
            raise RQ6NegativeFPError(
                f"unexpected training_seed: {training_seed}"
            )
        if seed_index != expected_seed_index[training_seed]:
            raise RQ6NegativeFPError(
                "seed_index does not match locked training_seed mapping"
            )

        identity = (architecture, budget, training_seed)
        if identity in seen:
            raise RQ6NegativeFPError(
                "duplicate architecture x budget x training_seed row"
            )
        seen.add(identity)

        by_seed[training_seed][(architecture, budget)] = delta

    result: dict[int, float] = {}
    for training_seed in OFFICIAL_TRAINING_SEEDS:
        seed = int(training_seed)
        observed_cells = set(by_seed[seed])
        if observed_cells != expected_cells:
            raise RQ6NegativeFPError(
                f"training_seed {seed} does not contain the exact "
                f"{RQ6_CELL_COUNT} architecture x budget cells"
            )

        values = np.asarray(
            [
                by_seed[seed][(architecture, budget)]
                for architecture in RQ6_ARCHITECTURES
                for budget in RQ6_BUDGETS
            ],
            dtype=float,
        )
        result[seed] = float(values.mean())

    return result


def rq6_primary_test(
    seed_effects: Mapping[int, float],
) -> dict[str, object]:
    """Run the locked two-sided one-sample t-test on H_s."""

    summary = summarize_seed_runs(seed_effects)

    values = np.asarray(
        [seed_effects[int(seed)] for seed in OFFICIAL_TRAINING_SEEDS],
        dtype=float,
    )
    result = stats.ttest_1samp(
        values,
        popmean=RQ6_NULL_MEAN,
        alternative=RQ6_ALTERNATIVE,
    )

    n = int(summary["n"])
    df = int(result.df)
    if n != OFFICIAL_SEED_COUNT or df != RQ6_DF:
        raise RQ6NegativeFPError(
            f"RQ6 requires n={OFFICIAL_SEED_COUNT} and df={RQ6_DF}; "
            f"observed n={n}, df={df}"
        )

    effect_estimate = float(summary["mean"])
    sample_sd = float(summary["sample_sd"])
    standard_error = sample_sd / math.sqrt(n)

    critical_value = float(
        stats.t.ppf(1.0 - RQ6_ALPHA / 2.0, df=df)
    )
    ci_low = effect_estimate - critical_value * standard_error
    ci_high = effect_estimate + critical_value * standard_error

    t_statistic = float(result.statistic)
    p_value = float(result.pvalue)
    if not all(
        math.isfinite(value)
        for value in (
            effect_estimate,
            sample_sd,
            standard_error,
            t_statistic,
            p_value,
            ci_low,
            ci_high,
        )
    ):
        raise RQ6NegativeFPError("RQ6 inference must be finite")

    return {
        "replication_unit": REPLICATION_UNIT,
        "effect": RQ6_EFFECT,
        "primitive_effect": "FP_SSL_NEG_MINUS_FP_SUP_NEG",
        "primary_metric": RQ6_PRIMARY_METRIC,
        "cell_count_per_seed": RQ6_CELL_COUNT,
        "cell_weight": RQ6_CELL_WEIGHT,
        "effect_estimate": effect_estimate,
        "sample_sd": sample_sd,
        "sample_sd_ddof": int(summary["sample_sd_ddof"]),
        "n": n,
        "df": df,
        "null_mean": RQ6_NULL_MEAN,
        "alternative": RQ6_ALTERNATIVE,
        "t_statistic": t_statistic,
        "p_value": p_value,
        "p_value_sidedness": "two-sided",
        "ci_level": RQ6_CI_LEVEL,
        "ci_sidedness": "two-sided",
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "negative_effect_interpretation": "SSL_LESS_FP_PER_NEGATIVE",
        "positive_effect_interpretation": "SSL_MORE_FP_PER_NEGATIVE",
        "far_secondary_outcome": RQ6_FAR_SECONDARY_OUTCOME,
        "far_formal_p_value_required": RQ6_FAR_FORMAL_P_VALUE_REQUIRED,
    }
