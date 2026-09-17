"""Locked RQ7 architecture-dependent SSL-effect statistical fixture.

Scientific contract:
- primitive paired effect is delta_mAP_test = SSL - SUP;
- for each architecture and training seed, A_{a,s} is the equal-weight mean
  across the four labeled-data budgets;
- D_s = A_{Swin-T,s} - A_{R50,s};
- inferential replication unit is training seed / trained run;
- official n = 10 and df = 9;
- primary inference is a two-sided one-sample t-test against zero;
- uncertainty is reported with a two-sided 95% confidence interval.
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

RQ7_ARCHITECTURES = ("R50", "Swin-T")
RQ7_BUDGETS = ("1%", "5%", "10%", "20%")
RQ7_BUDGET_COUNT = len(RQ7_BUDGETS)
RQ7_ROWS_PER_SEED = len(RQ7_ARCHITECTURES) * RQ7_BUDGET_COUNT
RQ7_TOTAL_ROWS = OFFICIAL_SEED_COUNT * RQ7_ROWS_PER_SEED
RQ7_DELTA_COLUMN = "delta_mAP_test"
RQ7_EFFECT = "D_s"
RQ7_NULL_MEAN = 0.0
RQ7_ALTERNATIVE = "two-sided"
RQ7_ALPHA = 0.05
RQ7_CI_LEVEL = 0.95
RQ7_DF = OFFICIAL_SEED_COUNT - 1


class RQ7ArchitectureEffectError(ValueError):
    """Raised when RQ7 input violates the locked protocol."""


def _coerce_int(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise RQ7ArchitectureEffectError(f"{field} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise RQ7ArchitectureEffectError(f"{field} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise RQ7ArchitectureEffectError(f"{field} must be an integer")
    return result


def _coerce_finite(value: object, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RQ7ArchitectureEffectError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise RQ7ArchitectureEffectError(f"{field} must be finite")
    return result


def compute_rq7_architecture_gain_by_seed(
    rows: Sequence[Mapping[str, object]],
) -> dict[int, dict[str, float]]:
    """Compute A_{a,s} for both architectures from the 80 paired rows."""
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise RQ7ArchitectureEffectError("rows must be a sequence of mappings")

    if len(rows) != RQ7_TOTAL_ROWS:
        raise RQ7ArchitectureEffectError(
            f"RQ7 requires exactly {RQ7_TOTAL_ROWS} rows; observed {len(rows)}"
        )

    expected_seed_index = {
        int(seed): index
        for index, seed in enumerate(OFFICIAL_TRAINING_SEEDS, start=1)
    }
    expected_cells = {
        (architecture, budget)
        for architecture in RQ7_ARCHITECTURES
        for budget in RQ7_BUDGETS
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
        RQ7_DELTA_COLUMN,
    }

    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise RQ7ArchitectureEffectError(
                f"row {row_number} must be a mapping"
            )
        missing = required_fields.difference(row)
        if missing:
            raise RQ7ArchitectureEffectError(
                f"row {row_number} missing fields: {sorted(missing)}"
            )

        architecture = str(row["architecture"])
        budget = str(row["budget"])
        seed_index = _coerce_int(row["seed_index"], "seed_index")
        training_seed = _coerce_int(row["training_seed"], "training_seed")
        delta = _coerce_finite(row[RQ7_DELTA_COLUMN], RQ7_DELTA_COLUMN)

        if architecture not in RQ7_ARCHITECTURES:
            raise RQ7ArchitectureEffectError(
                f"invalid architecture: {architecture}"
            )
        if budget not in RQ7_BUDGETS:
            raise RQ7ArchitectureEffectError(f"invalid budget: {budget}")
        if training_seed not in expected_seed_index:
            raise RQ7ArchitectureEffectError(
                f"unexpected training_seed: {training_seed}"
            )
        if seed_index != expected_seed_index[training_seed]:
            raise RQ7ArchitectureEffectError(
                "seed_index does not match locked training_seed mapping"
            )

        identity = (architecture, budget, training_seed)
        if identity in seen:
            raise RQ7ArchitectureEffectError(
                "duplicate architecture x budget x training_seed row"
            )
        seen.add(identity)
        by_seed[training_seed][(architecture, budget)] = delta

    result: dict[int, dict[str, float]] = {}
    for training_seed in OFFICIAL_TRAINING_SEEDS:
        seed = int(training_seed)
        observed_cells = set(by_seed[seed])
        if observed_cells != expected_cells:
            raise RQ7ArchitectureEffectError(
                f"training_seed {seed} does not contain the exact "
                f"{RQ7_ROWS_PER_SEED} architecture x budget cells"
            )

        architecture_gains: dict[str, float] = {}
        for architecture in RQ7_ARCHITECTURES:
            values = np.asarray(
                [
                    by_seed[seed][(architecture, budget)]
                    for budget in RQ7_BUDGETS
                ],
                dtype=float,
            )
            architecture_gains[architecture] = float(np.mean(values))
        result[seed] = architecture_gains

    return result


def compute_rq7_difference_by_seed(
    architecture_gains: Mapping[int, Mapping[str, float]],
) -> dict[int, float]:
    """Compute D_s = A_Swin-T,s - A_R50,s in locked seed order."""
    if not isinstance(architecture_gains, Mapping):
        raise RQ7ArchitectureEffectError(
            "architecture_gains must map training_seed to architecture gains"
        )

    observed_seeds = tuple(architecture_gains.keys())
    if observed_seeds != OFFICIAL_TRAINING_SEEDS:
        raise RQ7ArchitectureEffectError(
            "training seeds must match the exact locked ordered list; "
            f"expected={OFFICIAL_TRAINING_SEEDS}, observed={observed_seeds}"
        )

    result: dict[int, float] = {}
    for training_seed in OFFICIAL_TRAINING_SEEDS:
        gains = architecture_gains[int(training_seed)]
        if not isinstance(gains, Mapping):
            raise RQ7ArchitectureEffectError(
                f"training_seed {training_seed} architecture gains must be a mapping"
            )
        if set(gains.keys()) != set(RQ7_ARCHITECTURES):
            raise RQ7ArchitectureEffectError(
                f"training_seed {training_seed} must contain exactly R50 and Swin-T"
            )
        r50 = _coerce_finite(gains["R50"], "A_R50")
        swin = _coerce_finite(gains["Swin-T"], "A_Swin-T")
        result[int(training_seed)] = float(swin - r50)

    return result


def rq7_primary_test(seed_differences: Mapping[int, float]) -> dict[str, object]:
    """Run the locked two-sided one-sample t-test on D_s."""
    summary = summarize_seed_runs(seed_differences)

    values = np.asarray(
        [seed_differences[int(seed)] for seed in OFFICIAL_TRAINING_SEEDS],
        dtype=float,
    )
    result = stats.ttest_1samp(
        values,
        popmean=RQ7_NULL_MEAN,
        alternative=RQ7_ALTERNATIVE,
    )

    n = int(summary["n"])
    df = int(result.df)
    if n != OFFICIAL_SEED_COUNT or df != RQ7_DF:
        raise RQ7ArchitectureEffectError(
            f"RQ7 requires n={OFFICIAL_SEED_COUNT} and df={RQ7_DF}; "
            f"observed n={n}, df={df}"
        )

    effect_estimate = float(summary["mean"])
    sample_sd = float(summary["sample_sd"])
    if not math.isfinite(sample_sd) or sample_sd <= 0.0:
        raise RQ7ArchitectureEffectError(
            "RQ7 requires positive finite seed-level sample SD"
        )

    standard_error = sample_sd / math.sqrt(n)
    critical_value = float(
        stats.t.ppf(1.0 - RQ7_ALPHA / 2.0, df=df)
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
            t_statistic,
            p_value,
            ci_low,
            ci_high,
        )
    ):
        raise RQ7ArchitectureEffectError("RQ7 inference must be finite")

    return {
        "replication_unit": REPLICATION_UNIT,
        "effect": RQ7_EFFECT,
        "effect_definition": "A_Swin-T_minus_A_R50",
        "effect_estimate": effect_estimate,
        "sample_sd": sample_sd,
        "sample_sd_ddof": int(summary["sample_sd_ddof"]),
        "n": n,
        "df": df,
        "null_mean": RQ7_NULL_MEAN,
        "alternative": RQ7_ALTERNATIVE,
        "t_statistic": t_statistic,
        "p_value": p_value,
        "p_value_sidedness": "two-sided",
        "ci_level": RQ7_CI_LEVEL,
        "ci_sidedness": "two-sided",
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "interpretation_scope": (
            "DIFFERENCE_IN_SSL_GAIN_BETWEEN_ARCHITECTURES_NOT_ABSOLUTE_SUPERIORITY"
        ),
    }
