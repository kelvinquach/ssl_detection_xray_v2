"""Exact sign-flip robustness analysis for S6.11.

Locked contract:
- exactly 10 ordered training-seed paired effects;
- exhaustive 2^10 = 1024 sign configurations;
- observed statistic = arithmetic mean of the 10 paired effects;
- RQ2: one-sided greater, count(T_flip >= T_obs) / 1024;
- RQ6/RQ7: two-sided, count(abs(T_flip) >= abs(T_obs)) / 1024;
- equality/ties are included;
- no +1 correction;
- zero effects retain their seed and all 1024 configurations.
"""

from __future__ import annotations

import math
from itertools import product
from typing import Mapping

from .seed_summary import OFFICIAL_SEED_COUNT, OFFICIAL_TRAINING_SEEDS

SIGN_CONFIGURATION_COUNT = 2 ** OFFICIAL_SEED_COUNT
TEST_STATISTIC = "arithmetic_mean_of_paired_seed_effects"
REPLICATION_UNIT = "TRAINING_SEED_TRAINED_RUN"

_RQ_CONTRACT = {
    "RQ2": {"effect": "G_s", "alternative": "one-sided-greater"},
    "RQ6": {"effect": "H_s", "alternative": "two-sided"},
    "RQ7": {"effect": "D_s", "alternative": "two-sided"},
}


class ExactSignFlipError(ValueError):
    """Raised when input violates the locked S6.11 contract."""


def _coerce_finite(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ExactSignFlipError(f"{field} must be numeric, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ExactSignFlipError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise ExactSignFlipError(f"{field} must be finite")
    return result


def exact_sign_flip_test(
    seed_effects: Mapping[int, float],
    *,
    rq: str,
) -> dict[str, object]:
    """Run the locked exhaustive exact sign-flip robustness test."""
    if rq not in _RQ_CONTRACT:
        raise ExactSignFlipError("rq must be exactly one of RQ2, RQ6, RQ7")
    if not isinstance(seed_effects, Mapping):
        raise ExactSignFlipError("seed_effects must be a mapping")

    observed_seeds = tuple(seed_effects.keys())
    if observed_seeds != OFFICIAL_TRAINING_SEEDS:
        raise ExactSignFlipError(
            "training seeds must match the exact locked ordered list; "
            f"expected={OFFICIAL_TRAINING_SEEDS}, observed={observed_seeds}"
        )

    values = tuple(
        _coerce_finite(seed_effects[int(seed)], f"seed_effect[{seed}]")
        for seed in OFFICIAL_TRAINING_SEEDS
    )
    if len(values) != OFFICIAL_SEED_COUNT:
        raise ExactSignFlipError(
            f"exact sign-flip requires exactly {OFFICIAL_SEED_COUNT} seed effects"
        )

    observed_statistic = math.fsum(values) / OFFICIAL_SEED_COUNT
    contract = _RQ_CONTRACT[rq]
    alternative = contract["alternative"]

    extreme_configuration_count = 0
    enumerated_configuration_count = 0
    observed_configuration_found = False

    for signs in product((-1.0, 1.0), repeat=OFFICIAL_SEED_COUNT):
        enumerated_configuration_count += 1
        if all(sign == 1.0 for sign in signs):
            observed_configuration_found = True
        flipped_statistic = math.fsum(
            sign * value for sign, value in zip(signs, values)
        ) / OFFICIAL_SEED_COUNT

        if alternative == "one-sided-greater":
            is_extreme = flipped_statistic >= observed_statistic
        else:
            is_extreme = abs(flipped_statistic) >= abs(observed_statistic)

        if is_extreme:
            extreme_configuration_count += 1

    if enumerated_configuration_count != SIGN_CONFIGURATION_COUNT:
        raise ExactSignFlipError(
            "exhaustive enumeration did not produce exactly 1024 configurations"
        )
    if not observed_configuration_found:
        raise ExactSignFlipError("all-positive observed configuration was not enumerated")

    exact_p_value = extreme_configuration_count / SIGN_CONFIGURATION_COUNT
    zero_effect_count = sum(value == 0.0 for value in values)

    extremeness_rule = (
        "T_flip >= T_obs"
        if alternative == "one-sided-greater"
        else "abs(T_flip) >= abs(T_obs)"
    )

    return {
        "rq": rq,
        "effect": contract["effect"],
        "ordered_training_seeds": [int(seed) for seed in OFFICIAL_TRAINING_SEEDS],
        "seed_effect_count": OFFICIAL_SEED_COUNT,
        "seed_effects": list(values),
        "replication_unit": REPLICATION_UNIT,
        "test_statistic": TEST_STATISTIC,
        "observed_statistic": float(observed_statistic),
        "sign_configuration_count": SIGN_CONFIGURATION_COUNT,
        "enumerated_configuration_count": enumerated_configuration_count,
        "observed_all_positive_configuration_included": observed_configuration_found,
        "alternative": alternative,
        "extremeness_rule": extremeness_rule,
        "equality_ties_included": True,
        "plus_one_correction": False,
        "zero_effect_seed_retained": True,
        "zero_effect_count": zero_effect_count,
        "duplicate_statistics_counted_by_configuration": True,
        "extreme_configuration_count": extreme_configuration_count,
        "exact_p_value": float(exact_p_value),
    }

