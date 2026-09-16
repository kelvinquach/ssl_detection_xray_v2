"""Leave-one-seed-out robustness analysis for S6.12.

Locked contract:
- exactly 10 ordered training-seed paired effects;
- exactly 10 LOSO analyses;
- each analysis temporarily omits exactly one training seed;
- the effect is recomputed as the arithmetic mean of the remaining 9 seed effects;
- LOSO evaluates seed influence on effect direction and magnitude;
- LOSO never excludes a seed from the primary analysis;
- no LOSO p-value, confidence interval, or replacement primary test is introduced here.
"""

from __future__ import annotations

import math
from typing import Mapping

from .seed_summary import OFFICIAL_SEED_COUNT, OFFICIAL_TRAINING_SEEDS

LOSO_ANALYSIS_COUNT = OFFICIAL_SEED_COUNT
LOSO_RETAINED_SEED_COUNT = OFFICIAL_SEED_COUNT - 1
RECOMPUTED_STATISTIC = "arithmetic_mean_of_remaining_seed_effects"
REPLICATION_UNIT = "TRAINING_SEED_TRAINED_RUN"

_RQ_CONTRACT = {
    "RQ2": {"effect": "G_s"},
    "RQ6": {"effect": "H_s"},
    "RQ7": {"effect": "D_s"},
}


class LOSORobustnessError(ValueError):
    """Raised when input violates the locked S6.12 LOSO contract."""


def _coerce_finite(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise LOSORobustnessError(f"{field} must be numeric, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise LOSORobustnessError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise LOSORobustnessError(f"{field} must be finite")
    return result


def leave_one_seed_out(
    seed_effects: Mapping[int, float],
    *,
    rq: str,
) -> dict[str, object]:
    """Run the locked exactly-10 leave-one-seed-out robustness analysis."""
    if rq not in _RQ_CONTRACT:
        raise LOSORobustnessError("rq must be exactly one of RQ2, RQ6, RQ7")
    if not isinstance(seed_effects, Mapping):
        raise LOSORobustnessError("seed_effects must be a mapping")

    observed_seeds = tuple(seed_effects.keys())
    if observed_seeds != OFFICIAL_TRAINING_SEEDS:
        raise LOSORobustnessError(
            "training seeds must match the exact locked ordered list; "
            f"expected={OFFICIAL_TRAINING_SEEDS}, observed={observed_seeds}"
        )

    values = tuple(
        _coerce_finite(seed_effects[int(seed)], f"seed_effect[{seed}]")
        for seed in OFFICIAL_TRAINING_SEEDS
    )
    if len(values) != OFFICIAL_SEED_COUNT:
        raise LOSORobustnessError(
            f"LOSO requires exactly {OFFICIAL_SEED_COUNT} seed effects"
        )

    full_effect = float(sum(values) / OFFICIAL_SEED_COUNT)
    analyses: list[dict[str, object]] = []

    for omitted_index, omitted_seed in enumerate(OFFICIAL_TRAINING_SEEDS):
        retained_seeds = tuple(
            int(seed)
            for index, seed in enumerate(OFFICIAL_TRAINING_SEEDS)
            if index != omitted_index
        )
        retained_values = tuple(
            values[index]
            for index in range(OFFICIAL_SEED_COUNT)
            if index != omitted_index
        )
        if len(retained_values) != LOSO_RETAINED_SEED_COUNT:
            raise LOSORobustnessError(
                "each LOSO analysis must retain exactly 9 seed effects"
            )

        recomputed_effect = float(
            sum(retained_values) / LOSO_RETAINED_SEED_COUNT
        )
        analyses.append(
            {
                "analysis_index": omitted_index + 1,
                "omitted_seed": int(omitted_seed),
                "retained_seed_count": LOSO_RETAINED_SEED_COUNT,
                "retained_training_seeds": list(retained_seeds),
                "recomputed_effect": recomputed_effect,
            }
        )

    if len(analyses) != LOSO_ANALYSIS_COUNT:
        raise LOSORobustnessError(
            f"LOSO requires exactly {LOSO_ANALYSIS_COUNT} analyses"
        )

    return {
        "rq": rq,
        "effect": _RQ_CONTRACT[rq]["effect"],
        "replication_unit": REPLICATION_UNIT,
        "ordered_training_seeds": list(OFFICIAL_TRAINING_SEEDS),
        "seed_effect_count": OFFICIAL_SEED_COUNT,
        "full_effect": full_effect,
        "loso_analysis_count": LOSO_ANALYSIS_COUNT,
        "retained_seed_count_per_analysis": LOSO_RETAINED_SEED_COUNT,
        "recomputed_statistic": RECOMPUTED_STATISTIC,
        "primary_seed_exclusion_allowed": False,
        "analyses": analyses,
    }
