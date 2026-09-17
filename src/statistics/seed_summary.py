"""Seed-level descriptive statistics for the locked SSOD protocol.

S6.01 scientific contract:
- inferential replication unit: training seed / trained run;
- official aggregation uses the exact prespecified ordered 10-seed list;
- arithmetic mean across seed-level scalar results;
- sample standard deviation with ddof = 1;
- image, bbox, detection, and pooled predictions are not replications.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


REPLICATION_UNIT = "TRAINING_SEED_TRAINED_RUN"
OFFICIAL_TRAINING_SEEDS = (
    204886845,
    1480646854,
    1798418854,
    2045683682,
    1814859839,
    1603952859,
    1878351743,
    875651179,
    477581743,
    869675675,
)
OFFICIAL_SEED_COUNT = 10
SAMPLE_SD_DDOF = 1
CROSS_SEED_PREDICTION_POOLING_ALLOWED = False


class SeedSummaryError(ValueError):
    """Raised when seed-level input violates the locked protocol."""


def summarize_seed_runs(
    seed_values: Mapping[int, float],
) -> dict[str, object]:
    """Summarize exactly one scalar result from each official training seed."""
    if not isinstance(seed_values, Mapping):
        raise SeedSummaryError(
            "seed_values must be a mapping of training_seed -> scalar metric"
        )

    observed_seeds = tuple(seed_values.keys())
    if observed_seeds != OFFICIAL_TRAINING_SEEDS:
        raise SeedSummaryError(
            "training seeds must match the exact locked ordered list; "
            f"expected={OFFICIAL_TRAINING_SEEDS}, observed={observed_seeds}"
        )

    values = np.asarray(list(seed_values.values()), dtype=float)
    if values.ndim != 1 or values.size != OFFICIAL_SEED_COUNT:
        raise SeedSummaryError(
            "each official training seed must contribute exactly one scalar metric"
        )

    if not np.all(np.isfinite(values)):
        raise SeedSummaryError("seed-level metric values must all be finite")

    mean = float(np.mean(values))
    sample_sd = float(np.std(values, ddof=SAMPLE_SD_DDOF))

    return {
        "replication_unit": REPLICATION_UNIT,
        "n": int(values.size),
        "training_seeds": list(OFFICIAL_TRAINING_SEEDS),
        "mean": mean,
        "sample_sd": sample_sd,
        "sample_sd_ddof": SAMPLE_SD_DDOF,
        "sample_sd_denominator": int(values.size - 1),
        "cross_seed_prediction_pooling_allowed":
            CROSS_SEED_PREDICTION_POOLING_ALLOWED,
    }
