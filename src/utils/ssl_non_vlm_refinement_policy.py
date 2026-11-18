"""Threshold calculations for semi-supervised learning with non-VLM refinement.

SSL + REF means:
Semi-Supervised Learning with non-VLM pseudo-label refinement.

This module is independent of MMDetection and PyTorch.
It contains only the class-specific adaptive threshold calculation used
by the non-VLM pseudo-label refinement method.

Only confidence scores from earlier successful optimizer updates are
allowed when calculating the threshold for the current training batch.
"""

from math import floor, isfinite
from typing import Sequence


def linear_quantile(
    values: Sequence[float],
    quantile: float,
) -> float:
    """Calculate a quantile using linear interpolation."""
    if not isfinite(quantile) or not 0.0 <= quantile <= 1.0:
        raise ValueError(
            "quantile must be a finite value between 0 and 1"
        )

    if not values:
        raise ValueError(
            "values must not be empty"
        )

    ordered_values = sorted(
        float(value)
        for value in values
    )

    if any(
        not isfinite(value)
        for value in ordered_values
    ):
        raise ValueError(
            "values must contain only finite numbers"
        )

    position = (
        (len(ordered_values) - 1)
        * quantile
    )

    lower_index = floor(position)

    upper_index = min(
        lower_index + 1,
        len(ordered_values) - 1,
    )

    if lower_index == upper_index:
        return ordered_values[lower_index]

    interpolation_fraction = (
        position - lower_index
    )

    lower_value = ordered_values[lower_index]
    upper_value = ordered_values[upper_index]

    return (
        lower_value
        + interpolation_fraction
        * (upper_value - lower_value)
    )


def adaptive_class_threshold(
    previous_committed_scores: Sequence[float],
    *,
    global_threshold: float = 0.90,
    minimum_threshold: float = 0.70,
    maximum_threshold: float = 0.90,
    shrinkage_strength: int = 30,
    target_quantile: float = 0.80,
    minimum_history_size: int = 32,
) -> float:
    """Calculate the class-specific pseudo-label confidence threshold.

    ``previous_committed_scores`` contains Teacher confidence scores from
    earlier optimizer updates that were successfully executed.

    Scores from the current training batch must not be included when
    calculating the threshold for that same batch.
    """
    committed_scores = [
        float(score)
        for score in previous_committed_scores
    ]

    if any(
        not isfinite(score)
        for score in committed_scores
    ):
        raise ValueError(
            "previous_committed_scores must contain only finite values"
        )

    if any(
        score < 0.0 or score > 1.0
        for score in committed_scores
    ):
        raise ValueError(
            "previous_committed_scores must be between 0 and 1"
        )

    threshold_parameters = (
        global_threshold,
        minimum_threshold,
        maximum_threshold,
        target_quantile,
    )

    if any(
        not isfinite(value)
        for value in threshold_parameters
    ):
        raise ValueError(
            "threshold parameters must contain only finite values"
        )

    if not 0.0 <= global_threshold <= 1.0:
        raise ValueError(
            "global_threshold must be between 0 and 1"
        )

    if not 0.0 <= minimum_threshold <= 1.0:
        raise ValueError(
            "minimum_threshold must be between 0 and 1"
        )

    if not 0.0 <= maximum_threshold <= 1.0:
        raise ValueError(
            "maximum_threshold must be between 0 and 1"
        )

    if minimum_threshold > maximum_threshold:
        raise ValueError(
            "minimum_threshold must not exceed maximum_threshold"
        )

    if not 0.0 <= target_quantile <= 1.0:
        raise ValueError(
            "target_quantile must be between 0 and 1"
        )

    if shrinkage_strength <= 0:
        raise ValueError(
            "shrinkage_strength must be positive"
        )

    if minimum_history_size <= 0:
        raise ValueError(
            "minimum_history_size must be positive"
        )

    history_size = len(
        committed_scores
    )

    if history_size >= minimum_history_size:
        history_quantile = linear_quantile(
            committed_scores,
            target_quantile,
        )
    else:
        history_quantile = global_threshold

    history_weight = (
        history_size
        / (
            history_size
            + shrinkage_strength
        )
    )

    threshold = (
        history_weight
        * history_quantile
        + (
            1.0 - history_weight
        )
        * global_threshold
    )

    threshold = max(
        minimum_threshold,
        threshold,
    )

    threshold = min(
        maximum_threshold,
        threshold,
    )

    return threshold
