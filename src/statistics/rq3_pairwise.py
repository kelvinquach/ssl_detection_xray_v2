"""RQ3 prespecified paired budget contrasts for S6.04.

Locked semantics:
- input is B_{b,s} from the S6.03 budget-gain definition;
- exactly six prespecified budget contrasts;
- D_{c,s} = B_{b2,s} - B_{b1,s};
- inferential unit is training seed / trained run;
- n=10 paired training seeds and df=9;
- two-sided one-sample t-test on paired differences against zero;
- mathematically equivalent to a paired t-test between the two budgets;
- sample SD uses ddof=1;
- each contrast reports an individual two-sided 95% CI;
- S6.04 reports raw two-sided p-values only;
- Holm F3 adjustment is deferred to S6.10.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from scipy import stats

from .rq3_primary import RQ3_BUDGET_LEVELS
from .seed_summary import (
    OFFICIAL_SEED_COUNT,
    OFFICIAL_TRAINING_SEEDS,
    REPLICATION_UNIT,
    SAMPLE_SD_DDOF,
    summarize_seed_runs,
)

RQ3_PAIRWISE_CONTRASTS = (
    ("5%-1%", "5%", "1%"),
    ("10%-1%", "10%", "1%"),
    ("20%-1%", "20%", "1%"),
    ("10%-5%", "10%", "5%"),
    ("20%-5%", "20%", "5%"),
    ("20%-10%", "20%", "10%"),
)
RQ3_PAIRWISE_CONTRAST_COUNT = 6
RQ3_PAIRWISE_NULL_MEAN = 0.0
RQ3_PAIRWISE_ALTERNATIVE = "two-sided"
RQ3_PAIRWISE_ALPHA = 0.05
RQ3_PAIRWISE_CI_LEVEL = 0.95
RQ3_PAIRWISE_DF = OFFICIAL_SEED_COUNT - 1
RQ3_PAIRWISE_HOLM_APPLIED = False
RQ3_PAIRWISE_HOLM_DEFERRED_TASK = "S6.10"


class RQ3PairwiseError(ValueError):
    """Raised when RQ3 pairwise input violates the locked protocol."""


def _validate_seed_budget_gains(
    seed_budget_gains: Mapping[int, Sequence[float]],
) -> np.ndarray:
    """Validate exact seed order and four ordered budget gains per seed."""
    if not isinstance(seed_budget_gains, Mapping):
        raise RQ3PairwiseError(
            "seed_budget_gains must map training_seed -> 4 ordered budget gains"
        )

    observed_seeds = tuple(seed_budget_gains.keys())
    if observed_seeds != OFFICIAL_TRAINING_SEEDS:
        raise RQ3PairwiseError(
            "training seeds must match the exact locked ordered list; "
            f"expected={OFFICIAL_TRAINING_SEEDS}, observed={observed_seeds}"
        )

    rows: list[np.ndarray] = []
    for seed in OFFICIAL_TRAINING_SEEDS:
        try:
            values = np.asarray(seed_budget_gains[seed], dtype=float)
        except (TypeError, ValueError) as exc:
            raise RQ3PairwiseError(
                f"seed {seed} must provide exactly four numeric budget gains"
            ) from exc

        if values.ndim != 1 or values.shape != (len(RQ3_BUDGET_LEVELS),):
            raise RQ3PairwiseError(
                f"seed {seed} must provide shape ({len(RQ3_BUDGET_LEVELS)},); "
                f"observed={values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise RQ3PairwiseError(
                f"seed {seed} contains non-finite budget gain values"
            )
        rows.append(values)

    matrix = np.vstack(rows)
    expected_shape = (OFFICIAL_SEED_COUNT, len(RQ3_BUDGET_LEVELS))
    if matrix.shape != expected_shape:
        raise RQ3PairwiseError(
            f"expected seed-budget matrix shape {expected_shape}; observed={matrix.shape}"
        )
    return matrix


def rq3_pairwise_tests(
    seed_budget_gains: Mapping[int, Sequence[float]],
) -> dict[str, object]:
    """Run the six locked two-sided paired-difference t-tests without Holm."""
    values = _validate_seed_budget_gains(seed_budget_gains)
    budget_index = {budget: i for i, budget in enumerate(RQ3_BUDGET_LEVELS)}
    results: list[dict[str, object]] = []

    for contrast_name, budget_2, budget_1 in RQ3_PAIRWISE_CONTRASTS:
        differences = values[:, budget_index[budget_2]] - values[:, budget_index[budget_1]]
        seed_differences = {
            seed: float(value)
            for seed, value in zip(OFFICIAL_TRAINING_SEEDS, differences, strict=True)
        }
        summary = summarize_seed_runs(seed_differences)
        n = int(summary["n"])
        sample_sd = float(summary["sample_sd"])
        if n != OFFICIAL_SEED_COUNT:
            raise RQ3PairwiseError(
                f"contrast {contrast_name} requires n={OFFICIAL_SEED_COUNT}; observed={n}"
            )
        if not math.isfinite(sample_sd) or sample_sd <= 0.0:
            raise RQ3PairwiseError(
                f"contrast {contrast_name} requires positive finite paired-difference SD"
            )

        test = stats.ttest_1samp(
            differences,
            popmean=RQ3_PAIRWISE_NULL_MEAN,
            alternative=RQ3_PAIRWISE_ALTERNATIVE,
        )
        df = int(test.df)
        if df != RQ3_PAIRWISE_DF:
            raise RQ3PairwiseError(
                f"contrast {contrast_name} requires df={RQ3_PAIRWISE_DF}; observed={df}"
            )

        effect_estimate = float(summary["mean"])
        standard_error = sample_sd / math.sqrt(n)
        critical_value = float(
            stats.t.ppf(1.0 - RQ3_PAIRWISE_ALPHA / 2.0, df=df)
        )
        ci_low = effect_estimate - critical_value * standard_error
        ci_high = effect_estimate + critical_value * standard_error
        t_statistic = float(test.statistic)
        raw_p_value = float(test.pvalue)

        if not all(
            math.isfinite(value)
            for value in (effect_estimate, t_statistic, raw_p_value, ci_low, ci_high)
        ):
            raise RQ3PairwiseError(
                f"contrast {contrast_name} produced non-finite inferential output"
            )

        results.append(
            {
                "contrast": contrast_name,
                "budget_2": budget_2,
                "budget_1": budget_1,
                "direction": "b2-b1",
                "effect": "D_c_s",
                "paired_differences_by_seed": seed_differences,
                "effect_estimate": effect_estimate,
                "sample_sd": sample_sd,
                "sample_sd_ddof": int(summary["sample_sd_ddof"]),
                "n": n,
                "df": df,
                "null_mean": RQ3_PAIRWISE_NULL_MEAN,
                "alternative": RQ3_PAIRWISE_ALTERNATIVE,
                "t_statistic": t_statistic,
                "raw_p_value": raw_p_value,
                "p_value_sidedness": "two-sided",
                "ci_level": RQ3_PAIRWISE_CI_LEVEL,
                "ci_sidedness": "two-sided",
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
            }
        )

    return {
        "replication_unit": REPLICATION_UNIT,
        "budget_levels": list(RQ3_BUDGET_LEVELS),
        "contrast_count": RQ3_PAIRWISE_CONTRAST_COUNT,
        "contrasts": results,
        "holm_applied": RQ3_PAIRWISE_HOLM_APPLIED,
        "holm_deferred_task": RQ3_PAIRWISE_HOLM_DEFERRED_TASK,
    }
