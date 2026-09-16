from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

from scipy import stats

from src.statistics.rq3_primary import (
    RQ3_BUDGET_COUNT,
    RQ3_BUDGET_LEVELS,
    RQ3PrimaryError,
    _validate_seed_budget_gains,
 )
from src.statistics.seed_summary import (
    OFFICIAL_SEED_COUNT,
    REPLICATION_UNIT,
 )

FRIEDMAN_DF = RQ3_BUDGET_COUNT - 1
FRIEDMAN_ROLE = 'SENSITIVITY_ONLY'
FRIEDMAN_NULL = 'NO_SYSTEMATIC_DIFFERENCE_IN_WITHIN_SEED_RANK_STRUCTURE_ACROSS_BUDGETS'
FRIEDMAN_P_VALUE_FORM = 'RIGHT_TAIL_ASYMPTOTIC_CHI_SQUARE'
FRIEDMAN_TIE_POLICY = 'STANDARD_FRIEDMAN_TIE_CORRECTION'
FRIEDMAN_POSTHOC = 'NONE'
FRIEDMAN_MULTIPLICITY_FAMILY_ADDED = False
FRIEDMAN_PRIMARY_METHOD = 'GREENHOUSE_GEISSER_REPEATED_MEASURES_ANOVA'
FRIEDMAN_PRIMARY_METHOD_UNCHANGED = True


class FriedmanSensitivityError(ValueError):
    pass


def rq3_friedman_sensitivity(
    seed_budget_gains: Mapping[int, Sequence[float]],
) -> dict[str, object]:
    try:
        matrix = _validate_seed_budget_gains(seed_budget_gains)
    except RQ3PrimaryError as exc:
        raise FriedmanSensitivityError(str(exc)) from exc

    samples = [matrix[:, index] for index in range(RQ3_BUDGET_COUNT)]
    result = stats.friedmanchisquare(*samples)
    statistic = float(result.statistic)
    p_value = float(result.pvalue)

    if not math.isfinite(statistic) or not math.isfinite(p_value):
        raise FriedmanSensitivityError(
            'Friedman statistic and p-value must both be finite'
        )

    return {
        'input_effect': 'B_b_s',
        'inferential_replication_unit': REPLICATION_UNIT,
        'n_subjects': OFFICIAL_SEED_COUNT,
        'budget_count': RQ3_BUDGET_COUNT,
        'budget_levels': list(RQ3_BUDGET_LEVELS),
        'statistic_name': 'friedman_chi_square',
        'statistic': statistic,
        'df': FRIEDMAN_DF,
        'p_value': p_value,
        'p_value_form': FRIEDMAN_P_VALUE_FORM,
        'null_hypothesis': FRIEDMAN_NULL,
        'tie_handling': FRIEDMAN_TIE_POLICY,
        'role': FRIEDMAN_ROLE,
        'posthoc': FRIEDMAN_POSTHOC,
        'multiplicity_family_added': FRIEDMAN_MULTIPLICITY_FAMILY_ADDED,
        'primary_method': FRIEDMAN_PRIMARY_METHOD,
        'primary_method_unchanged': FRIEDMAN_PRIMARY_METHOD_UNCHANGED,
    }
