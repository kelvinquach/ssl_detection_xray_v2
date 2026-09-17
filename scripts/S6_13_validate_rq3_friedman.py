from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import scipy

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.statistics.rq3_friedman import (
    FRIEDMAN_DF,
    FRIEDMAN_MULTIPLICITY_FAMILY_ADDED,
    FRIEDMAN_NULL,
    FRIEDMAN_POSTHOC,
    FRIEDMAN_PRIMARY_METHOD,
    FRIEDMAN_PRIMARY_METHOD_UNCHANGED,
    FRIEDMAN_P_VALUE_FORM,
    FRIEDMAN_ROLE,
    FRIEDMAN_TIE_POLICY,
    FriedmanSensitivityError,
    rq3_friedman_sensitivity,
)
from src.statistics.rq3_primary import RQ3_BUDGET_LEVELS
from src.statistics.seed_summary import (
    OFFICIAL_TRAINING_SEEDS,
    REPLICATION_UNIT,
)

RQ3_FRIEDMAN_SHA256 = '499630d1983030addb51b529519a92cef34bb3840d22e0784e7f7319fff2fc1f'
RQ3_PRIMARY_SHA256 = '41ac57f7c4f9c19daef8f4f062001bcff1ff41b0af7b4683eb3459e436bf3676'
SEED_SUMMARY_SHA256 = '2a404ac0f6097cad16816b550fcd1c9ca240fab5535854dc0fd5e4433e49ff3f'

CHECKS: list[dict[str, object]] = []


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(check_id: str, condition: bool, detail: object) -> None:
    CHECKS.append({
        'check_id': check_id,
        'status': 'PASS' if condition else 'FAIL',
        'detail': detail,
    })


def expect_error(check_id: str, fn, detail: str) -> None:
    try:
        fn()
    except FriedmanSensitivityError:
        check(check_id, True, detail)
    else:
        check(check_id, False, detail)


def fixture_from_columns(*columns: list[float]) -> dict[int, tuple[float, ...]]:
    return {
        int(seed): tuple(float(column[index]) for column in columns)
        for index, seed in enumerate(OFFICIAL_TRAINING_SEEDS)
    }


def main() -> int:
    no_tie = fixture_from_columns(
        [1,2,3,4,5,6,7,8,9,10],
        [2,3,4,5,6,7,8,9,10,11],
        [3,4,5,6,7,8,9,10,11,12],
        [4,5,6,7,8,9,10,11,12,13],
    )
    tie = fixture_from_columns(
        [1,1,2,2,3,3,4,4,5,5],
        [1,2,2,3,3,4,4,5,5,6],
        [2,2,3,3,4,4,5,5,6,6],
        [2,3,3,4,4,5,5,6,6,7],
    )

    no_tie_result = rq3_friedman_sensitivity(no_tie)
    tie_result = rq3_friedman_sensitivity(tie)

    check('C01', tuple(RQ3_BUDGET_LEVELS) == ('1%', '5%', '10%', '20%'), list(RQ3_BUDGET_LEVELS))
    check('C02', FRIEDMAN_DF == 3, FRIEDMAN_DF)
    check('C03', FRIEDMAN_ROLE == 'SENSITIVITY_ONLY', FRIEDMAN_ROLE)
    check('C04', FRIEDMAN_NULL == 'NO_SYSTEMATIC_DIFFERENCE_IN_WITHIN_SEED_RANK_STRUCTURE_ACROSS_BUDGETS', FRIEDMAN_NULL)
    check('C05', FRIEDMAN_P_VALUE_FORM == 'RIGHT_TAIL_ASYMPTOTIC_CHI_SQUARE', FRIEDMAN_P_VALUE_FORM)
    check('C06', FRIEDMAN_TIE_POLICY == 'STANDARD_FRIEDMAN_TIE_CORRECTION', FRIEDMAN_TIE_POLICY)
    check('C07', FRIEDMAN_POSTHOC == 'NONE', FRIEDMAN_POSTHOC)
    check('C08', FRIEDMAN_MULTIPLICITY_FAMILY_ADDED is False, FRIEDMAN_MULTIPLICITY_FAMILY_ADDED)
    check('C09', FRIEDMAN_PRIMARY_METHOD == 'GREENHOUSE_GEISSER_REPEATED_MEASURES_ANOVA', FRIEDMAN_PRIMARY_METHOD)
    check('C10', FRIEDMAN_PRIMARY_METHOD_UNCHANGED is True, FRIEDMAN_PRIMARY_METHOD_UNCHANGED)

    check('C11', no_tie_result['input_effect'] == 'B_b_s', no_tie_result['input_effect'])
    check('C12', no_tie_result['inferential_replication_unit'] == REPLICATION_UNIT, no_tie_result['inferential_replication_unit'])
    check('C13', no_tie_result['n_subjects'] == 10, no_tie_result['n_subjects'])
    check('C14', no_tie_result['budget_count'] == 4, no_tie_result['budget_count'])
    check('C15', no_tie_result['budget_levels'] == ['1%', '5%', '10%', '20%'], no_tie_result['budget_levels'])
    check('C16', no_tie_result['statistic_name'] == 'friedman_chi_square', no_tie_result['statistic_name'])
    check('C17', math.isclose(no_tie_result['statistic'], 30.0, rel_tol=0.0, abs_tol=1e-12), no_tie_result['statistic'])
    check('C18', math.isclose(no_tie_result['p_value'], 1.3800570312932553e-06, rel_tol=1e-12, abs_tol=1e-18), no_tie_result['p_value'])
    check('C19', no_tie_result['df'] == 3, no_tie_result['df'])
    check('C20', no_tie_result['p_value_form'] == 'RIGHT_TAIL_ASYMPTOTIC_CHI_SQUARE', no_tie_result['p_value_form'])
    check('C21', no_tie_result['role'] == 'SENSITIVITY_ONLY', no_tie_result['role'])
    check('C22', no_tie_result['posthoc'] == 'NONE', no_tie_result['posthoc'])
    check('C23', no_tie_result['multiplicity_family_added'] is False, no_tie_result['multiplicity_family_added'])
    check('C24', no_tie_result['primary_method_unchanged'] is True, no_tie_result['primary_method_unchanged'])

    check('C25', math.isclose(tie_result['statistic'], 25.588235294117649, rel_tol=1e-12, abs_tol=1e-12), tie_result['statistic'])
    check('C26', math.isclose(tie_result['p_value'], 1.163105756259275e-05, rel_tol=1e-12, abs_tol=1e-18), tie_result['p_value'])
    check('C27', tie_result['tie_handling'] == 'STANDARD_FRIEDMAN_TIE_CORRECTION', tie_result['tie_handling'])

    missing = dict(no_tie)
    missing.pop(OFFICIAL_TRAINING_SEEDS[-1])
    expect_error('C28', lambda: rq3_friedman_sensitivity(missing), 'missing official seed rejected')
    expect_error('C29', lambda: rq3_friedman_sensitivity(dict(reversed(list(no_tie.items())))), 'wrong seed order rejected')
    wrong_budget_count = dict(no_tie)
    wrong_budget_count[OFFICIAL_TRAINING_SEEDS[0]] = (1.0, 2.0, 3.0)
    expect_error('C30', lambda: rq3_friedman_sensitivity(wrong_budget_count), 'wrong budget count rejected')
    nonnumeric = dict(no_tie)
    nonnumeric[OFFICIAL_TRAINING_SEEDS[0]] = ('x', 2.0, 3.0, 4.0)
    expect_error('C31', lambda: rq3_friedman_sensitivity(nonnumeric), 'nonnumeric value rejected')
    nan_input = dict(no_tie)
    nan_input[OFFICIAL_TRAINING_SEEDS[0]] = (float('nan'), 2.0, 3.0, 4.0)
    expect_error('C32', lambda: rq3_friedman_sensitivity(nan_input), 'NaN rejected')
    inf_input = dict(no_tie)
    inf_input[OFFICIAL_TRAINING_SEEDS[0]] = (float('inf'), 2.0, 3.0, 4.0)
    expect_error('C33', lambda: rq3_friedman_sensitivity(inf_input), 'infinity rejected')
    flat = {seed: (1.0, 1.0, 1.0, 1.0) for seed in OFFICIAL_TRAINING_SEEDS}
    expect_error('C34', lambda: rq3_friedman_sensitivity(flat), 'non-finite degenerate Friedman result rejected')

    core_path = REPO_ROOT / 'src/statistics/rq3_friedman.py'
    primary_path = REPO_ROOT / 'src/statistics/rq3_primary.py'
    seed_path = REPO_ROOT / 'src/statistics/seed_summary.py'
    validator_path = Path(__file__).resolve()
    check('C35', sha256_file(core_path) == RQ3_FRIEDMAN_SHA256, sha256_file(core_path))
    check('C36', sha256_file(primary_path) == RQ3_PRIMARY_SHA256, sha256_file(primary_path))
    check('C37', sha256_file(seed_path) == SEED_SUMMARY_SHA256, sha256_file(seed_path))

    failed = [item for item in CHECKS if item['status'] != 'PASS']
    report = {
        'task': 'S6.13',
        'tracker_action': 'Implement Friedman RQ3',
        'status': 'PASS' if not failed else 'FAIL',
        'checks_passed': len(CHECKS) - len(failed),
        'checks_total': len(CHECKS),
        'failed_count': len(failed),
        'failed_checks': failed,
        'scientific_contract': {
            'input_effect': 'B_b_s',
            'inferential_replication_unit': REPLICATION_UNIT,
            'n_subjects': 10,
            'budget_levels': list(RQ3_BUDGET_LEVELS),
            'test': 'standard_friedman_rank_test',
            'statistic': 'friedman_chi_square',
            'df': 3,
            'p_value': 'right_tail_asymptotic_chi_square',
            'tie_handling': 'standard_friedman_tie_correction',
            'role': 'sensitivity_only',
            'posthoc': 'NONE',
            'new_multiplicity_family': 'NONE',
            'primary_method': 'Greenhouse-Geisser repeated-measures ANOVA',
            'primary_method_unchanged': True,
        },
        'positive_fixture': {
            'no_tie': no_tie_result,
            'with_ties': tie_result,
        },
        'checks': CHECKS,
        'source_sha256': {
            'src/statistics/rq3_friedman.py': sha256_file(core_path),
            'src/statistics/rq3_primary.py': sha256_file(primary_path),
            'src/statistics/seed_summary.py': sha256_file(seed_path),
            'scripts/S6_13_validate_rq3_friedman.py': sha256_file(validator_path),
        },
        'statistical_environment': {
            'scipy_version': scipy.__version__,
        },
        'fixture_only_no_official_results': True,
        'test_used': False,
        'hidden_unlabeled_gt_used': False,
        'official_training_authorized': False,
        'final_test_authorized': False,
    }

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f'S6_13_FRIEDMAN_FIXTURE={len(CHECKS) - len(failed)}/{len(CHECKS)}_PASS')
    return 0 if not failed else 1


if __name__ == '__main__':
    raise SystemExit(main())
