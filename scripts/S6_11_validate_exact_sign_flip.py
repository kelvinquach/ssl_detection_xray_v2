#!/usr/bin/env python3
"""Validate the locked S6.11 exact sign-flip implementation."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.statistics.exact_sign_flip import (
    ExactSignFlipError,
    SIGN_CONFIGURATION_COUNT,
    TEST_STATISTIC,
    exact_sign_flip_test,
)
from src.statistics.seed_summary import OFFICIAL_TRAINING_SEEDS


CHECKS: list[dict[str, object]] = []


def check(check_id: str, condition: bool, detail: str) -> None:
    CHECKS.append(
        {
            "check_id": check_id,
            "status": "PASS" if condition else "FAIL",
            "detail": detail,
        }
    )
    if not condition:
        raise AssertionError(f"{check_id}: {detail}")


def ordered(values: list[float]) -> dict[int, float]:
    return {
        int(seed): float(value)
        for seed, value in zip(OFFICIAL_TRAINING_SEEDS, values)
    }


def expect_error(check_id: str, func, expected_text: str) -> None:
    try:
        func()
    except ExactSignFlipError as exc:
        check(check_id, expected_text in str(exc), str(exc))
    else:
        check(check_id, False, "Expected ExactSignFlipError")


def main() -> int:
    positive = [float(i) for i in range(1, 11)]
    rq2 = exact_sign_flip_test(ordered(positive), rq="RQ2")
    rq6 = exact_sign_flip_test(ordered(positive), rq="RQ6")
    rq7 = exact_sign_flip_test(
        ordered([float(i) / 10.0 for i in range(1, 11)]),
        rq="RQ7",
    )

    check("C01", SIGN_CONFIGURATION_COUNT == 1024, "2^10 must equal 1024")
    check("C02", TEST_STATISTIC == "arithmetic_mean_of_paired_seed_effects", "locked test statistic")
    check("C03", rq2["rq"] == "RQ2" and rq2["effect"] == "G_s", "RQ2 effect identity")
    check("C04", rq6["rq"] == "RQ6" and rq6["effect"] == "H_s", "RQ6 effect identity")
    check("C05", rq7["rq"] == "RQ7" and rq7["effect"] == "D_s", "RQ7 effect identity")
    check("C06", rq2["alternative"] == "one-sided-greater", "RQ2 sidedness")
    check("C07", rq6["alternative"] == "two-sided", "RQ6 sidedness")
    check("C08", rq7["alternative"] == "two-sided", "RQ7 sidedness")
    check("C09", rq2["extremeness_rule"] == "T_flip >= T_obs", "RQ2 extremeness rule")
    check("C10", rq6["extremeness_rule"] == "abs(T_flip) >= abs(T_obs)", "RQ6 extremeness rule")
    check("C11", rq7["extremeness_rule"] == "abs(T_flip) >= abs(T_obs)", "RQ7 extremeness rule")
    check("C12", rq2["seed_effect_count"] == 10, "RQ2 seed count")
    check("C13", rq6["seed_effect_count"] == 10, "RQ6 seed count")
    check("C14", rq7["seed_effect_count"] == 10, "RQ7 seed count")
    check("C15", rq2["ordered_training_seeds"] == [int(x) for x in OFFICIAL_TRAINING_SEEDS], "locked seed order")
    check("C16", rq2["sign_configuration_count"] == 1024, "RQ2 sign configuration count")
    check("C17", rq6["sign_configuration_count"] == 1024, "RQ6 sign configuration count")
    check("C18", rq7["sign_configuration_count"] == 1024, "RQ7 sign configuration count")
    check("C19", rq2["enumerated_configuration_count"] == 1024, "RQ2 exhaustive enumeration")
    check("C20", rq6["enumerated_configuration_count"] == 1024, "RQ6 exhaustive enumeration")
    check("C21", rq7["enumerated_configuration_count"] == 1024, "RQ7 exhaustive enumeration")
    check("C22", rq2["observed_all_positive_configuration_included"] is True, "observed configuration included")
    check("C23", rq2["equality_ties_included"] is True, "ties included")
    check("C24", rq2["plus_one_correction"] is False, "no +1 correction")
    check("C25", rq2["zero_effect_seed_retained"] is True, "zero-effect seed retained")
    check("C26", rq2["duplicate_statistics_counted_by_configuration"] is True, "duplicate statistics counted by configuration")
    check("C27", math.isclose(rq2["observed_statistic"], 5.5, rel_tol=0.0, abs_tol=1e-15), "RQ2 observed mean")
    check("C28", rq2["extreme_configuration_count"] == 1, "RQ2 positive fixture extreme count")
    check("C29", math.isclose(rq2["exact_p_value"], 1.0 / 1024.0, rel_tol=0.0, abs_tol=1e-15), "RQ2 exact p")
    check("C30", rq6["extreme_configuration_count"] == 2, "RQ6 positive fixture extreme count")
    check("C31", math.isclose(rq6["exact_p_value"], 2.0 / 1024.0, rel_tol=0.0, abs_tol=1e-15), "RQ6 exact p")
    check("C32", math.isclose(rq7["observed_statistic"], 0.55, rel_tol=0.0, abs_tol=1e-15), "RQ7 observed mean")
    check("C33", rq7["extreme_configuration_count"] == 2, "RQ7 positive fixture extreme count")
    check("C34", math.isclose(rq7["exact_p_value"], 2.0 / 1024.0, rel_tol=0.0, abs_tol=1e-15), "RQ7 exact p")

    zeros = exact_sign_flip_test(ordered([0.0] * 10), rq="RQ2")
    check("C35", zeros["zero_effect_count"] == 10, "all zero effects retained")
    check("C36", zeros["seed_effect_count"] == 10, "zero fixture keeps n=10")
    check("C37", zeros["enumerated_configuration_count"] == 1024, "zero fixture still enumerates 1024")
    check("C38", zeros["extreme_configuration_count"] == 1024, "equality includes all zero configurations")
    check("C39", zeros["exact_p_value"] == 1.0, "all-zero exact p equals 1")

    one_zero = exact_sign_flip_test(
        ordered([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
        rq="RQ2",
    )
    check("C40", one_zero["zero_effect_count"] == 1, "one zero effect retained")
    check("C41", one_zero["extreme_configuration_count"] == 2, "zero-sign duplicate configurations counted separately")
    check("C42", math.isclose(one_zero["exact_p_value"], 2.0 / 1024.0, rel_tol=0.0, abs_tol=1e-15), "duplicate configurations remain in denominator")

    expect_error(
        "C43",
        lambda: exact_sign_flip_test(ordered(positive), rq="RQ3"),
        "rq must be exactly one of RQ2, RQ6, RQ7",
    )
    reversed_mapping = dict(reversed(list(ordered(positive).items())))
    expect_error(
        "C44",
        lambda: exact_sign_flip_test(reversed_mapping, rq="RQ2"),
        "training seeds must match the exact locked ordered list",
    )
    short_mapping = {
        int(seed): float(i + 1)
        for i, seed in enumerate(OFFICIAL_TRAINING_SEEDS[:-1])
    }
    expect_error(
        "C45",
        lambda: exact_sign_flip_test(short_mapping, rq="RQ2"),
        "training seeds must match the exact locked ordered list",
    )
    bad_nan = ordered(positive)
    bad_nan[int(OFFICIAL_TRAINING_SEEDS[0])] = float("nan")
    expect_error("C46", lambda: exact_sign_flip_test(bad_nan, rq="RQ2"), "must be finite")
    bad_inf = ordered(positive)
    bad_inf[int(OFFICIAL_TRAINING_SEEDS[0])] = float("inf")
    expect_error("C47", lambda: exact_sign_flip_test(bad_inf, rq="RQ2"), "must be finite")
    bad_bool = ordered(positive)
    bad_bool[int(OFFICIAL_TRAINING_SEEDS[0])] = True
    expect_error("C48", lambda: exact_sign_flip_test(bad_bool, rq="RQ2"), "must be numeric, not bool")
    expect_error(
        "C49",
        lambda: exact_sign_flip_test([1.0] * 10, rq="RQ2"),
        "seed_effects must be a mapping",
    )

    check("C50", rq2["replication_unit"] == "TRAINING_SEED_TRAINED_RUN", "inferential replication unit")
    check("C51", rq2["exact_p_value"] == rq2["extreme_configuration_count"] / 1024, "RQ2 denominator exactly 1024")
    check("C52", rq6["exact_p_value"] == rq6["extreme_configuration_count"] / 1024, "RQ6 denominator exactly 1024")
    check("C53", rq7["exact_p_value"] == rq7["extreme_configuration_count"] / 1024, "RQ7 denominator exactly 1024")

    failed = [item for item in CHECKS if item["status"] != "PASS"]
    report = {
        "task": "S6.11",
        "tracker_action": "Implement exact sign-flip 2^10",
        "status": "PASS" if not failed else "FAIL",
        "checks_passed": len(CHECKS) - len(failed),
        "checks_total": len(CHECKS),
        "failed_count": len(failed),
        "failed_checks": failed,
        "scientific_contract": {
            "seed_effect_count": 10,
            "sign_configuration_count": 1024,
            "test_statistic": "arithmetic_mean_of_paired_seed_effects",
            "rq2_alternative": "one-sided-greater",
            "rq6_alternative": "two-sided",
            "rq7_alternative": "two-sided",
            "equality_ties_included": True,
            "plus_one_correction": False,
            "zero_effect_seed_retained": True,
            "duplicate_statistics_counted_by_configuration": True,
        },
        "positive_fixture": {
            "rq2": rq2,
            "rq6": rq6,
            "rq7": rq7,
        },
        "zero_effect_fixture": {
            "all_zero_rq2": zeros,
            "one_zero_rq2": one_zero,
        },
        "checks": CHECKS,
        "test_used": False,
        "hidden_unlabeled_gt_used": False,
        "official_training_authorized": False,
        "final_test_authorized": False,
    }

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"S6_11_EXACT_SIGN_FLIP_FIXTURE={len(CHECKS) - len(failed)}/{len(CHECKS)}_PASS")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

