"""Synthetic fixture validator for S6.12 LOSO robustness."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.statistics.loso_robustness import (
    LOSO_ANALYSIS_COUNT,
    LOSO_RETAINED_SEED_COUNT,
    LOSORobustnessError,
    leave_one_seed_out,
)
from src.statistics.seed_summary import OFFICIAL_TRAINING_SEEDS

CHECKS: list[dict[str, object]] = []


def check(check_id: str, condition: bool, detail: str) -> None:
    CHECKS.append({
        "check_id": check_id,
        "status": "PASS" if condition else "FAIL",
        "detail": detail,
    })


def expect_error(check_id: str, fn, detail: str) -> None:
    try:
        fn()
    except LOSORobustnessError:
        check(check_id, True, detail)
    else:
        check(check_id, False, detail)


def fixture(values: list[float]) -> dict[int, float]:
    return {
        int(seed): float(value)
        for seed, value in zip(OFFICIAL_TRAINING_SEEDS, values, strict=True)
    }


def main() -> int:
    rq2_input = fixture([1,2,3,4,5,6,7,8,9,10])
    rq6_input = fixture([1,2,3,4,5,6,7,8,9,10])
    rq7_input = fixture([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])

    rq2 = leave_one_seed_out(rq2_input, rq="RQ2")
    rq6 = leave_one_seed_out(rq6_input, rq="RQ6")
    rq7 = leave_one_seed_out(rq7_input, rq="RQ7")

    check("C01", LOSO_ANALYSIS_COUNT == 10, "exactly 10 LOSO analyses")
    check("C02", LOSO_RETAINED_SEED_COUNT == 9, "each LOSO analysis retains 9 seeds")
    check("C03", rq2["loso_analysis_count"] == 10, "RQ2 has exactly 10 analyses")
    check("C04", rq6["loso_analysis_count"] == 10, "RQ6 has exactly 10 analyses")
    check("C05", rq7["loso_analysis_count"] == 10, "RQ7 has exactly 10 analyses")
    check("C06", math.isclose(rq2["full_effect"], 5.5), "RQ2 full effect is 5.5")
    check("C07", math.isclose(rq6["full_effect"], 5.5), "RQ6 full effect is 5.5")
    check("C08", math.isclose(rq7["full_effect"], 0.55), "RQ7 full effect is 0.55")
    check("C09", rq2["primary_seed_exclusion_allowed"] is False, "RQ2 cannot exclude primary seed")
    check("C10", rq6["primary_seed_exclusion_allowed"] is False, "RQ6 cannot exclude primary seed")
    check("C11", rq7["primary_seed_exclusion_allowed"] is False, "RQ7 cannot exclude primary seed")

    for offset, result in enumerate((rq2, rq6, rq7), start=12):
        omitted = [row["omitted_seed"] for row in result["analyses"]]
        check(f"C{offset:02d}", tuple(omitted) == OFFICIAL_TRAINING_SEEDS, f"{result['rq']} omits each official seed exactly once in locked order")

    for index, row in enumerate(rq2["analyses"]):
        omitted_value = float(index + 1)
        expected = (55.0 - omitted_value) / 9.0
        check(f"C{15 + index:02d}", math.isclose(row["recomputed_effect"], expected), f"RQ2 omission {index + 1} recomputed mean")

    for index, row in enumerate(rq6["analyses"]):
        omitted_value = float(index + 1)
        expected = (55.0 - omitted_value) / 9.0
        check(f"C{25 + index:02d}", math.isclose(row["recomputed_effect"], expected), f"RQ6 omission {index + 1} recomputed mean")

    for index, row in enumerate(rq7["analyses"]):
        omitted_value = float(index + 1) / 10.0
        expected = (5.5 - omitted_value) / 9.0
        check(f"C{35 + index:02d}", math.isclose(row["recomputed_effect"], expected), f"RQ7 omission {index + 1} recomputed mean")

    check("C45", all(row["retained_seed_count"] == 9 for row in rq2["analyses"]), "RQ2 retains 9 seeds every time")
    check("C46", all(row["retained_seed_count"] == 9 for row in rq6["analyses"]), "RQ6 retains 9 seeds every time")
    check("C47", all(row["retained_seed_count"] == 9 for row in rq7["analyses"]), "RQ7 retains 9 seeds every time")
    check("C48", all(row["omitted_seed"] not in row["retained_training_seeds"] for row in rq2["analyses"]), "RQ2 omitted seed absent only from its LOSO replicate")
    check("C49", all(row["omitted_seed"] not in row["retained_training_seeds"] for row in rq6["analyses"]), "RQ6 omitted seed absent only from its LOSO replicate")
    check("C50", all(row["omitted_seed"] not in row["retained_training_seeds"] for row in rq7["analyses"]), "RQ7 omitted seed absent only from its LOSO replicate")

    expect_error("C51", lambda: leave_one_seed_out(rq2_input, rq="RQ3"), "invalid RQ rejected")
    expect_error("C52", lambda: leave_one_seed_out({k:v for k,v in list(rq2_input.items())[:-1]}, rq="RQ2"), "missing seed rejected")
    expect_error("C53", lambda: leave_one_seed_out(dict(reversed(list(rq2_input.items()))), rq="RQ2"), "wrong seed order rejected")
    bad_bool = dict(rq2_input); bad_bool[OFFICIAL_TRAINING_SEEDS[0]] = True
    expect_error("C54", lambda: leave_one_seed_out(bad_bool, rq="RQ2"), "boolean effect rejected")
    bad_nan = dict(rq2_input); bad_nan[OFFICIAL_TRAINING_SEEDS[0]] = float("nan")
    expect_error("C55", lambda: leave_one_seed_out(bad_nan, rq="RQ2"), "NaN effect rejected")
    bad_inf = dict(rq2_input); bad_inf[OFFICIAL_TRAINING_SEEDS[0]] = float("inf")
    expect_error("C56", lambda: leave_one_seed_out(bad_inf, rq="RQ2"), "infinite effect rejected")

    failed = [item for item in CHECKS if item["status"] != "PASS"]
    report = {
        "task": "S6.12",
        "tracker_action": "Implement exactly 10 LOSO",
        "status": "PASS" if not failed else "FAIL",
        "checks_passed": len(CHECKS) - len(failed),
        "checks_total": len(CHECKS),
        "failed_count": len(failed),
        "failed_checks": failed,
        "scientific_contract": {
            "seed_effect_count": 10,
            "loso_analysis_count": 10,
            "retained_seed_count_per_analysis": 9,
            "recomputed_statistic": "arithmetic_mean_of_remaining_seed_effects",
            "primary_seed_exclusion_allowed": False,
            "loso_p_value_required": False,
            "loso_confidence_interval_required": False,
        },
        "positive_fixture": {"rq2": rq2, "rq6": rq6, "rq7": rq7},
        "checks": CHECKS,
        "test_used": False,
        "hidden_unlabeled_gt_used": False,
        "official_training_authorized": False,
        "final_test_authorized": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"S6_12_LOSO_FIXTURE={len(CHECKS) - len(failed)}/{len(CHECKS)}_PASS")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
