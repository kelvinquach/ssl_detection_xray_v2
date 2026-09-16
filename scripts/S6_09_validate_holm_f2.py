"""Validate the locked S6.09 Holm F2 multiplicity fixture."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from src.statistics.holm_f2 import (
    F2_ALPHA,
    F2_FAMILY_ID,
    F2_M,
    F2_MEMBER_ORDER,
    F2_METHOD,
    HolmF2Error,
    holm_f2,
)

ROOT = Path(__file__).resolve().parents[1]
STATISTICAL_REPORT = ROOT / "artifacts" / "preflight" / "statistics" / "statistical_fixture_report.json"
OUTPUT_REPORT = ROOT / "artifacts" / "preflight" / "statistics" / "multiplicity_fixture_report.json"
CORE = ROOT / "src" / "statistics" / "holm_f2.py"
VALIDATOR = ROOT / "scripts" / "S6_09_validate_holm_f2.py"

EXPECTED_STATISTICAL_REPORT_SHA256 = (
    "c0874e61745f8d2cae5be707fbe0d82c802b780a63f8fb5510458614f985b3bc"
)
EXPECTED_CORE_SHA256 = (
    "d080f91729bb44838d64bba5b4d2c8a351dda76966e09d365b9e4dc9d4225707"
)
TOL = 1e-12

checks: list[dict[str, object]] = []


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check(name: str, passed: bool, observed: object) -> None:
    checks.append({"name": name, "pass": bool(passed), "observed": observed})
    if not passed:
        raise AssertionError(f"{name}: {observed}")


def expect_error(name: str, fn) -> None:
    try:
        fn()
    except (HolmF2Error, ValueError, TypeError) as exc:
        check(name, True, type(exc).__name__)
        return
    check(name, False, "NO_ERROR")


def main() -> None:
    if OUTPUT_REPORT.exists():
        raise AssertionError("multiplicity_fixture_report.json must not preexist")

    statistical_bytes = STATISTICAL_REPORT.read_bytes()
    statistical_sha = sha256_bytes(statistical_bytes)
    check(
        "statistical_fixture_report_sha256",
        statistical_sha == EXPECTED_STATISTICAL_REPORT_SHA256,
        statistical_sha,
    )
    check(
        "statistical_fixture_report_lf_only",
        b"\r" not in statistical_bytes,
        b"\r" not in statistical_bytes,
    )

    core_bytes = CORE.read_bytes()
    validator_bytes = VALIDATOR.read_bytes()
    core_sha = sha256_bytes(core_bytes)
    validator_sha = sha256_bytes(validator_bytes)
    check("holm_f2_core_sha256", core_sha == EXPECTED_CORE_SHA256, core_sha)
    check("holm_f2_core_lf_only", b"\r" not in core_bytes, b"\r" not in core_bytes)
    check("validator_lf_only", b"\r" not in validator_bytes, b"\r" not in validator_bytes)

    report = json.loads(statistical_bytes.decode("utf-8"))
    section_keys = {
        "RQ3": "s6_03_rq3_gg_repeated_measures_anova",
        "RQ4": "s6_05_rq4_mixed_model",
        "RQ6": "s6_07_rq6_negative_fp_primary_t_test",
        "RQ7": "s6_08_rq7_architecture_effect_primary_t_test",
    }
    for member, key in section_keys.items():
        section = report.get(key)
        check(f"{member.lower()}_section_present", isinstance(section, dict), key)
        check(f"{member.lower()}_section_status", section.get("status") == "PASS", section.get("status"))

    rq3 = report[section_keys["RQ3"]]
    rq4 = report[section_keys["RQ4"]]
    rq6 = report[section_keys["RQ6"]]
    rq7 = report[section_keys["RQ7"]]

    check(
        "rq3_gg_always",
        rq3["scientific_contract"]["greenhouse_geisser_policy"] == "ALWAYS",
        rq3["scientific_contract"]["greenhouse_geisser_policy"],
    )
    rq3_p = float(rq3["positive_fixture"]["p_gg"])
    check("rq3_p_gg_finite", math.isfinite(rq3_p), rq3_p)

    check(
        "rq4_kenward_roger",
        rq4["scientific_contract"]["fixed_effect_inference"] == "Kenward-Roger",
        rq4["scientific_contract"]["fixed_effect_inference"],
    )
    check(
        "rq4_one_sided_greater",
        rq4["scientific_contract"]["beta_Q_alternative"] == "greater",
        rq4["scientific_contract"]["beta_Q_alternative"],
    )
    rq4_p = float(rq4["positive_fixture"]["golden_result"]["one_sided_p_value"])
    check("rq4_one_sided_p_finite", math.isfinite(rq4_p), rq4_p)

    check("rq6_two_sided", rq6["alternative"] == "two-sided", rq6["alternative"])
    rq6_p = float(rq6["positive_fixture"]["p_value"])
    check("rq6_two_sided_p_finite", math.isfinite(rq6_p), rq6_p)

    check(
        "rq7_two_sided",
        rq7["scientific_contract"]["alternative"] == "two-sided",
        rq7["scientific_contract"]["alternative"],
    )
    rq7_p = float(rq7["positive_fixture"]["primary_result"]["p_value"])
    check("rq7_two_sided_p_finite", math.isfinite(rq7_p), rq7_p)

    check("f2_family_id", F2_FAMILY_ID == "F2", F2_FAMILY_ID)
    check("f2_member_order", F2_MEMBER_ORDER == ("RQ3", "RQ4", "RQ6", "RQ7"), F2_MEMBER_ORDER)
    check("f2_m", F2_M == 4, F2_M)
    check("f2_method", F2_METHOD == "HOLM_STEP_DOWN", F2_METHOD)
    check("f2_fwer", math.isclose(F2_ALPHA, 0.05), F2_ALPHA)

    raw_p_values = {
        "RQ3": rq3_p,
        "RQ4": rq4_p,
        "RQ6": rq6_p,
        "RQ7": rq7_p,
    }
    result = holm_f2(raw_p_values)

    check("result_family", result["family"] == "F2", result["family"])
    check("result_method", result["method"] == "HOLM_STEP_DOWN", result["method"])
    check("result_family_size", result["family_size"] == 4, result["family_size"])
    check("result_fwer", math.isclose(float(result["fwer"]), 0.05), result["fwer"])
    check(
        "result_membership_locked",
        result["family_membership_locked"] is True,
        result["family_membership_locked"],
    )
    check(
        "result_ci_not_adjusted",
        result["individual_ci_adjusted"] is False,
        result["individual_ci_adjusted"],
    )

    ordered_members = [row["member"] for row in result["ordered_results"]]
    check(
        "ordered_members",
        ordered_members == ["RQ4", "RQ3", "RQ6", "RQ7"],
        ordered_members,
    )

    expected_adjusted = {
        "RQ4": 4.0 * rq4_p,
        "RQ3": 3.0 * rq3_p,
        "RQ6": 2.0 * rq6_p,
        "RQ7": 2.0 * rq6_p,
    }
    for member in F2_MEMBER_ORDER:
        observed = float(result["by_member"][member]["adjusted_p_value"])
        expected = float(expected_adjusted[member])
        check(
            f"golden_adjusted_{member.lower()}",
            math.isclose(observed, expected, rel_tol=TOL, abs_tol=TOL),
            observed,
        )
        check(
            f"golden_reject_{member.lower()}",
            result["by_member"][member]["reject_at_fwer_0_05"] is True,
            result["by_member"][member]["reject_at_fwer_0_05"],
        )

    adjusted_in_rank_order = [
        float(row["adjusted_p_value"]) for row in result["ordered_results"]
    ]
    check(
        "adjusted_p_monotone",
        all(
            adjusted_in_rank_order[i] <= adjusted_in_rank_order[i + 1]
            for i in range(len(adjusted_in_rank_order) - 1)
        ),
        adjusted_in_rank_order,
    )

    expect_error(
        "negative_missing_member",
        lambda: holm_f2({"RQ3": 0.01, "RQ4": 0.02, "RQ6": 0.03}),
    )
    expect_error(
        "negative_extra_member",
        lambda: holm_f2(
            {"RQ3": 0.01, "RQ4": 0.02, "RQ6": 0.03, "RQ7": 0.04, "RQ2": 0.05}
        ),
    )
    expect_error(
        "negative_wrong_member",
        lambda: holm_f2({"RQ3": 0.01, "RQ4": 0.02, "RQ6": 0.03, "RQ5": 0.04}),
    )
    bad = copy.deepcopy(raw_p_values)
    bad["RQ3"] = float("nan")
    expect_error("negative_nan_p", lambda: holm_f2(bad))
    bad = copy.deepcopy(raw_p_values)
    bad["RQ4"] = -0.01
    expect_error("negative_below_zero_p", lambda: holm_f2(bad))
    bad = copy.deepcopy(raw_p_values)
    bad["RQ6"] = 1.01
    expect_error("negative_above_one_p", lambda: holm_f2(bad))
    bad = copy.deepcopy(raw_p_values)
    bad["RQ7"] = True
    expect_error("negative_boolean_p", lambda: holm_f2(bad))

    boundary = {
        "fixture_only_no_official_results": True,
        "family_shrunk_after_results": False,
        "holm_f3_applied_in_s6_09": False,
        "individual_ci_holm_adjusted": False,
        "exact_sign_flip_applied_in_s6_09": False,
        "loso_applied_in_s6_09": False,
        "test_used": False,
        "test_gt_used": False,
        "hidden_unlabeled_gt_used": False,
        "official_training_authorized": False,
        "final_test_authorized": False,
    }

    fixture = {
        "task": "S6.09",
        "category": "Multiplicity",
        "tracker_action": "Implement Holm F2 m=4",
        "status": "PASS",
        "family": "F2",
        "scientific_contract": {
            "members": list(F2_MEMBER_ORDER),
            "member_input_semantics": {
                "RQ3": "GG_CORRECTED_OMNIBUS_P",
                "RQ4": "ONE_SIDED_KENWARD_ROGER_BETA_Q_P",
                "RQ6": "TWO_SIDED_PRIMARY_P",
                "RQ7": "TWO_SIDED_PRIMARY_P",
            },
            "m": 4,
            "method": "HOLM_STEP_DOWN",
            "fwer": 0.05,
            "family_membership_locked": True,
            "individual_ci_holm_adjusted": False,
        },
        "input_source": {
            "path": "artifacts/preflight/statistics/statistical_fixture_report.json",
            "sha256": statistical_sha,
            "synthetic_fixture_only": True,
        },
        "raw_p_values": raw_p_values,
        "holm_result": result,
        "scientific_boundaries": boundary,
        "source_sha256": {
            "src/statistics/holm_f2.py": core_sha,
            "scripts/S6_09_validate_holm_f2.py": validator_sha,
        },
    }

    fixture["checks_passed"] = sum(1 for item in checks if item["pass"])
    fixture["checks_total"] = len(checks)
    fixture["failed_count"] = sum(1 for item in checks if not item["pass"])
    fixture["failed_checks"] = [item["name"] for item in checks if not item["pass"]]
    fixture["checks"] = checks

    if fixture["failed_count"] != 0:
        raise AssertionError(fixture["failed_checks"])

    output = {
        "schema_version": "1.0",
        "status": "PASS",
        "s6_09_holm_f2": fixture,
    }
    payload = json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.write_bytes(payload.encode("utf-8"))

    output_sha = sha256_bytes(OUTPUT_REPORT.read_bytes())
    print(f"S6_09_HOLM_F2_FIXTURE={fixture['checks_passed']}/{fixture['checks_total']}_PASS")
    print("FAMILY=F2")
    print("MEMBERS=RQ3,RQ4,RQ6,RQ7")
    print("M=4")
    print("METHOD=HOLM_STEP_DOWN")
    print("FWER=0.05")
    print("ORDER=" + ",".join(ordered_members))
    print("RQ3_RAW_P=" + repr(rq3_p))
    print("RQ4_RAW_P=" + repr(rq4_p))
    print("RQ6_RAW_P=" + repr(rq6_p))
    print("RQ7_RAW_P=" + repr(rq7_p))
    print("HOLM_F2_CORE_SHA256=" + core_sha)
    print("HOLM_F2_VALIDATOR_SHA256=" + validator_sha)
    print("STATISTICAL_FIXTURE_REPORT_SHA256=" + statistical_sha)
    print("MULTIPLICITY_FIXTURE_REPORT_SHA256=" + output_sha)
    print("TEST_USED=False")
    print("HIDDEN_UNLABELED_GT_USED=False")
    print("OFFICIAL_TRAINING_AUTHORIZED=False")
    print("FINAL_TEST_AUTHORIZED=False")


if __name__ == "__main__":
    main()
