"""Validate and extend the locked S6.10 Holm F3 multiplicity fixture."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from src.statistics.holm_f3 import (
    F3_ALPHA,
    F3_FAMILY_ID,
    F3_M,
    F3_MEMBER_ORDER,
    F3_METHOD,
    HolmF3Error,
    holm_f3,
)

ROOT = Path(__file__).resolve().parents[1]
STATISTICAL_REPORT = ROOT / "artifacts" / "preflight" / "statistics" / "statistical_fixture_report.json"
OUTPUT_REPORT = ROOT / "artifacts" / "preflight" / "statistics" / "multiplicity_fixture_report.json"
CORE = ROOT / "src" / "statistics" / "holm_f3.py"
VALIDATOR = ROOT / "scripts" / "S6_10_validate_holm_f3.py"

EXPECTED_STATISTICAL_REPORT_SHA256 = (
    "c0874e61745f8d2cae5be707fbe0d82c802b780a63f8fb5510458614f985b3bc"
)
EXPECTED_INPUT_MULTIPLICITY_REPORT_SHA256 = (
    "b40686709ea6b6dbe0aff5cf89b2b6eede459e1164754052ede62008e3a498d0"
)
EXPECTED_CORE_SHA256 = (
    "5a3ebb59f83927c8df55ecc9f2dcf512d333914cdeed3bd6c321d1f5f3883c80"
)
TOL = 1e-12

checks: list[dict[str, object]] = []


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(payload)


def check(name: str, passed: bool, observed: object) -> None:
    checks.append({"name": name, "pass": bool(passed), "observed": observed})
    if not passed:
        raise AssertionError(f"{name}: {observed}")


def expect_error(name: str, fn) -> None:
    try:
        fn()
    except (HolmF3Error, ValueError, TypeError) as exc:
        check(name, True, type(exc).__name__)
        return
    check(name, False, "NO_ERROR")


def main() -> None:
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

    input_multiplicity_bytes = OUTPUT_REPORT.read_bytes()
    input_multiplicity_sha = sha256_bytes(input_multiplicity_bytes)
    check(
        "input_multiplicity_fixture_report_sha256",
        input_multiplicity_sha == EXPECTED_INPUT_MULTIPLICITY_REPORT_SHA256,
        input_multiplicity_sha,
    )
    check(
        "input_multiplicity_fixture_report_lf_only",
        b"\r" not in input_multiplicity_bytes,
        b"\r" not in input_multiplicity_bytes,
    )

    core_bytes = CORE.read_bytes()
    validator_bytes = VALIDATOR.read_bytes()
    core_sha = sha256_bytes(core_bytes)
    validator_sha = sha256_bytes(validator_bytes)
    check("holm_f3_core_sha256", core_sha == EXPECTED_CORE_SHA256, core_sha)
    check("holm_f3_core_lf_only", b"\r" not in core_bytes, b"\r" not in core_bytes)
    check("validator_lf_only", b"\r" not in validator_bytes, b"\r" not in validator_bytes)

    statistical_report = json.loads(statistical_bytes.decode("utf-8"))
    input_multiplicity = json.loads(input_multiplicity_bytes.decode("utf-8"))

    check("input_schema_version", input_multiplicity.get("schema_version") == "1.0", input_multiplicity.get("schema_version"))
    check("input_top_status", input_multiplicity.get("status") == "PASS", input_multiplicity.get("status"))
    check(
        "input_top_level_keys",
        set(input_multiplicity.keys()) == {"schema_version", "status", "s6_09_holm_f2"},
        sorted(input_multiplicity.keys()),
    )
    historical_f2 = input_multiplicity.get("s6_09_holm_f2")
    check("historical_s6_09_present", isinstance(historical_f2, dict), type(historical_f2).__name__)
    check("historical_s6_09_status", historical_f2.get("status") == "PASS", historical_f2.get("status"))
    check("historical_s6_09_checks", historical_f2.get("checks_passed") == 50 and historical_f2.get("checks_total") == 50, [historical_f2.get("checks_passed"), historical_f2.get("checks_total")])
    check("historical_s6_09_failed_count", historical_f2.get("failed_count") == 0, historical_f2.get("failed_count"))
    historical_f2_sha = canonical_sha256(historical_f2)

    section = statistical_report.get("s6_04_rq3_pairwise_contrasts")
    check("s6_04_section_present", isinstance(section, dict), "s6_04_rq3_pairwise_contrasts")
    check("s6_04_section_status", section.get("status") == "PASS", section.get("status"))
    check("s6_04_checks", section.get("checks_passed") == 99 and section.get("checks_total") == 99, [section.get("checks_passed"), section.get("checks_total")])
    check("s6_04_failed_count", section.get("failed_count") == 0, section.get("failed_count"))

    contract = section["scientific_contract"]
    check("s6_04_contrast_count", contract.get("contrast_count") == 6, contract.get("contrast_count"))
    check("s6_04_contrasts", tuple(contract.get("contrasts", [])) == F3_MEMBER_ORDER, contract.get("contrasts"))
    check("s6_04_raw_p_two_sided", contract.get("raw_p_value") == "TWO_SIDED", contract.get("raw_p_value"))
    check("s6_04_holm_not_applied", contract.get("holm_applied") is False, contract.get("holm_applied"))
    check("s6_04_holm_deferred_s6_10", contract.get("holm_deferred_task") == "S6.10", contract.get("holm_deferred_task"))

    golden_rows = section["positive_fixture"]["golden_results"]
    check("golden_row_count", len(golden_rows) == 6, len(golden_rows))
    by_contrast = {row["contrast"]: row for row in golden_rows}
    check("golden_members_exact", set(by_contrast) == set(F3_MEMBER_ORDER), sorted(by_contrast))

    raw_p_values = {
        member: float(by_contrast[member]["raw_two_sided_p_value"])
        for member in F3_MEMBER_ORDER
    }
    for member, p_value in raw_p_values.items():
        check(
            f"raw_p_finite_{member}",
            math.isfinite(p_value) and 0.0 <= p_value <= 1.0,
            p_value,
        )

    check("f3_family_id", F3_FAMILY_ID == "F3", F3_FAMILY_ID)
    check("f3_member_order", F3_MEMBER_ORDER == ("5%-1%", "10%-1%", "20%-1%", "10%-5%", "20%-5%", "20%-10%"), F3_MEMBER_ORDER)
    check("f3_m", F3_M == 6, F3_M)
    check("f3_method", F3_METHOD == "HOLM_STEP_DOWN", F3_METHOD)
    check("f3_fwer", math.isclose(F3_ALPHA, 0.05), F3_ALPHA)

    result = holm_f3(raw_p_values)

    check("result_family", result["family"] == "F3", result["family"])
    check("result_method", result["method"] == "HOLM_STEP_DOWN", result["method"])
    check("result_family_size", result["family_size"] == 6, result["family_size"])
    check("result_fwer", math.isclose(float(result["fwer"]), 0.05), result["fwer"])
    check("result_membership_locked", result["family_membership_locked"] is True, result["family_membership_locked"])
    check("result_ci_not_adjusted", result["individual_ci_adjusted"] is False, result["individual_ci_adjusted"])

    ordered_members = [row["member"] for row in result["ordered_results"]]
    expected_order = ["20%-1%", "20%-5%", "5%-1%", "10%-1%", "20%-10%", "10%-5%"]
    check("ordered_members", ordered_members == expected_order, ordered_members)

    expected_adjusted = {
        "20%-1%": 2.5334220649443847e-11,
        "20%-5%": 1.4097118740743502e-09,
        "5%-1%": 1.1339892241091992e-07,
        "10%-1%": 1.9689415388994918e-07,
        "20%-10%": 8.901738559491669e-05,
        "10%-5%": 8.901738559491669e-05,
    }
    for member in F3_MEMBER_ORDER:
        observed = float(result["by_member"][member]["adjusted_p_value"])
        expected = expected_adjusted[member]
        check(
            f"golden_adjusted_{member}",
            math.isclose(observed, expected, rel_tol=TOL, abs_tol=TOL),
            observed,
        )
        check(
            f"golden_reject_{member}",
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
        lambda: holm_f3({member: 0.01 for member in F3_MEMBER_ORDER[:-1]}),
    )
    extra = {member: 0.01 for member in F3_MEMBER_ORDER}
    extra["EXTRA"] = 0.01
    expect_error("negative_extra_member", lambda: holm_f3(extra))
    bad = {member: 0.01 for member in F3_MEMBER_ORDER}
    bad["5%-1%"] = float("nan")
    expect_error("negative_nan_p", lambda: holm_f3(bad))
    bad = {member: 0.01 for member in F3_MEMBER_ORDER}
    bad["10%-1%"] = -0.01
    expect_error("negative_below_zero_p", lambda: holm_f3(bad))
    bad = {member: 0.01 for member in F3_MEMBER_ORDER}
    bad["20%-1%"] = 1.01
    expect_error("negative_above_one_p", lambda: holm_f3(bad))
    bad = {member: 0.01 for member in F3_MEMBER_ORDER}
    bad["10%-5%"] = True
    expect_error("negative_boolean_p", lambda: holm_f3(bad))

    tie_result = holm_f3({member: 0.01 for member in F3_MEMBER_ORDER})
    check(
        "deterministic_tie_order",
        [row["member"] for row in tie_result["ordered_results"]] == list(F3_MEMBER_ORDER),
        [row["member"] for row in tie_result["ordered_results"]],
    )

    boundary = {
        "fixture_only_no_official_results": True,
        "family_shrunk_after_results": False,
        "holm_f2_recomputed_in_s6_10": False,
        "individual_ci_holm_adjusted": False,
        "exact_sign_flip_applied_in_s6_10": False,
        "loso_applied_in_s6_10": False,
        "test_used": False,
        "test_gt_used": False,
        "hidden_unlabeled_gt_used": False,
        "official_training_authorized": False,
        "final_test_authorized": False,
    }

    fixture = {
        "task": "S6.10",
        "category": "Multiplicity",
        "tracker_action": "Implement Holm F3 m=6",
        "status": "PASS",
        "family": "F3",
        "scientific_contract": {
            "members": list(F3_MEMBER_ORDER),
            "member_input_semantics": "S6.04_RAW_TWO_SIDED_RQ3_PAIRWISE_P",
            "m": 6,
            "method": "HOLM_STEP_DOWN",
            "fwer": 0.05,
            "family_membership_locked": True,
            "individual_ci_holm_adjusted": False,
        },
        "input_source": {
            "path": "artifacts/preflight/statistics/statistical_fixture_report.json",
            "section": "s6_04_rq3_pairwise_contrasts",
            "field": "positive_fixture.golden_results[*].raw_two_sided_p_value",
            "sha256": statistical_sha,
            "synthetic_fixture_only": True,
        },
        "raw_p_values": raw_p_values,
        "holm_result": result,
        "scientific_boundaries": boundary,
        "historical_s6_09_section_canonical_sha256": historical_f2_sha,
        "source_sha256": {
            "src/statistics/holm_f3.py": core_sha,
            "scripts/S6_10_validate_holm_f3.py": validator_sha,
        },
    }

    fixture["checks_passed"] = sum(1 for item in checks if item["pass"])
    fixture["checks_total"] = len(checks)
    fixture["failed_count"] = sum(1 for item in checks if not item["pass"])
    fixture["failed_checks"] = [item["name"] for item in checks if not item["pass"]]
    fixture["checks"] = checks

    if fixture["failed_count"] != 0:
        raise AssertionError(fixture["failed_checks"])

    output = copy.deepcopy(input_multiplicity)
    output["schema_version"] = "1.0"
    output["status"] = "PASS"
    output["s6_10_holm_f3"] = fixture

    check(
        "historical_s6_09_preserved_before_write",
        output["s6_09_holm_f2"] == historical_f2,
        canonical_sha256(output["s6_09_holm_f2"]),
    )

    payload = json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    OUTPUT_REPORT.write_bytes(payload.encode("utf-8"))

    written = json.loads(OUTPUT_REPORT.read_text(encoding="utf-8"))
    check(
        "historical_s6_09_preserved_after_write",
        written["s6_09_holm_f2"] == historical_f2,
        canonical_sha256(written["s6_09_holm_f2"]),
    )
    check(
        "historical_s6_09_canonical_sha_unchanged",
        canonical_sha256(written["s6_09_holm_f2"]) == historical_f2_sha,
        canonical_sha256(written["s6_09_holm_f2"]),
    )
    check("written_s6_10_present", "s6_10_holm_f3" in written, sorted(written.keys()))
    check("written_top_status", written.get("status") == "PASS", written.get("status"))

    output_sha = sha256_bytes(OUTPUT_REPORT.read_bytes())
    print(f"S6_10_HOLM_F3_FIXTURE={fixture['checks_passed']}/{fixture['checks_total']}_PASS")
    print("FAMILY=F3")
    print("MEMBERS=" + ",".join(F3_MEMBER_ORDER))
    print("M=6")
    print("METHOD=HOLM_STEP_DOWN")
    print("FWER=0.05")
    print("ORDER=" + ",".join(ordered_members))
    for member in F3_MEMBER_ORDER:
        print(member + "_RAW_P=" + repr(raw_p_values[member]))
        print(member + "_ADJUSTED_P=" + repr(result["by_member"][member]["adjusted_p_value"]))
        print(member + "_REJECT=" + str(result["by_member"][member]["reject_at_fwer_0_05"]))
    print("HISTORICAL_S6_09_SECTION_CANONICAL_SHA256=" + historical_f2_sha)
    print("HOLM_F3_CORE_SHA256=" + core_sha)
    print("HOLM_F3_VALIDATOR_SHA256=" + validator_sha)
    print("STATISTICAL_FIXTURE_REPORT_SHA256=" + statistical_sha)
    print("INPUT_MULTIPLICITY_FIXTURE_REPORT_SHA256=" + input_multiplicity_sha)
    print("MULTIPLICITY_FIXTURE_REPORT_SHA256=" + output_sha)
    print("TEST_USED=False")
    print("HIDDEN_UNLABELED_GT_USED=False")
    print("OFFICIAL_TRAINING_AUTHORIZED=False")
    print("FINAL_TEST_AUTHORIZED=False")


if __name__ == "__main__":
    main()
