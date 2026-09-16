#!/usr/bin/env python3
"""Validate the locked S6.07 RQ6 primary statistical fixture."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from src.statistics.rq6_negative_fp import (
    RQ6_ALTERNATIVE,
    RQ6_CELL_COUNT,
    RQ6_CI_LEVEL,
    RQ6_DELTA_COLUMN,
    RQ6_DF,
    RQ6_FAR_FORMAL_P_VALUE_REQUIRED,
    RQ6_FAR_SECONDARY_OUTCOME,
    RQ6NegativeFPError,
    compute_rq6_seed_effects,
    rq6_primary_test,
)
from src.statistics.seed_summary import (
    OFFICIAL_SEED_COUNT,
    OFFICIAL_TRAINING_SEEDS,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts" / "preflight" / "statistics" / "statistical_fixture_report.json"
CORE = ROOT / "src" / "statistics" / "rq6_negative_fp.py"

EXPECTED_HISTORICAL_REPORT_SHA256 = (
    "af3c3e6fee343a900a04de23ef6ea8272d98da109730e7e41e01795ad1611bad"
)
EXPECTED_CORE_SHA256 = (
    "69579793b2927c7a3ed39f7f94d3a2a598b35a0eacbe2b591cce48f69c2d0907"
)
SECTION_KEY = "s6_07_rq6_negative_fp_primary_t_test"
TOL = 1e-9

checks: list[dict[str, object]] = []


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check(name: str, condition: bool, observed: object = None) -> None:
    checks.append(
        {
            "name": name,
            "pass": bool(condition),
            "observed": observed,
        }
    )
    if not condition:
        raise AssertionError(f"{name} failed: {observed!r}")


def expect_error(name: str, rows: list[dict[str, object]]) -> None:
    try:
        compute_rq6_seed_effects(rows)
    except RQ6NegativeFPError:
        check(name, True, "RQ6NegativeFPError")
        return
    check(name, False, "NO_ERROR")


def make_positive_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    architectures = ("R50", "Swin-T")
    budgets = ("1%", "5%", "10%", "20%")

    for seed_index, training_seed in enumerate(
        OFFICIAL_TRAINING_SEEDS,
        start=1,
    ):
        for architecture in architectures:
            for budget in budgets:
                rows.append(
                    {
                        "architecture": architecture,
                        "budget": budget,
                        "seed_index": seed_index,
                        "training_seed": int(training_seed),
                        RQ6_DELTA_COLUMN: float(seed_index),
                    }
                )
    return rows


def main() -> None:
    historical_bytes = REPORT.read_bytes()
    historical_sha = sha256_bytes(historical_bytes)

    check(
        "historical_report_sha256",
        historical_sha == EXPECTED_HISTORICAL_REPORT_SHA256,
        historical_sha,
    )

    historical_text = historical_bytes.decode("utf-8")
    check(
        "historical_report_lf_closing",
        historical_bytes.endswith(b"}\n"),
        historical_bytes[-8:].hex(),
    )

    report = json.loads(historical_text)

    check("s6_01_status", report.get("status") == "PASS", report.get("status"))
    check(
        "s6_02_status",
        report.get("s6_02_rq2_primary_t_test", {}).get("status") == "PASS",
        report.get("s6_02_rq2_primary_t_test", {}).get("status"),
    )
    check(
        "s6_03_status",
        report.get("s6_03_rq3_gg_repeated_measures_anova", {}).get("status") == "PASS",
        report.get("s6_03_rq3_gg_repeated_measures_anova", {}).get("status"),
    )
    check(
        "s6_04_status",
        report.get("s6_04_rq3_pairwise_contrasts", {}).get("status") == "PASS",
        report.get("s6_04_rq3_pairwise_contrasts", {}).get("status"),
    )
    check(
        "s6_05_status",
        report.get("s6_05_rq4_mixed_model", {}).get("status") == "PASS",
        report.get("s6_05_rq4_mixed_model", {}).get("status"),
    )
    check(
        "s6_06_status",
        report.get("s6_06_rq5_rare_class_summary", {}).get("status") == "PASS",
        report.get("s6_06_rq5_rare_class_summary", {}).get("status"),
    )
    check(
        "s6_07_not_preexisting",
        SECTION_KEY not in report,
        SECTION_KEY in report,
    )

    core_sha = sha256_bytes(CORE.read_bytes())
    check(
        "rq6_core_sha256",
        core_sha == EXPECTED_CORE_SHA256,
        core_sha,
    )

    check("official_seed_count", OFFICIAL_SEED_COUNT == 10, OFFICIAL_SEED_COUNT)
    check("rq6_cell_count", RQ6_CELL_COUNT == 8, RQ6_CELL_COUNT)
    check("rq6_alternative", RQ6_ALTERNATIVE == "two-sided", RQ6_ALTERNATIVE)
    check("rq6_df", RQ6_DF == 9, RQ6_DF)
    check("rq6_ci_level", RQ6_CI_LEVEL == 0.95, RQ6_CI_LEVEL)
    check(
        "far_secondary_outcome",
        RQ6_FAR_SECONDARY_OUTCOME is True,
        RQ6_FAR_SECONDARY_OUTCOME,
    )
    check(
        "far_no_formal_p_value",
        RQ6_FAR_FORMAL_P_VALUE_REQUIRED is False,
        RQ6_FAR_FORMAL_P_VALUE_REQUIRED,
    )

    rows = make_positive_rows()
    check("positive_fixture_row_count", len(rows) == 80, len(rows))

    seed_effects = compute_rq6_seed_effects(rows)
    expected_seed_effects = {
        int(seed): float(index)
        for index, seed in enumerate(OFFICIAL_TRAINING_SEEDS, start=1)
    }
    check(
        "positive_seed_effects",
        seed_effects == expected_seed_effects,
        seed_effects,
    )

    result = rq6_primary_test(seed_effects)

    check("primary_effect", result["effect"] == "H_s", result["effect"])
    check(
        "primitive_effect_direction",
        result["primitive_effect"] == "FP_SSL_NEG_MINUS_FP_SUP_NEG",
        result["primitive_effect"],
    )
    check(
        "primary_metric",
        result["primary_metric"] == "FP_PER_NEGATIVE",
        result["primary_metric"],
    )
    check(
        "effect_estimate",
        abs(float(result["effect_estimate"]) - 5.5) <= TOL,
        result["effect_estimate"],
    )
    check(
        "sample_sd",
        abs(float(result["sample_sd"]) - 3.0276503540974917) <= TOL,
        result["sample_sd"],
    )
    check("sample_sd_ddof", result["sample_sd_ddof"] == 1, result["sample_sd_ddof"])
    check("n", result["n"] == 10, result["n"])
    check("df", result["df"] == 9, result["df"])
    check("null_mean", result["null_mean"] == 0.0, result["null_mean"])
    check("alternative", result["alternative"] == "two-sided", result["alternative"])
    check(
        "t_statistic",
        abs(float(result["t_statistic"]) - 5.744562646538029) <= TOL,
        result["t_statistic"],
    )
    check(
        "two_sided_p_value",
        abs(float(result["p_value"]) - 0.0002781960110481857) <= TOL,
        result["p_value"],
    )
    check(
        "p_value_sidedness",
        result["p_value_sidedness"] == "two-sided",
        result["p_value_sidedness"],
    )
    check("ci_level", result["ci_level"] == 0.95, result["ci_level"])
    check(
        "ci_sidedness",
        result["ci_sidedness"] == "two-sided",
        result["ci_sidedness"],
    )
    check(
        "ci_low",
        abs(float(result["ci_low"]) - 3.3341494102783162) <= TOL,
        result["ci_low"],
    )
    check(
        "ci_high",
        abs(float(result["ci_high"]) - 7.665850589721684) <= TOL,
        result["ci_high"],
    )
    check(
        "negative_interpretation",
        result["negative_effect_interpretation"] == "SSL_LESS_FP_PER_NEGATIVE",
        result["negative_effect_interpretation"],
    )
    check(
        "positive_interpretation",
        result["positive_effect_interpretation"] == "SSL_MORE_FP_PER_NEGATIVE",
        result["positive_effect_interpretation"],
    )
    check(
        "result_far_secondary",
        result["far_secondary_outcome"] is True,
        result["far_secondary_outcome"],
    )
    check(
        "result_far_no_formal_p",
        result["far_formal_p_value_required"] is False,
        result["far_formal_p_value_required"],
    )
    check(
        "no_far_p_value_field",
        "far_p_value" not in result,
        sorted(result),
    )

    expect_error("reject_wrong_row_count", rows[:-1])

    duplicate_rows = [dict(row) for row in rows]
    duplicate_rows[-1] = dict(duplicate_rows[0])
    expect_error("reject_duplicate_cell", duplicate_rows)

    bad_arch = [dict(row) for row in rows]
    bad_arch[0]["architecture"] = "BAD_ARCH"
    expect_error("reject_bad_architecture", bad_arch)

    bad_budget = [dict(row) for row in rows]
    bad_budget[0]["budget"] = "50%"
    expect_error("reject_bad_budget", bad_budget)

    bad_seed_mapping = [dict(row) for row in rows]
    bad_seed_mapping[0]["seed_index"] = 10
    expect_error("reject_bad_seed_mapping", bad_seed_mapping)

    nan_rows = [dict(row) for row in rows]
    nan_rows[0][RQ6_DELTA_COLUMN] = float("nan")
    expect_error("reject_nan_delta", nan_rows)

    inf_rows = [dict(row) for row in rows]
    inf_rows[0][RQ6_DELTA_COLUMN] = float("inf")
    expect_error("reject_infinite_delta", inf_rows)

    out_of_range_rows = [dict(row) for row in rows]
    out_of_range_rows[0][RQ6_DELTA_COLUMN] = 100.001
    expect_error("reject_out_of_range_delta", out_of_range_rows)

    check("test_used", True, False)
    check("hidden_unlabeled_gt_used", True, False)
    check("official_training_authorized", True, False)
    check("final_test_authorized", True, False)
    check("fixture_is_synthetic", True, "SYNTHETIC_ONLY")

    validator_sha = sha256_bytes(Path(__file__).read_bytes())

    section = {
        "task": "S6.07",
        "status": "PASS",
        "rq": "RQ6",
        "fixture": "SYNTHETIC_ONLY",
        "primary_effect": "H_s",
        "primitive_effect": "FP_SSL_NEG_MINUS_FP_SUP_NEG",
        "primary_metric": "FP_PER_NEGATIVE",
        "cell_count_per_seed": 8,
        "training_seed_count": 10,
        "sample_sd_ddof": 1,
        "primary_test": "TWO_SIDED_ONE_SAMPLE_T_TEST",
        "alternative": "two-sided",
        "df": 9,
        "ci": "TWO_SIDED_95_PERCENT",
        "far_secondary_outcome": True,
        "far_formal_p_value_required": False,
        "negative_effect_interpretation": "SSL_LESS_FP_PER_NEGATIVE",
        "positive_effect_interpretation": "SSL_MORE_FP_PER_NEGATIVE",
        "positive_fixture": {
            "seed_effects": seed_effects,
            "effect_estimate": result["effect_estimate"],
            "sample_sd": result["sample_sd"],
            "t_statistic": result["t_statistic"],
            "p_value": result["p_value"],
            "ci_low": result["ci_low"],
            "ci_high": result["ci_high"],
        },
        "checks_passed": sum(1 for item in checks if item["pass"]),
        "checks_total": len(checks),
        "checks": checks,
        "core_sha256": core_sha,
        "validator_sha256": validator_sha,
        "historical_report_sha256": historical_sha,
        "test_used": False,
        "hidden_unlabeled_gt_used": False,
        "official_training_authorized": False,
        "final_test_authorized": False,
    }

    check(
        "all_checks_pass",
        all(bool(item["pass"]) for item in checks),
        f"{sum(1 for item in checks if item['pass'])}/{len(checks)}",
    )

    section["checks_passed"] = sum(1 for item in checks if item["pass"])
    section["checks_total"] = len(checks)
    section["checks"] = checks

    payload = json.dumps(
        section,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    payload = payload.replace("\n", "\n  ")

    appended = (
        historical_bytes[:-2]
        + (',\n  "' + SECTION_KEY + '": ' + payload + "\n}\n").encode("utf-8")
    )

    REPORT.write_bytes(appended)

    written = REPORT.read_bytes()
    marker = (',\n  "' + SECTION_KEY + '": ').encode("utf-8")
    prefix, separator, _ = written.partition(marker)
    check("section_marker_present", bool(separator), bool(separator))

    reconstructed_historical = prefix + b"}\n"
    check(
        "historical_bytes_preserved",
        reconstructed_historical == historical_bytes,
        sha256_bytes(reconstructed_historical),
    )

    section["checks_passed"] = sum(1 for item in checks if item["pass"])
    section["checks_total"] = len(checks)
    section["checks"] = checks

    payload = json.dumps(
        section,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    payload = payload.replace("\n", "\n  ")
    appended = (
        historical_bytes[:-2]
        + (',\n  "' + SECTION_KEY + '": ' + payload + "\n}\n").encode("utf-8")
    )
    REPORT.write_bytes(appended)

    final_report_sha = sha256_bytes(REPORT.read_bytes())

    print(
        "S6_07_RQ6_FIXTURE="
        f"{sum(1 for item in checks if item['pass'])}/{len(checks)}_PASS"
    )
    print("PRIMARY_EFFECT=H_s")
    print("CELL_COUNT_PER_SEED=8")
    print("TRAINING_SEEDS=10")
    print("SAMPLE_SD_DDOF=1")
    print("PRIMARY_TEST=TWO_SIDED_ONE_SAMPLE_T_TEST")
    print("N=10")
    print("DF=9")
    print("CI=TWO_SIDED_95_PERCENT")
    print("FAR_FORMAL_P_VALUE_REQUIRED=False")
    print("TEST_USED=False")
    print("HIDDEN_UNLABELED_GT_USED=False")
    print("OFFICIAL_TRAINING_AUTHORIZED=False")
    print("FINAL_TEST_AUTHORIZED=False")
    print(f"CORE_SHA256={core_sha}")
    print(f"VALIDATOR_SHA256={validator_sha}")
    print(f"REPORT_SHA256={final_report_sha}")


if __name__ == "__main__":
    main()
