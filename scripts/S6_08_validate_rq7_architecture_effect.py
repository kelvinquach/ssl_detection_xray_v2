"""Validate the locked S6.08 / RQ7 architecture-effect statistical fixture."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from src.statistics.rq7_architecture_effect import (
    RQ7_ALTERNATIVE,
    RQ7_ARCHITECTURES,
    RQ7_BUDGETS,
    RQ7_CI_LEVEL,
    RQ7_DF,
    RQ7_TOTAL_ROWS,
    RQ7ArchitectureEffectError,
    compute_rq7_architecture_gain_by_seed,
    compute_rq7_difference_by_seed,
    rq7_primary_test,
)
from src.statistics.seed_summary import (
    OFFICIAL_SEED_COUNT,
    OFFICIAL_TRAINING_SEEDS,
    SAMPLE_SD_DDOF,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts" / "preflight" / "statistics" / "statistical_fixture_report.json"
CORE = ROOT / "src" / "statistics" / "rq7_architecture_effect.py"
VALIDATOR = ROOT / "scripts" / "S6_08_validate_rq7_architecture_effect.py"

EXPECTED_HISTORICAL_REPORT_SHA256 = (
    "2ecb5ef5feb68f5517dcb326457f4e3cd5114879e14f52b9d0b5814b7c9ce34f"
)
EXPECTED_CORE_SHA256 = (
    "3bc1d87feee80ecaad2ea00142f7dd56fb6491d4f2b626db005f32c1c44e6471"
)
SECTION_KEY = "s6_08_rq7_architecture_effect_primary_t_test"
TOL = 1e-9

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
    except (RQ7ArchitectureEffectError, ValueError, TypeError) as exc:
        check(name, True, type(exc).__name__)
        return
    check(name, False, "NO_ERROR")


def build_positive_rows() -> tuple[list[dict[str, object]], list[float]]:
    rows: list[dict[str, object]] = []
    expected_d: list[float] = []
    for seed_index, training_seed in enumerate(OFFICIAL_TRAINING_SEEDS, start=1):
        d_value = seed_index / 10.0
        expected_d.append(d_value)
        r50_base = 1.0 + seed_index / 100.0
        for budget_index, budget in enumerate(RQ7_BUDGETS, start=1):
            r50 = r50_base + budget_index / 1000.0
            swin = r50 + d_value
            rows.append({
                "architecture": "R50",
                "budget": budget,
                "seed_index": seed_index,
                "training_seed": int(training_seed),
                "delta_mAP_test": r50,
            })
            rows.append({
                "architecture": "Swin-T",
                "budget": budget,
                "seed_index": seed_index,
                "training_seed": int(training_seed),
                "delta_mAP_test": swin,
            })
    return rows, expected_d


def main() -> None:
    historical_bytes = REPORT.read_bytes()
    historical_sha = sha256_bytes(historical_bytes)
    check(
        "historical_report_sha256",
        historical_sha == EXPECTED_HISTORICAL_REPORT_SHA256,
        historical_sha,
    )
    check(
        "historical_report_lf_closing",
        historical_bytes.endswith(b"}\n"),
        historical_bytes[-8:].hex(),
    )

    report = json.loads(historical_bytes.decode("utf-8"))
    check("s6_01_status", report.get("status") == "PASS", report.get("status"))
    for key in (
        "s6_02_rq2_primary_t_test",
        "s6_03_rq3_gg_repeated_measures_anova",
        "s6_04_rq3_pairwise_contrasts",
        "s6_05_rq4_mixed_model",
        "s6_06_rq5_rare_class_summary",
        "s6_07_rq6_negative_fp_primary_t_test",
    ):
        section = report.get(key, {})
        check(f"{key}_status", section.get("status") == "PASS", section.get("status"))
    check("s6_08_not_preexisting", SECTION_KEY not in report, SECTION_KEY in report)

    core_sha = sha256_bytes(CORE.read_bytes())
    validator_sha = sha256_bytes(VALIDATOR.read_bytes())
    check("rq7_core_sha256", core_sha == EXPECTED_CORE_SHA256, core_sha)
    check("rq7_core_lf_only", b"\r" not in CORE.read_bytes(), True)
    check("rq7_validator_lf_only", b"\r" not in VALIDATOR.read_bytes(), True)

    check("official_seed_count", OFFICIAL_SEED_COUNT == 10, OFFICIAL_SEED_COUNT)
    check("sample_sd_ddof", SAMPLE_SD_DDOF == 1, SAMPLE_SD_DDOF)
    check("rq7_architectures", RQ7_ARCHITECTURES == ("R50", "Swin-T"), RQ7_ARCHITECTURES)
    check("rq7_budgets", RQ7_BUDGETS == ("1%", "5%", "10%", "20%"), RQ7_BUDGETS)
    check("rq7_total_rows", RQ7_TOTAL_ROWS == 80, RQ7_TOTAL_ROWS)
    check("rq7_alternative", RQ7_ALTERNATIVE == "two-sided", RQ7_ALTERNATIVE)
    check("rq7_df", RQ7_DF == 9, RQ7_DF)
    check("rq7_ci_level", math.isclose(RQ7_CI_LEVEL, 0.95), RQ7_CI_LEVEL)

    rows, expected_d = build_positive_rows()
    check("positive_fixture_row_count", len(rows) == 80, len(rows))
    gains = compute_rq7_architecture_gain_by_seed(rows)
    check("architecture_gain_seed_count", len(gains) == 10, len(gains))
    differences = compute_rq7_difference_by_seed(gains)
    check(
        "difference_seed_order",
        tuple(differences.keys()) == OFFICIAL_TRAINING_SEEDS,
        tuple(differences.keys()),
    )

    for index, training_seed in enumerate(OFFICIAL_TRAINING_SEEDS):
        observed = differences[int(training_seed)]
        expected = expected_d[index]
        check(
            f"d_s_seed_{index + 1:02d}",
            math.isclose(observed, expected, rel_tol=TOL, abs_tol=TOL),
            observed,
        )

    result = rq7_primary_test(differences)
    golden = {
        "effect_estimate": 0.55,
        "sample_sd": 0.30276503540974914,
        "t_statistic": 5.744562646538029,
        "p_value": 0.0002781960110481857,
        "ci_low": 0.33341494103318314,
        "ci_high": 0.766585058966817,
    }
    for key, expected in golden.items():
        observed = float(result[key])
        check(
            f"golden_{key}",
            math.isclose(observed, expected, rel_tol=TOL, abs_tol=TOL),
            observed,
        )

    check("result_effect", result["effect"] == "D_s", result["effect"])
    check(
        "result_effect_definition",
        result["effect_definition"] == "A_Swin-T_minus_A_R50",
        result["effect_definition"],
    )
    check("result_n", result["n"] == 10, result["n"])
    check("result_df", result["df"] == 9, result["df"])
    check("result_ddof", result["sample_sd_ddof"] == 1, result["sample_sd_ddof"])
    check("result_alternative", result["alternative"] == "two-sided", result["alternative"])
    check("result_p_sidedness", result["p_value_sidedness"] == "two-sided", result["p_value_sidedness"])
    check("result_ci_sidedness", result["ci_sidedness"] == "two-sided", result["ci_sidedness"])
    check(
        "interpretation_scope",
        result["interpretation_scope"] == "DIFFERENCE_IN_SSL_GAIN_BETWEEN_ARCHITECTURES_NOT_ABSOLUTE_SUPERIORITY",
        result["interpretation_scope"],
    )

    expect_error("negative_wrong_row_count", lambda: compute_rq7_architecture_gain_by_seed(rows[:-1]))

    bad = copy.deepcopy(rows)
    bad[-1] = copy.deepcopy(bad[0])
    expect_error("negative_duplicate_cell", lambda: compute_rq7_architecture_gain_by_seed(bad))

    bad = copy.deepcopy(rows)
    bad[0]["architecture"] = "INVALID"
    expect_error("negative_invalid_architecture", lambda: compute_rq7_architecture_gain_by_seed(bad))

    bad = copy.deepcopy(rows)
    bad[0]["budget"] = "100%"
    expect_error("negative_invalid_budget", lambda: compute_rq7_architecture_gain_by_seed(bad))

    bad = copy.deepcopy(rows)
    bad[0]["seed_index"] = 10
    expect_error("negative_seed_index_mapping", lambda: compute_rq7_architecture_gain_by_seed(bad))

    bad = copy.deepcopy(rows)
    bad[0]["training_seed"] = 999999
    expect_error("negative_unknown_training_seed", lambda: compute_rq7_architecture_gain_by_seed(bad))

    bad = copy.deepcopy(rows)
    bad[0]["delta_mAP_test"] = float("nan")
    expect_error("negative_nonfinite_delta", lambda: compute_rq7_architecture_gain_by_seed(bad))

    bad = copy.deepcopy(rows)
    del bad[0]["delta_mAP_test"]
    expect_error("negative_missing_delta_field", lambda: compute_rq7_architecture_gain_by_seed(bad))

    reversed_gains = dict(reversed(list(gains.items())))
    expect_error("negative_wrong_seed_order", lambda: compute_rq7_difference_by_seed(reversed_gains))

    bad_gains = copy.deepcopy(gains)
    del bad_gains[int(OFFICIAL_TRAINING_SEEDS[0])]["Swin-T"]
    expect_error("negative_missing_architecture_gain", lambda: compute_rq7_difference_by_seed(bad_gains))

    zero_d = {int(seed): 0.5 for seed in OFFICIAL_TRAINING_SEEDS}
    expect_error("negative_zero_seed_sd", lambda: rq7_primary_test(zero_d))

    scientific_contract = {
        "primitive_effect": "delta_mAP_test=SSL-SUP",
        "architecture_gain": "A_{a,s}=mean_of_four_budget_delta_mAP_test_values",
        "architecture_difference": "D_s=A_Swin-T,s-A_R50,s",
        "replication_unit": "TRAINING_SEED_TRAINED_RUN",
        "official_seed_count": 10,
        "n": 10,
        "df": 9,
        "null_mean": 0.0,
        "alternative": "two-sided",
        "alpha": 0.05,
        "ci": "individual_two-sided_95_percent",
        "absolute_swin_superiority_test": False,
        "holm_f2_applied_in_s6_08": False,
        "exact_sign_flip_applied_in_s6_08": False,
        "loso_applied_in_s6_08": False,
    }

    scientific_boundaries = {
        "test_used": False,
        "test_gt_used": False,
        "hidden_unlabeled_gt_used": False,
        "official_training_authorized": False,
        "final_test_authorized": False,
    }

    section = {
        "task": "S6.08",
        "rq": "RQ7",
        "tracker_action": "Implement two-sided architecture effect",
        "status": "PASS",
        "scientific_contract": scientific_contract,
        "positive_fixture": {
            "row_count": len(rows),
            "training_seed_count": len(differences),
            "seed_differences": [float(differences[int(seed)]) for seed in OFFICIAL_TRAINING_SEEDS],
            "primary_result": result,
        },
        "scientific_boundaries": scientific_boundaries,
        "historical_s6_07_evidence_sha256_before_extension": historical_sha,
        "source_sha256": {
            "src/statistics/rq7_architecture_effect.py": core_sha,
            "scripts/S6_08_validate_rq7_architecture_effect.py": validator_sha,
        },
    }

    section["checks_passed"] = sum(1 for item in checks if item["pass"])
    section["checks_total"] = len(checks)
    section["failed_count"] = sum(1 for item in checks if not item["pass"])
    section["failed_checks"] = [item["name"] for item in checks if not item["pass"]]
    section["checks"] = checks

    payload = json.dumps(section, ensure_ascii=False, indent=2, sort_keys=True)
    payload = payload.replace("\n", "\n  ")
    candidate = (
        historical_bytes[:-2]
        + (",\n  \"" + SECTION_KEY + "\": " + payload + "\n}\n").encode("utf-8")
    )

    parsed_candidate = json.loads(candidate.decode("utf-8"))
    check(
        "candidate_section_present",
        parsed_candidate.get(SECTION_KEY, {}).get("status") == "PASS",
        parsed_candidate.get(SECTION_KEY, {}).get("status"),
    )
    separator = candidate.rpartition((",\n  \"" + SECTION_KEY + "\": ").encode("utf-8"))
    check("candidate_section_marker_present", bool(separator[1]), bool(separator[1]))
    reconstructed_historical = separator[0] + b"}\n"
    check(
        "historical_bytes_preserved",
        reconstructed_historical == historical_bytes,
        sha256_bytes(reconstructed_historical),
    )

    section["checks_passed"] = sum(1 for item in checks if item["pass"])
    section["checks_total"] = len(checks)
    section["failed_count"] = sum(1 for item in checks if not item["pass"])
    section["failed_checks"] = [item["name"] for item in checks if not item["pass"]]
    section["checks"] = checks

    if section["failed_count"] != 0:
        raise AssertionError(section["failed_checks"])

    payload = json.dumps(section, ensure_ascii=False, indent=2, sort_keys=True)
    payload = payload.replace("\n", "\n  ")
    final_bytes = (
        historical_bytes[:-2]
        + (",\n  \"" + SECTION_KEY + "\": " + payload + "\n}\n").encode("utf-8")
    )
    REPORT.write_bytes(final_bytes)

    final_sha = sha256_bytes(final_bytes)
    print(f"S6_08_RQ7_FIXTURE={section['checks_passed']}/{section['checks_total']}_PASS")
    print("PRIMARY_EFFECT=D_s")
    print("ARCHITECTURE_GAIN=A_a_s_MEAN_4_BUDGETS")
    print("DIRECTION=A_Swin-T_MINUS_A_R50")
    print("TRAINING_SEEDS=10")
    print("DF=9")
    print("ALTERNATIVE=two-sided")
    print(f"RQ7_CORE_SHA256={core_sha}")
    print(f"RQ7_VALIDATOR_SHA256={validator_sha}")
    print(f"HISTORICAL_REPORT_SHA256={historical_sha}")
    print(f"FINAL_REPORT_SHA256={final_sha}")
    print("TEST_USED=False")
    print("HIDDEN_UNLABELED_GT_USED=False")
    print("OFFICIAL_TRAINING_AUTHORIZED=False")
    print("FINAL_TEST_AUTHORIZED=False")


if __name__ == "__main__":
    main()
