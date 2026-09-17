"""Validate S6.06 RQ5 rare-class summary fixtures and extend shared evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.statistics.rq5_rare_class import (
    RQ5_ALPHA,
    RQ5_ARCHITECTURES,
    RQ5_BUDGETS,
    RQ5_CELL_COUNT,
    RQ5_CI_LEVEL,
    RQ5_DEFINED_STATUS,
    RQ5_DELTA_COLUMNS,
    RQ5_FORMAL_P_VALUE_REQUIRED,
    RQ5_INPUT_EFFECT_UNIT,
    RQ5_MISSINGNESS_POLICY,
    RQ5_RARE_CLASSES,
    RQ5_RARE_MAP_CREATED,
    RQ5_UNDEFINED_STATUS,
    RQ5RareClassError,
    compute_rq5_seed_gains,
    summarize_rq5_classes,
)
from src.statistics.seed_summary import (
    OFFICIAL_SEED_COUNT,
    OFFICIAL_TRAINING_SEEDS,
    REPLICATION_UNIT,
    SAMPLE_SD_DDOF,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/preflight/statistics/statistical_fixture_report.json"
CORE = ROOT / "src/statistics/rq5_rare_class.py"
VALIDATOR = Path(__file__).resolve()

EXPECTED_HISTORICAL_REPORT_SHA256 = (
    "9a8da61e528a7edcac30246295b3bbaf1a7b2ba4e4403692312550249b191303"
)
EXPECTED_CORE_SHA256 = (
    "d284a41528ae0340a7e881877a10abcfd09d2a712fa47d6e44ce6e72b6336a7a"
)

checks: list[dict[str, object]] = []
negative: dict[str, dict[str, object]] = {}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check(name: str, condition: bool, observed: object) -> None:
    checks.append({
        "name": name,
        "status": "PASS" if condition else "FAIL",
        "observed": observed,
    })


def expect_failure(name: str, fn) -> None:
    try:
        fn()
    except RQ5RareClassError as exc:
        negative[name] = {"status": "PASS", "observed": str(exc)}
        check(name, True, str(exc))
    else:
        negative[name] = {"status": "FAIL", "observed": "NOT_BLOCKED"}
        check(name, False, "NOT_BLOCKED")


historical_sha = sha256_file(REPORT)
core_sha = sha256_file(CORE)
historical_text = REPORT.read_text(encoding="utf-8")
historical = json.loads(historical_text)

check(
    "historical::shared_evidence_sha256",
    historical_sha == EXPECTED_HISTORICAL_REPORT_SHA256,
    historical_sha,
)
check(
    "historical::s6_05_status",
    historical.get("s6_05_rq4_mixed_model", {}).get("status") == "PASS",
    historical.get("s6_05_rq4_mixed_model", {}).get("status"),
)
check(
    "historical::s6_05_checks_42_of_42",
    historical.get("s6_05_rq4_mixed_model", {}).get("checks_passed") == 42
    and historical.get("s6_05_rq4_mixed_model", {}).get("checks_total") == 42,
    {
        "passed": historical.get("s6_05_rq4_mixed_model", {}).get("checks_passed"),
        "total": historical.get("s6_05_rq4_mixed_model", {}).get("checks_total"),
    },
)
check("source::rq5_core_sha256", core_sha == EXPECTED_CORE_SHA256, core_sha)

check(
    "contract::rare_classes",
    RQ5_RARE_CLASSES == ("Atelectasis", "Pneumothorax"),
    list(RQ5_RARE_CLASSES),
)
check(
    "contract::architectures",
    RQ5_ARCHITECTURES == ("R50", "Swin-T"),
    list(RQ5_ARCHITECTURES),
)
check(
    "contract::budgets",
    RQ5_BUDGETS == ("1%", "5%", "10%", "20%"),
    list(RQ5_BUDGETS),
)
check("contract::cell_count_8", RQ5_CELL_COUNT == 8, RQ5_CELL_COUNT)
check("contract::seed_count_10", OFFICIAL_SEED_COUNT == 10, OFFICIAL_SEED_COUNT)
check(
    "contract::ordered_training_seeds",
    OFFICIAL_TRAINING_SEEDS
    == (
        204886845,
        1480646854,
        1798418854,
        2045683682,
        1814859839,
        1603952859,
        1878351743,
        875651179,
        477581743,
        869675675,
    ),
    list(OFFICIAL_TRAINING_SEEDS),
)
check(
    "contract::replication_unit",
    REPLICATION_UNIT == "TRAINING_SEED_TRAINED_RUN",
    REPLICATION_UNIT,
)
check("contract::sample_sd_ddof_1", SAMPLE_SD_DDOF == 1, SAMPLE_SD_DDOF)
check("contract::ci_level_095", RQ5_CI_LEVEL == 0.95, RQ5_CI_LEVEL)
check("contract::alpha_005", RQ5_ALPHA == 0.05, RQ5_ALPHA)
check(
    "contract::effect_unit_ap_points",
    RQ5_INPUT_EFFECT_UNIT == "AP_POINTS",
    RQ5_INPUT_EFFECT_UNIT,
)
check(
    "contract::missingness_policy",
    RQ5_MISSINGNESS_POLICY == "PROPAGATE_NA_NO_AVAILABLE_CASE_MEAN",
    RQ5_MISSINGNESS_POLICY,
)
check(
    "contract::formal_p_value_not_required",
    RQ5_FORMAL_P_VALUE_REQUIRED is False,
    RQ5_FORMAL_P_VALUE_REQUIRED,
)
check(
    "contract::rare_map_not_created",
    RQ5_RARE_MAP_CREATED is False,
    RQ5_RARE_MAP_CREATED,
)
check(
    "contract::delta_columns",
    RQ5_DELTA_COLUMNS
    == {
        "Atelectasis": "delta_AP_test_Atelectasis",
        "Pneumothorax": "delta_AP_test_Pneumothorax",
    },
    RQ5_DELTA_COLUMNS,
)

rows: list[dict[str, object]] = []
for seed_index, training_seed in enumerate(OFFICIAL_TRAINING_SEEDS, start=1):
    for architecture in RQ5_ARCHITECTURES:
        for budget in RQ5_BUDGETS:
            rows.append({
                "architecture": architecture,
                "budget": budget,
                "seed_index": seed_index,
                "training_seed": training_seed,
                "delta_AP_test_Atelectasis": float(seed_index),
                "delta_AP_test_Pneumothorax": float(2 * seed_index),
            })

seed_gains = compute_rq5_seed_gains(rows)
summaries = summarize_rq5_classes(seed_gains)

ate_seed = [x for x in seed_gains if x["class_name"] == "Atelectasis"]
ptx_seed = [x for x in seed_gains if x["class_name"] == "Pneumothorax"]
ate = next(x for x in summaries if x["class_name"] == "Atelectasis")
ptx = next(x for x in summaries if x["class_name"] == "Pneumothorax")

expected_sd = 3.0276503540974917
expected_ate_ci = (3.3341494102783162, 7.665850589721684)
expected_ptx_ci = (6.6682988205566324, 15.331701179443368)

check("positive::input_rows_80", len(rows) == 80, len(rows))
check("positive::seed_class_rows_20", len(seed_gains) == 20, len(seed_gains))
check("positive::summary_rows_2", len(summaries) == 2, len(summaries))
check(
    "positive::atelectasis_seed_gains",
    np.allclose(
        [x["g_c_s"] for x in ate_seed],
        [float(i) for i in range(1, 11)],
        rtol=0.0,
        atol=1e-12,
    ),
    [x["g_c_s"] for x in ate_seed],
)
check(
    "positive::pneumothorax_seed_gains",
    np.allclose(
        [x["g_c_s"] for x in ptx_seed],
        [float(2 * i) for i in range(1, 11)],
        rtol=0.0,
        atol=1e-12,
    ),
    [x["g_c_s"] for x in ptx_seed],
)
check(
    "positive::all_seed_rows_defined",
    all(x["status"] == RQ5_DEFINED_STATUS for x in seed_gains),
    [x["status"] for x in seed_gains],
)
check(
    "positive::atelectasis_mean",
    math.isclose(float(ate["mean_paired_gain"]), 5.5, rel_tol=0.0, abs_tol=1e-12),
    ate["mean_paired_gain"],
)
check(
    "positive::pneumothorax_mean",
    math.isclose(float(ptx["mean_paired_gain"]), 11.0, rel_tol=0.0, abs_tol=1e-12),
    ptx["mean_paired_gain"],
)
check(
    "positive::atelectasis_sample_sd_ddof1",
    math.isclose(float(ate["sample_sd"]), expected_sd, rel_tol=1e-12, abs_tol=1e-12)
    and ate["sample_sd_ddof"] == 1,
    {"sample_sd": ate["sample_sd"], "ddof": ate["sample_sd_ddof"]},
)
check(
    "positive::pneumothorax_sample_sd_ddof1",
    math.isclose(float(ptx["sample_sd"]), 2.0 * expected_sd, rel_tol=1e-12, abs_tol=1e-12)
    and ptx["sample_sd_ddof"] == 1,
    {"sample_sd": ptx["sample_sd"], "ddof": ptx["sample_sd_ddof"]},
)
check(
    "positive::n10_df9",
    ate["n"] == 10 and ate["df"] == 9 and ptx["n"] == 10 and ptx["df"] == 9,
    {"ate_n": ate["n"], "ate_df": ate["df"], "ptx_n": ptx["n"], "ptx_df": ptx["df"]},
)
check(
    "positive::two_sided_95_ci_contract",
    ate["ci_level"] == 0.95
    and ate["ci_sidedness"] == "two-sided"
    and ptx["ci_level"] == 0.95
    and ptx["ci_sidedness"] == "two-sided",
    {
        "ate": [ate["ci_sidedness"], ate["ci_level"]],
        "ptx": [ptx["ci_sidedness"], ptx["ci_level"]],
    },
)
check(
    "positive::atelectasis_ci95_golden",
    math.isclose(float(ate["ci_low"]), expected_ate_ci[0], rel_tol=1e-9, abs_tol=1e-9)
    and math.isclose(float(ate["ci_high"]), expected_ate_ci[1], rel_tol=1e-9, abs_tol=1e-9),
    {"low": ate["ci_low"], "high": ate["ci_high"]},
)
check(
    "positive::pneumothorax_ci95_golden",
    math.isclose(float(ptx["ci_low"]), expected_ptx_ci[0], rel_tol=1e-9, abs_tol=1e-9)
    and math.isclose(float(ptx["ci_high"]), expected_ptx_ci[1], rel_tol=1e-9, abs_tol=1e-9),
    {"low": ptx["ci_low"], "high": ptx["ci_high"]},
)
check(
    "positive::formal_p_value_false",
    ate["formal_p_value_required"] is False
    and ptx["formal_p_value_required"] is False,
    [ate["formal_p_value_required"], ptx["formal_p_value_required"]],
)
check(
    "positive::no_p_value_result_fields",
    all(
        key not in row
        for row in summaries
        for key in ("p_value", "raw_p_value", "one_sided_p_value")
    ),
    [sorted(row.keys()) for row in summaries],
)
check(
    "positive::no_rare_map_field",
    all("rare_mAP" not in row and "rare_map" not in row for row in summaries),
    [sorted(row.keys()) for row in summaries],
)

na_rows = [dict(row) for row in rows]
na_rows[0]["delta_AP_test_Atelectasis"] = None
na_seed_gains = compute_rq5_seed_gains(na_rows)
na_summaries = summarize_rq5_classes(na_seed_gains)
na_seed = next(
    x for x in na_seed_gains
    if x["class_name"] == "Atelectasis"
    and x["training_seed"] == OFFICIAL_TRAINING_SEEDS[0]
)
na_ate = next(x for x in na_summaries if x["class_name"] == "Atelectasis")
na_ptx = next(x for x in na_summaries if x["class_name"] == "Pneumothorax")

check(
    "na::cell_propagates_to_seed",
    na_seed["status"] == RQ5_UNDEFINED_STATUS and na_seed["g_c_s"] is None,
    na_seed,
)
check(
    "na::no_available_case_7_of_8_mean",
    na_seed["defined_cell_count"] == 7
    and na_seed["required_cell_count"] == 8
    and na_seed["g_c_s"] is None,
    na_seed,
)
check(
    "na::seed_propagates_to_class_summary",
    na_ate["status"] == RQ5_UNDEFINED_STATUS
    and na_ate["mean_paired_gain"] is None
    and na_ate["sample_sd"] is None
    and na_ate["ci_low"] is None
    and na_ate["ci_high"] is None,
    na_ate,
)
check(
    "na::undefined_seed_identity",
    na_ate["undefined_training_seeds"] == [OFFICIAL_TRAINING_SEEDS[0]],
    na_ate["undefined_training_seeds"],
)
check(
    "na::unrelated_class_remains_defined",
    na_ptx["status"] == RQ5_DEFINED_STATUS
    and math.isclose(float(na_ptx["mean_paired_gain"]), 11.0, rel_tol=0.0, abs_tol=1e-12),
    na_ptx,
)

nan_rows = [dict(row) for row in rows]
nan_rows[0]["delta_AP_test_Atelectasis"] = float("nan")
nan_gains = compute_rq5_seed_gains(nan_rows)
nan_first = next(
    x for x in nan_gains
    if x["class_name"] == "Atelectasis"
    and x["training_seed"] == OFFICIAL_TRAINING_SEEDS[0]
)
check(
    "na::nan_semantics_match_none",
    nan_first["status"] == RQ5_UNDEFINED_STATUS and nan_first["g_c_s"] is None,
    nan_first,
)

expect_failure(
    "negative::missing_row",
    lambda: compute_rq5_seed_gains(rows[:-1]),
)

duplicate_rows = [dict(row) for row in rows]
duplicate_rows[-1] = dict(duplicate_rows[0])
expect_failure(
    "negative::duplicate_cell_seed",
    lambda: compute_rq5_seed_gains(duplicate_rows),
)

bad_seed = [dict(row) for row in rows]
bad_seed[0]["training_seed"] = 123
expect_failure(
    "negative::bad_seed_mapping",
    lambda: compute_rq5_seed_gains(bad_seed),
)

bad_arch = [dict(row) for row in rows]
bad_arch[0]["architecture"] = "INVALID"
expect_failure(
    "negative::invalid_architecture",
    lambda: compute_rq5_seed_gains(bad_arch),
)

bad_budget = [dict(row) for row in rows]
bad_budget[0]["budget"] = "99%"
expect_failure(
    "negative::invalid_budget",
    lambda: compute_rq5_seed_gains(bad_budget),
)

bad_inf = [dict(row) for row in rows]
bad_inf[0]["delta_AP_test_Atelectasis"] = float("inf")
expect_failure(
    "negative::infinite_effect",
    lambda: compute_rq5_seed_gains(bad_inf),
)

bad_range = [dict(row) for row in rows]
bad_range[0]["delta_AP_test_Atelectasis"] = 101.0
expect_failure(
    "negative::effect_outside_ap_point_range",
    lambda: compute_rq5_seed_gains(bad_range),
)

reordered = list(reversed(seed_gains))
expect_failure(
    "negative::reordered_seed_summary_rows",
    lambda: summarize_rq5_classes(reordered),
)

missing_seed_summary = [
    row
    for row in seed_gains
    if not (
        row["class_name"] == "Atelectasis"
        and row["training_seed"] == OFFICIAL_TRAINING_SEEDS[-1]
    )
]
expect_failure(
    "negative::missing_seed_summary_row",
    lambda: summarize_rq5_classes(missing_seed_summary),
)

failed = [row for row in checks if row["status"] != "PASS"]
status = "PASS" if not failed else "FAIL"

validator_sha = sha256_file(VALIDATOR)

entry = {
    "task": "S6.06",
    "rq": "RQ5",
    "tracker_action": "Implement rare-class summaries",
    "status": status,
    "checks_total": len(checks),
    "checks_passed": len(checks) - len(failed),
    "failed_count": len(failed),
    "failed_checks": [row["name"] for row in failed],
    "scientific_contract": {
        "rare_classes": list(RQ5_RARE_CLASSES),
        "effect": "G_c_s",
        "g_c_s_definition": "EQUAL_WEIGHT_MEAN_OF_8_PAIRED_ARCHITECTURE_BUDGET_DELTA_AP_CELLS",
        "cell_count_per_class_seed": RQ5_CELL_COUNT,
        "cell_weight": 1.0 / RQ5_CELL_COUNT,
        "effect_unit": RQ5_INPUT_EFFECT_UNIT,
        "inferential_replication_unit": REPLICATION_UNIT,
        "n_training_seeds": OFFICIAL_SEED_COUNT,
        "sample_sd_ddof": SAMPLE_SD_DDOF,
        "ci": "TWO_SIDED_95_PERCENT",
        "formal_p_value_required": False,
        "rare_map_created": False,
        "missingness_policy": RQ5_MISSINGNESS_POLICY,
        "undefined_ap": "NA_NOT_ZERO",
    },
    "positive_fixture": {
        "synthetic_only": True,
        "input_rows": len(rows),
        "seed_class_rows": len(seed_gains),
        "class_summaries": summaries,
    },
    "na_fixture": {
        "synthetic_only": True,
        "single_undefined_cell_class": "Atelectasis",
        "single_undefined_cell_seed": OFFICIAL_TRAINING_SEEDS[0],
        "seed_result": na_seed,
        "class_result": na_ate,
        "unrelated_class_result": na_ptx,
    },
    "negative_fixtures": negative,
    "scientific_boundaries": {
        "fixture_only_no_official_results": True,
        "official_training_authorized": False,
        "final_test_authorized": False,
        "test_used": False,
        "hidden_unlabeled_gt_used": False,
        "available_case_mean_used": False,
        "undefined_ap_imputed_zero": False,
        "rare_map_created": False,
    },
    "historical_s6_05_evidence_sha256_before_extension": historical_sha,
    "source_sha256": {
        "src/statistics/rq5_rare_class.py": core_sha,
        "scripts/S6_06_validate_rq5_rare_class.py": validator_sha,
    },
    "checks": checks,
}

if failed:
    print(f"S6_06_RQ5_FIXTURE={len(checks)-len(failed)}/{len(checks)}_FAIL")
    for row in failed:
        print(f"FAILED={row['name']} OBSERVED={row['observed']}")
    raise SystemExit(1)

if "s6_06_rq5_rare_class_summary" in historical:
    raise RuntimeError("S6.06 section already exists in historical report")
section_text = json.dumps(
    {"s6_06_rq5_rare_class_summary": entry},
    indent=2,
    ensure_ascii=False,
)
if not historical_text.endswith("}\n"):
    raise RuntimeError("historical report must end with exactly closing brace plus newline")
section_body = section_text[2:-2]
REPORT.write_text(
    historical_text[:-2] + ",\n" + section_body + "\n}\n",
    encoding="utf-8",
)

print(f"S6_06_RQ5_FIXTURE={len(checks)}/{len(checks)}_PASS")
print("RARE_CLASSES=Atelectasis,Pneumothorax")
print("CELL_COUNT_PER_CLASS_SEED=8")
print("TRAINING_SEEDS=10")
print("SAMPLE_SD_DDOF=1")
print("CI=TWO_SIDED_95_PERCENT")
print("FORMAL_P_VALUE_REQUIRED=False")
print("RARE_MAP_CREATED=False")
print("UNDEFINED_AP=NA_NOT_ZERO")
print("AVAILABLE_CASE_MEAN_USED=False")
print("TEST_USED=False")
print("HIDDEN_UNLABELED_GT_USED=False")
print("OFFICIAL_TRAINING_AUTHORIZED=False")
print("FINAL_TEST_AUTHORIZED=False")
print("CORE_SHA256=" + core_sha)
print("VALIDATOR_SHA256=" + validator_sha)
print("REPORT_SHA256=" + sha256_file(REPORT))
