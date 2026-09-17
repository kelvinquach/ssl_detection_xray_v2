#!/usr/bin/env python
"""Validate the locked S6.04 RQ3 prespecified paired budget contrasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from src.statistics.rq3_pairwise import (
    RQ3_PAIRWISE_ALPHA,
    RQ3_PAIRWISE_ALTERNATIVE,
    RQ3_PAIRWISE_CI_LEVEL,
    RQ3_PAIRWISE_CONTRAST_COUNT,
    RQ3_PAIRWISE_CONTRASTS,
    RQ3_PAIRWISE_DF,
    RQ3_PAIRWISE_HOLM_APPLIED,
    RQ3_PAIRWISE_HOLM_DEFERRED_TASK,
    RQ3_PAIRWISE_NULL_MEAN,
    RQ3PairwiseError,
    rq3_pairwise_tests,
)
from src.statistics.seed_summary import (
    OFFICIAL_SEED_COUNT,
    OFFICIAL_TRAINING_SEEDS,
    REPLICATION_UNIT,
    SAMPLE_SD_DDOF,
)


GOLDEN = np.asarray([
    [1.0, 1.4, 1.8, 2.1],
    [1.2, 1.5, 1.9, 2.4],
    [0.9, 1.3, 1.6, 2.0],
    [1.4, 1.7, 2.2, 2.5],
    [1.1, 1.6, 1.7, 2.3],
    [1.3, 1.8, 2.0, 2.6],
    [0.8, 1.2, 1.5, 1.9],
    [1.5, 1.9, 2.4, 2.7],
    [1.0, 1.5, 2.1, 2.2],
    [1.2, 1.6, 1.8, 2.5],
], dtype=float)

EXPECTED_NAMES = (
    "5%-1%",
    "10%-1%",
    "20%-1%",
    "10%-5%",
    "20%-5%",
    "20%-10%",
)

EXPECTED_MEANS = (
    0.41,
    0.76,
    1.18,
    0.35,
    0.77,
    0.42,
)

EXPECTED_SDS = (
    0.07378647873726217,
    0.15055453054181625,
    0.07888106377466154,
    0.15811388300841903,
    0.08232726023485644,
    0.18135294011647254,
)

EXPECTED_T = (
    17.571428571428573,
    15.963192957919308,
    47.305239818499125,
    6.9999999999999964,
    29.576519264498973,
    7.323601240860617,
)

EXPECTED_P = (
    2.834973060273001e-08,
    6.563138462998302e-08,
    4.222370108240648e-12,
    6.324738963736722e-05,
    2.8194237481486984e-10,
    4.4508692797458326e-05,
)

EXPECTED_CI = (
    (0.3572163328680419, 0.46278366713195807),
    (0.6522997768517405, 0.8677002231482596),
    (1.1235718862784836, 1.2364281137215167),
    (0.2368921418600897, 0.4631078581399103),
    (0.7111066258413513, 0.8288933741586485),
    (0.29026792186959705, 0.549732078130403),
)


def build_report() -> dict[str, object]:
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, observed: object) -> None:
        checks.append(
            {
                "name": name,
                "status": "PASS" if bool(passed) else "FAIL",
                "observed": observed,
            }
        )

    seed_budget = {
        seed: [float(v) for v in row]
        for seed, row in zip(OFFICIAL_TRAINING_SEEDS, GOLDEN, strict=True)
    }

    check("contract::replication_unit", REPLICATION_UNIT == "TRAINING_SEED_TRAINED_RUN", REPLICATION_UNIT)
    check("contract::official_seed_count", OFFICIAL_SEED_COUNT == 10, OFFICIAL_SEED_COUNT)
    check("contract::sample_sd_ddof", SAMPLE_SD_DDOF == 1, SAMPLE_SD_DDOF)
    check("contract::contrast_count", RQ3_PAIRWISE_CONTRAST_COUNT == 6, RQ3_PAIRWISE_CONTRAST_COUNT)
    check("contract::contrast_names", tuple(x[0] for x in RQ3_PAIRWISE_CONTRASTS) == EXPECTED_NAMES, [x[0] for x in RQ3_PAIRWISE_CONTRASTS])
    check("contract::contrast_direction_5_1", RQ3_PAIRWISE_CONTRASTS[0] == ("5%-1%", "5%", "1%"), RQ3_PAIRWISE_CONTRASTS[0])
    check("contract::contrast_direction_10_1", RQ3_PAIRWISE_CONTRASTS[1] == ("10%-1%", "10%", "1%"), RQ3_PAIRWISE_CONTRASTS[1])
    check("contract::contrast_direction_20_1", RQ3_PAIRWISE_CONTRASTS[2] == ("20%-1%", "20%", "1%"), RQ3_PAIRWISE_CONTRASTS[2])
    check("contract::contrast_direction_10_5", RQ3_PAIRWISE_CONTRASTS[3] == ("10%-5%", "10%", "5%"), RQ3_PAIRWISE_CONTRASTS[3])
    check("contract::contrast_direction_20_5", RQ3_PAIRWISE_CONTRASTS[4] == ("20%-5%", "20%", "5%"), RQ3_PAIRWISE_CONTRASTS[4])
    check("contract::contrast_direction_20_10", RQ3_PAIRWISE_CONTRASTS[5] == ("20%-10%", "20%", "10%"), RQ3_PAIRWISE_CONTRASTS[5])
    check("contract::null_mean", RQ3_PAIRWISE_NULL_MEAN == 0.0, RQ3_PAIRWISE_NULL_MEAN)
    check("contract::alternative_two_sided", RQ3_PAIRWISE_ALTERNATIVE == "two-sided", RQ3_PAIRWISE_ALTERNATIVE)
    check("contract::alpha", RQ3_PAIRWISE_ALPHA == 0.05, RQ3_PAIRWISE_ALPHA)
    check("contract::ci_level", RQ3_PAIRWISE_CI_LEVEL == 0.95, RQ3_PAIRWISE_CI_LEVEL)
    check("contract::df", RQ3_PAIRWISE_DF == 9, RQ3_PAIRWISE_DF)
    check("contract::holm_not_applied", RQ3_PAIRWISE_HOLM_APPLIED is False, RQ3_PAIRWISE_HOLM_APPLIED)
    check("contract::holm_deferred_task", RQ3_PAIRWISE_HOLM_DEFERRED_TASK == "S6.10", RQ3_PAIRWISE_HOLM_DEFERRED_TASK)

    result = rq3_pairwise_tests(seed_budget)
    contrasts = result["contrasts"]

    check("positive::top_level_contrast_count", result["contrast_count"] == 6, result["contrast_count"])
    check("positive::top_level_holm_not_applied", result["holm_applied"] is False, result["holm_applied"])
    check("positive::top_level_holm_deferred", result["holm_deferred_task"] == "S6.10", result["holm_deferred_task"])
    check("positive::result_contrast_order", tuple(x["contrast"] for x in contrasts) == EXPECTED_NAMES, [x["contrast"] for x in contrasts])

    numerical_rows: list[dict[str, object]] = []
    budget_index = {"1%": 0, "5%": 1, "10%": 2, "20%": 3}

    for index, row in enumerate(contrasts):
        name = EXPECTED_NAMES[index]
        check(f"positive::{name}::direction", row["direction"] == "b2-b1", row["direction"])
        check(f"positive::{name}::n_df_ddof", row["n"] == 10 and row["df"] == 9 and row["sample_sd_ddof"] == 1, {"n": row["n"], "df": row["df"], "ddof": row["sample_sd_ddof"]})
        check(f"positive::{name}::two_sided_contract", row["alternative"] == "two-sided" and row["p_value_sidedness"] == "two-sided" and row["ci_sidedness"] == "two-sided" and row["ci_level"] == 0.95, {"alternative": row["alternative"], "p_sidedness": row["p_value_sidedness"], "ci_sidedness": row["ci_sidedness"], "ci_level": row["ci_level"]})
        check(f"positive::{name}::mean", np.isclose(row["effect_estimate"], EXPECTED_MEANS[index], rtol=0.0, atol=1e-12), row["effect_estimate"])
        check(f"positive::{name}::sample_sd", np.isclose(row["sample_sd"], EXPECTED_SDS[index], rtol=1e-12, atol=1e-12), row["sample_sd"])
        check(f"positive::{name}::t_statistic", np.isclose(row["t_statistic"], EXPECTED_T[index], rtol=1e-12, atol=1e-12), row["t_statistic"])
        check(f"positive::{name}::raw_p", np.isclose(row["raw_p_value"], EXPECTED_P[index], rtol=1e-10, atol=1e-20), row["raw_p_value"])
        check(f"positive::{name}::ci95", np.isclose(row["ci_low"], EXPECTED_CI[index][0], rtol=1e-12, atol=5e-12) and np.isclose(row["ci_high"], EXPECTED_CI[index][1], rtol=1e-12, atol=5e-12), {"low": row["ci_low"], "high": row["ci_high"]})

        j2 = budget_index[row["budget_2"]]
        j1 = budget_index[row["budget_1"]]
        paired = stats.ttest_rel(GOLDEN[:, j2], GOLDEN[:, j1], alternative="two-sided")
        check(f"positive::{name}::paired_t_equivalence", np.isclose(row["t_statistic"], paired.statistic, rtol=1e-12, atol=1e-12) and np.isclose(row["raw_p_value"], paired.pvalue, rtol=1e-10, atol=1e-20), {"paired_t": float(paired.statistic), "paired_p": float(paired.pvalue)})

        expected_diff = GOLDEN[:, j2] - GOLDEN[:, j1]
        observed_diff = np.asarray([row["paired_differences_by_seed"][seed] for seed in OFFICIAL_TRAINING_SEEDS], dtype=float)
        check(f"positive::{name}::paired_difference_values", np.allclose(observed_diff, expected_diff, rtol=0.0, atol=1e-12), observed_diff.tolist())
        check(f"positive::{name}::seed_order", tuple(row["paired_differences_by_seed"].keys()) == OFFICIAL_TRAINING_SEEDS, list(row["paired_differences_by_seed"].keys()))
        check(f"positive::{name}::no_holm_fields", not any(key.startswith("holm") or "adjusted" in key for key in row.keys()), sorted(row.keys()))

        numerical_rows.append(
            {
                "contrast": name,
                "mean_paired_difference": float(row["effect_estimate"]),
                "sample_sd": float(row["sample_sd"]),
                "n": int(row["n"]),
                "df": int(row["df"]),
                "t_statistic": float(row["t_statistic"]),
                "raw_two_sided_p_value": float(row["raw_p_value"]),
                "ci95_low": float(row["ci_low"]),
                "ci95_high": float(row["ci_high"]),
            }
        )

    def expect_failure(name: str, candidate: object) -> None:
        failed_closed = False
        detail = None
        try:
            rq3_pairwise_tests(candidate)
        except RQ3PairwiseError as exc:
            failed_closed = True
            detail = str(exc)
        check(name, failed_closed, detail)

    missing = dict(seed_budget)
    missing.pop(OFFICIAL_TRAINING_SEEDS[-1])
    expect_failure("negative::missing_seed", missing)

    reordered = {seed: seed_budget[seed] for seed in reversed(OFFICIAL_TRAINING_SEEDS)}
    expect_failure("negative::reordered_seeds", reordered)

    wrong_shape = {seed: list(values) for seed, values in seed_budget.items()}
    wrong_shape[OFFICIAL_TRAINING_SEEDS[0]] = wrong_shape[OFFICIAL_TRAINING_SEEDS[0]][:3]
    expect_failure("negative::wrong_shape", wrong_shape)

    nonfinite = {seed: list(values) for seed, values in seed_budget.items()}
    nonfinite[OFFICIAL_TRAINING_SEEDS[0]][1] = float("nan")
    expect_failure("negative::nonfinite", nonfinite)

    degenerate = {seed: [1.0, 1.0, 1.0, 1.0] for seed in OFFICIAL_TRAINING_SEEDS}
    expect_failure("negative::degenerate_paired_differences", degenerate)

    failed = [item for item in checks if item["status"] != "PASS"]

    return {
        "task": "S6.04",
        "rq": "RQ3",
        "action": "Implement six prespecified paired budget contrasts",
        "status": "PASS" if not failed else "FAIL",
        "checks_total": len(checks),
        "checks_passed": len(checks) - len(failed),
        "failed_count": len(failed),
        "replication_unit": REPLICATION_UNIT,
        "training_seed_count": OFFICIAL_SEED_COUNT,
        "contrast_count": RQ3_PAIRWISE_CONTRAST_COUNT,
        "test": "TWO_SIDED_ONE_SAMPLE_T_ON_PAIRED_DIFFERENCES",
        "paired_t_equivalence": True,
        "sample_sd_ddof": SAMPLE_SD_DDOF,
        "df": RQ3_PAIRWISE_DF,
        "ci": "INDIVIDUAL_TWO_SIDED_95_PERCENT",
        "raw_p_value": "TWO_SIDED",
        "holm_applied": False,
        "holm_deferred_task": "S6.10",
        "test_gt_used": False,
        "hidden_u_gt_used": False,
        "official_training_authorized": False,
        "final_test_authorized": False,
        "golden_results": numerical_rows,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = build_report()
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    print(text)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8", newline="\n")

    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
