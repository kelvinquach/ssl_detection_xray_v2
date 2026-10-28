#!/usr/bin/env python3

import argparse
import csv
import hashlib
import math
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.statistics.seed_summary import summarize_seed_runs
from src.statistics.rq3_pairwise import (
    RQ3_PAIRWISE_ALPHA,
    RQ3_PAIRWISE_CI_LEVEL,
    RQ3_PAIRWISE_DF,
)

EXPECTED_SEED_SUMMARY_SHA256 = "2a404ac0f6097cad16816b550fcd1c9ca240fab5535854dc0fd5e4433e49ff3f"
EXPECTED_RQ3_PAIRWISE_SHA256 = "18c0c84fd396193e6019c9a55f051e6ab68ea9cb08e132017d92ddd017be8381"

EXPECTED_PYTHON = "3.10.13"
EXPECTED_NUMPY = "1.26.4"
EXPECTED_SCIPY = "1.15.3"

SEEDS = [
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
]

ARCHITECTURES = ["R50-FPN", "Swin-T-FPN"]
BUDGETS = ["1pct", "5pct", "10pct", "20pct"]

SOURCE_REL = "artifacts/final_test/official_final_test_results.csv"
METRIC = "test_mAP_50_95"
STATISTICS_SEMANTICS = "FROZEN_S6_REUSE"
REPRESENTATION_AUTHORITY = "S10_IMPLEMENTATION_SCHEMA_MINIMAL_FROM_RQ1_CONTRACT"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
            lineterminator="\n",
            extrasaction="raise",
        )
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--seed-output", required=True)
    ap.add_argument("--summary-output", required=True)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    source = Path(args.source)
    seed_output = Path(args.seed_output)
    summary_output = Path(args.summary_output)

    require(
        sha256_file(repo / "src/statistics/seed_summary.py")
        == EXPECTED_SEED_SUMMARY_SHA256,
        "SEED_SUMMARY_SHA256_MISMATCH",
    )
    require(
        sha256_file(repo / "src/statistics/rq3_pairwise.py")
        == EXPECTED_RQ3_PAIRWISE_SHA256,
        "RQ3_PAIRWISE_SHA256_MISMATCH",
    )

    require(
        platform.python_version() == EXPECTED_PYTHON,
        f"PYTHON_VERSION_MISMATCH:{platform.python_version()}!={EXPECTED_PYTHON}",
    )
    require(
        np.__version__ == EXPECTED_NUMPY,
        f"NUMPY_VERSION_MISMATCH:{np.__version__}!={EXPECTED_NUMPY}",
    )
    require(
        scipy.__version__ == EXPECTED_SCIPY,
        f"SCIPY_VERSION_MISMATCH:{scipy.__version__}!={EXPECTED_SCIPY}",
    )

    require(RQ3_PAIRWISE_ALPHA == 0.05, "ALPHA_NOT_0_05")
    require(RQ3_PAIRWISE_CI_LEVEL == 0.95, "CI_LEVEL_NOT_0_95")
    require(RQ3_PAIRWISE_DF == 9, "CI_DF_NOT_9")

    rows = load_csv(source)
    require(len(rows) == 180, f"SOURCE_ROWS={len(rows)}")

    rq1 = [
        r
        for r in rows
        if r["condition_role"] == "MAIN_LOW_LABEL"
        and r["method"] == "SUP"
        and r["budget"] in BUDGETS
    ]

    require(len(rq1) == 80, f"RQ1_ROWS={len(rq1)}")
    require(
        len({r["run_id"] for r in rq1}) == 80,
        "RQ1_RUN_ID_NOT_UNIQUE",
    )

    seed_fields = [
        "run_id",
        "condition_role",
        "method",
        "architecture",
        "budget",
        "seed_index",
        "training_seed",
        "metric_name",
        "test_mAP_50_95",
        "final_evaluation_status",
        "source_artifact",
        "formal_hypothesis_test",
        "representation_authority",
    ]

    summary_fields = [
        "architecture",
        "budget",
        "metric_name",
        "n_seeds",
        "mean",
        "sample_SD_ddof1",
        "mean_two_sided_95CI_low",
        "mean_two_sided_95CI_high",
        "ci_level",
        "df",
        "uncertainty_role",
        "formal_hypothesis_test",
        "p_value_computed",
        "statistics_semantics",
        "source_artifact",
        "representation_authority",
    ]

    seed_rows = []
    summary_rows = []

    for arch in ARCHITECTURES:
        for budget in BUDGETS:
            group = [
                r
                for r in rq1
                if r["architecture"] == arch
                and r["budget"] == budget
            ]
            require(
                len(group) == 10,
                f"GROUP_ROWS:{arch}:{budget}:{len(group)}",
            )

            by_seed = {}
            for r in group:
                seed = int(r["training_seed"])
                require(seed not in by_seed, f"DUPLICATE_SEED:{arch}:{budget}:{seed}")
                require(
                    r["final_evaluation_status"] == "CLOSED_VERIFIED",
                    f"FINAL_STATUS_NOT_CLOSED_VERIFIED:{r['run_id']}",
                )
                try:
                    value = float(r[METRIC])
                except Exception as e:
                    raise RuntimeError(
                        f"NON_NUMERIC_METRIC:{r['run_id']}:{r[METRIC]}"
                    ) from e
                require(math.isfinite(value), f"NONFINITE_METRIC:{r['run_id']}")
                by_seed[seed] = value

            require(
                set(by_seed.keys()) == set(SEEDS),
                f"FROZEN_SEED_SET_MISMATCH:{arch}:{budget}",
            )

            ordered_group = sorted(
                group,
                key=lambda r: int(r["seed_index"]),
            )

            require(
                [int(r["training_seed"]) for r in ordered_group] == SEEDS,
                f"FROZEN_SEED_ORDER_MISMATCH:{arch}:{budget}",
            )

            for r in ordered_group:
                seed_rows.append(
                    {
                        "run_id": r["run_id"],
                        "condition_role": r["condition_role"],
                        "method": r["method"],
                        "architecture": r["architecture"],
                        "budget": r["budget"],
                        "seed_index": r["seed_index"],
                        "training_seed": r["training_seed"],
                        "metric_name": METRIC,
                        "test_mAP_50_95": r[METRIC],
                        "final_evaluation_status": r["final_evaluation_status"],
                        "source_artifact": SOURCE_REL,
                        "formal_hypothesis_test": "false",
                        "representation_authority": REPRESENTATION_AUTHORITY,
                    }
                )

            s = summarize_seed_runs(by_seed)
            require(int(s["n"]) == 10, f"N_NOT_10:{arch}:{budget}")
            require(
                int(s["sample_sd_ddof"]) == 1,
                f"DDOF_NOT_1:{arch}:{budget}",
            )

            mean = float(s["mean"])
            sample_sd = float(s["sample_sd"])
            standard_error = sample_sd / math.sqrt(10)
            critical_value = float(
                stats.t.ppf(
                    1.0 - RQ3_PAIRWISE_ALPHA / 2.0,
                    df=RQ3_PAIRWISE_DF,
                )
            )
            ci_low = mean - critical_value * standard_error
            ci_high = mean + critical_value * standard_error

            for value in (
                mean,
                sample_sd,
                standard_error,
                critical_value,
                ci_low,
                ci_high,
            ):
                require(math.isfinite(value), f"NONFINITE_SUMMARY:{arch}:{budget}")

            summary_rows.append(
                {
                    "architecture": arch,
                    "budget": budget,
                    "metric_name": METRIC,
                    "n_seeds": 10,
                    "mean": repr(mean),
                    "sample_SD_ddof1": repr(sample_sd),
                    "mean_two_sided_95CI_low": repr(ci_low),
                    "mean_two_sided_95CI_high": repr(ci_high),
                    "ci_level": repr(RQ3_PAIRWISE_CI_LEVEL),
                    "df": RQ3_PAIRWISE_DF,
                    "uncertainty_role": "DESCRIPTIVE",
                    "formal_hypothesis_test": "false",
                    "p_value_computed": "false",
                    "statistics_semantics": STATISTICS_SEMANTICS,
                    "source_artifact": SOURCE_REL,
                    "representation_authority": REPRESENTATION_AUTHORITY,
                }
            )

    require(len(seed_rows) == 80, f"SEED_OUTPUT_ROWS={len(seed_rows)}")
    require(len(summary_rows) == 8, f"SUMMARY_OUTPUT_ROWS={len(summary_rows)}")

    write_csv(seed_output, seed_fields, seed_rows)
    write_csv(summary_output, summary_fields, summary_rows)

    print("=== S10.1.10 RQ1 ===")
    print(f"SEED_ROWS={len(seed_rows)}")
    print(f"SUMMARY_ROWS={len(summary_rows)}")
    print("ARCHITECTURES=2")
    print("BUDGETS=4")
    print("SEEDS_PER_GROUP=10")
    print("METRIC=test_mAP_50_95")
    print("SAMPLE_SD_DDOF=1")
    print("CI_METHOD=STUDENT_T")
    print("CI_LEVEL=0.95")
    print("CI_DF=9")
    print("UNCERTAINTY_ROLE=DESCRIPTIVE")
    print("FORMAL_HYPOTHESIS_TEST=FALSE")
    print("P_VALUE_COMPUTED=FALSE")
    print("STATISTICS_CORE=FROZEN_S6_REUSE")
    print("S10_1_10_STATUS=PASS")


if __name__ == "__main__":
    main()
