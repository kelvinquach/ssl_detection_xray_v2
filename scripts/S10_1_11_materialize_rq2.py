#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
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

from src.statistics.seed_summary import (
    OFFICIAL_TRAINING_SEEDS,
    summarize_seed_runs,
)
from src.statistics.rq2_primary import (
    RQ2_ALPHA,
    RQ2_ALTERNATIVE,
    RQ2_CELL_COUNT,
    RQ2_CELL_WEIGHT,
    RQ2_CI_LEVEL,
    RQ2_DF,
    RQ2_NULL_MEAN,
    compute_rq2_gain_by_seed,
    rq2_primary_test,
)

EXPECTED_SEED_SUMMARY_SHA256 = (
    "2a404ac0f6097cad16816b550fcd1c9ca240fab5535854dc0fd5e4433e49ff3f"
)
EXPECTED_RQ2_PRIMARY_SHA256 = (
    "a9dea0204ec7f38cba428736edbe9f1c70f4583f71103e2b8dbc21be83121c3a"
)
EXPECTED_PAIRED_EFFECTS_SHA256 = (
    "bad5a179711443636a13bd6501a1d90a7e8c5dadfb16b9c09df74bf502b727bb"
)

EXPECTED_PYTHON = "3.10.13"
EXPECTED_NUMPY = "1.26.4"
EXPECTED_SCIPY = "1.15.3"

ARCHITECTURES = ["R50-FPN", "Swin-T-FPN"]
BUDGETS = ["1pct", "5pct", "10pct", "20pct"]

SOURCE_REL = "artifacts/results/paired_effects.csv"
METRIC = "delta_mAP_test"
STATISTICS_SEMANTICS = "FROZEN_S6_REUSE"
REPRESENTATION_AUTHORITY = (
    "S10_IMPLEMENTATION_SCHEMA_MINIMAL_FROM_RQ2_CONTRACT"
)


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


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--gain-output", required=True)
    ap.add_argument("--primary-output", required=True)
    ap.add_argument("--cell-output", required=True)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    source = Path(args.source)
    gain_output = Path(args.gain_output)
    primary_output = Path(args.primary_output)
    cell_output = Path(args.cell_output)

    # Frozen source identity.
    require(
        sha256_file(repo / "src/statistics/seed_summary.py")
        == EXPECTED_SEED_SUMMARY_SHA256,
        "SEED_SUMMARY_SHA256_MISMATCH",
    )
    require(
        sha256_file(repo / "src/statistics/rq2_primary.py")
        == EXPECTED_RQ2_PRIMARY_SHA256,
        "RQ2_PRIMARY_SHA256_MISMATCH",
    )

    # Canonical S10 source identity.
    require(
        sha256_file(source) == EXPECTED_PAIRED_EFFECTS_SHA256,
        "PAIRED_EFFECTS_SHA256_MISMATCH",
    )

    # Canonical Python implementation reference.
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

    # Frozen RQ2 contract.
    require(RQ2_CELL_COUNT == 8, "RQ2_CELL_COUNT_NOT_8")
    require(math.isclose(RQ2_CELL_WEIGHT, 0.125), "RQ2_CELL_WEIGHT_NOT_0_125")
    require(RQ2_NULL_MEAN == 0.0, "RQ2_NULL_MEAN_NOT_ZERO")
    require(RQ2_ALTERNATIVE == "greater", "RQ2_ALTERNATIVE_NOT_GREATER")
    require(RQ2_ALPHA == 0.05, "RQ2_ALPHA_NOT_0_05")
    require(RQ2_CI_LEVEL == 0.95, "RQ2_CI_LEVEL_NOT_0_95")
    require(RQ2_DF == 9, "RQ2_DF_NOT_9")

    rows = load_csv(source)
    require(len(rows) == 80, f"SOURCE_ROWS={len(rows)}")
    require(
        len({r["pair_id"] for r in rows}) == 80,
        "PAIR_ID_NOT_UNIQUE",
    )

    expected_cells = {
        (architecture, budget)
        for architecture in ARCHITECTURES
        for budget in BUDGETS
    }

    by_identity = {}
    for r in rows:
        architecture = r["architecture"]
        budget = r["budget"]
        seed = int(r["training_seed"])

        require(
            architecture in ARCHITECTURES,
            f"UNEXPECTED_ARCHITECTURE:{architecture}",
        )
        require(
            budget in BUDGETS,
            f"UNEXPECTED_BUDGET:{budget}",
        )
        require(
            seed in OFFICIAL_TRAINING_SEEDS,
            f"UNEXPECTED_SEED:{seed}",
        )

        identity = (architecture, budget, seed)
        require(
            identity not in by_identity,
            f"DUPLICATE_CELL_SEED:{architecture}:{budget}:{seed}",
        )

        try:
            sup = float(r["mAP_test_SUP"])
            ssl = float(r["mAP_test_SSL"])
            delta = float(r["delta_mAP_test"])
        except Exception as e:
            raise RuntimeError(
                f"NON_NUMERIC_RQ2_SOURCE:{r['pair_id']}"
            ) from e

        require(
            all(math.isfinite(v) for v in (sup, ssl, delta)),
            f"NONFINITE_RQ2_SOURCE:{r['pair_id']}",
        )
        require(
            math.isclose(
                ssl - sup,
                delta,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"DELTA_MISMATCH:{r['pair_id']}",
        )

        by_identity[identity] = delta

    # Build the exact frozen RQ2 input:
    # ordered seed -> exactly eight architecture x budget delta-mAP cells.
    seed_cell_effects = {}
    for seed in OFFICIAL_TRAINING_SEEDS:
        observed = {
            (architecture, budget)
            for architecture in ARCHITECTURES
            for budget in BUDGETS
            if (architecture, budget, int(seed)) in by_identity
        }
        require(
            observed == expected_cells,
            f"SEED_CELL_SET_MISMATCH:{seed}",
        )

        seed_cell_effects[int(seed)] = [
            by_identity[(architecture, budget, int(seed))]
            for architecture in ARCHITECTURES
            for budget in BUDGETS
        ]

    # DO NOT reimplement RQ2 aggregation.
    gains = compute_rq2_gain_by_seed(seed_cell_effects)

    # DO NOT reimplement the primary statistical test.
    primary = rq2_primary_test(gains)

    gain_fields = [
        "seed_index",
        "training_seed",
        "effect",
        "cell_count",
        "cell_weight",
        "G_s",
        "source_artifact",
        "statistics_semantics",
        "representation_authority",
    ]

    gain_rows = []
    for seed_index, seed in enumerate(
        OFFICIAL_TRAINING_SEEDS,
        start=1,
    ):
        gain_rows.append(
            {
                "seed_index": seed_index,
                "training_seed": int(seed),
                "effect": "G_s",
                "cell_count": RQ2_CELL_COUNT,
                "cell_weight": repr(RQ2_CELL_WEIGHT),
                "G_s": repr(float(gains[int(seed)])),
                "source_artifact": SOURCE_REL,
                "statistics_semantics": STATISTICS_SEMANTICS,
                "representation_authority": REPRESENTATION_AUTHORITY,
            }
        )

    # RQ2 cell descriptives:
    # exactly 8 architecture x budget cells, each summarized over 10 seeds.
    # This is descriptive only: mean, sample SD ddof=1, two-sided 95% CI.
    cell_fields = [
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
        "source_artifact",
        "statistics_semantics",
        "representation_authority",
    ]

    cell_rows = []
    critical_value = float(
        stats.t.ppf(
            1.0 - RQ2_ALPHA / 2.0,
            df=RQ2_DF,
        )
    )

    for architecture in ARCHITECTURES:
        for budget in BUDGETS:
            values_by_seed = {
                int(seed): by_identity[
                    (architecture, budget, int(seed))
                ]
                for seed in OFFICIAL_TRAINING_SEEDS
            }

            summary = summarize_seed_runs(values_by_seed)

            require(
                int(summary["n"]) == 10,
                f"CELL_N_NOT_10:{architecture}:{budget}",
            )
            require(
                int(summary["sample_sd_ddof"]) == 1,
                f"CELL_DDOF_NOT_1:{architecture}:{budget}",
            )

            mean = float(summary["mean"])
            sample_sd = float(summary["sample_sd"])
            standard_error = sample_sd / math.sqrt(10)
            ci_low = mean - critical_value * standard_error
            ci_high = mean + critical_value * standard_error

            require(
                all(
                    math.isfinite(v)
                    for v in (
                        mean,
                        sample_sd,
                        standard_error,
                        ci_low,
                        ci_high,
                    )
                ),
                f"NONFINITE_CELL_SUMMARY:{architecture}:{budget}",
            )

            cell_rows.append(
                {
                    "architecture": architecture,
                    "budget": budget,
                    "metric_name": METRIC,
                    "n_seeds": 10,
                    "mean": repr(mean),
                    "sample_SD_ddof1": repr(sample_sd),
                    "mean_two_sided_95CI_low": repr(ci_low),
                    "mean_two_sided_95CI_high": repr(ci_high),
                    "ci_level": repr(RQ2_CI_LEVEL),
                    "df": RQ2_DF,
                    "uncertainty_role": "DESCRIPTIVE",
                    "formal_hypothesis_test": "false",
                    "p_value_computed": "false",
                    "source_artifact": SOURCE_REL,
                    "statistics_semantics": STATISTICS_SEMANTICS,
                    "representation_authority": REPRESENTATION_AUTHORITY,
                }
            )

    require(len(gain_rows) == 10, f"GAIN_ROWS={len(gain_rows)}")
    require(len(cell_rows) == 8, f"CELL_ROWS={len(cell_rows)}")

    primary_output_obj = {
        "schema_version": "1.0.0",
        "rq": "RQ2",
        "analysis_role": "PRIMARY_HYPOTHESIS",
        "scientific_contract": {
            "primitive_effect": "delta_mAP_test=SSL-SUP",
            "effect": "G_s",
            "g_s_definition": (
                "EQUAL_WEIGHT_MEAN_OF_8_PAIRED_ARCHITECTURE_BUDGET_DELTA_MAP_CELLS"
            ),
            "cell_count_per_seed": RQ2_CELL_COUNT,
            "cell_weight": RQ2_CELL_WEIGHT,
            "null_mean": RQ2_NULL_MEAN,
            "alternative": RQ2_ALTERNATIVE,
            "p_value_sidedness": "one-sided",
            "ci_level": RQ2_CI_LEVEL,
            "ci_sidedness": "two-sided",
            "n": int(primary["n"]),
            "df": int(primary["df"]),
        },
        "result": primary,
        "source_artifact": SOURCE_REL,
        "source_sha256": EXPECTED_PAIRED_EFFECTS_SHA256,
        "frozen_statistics_sources": {
            "src/statistics/rq2_primary.py": EXPECTED_RQ2_PRIMARY_SHA256,
            "src/statistics/seed_summary.py": EXPECTED_SEED_SUMMARY_SHA256,
        },
        "statistics_semantics": STATISTICS_SEMANTICS,
        "representation_authority": REPRESENTATION_AUTHORITY,
    }

    write_csv(gain_output, gain_fields, gain_rows)
    write_json(primary_output, primary_output_obj)
    write_csv(cell_output, cell_fields, cell_rows)

    print("=== S10.1.11 RQ2 ===")
    print(f"SOURCE_ROWS={len(rows)}")
    print(f"GAIN_ROWS={len(gain_rows)}")
    print(f"CELL_DESCRIPTIVE_ROWS={len(cell_rows)}")
    print("ARCHITECTURES=2")
    print("BUDGETS=4")
    print("CELLS_PER_SEED=8")
    print("SEEDS=10")
    print("PRIMITIVE_EFFECT=delta_mAP_test=SSL-SUP")
    print("OVERALL_EFFECT=G_s")
    print("PRIMARY_TEST=ONE_SAMPLE_T")
    print("ALTERNATIVE=greater")
    print("P_VALUE_SIDEDNESS=one-sided")
    print("CI_SIDEDNESS=two-sided")
    print("CI_LEVEL=0.95")
    print("DF=9")
    print("CELL_DESCRIPTIVES=MEAN_SAMPLE_SD_DDOF1_TWO_SIDED_95CI")
    print("CELL_FORMAL_HYPOTHESIS_TEST=FALSE")
    print("STATISTICS_CORE=FROZEN_S6_REUSE")
    print("S10_1_11_STATUS=PASS")


if __name__ == "__main__":
    main()
