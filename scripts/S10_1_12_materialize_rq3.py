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

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.statistics.seed_summary import OFFICIAL_TRAINING_SEEDS
from src.statistics.rq3_primary import (
    RQ3_ARCHITECTURE_COUNT,
    RQ3_ARCHITECTURE_WEIGHT,
    RQ3_BUDGET_COUNT,
    RQ3_BUDGET_LEVELS,
    RQ3_GREENHOUSE_GEISSER_POLICY,
    RQ3_MAUCHLY_ROLE,
    RQ3_UNCORRECTED_DF_DENOMINATOR,
    RQ3_UNCORRECTED_DF_NUMERATOR,
    compute_rq3_budget_gain_by_seed,
    rq3_primary_anova,
)
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
    rq3_pairwise_tests,
)
from src.statistics.rq3_friedman import (
    FRIEDMAN_DF,
    FRIEDMAN_MULTIPLICITY_FAMILY_ADDED,
    FRIEDMAN_POSTHOC,
    FRIEDMAN_PRIMARY_METHOD,
    FRIEDMAN_PRIMARY_METHOD_UNCHANGED,
    FRIEDMAN_ROLE,
    rq3_friedman_sensitivity,
)

EXPECTED_SEED_SUMMARY_SHA256 = (
    "2a404ac0f6097cad16816b550fcd1c9ca240fab5535854dc0fd5e4433e49ff3f"
)
EXPECTED_RQ3_PRIMARY_SHA256 = (
    "41ac57f7c4f9c19daef8f4f062001bcff1ff41b0af7b4683eb3459e436bf3676"
)
EXPECTED_RQ3_PAIRWISE_SHA256 = (
    "18c0c84fd396193e6019c9a55f051e6ab68ea9cb08e132017d92ddd017be8381"
)
EXPECTED_RQ3_FRIEDMAN_SHA256 = (
    "499630d1983030addb51b529519a92cef34bb3840d22e0784e7f7319fff2fc1f"
)
EXPECTED_PAIRED_EFFECTS_SHA256 = (
    "bad5a179711443636a13bd6501a1d90a7e8c5dadfb16b9c09df74bf502b727bb"
)

EXPECTED_PYTHON = "3.10.13"
EXPECTED_NUMPY = "1.26.4"
EXPECTED_SCIPY = "1.15.3"

ARCHITECTURES = ["R50-FPN", "Swin-T-FPN"]
SOURCE_BUDGETS = ["1pct", "5pct", "10pct", "20pct"]
FROZEN_BUDGETS = ["1%", "5%", "10%", "20%"]

SOURCE_REL = "artifacts/results/paired_effects.csv"
STATISTICS_SEMANTICS = "FROZEN_S6_REUSE"
REPRESENTATION_AUTHORITY = (
    "S10_IMPLEMENTATION_SCHEMA_MINIMAL_FROM_RQ3_CONTRACT"
)
MULTIPLICITY_FAMILY = "F3"


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
    ap.add_argument("--pairwise-output", required=True)
    ap.add_argument("--friedman-output", required=True)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    source = Path(args.source)
    gain_output = Path(args.gain_output)
    primary_output = Path(args.primary_output)
    pairwise_output = Path(args.pairwise_output)
    friedman_output = Path(args.friedman_output)

    # Frozen source identities.
    require(
        sha256_file(repo / "src/statistics/seed_summary.py")
        == EXPECTED_SEED_SUMMARY_SHA256,
        "SEED_SUMMARY_SHA256_MISMATCH",
    )
    require(
        sha256_file(repo / "src/statistics/rq3_primary.py")
        == EXPECTED_RQ3_PRIMARY_SHA256,
        "RQ3_PRIMARY_SHA256_MISMATCH",
    )
    require(
        sha256_file(repo / "src/statistics/rq3_pairwise.py")
        == EXPECTED_RQ3_PAIRWISE_SHA256,
        "RQ3_PAIRWISE_SHA256_MISMATCH",
    )
    require(
        sha256_file(repo / "src/statistics/rq3_friedman.py")
        == EXPECTED_RQ3_FRIEDMAN_SHA256,
        "RQ3_FRIEDMAN_SHA256_MISMATCH",
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

    # Frozen RQ3 contract.
    require(
        tuple(RQ3_BUDGET_LEVELS) == tuple(FROZEN_BUDGETS),
        f"RQ3_BUDGET_LEVELS_MISMATCH:{RQ3_BUDGET_LEVELS}",
    )
    require(RQ3_BUDGET_COUNT == 4, "RQ3_BUDGET_COUNT_NOT_4")
    require(RQ3_ARCHITECTURE_COUNT == 2, "RQ3_ARCHITECTURE_COUNT_NOT_2")
    require(
        math.isclose(RQ3_ARCHITECTURE_WEIGHT, 0.5),
        "RQ3_ARCHITECTURE_WEIGHT_NOT_0_5",
    )
    require(
        RQ3_UNCORRECTED_DF_NUMERATOR == 3,
        "RQ3_UNCORRECTED_DF_NUMERATOR_NOT_3",
    )
    require(
        RQ3_UNCORRECTED_DF_DENOMINATOR == 27,
        "RQ3_UNCORRECTED_DF_DENOMINATOR_NOT_27",
    )
    require(
        RQ3_GREENHOUSE_GEISSER_POLICY == "ALWAYS",
        "RQ3_GG_POLICY_NOT_ALWAYS",
    )
    require(
        RQ3_MAUCHLY_ROLE == "DIAGNOSTIC_ONLY",
        "RQ3_MAUCHLY_ROLE_NOT_DIAGNOSTIC_ONLY",
    )

    expected_contrasts = (
        ("5%-1%", "5%", "1%"),
        ("10%-1%", "10%", "1%"),
        ("20%-1%", "20%", "1%"),
        ("10%-5%", "10%", "5%"),
        ("20%-5%", "20%", "5%"),
        ("20%-10%", "20%", "10%"),
    )
    require(
        tuple(RQ3_PAIRWISE_CONTRASTS) == expected_contrasts,
        "RQ3_PAIRWISE_CONTRASTS_MISMATCH",
    )
    require(RQ3_PAIRWISE_CONTRAST_COUNT == 6, "RQ3_PAIRWISE_COUNT_NOT_6")
    require(RQ3_PAIRWISE_NULL_MEAN == 0.0, "RQ3_PAIRWISE_NULL_NOT_ZERO")
    require(
        RQ3_PAIRWISE_ALTERNATIVE == "two-sided",
        "RQ3_PAIRWISE_ALTERNATIVE_NOT_TWO_SIDED",
    )
    require(RQ3_PAIRWISE_ALPHA == 0.05, "RQ3_PAIRWISE_ALPHA_NOT_0_05")
    require(RQ3_PAIRWISE_CI_LEVEL == 0.95, "RQ3_PAIRWISE_CI_NOT_0_95")
    require(RQ3_PAIRWISE_DF == 9, "RQ3_PAIRWISE_DF_NOT_9")
    require(RQ3_PAIRWISE_HOLM_APPLIED is False, "RQ3_HOLM_ALREADY_APPLIED")
    require(
        RQ3_PAIRWISE_HOLM_DEFERRED_TASK == "S6.10",
        "RQ3_HOLM_DEFERRED_TASK_MISMATCH",
    )

    require(FRIEDMAN_DF == 3, "RQ3_FRIEDMAN_DF_NOT_3")
    require(FRIEDMAN_ROLE == "SENSITIVITY_ONLY", "RQ3_FRIEDMAN_ROLE_MISMATCH")
    require(FRIEDMAN_POSTHOC == "NONE", "RQ3_FRIEDMAN_POSTHOC_NOT_NONE")
    require(
        FRIEDMAN_MULTIPLICITY_FAMILY_ADDED is False,
        "RQ3_FRIEDMAN_ADDED_MULTIPLICITY_FAMILY",
    )
    require(
        FRIEDMAN_PRIMARY_METHOD
        == "GREENHOUSE_GEISSER_REPEATED_MEASURES_ANOVA",
        "RQ3_FRIEDMAN_PRIMARY_METHOD_MISMATCH",
    )
    require(
        FRIEDMAN_PRIMARY_METHOD_UNCHANGED is True,
        "RQ3_FRIEDMAN_PRIMARY_METHOD_CHANGED",
    )

    rows = load_csv(source)
    require(len(rows) == 80, f"SOURCE_ROWS={len(rows)}")
    require(
        len({r["pair_id"] for r in rows}) == 80,
        "PAIR_ID_NOT_UNIQUE",
    )

    expected_cells = {
        (architecture, budget)
        for architecture in ARCHITECTURES
        for budget in SOURCE_BUDGETS
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
            budget in SOURCE_BUDGETS,
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
                f"NON_NUMERIC_RQ3_SOURCE:{r['pair_id']}"
            ) from e

        require(
            all(math.isfinite(v) for v in (sup, ssl, delta)),
            f"NONFINITE_RQ3_SOURCE:{r['pair_id']}",
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

    # Build exact frozen RQ3 aggregation input:
    # ordered seed -> 4 budget rows x 2 architecture delta-mAP values.
    seed_architecture_budget_effects = {}
    for seed in OFFICIAL_TRAINING_SEEDS:
        observed = {
            (architecture, budget)
            for architecture in ARCHITECTURES
            for budget in SOURCE_BUDGETS
            if (architecture, budget, int(seed)) in by_identity
        }
        require(
            observed == expected_cells,
            f"SEED_CELL_SET_MISMATCH:{seed}",
        )

        seed_architecture_budget_effects[int(seed)] = [
            [
                by_identity[(architecture, budget, int(seed))]
                for architecture in ARCHITECTURES
            ]
            for budget in SOURCE_BUDGETS
        ]

    # DO NOT reimplement B_{b,s}.
    gains = compute_rq3_budget_gain_by_seed(
        seed_architecture_budget_effects
    )

    # DO NOT reimplement RQ3 primary, pairwise, or Friedman statistics.
    primary = rq3_primary_anova(gains)
    pairwise = rq3_pairwise_tests(gains)
    friedman = rq3_friedman_sensitivity(gains)

    gain_fields = [
        "seed_index",
        "training_seed",
        "budget",
        "effect",
        "architecture_count",
        "architecture_weight",
        "B_b_s",
        "source_artifact",
        "statistics_semantics",
        "representation_authority",
    ]

    gain_rows = []
    for seed_index, seed in enumerate(
        OFFICIAL_TRAINING_SEEDS,
        start=1,
    ):
        seed_values = gains[int(seed)]
        require(
            len(seed_values) == 4,
            f"GAIN_BUDGET_COUNT_NOT_4:{seed}",
        )
        for budget_index, budget in enumerate(FROZEN_BUDGETS):
            value = float(seed_values[budget_index])
            require(
                math.isfinite(value),
                f"NONFINITE_B_B_S:{seed}:{budget}",
            )
            gain_rows.append(
                {
                    "seed_index": seed_index,
                    "training_seed": int(seed),
                    "budget": budget,
                    "effect": "B_b_s",
                    "architecture_count": RQ3_ARCHITECTURE_COUNT,
                    "architecture_weight": repr(RQ3_ARCHITECTURE_WEIGHT),
                    "B_b_s": repr(value),
                    "source_artifact": SOURCE_REL,
                    "statistics_semantics": STATISTICS_SEMANTICS,
                    "representation_authority": REPRESENTATION_AUTHORITY,
                }
            )

    pairwise_fields = [
        "contrast",
        "budget_2",
        "budget_1",
        "direction",
        "effect",
        "effect_estimate",
        "sample_sd",
        "sample_sd_ddof",
        "n",
        "df",
        "null_mean",
        "alternative",
        "t_statistic",
        "raw_p_value",
        "p_value_sidedness",
        "ci_level",
        "ci_sidedness",
        "ci_low",
        "ci_high",
        "holm_applied",
        "holm_deferred_task",
        "multiplicity_family",
        "source_artifact",
        "statistics_semantics",
        "representation_authority",
    ]

    pairwise_rows = []
    require(
        int(pairwise["contrast_count"]) == 6,
        f"PAIRWISE_CONTRAST_COUNT={pairwise['contrast_count']}",
    )
    require(
        pairwise["holm_applied"] is False,
        "PAIRWISE_HOLM_APPLIED_TRUE",
    )
    require(
        pairwise["holm_deferred_task"] == "S6.10",
        "PAIRWISE_HOLM_DEFERRED_TASK_MISMATCH",
    )

    for result in pairwise["contrasts"]:
        pairwise_rows.append(
            {
                "contrast": result["contrast"],
                "budget_2": result["budget_2"],
                "budget_1": result["budget_1"],
                "direction": result["direction"],
                "effect": result["effect"],
                "effect_estimate": repr(float(result["effect_estimate"])),
                "sample_sd": repr(float(result["sample_sd"])),
                "sample_sd_ddof": int(result["sample_sd_ddof"]),
                "n": int(result["n"]),
                "df": int(result["df"]),
                "null_mean": repr(float(result["null_mean"])),
                "alternative": result["alternative"],
                "t_statistic": repr(float(result["t_statistic"])),
                "raw_p_value": repr(float(result["raw_p_value"])),
                "p_value_sidedness": result["p_value_sidedness"],
                "ci_level": repr(float(result["ci_level"])),
                "ci_sidedness": result["ci_sidedness"],
                "ci_low": repr(float(result["ci_low"])),
                "ci_high": repr(float(result["ci_high"])),
                "holm_applied": "false",
                "holm_deferred_task": pairwise["holm_deferred_task"],
                "multiplicity_family": MULTIPLICITY_FAMILY,
                "source_artifact": SOURCE_REL,
                "statistics_semantics": STATISTICS_SEMANTICS,
                "representation_authority": REPRESENTATION_AUTHORITY,
            }
        )

    require(len(gain_rows) == 40, f"GAIN_ROWS={len(gain_rows)}")
    require(len(pairwise_rows) == 6, f"PAIRWISE_ROWS={len(pairwise_rows)}")

    primary_output_obj = {
        "schema_version": "1.0.0",
        "rq": "RQ3",
        "analysis_role": "SECONDARY_FORMAL_OMNIBUS",
        "scientific_contract": {
            "primitive_effect": "delta_mAP_test=SSL-SUP",
            "effect": "B_b_s",
            "b_b_s_definition": (
                "EQUAL_WEIGHT_MEAN_OF_R50_AND_SWIN_DELTA_MAP_WITHIN_BUDGET_AND_SEED"
            ),
            "architecture_count": RQ3_ARCHITECTURE_COUNT,
            "architecture_weight": RQ3_ARCHITECTURE_WEIGHT,
            "budget_levels": list(RQ3_BUDGET_LEVELS),
            "n_subjects": 10,
            "primary_method": "ONE_WAY_REPEATED_MEASURES_ANOVA",
            "greenhouse_geisser_policy": RQ3_GREENHOUSE_GEISSER_POLICY,
            "mauchly_role": RQ3_MAUCHLY_ROLE,
            "hypothesis_direction": "NON_DIRECTIONAL",
        },
        "result": primary,
        "source_artifact": SOURCE_REL,
        "source_sha256": EXPECTED_PAIRED_EFFECTS_SHA256,
        "frozen_statistics_sources": {
            "src/statistics/rq3_primary.py": EXPECTED_RQ3_PRIMARY_SHA256,
            "src/statistics/seed_summary.py": EXPECTED_SEED_SUMMARY_SHA256,
        },
        "statistics_semantics": STATISTICS_SEMANTICS,
        "representation_authority": REPRESENTATION_AUTHORITY,
    }

    friedman_output_obj = {
        "schema_version": "1.0.0",
        "rq": "RQ3",
        "analysis_role": "SENSITIVITY_ONLY",
        "scientific_contract": {
            "input_effect": "B_b_s",
            "budget_levels": list(RQ3_BUDGET_LEVELS),
            "n_subjects": 10,
            "posthoc": "NONE",
            "multiplicity_family_added": False,
            "primary_method_unchanged": True,
        },
        "result": friedman,
        "source_artifact": SOURCE_REL,
        "source_sha256": EXPECTED_PAIRED_EFFECTS_SHA256,
        "frozen_statistics_sources": {
            "src/statistics/rq3_friedman.py": EXPECTED_RQ3_FRIEDMAN_SHA256,
            "src/statistics/rq3_primary.py": EXPECTED_RQ3_PRIMARY_SHA256,
        },
        "statistics_semantics": STATISTICS_SEMANTICS,
        "representation_authority": REPRESENTATION_AUTHORITY,
    }

    write_csv(gain_output, gain_fields, gain_rows)
    write_json(primary_output, primary_output_obj)
    write_csv(pairwise_output, pairwise_fields, pairwise_rows)
    write_json(friedman_output, friedman_output_obj)

    print("=== S10.1.12 RQ3 ===")
    print(f"SOURCE_ROWS={len(rows)}")
    print(f"BUDGET_GAIN_ROWS={len(gain_rows)}")
    print(f"PAIRWISE_ROWS={len(pairwise_rows)}")
    print("ARCHITECTURES=2")
    print("BUDGETS=4")
    print("SEEDS=10")
    print("PRIMITIVE_EFFECT=delta_mAP_test=SSL-SUP")
    print("BUDGET_EFFECT=B_b_s")
    print("PRIMARY_TEST=ONE_WAY_REPEATED_MEASURES_ANOVA")
    print("GREENHOUSE_GEISSER=ALWAYS")
    print("MAUCHLY=DIAGNOSTIC_ONLY")
    print("PAIRWISE_CONTRASTS=6")
    print("PAIRWISE_TEST=TWO_SIDED_ONE_SAMPLE_T_ON_PAIRED_DIFFERENCES")
    print("PAIRWISE_HOLM_APPLIED=FALSE")
    print("PAIRWISE_MULTIPLICITY_FAMILY=F3")
    print("FRIEDMAN_ROLE=SENSITIVITY_ONLY")
    print("FRIEDMAN_POSTHOC=NONE")
    print("STATISTICS_CORE=FROZEN_S6_REUSE")
    print("S10_1_12_STATUS=PASS")


if __name__ == "__main__":
    main()
