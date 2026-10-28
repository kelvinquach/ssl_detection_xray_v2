from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import scipy

from src.statistics.seed_summary import OFFICIAL_TRAINING_SEEDS
from src.statistics.exact_sign_flip import (
    SIGN_CONFIGURATION_COUNT,
    exact_sign_flip_test,
)
from src.statistics.loso_robustness import (
    LOSO_ANALYSIS_COUNT,
    LOSO_RETAINED_SEED_COUNT,
    leave_one_seed_out,
)


EXPECTED_SEED_SUMMARY_SHA256 = (
    "2a404ac0f6097cad16816b550fcd1c9ca240fab5535854dc0fd5e4433e49ff3f"
)
EXPECTED_EXACT_SIGN_FLIP_SHA256 = (
    "56f352002e0e69a308eaffc8456f598fae666f18811171e24949ee5d299fbf20"
)
EXPECTED_LOSO_SHA256 = (
    "4cdddff20648d241c17d1e3d7917b36d692ee68eab9a98ddd9af1763a1431b50"
)
EXPECTED_STATISTICAL_IMPLEMENTATION_MANIFEST_SHA256 = (
    "eb2f5eb19164030d4e1435a181996315c6009aef61058ddfed10c13431f9f30e"
)
EXPECTED_RQ7_SEED_SHA256 = (
    "30230a4c0a28ac3027deb0bc9fcdab2a4d9d36c1ccf7aae0f28faba752fd67f6"
)

EXPECTED_PYTHON = "3.10.13"
EXPECTED_NUMPY = "1.26.4"
EXPECTED_SCIPY = "1.15.3"

SOURCE_REL = "analysis/rq7/rq7_architecture_gain_by_seed.csv"
STATISTICAL_IMPLEMENTATION_MANIFEST_REL = (
    "artifacts/preflight/statistics/statistical_implementation_manifest.json"
)

STATISTICS_SEMANTICS = "FROZEN_S6_REUSE"
REPRESENTATION_AUTHORITY = "S10_CANONICAL_MACHINE_READABLE"


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
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        f.write("\n")


def finite_float(value, label):
    try:
        result = float(value)
    except Exception as exc:
        raise RuntimeError(f"NON_NUMERIC:{label}:{value!r}") from exc
    require(math.isfinite(result), f"NONFINITE:{label}:{value!r}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--exact-output", required=True)
    parser.add_argument("--loso-output", required=True)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    source = Path(args.source)
    exact_output = Path(args.exact_output)
    loso_output = Path(args.loso_output)

    generation_script = Path(__file__).resolve()
    generation_script_sha256 = sha256_file(generation_script)

    # ------------------------------------------------------------
    # Frozen source identities
    # ------------------------------------------------------------
    require(
        sha256_file(repo / "src/statistics/seed_summary.py")
        == EXPECTED_SEED_SUMMARY_SHA256,
        "SEED_SUMMARY_SHA256_MISMATCH",
    )
    require(
        sha256_file(repo / "src/statistics/exact_sign_flip.py")
        == EXPECTED_EXACT_SIGN_FLIP_SHA256,
        "EXACT_SIGN_FLIP_SHA256_MISMATCH",
    )
    require(
        sha256_file(repo / "src/statistics/loso_robustness.py")
        == EXPECTED_LOSO_SHA256,
        "LOSO_SHA256_MISMATCH",
    )

    statistical_manifest = repo / STATISTICAL_IMPLEMENTATION_MANIFEST_REL
    require(
        sha256_file(statistical_manifest)
        == EXPECTED_STATISTICAL_IMPLEMENTATION_MANIFEST_SHA256,
        "STATISTICAL_IMPLEMENTATION_MANIFEST_SHA256_MISMATCH",
    )

    require(
        sha256_file(source) == EXPECTED_RQ7_SEED_SHA256,
        "RQ7_SEED_SHA256_MISMATCH",
    )

    # ------------------------------------------------------------
    # Canonical runtime
    # ------------------------------------------------------------
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

    require(
        SIGN_CONFIGURATION_COUNT == 1024,
        "SIGN_CONFIGURATION_COUNT_NOT_1024",
    )
    require(
        LOSO_ANALYSIS_COUNT == 10,
        "LOSO_ANALYSIS_COUNT_NOT_10",
    )
    require(
        LOSO_RETAINED_SEED_COUNT == 9,
        "LOSO_RETAINED_SEED_COUNT_NOT_9",
    )

    # ------------------------------------------------------------
    # Canonical RQ7 seed-level input
    # ------------------------------------------------------------
    rows = load_csv(source)
    require(len(rows) == 10, f"RQ7_SEED_ROWS={len(rows)}")

    required_fields = {
        "seed_index",
        "training_seed",
        "A_R50_s",
        "A_Swin-T_s",
        "D_s",
    }
    require(
        required_fields.issubset(rows[0].keys()),
        "RQ7_REQUIRED_FIELDS_MISSING",
    )

    observed_seeds = tuple(int(row["training_seed"]) for row in rows)
    require(
        observed_seeds == tuple(OFFICIAL_TRAINING_SEEDS),
        "RQ7_SEED_ORDER_MISMATCH",
    )

    seed_effects = {}

    for expected_seed_index, row in enumerate(rows, start=1):
        seed_index = int(row["seed_index"])
        training_seed = int(row["training_seed"])

        require(
            seed_index == expected_seed_index,
            f"RQ7_SEED_INDEX_MISMATCH:{seed_index}:{expected_seed_index}",
        )

        a_r50 = finite_float(row["A_R50_s"], f"A_R50_s:{training_seed}")
        a_swin = finite_float(row["A_Swin-T_s"], f"A_Swin-T_s:{training_seed}")
        d_s = finite_float(row["D_s"], f"D_s:{training_seed}")

        require(
            math.isclose(
                d_s,
                a_swin - a_r50,
                rel_tol=1e-15,
                abs_tol=1e-15,
            ),
            f"RQ7_D_S_IDENTITY_MISMATCH:{training_seed}",
        )

        seed_effects[training_seed] = d_s

    require(
        tuple(seed_effects.keys()) == tuple(OFFICIAL_TRAINING_SEEDS),
        "RQ7_SEED_EFFECT_ORDER_MISMATCH",
    )

    # ------------------------------------------------------------
    # Frozen S6 robustness reuse — DO NOT reimplement statistics
    # ------------------------------------------------------------
    exact_result = exact_sign_flip_test(seed_effects, rq="RQ7")
    loso_result = leave_one_seed_out(seed_effects, rq="RQ7")

    require(exact_result["rq"] == "RQ7", "EXACT_RQ_MISMATCH")
    require(exact_result["effect"] == "D_s", "EXACT_EFFECT_MISMATCH")
    require(
        exact_result["alternative"] == "two-sided",
        "EXACT_ALTERNATIVE_MISMATCH",
    )
    require(
        exact_result["sign_configuration_count"] == 1024,
        "EXACT_SIGN_CONFIGURATION_COUNT_MISMATCH",
    )
    require(
        exact_result["enumerated_configuration_count"] == 1024,
        "EXACT_ENUMERATED_CONFIGURATION_COUNT_MISMATCH",
    )
    require(
        exact_result["extremeness_rule"] == "abs(T_flip) >= abs(T_obs)",
        "EXACT_EXTREMENESS_RULE_MISMATCH",
    )
    require(
        exact_result["equality_ties_included"] is True,
        "EXACT_EQUALITY_TIES_NOT_INCLUDED",
    )
    require(
        exact_result["plus_one_correction"] is False,
        "EXACT_PLUS_ONE_CORRECTION_TRUE",
    )
    require(
        exact_result["zero_effect_seed_retained"] is True,
        "EXACT_ZERO_EFFECT_SEED_NOT_RETAINED",
    )
    require(
        exact_result["duplicate_statistics_counted_by_configuration"] is True,
        "EXACT_DUPLICATES_NOT_COUNTED_BY_CONFIGURATION",
    )

    require(loso_result["rq"] == "RQ7", "LOSO_RQ_MISMATCH")
    require(loso_result["effect"] == "D_s", "LOSO_EFFECT_MISMATCH")
    require(
        int(loso_result["loso_analysis_count"]) == 10,
        "LOSO_RESULT_COUNT_MISMATCH",
    )
    require(
        int(loso_result["retained_seed_count_per_analysis"]) == 9,
        "LOSO_RESULT_RETAINED_COUNT_MISMATCH",
    )
    require(
        loso_result["primary_seed_exclusion_allowed"] is False,
        "LOSO_PRIMARY_SEED_EXCLUSION_ALLOWED",
    )

    # ------------------------------------------------------------
    # Provenance/runtime reference
    # ------------------------------------------------------------
    runtime_reference = {
        "manifest_path": STATISTICAL_IMPLEMENTATION_MANIFEST_REL,
        "manifest_sha256": EXPECTED_STATISTICAL_IMPLEMENTATION_MANIFEST_SHA256,
        "claim_scope": (
            "REFERENCE_FOR_CURRENT_CANONICAL_PYTHON_IMPLEMENTATION_"
            "NOT_RETROACTIVE_EXECUTION_CLAIM"
        ),
        "python": EXPECTED_PYTHON,
        "numpy": EXPECTED_NUMPY,
        "scipy": EXPECTED_SCIPY,
        "s6_11_historical_python_version_recorded": False,
    }

    exact_output_obj = {
        "schema_version": "1.0.0",
        "rq": "RQ7",
        "analysis_role": "ROBUSTNESS",
        "scientific_contract": {
            "input_effect": "D_s",
            "effect_definition": "A_Swin-T_minus_A_R50",
            "replication_unit": "TRAINING_SEED_TRAINED_RUN",
            "seed_count": 10,
            "method": "EXACT_SIGN_FLIP",
            "alternative": "two-sided",
            "sign_configuration_count": 1024,
            "extremeness_rule": "abs(T_flip) >= abs(T_obs)",
            "equality_ties_included": True,
            "plus_one_correction": False,
            "zero_effect_seed_retained": True,
        },
        "result": exact_result,
        "source_artifact": SOURCE_REL,
        "source_sha256": EXPECTED_RQ7_SEED_SHA256,
        "generation_script": generation_script.relative_to(repo).as_posix(),
        "generation_script_sha256": generation_script_sha256,
        "frozen_statistics_sources": {
            "src/statistics/seed_summary.py": EXPECTED_SEED_SUMMARY_SHA256,
            "src/statistics/exact_sign_flip.py": EXPECTED_EXACT_SIGN_FLIP_SHA256,
        },
        "statistical_environment_reference": runtime_reference,
        "statistics_semantics": STATISTICS_SEMANTICS,
        "representation_authority": REPRESENTATION_AUTHORITY,
    }

    loso_fields = [
        "schema_version",
        "rq",
        "analysis_role",
        "effect",
        "analysis_index",
        "omitted_seed",
        "retained_seed_count",
        "retained_training_seeds",
        "full_effect",
        "recomputed_effect",
        "primary_seed_exclusion_allowed",
        "source_artifact",
        "source_sha256",
        "generation_script",
        "generation_script_sha256",
        "frozen_loso_source_sha256",
        "statistical_environment_reference",
        "statistical_environment_sha256",
        "statistics_semantics",
        "representation_authority",
    ]

    generation_script_rel = generation_script.relative_to(repo).as_posix()
    loso_rows = []

    for analysis in loso_result["analyses"]:
        loso_rows.append(
            {
                "schema_version": "1.0.0",
                "rq": "RQ7",
                "analysis_role": "ROBUSTNESS",
                "effect": "D_s",
                "analysis_index": int(analysis["analysis_index"]),
                "omitted_seed": int(analysis["omitted_seed"]),
                "retained_seed_count": int(analysis["retained_seed_count"]),
                "retained_training_seeds": json.dumps(
                    analysis["retained_training_seeds"],
                    separators=(",", ":"),
                ),
                "full_effect": repr(float(loso_result["full_effect"])),
                "recomputed_effect": repr(
                    float(analysis["recomputed_effect"])
                ),
                "primary_seed_exclusion_allowed": "false",
                "source_artifact": SOURCE_REL,
                "source_sha256": EXPECTED_RQ7_SEED_SHA256,
                "generation_script": generation_script_rel,
                "generation_script_sha256": generation_script_sha256,
                "frozen_loso_source_sha256": EXPECTED_LOSO_SHA256,
                "statistical_environment_reference": (
                    STATISTICAL_IMPLEMENTATION_MANIFEST_REL
                ),
                "statistical_environment_sha256": (
                    EXPECTED_STATISTICAL_IMPLEMENTATION_MANIFEST_SHA256
                ),
                "statistics_semantics": STATISTICS_SEMANTICS,
                "representation_authority": REPRESENTATION_AUTHORITY,
            }
        )

    require(len(loso_rows) == 10, f"LOSO_ROWS={len(loso_rows)}")

    write_json(exact_output, exact_output_obj)
    write_csv(loso_output, loso_fields, loso_rows)

    print("=== S10.1.16 RQ7 ROBUSTNESS ===")
    print("SOURCE=" + SOURCE_REL)
    print("SOURCE_ROWS=10")
    print("EFFECT=D_s")
    print("EXACT_SIGN_FLIP=FROZEN_S6_REUSE")
    print("EXACT_SIGN_CONFIGURATIONS=1024")
    print("EXACT_ALTERNATIVE=two-sided")
    print("EXACT_EXTREMENESS_RULE=abs(T_flip) >= abs(T_obs)")
    print("EXACT_EQUALITY_TIES_INCLUDED=TRUE")
    print("EXACT_PLUS_ONE_CORRECTION=FALSE")
    print("LOSO=FROZEN_S6_REUSE")
    print("LOSO_ANALYSES=10")
    print("LOSO_RETAINED_SEEDS_PER_ANALYSIS=9")
    print("LOSO_P_VALUE_COMPUTED=FALSE")
    print("LOSO_CI_COMPUTED=FALSE")
    print("STATISTICS_REIMPLEMENTED=FALSE")
    print("EXACT_OUTPUT_SHA256=" + sha256_file(exact_output))
    print("LOSO_OUTPUT_SHA256=" + sha256_file(loso_output))


if __name__ == "__main__":
    main()
