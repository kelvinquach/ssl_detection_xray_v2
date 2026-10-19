from __future__ import annotations

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
from src.statistics.rq6_negative_fp import (
    RQ6_ALPHA,
    RQ6_ARCHITECTURES,
    RQ6_BUDGETS,
    RQ6_CELL_COUNT,
    RQ6_CELL_WEIGHT,
    RQ6_CI_LEVEL,
    RQ6_DF,
    RQ6_EFFECT,
    RQ6_FAR_FORMAL_P_VALUE_REQUIRED,
    RQ6_FAR_SECONDARY_OUTCOME,
    compute_rq6_seed_effects,
    rq6_primary_test,
)

EXPECTED_SEED_SUMMARY_SHA256 = (
    "2a404ac0f6097cad16816b550fcd1c9ca240fab5535854dc0fd5e4433e49ff3f"
)
EXPECTED_RQ6_CORE_SHA256 = (
    "69579793b2927c7a3ed39f7f94d3a2a598b35a0eacbe2b591cce48f69c2d0907"
)
EXPECTED_PAIRED_EFFECTS_SHA256 = (
    "bad5a179711443636a13bd6501a1d90a7e8c5dadfb16b9c09df74bf502b727bb"
)
EXPECTED_FINAL_RESULTS_SHA256 = (
    "07666def77ee911878e8ba3c99b7d34353991b5b7fdaed64db495b9cd858c9f5"
)
EXPECTED_CONTROLLED_REVISION_SHA256 = (
    "827b3ca0c55ecc9dcd8704d6caea7c6d087e4630c319d3d1f92897f88c83f156"
)

EXPECTED_PYTHON = "3.10.13"
EXPECTED_NUMPY = "1.26.4"
EXPECTED_SCIPY = "1.15.3"

PAIRED_SOURCE_REL = "artifacts/results/paired_effects.csv"
FINAL_SOURCE_REL = "artifacts/final_test/official_final_test_results.csv"
CONTROLLED_REVISION_REL = (
    "artifacts/governance/controlled_revisions/"
    "RQ6_FAR_CONTROLLED_CLARIFICATION_20261019.md"
)

ARCH_MAP = {
    "R50-FPN": "R50",
    "Swin-T-FPN": "Swin-T",
}
BUDGET_MAP = {
    "1pct": "1%",
    "5pct": "5%",
    "10pct": "10%",
    "20pct": "20%",
}

STATISTICS_SEMANTICS = "FROZEN_S6_REUSE_PLUS_RESEARCHER_APPROVED_RQ6_FAR_CLARIFICATION"
REPRESENTATION_AUTHORITY = "S10_CANONICAL_MACHINE_READABLE"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(
    path: Path,
    fields: list[str],
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        f.write("\n")


def finite_float(value: object, field: str) -> float:
    try:
        parsed = float(value)
    except Exception as exc:
        raise RuntimeError(f"NON_NUMERIC:{field}") from exc
    require(math.isfinite(parsed), f"NONFINITE:{field}")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--seed-output", required=True)
    parser.add_argument("--primary-output", required=True)
    parser.add_argument("--far-output", required=True)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    paired_source = repo / PAIRED_SOURCE_REL
    final_source = repo / FINAL_SOURCE_REL
    controlled_revision = repo / CONTROLLED_REVISION_REL

    seed_output = Path(args.seed_output)
    primary_output = Path(args.primary_output)
    far_output = Path(args.far_output)

    generation_script = Path(__file__).resolve()
    generation_script_sha256 = sha256_file(generation_script)

    # ------------------------------------------------------------------
    # Dependency/source identity locks
    # ------------------------------------------------------------------
    require(
        sha256_file(repo / "src/statistics/seed_summary.py")
        == EXPECTED_SEED_SUMMARY_SHA256,
        "SEED_SUMMARY_SHA256_MISMATCH",
    )
    require(
        sha256_file(repo / "src/statistics/rq6_negative_fp.py")
        == EXPECTED_RQ6_CORE_SHA256,
        "RQ6_CORE_SHA256_MISMATCH",
    )
    require(
        sha256_file(paired_source) == EXPECTED_PAIRED_EFFECTS_SHA256,
        "PAIRED_EFFECTS_SHA256_MISMATCH",
    )
    require(
        sha256_file(final_source) == EXPECTED_FINAL_RESULTS_SHA256,
        "OFFICIAL_FINAL_TEST_RESULTS_SHA256_MISMATCH",
    )
    require(
        sha256_file(controlled_revision)
        == EXPECTED_CONTROLLED_REVISION_SHA256,
        "RQ6_FAR_CONTROLLED_REVISION_SHA256_MISMATCH",
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

    require(RQ6_CELL_COUNT == 8, "RQ6_CELL_COUNT_NOT_8")
    require(math.isclose(RQ6_CELL_WEIGHT, 0.125), "RQ6_CELL_WEIGHT_NOT_0_125")
    require(RQ6_DF == 9, "RQ6_DF_NOT_9")
    require(RQ6_CI_LEVEL == 0.95, "RQ6_CI_LEVEL_NOT_0_95")
    require(RQ6_FAR_SECONDARY_OUTCOME is True, "RQ6_FAR_NOT_SECONDARY")
    require(
        RQ6_FAR_FORMAL_P_VALUE_REQUIRED is False,
        "RQ6_FAR_P_VALUE_POLICY_MISMATCH",
    )

    # ------------------------------------------------------------------
    # RQ6 primary: reuse frozen S6 core on paired_effects.csv
    # ------------------------------------------------------------------
    paired_raw = load_csv(paired_source)
    require(len(paired_raw) == 80, f"PAIRED_EFFECTS_ROWS={len(paired_raw)}")

    primary_rows: list[dict[str, object]] = []
    for i, row in enumerate(paired_raw, start=1):
        require(
            row["architecture"] in ARCH_MAP,
            f"ROW_{i}_UNEXPECTED_ARCHITECTURE:{row['architecture']}",
        )
        require(
            row["budget"] in BUDGET_MAP,
            f"ROW_{i}_UNEXPECTED_BUDGET:{row['budget']}",
        )

        primary_rows.append(
            {
                "architecture": ARCH_MAP[row["architecture"]],
                "budget": BUDGET_MAP[row["budget"]],
                "seed_index": int(row["seed_index"]),
                "training_seed": int(row["training_seed"]),
                "delta_FP_per_negative_test": finite_float(
                    row["delta_FP_per_negative_test"],
                    f"row_{i}.delta_FP_per_negative_test",
                ),
            }
        )

    # DO NOT reimplement H_s.
    seed_effects = compute_rq6_seed_effects(primary_rows)

    require(
        tuple(seed_effects.keys()) == tuple(OFFICIAL_TRAINING_SEEDS),
        "RQ6_SEED_EFFECT_ORDER_MISMATCH",
    )

    # DO NOT reimplement primary t-test.
    primary_result = rq6_primary_test(seed_effects)

    require(primary_result["effect"] == "H_s", "RQ6_PRIMARY_EFFECT_MISMATCH")
    require(int(primary_result["n"]) == 10, "RQ6_PRIMARY_N_NOT_10")
    require(int(primary_result["df"]) == 9, "RQ6_PRIMARY_DF_NOT_9")
    require(
        primary_result["alternative"] == "two-sided",
        "RQ6_PRIMARY_ALTERNATIVE_MISMATCH",
    )

    seed_fields = [
        "seed_index",
        "training_seed",
        "effect",
        "cell_count",
        "cell_weight",
        "H_s",
        "source_artifact",
        "source_sha256",
        "statistics_semantics",
        "representation_authority",
    ]

    seed_rows: list[dict[str, object]] = []
    for seed_index, seed in enumerate(OFFICIAL_TRAINING_SEEDS, start=1):
        seed_rows.append(
            {
                "seed_index": seed_index,
                "training_seed": int(seed),
                "effect": RQ6_EFFECT,
                "cell_count": RQ6_CELL_COUNT,
                "cell_weight": repr(RQ6_CELL_WEIGHT),
                "H_s": repr(float(seed_effects[int(seed)])),
                "source_artifact": PAIRED_SOURCE_REL,
                "source_sha256": EXPECTED_PAIRED_EFFECTS_SHA256,
                "statistics_semantics": STATISTICS_SEMANTICS,
                "representation_authority": REPRESENTATION_AUTHORITY,
            }
        )

    require(len(seed_rows) == 10, f"RQ6_SEED_ROWS={len(seed_rows)}")

    primary_output_obj = {
        "schema_version": "1.0.0",
        "rq": "RQ6",
        "analysis_role": "PRIMARY_HYPOTHESIS",
        "scientific_contract": {
            "primitive_effect": "delta_FP_per_negative_test=SSL-SUP",
            "effect": "H_s",
            "h_s_definition": (
                "EQUAL_WEIGHT_MEAN_OF_8_PAIRED_ARCHITECTURE_BUDGET_"
                "DELTA_FP_PER_NEGATIVE_CELLS"
            ),
            "cell_count_per_seed": RQ6_CELL_COUNT,
            "cell_weight": RQ6_CELL_WEIGHT,
            "replication_unit": "TRAINING_SEED_TRAINED_RUN",
            "n": 10,
            "df": 9,
            "alternative": "two-sided",
            "p_value_sidedness": "two-sided",
            "ci_level": RQ6_CI_LEVEL,
            "ci_sidedness": "two-sided",
            "far_secondary_outcome": True,
            "far_formal_p_value_required": False,
        },
        "result": primary_result,
        "source_artifact": PAIRED_SOURCE_REL,
        "source_sha256": EXPECTED_PAIRED_EFFECTS_SHA256,
        "generation_script": generation_script.relative_to(repo).as_posix(),
        "generation_script_sha256": generation_script_sha256,
        "frozen_statistics_sources": {
            "src/statistics/seed_summary.py": EXPECTED_SEED_SUMMARY_SHA256,
            "src/statistics/rq6_negative_fp.py": EXPECTED_RQ6_CORE_SHA256,
        },
        "statistics_semantics": STATISTICS_SEMANTICS,
        "representation_authority": REPRESENTATION_AUTHORITY,
    }

    # ------------------------------------------------------------------
    # FAR secondary outcome:
    # researcher-approved controlled clarification.
    # ------------------------------------------------------------------
    final_rows = load_csv(final_source)
    require(len(final_rows) == 180, f"FINAL_RESULTS_ROWS={len(final_rows)}")

    far_by_method: dict[str, dict[tuple[str, str, int, int], float]] = {
        "SUP": {},
        "SSL": {},
    }

    for i, row in enumerate(final_rows, start=1):
        if row["condition_role"] != "MAIN_LOW_LABEL":
            continue

        method = row["method"]
        if method not in ("SUP", "SSL"):
            continue

        require(
            row["architecture"] in ARCH_MAP,
            f"FINAL_ROW_{i}_UNEXPECTED_ARCHITECTURE:{row['architecture']}",
        )
        require(
            row["budget"] in BUDGET_MAP,
            f"FINAL_ROW_{i}_UNEXPECTED_BUDGET:{row['budget']}",
        )

        architecture = ARCH_MAP[row["architecture"]]
        budget = BUDGET_MAP[row["budget"]]
        seed_index = int(row["seed_index"])
        training_seed = int(row["training_seed"])
        far = finite_float(row["negative_FAR"], f"final_row_{i}.negative_FAR")

        require(0.0 <= far <= 1.0, f"FINAL_ROW_{i}_FAR_OUT_OF_RANGE")

        key = (architecture, budget, seed_index, training_seed)
        require(
            key not in far_by_method[method],
            f"DUPLICATE_FAR_KEY:{method}:{key}",
        )
        far_by_method[method][key] = far

    require(len(far_by_method["SUP"]) == 80, f"SUP_FAR_ROWS={len(far_by_method['SUP'])}")
    require(len(far_by_method["SSL"]) == 80, f"SSL_FAR_ROWS={len(far_by_method['SSL'])}")
    require(
        set(far_by_method["SUP"]) == set(far_by_method["SSL"]),
        "SUP_SSL_FAR_KEY_SET_MISMATCH",
    )

    expected_cells = {
        (architecture, budget)
        for architecture in RQ6_ARCHITECTURES
        for budget in RQ6_BUDGETS
    }

    far_seed_raw: dict[int, float] = {}

    for expected_seed_index, seed in enumerate(OFFICIAL_TRAINING_SEEDS, start=1):
        seed = int(seed)

        cell_deltas: list[float] = []
        observed_cells: set[tuple[str, str]] = set()

        for architecture in RQ6_ARCHITECTURES:
            for budget in RQ6_BUDGETS:
                key = (architecture, budget, expected_seed_index, seed)

                require(key in far_by_method["SUP"], f"MISSING_SUP_FAR_KEY:{key}")
                require(key in far_by_method["SSL"], f"MISSING_SSL_FAR_KEY:{key}")

                observed_cells.add((architecture, budget))
                cell_deltas.append(
                    far_by_method["SSL"][key] - far_by_method["SUP"][key]
                )

        require(
            observed_cells == expected_cells,
            f"FAR_SEED_CELL_SET_MISMATCH:{seed}",
        )
        require(len(cell_deltas) == 8, f"FAR_CELL_COUNT_NOT_8:{seed}")

        # Researcher-approved K_s clarification.
        far_seed_raw[seed] = float(np.mean(np.asarray(cell_deltas, dtype=float)))

    require(
        tuple(far_seed_raw.keys()) == tuple(OFFICIAL_TRAINING_SEEDS),
        "FAR_SEED_ORDER_MISMATCH",
    )

    # Summary is based on the 10 locked seed-level K_s values.
    far_summary = summarize_seed_runs(far_seed_raw)
    require(int(far_summary["n"]) == 10, "FAR_SUMMARY_N_NOT_10")
    require(int(far_summary["sample_sd_ddof"]) == 1, "FAR_SUMMARY_DDOF_NOT_1")

    far_mean_raw = float(far_summary["mean"])
    far_sd_raw = float(far_summary["sample_sd"])
    far_se_raw = far_sd_raw / math.sqrt(10)

    critical_value = float(
        stats.t.ppf(
            1.0 - RQ6_ALPHA / 2.0,
            df=RQ6_DF,
        )
    )

    far_ci_low_raw = far_mean_raw - critical_value * far_se_raw
    far_ci_high_raw = far_mean_raw + critical_value * far_se_raw

    require(
        all(
            math.isfinite(v)
            for v in (
                far_mean_raw,
                far_sd_raw,
                far_se_raw,
                far_ci_low_raw,
                far_ci_high_raw,
            )
        ),
        "NONFINITE_FAR_SUMMARY",
    )

    # Thesis/reporting unit = percentage points.
    far_seed_pp = {
        int(seed): 100.0 * float(value)
        for seed, value in far_seed_raw.items()
    }

    far_fields = [
        "schema_version",
        "rq",
        "analysis_role",
        "effect",
        "primitive_effect",
        "metric_name",
        "replication_unit",
        "n_seeds",
        "cell_count_per_seed",
        "cell_weight",
        "ordered_training_seeds",
        "seed_effects_K_s_raw_proportion",
        "seed_effects_K_s_percentage_points",
        "effect_estimate_percentage_points",
        "sample_SD_ddof1_percentage_points",
        "mean_two_sided_95CI_low_percentage_points",
        "mean_two_sided_95CI_high_percentage_points",
        "ci_level",
        "ci_sidedness",
        "formal_hypothesis_test",
        "p_value_computed",
        "multiplicity_family",
        "reporting_unit",
        "source_artifact",
        "source_sha256",
        "controlled_revision_artifact",
        "controlled_revision_sha256",
        "generation_script",
        "generation_script_sha256",
        "statistics_semantics",
        "representation_authority",
    ]

    far_row = {
        "schema_version": "1.0.0",
        "rq": "RQ6",
        "analysis_role": "SECONDARY_ESTIMATION",
        "effect": "K_s",
        "primitive_effect": "negative_FAR_SSL_MINUS_SUP",
        "metric_name": "negative_FAR",
        "replication_unit": "TRAINING_SEED_TRAINED_RUN",
        "n_seeds": 10,
        "cell_count_per_seed": 8,
        "cell_weight": repr(1.0 / 8.0),
        "ordered_training_seeds": json.dumps(
            [int(s) for s in OFFICIAL_TRAINING_SEEDS],
            separators=(",", ":"),
        ),
        "seed_effects_K_s_raw_proportion": json.dumps(
            [far_seed_raw[int(s)] for s in OFFICIAL_TRAINING_SEEDS],
            separators=(",", ":"),
        ),
        "seed_effects_K_s_percentage_points": json.dumps(
            [far_seed_pp[int(s)] for s in OFFICIAL_TRAINING_SEEDS],
            separators=(",", ":"),
        ),
        "effect_estimate_percentage_points": repr(100.0 * far_mean_raw),
        "sample_SD_ddof1_percentage_points": repr(100.0 * far_sd_raw),
        "mean_two_sided_95CI_low_percentage_points": repr(
            100.0 * far_ci_low_raw
        ),
        "mean_two_sided_95CI_high_percentage_points": repr(
            100.0 * far_ci_high_raw
        ),
        "ci_level": repr(RQ6_CI_LEVEL),
        "ci_sidedness": "two-sided",
        "formal_hypothesis_test": "false",
        "p_value_computed": "false",
        "multiplicity_family": "NONE",
        "reporting_unit": "PERCENTAGE_POINTS",
        "source_artifact": FINAL_SOURCE_REL,
        "source_sha256": EXPECTED_FINAL_RESULTS_SHA256,
        "controlled_revision_artifact": CONTROLLED_REVISION_REL,
        "controlled_revision_sha256": EXPECTED_CONTROLLED_REVISION_SHA256,
        "generation_script": generation_script.relative_to(repo).as_posix(),
        "generation_script_sha256": generation_script_sha256,
        "statistics_semantics": STATISTICS_SEMANTICS,
        "representation_authority": REPRESENTATION_AUTHORITY,
    }

    write_csv(seed_output, seed_fields, seed_rows)
    write_json(primary_output, primary_output_obj)
    write_csv(far_output, far_fields, [far_row])

    print("=== S10.1.15 RQ6 MATERIALIZER ===")
    print(f"PAIRED_SOURCE_SHA256={sha256_file(paired_source)}")
    print(f"FINAL_SOURCE_SHA256={sha256_file(final_source)}")
    print(f"CONTROLLED_REVISION_SHA256={sha256_file(controlled_revision)}")
    print("PRIMARY_SOURCE_ROWS=80")
    print("PRIMARY_SEED_ROWS=10")
    print("PRIMARY_EFFECT=H_s")
    print("PRIMARY_TEST=ONE_SAMPLE_T")
    print("PRIMARY_ALTERNATIVE=two-sided")
    print("PRIMARY_N=10")
    print("PRIMARY_DF=9")
    print("FAR_SUP_ROWS=80")
    print("FAR_SSL_ROWS=80")
    print("FAR_MATCHED_PAIRS=80")
    print("FAR_EFFECT=K_s")
    print("FAR_CELLS_PER_SEED=8")
    print("FAR_SEEDS=10")
    print("FAR_FORMAL_HYPOTHESIS_TEST=FALSE")
    print("FAR_P_VALUE_COMPUTED=FALSE")
    print("FAR_CI=INDIVIDUAL_TWO_SIDED_95_PERCENT")
    print("FAR_REPORTING_UNIT=PERCENTAGE_POINTS")
    print("STATISTICS_CORE=FROZEN_S6_REUSE")
    print(f"SEED_OUTPUT_SHA256={sha256_file(seed_output)}")
    print(f"PRIMARY_OUTPUT_SHA256={sha256_file(primary_output)}")
    print(f"FAR_OUTPUT_SHA256={sha256_file(far_output)}")


if __name__ == "__main__":
    main()
