from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import scipy

from src.statistics.seed_summary import (
    OFFICIAL_TRAINING_SEEDS,
    REPLICATION_UNIT,
)
from src.statistics.rq7_architecture_effect import (
    RQ7_ALTERNATIVE,
    RQ7_ARCHITECTURES,
    RQ7_BUDGETS,
    RQ7_CI_LEVEL,
    RQ7_DF,
    RQ7_EFFECT,
    RQ7_TOTAL_ROWS,
    compute_rq7_architecture_gain_by_seed,
    compute_rq7_difference_by_seed,
    rq7_primary_test,
)


ROOT = Path(__file__).resolve().parents[1]

SOURCE_REL = "artifacts/results/paired_effects.csv"
SOURCE = ROOT / SOURCE_REL

OUTPUT_DIR = ROOT / "analysis" / "rq7"
GAIN_OUTPUT = OUTPUT_DIR / "rq7_architecture_gain_by_seed.csv"
PRIMARY_OUTPUT = OUTPUT_DIR / "rq7_primary_test.json"

RQ7_CORE = ROOT / "src" / "statistics" / "rq7_architecture_effect.py"
RQ7_VALIDATOR = ROOT / "scripts" / "S6_08_validate_rq7_architecture_effect.py"
SEED_SUMMARY = ROOT / "src" / "statistics" / "seed_summary.py"
STATISTICAL_MANIFEST = (
    ROOT
    / "artifacts"
    / "preflight"
    / "statistics"
    / "statistical_implementation_manifest.json"
)

EXPECTED_PYTHON = "3.10.13"
EXPECTED_NUMPY = "1.26.4"
EXPECTED_SCIPY = "1.15.3"

EXPECTED_SOURCE_SHA256 = (
    "bad5a179711443636a13bd6501a1d90a7e8c5dadfb16b9c09df74bf502b727bb"
)
EXPECTED_RQ7_CORE_SHA256 = (
    "3bc1d87feee80ecaad2ea00142f7dd56fb6491d4f2b626db005f32c1c44e6471"
)
EXPECTED_RQ7_VALIDATOR_SHA256 = (
    "d14f7688f0c9af7c10d99ee6dfa3e76a2010913a3e6926216d81d0e14a861b7d"
)
EXPECTED_SEED_SUMMARY_SHA256 = (
    "2a404ac0f6097cad16816b550fcd1c9ca240fab5535854dc0fd5e4433e49ff3f"
)
EXPECTED_STATISTICAL_MANIFEST_SHA256 = (
    "eb2f5eb19164030d4e1435a181996315c6009aef61058ddfed10c13431f9f30e"
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

STATISTICS_SEMANTICS = "FROZEN_S6_REUSE"
REPRESENTATION_AUTHORITY = "S10_CANONICAL_MACHINE_READABLE"

GAIN_FIELDS = [
    "seed_index",
    "training_seed",
    "A_R50_s",
    "A_Swin-T_s",
    "D_s",
    "source_artifact",
    "statistics_semantics",
    "representation_authority",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite_float(value: object, label: str) -> float:
    try:
        result = float(value)
    except Exception as exc:
        raise RuntimeError(f"NON_NUMERIC:{label}:{value!r}") from exc
    require(math.isfinite(result), f"NONFINITE:{label}:{value!r}")
    return result


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=GAIN_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    # ------------------------------------------------------------
    # Runtime / provenance locks
    # ------------------------------------------------------------
    require(platform.python_version() == EXPECTED_PYTHON,
            f"PYTHON_VERSION:{platform.python_version()}")
    require(np.__version__ == EXPECTED_NUMPY,
            f"NUMPY_VERSION:{np.__version__}")
    require(scipy.__version__ == EXPECTED_SCIPY,
            f"SCIPY_VERSION:{scipy.__version__}")

    locked_files = [
        (SOURCE, EXPECTED_SOURCE_SHA256, "PAIRED_EFFECTS"),
        (RQ7_CORE, EXPECTED_RQ7_CORE_SHA256, "RQ7_CORE"),
        (RQ7_VALIDATOR, EXPECTED_RQ7_VALIDATOR_SHA256, "RQ7_VALIDATOR"),
        (SEED_SUMMARY, EXPECTED_SEED_SUMMARY_SHA256, "SEED_SUMMARY"),
        (
            STATISTICAL_MANIFEST,
            EXPECTED_STATISTICAL_MANIFEST_SHA256,
            "STATISTICAL_MANIFEST",
        ),
    ]

    for path, expected_sha, label in locked_files:
        require(path.is_file(), f"MISSING_{label}:{path}")
        observed_sha = sha256_file(path)
        require(
            observed_sha == expected_sha,
            f"{label}_SHA256_MISMATCH:{observed_sha}",
        )

    # ------------------------------------------------------------
    # Frozen RQ7 implementation contract
    # ------------------------------------------------------------
    require(tuple(RQ7_ARCHITECTURES) == ("R50", "Swin-T"),
            f"RQ7_ARCHITECTURES:{RQ7_ARCHITECTURES}")
    require(tuple(RQ7_BUDGETS) == ("1%", "5%", "10%", "20%"),
            f"RQ7_BUDGETS:{RQ7_BUDGETS}")
    require(RQ7_TOTAL_ROWS == 80, f"RQ7_TOTAL_ROWS:{RQ7_TOTAL_ROWS}")
    require(RQ7_EFFECT == "D_s", f"RQ7_EFFECT:{RQ7_EFFECT}")
    require(RQ7_ALTERNATIVE == "two-sided",
            f"RQ7_ALTERNATIVE:{RQ7_ALTERNATIVE}")
    require(RQ7_DF == 9, f"RQ7_DF:{RQ7_DF}")
    require(math.isclose(RQ7_CI_LEVEL, 0.95),
            f"RQ7_CI_LEVEL:{RQ7_CI_LEVEL}")

    # ------------------------------------------------------------
    # Canonical paired-effect input
    # ------------------------------------------------------------
    raw_rows = read_csv(SOURCE)
    require(len(raw_rows) == 80, f"PAIRED_EFFECTS_ROWS:{len(raw_rows)}")

    required_fields = {
        "architecture",
        "budget",
        "seed_index",
        "training_seed",
        "mAP_test_SUP",
        "mAP_test_SSL",
        "delta_mAP_test",
    }
    require(
        required_fields.issubset(raw_rows[0].keys()),
        "PAIRED_EFFECTS_REQUIRED_FIELDS_MISSING",
    )

    normalized: list[dict[str, object]] = []
    raw_keys: set[tuple[str, str, int, int]] = set()

    for index, row in enumerate(raw_rows, start=1):
        arch_raw = row["architecture"]
        budget_raw = row["budget"]

        require(
            arch_raw in ARCH_MAP,
            f"ROW_{index}_UNEXPECTED_ARCHITECTURE:{arch_raw}",
        )
        require(
            budget_raw in BUDGET_MAP,
            f"ROW_{index}_UNEXPECTED_BUDGET:{budget_raw}",
        )

        seed_index = int(row["seed_index"])
        training_seed = int(row["training_seed"])

        require(
            1 <= seed_index <= 10,
            f"ROW_{index}_INVALID_SEED_INDEX:{seed_index}",
        )
        require(
            training_seed == int(OFFICIAL_TRAINING_SEEDS[seed_index - 1]),
            f"ROW_{index}_SEED_MAPPING_MISMATCH",
        )

        key = (
            arch_raw,
            budget_raw,
            seed_index,
            training_seed,
        )
        require(key not in raw_keys, f"DUPLICATE_SOURCE_CELL:{key}")
        raw_keys.add(key)

        sup = finite_float(row["mAP_test_SUP"], f"row_{index}.mAP_test_SUP")
        ssl = finite_float(row["mAP_test_SSL"], f"row_{index}.mAP_test_SSL")
        delta = finite_float(
            row["delta_mAP_test"],
            f"row_{index}.delta_mAP_test",
        )

        require(
            math.isclose(
                delta,
                ssl - sup,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ),
            f"ROW_{index}_DELTA_MISMATCH",
        )

        normalized.append(
            {
                "architecture": ARCH_MAP[arch_raw],
                "budget": BUDGET_MAP[budget_raw],
                "seed_index": seed_index,
                "training_seed": training_seed,
                "delta_mAP_test": delta,
            }
        )

    require(len(raw_keys) == 80, f"UNIQUE_SOURCE_KEYS:{len(raw_keys)}")

    # ------------------------------------------------------------
    # Reuse frozen S6 RQ7 computation exactly
    # ------------------------------------------------------------
    architecture_gains = compute_rq7_architecture_gain_by_seed(normalized)
    seed_differences = compute_rq7_difference_by_seed(architecture_gains)
    primary_result = rq7_primary_test(seed_differences)

    require(
        tuple(architecture_gains.keys()) == OFFICIAL_TRAINING_SEEDS,
        "ARCHITECTURE_GAIN_SEED_ORDER_MISMATCH",
    )
    require(
        tuple(seed_differences.keys()) == OFFICIAL_TRAINING_SEEDS,
        "D_S_SEED_ORDER_MISMATCH",
    )
    require(len(seed_differences) == 10, "D_S_SEED_COUNT_NOT_10")

    require(primary_result["effect"] == "D_s", "PRIMARY_EFFECT_NOT_D_S")
    require(
        primary_result["effect_definition"] == "A_Swin-T_minus_A_R50",
        "PRIMARY_EFFECT_DEFINITION_MISMATCH",
    )
    require(primary_result["n"] == 10, "PRIMARY_N_NOT_10")
    require(primary_result["df"] == 9, "PRIMARY_DF_NOT_9")
    require(
        primary_result["sample_sd_ddof"] == 1,
        "PRIMARY_DDOF_NOT_1",
    )
    require(
        primary_result["alternative"] == "two-sided",
        "PRIMARY_ALTERNATIVE_NOT_TWO_SIDED",
    )
    require(
        primary_result["p_value_sidedness"] == "two-sided",
        "PRIMARY_P_VALUE_NOT_TWO_SIDED",
    )
    require(
        primary_result["ci_sidedness"] == "two-sided",
        "PRIMARY_CI_NOT_TWO_SIDED",
    )
    require(
        primary_result["interpretation_scope"]
        == "DIFFERENCE_IN_SSL_GAIN_BETWEEN_ARCHITECTURES_NOT_ABSOLUTE_SUPERIORITY",
        "PRIMARY_INTERPRETATION_SCOPE_MISMATCH",
    )

    # ------------------------------------------------------------
    # Canonical seed-level output
    # ------------------------------------------------------------
    gain_rows: list[dict[str, object]] = []

    for seed_index, seed in enumerate(OFFICIAL_TRAINING_SEEDS, start=1):
        gains = architecture_gains[int(seed)]

        require(set(gains.keys()) == {"R50", "Swin-T"},
                f"ARCHITECTURE_GAIN_KEYS:{seed}:{gains.keys()}")

        a_r50 = finite_float(gains["R50"], f"A_R50:{seed}")
        a_swin = finite_float(gains["Swin-T"], f"A_Swin-T:{seed}")
        d_s = finite_float(seed_differences[int(seed)], f"D_s:{seed}")

        require(
            math.isclose(
                d_s,
                a_swin - a_r50,
                rel_tol=1e-15,
                abs_tol=1e-15,
            ),
            f"D_S_IDENTITY_MISMATCH:{seed}",
        )

        gain_rows.append(
            {
                "seed_index": seed_index,
                "training_seed": int(seed),
                "A_R50_s": repr(a_r50),
                "A_Swin-T_s": repr(a_swin),
                "D_s": repr(d_s),
                "source_artifact": SOURCE_REL,
                "statistics_semantics": STATISTICS_SEMANTICS,
                "representation_authority": REPRESENTATION_AUTHORITY,
            }
        )

    # ------------------------------------------------------------
    # Canonical primary JSON
    # ------------------------------------------------------------
    primary_payload = {
        "rq": "RQ7",
        "analysis_role": "SECONDARY_FORMAL_HYPOTHESIS",
        "hypothesis_direction": "NON_DIRECTIONAL",
        "scientific_contract": {
            "primitive_effect": "delta_mAP_test=SSL-SUP",
            "architecture_gain": (
                "A_{a,s}=mean_of_four_budget_delta_mAP_test_values"
            ),
            "architecture_difference": "D_s=A_Swin-T,s-A_R50,s",
            "replication_unit": REPLICATION_UNIT,
            "n": 10,
            "df": 9,
            "alternative": "two-sided",
            "ci": "individual_two_sided_95_percent",
            "absolute_swin_superiority_test": False,
            "holm_f2_applied_here": False,
        },
        "source_artifact": SOURCE_REL,
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "frozen_implementation": {
            "rq7_core_sha256": EXPECTED_RQ7_CORE_SHA256,
            "rq7_validator_sha256": EXPECTED_RQ7_VALIDATOR_SHA256,
            "seed_summary_sha256": EXPECTED_SEED_SUMMARY_SHA256,
            "statistical_manifest_sha256": (
                EXPECTED_STATISTICAL_MANIFEST_SHA256
            ),
        },
        "identity_normalization": {
            "architecture": ARCH_MAP,
            "budget": BUDGET_MAP,
            "semantic_change": False,
        },
        "statistics_semantics": STATISTICS_SEMANTICS,
        "statistics_reimplemented": False,
        "primary_result": primary_result,
    }

    write_csv(GAIN_OUTPUT, gain_rows)
    write_json(PRIMARY_OUTPUT, primary_payload)

    print("=== S10.1.16 RQ7 MATERIALIZATION ===")
    print(f"SOURCE_ROWS={len(raw_rows)}")
    print(f"GAIN_ROWS={len(gain_rows)}")
    print("ARCHITECTURES=2")
    print("BUDGETS=4")
    print("SEEDS=10")
    print("PRIMITIVE_EFFECT=delta_mAP_test=SSL-SUP")
    print("ARCHITECTURE_GAIN=A_a_s_MEAN_4_BUDGETS")
    print("PRIMARY_EFFECT=D_s")
    print("DIRECTION=A_Swin-T_MINUS_A_R50")
    print("PRIMARY_TEST=TWO_SIDED_ONE_SAMPLE_T")
    print("DF=9")
    print("CI=INDIVIDUAL_TWO_SIDED_95_PERCENT")
    print("IDENTITY_NORMALIZATION_ONLY=True")
    print("STATISTICS_REIMPLEMENTED=False")
    print("HOLM_F2_APPLIED=False")
    print(f"GAIN_OUTPUT_SHA256={sha256_file(GAIN_OUTPUT)}")
    print(f"PRIMARY_OUTPUT_SHA256={sha256_file(PRIMARY_OUTPUT)}")


if __name__ == "__main__":
    main()
