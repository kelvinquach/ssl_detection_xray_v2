from __future__ import annotations

import csv
import hashlib
import math
import platform
from pathlib import Path

import numpy as np
import scipy

from src.statistics.holm_f3 import (
    F3_ALPHA,
    F3_FAMILY_ID,
    F3_M,
    F3_MEMBER_ORDER,
    F3_METHOD,
    holm_f3,
)

ROOT = Path(__file__).resolve().parents[1]

SOURCE_REL = "analysis/rq3/rq3_pairwise_contrasts.csv"
SOURCE = ROOT / SOURCE_REL

OUTPUT_REL = "analysis/multiplicity/holm_F3.csv"
OUTPUT = ROOT / OUTPUT_REL

HOLM_CORE = ROOT / "src/statistics/holm_f3.py"
STAT_MANIFEST = ROOT / "artifacts/preflight/statistics/statistical_implementation_manifest.json"

EXPECTED_SOURCE_SHA256 = (
    "b0c686b9904bee4fcdce1619ccfbe0bca3292b6f6b58a5ff9d542a2966e38750"
)
EXPECTED_HOLM_CORE_SHA256 = (
    "5a3ebb59f83927c8df55ecc9f2dcf512d333914cdeed3bd6c321d1f5f3883c80"
)
EXPECTED_STAT_MANIFEST_SHA256 = (
    "eb2f5eb19164030d4e1435a181996315c6009aef61058ddfed10c13431f9f30e"
)

EXPECTED_PYTHON = "3.10.13"
EXPECTED_NUMPY = "1.26.4"
EXPECTED_SCIPY = "1.15.3"

EXPECTED_RAW_P = {
    "5%-1%": 8.439822480900986e-06,
    "10%-1%": 0.05908257253144409,
    "20%-1%": 0.001673997481948107,
    "10%-5%": 1.7727785908857152e-05,
    "20%-5%": 7.4353987591065255e-09,
    "20%-10%": 6.302947272483086e-05,
}

FIELDS = [
    "schema_version",
    "family",
    "method",
    "family_size",
    "fwer",
    "rank",
    "member",
    "raw_p_value",
    "holm_multiplier",
    "holm_threshold",
    "adjusted_p_value",
    "reject_at_fwer_0_05",
    "source_artifact",
    "source_field",
    "source_sha256",
    "family_membership_locked",
    "individual_ci_adjusted",
    "statistics_semantics",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_p(value: object, label: str) -> float:
    try:
        p = float(value)
    except Exception as exc:
        raise RuntimeError(f"NON_NUMERIC_P:{label}:{value!r}") from exc

    require(math.isfinite(p), f"NONFINITE_P:{label}")
    require(0.0 <= p <= 1.0, f"P_OUT_OF_RANGE:{label}:{p}")
    return p


def main() -> None:
    require(
        platform.python_version() == EXPECTED_PYTHON,
        f"PYTHON_VERSION:{platform.python_version()}",
    )
    require(np.__version__ == EXPECTED_NUMPY, f"NUMPY_VERSION:{np.__version__}")
    require(scipy.__version__ == EXPECTED_SCIPY, f"SCIPY_VERSION:{scipy.__version__}")

    require(F3_FAMILY_ID == "F3", f"F3_FAMILY_ID:{F3_FAMILY_ID}")
    require(
        tuple(F3_MEMBER_ORDER)
        == ("5%-1%", "10%-1%", "20%-1%", "10%-5%", "20%-5%", "20%-10%"),
        f"F3_MEMBER_ORDER:{F3_MEMBER_ORDER}",
    )
    require(F3_M == 6, f"F3_M:{F3_M}")
    require(math.isclose(F3_ALPHA, 0.05), f"F3_ALPHA:{F3_ALPHA}")
    require(F3_METHOD == "HOLM_STEP_DOWN", f"F3_METHOD:{F3_METHOD}")

    require(SOURCE.is_file(), f"MISSING_SOURCE:{SOURCE}")
    require(HOLM_CORE.is_file(), f"MISSING_CORE:{HOLM_CORE}")
    require(STAT_MANIFEST.is_file(), f"MISSING_MANIFEST:{STAT_MANIFEST}")

    source_sha = sha256_file(SOURCE)
    core_sha = sha256_file(HOLM_CORE)
    manifest_sha = sha256_file(STAT_MANIFEST)

    require(source_sha == EXPECTED_SOURCE_SHA256, "SOURCE_SHA256_MISMATCH")
    require(core_sha == EXPECTED_HOLM_CORE_SHA256, "HOLM_CORE_SHA256_MISMATCH")
    require(
        manifest_sha == EXPECTED_STAT_MANIFEST_SHA256,
        "STAT_MANIFEST_SHA256_MISMATCH",
    )

    with SOURCE.open("r", encoding="utf-8-sig", newline="") as f:
        source_rows = list(csv.DictReader(f))

    require(len(source_rows) == 6, f"SOURCE_ROWS:{len(source_rows)}")

    by_contrast = {}

    for row in source_rows:
        contrast = row["contrast"]

        require(contrast not in by_contrast, f"DUPLICATE_CONTRAST:{contrast}")
        require(
            row["p_value_sidedness"] == "two-sided",
            f"NON_TWO_SIDED:{contrast}:{row['p_value_sidedness']}",
        )
        require(
            row["holm_applied"].lower() == "false",
            f"HOLM_ALREADY_APPLIED:{contrast}",
        )
        require(
            row["holm_deferred_task"] == "S6.10",
            f"HOLM_DEFERRED_TASK:{contrast}:{row['holm_deferred_task']}",
        )
        require(
            row["multiplicity_family"] == "F3",
            f"FAMILY_MISMATCH:{contrast}:{row['multiplicity_family']}",
        )

        by_contrast[contrast] = row

    require(
        set(by_contrast) == set(F3_MEMBER_ORDER),
        f"MEMBER_SET_MISMATCH:{sorted(by_contrast)}",
    )

    raw_p_values = {
        member: finite_p(
            by_contrast[member]["raw_p_value"],
            f"{member}.raw_p_value",
        )
        for member in F3_MEMBER_ORDER
    }

    for member in F3_MEMBER_ORDER:
        require(
            math.isclose(
                raw_p_values[member],
                EXPECTED_RAW_P[member],
                rel_tol=1e-15,
                abs_tol=1e-18,
            ),
            f"RAW_P_MISMATCH:{member}:{raw_p_values[member]}",
        )

    result = holm_f3(raw_p_values)

    require(result["family"] == "F3", "RESULT_FAMILY_MISMATCH")
    require(result["family_size"] == 6, "RESULT_FAMILY_SIZE_MISMATCH")
    require(result["method"] == "HOLM_STEP_DOWN", "RESULT_METHOD_MISMATCH")
    require(math.isclose(float(result["fwer"]), 0.05), "RESULT_FWER_MISMATCH")
    require(
        result["family_membership_locked"] is True,
        "RESULT_FAMILY_NOT_LOCKED",
    )
    require(
        result["individual_ci_adjusted"] is False,
        "RESULT_CI_ADJUSTED_TRUE",
    )
    require(len(result["ordered_results"]) == 6, "RESULT_ROWS_NOT_6")

    adjusted = [
        float(row["adjusted_p_value"])
        for row in result["ordered_results"]
    ]
    require(
        all(
            adjusted[i] <= adjusted[i + 1]
            for i in range(len(adjusted) - 1)
        ),
        "ADJUSTED_P_NOT_MONOTONE",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=FIELDS,
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()

        for item in result["ordered_results"]:
            writer.writerow(
                {
                    "schema_version": "1.0.0",
                    "family": result["family"],
                    "method": result["method"],
                    "family_size": result["family_size"],
                    "fwer": repr(float(result["fwer"])),
                    "rank": int(item["rank"]),
                    "member": item["member"],
                    "raw_p_value": repr(float(item["raw_p_value"])),
                    "holm_multiplier": int(item["holm_multiplier"]),
                    "holm_threshold": repr(float(item["holm_threshold"])),
                    "adjusted_p_value": repr(float(item["adjusted_p_value"])),
                    "reject_at_fwer_0_05": (
                        "true"
                        if item["reject_at_fwer_0_05"]
                        else "false"
                    ),
                    "source_artifact": SOURCE_REL,
                    "source_field": "raw_p_value",
                    "source_sha256": source_sha,
                    "family_membership_locked": "true",
                    "individual_ci_adjusted": "false",
                    "statistics_semantics": "FROZEN_S6_REUSE",
                }
            )

    print("=== S10.1 HOLM F3 MATERIALIZATION ===")
    print("FAMILY=F3")
    print("MEMBERS=" + ",".join(F3_MEMBER_ORDER))
    print("M=6")
    print("METHOD=HOLM_STEP_DOWN")
    print("FWER=0.05")
    print("INDIVIDUAL_CI_ADJUSTED=FALSE")
    print("STATISTICS_REIMPLEMENTED=FALSE")
    print("SOURCE=" + SOURCE_REL)
    print("SOURCE_SHA256=" + source_sha)
    print("OUTPUT=" + OUTPUT_REL)
    print("OUTPUT_SHA256=" + sha256_file(OUTPUT))


if __name__ == "__main__":
    main()
