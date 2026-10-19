from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import scipy

from src.statistics.holm_f2 import (
    F2_ALPHA,
    F2_FAMILY_ID,
    F2_M,
    F2_MEMBER_ORDER,
    F2_METHOD,
    holm_f2,
)

ROOT = Path(__file__).resolve().parents[1]

RQ3_REL = "analysis/rq3/rq3_gg_anova.json"
RQ4_REL = "analysis/rq4/rq4_primary_mixed_model.json"
RQ6_REL = "analysis/rq6/rq6_primary_test.json"
RQ7_REL = "analysis/rq7/rq7_primary_test.json"

OUTPUT_REL = "analysis/multiplicity/holm_F2.csv"
OUTPUT = ROOT / OUTPUT_REL

HOLM_CORE = ROOT / "src/statistics/holm_f2.py"
STAT_MANIFEST = ROOT / "artifacts/preflight/statistics/statistical_implementation_manifest.json"

EXPECTED_HOLM_CORE_SHA256 = (
    "d080f91729bb44838d64bba5b4d2c8a351dda76966e09d365b9e4dc9d4225707"
)
EXPECTED_STAT_MANIFEST_SHA256 = (
    "eb2f5eb19164030d4e1435a181996315c6009aef61058ddfed10c13431f9f30e"
)

EXPECTED_RQ3_SHA256 = None
EXPECTED_RQ4_SHA256 = (
    "510e2a68e88914069ecba4d54f061ab9a863d384a5db359cd678285435c559f5"
)
EXPECTED_RQ6_SHA256 = (
    "335009d7a209c57a137f7f25cb9177e0fb971570d14591e213137b64bd76622a"
)
EXPECTED_RQ7_SHA256 = (
    "c67a14b231f309d35000a29fe48a9604a1bf9e959fa29d4d83c60c75d050ea6e"
)

EXPECTED_PYTHON = "3.10.13"
EXPECTED_NUMPY = "1.26.4"
EXPECTED_SCIPY = "1.15.3"

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


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=FIELDS,
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)


def finite_p(value: object, label: str) -> float:
    try:
        p = float(value)
    except Exception as exc:
        raise RuntimeError(f"NON_NUMERIC_P:{label}:{value!r}") from exc
    require(math.isfinite(p), f"NONFINITE_P:{label}")
    require(0.0 <= p <= 1.0, f"P_OUT_OF_RANGE:{label}:{p}")
    return p


def main() -> None:
    require(platform.python_version() == EXPECTED_PYTHON,
            f"PYTHON_VERSION:{platform.python_version()}")
    require(np.__version__ == EXPECTED_NUMPY,
            f"NUMPY_VERSION:{np.__version__}")
    require(scipy.__version__ == EXPECTED_SCIPY,
            f"SCIPY_VERSION:{scipy.__version__}")

    require(F2_FAMILY_ID == "F2", f"F2_FAMILY_ID:{F2_FAMILY_ID}")
    require(tuple(F2_MEMBER_ORDER) == ("RQ3", "RQ4", "RQ6", "RQ7"),
            f"F2_MEMBER_ORDER:{F2_MEMBER_ORDER}")
    require(F2_M == 4, f"F2_M:{F2_M}")
    require(math.isclose(F2_ALPHA, 0.05), f"F2_ALPHA:{F2_ALPHA}")
    require(F2_METHOD == "HOLM_STEP_DOWN", f"F2_METHOD:{F2_METHOD}")

    require(
        sha256_file(HOLM_CORE) == EXPECTED_HOLM_CORE_SHA256,
        "HOLM_CORE_SHA256_MISMATCH",
    )
    require(
        sha256_file(STAT_MANIFEST) == EXPECTED_STAT_MANIFEST_SHA256,
        "STAT_MANIFEST_SHA256_MISMATCH",
    )

    rq3_path = ROOT / RQ3_REL
    rq4_path = ROOT / RQ4_REL
    rq6_path = ROOT / RQ6_REL
    rq7_path = ROOT / RQ7_REL

    for p in (rq3_path, rq4_path, rq6_path, rq7_path):
        require(p.is_file(), f"MISSING_SOURCE:{p}")

    rq3_sha = sha256_file(rq3_path)
    rq4_sha = sha256_file(rq4_path)
    rq6_sha = sha256_file(rq6_path)
    rq7_sha = sha256_file(rq7_path)

    if EXPECTED_RQ3_SHA256 is not None:
        require(rq3_sha == EXPECTED_RQ3_SHA256, "RQ3_SHA256_MISMATCH")

    require(rq4_sha == EXPECTED_RQ4_SHA256, "RQ4_SHA256_MISMATCH")
    require(rq6_sha == EXPECTED_RQ6_SHA256, "RQ6_SHA256_MISMATCH")
    require(rq7_sha == EXPECTED_RQ7_SHA256, "RQ7_SHA256_MISMATCH")

    rq3 = load_json(rq3_path)
    rq4 = load_json(rq4_path)
    rq6 = load_json(rq6_path)
    rq7 = load_json(rq7_path)

    raw_p_values = {
        "RQ3": finite_p(rq3["result"]["p_gg"], "RQ3.result.p_gg"),
        "RQ4": finite_p(
            rq4["result"]["one_sided_p_value"],
            "RQ4.result.one_sided_p_value",
        ),
        "RQ6": finite_p(rq6["result"]["p_value"], "RQ6.result.p_value"),
        "RQ7": finite_p(
            rq7["primary_result"]["p_value"],
            "RQ7.primary_result.p_value",
        ),
    }

    expected_raw = {
        "RQ3": 2.737467932962183e-10,
        "RQ4": 0.13796448268749345,
        "RQ6": 6.278732094208282e-10,
        "RQ7": 0.18513455725443084,
    }

    for member in F2_MEMBER_ORDER:
        require(
            math.isclose(
                raw_p_values[member],
                expected_raw[member],
                rel_tol=1e-15,
                abs_tol=1e-18,
            ),
            f"RAW_P_MISMATCH:{member}:{raw_p_values[member]}",
        )

    result = holm_f2(raw_p_values)

    require(result["family"] == "F2", "RESULT_FAMILY_MISMATCH")
    require(result["method"] == "HOLM_STEP_DOWN", "RESULT_METHOD_MISMATCH")
    require(result["family_size"] == 4, "RESULT_FAMILY_SIZE_MISMATCH")
    require(math.isclose(result["fwer"], 0.05), "RESULT_FWER_MISMATCH")
    require(result["family_membership_locked"] is True,
            "RESULT_FAMILY_NOT_LOCKED")
    require(result["individual_ci_adjusted"] is False,
            "RESULT_CI_ADJUSTED_TRUE")

    source_meta = {
        "RQ3": (RQ3_REL, "result.p_gg", rq3_sha),
        "RQ4": (RQ4_REL, "result.one_sided_p_value", rq4_sha),
        "RQ6": (RQ6_REL, "result.p_value", rq6_sha),
        "RQ7": (RQ7_REL, "primary_result.p_value", rq7_sha),
    }

    rows = []

    for item in result["ordered_results"]:
        member = item["member"]
        src, field, sha = source_meta[member]

        rows.append(
            {
                "schema_version": "1.0.0",
                "family": result["family"],
                "method": result["method"],
                "family_size": result["family_size"],
                "fwer": repr(float(result["fwer"])),
                "rank": int(item["rank"]),
                "member": member,
                "raw_p_value": repr(float(item["raw_p_value"])),
                "holm_multiplier": int(item["holm_multiplier"]),
                "holm_threshold": repr(float(item["holm_threshold"])),
                "adjusted_p_value": repr(float(item["adjusted_p_value"])),
                "reject_at_fwer_0_05": (
                    "true" if item["reject_at_fwer_0_05"] else "false"
                ),
                "source_artifact": src,
                "source_field": field,
                "source_sha256": sha,
                "family_membership_locked": "true",
                "individual_ci_adjusted": "false",
                "statistics_semantics": "FROZEN_S6_REUSE",
            }
        )

    require(len(rows) == 4, f"F2_OUTPUT_ROWS:{len(rows)}")

    write_csv(OUTPUT, rows)

    print("=== S10.1.16 F2 HOLM MATERIALIZATION ===")
    print("FAMILY=F2")
    print("MEMBERS=RQ3,RQ4,RQ6,RQ7")
    print("M=4")
    print("METHOD=HOLM_STEP_DOWN")
    print("FWER=0.05")
    print("INDIVIDUAL_CI_ADJUSTED=FALSE")
    print("STATISTICS_REIMPLEMENTED=FALSE")
    print("OUTPUT=" + OUTPUT_REL)
    print("OUTPUT_SHA256=" + sha256_file(OUTPUT))


if __name__ == "__main__":
    main()
