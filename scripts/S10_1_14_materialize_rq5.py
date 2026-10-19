#!/usr/bin/env python3
"""Materialize canonical S10.1.14 / RQ5 rare-class outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.statistics.rq5_rare_class import (
    RQ5_DEFINED_STATUS,
    RQ5_FORMAL_P_VALUE_REQUIRED,
    RQ5_MISSINGNESS_POLICY,
    RQ5_RARE_CLASSES,
    compute_rq5_seed_gains,
    summarize_rq5_classes,
)

SOURCE_REL = "artifacts/results/paired_effects.csv"

EXPECTED_PAIRED_EFFECTS_SHA256 = (
    "bad5a179711443636a13bd6501a1d90a7e8c5dadfb16b9c09df74bf502b727bb"
)

EXPECTED_RQ5_COMMITTED_SHA256 = (
    "21fd5694902ceccc6956e499bb2739a67e8d4533e56ee39f42280ef7031ef41d"
)

EXPECTED_SEED_SUMMARY_SHA256 = (
    "2a404ac0f6097cad16816b550fcd1c9ca240fab5535854dc0fd5e4433e49ff3f"
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

GAIN_FIELDS = [
    "class_name",
    "seed_index",
    "training_seed",
    "g_c_s",
    "status",
    "defined_cell_count",
    "required_cell_count",
    "undefined_cells",
    "effect_unit",
]

SUMMARY_FIELDS = [
    "class_name",
    "status",
    "mean_paired_gain",
    "sample_sd",
    "sample_sd_ddof",
    "n",
    "df",
    "ci_level",
    "ci_sidedness",
    "ci_low",
    "ci_high",
    "formal_p_value_required",
    "undefined_training_seeds",
    "missingness_policy",
    "effect_unit",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"CSV_NOT_FOUND:{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(rows, f"CSV_EMPTY:{path}")
    return rows


def optional_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.upper() in {"NA", "N/A", "NULL", "NONE"}:
        return None
    x = float(text)
    if math.isnan(x):
        return None
    require(math.isfinite(x), f"NONFINITE_VALUE:{value}")
    return x


def csv_value(value: object) -> object:
    if value is None:
        return "NA"
    if isinstance(value, list):
        return ";".join(str(x) for x in value)
    return value


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gain-output", required=True)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()

    source = REPO / SOURCE_REL
    gain_output = Path(args.gain_output)
    summary_output = Path(args.summary_output)

    require(
        sha256_file(source) == EXPECTED_PAIRED_EFFECTS_SHA256,
        "PAIRED_EFFECTS_SHA256_MISMATCH",
    )
    require(
        sha256_file(REPO / "src/statistics/rq5_rare_class.py")
        == EXPECTED_RQ5_COMMITTED_SHA256,
        "RQ5_COMMITTED_SOURCE_SHA256_MISMATCH",
    )
    require(
        sha256_file(REPO / "src/statistics/seed_summary.py")
        == EXPECTED_SEED_SUMMARY_SHA256,
        "SEED_SUMMARY_SHA256_MISMATCH",
    )

    raw_rows = load_csv(source)
    require(len(raw_rows) == 80, f"PAIRED_EFFECTS_ROWS={len(raw_rows)}")

    normalized: list[dict[str, object]] = []

    for index, row in enumerate(raw_rows, start=1):
        arch_raw = row.get("architecture", "")
        budget_raw = row.get("budget", "")

        require(
            arch_raw in ARCH_MAP,
            f"ROW_{index}_UNEXPECTED_ARCHITECTURE:{arch_raw}",
        )
        require(
            budget_raw in BUDGET_MAP,
            f"ROW_{index}_UNEXPECTED_BUDGET:{budget_raw}",
        )

        normalized.append(
            {
                "architecture": ARCH_MAP[arch_raw],
                "budget": BUDGET_MAP[budget_raw],
                "seed_index": int(row["seed_index"]),
                "training_seed": int(row["training_seed"]),
                "delta_AP_test_Atelectasis": optional_float(
                    row.get("delta_AP_test_Atelectasis")
                ),
                "delta_AP_test_Pneumothorax": optional_float(
                    row.get("delta_AP_test_Pneumothorax")
                ),
            }
        )

    seed_rows = compute_rq5_seed_gains(normalized)
    summary_rows = summarize_rq5_classes(seed_rows)

    require(len(seed_rows) == 20, f"RQ5_GAIN_ROWS={len(seed_rows)}")
    require(len(summary_rows) == 2, f"RQ5_SUMMARY_ROWS={len(summary_rows)}")

    require(
        tuple(row["class_name"] for row in summary_rows) == RQ5_RARE_CLASSES,
        "RQ5_CLASS_ORDER_MISMATCH",
    )

    require(
        all(row["status"] == RQ5_DEFINED_STATUS for row in seed_rows),
        "RQ5_UNDEFINED_SEED_GAIN",
    )
    require(
        all(row["status"] == RQ5_DEFINED_STATUS for row in summary_rows),
        "RQ5_UNDEFINED_CLASS_SUMMARY",
    )

    require(
        all(
            row["formal_p_value_required"] is RQ5_FORMAL_P_VALUE_REQUIRED
            for row in summary_rows
        ),
        "RQ5_FORMAL_P_VALUE_POLICY_MISMATCH",
    )
    require(
        all(
            row["missingness_policy"] == RQ5_MISSINGNESS_POLICY
            for row in summary_rows
        ),
        "RQ5_MISSINGNESS_POLICY_MISMATCH",
    )

    write_csv(gain_output, GAIN_FIELDS, seed_rows)
    write_csv(summary_output, SUMMARY_FIELDS, summary_rows)

    print("=== S10.1.14 RQ5 MATERIALIZER ===")
    print(f"SOURCE={SOURCE_REL}")
    print(f"SOURCE_SHA256={sha256_file(source)}")
    print(
        "RQ5_CORE_COMMITTED_SHA256="
        + sha256_file(REPO / "src/statistics/rq5_rare_class.py")
    )
    print("INPUT_ROWS=80")
    print(f"GAIN_ROWS={len(seed_rows)}")
    print(f"SUMMARY_ROWS={len(summary_rows)}")
    print("RARE_CLASSES=" + ",".join(RQ5_RARE_CLASSES))
    print("ALL_SEED_GAINS_DEFINED=True")
    print("ALL_CLASS_SUMMARIES_DEFINED=True")
    print("FORMAL_P_VALUE_REQUIRED=False")
    print("RARE_MAP_CREATED=False")
    print("IDENTITY_NORMALIZATION_ONLY=True")
    print("STATISTICS_REIMPLEMENTED=False")
    print(f"GAIN_OUTPUT_SHA256={sha256_file(gain_output)}")
    print(f"SUMMARY_OUTPUT_SHA256={sha256_file(summary_output)}")


if __name__ == "__main__":
    main()
