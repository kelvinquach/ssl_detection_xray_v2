"""RQ5 rare-class summaries under the locked SSOD statistical contract."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import numpy as np
from scipy import stats

from src.statistics.seed_summary import (
    OFFICIAL_SEED_COUNT,
    OFFICIAL_TRAINING_SEEDS,
    REPLICATION_UNIT,
    SAMPLE_SD_DDOF,
    summarize_seed_runs,
)

RQ5_RARE_CLASSES = ("Atelectasis", "Pneumothorax")
RQ5_ARCHITECTURES = ("R50", "Swin-T")
RQ5_BUDGETS = ("1%", "5%", "10%", "20%")
RQ5_CELL_COUNT = len(RQ5_ARCHITECTURES) * len(RQ5_BUDGETS)
RQ5_CI_LEVEL = 0.95
RQ5_ALPHA = 0.05
RQ5_INPUT_EFFECT_UNIT = "AP_POINTS"
RQ5_UNDEFINED_STATUS = "UNDEFINED"
RQ5_DEFINED_STATUS = "DEFINED"
RQ5_MISSINGNESS_POLICY = "PROPAGATE_NA_NO_AVAILABLE_CASE_MEAN"
RQ5_FORMAL_P_VALUE_REQUIRED = False
RQ5_RARE_MAP_CREATED = False

RQ5_DELTA_COLUMNS = {
    "Atelectasis": "delta_AP_test_Atelectasis",
    "Pneumothorax": "delta_AP_test_Pneumothorax",
}


class RQ5RareClassError(ValueError):
    """Raised when RQ5 inputs violate the locked analysis contract."""


def _expected_cell_keys() -> tuple[tuple[str, str], ...]:
    return tuple(
        (architecture, budget)
        for architecture in RQ5_ARCHITECTURES
        for budget in RQ5_BUDGETS
    )


def _validate_rows(rows: Iterable[Mapping[str, object]]) -> list[Mapping[str, object]]:
    rows = list(rows)
    if len(rows) != OFFICIAL_SEED_COUNT * RQ5_CELL_COUNT:
        raise RQ5RareClassError(
            f"RQ5 requires exactly {OFFICIAL_SEED_COUNT * RQ5_CELL_COUNT} rows; "
            f"observed={len(rows)}"
        )

    required = {
        "architecture",
        "budget",
        "seed_index",
        "training_seed",
        *RQ5_DELTA_COLUMNS.values(),
    }
    expected_seed_map = {
        index: seed
        for index, seed in enumerate(OFFICIAL_TRAINING_SEEDS, start=1)
    }
    observed_keys: set[tuple[str, str, int]] = set()

    for row_index, row in enumerate(rows, start=1):
        missing = required.difference(row)
        if missing:
            raise RQ5RareClassError(
                f"row {row_index} missing required columns: {sorted(missing)}"
            )

        architecture = str(row["architecture"])
        budget = str(row["budget"])
        if architecture not in RQ5_ARCHITECTURES:
            raise RQ5RareClassError(
                f"row {row_index} invalid architecture={architecture}"
            )
        if budget not in RQ5_BUDGETS:
            raise RQ5RareClassError(f"row {row_index} invalid budget={budget}")

        try:
            seed_index = int(row["seed_index"])
            training_seed = int(row["training_seed"])
        except (TypeError, ValueError) as exc:
            raise RQ5RareClassError(
                f"row {row_index} seed identity must be integer-valued"
            ) from exc

        if seed_index not in expected_seed_map:
            raise RQ5RareClassError(
                f"row {row_index} seed_index must be in 1..{OFFICIAL_SEED_COUNT}"
            )
        if training_seed != expected_seed_map[seed_index]:
            raise RQ5RareClassError(
                f"row {row_index} training_seed does not match locked seed_index mapping"
            )

        key = (architecture, budget, training_seed)
        if key in observed_keys:
            raise RQ5RareClassError(
                f"duplicate architecture-budget-seed row: {key}"
            )
        observed_keys.add(key)

        for class_name, column in RQ5_DELTA_COLUMNS.items():
            value = row[column]
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise RQ5RareClassError(
                    f"row {row_index} {column} must be numeric or NA"
                ) from exc
            if not math.isnan(numeric) and not math.isfinite(numeric):
                raise RQ5RareClassError(
                    f"row {row_index} {column} must be finite or NA"
                )
            if math.isfinite(numeric) and not (-100.0 <= numeric <= 100.0):
                raise RQ5RareClassError(
                    f"row {row_index} {column} outside AP-point difference range [-100,100]"
                )

    expected_cells = set(_expected_cell_keys())
    for training_seed in OFFICIAL_TRAINING_SEEDS:
        observed_cells = {
            (str(row["architecture"]), str(row["budget"]))
            for row in rows
            if int(row["training_seed"]) == training_seed
        }
        if observed_cells != expected_cells:
            raise RQ5RareClassError(
                f"seed {training_seed} must contain exactly the 8 locked "
                f"architecture-budget cells"
            )

    return rows


def compute_rq5_seed_gains(
    rows: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Compute G_{c,s}; any undefined cell propagates NA for that class/seed."""
    rows = _validate_rows(rows)
    output: list[dict[str, object]] = []

    for class_name in RQ5_RARE_CLASSES:
        column = RQ5_DELTA_COLUMNS[class_name]
        for seed_index, training_seed in enumerate(
            OFFICIAL_TRAINING_SEEDS, start=1
        ):
            seed_rows = [
                row for row in rows
                if int(row["training_seed"]) == training_seed
            ]
            values: list[float] = []
            undefined_cells: list[str] = []

            for architecture, budget in _expected_cell_keys():
                row = next(
                    row
                    for row in seed_rows
                    if str(row["architecture"]) == architecture
                    and str(row["budget"]) == budget
                )
                raw = row[column]
                value = float("nan") if raw is None else float(raw)
                if math.isnan(value):
                    undefined_cells.append(f"{architecture}__{budget}")
                else:
                    values.append(value)

            if undefined_cells:
                output.append(
                    {
                        "class_name": class_name,
                        "seed_index": seed_index,
                        "training_seed": training_seed,
                        "g_c_s": None,
                        "status": RQ5_UNDEFINED_STATUS,
                        "defined_cell_count": len(values),
                        "required_cell_count": RQ5_CELL_COUNT,
                        "undefined_cells": undefined_cells,
                        "effect_unit": RQ5_INPUT_EFFECT_UNIT,
                    }
                )
            else:
                output.append(
                    {
                        "class_name": class_name,
                        "seed_index": seed_index,
                        "training_seed": training_seed,
                        "g_c_s": float(np.mean(values)),
                        "status": RQ5_DEFINED_STATUS,
                        "defined_cell_count": RQ5_CELL_COUNT,
                        "required_cell_count": RQ5_CELL_COUNT,
                        "undefined_cells": [],
                        "effect_unit": RQ5_INPUT_EFFECT_UNIT,
                    }
                )

    return output


def summarize_rq5_classes(
    seed_gain_rows: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Summarize each rare class across the exact 10 locked training seeds."""
    rows = list(seed_gain_rows)
    summaries: list[dict[str, object]] = []

    for class_name in RQ5_RARE_CLASSES:
        class_rows = [row for row in rows if row.get("class_name") == class_name]
        if len(class_rows) != OFFICIAL_SEED_COUNT:
            raise RQ5RareClassError(
                f"{class_name} requires exactly {OFFICIAL_SEED_COUNT} seed rows"
            )

        observed_seed_order = tuple(int(row["training_seed"]) for row in class_rows)
        if observed_seed_order != OFFICIAL_TRAINING_SEEDS:
            raise RQ5RareClassError(
                f"{class_name} seed rows must preserve exact locked seed order"
            )

        undefined = [
            int(row["training_seed"])
            for row in class_rows
            if row.get("status") != RQ5_DEFINED_STATUS
            or row.get("g_c_s") is None
        ]
        if undefined:
            summaries.append(
                {
                    "class_name": class_name,
                    "status": RQ5_UNDEFINED_STATUS,
                    "mean_paired_gain": None,
                    "sample_sd": None,
                    "sample_sd_ddof": SAMPLE_SD_DDOF,
                    "n": OFFICIAL_SEED_COUNT,
                    "df": OFFICIAL_SEED_COUNT - 1,
                    "ci_level": RQ5_CI_LEVEL,
                    "ci_sidedness": "two-sided",
                    "ci_low": None,
                    "ci_high": None,
                    "formal_p_value_required": RQ5_FORMAL_P_VALUE_REQUIRED,
                    "undefined_training_seeds": undefined,
                    "missingness_policy": RQ5_MISSINGNESS_POLICY,
                    "effect_unit": RQ5_INPUT_EFFECT_UNIT,
                }
            )
            continue

        values_by_seed = {
            int(row["training_seed"]): float(row["g_c_s"])
            for row in class_rows
        }
        summary = summarize_seed_runs(values_by_seed)
        mean = float(summary["mean"])
        sample_sd = float(summary["sample_sd"])
        n = OFFICIAL_SEED_COUNT
        df = n - 1
        standard_error = sample_sd / math.sqrt(n)
        critical_value = float(stats.t.ppf(1.0 - RQ5_ALPHA / 2.0, df=df))

        summaries.append(
            {
                "class_name": class_name,
                "status": RQ5_DEFINED_STATUS,
                "mean_paired_gain": mean,
                "sample_sd": sample_sd,
                "sample_sd_ddof": int(summary["sample_sd_ddof"]),
                "n": n,
                "df": df,
                "ci_level": RQ5_CI_LEVEL,
                "ci_sidedness": "two-sided",
                "ci_low": mean - critical_value * standard_error,
                "ci_high": mean + critical_value * standard_error,
                "formal_p_value_required": RQ5_FORMAL_P_VALUE_REQUIRED,
                "undefined_training_seeds": [],
                "missingness_policy": RQ5_MISSINGNESS_POLICY,
                "effect_unit": RQ5_INPUT_EFFECT_UNIT,
            }
        )

    return summaries
