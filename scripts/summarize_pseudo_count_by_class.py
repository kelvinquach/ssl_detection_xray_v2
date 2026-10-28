from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(r"D:\ssl_detection_xray_v2")

INPUT_CSV = (
    REPO
    / "analysis"
    / "pseudo_count_diagnostics"
    / "pseudo_count_seed_class.csv"
)

OUTPUT_DIR = (
    REPO
    / "analysis"
    / "pseudo_count_diagnostics"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "pseudo_count_arch_budget_class_summary.csv"
)

EXPECTED_RUNS = 80
EXPECTED_CLASSES = 14
EXPECTED_SEEDS_PER_CELL = 10
EXPECTED_ROWS = EXPECTED_RUNS * EXPECTED_CLASSES

EXPECTED_ARCHITECTURES = {
    "R50-FPN",
    "Swin-T-FPN",
}

EXPECTED_BUDGETS = {
    "1%",
    "5%",
    "10%",
    "20%",
}

EXPECTED_THRESHOLD = 0.90
EXPECTED_COMPARATOR = "STRICT_GREATER"

ARCH_ORDER = {
    "R50-FPN": 0,
    "Swin-T-FPN": 1,
}

BUDGET_ORDER = {
    "1%": 0,
    "5%": 1,
    "10%": 2,
    "20%": 3,
}


def q1(x: pd.Series) -> float:
    """
    First quartile using NumPy/Pandas linear interpolation.

    This is the default continuous-sample quantile definition used by
    pandas Series.quantile(interpolation='linear').
    """
    return float(x.quantile(0.25, interpolation="linear"))


def q3(x: pd.Series) -> float:
    """
    Third quartile using NumPy/Pandas linear interpolation.
    """
    return float(x.quantile(0.75, interpolation="linear"))


def fmt_number(value: float) -> str:
    """
    Reviewer/manuscript-friendly formatting.

    Counts are summarized across seeds, so means/SD/quartiles may be
    non-integers even though individual seed counts are integers.
    """
    return f"{float(value):.2f}"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.is_file():
        raise FileNotFoundError(
            f"Input CSV not found: {INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    required_columns = {
        "run_id",
        "architecture",
        "budget",
        "seed_index",
        "attempt_id",
        "category_id",
        "class_name",
        "classification_threshold",
        "acceptance_comparator",
        "accepted_pseudo_count",
    }

    missing = required_columns.difference(df.columns)
    if missing:
        raise RuntimeError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    # ------------------------------------------------------------
    # 1. Global integrity checks
    # ------------------------------------------------------------

    if len(df) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} rows, observed={len(df)}"
        )

    unique_runs = df["run_id"].nunique()
    if unique_runs != EXPECTED_RUNS:
        raise RuntimeError(
            f"Expected {EXPECTED_RUNS} unique runs, "
            f"observed={unique_runs}"
        )

    observed_arch = set(df["architecture"].unique())
    if observed_arch != EXPECTED_ARCHITECTURES:
        raise RuntimeError(
            f"Architecture mismatch: {observed_arch}"
        )

    observed_budgets = set(df["budget"].unique())
    if observed_budgets != EXPECTED_BUDGETS:
        raise RuntimeError(
            f"Budget mismatch: {observed_budgets}"
        )

    thresholds = set(
        df["classification_threshold"]
        .astype(float)
        .unique()
    )

    if thresholds != {EXPECTED_THRESHOLD}:
        raise RuntimeError(
            f"Threshold mismatch: {thresholds}"
        )

    comparators = set(
        df["acceptance_comparator"].astype(str).unique()
    )

    if comparators != {EXPECTED_COMPARATOR}:
        raise RuntimeError(
            f"Comparator mismatch: {comparators}"
        )

    # Counts must be non-negative integer-valued.
    counts_numeric = pd.to_numeric(
        df["accepted_pseudo_count"],
        errors="raise",
    )

    if (counts_numeric < 0).any():
        raise RuntimeError(
            "accepted_pseudo_count contains negative values"
        )

    if not np.all(
        np.isclose(
            counts_numeric.to_numpy(),
            np.round(counts_numeric.to_numpy()),
        )
    ):
        raise RuntimeError(
            "accepted_pseudo_count contains non-integer values"
        )

    df["accepted_pseudo_count"] = (
        counts_numeric.astype(int)
    )

    # ------------------------------------------------------------
    # 2. Cell integrity:
    #    architecture × budget × class must have exactly 10 seeds
    # ------------------------------------------------------------

    cell_check = (
        df.groupby(
            [
                "architecture",
                "budget",
                "category_id",
                "class_name",
            ],
            dropna=False,
        )
        .agg(
            n_rows=("accepted_pseudo_count", "size"),
            n_seeds=("seed_index", "nunique"),
            n_runs=("run_id", "nunique"),
        )
        .reset_index()
    )

    bad_cells = cell_check[
        (cell_check["n_rows"] != EXPECTED_SEEDS_PER_CELL)
        | (
            cell_check["n_seeds"]
            != EXPECTED_SEEDS_PER_CELL
        )
        | (
            cell_check["n_runs"]
            != EXPECTED_SEEDS_PER_CELL
        )
    ]

    if not bad_cells.empty:
        raise RuntimeError(
            "Some architecture × budget × class cells "
            "do not contain exactly 10 unique seeds/runs:\n"
            + bad_cells.to_string(index=False)
        )

    expected_cells = (
        len(EXPECTED_ARCHITECTURES)
        * len(EXPECTED_BUDGETS)
        * EXPECTED_CLASSES
    )

    if len(cell_check) != expected_cells:
        raise RuntimeError(
            f"Expected {expected_cells} aggregate cells, "
            f"observed={len(cell_check)}"
        )

    # ------------------------------------------------------------
    # 3. Aggregate across the 10 training seeds
    # ------------------------------------------------------------

    grouped = (
        df.groupby(
            [
                "architecture",
                "budget",
                "category_id",
                "class_name",
            ],
            as_index=False,
            sort=False,
        )
        .agg(
            n_seeds=("seed_index", "nunique"),
            mean_count=("accepted_pseudo_count", "mean"),
            sd_count=("accepted_pseudo_count", "std"),
            median_count=("accepted_pseudo_count", "median"),
            q1_count=("accepted_pseudo_count", q1),
            q3_count=("accepted_pseudo_count", q3),
            min_count=("accepted_pseudo_count", "min"),
            max_count=("accepted_pseudo_count", "max"),
            zero_seed_count=(
                "accepted_pseudo_count",
                lambda x: int((x == 0).sum()),
            ),
        )
    )

    # Explicit IQR width as a numeric audit field.
    grouped["iqr_count"] = (
        grouped["q3_count"]
        - grouped["q1_count"]
    )

    # Percentage is useful diagnostically but zero_seed_count remains
    # the reviewer-facing requested field.
    grouped["zero_seed_pct"] = (
        grouped["zero_seed_count"]
        / grouped["n_seeds"]
        * 100.0
    )

    # ------------------------------------------------------------
    # 4. Reviewer-ready formatted fields
    # ------------------------------------------------------------

    grouped["mean_sd"] = grouped.apply(
        lambda r: (
            f"{fmt_number(r['mean_count'])} "
            f"± {fmt_number(r['sd_count'])}"
        ),
        axis=1,
    )

    grouped["median_iqr"] = grouped.apply(
        lambda r: (
            f"{fmt_number(r['median_count'])} "
            f"[{fmt_number(r['q1_count'])}, "
            f"{fmt_number(r['q3_count'])}]"
        ),
        axis=1,
    )

    grouped["min_max"] = grouped.apply(
        lambda r: (
            f"{int(r['min_count'])}–"
            f"{int(r['max_count'])}"
        ),
        axis=1,
    )

    grouped["zero_seeds"] = grouped.apply(
        lambda r: (
            f"{int(r['zero_seed_count'])}/"
            f"{int(r['n_seeds'])}"
        ),
        axis=1,
    )

    grouped["classification_threshold"] = (
        EXPECTED_THRESHOLD
    )
    grouped["acceptance_comparator"] = (
        EXPECTED_COMPARATOR
    )

    # ------------------------------------------------------------
    # 5. Stable scientific ordering
    # ------------------------------------------------------------

    grouped["_arch_order"] = (
        grouped["architecture"].map(ARCH_ORDER)
    )
    grouped["_budget_order"] = (
        grouped["budget"].map(BUDGET_ORDER)
    )

    grouped = grouped.sort_values(
        [
            "_arch_order",
            "_budget_order",
            "category_id",
        ]
    ).reset_index(drop=True)

    grouped = grouped.drop(
        columns=[
            "_arch_order",
            "_budget_order",
        ]
    )

    # ------------------------------------------------------------
    # 6. Final column order
    # ------------------------------------------------------------

    output_columns = [
        "architecture",
        "budget",
        "category_id",
        "class_name",
        "n_seeds",

        # Reviewer-ready columns
        "mean_sd",
        "median_iqr",
        "min_max",
        "zero_seeds",

        # Numeric audit columns
        "mean_count",
        "sd_count",
        "median_count",
        "q1_count",
        "q3_count",
        "iqr_count",
        "min_count",
        "max_count",
        "zero_seed_count",
        "zero_seed_pct",

        # Frozen acceptance semantics
        "classification_threshold",
        "acceptance_comparator",
    ]

    grouped = grouped[output_columns]

    grouped.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_MINIMAL,
    )

    # ------------------------------------------------------------
    # 7. Final validation
    # ------------------------------------------------------------

    expected_summary_rows = expected_cells

    if len(grouped) != expected_summary_rows:
        raise RuntimeError(
            f"Expected {expected_summary_rows} output rows, "
            f"observed={len(grouped)}"
        )

    if not (grouped["n_seeds"] == 10).all():
        raise RuntimeError(
            "Output contains cell(s) with n_seeds != 10"
        )

    print("STATUS=PASS")
    print(f"INPUT_ROWS={len(df)}")
    print(f"UNIQUE_RUNS={unique_runs}")
    print(
        f"AGGREGATE_ROWS={len(grouped)}"
    )
    print(
        "EXPECTED_AGGREGATE_ROWS="
        f"{2 * 4 * 14}"
    )
    print(
        "CELLS_WITH_ALL_10_ZERO="
        f"{int((grouped['zero_seed_count'] == 10).sum())}"
    )
    print(
        "CELLS_WITH_ANY_NONZERO="
        f"{int((grouped['zero_seed_count'] < 10).sum())}"
    )
    print(f"OUTPUT={OUTPUT_CSV}")


if __name__ == "__main__":
    main()
