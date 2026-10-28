from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(r"D:\ssl_detection_xray_v2")

INPUT_CSV = (
    REPO
    / "analysis"
    / "pseudo_pr_diagnostics"
    / "pseudo_pr_seed_class_iou.csv"
)

OUTPUT_DIR = (
    REPO
    / "analysis"
    / "pseudo_pr_diagnostics"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "pseudo_pr_arch_budget_class_iou_summary.csv"
)


EXPECTED_RUNS = 80
EXPECTED_CLASSES = 14
EXPECTED_SEEDS_PER_CELL = 10
EXPECTED_IOUS = [
    0.50, 0.55, 0.60, 0.65, 0.70,
    0.75, 0.80, 0.85, 0.90, 0.95,
]

EXPECTED_INPUT_ROWS = (
    EXPECTED_RUNS
    * EXPECTED_CLASSES
    * len(EXPECTED_IOUS)
)

EXPECTED_OUTPUT_ROWS = (
    2
    * 4
    * EXPECTED_CLASSES
    * len(EXPECTED_IOUS)
)

EXPECTED_THRESHOLD = 0.90
EXPECTED_COMPARATOR = "STRICT_GREATER"
EXPECTED_MAX_DETS = 100
EXPECTED_AREA = "all"
EXPECTED_MATCHING = "COCOeval_bbox_useCats1"


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
    return float(
        x.quantile(
            0.25,
            interpolation="linear",
        )
    )


def q3(x: pd.Series) -> float:
    return float(
        x.quantile(
            0.75,
            interpolation="linear",
        )
    )


def fmt2(value) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.2f}"


def fmt3(value) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.3f}"


def mean_sd(series: pd.Series) -> str:
    x = series.dropna()

    if len(x) == 0:
        return "NA"

    if len(x) == 1:
        return f"{fmt3(x.iloc[0])} ± NA"

    return (
        f"{fmt3(x.mean())} "
        f"± {fmt3(x.std(ddof=1))}"
    )


def median_iqr(series: pd.Series) -> str:
    x = series.dropna()

    if len(x) == 0:
        return "NA"

    med = x.median()
    low = x.quantile(
        0.25,
        interpolation="linear",
    )
    high = x.quantile(
        0.75,
        interpolation="linear",
    )

    return (
        f"{fmt3(med)} "
        f"[{fmt3(low)}, {fmt3(high)}]"
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not INPUT_CSV.is_file():
        raise FileNotFoundError(
            f"Input not found: {INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    required = {
        "run_id",
        "architecture",
        "budget",
        "seed_index",
        "attempt_id",
        "category_id",
        "class_name",
        "iou_threshold",
        "tp",
        "fp",
        "fn",
        "gt_count",
        "prediction_count",
        "precision",
        "recall",
        "classification_threshold",
        "acceptance_comparator",
        "max_dets_per_image",
        "area_range",
        "matching_semantics",
    }

    missing = required.difference(df.columns)

    if missing:
        raise RuntimeError(
            "Missing columns: "
            + ", ".join(sorted(missing))
        )

    # ----------------------------------------------------------
    # 1. Global integrity
    # ----------------------------------------------------------

    if len(df) != EXPECTED_INPUT_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_INPUT_ROWS} input rows, "
            f"observed={len(df)}"
        )

    if df["run_id"].nunique() != EXPECTED_RUNS:
        raise RuntimeError(
            f"Expected {EXPECTED_RUNS} runs, "
            f"observed={df['run_id'].nunique()}"
        )

    if df["category_id"].nunique() != EXPECTED_CLASSES:
        raise RuntimeError(
            f"Expected {EXPECTED_CLASSES} classes"
        )

    observed_ious = sorted(
        round(float(x), 2)
        for x in df["iou_threshold"].unique()
    )

    if observed_ious != EXPECTED_IOUS:
        raise RuntimeError(
            f"IoU grid mismatch: {observed_ious}"
        )

    if set(
        df["classification_threshold"]
        .astype(float)
        .unique()
    ) != {EXPECTED_THRESHOLD}:
        raise RuntimeError(
            "classification_threshold mismatch"
        )

    if set(
        df["acceptance_comparator"]
        .astype(str)
        .unique()
    ) != {EXPECTED_COMPARATOR}:
        raise RuntimeError(
            "acceptance_comparator mismatch"
        )

    if set(
        df["max_dets_per_image"]
        .astype(int)
        .unique()
    ) != {EXPECTED_MAX_DETS}:
        raise RuntimeError(
            "max_dets_per_image mismatch"
        )

    if set(
        df["area_range"]
        .astype(str)
        .unique()
    ) != {EXPECTED_AREA}:
        raise RuntimeError(
            "area_range mismatch"
        )

    if set(
        df["matching_semantics"]
        .astype(str)
        .unique()
    ) != {EXPECTED_MATCHING}:
        raise RuntimeError(
            "matching_semantics mismatch"
        )

    # ----------------------------------------------------------
    # 2. Arithmetic identities
    # ----------------------------------------------------------

    if not (
        df["tp"] + df["fp"]
        == df["prediction_count"]
    ).all():
        raise RuntimeError(
            "TP + FP != prediction_count"
        )

    if not (
        df["tp"] + df["fn"]
        == df["gt_count"]
    ).all():
        raise RuntimeError(
            "TP + FN != gt_count"
        )

    # Precision must be undefined exactly when no predictions.
    no_pred = (
        df["prediction_count"] == 0
    )

    if not df.loc[
        no_pred,
        "precision",
    ].isna().all():
        raise RuntimeError(
            "Precision must be NA when prediction_count=0"
        )

    # Where predictions exist, precision must be numeric.
    if df.loc[
        ~no_pred,
        "precision",
    ].isna().any():
        raise RuntimeError(
            "Precision unexpectedly NA when predictions exist"
        )

    # Recall should be defined because every class has GT.
    if df["recall"].isna().any():
        raise RuntimeError(
            "Recall contains unexpected NA"
        )

    precision_calc = (
        df.loc[~no_pred, "tp"]
        / df.loc[~no_pred, "prediction_count"]
    )

    if not np.allclose(
        df.loc[~no_pred, "precision"],
        precision_calc,
        rtol=0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Precision arithmetic mismatch"
        )

    recall_calc = (
        df["tp"]
        / df["gt_count"]
    )

    if not np.allclose(
        df["recall"],
        recall_calc,
        rtol=0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Recall arithmetic mismatch"
        )

    # ----------------------------------------------------------
    # 3. Verify each aggregate cell has exactly 10 seeds
    # ----------------------------------------------------------

    group_keys = [
        "architecture",
        "budget",
        "category_id",
        "class_name",
        "iou_threshold",
    ]

    cell_check = (
        df.groupby(
            group_keys,
            dropna=False,
        )
        .agg(
            n_rows=("run_id", "size"),
            n_runs=("run_id", "nunique"),
            n_seeds=("seed_index", "nunique"),
        )
        .reset_index()
    )

    bad = cell_check[
        (cell_check["n_rows"] != 10)
        | (cell_check["n_runs"] != 10)
        | (cell_check["n_seeds"] != 10)
    ]

    if not bad.empty:
        raise RuntimeError(
            "Some aggregate cells do not contain "
            "exactly 10 seeds/runs:\n"
            + bad.to_string(index=False)
        )

    if len(cell_check) != EXPECTED_OUTPUT_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_OUTPUT_ROWS} cells, "
            f"observed={len(cell_check)}"
        )

    # ----------------------------------------------------------
    # 4. Aggregate
    # ----------------------------------------------------------

    rows = []

    for keys, g in df.groupby(
        group_keys,
        sort=False,
        dropna=False,
    ):
        (
            architecture,
            budget,
            category_id,
            class_name,
            iou_threshold,
        ) = keys

        p = g["precision"]
        r = g["recall"]

        p_valid = p.dropna()
        r_valid = r.dropna()

        row = {
            "architecture": architecture,
            "budget": budget,
            "category_id": int(category_id),
            "class_name": class_name,
            "iou_threshold": round(
                float(iou_threshold),
                2,
            ),

            "n_seeds": int(
                g["seed_index"].nunique()
            ),

            # --------------------------------------------------
            # Count summaries across seeds
            # --------------------------------------------------
            "tp_mean": float(g["tp"].mean()),
            "tp_sd": float(g["tp"].std(ddof=1)),
            "fp_mean": float(g["fp"].mean()),
            "fp_sd": float(g["fp"].std(ddof=1)),
            "fn_mean": float(g["fn"].mean()),
            "fn_sd": float(g["fn"].std(ddof=1)),
            "prediction_count_mean": float(
                g["prediction_count"].mean()
            ),
            "prediction_count_sd": float(
                g["prediction_count"].std(ddof=1)
            ),

            # --------------------------------------------------
            # Precision
            # --------------------------------------------------
            "precision_n_defined": int(
                p_valid.shape[0]
            ),
            "precision_mean": (
                float(p_valid.mean())
                if len(p_valid)
                else np.nan
            ),
            "precision_sd": (
                float(p_valid.std(ddof=1))
                if len(p_valid) > 1
                else np.nan
            ),
            "precision_median": (
                float(p_valid.median())
                if len(p_valid)
                else np.nan
            ),
            "precision_q1": (
                q1(p_valid)
                if len(p_valid)
                else np.nan
            ),
            "precision_q3": (
                q3(p_valid)
                if len(p_valid)
                else np.nan
            ),
            "precision_min": (
                float(p_valid.min())
                if len(p_valid)
                else np.nan
            ),
            "precision_max": (
                float(p_valid.max())
                if len(p_valid)
                else np.nan
            ),

            # --------------------------------------------------
            # Recall
            # --------------------------------------------------
            "recall_mean": float(
                r_valid.mean()
            ),
            "recall_sd": float(
                r_valid.std(ddof=1)
            ),
            "recall_median": float(
                r_valid.median()
            ),
            "recall_q1": q1(r_valid),
            "recall_q3": q3(r_valid),
            "recall_min": float(
                r_valid.min()
            ),
            "recall_max": float(
                r_valid.max()
            ),

            # --------------------------------------------------
            # Zero / undefined seeds
            # --------------------------------------------------
            "zero_prediction_seed_count": int(
                (g["prediction_count"] == 0).sum()
            ),
            "undefined_precision_seed_count": int(
                g["precision"].isna().sum()
            ),
            "zero_tp_seed_count": int(
                (g["tp"] == 0).sum()
            ),

            # GT should be invariant across seeds.
            "gt_count": int(
                g["gt_count"].iloc[0]
            ),

            # Locked semantics
            "classification_threshold":
                EXPECTED_THRESHOLD,
            "acceptance_comparator":
                EXPECTED_COMPARATOR,
            "max_dets_per_image":
                EXPECTED_MAX_DETS,
            "area_range":
                EXPECTED_AREA,
            "matching_semantics":
                EXPECTED_MATCHING,
        }

        # ------------------------------------------------------
        # Reviewer-ready formatted fields
        # ------------------------------------------------------

        row["tp_mean_sd"] = (
            f"{row['tp_mean']:.2f} "
            f"± {row['tp_sd']:.2f}"
        )

        row["fp_mean_sd"] = (
            f"{row['fp_mean']:.2f} "
            f"± {row['fp_sd']:.2f}"
        )

        row["fn_mean_sd"] = (
            f"{row['fn_mean']:.2f} "
            f"± {row['fn_sd']:.2f}"
        )

        row["precision_mean_sd"] = (
            mean_sd(p)
        )

        row["precision_median_iqr"] = (
            median_iqr(p)
        )

        row["recall_mean_sd"] = (
            mean_sd(r)
        )

        row["recall_median_iqr"] = (
            median_iqr(r)
        )

        row["zero_prediction_seeds"] = (
            f"{row['zero_prediction_seed_count']}"
            f"/{row['n_seeds']}"
        )

        row["undefined_precision_seeds"] = (
            f"{row['undefined_precision_seed_count']}"
            f"/{row['n_seeds']}"
        )

        rows.append(row)

    out = pd.DataFrame(rows)

    # ----------------------------------------------------------
    # 5. Additional audit checks
    # ----------------------------------------------------------

    if len(out) != EXPECTED_OUTPUT_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_OUTPUT_ROWS} output rows, "
            f"observed={len(out)}"
        )

    if not (
        out["zero_prediction_seed_count"]
        == out["undefined_precision_seed_count"]
    ).all():
        raise RuntimeError(
            "Zero-prediction seeds must equal "
            "undefined-precision seeds"
        )

    # GT count must be invariant for same class across everything.
    gt_check = (
        out.groupby(
            ["category_id", "class_name"]
        )["gt_count"]
        .nunique()
    )

    if not (gt_check == 1).all():
        raise RuntimeError(
            "GT count varies unexpectedly within class"
        )

    # ----------------------------------------------------------
    # 6. Stable order
    # ----------------------------------------------------------

    out["_arch_order"] = (
        out["architecture"].map(
            ARCH_ORDER
        )
    )

    out["_budget_order"] = (
        out["budget"].map(
            BUDGET_ORDER
        )
    )

    out = out.sort_values(
        [
            "_arch_order",
            "_budget_order",
            "category_id",
            "iou_threshold",
        ]
    ).reset_index(drop=True)

    out = out.drop(
        columns=[
            "_arch_order",
            "_budget_order",
        ]
    )

    # ----------------------------------------------------------
    # 7. Final column order
    # ----------------------------------------------------------

    columns = [
        "architecture",
        "budget",
        "category_id",
        "class_name",
        "iou_threshold",
        "n_seeds",
        "gt_count",

        # Reviewer-facing
        "tp_mean_sd",
        "fp_mean_sd",
        "fn_mean_sd",
        "precision_mean_sd",
        "precision_median_iqr",
        "recall_mean_sd",
        "recall_median_iqr",
        "zero_prediction_seeds",
        "undefined_precision_seeds",

        # Numeric counts
        "tp_mean",
        "tp_sd",
        "fp_mean",
        "fp_sd",
        "fn_mean",
        "fn_sd",
        "prediction_count_mean",
        "prediction_count_sd",

        # Numeric precision
        "precision_n_defined",
        "precision_mean",
        "precision_sd",
        "precision_median",
        "precision_q1",
        "precision_q3",
        "precision_min",
        "precision_max",

        # Numeric recall
        "recall_mean",
        "recall_sd",
        "recall_median",
        "recall_q1",
        "recall_q3",
        "recall_min",
        "recall_max",

        # Diagnostics
        "zero_prediction_seed_count",
        "undefined_precision_seed_count",
        "zero_tp_seed_count",

        # Locked semantics
        "classification_threshold",
        "acceptance_comparator",
        "max_dets_per_image",
        "area_range",
        "matching_semantics",
    ]

    out = out[columns]

    out.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_MINIMAL,
    )

    # ----------------------------------------------------------
    # 8. Final console audit
    # ----------------------------------------------------------

    print("STATUS=PASS")
    print(
        f"INPUT_ROWS={len(df)}"
    )
    print(
        f"UNIQUE_RUNS={df['run_id'].nunique()}"
    )
    print(
        f"AGGREGATE_ROWS={len(out)}"
    )
    print(
        f"EXPECTED_AGGREGATE_ROWS="
        f"{EXPECTED_OUTPUT_ROWS}"
    )
    print(
        "CELLS_ALL_10_ZERO_PRED="
        f"{int((out['zero_prediction_seed_count'] == 10).sum())}"
    )
    print(
        "CELLS_ANY_PRED="
        f"{int((out['zero_prediction_seed_count'] < 10).sum())}"
    )
    print(
        "CELLS_PRECISION_UNDEFINED_ALL_SEEDS="
        f"{int((out['precision_n_defined'] == 0).sum())}"
    )
    print(
        f"OUTPUT={OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()
