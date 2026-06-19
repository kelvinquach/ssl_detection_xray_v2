#!/usr/bin/env python3
"""Phase 1A — VinBigData Chest X-ray dataset overview.

Reads ONLY the annotation CSV and produces statistical summaries about the
dataset: class distribution, image-level labels, and bounding-box quality.

Scope guardrails (Phase 1A): this script does NOT
  - create train/val/test splits
  - convert to COCO
  - copy or read images (no DICOM/PNG access)
  - train any model, generate pseudo-labels, or tune thresholds
  - touch the test set

"No Finding" is treated as a NEGATIVE image label, NOT a detection class.

Usage:
    python scripts/01A_dataset_overview.py \
        --train-csv data/raw/vinbigdata/annotations/train.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    print(
        "ERROR: pandas is required for this script but could not be imported.\n"
        f"       Underlying error: {exc!r}\n"
        "       Install it (see requirements.txt) and try again.",
        file=sys.stderr,
    )
    raise SystemExit(2)


# --- "No Finding" handling ------------------------------------------------

# Case-insensitive set of labels treated as the negative (no-abnormality) class.
NO_FINDING_LABELS = {"no finding"}

# Required and optional columns in the VinBigData annotation CSV.
REQUIRED_COLUMNS = ["image_id", "class_name"]
BBOX_COLUMNS = ["x_min", "y_min", "x_max", "y_max"]
OPTIONAL_COLUMNS = ["class_id"] + BBOX_COLUMNS


def is_no_finding(label: Any) -> bool:
    """Return True if `label` is a No Finding negative label (case-insensitive)."""
    if label is None or (isinstance(label, float) and math.isnan(label)):
        return False
    return str(label).strip().lower() in NO_FINDING_LABELS


# --- CLI ------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1A — VinBigData dataset overview (CSV only).",
    )
    parser.add_argument(
        "--train-csv",
        required=True,
        type=str,
        help="Path to the VinBigData annotation CSV (e.g. train.csv).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="reports/phase1A_dataset_overview.json",
        help="Path to the JSON overview output.",
    )
    parser.add_argument(
        "--class-csv",
        type=str,
        default="reports/phase1A_class_distribution.csv",
        help="Path to the per-class distribution CSV.",
    )
    parser.add_argument(
        "--image-summary-csv",
        type=str,
        default="reports/phase1A_image_level_summary.csv",
        help="Path to the per-image summary CSV.",
    )
    parser.add_argument(
        "--bbox-quality-csv",
        type=str,
        default="reports/phase1A_bbox_quality_summary.csv",
        help="Path to the bbox-quality summary CSV.",
    )
    parser.add_argument(
        "--report-md",
        type=str,
        default="reports/phase1A_dataset_overview.md",
        help="Path to the human-readable Markdown report.",
    )
    return parser.parse_args(argv)


# --- IO helpers -----------------------------------------------------------


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_annotations(csv_path: str) -> pd.DataFrame:
    """Load the annotation CSV, validating required columns explicitly."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Annotation CSV not found: {path}\n"
            "       Check the --train-csv path. (Phase 1A reads ONLY this CSV.)"
        )
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(
            f"Failed to read CSV '{path}': {exc!r}\n"
            "       Ensure it is a valid comma-separated VinBigData annotation file."
        ) from exc

    if df.empty:
        raise ValueError(f"Annotation CSV '{path}' is empty (0 rows).")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required column(s) in annotation CSV: "
            f"{missing}\n"
            f"       Found columns: {list(df.columns)}\n"
            f"       Required: {REQUIRED_COLUMNS}; "
            f"optional: {OPTIONAL_COLUMNS}"
        )
    return df


def detect_columns(df: pd.DataFrame) -> Dict[str, bool]:
    """Report which optional/bbox columns are present."""
    present = {col: (col in df.columns) for col in OPTIONAL_COLUMNS}
    present["class_id"] = "class_id" in df.columns
    present["has_all_bbox_columns"] = all(c in df.columns for c in BBOX_COLUMNS)
    return present


# --- Core analysis --------------------------------------------------------


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def compute_bbox_quality(
    df: pd.DataFrame, has_bbox: bool
) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    """Compute bbox quality stats. Returns (summary_dict, valid_bbox_df_or_None)."""
    quality: Dict[str, Any] = {
        "has_bbox_columns": has_bbox,
        "bbox_rows_considered": 0,
        "missing_coordinate": 0,
        "x_min_ge_x_max": 0,
        "y_min_ge_y_max": 0,
        "nonpositive_width_or_height": 0,
        "valid_bbox_count": 0,
        "dimensions": None,
    }
    if not has_bbox:
        return quality, None

    # Only rows that are abnormal (have a class that is not No Finding) are
    # expected to carry coordinates. We still inspect all rows that have any
    # bbox value present.
    work = df.copy()
    for col in BBOX_COLUMNS:
        work[col] = _to_numeric(work[col])

    # A row "carries bbox info" if at least one coordinate is non-null.
    any_coord = work[BBOX_COLUMNS].notna().any(axis=1)
    bbox_rows = work[any_coord]
    quality["bbox_rows_considered"] = int(len(bbox_rows))

    # Missing coordinate = carries some coord but not all four.
    all_coord = work[BBOX_COLUMNS].notna().all(axis=1)
    missing_mask = any_coord & ~all_coord
    quality["missing_coordinate"] = int(missing_mask.sum())

    # Rows with all four coords present: validate geometry.
    full = work[all_coord].copy()
    if not full.empty:
        x_bad = full["x_min"] >= full["x_max"]
        y_bad = full["y_min"] >= full["y_max"]
        width = full["x_max"] - full["x_min"]
        height = full["y_max"] - full["y_min"]
        nonpos = (width <= 0) | (height <= 0)

        quality["x_min_ge_x_max"] = int(x_bad.sum())
        quality["y_min_ge_y_max"] = int(y_bad.sum())
        quality["nonpositive_width_or_height"] = int(nonpos.sum())

        valid_mask = (~x_bad) & (~y_bad) & (~nonpos)
        valid = full[valid_mask].copy()
        quality["valid_bbox_count"] = int(len(valid))

        if not valid.empty:
            valid["_w"] = valid["x_max"] - valid["x_min"]
            valid["_h"] = valid["y_max"] - valid["y_min"]
            valid["_area"] = valid["_w"] * valid["_h"]
            quality["dimensions"] = {
                "width": {
                    "min": float(valid["_w"].min()),
                    "mean": float(valid["_w"].mean()),
                    "max": float(valid["_w"].max()),
                },
                "height": {
                    "min": float(valid["_h"].min()),
                    "mean": float(valid["_h"].mean()),
                    "max": float(valid["_h"].max()),
                },
                "area": {
                    "min": float(valid["_area"].min()),
                    "mean": float(valid["_area"].mean()),
                    "max": float(valid["_area"].max()),
                },
            }
            return quality, valid
    return quality, None


def compute_overview(
    df: pd.DataFrame, columns_present: Dict[str, bool]
) -> Dict[str, Any]:
    """Compute the full statistical overview."""
    has_class_id = columns_present.get("class_id", False)
    has_bbox = columns_present.get("has_all_bbox_columns", False)

    # Normalize a helper boolean column for No Finding.
    df = df.copy()
    df["_is_no_finding"] = df["class_name"].apply(is_no_finding)

    total_rows = int(len(df))
    unique_images = int(df["image_id"].nunique())

    class_names = sorted(df["class_name"].dropna().astype(str).unique().tolist())
    class_ids: Optional[List[int]] = None
    if has_class_id:
        class_ids = sorted(
            _to_numeric(df["class_id"]).dropna().astype(int).unique().tolist()
        )

    rows_per_class = (
        df["class_name"].astype(str).value_counts().sort_index().to_dict()
    )
    rows_per_class = {str(k): int(v) for k, v in rows_per_class.items()}

    images_per_class = (
        df.groupby(df["class_name"].astype(str))["image_id"]
        .nunique()
        .sort_index()
        .to_dict()
    )
    images_per_class = {str(k): int(v) for k, v in images_per_class.items()}

    # Abnormal = not No Finding.
    abnormal_df = df[~df["_is_no_finding"]]
    bbox_per_abnormal_class = (
        abnormal_df["class_name"].astype(str).value_counts().sort_index().to_dict()
    )
    bbox_per_abnormal_class = {
        str(k): int(v) for k, v in bbox_per_abnormal_class.items()
    }

    no_finding_rows = int(df["_is_no_finding"].sum())
    no_finding_images = int(df[df["_is_no_finding"]]["image_id"].nunique())

    abnormal_image_ids = set(abnormal_df["image_id"].unique())
    no_finding_image_ids = set(df[df["_is_no_finding"]]["image_id"].unique())
    abnormal_images = int(len(abnormal_image_ids))

    # Images that carry BOTH a No Finding and an abnormal label.
    both_image_ids = sorted(abnormal_image_ids & no_finding_image_ids)
    images_with_both = int(len(both_image_ids))

    overview: Dict[str, Any] = {
        "total_rows": total_rows,
        "unique_images": unique_images,
        "class_names": class_names,
        "class_ids": class_ids,
        "num_abnormal_classes_excluding_no_finding": int(
            len([c for c in class_names if not is_no_finding(c)])
        ),
        "rows_per_class_name": rows_per_class,
        "images_per_class_name": images_per_class,
        "bbox_per_abnormal_class": bbox_per_abnormal_class,
        "no_finding_rows": no_finding_rows,
        "no_finding_images": no_finding_images,
        "abnormal_images": abnormal_images,
        "images_with_both_no_finding_and_abnormal": images_with_both,
        "_both_image_ids_sample": both_image_ids[:20],
    }
    return overview


def compute_no_finding_policy_check(
    df: pd.DataFrame, has_bbox: bool
) -> Tuple[Dict[str, Any], List[str]]:
    """Apply No Finding policy and collect warnings."""
    warnings: List[str] = []
    df = df.copy()
    df["_is_no_finding"] = df["class_name"].apply(is_no_finding)

    check: Dict[str, Any] = {
        "no_finding_labels_recognized": sorted(NO_FINDING_LABELS),
        "no_finding_excluded_from_detection_classes": True,
        "no_finding_rows_with_bbox": 0,
        "abnormal_rows_missing_bbox": 0,
        "images_with_both_no_finding_and_abnormal": 0,
    }

    if has_bbox:
        work = df.copy()
        for col in BBOX_COLUMNS:
            work[col] = _to_numeric(work[col])
        all_coord = work[BBOX_COLUMNS].notna().all(axis=1)

        # 1) No Finding rows that nonetheless carry bbox coordinates.
        nf_with_bbox = int((work["_is_no_finding"] & all_coord).sum())
        check["no_finding_rows_with_bbox"] = nf_with_bbox
        if nf_with_bbox > 0:
            warnings.append(
                f"{nf_with_bbox} 'No Finding' row(s) have non-null bbox "
                "coordinates; these should be negative labels without boxes."
            )

        # 2) Abnormal rows missing bbox coordinates.
        abn_missing = int(((~work["_is_no_finding"]) & (~all_coord)).sum())
        check["abnormal_rows_missing_bbox"] = abn_missing
        if abn_missing > 0:
            warnings.append(
                f"{abn_missing} abnormal row(s) are missing one or more bbox "
                "coordinates."
            )
    else:
        warnings.append(
            "Bounding-box columns (x_min, y_min, x_max, y_max) not all present; "
            "skipped bbox-based No Finding policy checks."
        )

    # 3) Images with both No Finding and abnormal labels.
    abnormal_ids = set(df[~df["_is_no_finding"]]["image_id"].unique())
    nf_ids = set(df[df["_is_no_finding"]]["image_id"].unique())
    both = sorted(abnormal_ids & nf_ids)
    check["images_with_both_no_finding_and_abnormal"] = len(both)
    if both:
        warnings.append(
            f"{len(both)} image(s) carry BOTH a 'No Finding' and an abnormal "
            f"label. Sample image_ids: {both[:10]}"
        )

    return check, warnings


# --- Output writers -------------------------------------------------------


def write_class_distribution_csv(
    path: str, overview: Dict[str, Any]
) -> None:
    rows = []
    for cname in overview["class_names"]:
        rows.append(
            {
                "class_name": cname,
                "is_no_finding": is_no_finding(cname),
                "rows": overview["rows_per_class_name"].get(cname, 0),
                "images": overview["images_per_class_name"].get(cname, 0),
            }
        )
    out = ensure_parent(path)
    pd.DataFrame(rows).to_csv(out, index=False)


def write_image_level_summary_csv(path: str, df: pd.DataFrame) -> None:
    work = df.copy()
    work["_is_no_finding"] = work["class_name"].apply(is_no_finding)
    grouped = work.groupby("image_id").agg(
        total_rows=("class_name", "size"),
        no_finding_rows=("_is_no_finding", "sum"),
    )
    grouped["abnormal_rows"] = grouped["total_rows"] - grouped["no_finding_rows"]
    grouped["has_no_finding"] = grouped["no_finding_rows"] > 0
    grouped["has_abnormal"] = grouped["abnormal_rows"] > 0
    grouped["label_type"] = grouped.apply(
        lambda r: (
            "both"
            if r["has_no_finding"] and r["has_abnormal"]
            else "no_finding"
            if r["has_no_finding"]
            else "abnormal"
        ),
        axis=1,
    )
    out = ensure_parent(path)
    grouped.reset_index().to_csv(out, index=False)


def write_bbox_quality_csv(path: str, bbox_quality: Dict[str, Any]) -> None:
    flat = {
        "has_bbox_columns": bbox_quality["has_bbox_columns"],
        "bbox_rows_considered": bbox_quality["bbox_rows_considered"],
        "missing_coordinate": bbox_quality["missing_coordinate"],
        "x_min_ge_x_max": bbox_quality["x_min_ge_x_max"],
        "y_min_ge_y_max": bbox_quality["y_min_ge_y_max"],
        "nonpositive_width_or_height": bbox_quality[
            "nonpositive_width_or_height"
        ],
        "valid_bbox_count": bbox_quality["valid_bbox_count"],
    }
    dims = bbox_quality.get("dimensions")
    if dims:
        for key in ("width", "height", "area"):
            flat[f"{key}_min"] = dims[key]["min"]
            flat[f"{key}_mean"] = dims[key]["mean"]
            flat[f"{key}_max"] = dims[key]["max"]
    out = ensure_parent(path)
    pd.DataFrame([flat]).to_csv(out, index=False)


def write_markdown_report(
    path: str,
    csv_path: str,
    overview: Dict[str, Any],
    nf_check: Dict[str, Any],
    bbox_quality: Dict[str, Any],
    warnings: List[str],
    generated_files: Dict[str, str],
) -> None:
    lines: List[str] = []
    lines.append("# Phase 1A — VinBigData Dataset Overview")
    lines.append("")
    lines.append(
        f"_Generated {datetime.now(timezone.utc).isoformat()} from "
        f"`{csv_path}`._"
    )
    lines.append("")
    lines.append(
        "> Scope: CSV-only statistics. No split, no COCO, no image reads, "
        "no training. 'No Finding' is a negative image label, not a "
        "detection class."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total rows: **{overview['total_rows']}**")
    lines.append(f"- Unique images: **{overview['unique_images']}**")
    lines.append(f"- Abnormal images: **{overview['abnormal_images']}**")
    lines.append(f"- No Finding images: **{overview['no_finding_images']}**")
    lines.append(
        "- Abnormal classes (excluding No Finding): "
        f"**{overview['num_abnormal_classes_excluding_no_finding']}**"
    )
    invalid = (
        bbox_quality["x_min_ge_x_max"]
        + bbox_quality["y_min_ge_y_max"]
        + bbox_quality["nonpositive_width_or_height"]
        + bbox_quality["missing_coordinate"]
    )
    lines.append(f"- Bbox invalid count (total flags): **{invalid}**")
    lines.append("")

    lines.append("## Class distribution")
    lines.append("")
    lines.append("| class_name | No Finding? | rows | images |")
    lines.append("|---|---|---|---|")
    for cname in overview["class_names"]:
        lines.append(
            f"| {cname} | {is_no_finding(cname)} | "
            f"{overview['rows_per_class_name'].get(cname, 0)} | "
            f"{overview['images_per_class_name'].get(cname, 0)} |"
        )
    lines.append("")

    lines.append("## Bbox quality")
    lines.append("")
    lines.append(f"- Has bbox columns: {bbox_quality['has_bbox_columns']}")
    lines.append(
        f"- Bbox rows considered: {bbox_quality['bbox_rows_considered']}"
    )
    lines.append(f"- Missing coordinate: {bbox_quality['missing_coordinate']}")
    lines.append(f"- x_min >= x_max: {bbox_quality['x_min_ge_x_max']}")
    lines.append(f"- y_min >= y_max: {bbox_quality['y_min_ge_y_max']}")
    lines.append(
        "- Non-positive width/height: "
        f"{bbox_quality['nonpositive_width_or_height']}"
    )
    lines.append(f"- Valid bboxes: {bbox_quality['valid_bbox_count']}")
    dims = bbox_quality.get("dimensions")
    if dims:
        lines.append("")
        lines.append("| dim | min | mean | max |")
        lines.append("|---|---|---|---|")
        for key in ("width", "height", "area"):
            lines.append(
                f"| {key} | {dims[key]['min']:.2f} | "
                f"{dims[key]['mean']:.2f} | {dims[key]['max']:.2f} |"
            )
    lines.append("")

    lines.append("## No Finding policy check")
    lines.append("")
    lines.append(
        f"- No Finding labels recognized: {nf_check['no_finding_labels_recognized']}"
    )
    lines.append(
        "- No Finding rows with bbox: "
        f"{nf_check['no_finding_rows_with_bbox']}"
    )
    lines.append(
        "- Abnormal rows missing bbox: "
        f"{nf_check['abnormal_rows_missing_bbox']}"
    )
    lines.append(
        "- Images with both No Finding and abnormal: "
        f"{nf_check['images_with_both_no_finding_and_abnormal']}"
    )
    lines.append("")

    lines.append("## Warnings")
    lines.append("")
    if warnings:
        for w in warnings:
            lines.append(f"- ⚠️ {w}")
    else:
        lines.append("- None.")
    lines.append("")

    lines.append("## Generated files")
    lines.append("")
    for name, fpath in generated_files.items():
        lines.append(f"- `{name}`: `{fpath}`")
    lines.append("")

    out = ensure_parent(path)
    out.write_text("\n".join(lines), encoding="utf-8")


# --- Main -----------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    try:
        df = load_annotations(args.train_csv)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    columns_present = detect_columns(df)
    has_bbox = columns_present["has_all_bbox_columns"]

    overview = compute_overview(df, columns_present)
    bbox_quality, _ = compute_bbox_quality(df, has_bbox)
    nf_check, warnings = compute_no_finding_policy_check(df, has_bbox)

    generated_files = {
        "overview_json": args.output_json,
        "class_distribution_csv": args.class_csv,
        "image_level_summary_csv": args.image_summary_csv,
        "bbox_quality_csv": args.bbox_quality_csv,
        "report_md": args.report_md,
    }

    # Assemble JSON with the requested review-friendly structure.
    report = {
        "report_type": "phase1A_dataset_overview",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(args.train_csv),
        "columns_present": columns_present,
        "summary": overview,
        "no_finding_policy_check": nf_check,
        "bbox_quality": bbox_quality,
        "warnings": warnings,
        "generated_files": generated_files,
    }

    # Write all outputs.
    out_json = ensure_parent(args.output_json)
    out_json.write_text(
        json.dumps(report, indent=2, sort_keys=False), encoding="utf-8"
    )
    write_class_distribution_csv(args.class_csv, overview)
    write_image_level_summary_csv(args.image_summary_csv, df)
    write_bbox_quality_csv(args.bbox_quality_csv, bbox_quality)
    write_markdown_report(
        args.report_md,
        args.train_csv,
        overview,
        nf_check,
        bbox_quality,
        warnings,
        generated_files,
    )

    # Console summary.
    invalid_total = (
        bbox_quality["missing_coordinate"]
        + bbox_quality["x_min_ge_x_max"]
        + bbox_quality["y_min_ge_y_max"]
        + bbox_quality["nonpositive_width_or_height"]
    )
    print("=" * 60)
    print("Phase 1A — VinBigData dataset overview")
    print("=" * 60)
    print(f"Source CSV               : {args.train_csv}")
    print(f"Total rows               : {overview['total_rows']}")
    print(f"Unique images            : {overview['unique_images']}")
    print(f"Abnormal images          : {overview['abnormal_images']}")
    print(f"No Finding images        : {overview['no_finding_images']}")
    print(
        "Abnormal classes (excl. NF): "
        f"{overview['num_abnormal_classes_excluding_no_finding']}"
    )
    print(f"Bbox invalid count       : {invalid_total}")
    print("-" * 60)
    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  ⚠️  {w}")
    else:
        print("Warnings                 : none")
    print("-" * 60)
    print("Generated files:")
    for name, fpath in generated_files.items():
        print(f"  {name:<24} -> {fpath}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
