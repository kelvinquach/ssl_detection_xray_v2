#!/usr/bin/env python3
"""Phase 1B — VinBigData Chest X-ray annotation quality.

Reads ONLY the annotation CSV (full source metadata train.csv) and reports
annotation-quality issues: bbox coordinate sanity, boundary checks (only when
image dimensions exist in the CSV), No Finding policy, abnormal-annotation
consistency, exact/near-duplicate bbox candidates, and class-id<->class_name
mapping integrity.

This script is REPORT-ONLY. It never deletes or edits annotations. It flags
candidates and lets a human / GPT review decide.

Scope guardrails (Phase 1B): this script does NOT
  - split train/val/test
  - convert to COCO
  - create the downstream 4,894 subset
  - copy or read images (no DICOM/PNG access)
  - train, pseudo-label, or tune thresholds
  - touch the test set

"No Finding" is a NEGATIVE image label, NOT a detection class.

Usage:
    python scripts/01B_annotation_quality.py \
        --train-csv data/raw/vinbigdata/annotations/train.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    import pandas as pd
except Exception as exc:  # pragma: no cover
    print(
        "ERROR: pandas and numpy are required but could not be imported.\n"
        f"       Underlying error: {exc!r}\n"
        "       Install them (see requirements.txt) and try again.",
        file=sys.stderr,
    )
    raise SystemExit(2)


# --- Configuration --------------------------------------------------------

NO_FINDING_LABELS = {"no finding"}

REQUIRED_COLUMNS = ["image_id", "class_name"]
BBOX_COLUMNS = ["x_min", "y_min", "x_max", "y_max"]
# Candidate column names that may encode image dimensions.
WIDTH_COL_CANDIDATES = ["image_width", "width", "img_width", "Width"]
HEIGHT_COL_CANDIDATES = ["image_height", "height", "img_height", "Height"]

FORBIDDEN_ACTIONS = [
    "create_split",
    "create_coco_json",
    "create_4894_subset",
    "copy_images",
    "read_dicom_or_png",
    "train_model",
    "pseudo_label",
    "tune_threshold",
    "use_test_set",
    "auto_delete_or_edit_annotation",
]


def is_no_finding(label: Any) -> bool:
    """True if label is a No Finding negative label (case-insensitive)."""
    if label is None:
        return False
    if isinstance(label, float) and np.isnan(label):
        return False
    return str(label).strip().lower() in NO_FINDING_LABELS


# --- CLI ------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1B — VinBigData annotation quality (CSV only).",
    )
    parser.add_argument("--train-csv", required=True, type=str)
    parser.add_argument(
        "--output-json",
        type=str,
        default="reports/phase1B_annotation_quality.json",
    )
    parser.add_argument(
        "--report-md",
        type=str,
        default="reports/phase1B_annotation_quality.md",
    )
    parser.add_argument(
        "--annotation-sanity-md",
        type=str,
        default="reports/annotation_sanity_report.md",
    )
    parser.add_argument(
        "--invalid-bbox-csv",
        type=str,
        default="reports/invalid_bbox_rows.csv",
    )
    parser.add_argument(
        "--duplicate-csv",
        type=str,
        default="reports/duplicate_bbox_candidates.csv",
    )
    parser.add_argument(
        "--class-mapping-csv",
        type=str,
        default="reports/phase1B_class_mapping.csv",
    )
    parser.add_argument(
        "--bbox-quality-by-class-csv",
        type=str,
        default="reports/phase1B_bbox_quality_by_class.csv",
    )
    parser.add_argument(
        "--image-label-consistency-csv",
        type=str,
        default="reports/phase1B_image_label_consistency.csv",
    )
    parser.add_argument(
        "--near-duplicate-iou",
        type=float,
        default=0.95,
    )
    return parser.parse_args(argv)


# --- IO helpers -----------------------------------------------------------


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_annotations(csv_path: str) -> pd.DataFrame:
    """Load annotation CSV, validating required columns explicitly."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Annotation CSV not found: {path}\n"
            "       Check the --train-csv path. (Phase 1B reads ONLY this CSV.)"
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
            f"Missing required column(s): {missing}\n"
            f"       Found columns: {list(df.columns)}\n"
            f"       Required: {REQUIRED_COLUMNS}"
        )
    return df


def detect_dimension_columns(
    df: pd.DataFrame,
) -> Tuple[Optional[str], Optional[str]]:
    """Find image-width / image-height columns if present."""
    w = next((c for c in WIDTH_COL_CANDIDATES if c in df.columns), None)
    h = next((c for c in HEIGHT_COL_CANDIDATES if c in df.columns), None)
    # Only meaningful as a pair.
    if w and h:
        return w, h
    return None, None


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# --- Bbox sanity ----------------------------------------------------------


def analyze_bbox_sanity(
    df: pd.DataFrame,
    has_bbox: bool,
    dim_cols: Tuple[Optional[str], Optional[str]],
) -> Tuple[pd.DataFrame, Dict[str, int], str]:
    """Per-row bbox sanity. Returns (invalid_rows_df, reason_counts, boundary_status).

    `invalid_rows_df` contains the original rows plus a `reasons` column.
    """
    reason_counts: Dict[str, int] = {
        "missing_coordinate_abnormal": 0,
        "non_numeric_coordinate": 0,
        "x_min_negative": 0,
        "y_min_negative": 0,
        "x_max_negative": 0,
        "y_max_negative": 0,
        "x_min_ge_x_max": 0,
        "y_min_ge_y_max": 0,
        "width_le_0": 0,
        "height_le_0": 0,
        "area_le_0": 0,
        "no_finding_with_bbox": 0,
        "x_max_gt_image_width": 0,
        "y_max_gt_image_height": 0,
        "x_min_gt_image_width": 0,
        "y_min_gt_image_height": 0,
    }

    boundary_status = "not_evaluable_without_image_dimensions"
    if not has_bbox:
        empty = df.iloc[0:0].copy()
        empty["reasons"] = []
        return empty, reason_counts, boundary_status

    work = df.copy()
    work["_is_no_finding"] = work["class_name"].apply(is_no_finding)

    # Raw string presence (to detect non-numeric vs truly missing).
    raw = {c: work[c] for c in BBOX_COLUMNS}
    num = {c: to_numeric(work[c]) for c in BBOX_COLUMNS}

    raw_present = pd.DataFrame(
        {c: raw[c].notna() for c in BBOX_COLUMNS}
    )  # has some string/value
    num_present = pd.DataFrame({c: num[c].notna() for c in BBOX_COLUMNS})

    any_raw = raw_present.any(axis=1)
    all_num = num_present.all(axis=1)

    w_col, h_col = dim_cols
    img_w = to_numeric(work[w_col]) if w_col else None
    img_h = to_numeric(work[h_col]) if h_col else None
    if w_col and h_col:
        boundary_status = "evaluated_from_csv_dimensions"

    reasons_per_row: List[List[str]] = [[] for _ in range(len(work))]
    idx = work.index.to_list()

    def flag(mask: pd.Series, reason: str) -> None:
        mask = mask.fillna(False)
        count = int(mask.sum())
        if count:
            reason_counts[reason] += count
            for pos, is_set in enumerate(mask.to_numpy()):
                if is_set:
                    reasons_per_row[pos].append(reason)

    # No Finding rows that carry any bbox value.
    flag(work["_is_no_finding"] & any_raw, "no_finding_with_bbox")

    # Abnormal rows missing one or more coordinates.
    flag((~work["_is_no_finding"]) & (~all_num), "missing_coordinate_abnormal")

    # Non-numeric: a raw value is present but not parseable as number.
    for c in BBOX_COLUMNS:
        non_numeric = raw_present[c] & (~num_present[c])
        flag(non_numeric, "non_numeric_coordinate")

    # Geometry checks only where all four numeric coords exist.
    xmin, ymin, xmax, ymax = (num["x_min"], num["y_min"], num["x_max"], num["y_max"])
    width = xmax - xmin
    height = ymax - ymin
    area = width * height

    flag(all_num & (xmin < 0), "x_min_negative")
    flag(all_num & (ymin < 0), "y_min_negative")
    flag(all_num & (xmax < 0), "x_max_negative")
    flag(all_num & (ymax < 0), "y_max_negative")
    flag(all_num & (xmin >= xmax), "x_min_ge_x_max")
    flag(all_num & (ymin >= ymax), "y_min_ge_y_max")
    flag(all_num & (width <= 0), "width_le_0")
    flag(all_num & (height <= 0), "height_le_0")
    flag(all_num & (area <= 0), "area_le_0")

    # Boundary checks (only if dimensions available).
    if w_col and h_col:
        flag(all_num & (xmax > img_w), "x_max_gt_image_width")
        flag(all_num & (ymax > img_h), "y_max_gt_image_height")
        flag(all_num & (xmin > img_w), "x_min_gt_image_width")
        flag(all_num & (ymin > img_h), "y_min_gt_image_height")

    has_issue = [len(r) > 0 for r in reasons_per_row]
    invalid = work.loc[[idx[i] for i, v in enumerate(has_issue) if v]].copy()
    invalid_reasons = [
        ";".join(reasons_per_row[i]) for i, v in enumerate(has_issue) if v
    ]
    invalid["reasons"] = invalid_reasons
    invalid = invalid.drop(columns=["_is_no_finding"], errors="ignore")

    return invalid, reason_counts, boundary_status


# --- Duplicate / near-duplicate -------------------------------------------


def _iou(a: Tuple[float, float, float, float], b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def analyze_duplicates(
    df: pd.DataFrame,
    has_bbox: bool,
    iou_threshold: float,
    has_rad_id: bool,
) -> Tuple[pd.DataFrame, int, int]:
    """Find exact + near-duplicate bbox candidates within (image_id, class).

    Returns (candidates_df, exact_count, near_count). Comparisons are done only
    within groups sharing image_id and class, keeping cost low even on large CSVs.
    """
    cols_out = ["image_id", "class_name"]
    if "class_id" in df.columns:
        cols_out.append("class_id")
    cols_out += BBOX_COLUMNS
    if has_rad_id:
        cols_out.append("rad_id")

    if not has_bbox:
        return pd.DataFrame(columns=cols_out + ["dup_type", "iou", "group_key"]), 0, 0

    work = df.copy()
    work["_is_no_finding"] = work["class_name"].apply(is_no_finding)
    for c in BBOX_COLUMNS:
        work[c] = to_numeric(work[c])
    # Only abnormal rows with full coordinates participate.
    valid = work[(~work["_is_no_finding"]) & work[BBOX_COLUMNS].notna().all(axis=1)]

    class_key = "class_id" if "class_id" in df.columns else "class_name"

    exact_records: List[Dict[str, Any]] = []
    near_records: List[Dict[str, Any]] = []
    exact_count = 0
    near_count = 0

    for (img, cls), grp in valid.groupby(["image_id", class_key], sort=False):
        if len(grp) < 2:
            continue
        rows = grp.to_dict("records")
        # Exact duplicates via grouping on rounded coords.
        coord_seen: Dict[Tuple[float, float, float, float], int] = {}
        for r in rows:
            key = (r["x_min"], r["y_min"], r["x_max"], r["y_max"])
            coord_seen[key] = coord_seen.get(key, 0) + 1
        for r in rows:
            key = (r["x_min"], r["y_min"], r["x_max"], r["y_max"])
            if coord_seen[key] > 1:
                rec = {k: r.get(k) for k in cols_out}
                rec["dup_type"] = "exact"
                rec["iou"] = 1.0
                rec["group_key"] = f"{img}|{cls}"
                exact_records.append(rec)
                exact_count += 1

        # Near duplicates: pairwise IoU within the group (small groups).
        boxes = [
            (r["x_min"], r["y_min"], r["x_max"], r["y_max"]) for r in rows
        ]
        flagged_near = set()
        for (i, j) in combinations(range(len(rows)), 2):
            # Skip pairs already identical (counted as exact).
            if boxes[i] == boxes[j]:
                continue
            iou = _iou(boxes[i], boxes[j])
            if iou >= iou_threshold:
                for k in (i, j):
                    if k in flagged_near:
                        continue
                    flagged_near.add(k)
                    rec = {col: rows[k].get(col) for col in cols_out}
                    rec["dup_type"] = "near"
                    rec["iou"] = round(float(iou), 4)
                    rec["group_key"] = f"{img}|{cls}"
                    near_records.append(rec)
                    near_count += 1

    candidates = pd.DataFrame(exact_records + near_records, columns=cols_out + ["dup_type", "iou", "group_key"])
    return candidates, exact_count, near_count


# --- Class mapping --------------------------------------------------------


def analyze_class_mapping(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, int, List[str]]:
    """Check class_id <-> class_name is a clean bijection.

    Returns (mapping_df, issue_count, warnings).
    """
    warnings: List[str] = []
    if "class_id" not in df.columns:
        warnings.append(
            "No 'class_id' column present; class mapping bijection not checkable."
        )
        names = sorted(df["class_name"].dropna().astype(str).unique())
        mapping = pd.DataFrame(
            {
                "class_name": names,
                "class_id": [None] * len(names),
                "is_no_finding": [is_no_finding(n) for n in names],
                "issue": [""] * len(names),
            }
        )
        return mapping, 0, warnings

    work = df[["class_name", "class_id"]].copy()
    work["class_name"] = work["class_name"].astype(str)

    issue_count = 0
    records: List[Dict[str, Any]] = []

    # class_id -> set of names
    id_to_names = work.groupby("class_id")["class_name"].unique()
    name_to_ids = work.groupby("class_name")["class_id"].unique()

    for cname, ids in name_to_ids.items():
        ids_clean = [i for i in ids.tolist()]
        issue = ""
        if len(ids_clean) > 1:
            issue = f"class_name maps to multiple class_id: {ids_clean}"
            issue_count += 1
        nf = is_no_finding(cname)
        # one representative id
        rep_id = ids_clean[0] if ids_clean else None
        records.append(
            {
                "class_name": cname,
                "class_id": rep_id,
                "is_no_finding": nf,
                "all_class_ids": ";".join(str(i) for i in ids_clean),
                "issue": issue,
            }
        )

    for cid, names in id_to_names.items():
        names_clean = [str(n) for n in names.tolist()]
        if len(names_clean) > 1:
            issue_count += 1
            for rec in records:
                if rec["class_id"] == cid:
                    extra = f"class_id maps to multiple class_name: {names_clean}"
                    rec["issue"] = (rec["issue"] + "; " + extra).strip("; ")

    mapping = pd.DataFrame(records).sort_values("class_name").reset_index(drop=True)

    # Confirm No Finding not treated as abnormal detection class downstream.
    nf_rows = mapping[mapping["is_no_finding"]]
    if not nf_rows.empty:
        warnings.append(
            "'No Finding' present in class table; it is excluded from abnormal "
            "detection classes by policy (kept here only for mapping integrity)."
        )

    return mapping, issue_count, warnings


# --- Image-level label consistency ----------------------------------------


def analyze_image_label_consistency(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, int]:
    """Per-image label-type summary; flag mixed No Finding + abnormal."""
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
    mixed_count = int((grouped["label_type"] == "both").sum())
    return grouped.reset_index(), mixed_count


# --- bbox quality by class ------------------------------------------------


def analyze_bbox_quality_by_class(
    df: pd.DataFrame, has_bbox: bool
) -> pd.DataFrame:
    """Per-class counts of rows, valid bboxes, and missing-bbox rows."""
    work = df.copy()
    work["_is_no_finding"] = work["class_name"].apply(is_no_finding)
    class_key_cols = ["class_name"]
    if "class_id" in df.columns:
        class_key_cols = ["class_id", "class_name"]

    if has_bbox:
        for c in BBOX_COLUMNS:
            work[c] = to_numeric(work[c])
        work["_all_coords"] = work[BBOX_COLUMNS].notna().all(axis=1)
    else:
        work["_all_coords"] = False

    records: List[Dict[str, Any]] = []
    for key, grp in work.groupby(class_key_cols, sort=True):
        if isinstance(key, tuple):
            keymap = dict(zip(class_key_cols, key))
        else:
            keymap = {class_key_cols[0]: key}
        cname = str(keymap.get("class_name"))
        rec = {
            "class_id": keymap.get("class_id"),
            "class_name": cname,
            "is_no_finding": is_no_finding(cname),
            "rows": int(len(grp)),
            "rows_with_full_bbox": int(grp["_all_coords"].sum()),
            "rows_missing_bbox": int((~grp["_all_coords"]).sum()),
        }
        records.append(rec)
    return pd.DataFrame(records)


# --- Markdown writers -----------------------------------------------------


def write_markdown_report(
    path: str, payload: Dict[str, Any], iou_threshold: float
) -> None:
    s = payload["summary"] if "summary" in payload else payload
    lines: List[str] = []
    lines.append("# Phase 1B — Annotation Quality Report")
    lines.append("")
    lines.append(f"_Generated {payload['created_utc']} from `{payload['train_csv']}`._")
    lines.append("")
    lines.append("## Executive summary")
    lines.append("")
    lines.append(
        f"Analyzed **{payload['total_rows']}** annotation rows across "
        f"**{payload['unique_images']}** images. "
        f"Invalid bbox flags: **{payload['invalid_bbox_total']}**. "
        f"Exact duplicate candidates: **{payload['exact_duplicate_candidate_count']}**, "
        f"near-duplicate candidates (IoU ≥ {iou_threshold}): "
        f"**{payload['near_duplicate_candidate_count']}**. "
        f"Class-mapping issues: **{payload['class_mapping_issue_count']}**."
    )
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "- Reads the **full source metadata `train.csv` only**. It does NOT "
        "build the downstream 4,894-image controlled subset."
    )
    lines.append(
        "- CSV-only: no split, no COCO, no image/DICOM/PNG reads, no training, "
        "no pseudo-labelling, no threshold tuning, no test-set access."
    )
    lines.append("- Report-only: no annotation is deleted or modified.")
    lines.append("")
    lines.append("## Checks performed")
    lines.append("")
    lines.append("- Bbox coordinate sanity (missing, non-numeric, negative, degenerate geometry, non-positive area).")
    lines.append("- Image-boundary checks (only if the CSV carries image dimensions).")
    lines.append("- No Finding policy (negative label, must carry no bbox).")
    lines.append("- Abnormal-annotation completeness (full bbox required).")
    lines.append("- Exact and near-duplicate bbox candidates within (image_id, class).")
    lines.append("- class_id <-> class_name mapping bijection.")
    lines.append("")
    lines.append("## Key findings")
    lines.append("")
    lines.append(f"- Abnormal rows: {payload['abnormal_rows']}; No Finding rows: {payload['no_finding_rows']}.")
    lines.append(f"- Abnormal images: {payload['abnormal_images']}; No Finding images: {payload['no_finding_images']}.")
    lines.append(
        f"- Abnormal detection classes (excl. No Finding): "
        f"{payload['abnormal_detection_classes_excluding_no_finding']}."
    )
    lines.append(f"- No Finding rows carrying bbox: {payload['no_finding_with_bbox_count']}.")
    lines.append(f"- Abnormal rows missing bbox: {payload['abnormal_missing_bbox_count']}.")
    lines.append(f"- Images mixing No Finding + abnormal: {payload['mixed_no_finding_abnormal_image_count']}.")
    lines.append(f"- Negative-coordinate rows: {payload['negative_coordinate_count']}.")
    lines.append(f"- Zero/negative-area rows: {payload['zero_or_negative_area_count']}.")
    lines.append("")
    lines.append("## Invalid bbox summary (by reason)")
    lines.append("")
    lines.append("| reason | count |")
    lines.append("|---|---|")
    for reason, count in payload["invalid_bbox_by_reason"].items():
        lines.append(f"| {reason} | {count} |")
    lines.append("")
    lines.append("## Duplicate / near-duplicate summary")
    lines.append("")
    lines.append(
        f"- Exact duplicate candidates: {payload['exact_duplicate_candidate_count']}."
    )
    lines.append(
        f"- Near-duplicate candidates (IoU ≥ {iou_threshold}): "
        f"{payload['near_duplicate_candidate_count']}."
    )
    lines.append(
        "- **Interpretation:** VinBigData is multi-radiologist. Duplicate and "
        "near-duplicate boxes on the same image+class are very likely "
        "independent annotations from different readers, NOT confirmed errors. "
        "They are recorded as *candidates* for review only."
    )
    lines.append("")
    lines.append("## Class mapping summary")
    lines.append("")
    lines.append(f"- Mapping issues detected: {payload['class_mapping_issue_count']}.")
    lines.append("- See `phase1B_class_mapping.csv` for the full table.")
    lines.append("")
    lines.append("## No Finding policy summary")
    lines.append("")
    lines.append("- 'No Finding' is treated as a negative image label, excluded from detection classes.")
    lines.append(f"- No Finding rows with bbox (policy violation candidates): {payload['no_finding_with_bbox_count']}.")
    lines.append(f"- Images mixing No Finding and abnormal labels: {payload['mixed_no_finding_abnormal_image_count']}.")
    lines.append("")
    lines.append("## Boundary check status")
    lines.append("")
    lines.append(f"- `{payload['boundary_check_status']}`")
    if payload["boundary_check_status"].startswith("not_evaluable"):
        lines.append(
            "- The CSV does not carry image dimensions; boundary checks are "
            "deliberately skipped. No image files are read in Phase 1B."
        )
    lines.append("")
    lines.append("## Research risk interpretation")
    lines.append("")
    lines.append(
        "- Coordinate-sanity violations would directly corrupt detection "
        "targets and must be resolved before any COCO conversion."
    )
    lines.append(
        "- Duplicate/near-duplicate candidates affect how multi-reader boxes "
        "are fused later; they are not necessarily defects."
    )
    lines.append(
        "- Mixed No Finding + abnormal images, if any, would break the "
        "negative-image assumption used for semi-supervised negatives."
    )
    lines.append("")
    lines.append("## Recommended next action")
    lines.append("")
    lines.append(
        "- **Send these outputs to GPT review BEFORE ticking the Phase 1B "
        "checklist.** Do not auto-correct annotations; decisions on duplicate "
        "fusion and any flagged rows are research decisions."
    )
    lines.append("")
    ensure_parent(path).write_text("\n".join(lines), encoding="utf-8")


def write_annotation_sanity_md(
    path: str, payload: Dict[str, Any], invalid_df: pd.DataFrame
) -> None:
    lines: List[str] = []
    lines.append("# Annotation Sanity Report (Phase 1B)")
    lines.append("")
    lines.append(f"_Source: `{payload['train_csv']}` — {payload['created_utc']}._")
    lines.append("")
    lines.append(f"Total invalid bbox rows flagged: **{payload['invalid_bbox_total']}**.")
    lines.append("")
    lines.append("## Counts by reason")
    lines.append("")
    lines.append("| reason | count |")
    lines.append("|---|---|")
    for reason, count in payload["invalid_bbox_by_reason"].items():
        lines.append(f"| {reason} | {count} |")
    lines.append("")
    lines.append("## Sample flagged rows (first 25)")
    lines.append("")
    if invalid_df.empty:
        lines.append("No invalid rows flagged. ✅")
    else:
        sample = invalid_df.head(25)
        cols = [c for c in ["image_id", "class_name", "class_id", *BBOX_COLUMNS, "reasons"] if c in sample.columns]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for _, row in sample.iterrows():
            lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    lines.append("")
    lines.append(
        "_Report-only. No rows were deleted or modified. Review with GPT before action._"
    )
    ensure_parent(path).write_text("\n".join(lines), encoding="utf-8")


# --- Main -----------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    try:
        df = load_annotations(args.train_csv)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    has_bbox = all(c in df.columns for c in BBOX_COLUMNS)
    has_rad_id = "rad_id" in df.columns
    dim_cols = detect_dimension_columns(df)

    warnings: List[str] = []
    if not has_bbox:
        warnings.append(
            "Not all bbox columns (x_min,y_min,x_max,y_max) present; "
            "bbox-geometry and duplicate checks limited."
        )

    # Core counts.
    df_nf = df["class_name"].apply(is_no_finding)
    total_rows = int(len(df))
    unique_images = int(df["image_id"].nunique())
    abnormal_rows = int((~df_nf).sum())
    no_finding_rows = int(df_nf.sum())
    abnormal_images = int(df[~df_nf]["image_id"].nunique())
    no_finding_images = int(df[df_nf]["image_id"].nunique())
    abnormal_classes = sorted(
        c for c in df["class_name"].astype(str).unique() if not is_no_finding(c)
    )

    # Analyses.
    invalid_df, reason_counts, boundary_status = analyze_bbox_sanity(
        df, has_bbox, dim_cols
    )
    dup_df, exact_count, near_count = analyze_duplicates(
        df, has_bbox, args.near_duplicate_iou, has_rad_id
    )
    mapping_df, mapping_issues, mapping_warnings = analyze_class_mapping(df)
    consistency_df, mixed_count = analyze_image_label_consistency(df)
    bbox_by_class_df = analyze_bbox_quality_by_class(df, has_bbox)

    warnings.extend(mapping_warnings)

    invalid_total = int(len(invalid_df))
    negative_coord = (
        reason_counts["x_min_negative"]
        + reason_counts["y_min_negative"]
        + reason_counts["x_max_negative"]
        + reason_counts["y_max_negative"]
    )
    zero_neg_area = reason_counts["area_le_0"]
    no_finding_with_bbox = reason_counts["no_finding_with_bbox"]
    abnormal_missing_bbox = reason_counts["missing_coordinate_abnormal"]

    if no_finding_with_bbox:
        warnings.append(
            f"{no_finding_with_bbox} No Finding row(s) carry bbox coordinates."
        )
    if abnormal_missing_bbox:
        warnings.append(
            f"{abnormal_missing_bbox} abnormal row(s) are missing bbox coordinates."
        )
    if mixed_count:
        warnings.append(
            f"{mixed_count} image(s) carry both No Finding and abnormal labels."
        )

    generated_files = {
        "output_json": args.output_json,
        "report_md": args.report_md,
        "annotation_sanity_md": args.annotation_sanity_md,
        "invalid_bbox_csv": args.invalid_bbox_csv,
        "duplicate_csv": args.duplicate_csv,
        "class_mapping_csv": args.class_mapping_csv,
        "bbox_quality_by_class_csv": args.bbox_quality_by_class_csv,
        "image_label_consistency_csv": args.image_label_consistency_csv,
    }

    payload: Dict[str, Any] = {
        "phase": "phase1B_annotation_quality",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "train_csv": str(args.train_csv),
        "total_rows": total_rows,
        "unique_images": unique_images,
        "abnormal_rows": abnormal_rows,
        "no_finding_rows": no_finding_rows,
        "abnormal_images": abnormal_images,
        "no_finding_images": no_finding_images,
        "abnormal_detection_classes_excluding_no_finding": len(abnormal_classes),
        "abnormal_class_names": abnormal_classes,
        "invalid_bbox_total": invalid_total,
        "invalid_bbox_by_reason": reason_counts,
        "no_finding_with_bbox_count": no_finding_with_bbox,
        "abnormal_missing_bbox_count": abnormal_missing_bbox,
        "mixed_no_finding_abnormal_image_count": mixed_count,
        "negative_coordinate_count": int(negative_coord),
        "zero_or_negative_area_count": int(zero_neg_area),
        "exact_duplicate_candidate_count": exact_count,
        "near_duplicate_candidate_count": near_count,
        "near_duplicate_iou_threshold": args.near_duplicate_iou,
        "class_mapping_issue_count": mapping_issues,
        "boundary_check_status": boundary_status,
        "has_rad_id": has_rad_id,
        "warnings": warnings,
        "forbidden_actions_confirmed": {a: "not_performed" for a in FORBIDDEN_ACTIONS},
        "generated_files": generated_files,
    }

    # Write outputs.
    ensure_parent(args.output_json).write_text(
        json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8"
    )
    invalid_df.to_csv(ensure_parent(args.invalid_bbox_csv), index=False)
    dup_df.to_csv(ensure_parent(args.duplicate_csv), index=False)
    mapping_df.to_csv(ensure_parent(args.class_mapping_csv), index=False)
    bbox_by_class_df.to_csv(
        ensure_parent(args.bbox_quality_by_class_csv), index=False
    )
    consistency_df.to_csv(
        ensure_parent(args.image_label_consistency_csv), index=False
    )
    write_markdown_report(args.report_md, payload, args.near_duplicate_iou)
    write_annotation_sanity_md(args.annotation_sanity_md, payload, invalid_df)

    # Console summary.
    print("=" * 64)
    print("Phase 1B — Annotation Quality")
    print("=" * 64)
    print(f"Source CSV                 : {args.train_csv}")
    print(f"Total rows                 : {total_rows}")
    print(f"Unique images              : {unique_images}")
    print(f"Invalid bbox total         : {invalid_total}")
    print(f"No Finding with bbox       : {no_finding_with_bbox}")
    print(f"Abnormal missing bbox      : {abnormal_missing_bbox}")
    print(f"Exact duplicate candidates : {exact_count}")
    print(f"Near duplicate candidates  : {near_count} (IoU >= {args.near_duplicate_iou})")
    print(f"Class mapping issues       : {mapping_issues}")
    print(f"Boundary check status      : {boundary_status}")
    print("-" * 64)
    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  WARN: {w}")
    else:
        print("Warnings                   : none")
    print("-" * 64)
    print("Generated files:")
    for name, fpath in generated_files.items():
        print(f"  {name:<28} -> {fpath}")
    print("=" * 64)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
