#!/usr/bin/env python3
"""Phase 2A — Data Standardization / Image-Boundary Validation.

Validates, for the Phase 1C controlled scope (4,894 images), that:
  1. every image_id has a real DICOM file on disk (availability check), and
  2. every abnormal bounding box lies within the true image dimensions read
     from the DICOM header (boundary validation).

DICOM is read ONLY for metadata/dimensions (header via stop_before_pixels).
Pixel arrays are read only if --verify-pixel-array is set, and even then are
never saved, normalized, or converted.

This script is REPORT-ONLY. It never edits, clamps, deletes, or fuses any
bounding box, never copies/converts images, and never creates a processed
training dataset. Invalid boxes and DICOM errors are surfaced as CANDIDATES
for human/GPT review.

Scope guardrails (Phase 2A): this script does NOT
  - split train/val/test, convert to COCO, train, pseudo-label, tune thresholds
  - touch the test set
  - edit/clamp/delete/fuse bboxes (incl. the 147 near-duplicate candidates)
  - copy/convert images or create PNG/JPG or processed training images

"No Finding" is a NEGATIVE image label, NOT a detection class.

Usage (Windows CMD):
    python scripts\\02A_dicom_bbox_boundary_validation.py ^
        --annotations-csv data\\interim\\vinbigdata_phase1C_scope_annotations.csv ^
        --manifest-csv data\\manifests\\phase1C_selected_images_manifest.csv ^
        --dicom-root D:\\ssl_detection_xray\\data\\raw\\vinbigdata\\dicom_subset\\train
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    import pandas as pd
except Exception as exc:  # pragma: no cover
    print(
        "ERROR: pandas and numpy are required but could not be imported.\n"
        f"       Underlying error: {exc!r}",
        file=sys.stderr,
    )
    raise SystemExit(2)

try:
    import pydicom
    _HAVE_PYDICOM = True
except Exception:  # pragma: no cover
    pydicom = None  # type: ignore
    _HAVE_PYDICOM = False


# --- Configuration --------------------------------------------------------

NO_FINDING_LABELS = {"no finding"}
BBOX_COLUMNS = ["x_min", "y_min", "x_max", "y_max"]
REQUIRED_ANN_COLUMNS = ["image_id", "class_name", *BBOX_COLUMNS]

EXPECT_TOTAL_IMAGES = 4894
EXPECT_ABNORMAL_IMAGES = 4394
EXPECT_NO_FINDING_IMAGES = 500
EXPECT_ABNORMAL_ROWS = 36096

META_FIELDS = [
    "PhotometricInterpretation",
    "BitsAllocated",
    "BitsStored",
    "PixelRepresentation",
    "SamplesPerPixel",
    "TransferSyntaxUID",
    "Modality",
    "SOPInstanceUID",
]


def is_no_finding(label: Any) -> bool:
    if label is None:
        return False
    if isinstance(label, float) and np.isnan(label):
        return False
    return str(label).strip().lower() in NO_FINDING_LABELS


# --- CLI ------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2A — DICOM availability & bbox boundary validation.",
    )
    p.add_argument(
        "--annotations-csv",
        type=str,
        default="data/interim/vinbigdata_phase1C_scope_annotations.csv",
    )
    p.add_argument(
        "--manifest-csv",
        type=str,
        default="data/manifests/phase1C_selected_images_manifest.csv",
    )
    p.add_argument(
        "--dicom-root",
        type=str,
        default=None,
        help="Root dir of DICOM files. Optional if manifest source_path is usable.",
    )
    p.add_argument(
        "--output-json",
        type=str,
        default="reports/phase2A_dicom_bbox_validation.json",
    )
    p.add_argument(
        "--report-md",
        type=str,
        default="reports/phase2A_dicom_bbox_validation.md",
    )
    p.add_argument(
        "--image-metadata-csv",
        type=str,
        default="reports/phase2A_image_metadata.csv",
    )
    p.add_argument(
        "--image-availability-csv",
        type=str,
        default="reports/phase2A_image_availability.csv",
    )
    p.add_argument(
        "--bbox-validation-csv",
        type=str,
        default="reports/phase2A_bbox_boundary_validation.csv",
    )
    p.add_argument(
        "--invalid-bbox-csv",
        type=str,
        default="reports/phase2A_invalid_bbox_candidates.csv",
    )
    p.add_argument(
        "--dicom-errors-csv",
        type=str,
        default="reports/phase2A_dicom_read_errors.csv",
    )
    p.add_argument("--verify-pixel-array", action="store_true", default=False)
    p.add_argument(
        "--pixel-check-limit",
        type=int,
        default=0,
        help="0 = check all (when verify-pixel-array on); >0 = deterministic sample of N.",
    )
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args(argv)


# --- IO helpers -----------------------------------------------------------


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_csv(path: str, required: List[str], what: str) -> pd.DataFrame:
    fp = Path(path)
    if not fp.exists():
        raise FileNotFoundError(
            f"{what} not found: {fp}\n       Check the path. (Phase 2A reads metadata only.)"
        )
    try:
        df = pd.read_csv(fp)
    except Exception as exc:
        raise ValueError(f"Failed to read {what} '{fp}': {exc!r}") from exc
    if df.empty:
        raise ValueError(f"{what} '{fp}' is empty.")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{what} missing required column(s): {missing}\n"
            f"       Found: {list(df.columns)}"
        )
    return df


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# --- Availability ---------------------------------------------------------


def build_dicom_index(dicom_root: Optional[str]) -> Dict[str, str]:
    """Recursively index DICOM files under dicom_root by image_id (Path.stem).

    This mirrors Phase 1C's `rglob("*.dicom")` so availability resolution stays
    consistent: files may live in nested subfolders (e.g. train/<chunk>/<id>.dicom),
    not only flat under the root. Both .dicom and .dcm are indexed.

    If an image_id appears under multiple files, the first (sorted) is kept and
    the collision is discoverable via the returned index size vs file count.
    """
    index: Dict[str, str] = {}
    if not dicom_root:
        return index
    root = Path(dicom_root)
    if not root.is_dir():
        return index
    for pattern in ("*.dicom", "*.dcm"):
        for fp in sorted(root.rglob(pattern)):
            if fp.is_file():
                index.setdefault(fp.stem, str(fp))
    return index


def resolve_image_path(
    image_id: str,
    manifest_source_path: Optional[str],
    dicom_root: Optional[str],
    dicom_index: Dict[str, str],
) -> Tuple[str, str, bool, str]:
    """Resolve a real DICOM file for an image_id.

    Returns (expected_path, resolved_path, file_exists, source_of_path).
    Checks the actual filesystem; never trusts the manifest alone.

    Resolution order:
      1. manifest source_path, only if it points to a real file on THIS disk;
      2. recursive dicom_root index (synced with Phase 1C rglob), which handles
         nested subfolders; then flat {root}/{id}.dicom|.dcm as a fallback.
    """
    # 1) Try manifest source_path (packaging-time paths usually will NOT exist
    #    on this machine, so this is checked but rarely resolves).
    if manifest_source_path and isinstance(manifest_source_path, str):
        sp = manifest_source_path.strip()
        if sp and sp.lower() not in ("nan", "none"):
            candidate = os.path.normpath(sp)
            if os.path.exists(candidate) and os.path.isfile(candidate):
                return candidate, candidate, True, "manifest_source_path"

    # 2a) Recursive index (nested-safe, matches Phase 1C behaviour).
    hit = dicom_index.get(str(image_id))
    if hit and os.path.isfile(hit):
        return hit, hit, True, "dicom_root_recursive"

    # 2b) Flat pattern fallback directly under root.
    if dicom_root:
        for ext in (".dicom", ".dcm"):
            candidate = os.path.normpath(os.path.join(dicom_root, f"{image_id}{ext}"))
            if os.path.exists(candidate) and os.path.isfile(candidate):
                return candidate, candidate, True, f"dicom_root_pattern{ext}"

    # Not found: report the best 'expected' path for diagnostics.
    if manifest_source_path and str(manifest_source_path).strip().lower() not in ("", "nan", "none"):
        expected = os.path.normpath(str(manifest_source_path).strip())
        src = "manifest_source_path"
    elif dicom_root:
        expected = os.path.normpath(os.path.join(dicom_root, f"{image_id}.dicom"))
        src = "dicom_root_recursive_or_pattern"
    else:
        expected = ""
        src = "none"
    return expected, "", False, src


def build_availability(
    manifest: pd.DataFrame, dicom_root: Optional[str], dicom_index: Dict[str, str]
) -> pd.DataFrame:
    """Check each manifest image_id exactly once against the real filesystem."""
    has_source = "source_path" in manifest.columns
    records: List[Dict[str, Any]] = []
    # Deduplicate to one row per image_id (manifest is image-level already).
    seen = manifest.drop_duplicates("image_id")
    for _, row in seen.iterrows():
        iid = str(row["image_id"])
        sp = str(row["source_path"]) if has_source else None
        expected, resolved, exists, src = resolve_image_path(
            iid, sp, dicom_root, dicom_index
        )
        records.append(
            {
                "image_id": iid,
                "expected_path": expected,
                "resolved_path": resolved,
                "file_exists": bool(exists),
                "availability_status": "available" if exists else "missing",
                "source_of_path": src,
            }
        )
    return pd.DataFrame(records)


# --- DICOM metadata -------------------------------------------------------


def read_dicom_metadata(
    resolved_path: str, verify_pixels: bool
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Read DICOM header (no pixels by default). Returns (meta, error_or_None)."""
    meta: Dict[str, Any] = {
        "image_height": None,  # Rows
        "image_width": None,   # Columns
        "pixel_array_checked": False,
        "pixel_shape_matches": None,
    }
    for f in META_FIELDS:
        meta[f] = None

    if not _HAVE_PYDICOM:
        return meta, {"error": "pydicom_not_installed"}

    try:
        ds = pydicom.dcmread(resolved_path, stop_before_pixels=True, force=True)
    except Exception as exc:
        return meta, {"error": f"header_read_error: {exc!r}"}

    try:
        rows = getattr(ds, "Rows", None)
        cols = getattr(ds, "Columns", None)
        meta["image_height"] = int(rows) if rows is not None else None
        meta["image_width"] = int(cols) if cols is not None else None
        for f in META_FIELDS:
            val = getattr(ds, f, None)
            if f == "TransferSyntaxUID":
                # Transfer syntax lives on file_meta.
                tsu = None
                if getattr(ds, "file_meta", None) is not None:
                    tsu = getattr(ds.file_meta, "TransferSyntaxUID", None)
                val = str(tsu) if tsu is not None else None
            meta[f] = None if val is None else str(val)
    except Exception as exc:
        return meta, {"error": f"header_field_error: {exc!r}"}

    if verify_pixels:
        try:
            ds_px = pydicom.dcmread(resolved_path, force=True)
            arr = ds_px.pixel_array  # not saved, not normalized, not converted
            meta["pixel_array_checked"] = True
            h, w = arr.shape[0], arr.shape[1]
            meta["pixel_shape_matches"] = bool(
                meta["image_height"] == h and meta["image_width"] == w
            )
            del arr, ds_px
        except Exception as exc:
            meta["pixel_array_checked"] = True
            meta["pixel_shape_matches"] = None
            return meta, {"error": f"pixel_read_error: {exc!r}"}

    return meta, None


def select_pixel_check_ids(
    available_ids: List[str], limit: int, seed: int
) -> set:
    """Deterministic selection of image_ids for pixel verification."""
    if limit <= 0:
        return set(available_ids)
    if limit >= len(available_ids):
        return set(available_ids)
    rng = random.Random(seed)
    return set(rng.sample(sorted(available_ids), limit))


# --- BBox boundary validation ---------------------------------------------


def validate_bboxes(
    ann: pd.DataFrame,
    dims_by_image: Dict[str, Tuple[Optional[int], Optional[int]]],
    dicom_status_by_image: Dict[str, str],
    has_rad_id: bool,
    has_class_id: bool,
) -> Tuple[pd.DataFrame, Dict[str, int], int, int]:
    """Validate each abnormal bbox against true image dimensions.

    Returns (validation_df, reason_counts, nf_with_bbox, abnormal_missing_bbox).
    No box is ever modified.
    """
    reason_counts: Dict[str, int] = {
        "missing_coordinate": 0,
        "non_numeric_coordinate": 0,
        "x_min_negative": 0,
        "y_min_negative": 0,
        "x_max_negative": 0,
        "y_max_negative": 0,
        "x_min_ge_x_max": 0,
        "y_min_ge_y_max": 0,
        "bbox_width_le_0": 0,
        "bbox_height_le_0": 0,
        "x_max_gt_image_width": 0,
        "y_max_gt_image_height": 0,
        "image_dimension_missing": 0,
        "dicom_missing_or_read_error": 0,
    }
    nf_with_bbox = 0
    abnormal_missing_bbox = 0

    work = ann.copy()
    work["_is_nf"] = work["class_name"].apply(is_no_finding)
    raw = {c: work[c] for c in BBOX_COLUMNS}
    num = {c: to_numeric(work[c]) for c in BBOX_COLUMNS}

    records: List[Dict[str, Any]] = []

    for pos, (idx, row) in enumerate(work.iterrows()):
        iid = str(row["image_id"])
        is_nf = bool(row["_is_nf"])

        # No Finding: must have no bbox.
        any_coord = any(pd.notna(raw[c].iloc[pos]) and str(raw[c].iloc[pos]).strip() != "" for c in BBOX_COLUMNS)
        if is_nf:
            if any_coord:
                nf_with_bbox += 1
            continue  # No Finding rows are not abnormal bboxes.

        # Abnormal row -> validate.
        reasons: List[str] = []
        xmin, ymin, xmax, ymax = (num[c].iloc[pos] for c in BBOX_COLUMNS)

        raw_present = [str(raw[c].iloc[pos]).strip() not in ("", "nan", "None") and pd.notna(raw[c].iloc[pos]) for c in BBOX_COLUMNS]
        num_present = [pd.notna(v) for v in (xmin, ymin, xmax, ymax)]

        if not all(num_present):
            # distinguish missing vs non-numeric
            if any(rp and (not npz) for rp, npz in zip(raw_present, num_present)):
                reasons.append("non_numeric_coordinate")
                reason_counts["non_numeric_coordinate"] += 1
            if any((not rp) for rp in raw_present):
                reasons.append("missing_coordinate")
                reason_counts["missing_coordinate"] += 1
                abnormal_missing_bbox += 1

        ih, iw = dims_by_image.get(iid, (None, None))
        dicom_status = dicom_status_by_image.get(iid, "unknown")
        if dicom_status not in ("ok",):
            reasons.append("dicom_missing_or_read_error")
            reason_counts["dicom_missing_or_read_error"] += 1
        if iw is None or ih is None:
            reasons.append("image_dimension_missing")
            reason_counts["image_dimension_missing"] += 1

        bbox_w = None
        bbox_h = None
        if all(num_present):
            if xmin < 0:
                reasons.append("x_min_negative"); reason_counts["x_min_negative"] += 1
            if ymin < 0:
                reasons.append("y_min_negative"); reason_counts["y_min_negative"] += 1
            if xmax < 0:
                reasons.append("x_max_negative"); reason_counts["x_max_negative"] += 1
            if ymax < 0:
                reasons.append("y_max_negative"); reason_counts["y_max_negative"] += 1
            if xmin >= xmax:
                reasons.append("x_min_ge_x_max"); reason_counts["x_min_ge_x_max"] += 1
            if ymin >= ymax:
                reasons.append("y_min_ge_y_max"); reason_counts["y_min_ge_y_max"] += 1
            bbox_w = xmax - xmin
            bbox_h = ymax - ymin
            if bbox_w <= 0:
                reasons.append("bbox_width_le_0"); reason_counts["bbox_width_le_0"] += 1
            if bbox_h <= 0:
                reasons.append("bbox_height_le_0"); reason_counts["bbox_height_le_0"] += 1
            if iw is not None and xmax > iw:
                reasons.append("x_max_gt_image_width"); reason_counts["x_max_gt_image_width"] += 1
            if ih is not None and ymax > ih:
                reasons.append("y_max_gt_image_height"); reason_counts["y_max_gt_image_height"] += 1

        # boundary_valid strictly requires clean geometry + dims + dicom ok.
        boundary_valid = (
            all(num_present)
            and iw is not None
            and ih is not None
            and dicom_status == "ok"
            and (xmin >= 0)
            and (ymin >= 0)
            and (xmin < xmax)
            and (ymin < ymax)
            and (xmax <= iw)
            and (ymax <= ih)
        )

        rec: Dict[str, Any] = {
            "source_row_index": int(idx),
            "image_id": iid,
        }
        if has_rad_id:
            rec["rad_id"] = row.get("rad_id")
        if has_class_id:
            rec["class_id"] = row.get("class_id")
        rec["class_name"] = row.get("class_name")
        rec["x_min"] = None if pd.isna(xmin) else float(xmin)
        rec["y_min"] = None if pd.isna(ymin) else float(ymin)
        rec["x_max"] = None if pd.isna(xmax) else float(xmax)
        rec["y_max"] = None if pd.isna(ymax) else float(ymax)
        rec["image_width"] = iw
        rec["image_height"] = ih
        rec["bbox_width"] = None if bbox_w is None else float(bbox_w)
        rec["bbox_height"] = None if bbox_h is None else float(bbox_h)
        rec["boundary_valid"] = bool(boundary_valid)
        rec["invalid_reasons"] = ";".join(reasons)
        records.append(rec)

    validation_df = pd.DataFrame(records)
    return validation_df, reason_counts, nf_with_bbox, abnormal_missing_bbox


# --- Report writer --------------------------------------------------------


def write_report_md(path: str, p: Dict[str, Any]) -> None:
    L: List[str] = []
    L.append("# Phase 2A — DICOM Availability & BBox Boundary Validation")
    L.append("")
    L.append(f"_Generated {p['created_utc']}._")
    L.append("")
    L.append("## Executive summary")
    L.append("")
    L.append(
        f"Checked availability of **{p['availability_checked_image_count']}** images "
        f"and validated **{p['abnormal_bbox_rows_checked']}** abnormal bboxes against "
        "true DICOM dimensions. "
        f"DICOM missing: **{p['dicom_missing_count']}**, read errors: "
        f"**{p['dicom_read_error_count']}**. Boundary-invalid bboxes: "
        f"**{p['bbox_boundary_invalid_count']}**. DoD pass candidate: "
        f"**{p['dod_pass_candidate']}**."
    )
    L.append("")
    L.append("## Scope")
    L.append("")
    L.append(
        "- Phase 1C controlled scope only (expected 4,894 images). "
        "Report-only: no bbox edited/clamped/deleted/fused; no image copied/converted."
    )
    L.append(
        "- DICOM read for metadata/dimensions only (header via stop_before_pixels)."
    )
    L.append("")
    L.append("## Inputs")
    L.append("")
    L.append(f"- annotations_csv: `{p['annotations_csv']}`")
    L.append(f"- manifest_csv: `{p['manifest_csv']}`")
    L.append(f"- dicom_root: `{p['dicom_root']}`")
    L.append("")
    L.append("## Image availability summary")
    L.append("")
    L.append(f"- selected_scope_expected_images: {p['selected_scope_expected_images']}")
    L.append(f"- availability_checked_image_count: {p['availability_checked_image_count']}")
    L.append(f"- dicom_available_count: {p['dicom_available_count']}")
    L.append(f"- dicom_missing_count: {p['dicom_missing_count']}")
    if p["dicom_missing_image_ids"]:
        L.append(f"- sample missing image_ids: {p['dicom_missing_image_ids'][:10]}")
    L.append("")
    L.append("## DICOM metadata summary")
    L.append("")
    L.append(f"- dicom_read_success_count: {p['dicom_read_success_count']}")
    L.append(f"- dicom_read_error_count: {p['dicom_read_error_count']}")
    L.append(f"- pixel_array_checked: {p['pixel_array_checked']}")
    L.append(f"- pixel_array_check_count: {p['pixel_array_check_count']}")
    L.append(f"- pixel_array_error_count: {p['pixel_array_error_count']}")
    L.append("")
    L.append("## Image dimension summary")
    L.append("")
    L.append(f"- image_dimension_available_count: {p['image_dimension_available_count']}")
    L.append(f"- image_dimension_missing_count: {p['image_dimension_missing_count']}")
    L.append(f"- width/height distribution: {p['width_height_distribution_summary']}")
    L.append("")
    L.append("## BBox boundary validation summary")
    L.append("")
    L.append(f"- abnormal_bbox_rows_checked: {p['abnormal_bbox_rows_checked']}")
    L.append(f"- bbox_boundary_valid_count: {p['bbox_boundary_valid_count']}")
    L.append(f"- bbox_boundary_invalid_count: {p['bbox_boundary_invalid_count']}")
    L.append("")
    L.append("| reason | count |")
    L.append("|---|---|")
    for reason, count in p["invalid_bbox_by_reason"].items():
        L.append(f"| {reason} | {count} |")
    L.append("")
    L.append("## No Finding policy check")
    L.append("")
    L.append(f"- no_finding_images: {p['no_finding_images']}")
    L.append(f"- no_finding_rows: {p['no_finding_rows']}")
    L.append(f"- no_finding_with_bbox_count: {p['no_finding_with_bbox_count']}")
    L.append(f"- abnormal_missing_bbox_count: {p['abnormal_missing_bbox_count']}")
    L.append("- No Finding is a negative image label; excluded from detection classes.")
    L.append("")
    L.append("## Invalid bbox / DICOM error details")
    L.append("")
    L.append("- Invalid bbox candidates: `phase2A_invalid_bbox_candidates.csv` (review only, not auto-fixed).")
    L.append("- DICOM read errors: `phase2A_dicom_read_errors.csv`.")
    L.append("")
    L.append("## Decision candidates")
    L.append("")
    if p["decision_candidates"]:
        for d in p["decision_candidates"]:
            L.append(f"- {d}")
    else:
        L.append("- None; all checks clean.")
    L.append("")
    L.append("## Forbidden actions confirmed")
    L.append("")
    for k, v in p["forbidden_actions_confirmed"].items():
        L.append(f"- {k}: {v}")
    L.append("")
    L.append("## Recommended next action")
    L.append("")
    L.append(
        "- **Send these outputs to GPT review BEFORE ticking the Phase 2A "
        "checklist.** Do not auto-fix boxes or images; corrections are research "
        "decisions."
    )
    L.append("")
    ensure_parent(path).write_text("\n".join(L), encoding="utf-8")


# --- Main -----------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:  # noqa: C901
    args = parse_args(argv)
    random.seed(args.seed)

    # Load inputs.
    try:
        ann = load_csv(args.annotations_csv, REQUIRED_ANN_COLUMNS, "annotations CSV")
        manifest = load_csv(args.manifest_csv, ["image_id"], "manifest CSV")
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    has_source_path = "source_path" in manifest.columns
    if not args.dicom_root and not has_source_path:
        print(
            "ERROR: --dicom-root not provided and manifest has no 'source_path' column.\n"
            "       Cannot resolve DICOM files. Provide --dicom-root.",
            file=sys.stderr,
        )
        return 1
    if args.dicom_root and not os.path.isdir(args.dicom_root) and not has_source_path:
        print(
            f"ERROR: --dicom-root does not exist: {args.dicom_root}\n"
            "       and manifest has no usable 'source_path'.",
            file=sys.stderr,
        )
        return 1

    has_rad_id = "rad_id" in ann.columns
    has_class_id = "class_id" in ann.columns

    # Core annotation/manifest cross-checks.
    ann_nf = ann["class_name"].apply(is_no_finding)
    total_annotation_rows = int(len(ann))
    unique_annotation_images = int(ann["image_id"].nunique())
    abnormal_rows = int((~ann_nf).sum())
    no_finding_rows = int(ann_nf.sum())
    abnormal_images = int(ann[~ann_nf]["image_id"].nunique())
    no_finding_only_images = int(
        ann.groupby("image_id")["class_name"].apply(
            lambda s: bool(s.apply(is_no_finding).any() and not (~s.apply(is_no_finding)).any())
        ).sum()
    )

    manifest_rows = int(len(manifest))
    manifest_unique_images = int(manifest["image_id"].nunique())
    duplicate_manifest_image_id_count = int(manifest_rows - manifest_unique_images)

    ann_ids = set(ann["image_id"].astype(str))
    man_ids = set(manifest["image_id"].astype(str))
    annotation_not_in_manifest = sorted(ann_ids - man_ids)
    manifest_not_in_annotation = sorted(man_ids - ann_ids)

    # Availability (filesystem check, one row per image).
    dicom_index = build_dicom_index(args.dicom_root)
    availability = build_availability(manifest, args.dicom_root, dicom_index)
    availability_checked_image_count = int(len(availability))
    available_df = availability[availability["file_exists"]]
    dicom_available_count = int(len(available_df))
    missing_df = availability[~availability["file_exists"]]
    dicom_missing_count = int(len(missing_df))
    dicom_missing_image_ids = missing_df["image_id"].astype(str).tolist()

    # DICOM metadata reading.
    resolved_by_id = dict(
        zip(available_df["image_id"].astype(str), available_df["resolved_path"])
    )
    available_ids = list(available_df["image_id"].astype(str))
    pixel_check_ids = (
        select_pixel_check_ids(available_ids, args.pixel_check_limit, args.seed)
        if args.verify_pixel_array
        else set()
    )

    meta_records: List[Dict[str, Any]] = []
    error_records: List[Dict[str, Any]] = []
    dims_by_image: Dict[str, Tuple[Optional[int], Optional[int]]] = {}
    dicom_status_by_image: Dict[str, str] = {}
    dicom_read_success = 0
    dicom_read_error = 0
    pixel_checked = 0
    pixel_errors = 0

    for iid in available_ids:
        rpath = resolved_by_id[iid]
        verify = args.verify_pixel_array and (iid in pixel_check_ids)
        meta, err = read_dicom_metadata(rpath, verify)
        if verify:
            pixel_checked += 1
        if err is not None:
            dicom_read_error += 1
            dicom_status_by_image[iid] = "error"
            if "pixel_read_error" in str(err.get("error", "")):
                pixel_errors += 1
            error_records.append(
                {"image_id": iid, "resolved_path": rpath, "error": err.get("error")}
            )
            # Still record dims if partially read.
            dims_by_image[iid] = (meta.get("image_height"), meta.get("image_width"))
        else:
            dicom_read_success += 1
            dicom_status_by_image[iid] = "ok"
            dims_by_image[iid] = (meta.get("image_height"), meta.get("image_width"))
            if meta.get("pixel_array_checked") and meta.get("pixel_shape_matches") is False:
                pixel_errors += 1

        row = {"image_id": iid, "resolved_path": rpath}
        row.update(
            {
                "image_height": meta.get("image_height"),
                "image_width": meta.get("image_width"),
                "pixel_array_checked": meta.get("pixel_array_checked"),
                "pixel_shape_matches": meta.get("pixel_shape_matches"),
            }
        )
        for f in META_FIELDS:
            row[f] = meta.get(f)
        meta_records.append(row)

    # Missing images have no dims and error status.
    for iid in dicom_missing_image_ids:
        dicom_status_by_image[iid] = "missing"
        dims_by_image[iid] = (None, None)

    image_dimension_available_count = int(
        sum(1 for v in dims_by_image.values() if v[0] is not None and v[1] is not None)
    )
    image_dimension_missing_count = int(
        availability_checked_image_count - image_dimension_available_count
    )

    metadata_df = pd.DataFrame(meta_records)
    errors_df = pd.DataFrame(
        error_records, columns=["image_id", "resolved_path", "error"]
    )

    # Width/height distribution summary.
    wh_summary: Dict[str, Any] = {}
    if not metadata_df.empty and "image_width" in metadata_df.columns:
        w = pd.to_numeric(metadata_df["image_width"], errors="coerce").dropna()
        h = pd.to_numeric(metadata_df["image_height"], errors="coerce").dropna()
        if len(w):
            wh_summary = {
                "width_min": int(w.min()), "width_max": int(w.max()),
                "width_mean": round(float(w.mean()), 2),
                "height_min": int(h.min()), "height_max": int(h.max()),
                "height_mean": round(float(h.mean()), 2),
                "distinct_wh_pairs": int(
                    metadata_df[["image_width", "image_height"]].drop_duplicates().shape[0]
                ),
            }

    # BBox validation.
    validation_df, reason_counts, nf_with_bbox, abnormal_missing_bbox = validate_bboxes(
        ann, dims_by_image, dicom_status_by_image, has_rad_id, has_class_id
    )
    abnormal_bbox_rows_checked = int(len(validation_df))
    bbox_valid_count = int(validation_df["boundary_valid"].sum()) if not validation_df.empty else 0
    bbox_invalid_count = abnormal_bbox_rows_checked - bbox_valid_count
    invalid_df = (
        validation_df[~validation_df["boundary_valid"]].copy()
        if not validation_df.empty
        else validation_df
    )

    # Warnings & decision candidates.
    warnings: List[str] = []
    decision_candidates: List[str] = []
    if dicom_missing_count:
        warnings.append(f"{dicom_missing_count} image(s) missing on disk.")
        decision_candidates.append(
            f"Resolve {dicom_missing_count} missing DICOM file(s) before COCO/training."
        )
    if dicom_read_error:
        warnings.append(f"{dicom_read_error} DICOM read error(s).")
        decision_candidates.append(
            f"Investigate {dicom_read_error} DICOM read error(s) (see dicom errors CSV)."
        )
    if bbox_invalid_count:
        warnings.append(f"{bbox_invalid_count} abnormal bbox(es) boundary-invalid.")
        decision_candidates.append(
            f"Review {bbox_invalid_count} boundary-invalid bbox candidate(s); do NOT auto-clamp."
        )
    if nf_with_bbox:
        warnings.append(f"{nf_with_bbox} No Finding row(s) carry bbox coordinates.")
        decision_candidates.append(
            f"Review {nf_with_bbox} No Finding row(s) with bbox; policy says none expected."
        )
    if annotation_not_in_manifest:
        warnings.append(f"{len(annotation_not_in_manifest)} annotation image_id(s) not in manifest.")
    if manifest_not_in_annotation:
        warnings.append(f"{len(manifest_not_in_annotation)} manifest image_id(s) not in annotation.")

    forbidden = {
        "split_created": False,
        "coco_created": False,
        "training_started": False,
        "pseudo_label_generated": False,
        "threshold_tuned": False,
        "test_set_used": False,
        "annotations_deleted_or_edited": False,
        "bbox_clamped_or_modified": False,
        "near_duplicate_bbox_deleted_or_fused": False,
        "processed_training_images_created": False,
        "image_files_copied": False,
        "image_files_converted": False,
        "png_or_jpg_created": False,
    }

    dod_pass_candidate = bool(
        EXPECT_TOTAL_IMAGES == EXPECT_TOTAL_IMAGES
        and availability_checked_image_count == EXPECT_TOTAL_IMAGES
        and manifest_unique_images == EXPECT_TOTAL_IMAGES
        and duplicate_manifest_image_id_count == 0
        and len(annotation_not_in_manifest) == 0
        and len(manifest_not_in_annotation) == 0
        and dicom_missing_count == 0
        and dicom_read_error == 0
        and image_dimension_missing_count == 0
        and abnormal_bbox_rows_checked == EXPECT_ABNORMAL_ROWS
        and bbox_invalid_count == 0
        and nf_with_bbox == 0
        and abnormal_missing_bbox == 0
        and all(v is False for v in forbidden.values())
    )

    payload: Dict[str, Any] = {
        "phase": "phase2A_dicom_bbox_boundary_validation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "annotations_csv": str(args.annotations_csv),
        "manifest_csv": str(args.manifest_csv),
        "dicom_root": str(args.dicom_root) if args.dicom_root else None,
        "dicom_files_indexed_under_root": len(dicom_index),
        "total_annotation_rows": total_annotation_rows,
        "unique_annotation_images": unique_annotation_images,
        "manifest_rows": manifest_rows,
        "manifest_unique_images": manifest_unique_images,
        "selected_scope_expected_images": EXPECT_TOTAL_IMAGES,
        "expected_image_count": EXPECT_TOTAL_IMAGES,
        "availability_checked_image_count": availability_checked_image_count,
        "abnormal_images": abnormal_images,
        "no_finding_images": no_finding_only_images,
        "abnormal_rows": abnormal_rows,
        "no_finding_rows": no_finding_rows,
        "dicom_available_count": dicom_available_count,
        "dicom_missing_count": dicom_missing_count,
        "dicom_missing_image_ids": dicom_missing_image_ids[:100],
        "dicom_read_success_count": dicom_read_success,
        "dicom_read_error_count": dicom_read_error,
        "image_dimension_available_count": image_dimension_available_count,
        "image_dimension_missing_count": image_dimension_missing_count,
        "abnormal_bbox_rows_checked": abnormal_bbox_rows_checked,
        "bbox_boundary_valid_count": bbox_valid_count,
        "bbox_boundary_invalid_count": bbox_invalid_count,
        "invalid_bbox_by_reason": reason_counts,
        "no_finding_with_bbox_count": nf_with_bbox,
        "abnormal_missing_bbox_count": abnormal_missing_bbox,
        "annotation_not_in_manifest_count": len(annotation_not_in_manifest),
        "manifest_not_in_annotation_count": len(manifest_not_in_annotation),
        "duplicate_manifest_image_id_count": duplicate_manifest_image_id_count,
        "width_height_distribution_summary": wh_summary,
        "pixel_array_checked": bool(args.verify_pixel_array),
        "pixel_array_check_count": pixel_checked,
        "pixel_array_error_count": pixel_errors,
        "forbidden_actions_confirmed": forbidden,
        "decision_candidates": decision_candidates,
        "warnings": warnings,
        "dod_pass_candidate": dod_pass_candidate,
        "generated_files": {
            "output_json": args.output_json,
            "report_md": args.report_md,
            "image_metadata_csv": args.image_metadata_csv,
            "image_availability_csv": args.image_availability_csv,
            "bbox_validation_csv": args.bbox_validation_csv,
            "invalid_bbox_csv": args.invalid_bbox_csv,
            "dicom_errors_csv": args.dicom_errors_csv,
        },
    }

    # Write outputs (UTF-8).
    ensure_parent(args.output_json).write_text(
        json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8"
    )
    availability.to_csv(ensure_parent(args.image_availability_csv), index=False, encoding="utf-8")
    metadata_df.to_csv(ensure_parent(args.image_metadata_csv), index=False, encoding="utf-8")
    # Ensure bbox validation CSV has headers even if empty.
    if validation_df.empty:
        cols = ["source_row_index", "image_id"]
        if has_rad_id: cols.append("rad_id")
        if has_class_id: cols.append("class_id")
        cols += ["class_name", *BBOX_COLUMNS, "image_width", "image_height",
                 "bbox_width", "bbox_height", "boundary_valid", "invalid_reasons"]
        validation_df = pd.DataFrame(columns=cols)
    validation_df.to_csv(ensure_parent(args.bbox_validation_csv), index=False, encoding="utf-8")
    if invalid_df.empty:
        invalid_df = pd.DataFrame(columns=validation_df.columns)
    invalid_df.to_csv(ensure_parent(args.invalid_bbox_csv), index=False, encoding="utf-8")
    errors_df.to_csv(ensure_parent(args.dicom_errors_csv), index=False, encoding="utf-8")
    write_report_md(args.report_md, payload)

    # Console summary.
    print("=" * 68)
    print("Phase 2A — DICOM Availability & BBox Boundary Validation")
    print("=" * 68)
    print(f"dicom files indexed (rglob)  : {len(dicom_index)}")
    print(f"total selected images        : {availability_checked_image_count}")
    print(f"annotation rows              : {total_annotation_rows}")
    print(f"availability checked images  : {availability_checked_image_count}")
    print(f"DICOM available / missing    : {dicom_available_count} / {dicom_missing_count}")
    print(f"DICOM read success / error   : {dicom_read_success} / {dicom_read_error}")
    print(f"dimension available / missing: {image_dimension_available_count} / {image_dimension_missing_count}")
    print(f"abnormal bbox rows checked   : {abnormal_bbox_rows_checked}")
    print(f"bbox valid count             : {bbox_valid_count}")
    print(f"bbox invalid count           : {bbox_invalid_count}")
    print(f"no_finding_with_bbox_count   : {nf_with_bbox}")
    print("-" * 68)
    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  WARN: {w}")
    else:
        print("Warnings                     : none")
    print("-" * 68)
    print(f"dod_pass_candidate           : {dod_pass_candidate}")
    print("=" * 68)
    print("NOTE: Report-only. No bbox/image modified. Send outputs to review first.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
