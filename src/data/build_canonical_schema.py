"""Phase 2B — Canonical Detection Annotation Schema (core module).

Builds a canonical, traceable annotation schema for the Phase 1C controlled
scope (4,894 images) by REORGANIZING existing metadata into three tables:

  * canonical_image_table  — one row per unique image_id
  * canonical_bbox_table   — one row per abnormal bbox (xyxy on original image)
  * canonical_class_mapping — 14 abnormal detection classes (No Finding excluded)

This module is a pure metadata transformation. It NEVER:
  - converts to COCO, splits data, trains, pseudo-labels, or tunes thresholds
  - reads pixels, copies/converts images, or creates processed training images
  - edits, clamps, deletes, or fuses any bbox (incl. 147 near-duplicate candidates)

"No Finding" is handled at the IMAGE level as a negative label; it is never a
detection class.

All functions are import-safe and free of GPU/MMDetection dependencies.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# --- Constants ------------------------------------------------------------

NO_FINDING_LABELS = {"no finding"}
BBOX_COLUMNS = ["x_min", "y_min", "x_max", "y_max"]

EXPECT_TOTAL_IMAGES = 4894
EXPECT_ABNORMAL_IMAGES = 4394
EXPECT_NO_FINDING_IMAGES = 500
EXPECT_ABNORMAL_ROWS = 36096
EXPECT_DETECTION_CLASSES = 14

BBOX_FORMAT = "xyxy_original_image"

# Portable path policy: canonical identifiers are image_id + relative_dicom_path.
# Absolute local paths are retained only as local evidence, never as keys.
PATH_ROOT_VARIABLE = "VINBIGDATA_DICOM_ROOT"
DICOM_EXT = ".dicom"
RELATIVE_DICOM_PREFIX = "train/"


def is_no_finding(label: Any) -> bool:
    """True if label is a No Finding negative label (case-insensitive)."""
    if label is None:
        return False
    if isinstance(label, float) and np.isnan(label):
        return False
    return str(label).strip().lower() in NO_FINDING_LABELS


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# --- Image metadata join --------------------------------------------------


def load_image_dimensions(
    image_metadata: pd.DataFrame,
) -> Tuple[Dict[str, Tuple[Optional[int], Optional[int]]], Dict[str, Optional[str]]]:
    """Map image_id -> (height, width) and image_id -> dicom_path from Phase 2A.

    Returns two dicts: dims_by_image and path_by_image.
    """
    dims: Dict[str, Tuple[Optional[int], Optional[int]]] = {}
    paths: Dict[str, Optional[str]] = {}
    if image_metadata is None or image_metadata.empty:
        return dims, paths

    h_col = "image_height" if "image_height" in image_metadata.columns else None
    w_col = "image_width" if "image_width" in image_metadata.columns else None
    p_col = None
    for cand in ("dicom_path", "resolved_path", "source_path"):
        if cand in image_metadata.columns:
            p_col = cand
            break

    for _, row in image_metadata.iterrows():
        iid = str(row["image_id"])
        h = row[h_col] if h_col else None
        w = row[w_col] if w_col else None
        try:
            h = int(h) if pd.notna(h) else None
        except Exception:
            h = None
        try:
            w = int(w) if pd.notna(w) else None
        except Exception:
            w = None
        dims[iid] = (h, w)
        paths[iid] = (str(row[p_col]) if p_col and pd.notna(row[p_col]) else None)
    return dims, paths


# --- Class mapping --------------------------------------------------------


def build_class_mapping(ann: pd.DataFrame) -> Tuple[pd.DataFrame, int, List[str]]:
    """Build canonical class mapping for abnormal detection classes only.

    canonical_class_id is deterministic: classes are sorted by class_id_original
    (numeric, when available) then class_name, and enumerated from 0.
    Returns (mapping_df, issue_count, warnings).
    """
    warnings: List[str] = []
    work = ann.copy()
    work["_is_nf"] = work["class_name"].apply(is_no_finding)
    abn = work[~work["_is_nf"]].copy()

    has_class_id = "class_id" in abn.columns

    # Determine one representative class_id per class_name and check bijection.
    issue_count = 0
    name_to_ids: Dict[str, List[Any]] = {}
    for cname, grp in abn.groupby(abn["class_name"].astype(str)):
        if has_class_id:
            ids = sorted(
                int(v) for v in pd.to_numeric(grp["class_id"], errors="coerce").dropna().unique()
            )
        else:
            ids = []
        name_to_ids[cname] = ids
        if len(ids) > 1:
            issue_count += 1
            warnings.append(
                f"class_name '{cname}' maps to multiple class_id: {ids}"
            )

    if has_class_id:
        id_to_names: Dict[int, List[str]] = {}
        for cid, grp in abn.groupby(pd.to_numeric(abn["class_id"], errors="coerce")):
            if pd.isna(cid):
                continue
            names = sorted(grp["class_name"].astype(str).unique())
            id_to_names[int(cid)] = names
            if len(names) > 1:
                issue_count += 1
                warnings.append(
                    f"class_id {int(cid)} maps to multiple class_name: {names}"
                )

    # Build deterministic ordering.
    records: List[Dict[str, Any]] = []
    for cname, ids in name_to_ids.items():
        cid_orig = ids[0] if ids else None
        records.append({"class_name": cname, "class_id_original": cid_orig})

    order_df = pd.DataFrame(records)
    # Sort by (class_id_original, class_name) deterministically.
    order_df["_sort_id"] = order_df["class_id_original"].apply(
        lambda v: (0, int(v)) if v is not None and pd.notna(v) else (1, 0)
    )
    order_df = order_df.sort_values(
        by=["_sort_id", "class_name"], key=lambda col: col if col.name == "class_name" else col
    ).reset_index(drop=True)
    order_df["canonical_class_id"] = range(len(order_df))

    # Attach counts.
    rows_by_name = abn["class_name"].astype(str).value_counts().to_dict()
    images_by_name = abn.groupby(abn["class_name"].astype(str))["image_id"].nunique().to_dict()
    # bbox_count = abnormal rows with full numeric bbox for that class.
    for c in BBOX_COLUMNS:
        abn[c] = to_numeric(abn[c])
    abn["_full_bbox"] = abn[BBOX_COLUMNS].notna().all(axis=1)
    bbox_by_name = abn[abn["_full_bbox"]].groupby(abn["class_name"].astype(str)).size().to_dict()

    mapping_records: List[Dict[str, Any]] = []
    for _, r in order_df.iterrows():
        cname = r["class_name"]
        mapping_records.append(
            {
                "canonical_class_id": int(r["canonical_class_id"]),
                "class_id_original": (
                    None if r["class_id_original"] is None or pd.isna(r["class_id_original"])
                    else int(r["class_id_original"])
                ),
                "class_name": cname,
                "is_detection_class": True,
                "is_no_finding": False,
                "row_count": int(rows_by_name.get(cname, 0)),
                "image_count": int(images_by_name.get(cname, 0)),
                "bbox_count": int(bbox_by_name.get(cname, 0)),
            }
        )
    mapping_df = pd.DataFrame(mapping_records)
    return mapping_df, issue_count, warnings


# --- Image table ----------------------------------------------------------


def _is_absolute_path(path: Optional[str]) -> bool:
    """True if `path` looks absolute on Windows or POSIX (env-agnostic)."""
    if not path or not isinstance(path, str):
        return False
    s = path.strip()
    if not s or s.lower() in ("nan", "none"):
        return False
    # POSIX absolute, Windows drive (C:\ or C:/), or UNC (\\server).
    if s.startswith("/"):
        return True
    if len(s) >= 3 and s[1] == ":" and s[2] in ("\\", "/"):
        return True
    if s.startswith("\\\\"):
        return True
    return False


def portable_paths_for(image_id: str) -> Dict[str, str]:
    """Canonical, environment-independent path fields for an image_id."""
    fname = f"{image_id}{DICOM_EXT}"
    return {
        "dicom_filename": fname,
        "relative_dicom_path": f"{RELATIVE_DICOM_PREFIX}{fname}",
    }


def build_image_table(
    ann: pd.DataFrame,
    manifest: pd.DataFrame,
    dims_by_image: Dict[str, Tuple[Optional[int], Optional[int]]],
    path_by_image: Dict[str, Optional[str]],
) -> pd.DataFrame:
    """One row per unique image_id in the controlled scope."""
    work = ann.copy()
    work["_is_nf"] = work["class_name"].apply(is_no_finding)
    for c in BBOX_COLUMNS:
        work[f"_{c}"] = to_numeric(work[c])
    work["_full_bbox"] = work[[f"_{c}" for c in BBOX_COLUMNS]].notna().all(axis=1)

    # scope_label from manifest if present.
    scope_by_id: Dict[str, Any] = {}
    if "scope_label" in manifest.columns:
        m = manifest.drop_duplicates("image_id")
        scope_by_id = dict(zip(m["image_id"].astype(str), m["scope_label"]))

    records: List[Dict[str, Any]] = []
    for iid, grp in work.groupby("image_id", sort=True):
        iid_s = str(iid)
        abn_mask = ~grp["_is_nf"]
        abn_names = sorted(grp.loc[abn_mask, "class_name"].astype(str).unique().tolist())
        bbox_count = int((abn_mask & grp["_full_bbox"]).sum())
        # No Finding rows that (incorrectly) carry bbox coordinates.
        nf_bbox_count = int((grp["_is_nf"] & grp["_full_bbox"]).sum())
        is_abnormal = bool(abn_mask.any())
        is_negative = bool((not is_abnormal) and grp["_is_nf"].any())
        h, w = dims_by_image.get(iid_s, (None, None))
        local_path = path_by_image.get(iid_s)
        pp = portable_paths_for(iid_s)
        local_is_abs = _is_absolute_path(local_path)
        records.append(
            {
                "canonical_image_id": None,  # filled after sort
                "image_id": iid_s,
                "dicom_filename": pp["dicom_filename"],
                "relative_dicom_path": pp["relative_dicom_path"],
                # Backward-compatible alias: now holds the RELATIVE path.
                "dicom_path": pp["relative_dicom_path"],
                "local_dicom_path": local_path,
                "local_dicom_path_is_absolute": bool(local_is_abs),
                "path_root_variable": PATH_ROOT_VARIABLE,
                "image_width": w,
                "image_height": h,
                "scope_label": scope_by_id.get(iid_s, ("abnormal" if is_abnormal else "no_finding")),
                "is_abnormal": is_abnormal,
                "is_negative": is_negative,
                "has_bbox": bool(bbox_count > 0),
                "bbox_count": bbox_count,
                "no_finding_bbox_count": nf_bbox_count,
                "abnormal_class_count": int(len(abn_names)),
                "abnormal_class_names": ";".join(abn_names),
                "source_row_count": int(len(grp)),
                "abnormal_row_count": int(abn_mask.sum()),
                "no_finding_row_count": int(grp["_is_nf"].sum()),
            }
        )
    image_df = pd.DataFrame(records).sort_values("image_id").reset_index(drop=True)
    image_df["canonical_image_id"] = range(len(image_df))
    return image_df


# --- BBox table -----------------------------------------------------------


def build_bbox_table(
    ann: pd.DataFrame,
    mapping_df: pd.DataFrame,
    dims_by_image: Dict[str, Tuple[Optional[int], Optional[int]]],
    boundary_by_row: Optional[Dict[int, bool]],
) -> pd.DataFrame:
    """One row per abnormal bbox. No bbox is modified, clamped, or dropped."""
    name_to_canon = dict(
        zip(mapping_df["class_name"].astype(str), mapping_df["canonical_class_id"])
    )
    work = ann.copy()
    work["_is_nf"] = work["class_name"].apply(is_no_finding)
    for c in BBOX_COLUMNS:
        work[f"_{c}"] = to_numeric(work[c])
    work["_full_bbox"] = work[[f"_{c}" for c in BBOX_COLUMNS]].notna().all(axis=1)

    has_rad_id = "rad_id" in work.columns
    has_class_id = "class_id" in work.columns

    # Only abnormal rows that carry a full bbox.
    abn = work[(~work["_is_nf"]) & work["_full_bbox"]].copy()

    records: List[Dict[str, Any]] = []
    ann_counter = 0
    for idx, row in abn.iterrows():
        iid = str(row["image_id"])
        xmin, ymin, xmax, ymax = (
            float(row["_x_min"]), float(row["_y_min"]),
            float(row["_x_max"]), float(row["_y_max"]),
        )
        h, w = dims_by_image.get(iid, (None, None))
        bbox_w = xmax - xmin
        bbox_h = ymax - ymin
        bbox_area = bbox_w * bbox_h
        cname = str(row["class_name"])
        canon = name_to_canon.get(cname)

        # is_valid_bbox: geometry only (no dependence on image dims).
        is_valid = bool(
            (xmin < xmax) and (ymin < ymax) and (bbox_w > 0) and (bbox_h > 0)
            and (xmin >= 0) and (ymin >= 0)
        )
        # boundary_valid: from Phase 2A if available, else recompute vs dims.
        if boundary_by_row is not None and int(idx) in boundary_by_row:
            boundary_valid = bool(boundary_by_row[int(idx)])
        elif w is not None and h is not None:
            boundary_valid = bool(is_valid and xmax <= w and ymax <= h)
        else:
            boundary_valid = False

        rec: Dict[str, Any] = {
            "canonical_ann_id": ann_counter,
            "image_id": iid,
            "source_row_id": int(idx),
        }
        if has_rad_id:
            rec["rad_id"] = row.get("rad_id")
        rec["class_id_original"] = (
            int(row["class_id"]) if has_class_id and pd.notna(row.get("class_id")) else None
        )
        rec["class_name"] = cname
        rec["canonical_class_id"] = None if canon is None else int(canon)
        rec["x_min"] = xmin
        rec["y_min"] = ymin
        rec["x_max"] = xmax
        rec["y_max"] = ymax
        rec["bbox_width"] = bbox_w
        rec["bbox_height"] = bbox_h
        rec["bbox_area"] = bbox_area
        rec["image_width"] = w
        rec["image_height"] = h
        rec["bbox_format"] = BBOX_FORMAT
        rec["is_valid_bbox"] = is_valid
        rec["boundary_valid"] = boundary_valid
        records.append(rec)
        ann_counter += 1

    return pd.DataFrame(records)


# --- No Finding audit -----------------------------------------------------


def build_no_finding_audit(image_df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """Audit the negative (No Finding) images. Returns (audit_df, policy_pass)."""
    neg = image_df[image_df["is_negative"]].copy()
    records: List[Dict[str, Any]] = []
    policy_pass_all = True
    for _, row in neg.iterrows():
        has_bbox = bool(row["has_bbox"])
        bbox_count = int(row["bbox_count"])
        nf_bbox_count = int(row.get("no_finding_bbox_count", 0))
        issue = ""
        ppass = True
        # A negative image must carry NO bbox at all: neither abnormal boxes nor
        # No Finding rows with coordinates.
        if has_bbox or bbox_count > 0 or nf_bbox_count > 0:
            ppass = False
            policy_pass_all = False
            issue = "no_finding_image_has_bbox"
        records.append(
            {
                "image_id": row["image_id"],
                "is_negative": True,
                "has_bbox": bool(has_bbox or nf_bbox_count > 0),
                "bbox_count": bbox_count + nf_bbox_count,
                "no_finding_row_count": int(row["no_finding_row_count"]),
                "abnormal_row_count": int(row["abnormal_row_count"]),
                "policy_pass": ppass,
                "issue": issue,
            }
        )
    audit_df = pd.DataFrame(
        records,
        columns=[
            "image_id", "is_negative", "has_bbox", "bbox_count",
            "no_finding_row_count", "abnormal_row_count", "policy_pass", "issue",
        ],
    )
    return audit_df, policy_pass_all


# --- Consistency validation -----------------------------------------------


def validate_schema(
    ann: pd.DataFrame,
    manifest: pd.DataFrame,
    image_df: pd.DataFrame,
    bbox_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    dims_by_image: Dict[str, Tuple[Optional[int], Optional[int]]],
    no_finding_policy_pass: bool,
    class_mapping_issue_count: int,
) -> Tuple[Dict[str, Any], pd.DataFrame, List[str]]:
    """Cross-check the canonical tables and produce validation metrics + errors."""
    errors: List[Dict[str, Any]] = []
    warnings: List[str] = []

    def add_error(kind: str, ref: str, detail: str) -> None:
        errors.append({"error_type": kind, "reference": ref, "detail": detail})

    # bbox -> image existence.
    image_ids = set(image_df["image_id"].astype(str))
    bbox_without_image = 0
    if not bbox_df.empty:
        for iid in bbox_df["image_id"].astype(str):
            if iid not in image_ids:
                bbox_without_image += 1
                add_error("bbox_without_image", iid, "bbox image_id not in image table")

    # image -> metadata (dimensions) presence.
    image_without_metadata = 0
    for iid in image_ids:
        h, w = dims_by_image.get(iid, (None, None))
        if h is None or w is None:
            image_without_metadata += 1
            add_error("image_without_metadata", iid, "missing image dimensions")

    # bbox missing dimension.
    bbox_missing_dim = 0
    if not bbox_df.empty:
        miss = bbox_df[bbox_df["image_width"].isna() | bbox_df["image_height"].isna()]
        bbox_missing_dim = int(len(miss))
        for iid in miss["image_id"].astype(str).unique():
            add_error("bbox_missing_dimension", iid, "bbox row has no image dims")

    # bbox invalid (geometry).
    bbox_invalid = int((~bbox_df["is_valid_bbox"]).sum()) if not bbox_df.empty else 0

    # bbox with no canonical_class_id.
    if not bbox_df.empty:
        no_canon = bbox_df[bbox_df["canonical_class_id"].isna()]
        for iid in no_canon["image_id"].astype(str).unique():
            add_error("bbox_without_canonical_class", iid, "canonical_class_id missing")

    # No Finding must not be a detection class.
    nf_in_detection = bool(mapping_df["is_no_finding"].any())
    if nf_in_detection:
        add_error("no_finding_in_detection_classes", "-", "No Finding present in class mapping")

    # Count expectations (warnings, not hard errors unless DoD).
    canonical_image_rows = int(len(image_df))
    canonical_image_unique = int(image_df["image_id"].nunique())
    canonical_bbox_rows = int(len(bbox_df))
    canonical_class_count = int(len(mapping_df))
    abnormal_images = int(image_df["is_abnormal"].sum())
    no_finding_images = int(image_df["is_negative"].sum())

    if canonical_image_rows != EXPECT_TOTAL_IMAGES:
        warnings.append(f"canonical_image_rows={canonical_image_rows}, expected {EXPECT_TOTAL_IMAGES}")
    if abnormal_images != EXPECT_ABNORMAL_IMAGES:
        warnings.append(f"abnormal_images={abnormal_images}, expected {EXPECT_ABNORMAL_IMAGES}")
    if no_finding_images != EXPECT_NO_FINDING_IMAGES:
        warnings.append(f"no_finding_images={no_finding_images}, expected {EXPECT_NO_FINDING_IMAGES}")
    if canonical_bbox_rows != EXPECT_ABNORMAL_ROWS:
        warnings.append(f"canonical_bbox_rows={canonical_bbox_rows}, expected {EXPECT_ABNORMAL_ROWS}")
    if canonical_class_count != EXPECT_DETECTION_CLASSES:
        warnings.append(f"canonical_class_count={canonical_class_count}, expected {EXPECT_DETECTION_CLASSES}")

    # --- Portable path policy ---
    rel_missing = 0
    rel_absolute = 0
    local_absolute = 0
    if "relative_dicom_path" in image_df.columns:
        rel = image_df["relative_dicom_path"]
        rel_missing = int(rel.isna().sum() + (rel.astype(str).str.strip() == "").sum())
        rel_absolute = int(rel.astype(str).apply(_is_absolute_path).sum())
    else:
        rel_missing = canonical_image_rows
    if "local_dicom_path_is_absolute" in image_df.columns:
        local_absolute = int(
            image_df["local_dicom_path_is_absolute"].fillna(False).astype(bool).sum()
        )
    portable_path_policy_pass = bool(rel_missing == 0 and rel_absolute == 0)
    if rel_missing:
        add_error("relative_dicom_path_missing", "-", f"{rel_missing} rows missing relative_dicom_path")
    if rel_absolute:
        add_error("relative_dicom_path_absolute", "-", f"{rel_absolute} relative paths are absolute")

    errors_df = pd.DataFrame(
        errors, columns=["error_type", "reference", "detail"]
    )
    schema_error_count = int(len(errors_df))

    metrics = {
        "canonical_image_rows": canonical_image_rows,
        "canonical_image_unique_images": canonical_image_unique,
        "canonical_bbox_rows": canonical_bbox_rows,
        "canonical_class_count": canonical_class_count,
        "abnormal_images": abnormal_images,
        "no_finding_images": no_finding_images,
        "no_finding_policy_pass": bool(no_finding_policy_pass),
        "no_finding_in_detection_classes": nf_in_detection,
        "bbox_without_image_count": bbox_without_image,
        "image_without_metadata_count": image_without_metadata,
        "bbox_missing_dimension_count": bbox_missing_dim,
        "bbox_invalid_count": bbox_invalid,
        "class_mapping_issue_count": class_mapping_issue_count,
        "portable_path_policy_pass": portable_path_policy_pass,
        "relative_dicom_path_missing_count": rel_missing,
        "relative_dicom_path_absolute_count": rel_absolute,
        "local_dicom_path_absolute_count": local_absolute,
        "path_root_variable": PATH_ROOT_VARIABLE,
        "schema_error_count": schema_error_count,
    }
    return metrics, errors_df, warnings
