#!/usr/bin/env python3
"""Phase 2D — COCO Master Conversion & Validation.

Builds the REAL COCO detection JSON at data/processed/coco/coco_master.json from
the Phase 2B canonical schema, then validates it exhaustively.

HARD CONSTRAINT — NO IMAGE / DICOM ACCESS OF ANY KIND:
    This script never opens, reads, decodes, or even stats a .dicom file. It does
    not import pydicom, cv2, or PIL. `file_name`, `width` and `height` come from
    canonical_image_table.csv ONLY. pycocotools is used solely to parse the
    annotation JSON, never to read images.

Also forbidden here: splits, MMDetection/Detectron2 dataset loading, training,
inference, pseudo-labels, threshold tuning, test-set use, AP/mAP computation, and
any edit/clamp/delete/fuse/NMS of bboxes. All 36,096 bboxes are preserved
one-to-one.

`dataset_training_ready` is ALWAYS false: the DICOM loader belongs to Phase 2D.1.

Usage (Windows CMD):
    python scripts\\02D_build_coco_master.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    print(f"ERROR: pandas required but not importable: {exc!r}", file=sys.stderr)
    raise SystemExit(2)

try:
    import yaml
    _HAVE_YAML = True
except Exception:  # pragma: no cover
    yaml = None  # type: ignore
    _HAVE_YAML = False


# --- Protocol configuration (SOURCE OF TRUTH = the protocol YAML) ----------
#
# There are deliberately NO hard-coded expected counts or tolerances here.
# Every operational value below is loaded from
# configs/protocol/phase2D_coco_master_validation.yaml by load_protocol_strict()
# and then cross-checked against the Phase 2B validation JSON and the actual
# canonical tables. A duplicate source of truth is exactly what would let the
# YAML drift silently, so it is not kept.


class ProtocolError(RuntimeError):
    """Raised when the protocol YAML is missing, malformed, or self-inconsistent."""


class ProtocolDriftError(ProtocolError):
    """Raised when the protocol YAML disagrees with Phase 2B / the canonical data."""


class ProtocolConfig:
    """Typed, validated view of the Phase 2D protocol YAML.

    Constructed only via load_protocol_strict(). Every validation function takes
    this object rather than reading module-level constants.
    """

    __slots__ = (
        "expect_images",
        "expect_annotations",
        "expect_categories",
        "expect_abnormal_images",
        "expect_no_finding_images",
        "expect_no_finding_annotations",
        "area_rel_tol",
        "area_abs_tol",
        "boundary_abs_tol",
        "coord_abs_tol",
        "bbox_source_format",
        "bbox_target_format",
        "supercategory",
        "path_root_variable",
        "forbidden_category_names",
        "raw",
    )

    def __init__(self, **kw: Any) -> None:
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def as_dict(self) -> Dict[str, Any]:
        """The effective protocol actually driving this run."""
        return {
            "expected_counts": {
                "images": self.expect_images,
                "annotations": self.expect_annotations,
                "categories": self.expect_categories,
                "abnormal_images": self.expect_abnormal_images,
                "no_finding_images": self.expect_no_finding_images,
                "no_finding_annotations": self.expect_no_finding_annotations,
            },
            "tolerance": {
                "area_rel_tol": self.area_rel_tol,
                "area_abs_tol": self.area_abs_tol,
                "boundary_abs_tol": self.boundary_abs_tol,
                "coordinate_abs_tol": self.coord_abs_tol,
            },
            "bbox_source_format": self.bbox_source_format,
            "bbox_target_format": self.bbox_target_format,
            "supercategory": self.supercategory,
            "image_root_env_var": self.path_root_variable,
            "forbidden_category_names": sorted(self.forbidden_category_names),
        }


def _require_section(doc: Dict[str, Any], name: str) -> Dict[str, Any]:
    if name not in doc:
        raise ProtocolError(f"protocol YAML is missing required section '{name}'.")
    sec = doc[name]
    if not isinstance(sec, dict):
        raise ProtocolError(
            f"protocol YAML section '{name}' must be a mapping, got {type(sec).__name__}."
        )
    return sec


def _require_count(sec: Dict[str, Any], key: str, where: str) -> int:
    if key not in sec:
        raise ProtocolError(f"protocol YAML is missing required key '{where}.{key}'.")
    v = sec[key]
    # bool is a subclass of int; reject it explicitly.
    if isinstance(v, bool) or not isinstance(v, int):
        raise ProtocolError(
            f"protocol YAML '{where}.{key}' must be an integer, got "
            f"{type(v).__name__} ({v!r})."
        )
    if v < 0:
        raise ProtocolError(f"protocol YAML '{where}.{key}' must be >= 0, got {v}.")
    return int(v)


def _require_tol(sec: Dict[str, Any], key: str, where: str) -> float:
    if key not in sec:
        raise ProtocolError(f"protocol YAML is missing required key '{where}.{key}'.")
    v = sec[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ProtocolError(
            f"protocol YAML '{where}.{key}' must be a number, got "
            f"{type(v).__name__} ({v!r})."
        )
    f = float(v)
    if not math.isfinite(f):
        raise ProtocolError(f"protocol YAML '{where}.{key}' must be finite, got {f!r}.")
    if f < 0:
        raise ProtocolError(f"protocol YAML '{where}.{key}' must be >= 0, got {f}.")
    return f


def _require_str(sec: Dict[str, Any], key: str, where: str) -> str:
    if key not in sec:
        raise ProtocolError(f"protocol YAML is missing required key '{where}.{key}'.")
    v = sec[key]
    if not isinstance(v, str) or not v.strip():
        raise ProtocolError(
            f"protocol YAML '{where}.{key}' must be a non-empty string, got {v!r}."
        )
    return v.strip()


def load_protocol_strict(path: str) -> ProtocolConfig:
    """Load the Phase 2D protocol YAML. Hard-fails; never falls back to {}.

    Raises ProtocolError on: missing file, missing PyYAML, parse failure, non-dict
    document, missing section/key, wrong type, negative count/tolerance, or a
    non-finite tolerance.
    """
    p = Path(path)
    if not p.exists():
        raise ProtocolError(f"protocol YAML not found: {p}")
    if not _HAVE_YAML:
        raise ProtocolError(
            "PyYAML is not available, so the Phase 2D protocol cannot be enforced. "
            "Install pyyaml; Phase 2D refuses to run on hard-coded defaults."
        )
    try:
        doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ProtocolError(f"protocol YAML failed to parse: {exc!r}") from exc
    if not isinstance(doc, dict):
        raise ProtocolError(
            f"protocol YAML must be a mapping at the top level, got "
            f"{type(doc).__name__}."
        )

    counts = _require_section(doc, "expected_counts")
    tol = _require_section(doc, "tolerance")
    bbox = _require_section(doc, "bbox_policy")
    catp = _require_section(doc, "category_policy")
    pathp = _require_section(doc, "path_policy")

    forbidden_raw = catp.get("forbidden_category_names")
    if forbidden_raw is None:
        raise ProtocolError(
            "protocol YAML is missing required key 'category_policy."
            "forbidden_category_names'."
        )
    if not isinstance(forbidden_raw, list) or not forbidden_raw:
        raise ProtocolError(
            "protocol YAML 'category_policy.forbidden_category_names' must be a "
            "non-empty list."
        )
    forbidden = set()
    for item in forbidden_raw:
        if not isinstance(item, str) or not item.strip():
            raise ProtocolError(
                "protocol YAML 'category_policy.forbidden_category_names' entries "
                f"must be non-empty strings, got {item!r}."
            )
        forbidden.add(item.strip().lower())

    cfg = ProtocolConfig(
        expect_images=_require_count(counts, "images", "expected_counts"),
        expect_annotations=_require_count(counts, "annotations", "expected_counts"),
        expect_categories=_require_count(counts, "categories", "expected_counts"),
        expect_abnormal_images=_require_count(
            counts, "abnormal_images", "expected_counts"
        ),
        expect_no_finding_images=_require_count(
            counts, "no_finding_images", "expected_counts"
        ),
        expect_no_finding_annotations=_require_count(
            counts, "no_finding_annotations", "expected_counts"
        ),
        area_rel_tol=_require_tol(tol, "area_rel_tol", "tolerance"),
        area_abs_tol=_require_tol(tol, "area_abs_tol", "tolerance"),
        boundary_abs_tol=_require_tol(tol, "boundary_abs_tol", "tolerance"),
        coord_abs_tol=_require_tol(tol, "coordinate_abs_tol", "tolerance"),
        bbox_source_format=_require_str(bbox, "source_format", "bbox_policy"),
        bbox_target_format=_require_str(bbox, "target_format", "bbox_policy"),
        supercategory=_require_str(catp, "supercategory", "category_policy"),
        path_root_variable=_require_str(pathp, "image_root_env_var", "path_policy"),
        forbidden_category_names=forbidden,
        raw=doc,
    )

    # Internal coherence: the parts must sum to the whole.
    if cfg.expect_abnormal_images + cfg.expect_no_finding_images != cfg.expect_images:
        raise ProtocolError(
            "protocol YAML expected_counts are internally inconsistent: "
            f"abnormal_images({cfg.expect_abnormal_images}) + "
            f"no_finding_images({cfg.expect_no_finding_images}) != "
            f"images({cfg.expect_images})."
        )
    if cfg.expect_no_finding_annotations != 0:
        raise ProtocolError(
            "protocol YAML expected_counts.no_finding_annotations must be 0; got "
            f"{cfg.expect_no_finding_annotations}."
        )
    return cfg


def crosscheck_protocol(
    cfg: ProtocolConfig,
    phase2b: Dict[str, Any],
    actual_image_rows: int,
    actual_bbox_rows: int,
    actual_class_rows: int,
    actual_abnormal_images: int,
    actual_no_finding_images: int,
) -> List[str]:
    """Independently reconcile YAML <-> Phase 2B <-> the actual canonical tables.

    A YAML edited to legitimise a different dataset must not be able to pass. Every
    expected count has to agree on all three sides. Returns a list of drift messages
    (empty means no drift).
    """
    drift: List[str] = []

    def three_way(label: str, yaml_v: int, p2b_key: str, actual_v: int) -> None:
        p2b_v = phase2b.get(p2b_key)
        if yaml_v != p2b_v:
            drift.append(
                f"PROTOCOL_DRIFT: YAML expected_{label}={yaml_v} but Phase 2B "
                f"{p2b_key}={p2b_v}. Refusing to continue."
            )
        if yaml_v != actual_v:
            drift.append(
                f"PROTOCOL_DRIFT: YAML expected_{label}={yaml_v} but the actual "
                f"canonical table has {actual_v} row(s). Refusing to continue."
            )

    three_way("images", cfg.expect_images, "canonical_image_rows", actual_image_rows)
    three_way(
        "annotations", cfg.expect_annotations, "canonical_bbox_rows", actual_bbox_rows
    )
    three_way(
        "categories", cfg.expect_categories, "canonical_class_count", actual_class_rows
    )
    three_way(
        "abnormal_images",
        cfg.expect_abnormal_images,
        "abnormal_images",
        actual_abnormal_images,
    )
    three_way(
        "no_finding_images",
        cfg.expect_no_finding_images,
        "no_finding_images",
        actual_no_finding_images,
    )
    return drift


# --- Small helpers --------------------------------------------------------


def to_py(v: Any) -> Any:
    """Convert numpy/pandas scalars to plain Python scalars (JSON-safe)."""
    if v is None:
        return None
    if isinstance(v, (bool,)):
        return bool(v)
    # numpy bool_/int64/float64 expose .item()
    item = getattr(v, "item", None)
    if callable(item) and not isinstance(v, (str, bytes)):
        try:
            v = v.item()
        except Exception:
            pass
    if isinstance(v, float):
        return float(v)
    if isinstance(v, int):
        return int(v)
    return v


def to_int(v: Any) -> int:
    return int(to_py(v))


def to_float(v: Any) -> float:
    return float(to_py(v))


def is_finite(v: Any) -> bool:
    try:
        f = float(v)
    except Exception:
        return False
    return math.isfinite(f)


def sha256_file(path: str | Path) -> Optional[str]:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def normalize_rel_path(raw: Any) -> str:
    """Normalize a path separator to '/'. Never touches the filesystem."""
    s = str(raw).strip().replace("\\", "/")
    while "//" in s:
        s = s.replace("//", "/")
    return s


def path_is_absolute_like(s: str) -> bool:
    """True if the string looks absolute (drive letter, leading slash, UNC)."""
    if not s:
        return False
    if s.startswith("/"):
        return True
    if s.startswith("//"):
        return True
    if len(s) >= 2 and s[1] == ":":
        return True
    return False


def pick_column(
    df: pd.DataFrame, candidates: Sequence[str], what: str, required: bool = True
) -> Optional[str]:
    """Resolve a semantic field to a REAL column name. Never guesses by position."""
    for c in candidates:
        if c in df.columns:
            return c
    if not required:
        return None
    raise ValueError(
        f"Required field '{what}' not found. Tried column names: {list(candidates)}.\n"
        f"       Actual columns present: {list(df.columns)}"
    )


# --- Protocol / preflight -------------------------------------------------


def preflight(args: argparse.Namespace) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Check inputs exist and Phase 2B passed. Returns (phase2b, hard_errors, warnings).

    Expected COUNTS are NOT checked here: they come from the protocol YAML and are
    reconciled three ways in crosscheck_protocol(), once the canonical tables have
    actually been read.
    """
    hard_errors: List[str] = []
    warnings: List[str] = []

    required = {
        "canonical_image_table": args.canonical_image_table,
        "canonical_bbox_table": args.canonical_bbox_table,
        "canonical_class_mapping": args.canonical_class_mapping,
        "phase2b_validation_json": args.phase2b_validation_json,
    }
    for name, path in required.items():
        if not Path(path).exists():
            hard_errors.append(f"PREFLIGHT: required input missing: {name} -> {path}")

    if hard_errors:
        return {}, hard_errors, warnings

    try:
        phase2b = json.loads(Path(args.phase2b_validation_json).read_text(encoding="utf-8"))
    except Exception as exc:
        hard_errors.append(f"PREFLIGHT: cannot parse Phase 2B validation JSON: {exc!r}")
        return {}, hard_errors, warnings

    if phase2b.get("dod_pass_candidate") is not True:
        hard_errors.append(
            "PREFLIGHT: Phase 2B dod_pass_candidate is not true; Phase 2D refuses to run."
        )
    if phase2b.get("no_finding_in_detection_classes") is True:
        hard_errors.append("PREFLIGHT: Phase 2B reports No Finding in detection classes.")

    return phase2b, hard_errors, warnings


# --- Column mapping -------------------------------------------------------


def map_image_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols: Dict[str, Optional[str]] = {}
    cols["canonical_image_id"] = pick_column(
        df, ["canonical_image_id"], "canonical_image_id", required=False
    )
    cols["image_id"] = pick_column(
        df, ["image_id", "original_image_id"], "original image identifier"
    )
    cols["relative_dicom_path"] = pick_column(
        df, ["relative_dicom_path"], "relative_dicom_path"
    )
    cols["image_width"] = pick_column(df, ["image_width", "width"], "image_width")
    cols["image_height"] = pick_column(df, ["image_height", "height"], "image_height")
    # Negative / abnormal determination: from METADATA, not from zero annotations.
    cols["is_negative"] = pick_column(df, ["is_negative"], "is_negative", required=False)
    cols["is_abnormal"] = pick_column(df, ["is_abnormal"], "is_abnormal", required=False)
    cols["scope_label"] = pick_column(df, ["scope_label"], "scope_label", required=False)
    if cols["is_negative"] is None and cols["scope_label"] is None and cols["is_abnormal"] is None:
        raise ValueError(
            "Required field 'abnormal/No Finding indicator' not found. Need at least "
            "one of: is_negative, is_abnormal, scope_label.\n"
            f"       Actual columns present: {list(df.columns)}"
        )
    cols["bbox_count"] = pick_column(df, ["bbox_count"], "bbox_count", required=False)
    return cols


def map_bbox_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols: Dict[str, Optional[str]] = {}
    cols["canonical_ann_id"] = pick_column(df, ["canonical_ann_id"], "canonical_ann_id")
    cols["image_id"] = pick_column(df, ["image_id"], "bbox->image join key")
    cols["canonical_class_id"] = pick_column(
        df, ["canonical_class_id"], "bbox->class join key", required=False
    )
    cols["class_name"] = pick_column(df, ["class_name"], "class_name", required=False)
    if cols["canonical_class_id"] is None and cols["class_name"] is None:
        raise ValueError(
            "Required field 'class identifier for join' not found. Need "
            "canonical_class_id or class_name.\n"
            f"       Actual columns present: {list(df.columns)}"
        )
    for k in ("x_min", "y_min", "x_max", "y_max"):
        cols[k] = pick_column(df, [k], k)
    cols["source_row_id"] = pick_column(
        df, ["source_row_id"], "source_row_id", required=False
    )
    cols["rad_id"] = pick_column(df, ["rad_id"], "rad_id", required=False)
    cols["class_id_original"] = pick_column(
        df, ["class_id_original"], "class_id_original", required=False
    )
    cols["bbox_format"] = pick_column(df, ["bbox_format"], "bbox_format", required=False)
    return cols


def map_class_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols: Dict[str, Optional[str]] = {}
    cols["canonical_class_id"] = pick_column(df, ["canonical_class_id"], "canonical_class_id")
    cols["class_name"] = pick_column(df, ["class_name", "name"], "class name")
    cols["class_id_original"] = pick_column(
        df, ["class_id_original"], "class_id_original", required=False
    )
    cols["is_no_finding"] = pick_column(
        df, ["is_no_finding"], "is_no_finding", required=False
    )
    return cols


# --- COCO construction ----------------------------------------------------


def build_categories(
    class_df: pd.DataFrame,
    cmap: Dict[str, Optional[str]],
    cfg: ProtocolConfig,
    hard_errors: List[str],
) -> Tuple[List[Dict[str, Any]], Dict[Any, int]]:
    """Categories with contiguous ids 1..14 following canonical_class_id order."""
    work = class_df.copy()

    # Drop any No Finding row defensively (it must not be a detection category).
    if cmap["is_no_finding"]:
        nf_mask = work[cmap["is_no_finding"]].fillna(False).astype(bool)
        if bool(nf_mask.any()):
            hard_errors.append(
                "CATEGORY: canonical_class_mapping contains an is_no_finding=True row; "
                "No Finding must not be a detection class."
            )
    names_lower = work[cmap["class_name"]].astype(str).str.strip().str.lower()
    if bool(names_lower.isin(cfg.forbidden_category_names).any()):
        bad = sorted(set(names_lower[names_lower.isin(cfg.forbidden_category_names)]))
        hard_errors.append(f"CATEGORY: forbidden category name(s) present: {bad}")

    # Follow canonical order (never alphabetical).
    work = work.sort_values(cmap["canonical_class_id"]).reset_index(drop=True)

    categories: List[Dict[str, Any]] = []
    canon_to_cat: Dict[Any, int] = {}
    for i, row in work.iterrows():
        cat_id = int(i) + 1  # contiguous 1..N, never 0
        canon_id = to_int(row[cmap["canonical_class_id"]])
        cat: Dict[str, Any] = {
            "id": cat_id,
            "name": str(row[cmap["class_name"]]),
            "supercategory": cfg.supercategory,
            "canonical_class_id": canon_id,
        }
        if cmap["class_id_original"] and pd.notna(row[cmap["class_id_original"]]):
            cat["class_id_original"] = to_int(row[cmap["class_id_original"]])
        categories.append(cat)
        canon_to_cat[canon_id] = cat_id
    return categories, canon_to_cat


def build_images(
    image_df: pd.DataFrame,
    imap: Dict[str, Optional[str]],
    cfg: ProtocolConfig,
    hard_errors: List[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, Dict[str, Any]]]:
    """Images with contiguous ids 1..N sorted by canonical_image_id (fallback image_id)."""
    work = image_df.copy()
    sort_key = imap["canonical_image_id"] or imap["image_id"]
    work = work.sort_values(sort_key).reset_index(drop=True)

    images: List[Dict[str, Any]] = []
    imgid_to_cocoid: Dict[str, int] = {}
    meta_by_imgid: Dict[str, Dict[str, Any]] = {}

    for i, row in work.iterrows():
        coco_id = int(i) + 1
        orig_id = str(row[imap["image_id"]])

        raw_path = row[imap["relative_dicom_path"]]
        if pd.isna(raw_path) or not str(raw_path).strip():
            hard_errors.append(f"PATH: image '{orig_id}' has an empty relative_dicom_path.")
            file_name = ""
        else:
            file_name = normalize_rel_path(raw_path)
            if path_is_absolute_like(file_name):
                hard_errors.append(
                    f"PATH: image '{orig_id}' file_name is absolute-like: '{file_name}'."
                )

        w = row[imap["image_width"]]
        h = row[imap["image_height"]]
        if pd.isna(w) or pd.isna(h) or not is_finite(w) or not is_finite(h):
            hard_errors.append(f"IMAGE: image '{orig_id}' has non-finite width/height.")
            wi, hi = 0, 0
        else:
            wi, hi = to_int(w), to_int(h)
            if wi <= 0 or hi <= 0:
                hard_errors.append(
                    f"IMAGE: image '{orig_id}' has non-positive dimensions: {wi}x{hi}."
                )

        # Negative determination from METADATA (never from zero-annotation count).
        is_neg: Optional[bool] = None
        if imap["is_negative"] is not None and pd.notna(row[imap["is_negative"]]):
            is_neg = bool(row[imap["is_negative"]])
        elif imap["is_abnormal"] is not None and pd.notna(row[imap["is_abnormal"]]):
            is_neg = not bool(row[imap["is_abnormal"]])
        elif imap["scope_label"] is not None and pd.notna(row[imap["scope_label"]]):
            is_neg = str(row[imap["scope_label"]]).strip().lower() in (
                "no_finding", "no finding", "negative",
            )
        if is_neg is None:
            hard_errors.append(f"IMAGE: cannot determine negativity for image '{orig_id}'.")
            is_neg = False

        img: Dict[str, Any] = {
            "id": coco_id,
            "file_name": file_name,
            "width": wi,
            "height": hi,
            "original_image_id": orig_id,
            "is_negative": bool(is_neg),
        }
        if imap["canonical_image_id"]:
            img["canonical_image_id"] = to_int(row[imap["canonical_image_id"]])
        if imap["scope_label"] and pd.notna(row[imap["scope_label"]]):
            img["scope_label"] = str(row[imap["scope_label"]])
        images.append(img)

        imgid_to_cocoid[orig_id] = coco_id
        meta_by_imgid[orig_id] = {
            "coco_image_id": coco_id,
            "canonical_image_id": img.get("canonical_image_id"),
            "file_name": file_name,
            "is_negative": bool(is_neg),
            "width": wi,
            "height": hi,
            "canonical_bbox_count": (
                to_int(row[imap["bbox_count"]])
                if imap["bbox_count"] and pd.notna(row[imap["bbox_count"]])
                else None
            ),
        }
    return images, imgid_to_cocoid, meta_by_imgid


def build_annotations(
    bbox_df: pd.DataFrame,
    bmap: Dict[str, Optional[str]],
    imgid_to_cocoid: Dict[str, int],
    canon_to_cat: Dict[Any, int],
    name_to_cat: Dict[str, int],
    hard_errors: List[str],
) -> List[Dict[str, Any]]:
    """One COCO annotation per canonical bbox row. No clamp/delete/fuse/NMS/rounding."""
    work = bbox_df.copy()
    sort_key = bmap["canonical_ann_id"] or bmap["source_row_id"]
    work = work.sort_values(sort_key).reset_index(drop=True)

    annotations: List[Dict[str, Any]] = []
    for i, row in work.iterrows():
        ann_id = int(i) + 1
        orig_img = str(row[bmap["image_id"]])
        coco_img_id = imgid_to_cocoid.get(orig_img)
        if coco_img_id is None:
            hard_errors.append(
                f"REFERENCE: annotation references unknown image_id '{orig_img}'."
            )
            continue

        # Category resolution: canonical_class_id first, else class_name.
        cat_id: Optional[int] = None
        canon_cls: Optional[int] = None
        if bmap["canonical_class_id"] and pd.notna(row[bmap["canonical_class_id"]]):
            canon_cls = to_int(row[bmap["canonical_class_id"]])
            cat_id = canon_to_cat.get(canon_cls)
        if cat_id is None and bmap["class_name"] and pd.notna(row[bmap["class_name"]]):
            cat_id = name_to_cat.get(str(row[bmap["class_name"]]))
        if cat_id is None:
            hard_errors.append(
                f"REFERENCE: annotation (canonical_ann_id="
                f"{row[bmap['canonical_ann_id']]}) has no resolvable category."
            )
            continue

        x0 = to_float(row[bmap["x_min"]])
        y0 = to_float(row[bmap["y_min"]])
        x1 = to_float(row[bmap["x_max"]])
        y1 = to_float(row[bmap["y_max"]])
        # Exact conversion. No clamping, no rounding, no deletion.
        w = x1 - x0
        h = y1 - y0
        area = w * h

        ann: Dict[str, Any] = {
            "id": ann_id,
            "image_id": int(coco_img_id),
            "category_id": int(cat_id),
            "bbox": [x0, y0, w, h],
            "area": area,
            "iscrowd": 0,
            "canonical_ann_id": to_int(row[bmap["canonical_ann_id"]]),
            "original_image_id": orig_img,
        }
        if canon_cls is not None:
            ann["canonical_class_id"] = canon_cls
        if bmap["source_row_id"] and pd.notna(row[bmap["source_row_id"]]):
            ann["source_row_id"] = to_int(row[bmap["source_row_id"]])
        if bmap["rad_id"] and pd.notna(row[bmap["rad_id"]]):
            ann["rad_id"] = str(row[bmap["rad_id"]])
        if bmap["class_id_original"] and pd.notna(row[bmap["class_id_original"]]):
            ann["class_id_original"] = to_int(row[bmap["class_id_original"]])
        annotations.append(ann)
    return annotations


# --- Validation -----------------------------------------------------------


def validate_coco(
    coco: Dict[str, Any],
    bbox_df: pd.DataFrame,
    bmap: Dict[str, Optional[str]],
    meta_by_imgid: Dict[str, Dict[str, Any]],
    canon_to_cat: Dict[Any, int],
    cfg: ProtocolConfig,
    hard_errors: List[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Full internal validation. Returns (validation_blocks, invalid_annotation_rows)."""
    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]
    invalid_rows: List[Dict[str, Any]] = []

    def flag(ann: Dict[str, Any], reason: str) -> None:
        invalid_rows.append(
            {
                "annotation_id": ann.get("id"),
                "canonical_ann_id": ann.get("canonical_ann_id"),
                "image_id": ann.get("image_id"),
                "category_id": ann.get("category_id"),
                "reason": reason,
                "bbox": json.dumps(ann.get("bbox")),
                "area": ann.get("area"),
                "image_width": img_dims.get(ann.get("image_id"), (None, None))[0],
                "image_height": img_dims.get(ann.get("image_id"), (None, None))[1],
            }
        )

    # ---- Images
    img_ids = [im["id"] for im in images]
    file_names = [im["file_name"] for im in images]
    img_dims = {im["id"]: (im["width"], im["height"]) for im in images}
    abs_paths = [f for f in file_names if path_is_absolute_like(f)]
    nonpos_dims = [im["id"] for im in images if im["width"] <= 0 or im["height"] <= 0]
    abnormal_count = sum(1 for im in images if not im.get("is_negative"))
    negative_count = sum(1 for im in images if im.get("is_negative"))

    image_validation = {
        "image_count": len(images),
        "image_count_pass": len(images) == cfg.expect_images,
        "image_ids_unique": len(set(img_ids)) == len(img_ids),
        "image_ids_contiguous_from_1": sorted(img_ids) == list(range(1, len(images) + 1)),
        "file_names_unique": len(set(file_names)) == len(file_names),
        "width_height_positive": len(nonpos_dims) == 0,
        "non_positive_dim_count": len(nonpos_dims),
        "absolute_path_count": len(abs_paths),
        "relative_path_only": len(abs_paths) == 0,
        "abnormal_images": abnormal_count,
        "abnormal_images_pass": abnormal_count == cfg.expect_abnormal_images,
        "no_finding_images": negative_count,
        "no_finding_images_pass": negative_count == cfg.expect_no_finding_images,
    }
    if not image_validation["image_count_pass"]:
        hard_errors.append(f"IMAGES: count={len(images)}, expected {cfg.expect_images}.")
    if not image_validation["image_ids_unique"]:
        hard_errors.append("IMAGES: image ids are not unique.")
    if not image_validation["file_names_unique"]:
        hard_errors.append("IMAGES: file_name values are not unique.")
    if not image_validation["relative_path_only"]:
        hard_errors.append(f"IMAGES: {len(abs_paths)} absolute-like file_name(s).")
    if not image_validation["width_height_positive"]:
        hard_errors.append(f"IMAGES: {len(nonpos_dims)} image(s) with non-positive dims.")
    if not image_validation["abnormal_images_pass"]:
        hard_errors.append(
            f"IMAGES: abnormal={abnormal_count}, expected {cfg.expect_abnormal_images}."
        )
    if not image_validation["no_finding_images_pass"]:
        hard_errors.append(
            f"IMAGES: No Finding={negative_count}, expected {cfg.expect_no_finding_images}."
        )

    # ---- Categories
    cat_ids = [c["id"] for c in categories]
    cat_names = [c["name"] for c in categories]
    cat_names_lower = {n.strip().lower() for n in cat_names}
    forbidden_present = sorted(cat_names_lower & cfg.forbidden_category_names)
    category_validation = {
        "category_count": len(categories),
        "category_count_pass": len(categories) == cfg.expect_categories,
        "ids_contiguous_1_to_n": sorted(cat_ids) == list(range(1, len(categories) + 1)),
        "ids_unique": len(set(cat_ids)) == len(cat_ids),
        "names_unique": len(set(cat_names)) == len(cat_names),
        "category_id_zero_present": 0 in cat_ids,
        "no_finding_absent": "no finding" not in cat_names_lower,
        "background_absent": "background" not in cat_names_lower,
        "normal_absent": "normal" not in cat_names_lower,
        "forbidden_names_present": forbidden_present,
        "supercategory_uniform": len({c.get("supercategory") for c in categories}) == 1,
    }
    if not category_validation["category_count_pass"]:
        hard_errors.append(f"CATEGORIES: count={len(categories)}, expected {cfg.expect_categories}.")
    if not category_validation["ids_contiguous_1_to_n"]:
        hard_errors.append("CATEGORIES: ids are not contiguous 1..N.")
    if category_validation["category_id_zero_present"]:
        hard_errors.append("CATEGORIES: category id 0 is present (forbidden).")
    if forbidden_present:
        hard_errors.append(f"CATEGORIES: forbidden names present: {forbidden_present}.")
    if not category_validation["names_unique"]:
        hard_errors.append("CATEGORIES: names are not unique.")

    # ---- Annotations (structure, geometry, boundary, area)
    ann_ids = [a["id"] for a in annotations]
    valid_img_ids = set(img_ids)
    valid_cat_ids = set(cat_ids)
    broken_ref = 0
    bad_bbox = 0
    bad_area = 0
    bad_boundary = 0
    bad_iscrowd = 0

    for a in annotations:
        bbox = a.get("bbox")
        if a["image_id"] not in valid_img_ids:
            broken_ref += 1
            flag(a, "image_id_not_found")
            continue
        if a["category_id"] not in valid_cat_ids:
            broken_ref += 1
            flag(a, "category_id_not_found")
            continue
        if not isinstance(bbox, list) or len(bbox) != 4 or not all(is_finite(v) for v in bbox):
            bad_bbox += 1
            flag(a, "bbox_not_four_finite_values")
            continue
        x, y, w, h = (float(v) for v in bbox)
        if x < 0 or y < 0:
            bad_bbox += 1
            flag(a, "negative_origin")
            continue
        if w <= 0 or h <= 0:
            bad_bbox += 1
            flag(a, "non_positive_width_or_height")
            continue
        iw, ih = img_dims[a["image_id"]]
        if x + w > iw + cfg.boundary_abs_tol or y + h > ih + cfg.boundary_abs_tol:
            bad_boundary += 1
            flag(a, "bbox_exceeds_image_boundary")
            continue
        if not math.isclose(float(a["area"]), w * h, rel_tol=cfg.area_rel_tol, abs_tol=cfg.area_abs_tol):
            bad_area += 1
            flag(a, "area_mismatch")
            continue
        if a.get("iscrowd") != 0:
            bad_iscrowd += 1
            flag(a, "iscrowd_not_zero")
            continue

    annotation_validation = {
        "annotation_count": len(annotations),
        "annotation_count_pass": len(annotations) == cfg.expect_annotations,
        "annotation_ids_unique": len(set(ann_ids)) == len(ann_ids),
        "annotation_ids_contiguous_from_1": sorted(ann_ids)
        == list(range(1, len(annotations) + 1)),
        "broken_reference_count": broken_ref,
        "all_image_ids_exist": broken_ref == 0,
        "all_category_ids_exist": broken_ref == 0,
        "invalid_bbox_count": bad_bbox,
        "boundary_violation_count": bad_boundary,
        "area_mismatch_count": bad_area,
        "iscrowd_violation_count": bad_iscrowd,
        "invalid_annotation_total": len(invalid_rows),
    }
    if not annotation_validation["annotation_count_pass"]:
        hard_errors.append(
            f"ANNOTATIONS: count={len(annotations)}, expected {cfg.expect_annotations}."
        )
    if not annotation_validation["annotation_ids_unique"]:
        hard_errors.append("ANNOTATIONS: ids are not unique.")
    if invalid_rows:
        hard_errors.append(f"ANNOTATIONS: {len(invalid_rows)} invalid annotation(s).")

    relationship_validation = {
        "all_image_ids_exist": broken_ref == 0,
        "all_category_ids_exist": broken_ref == 0,
        "broken_reference_count": broken_ref,
    }
    bbox_validation = {
        "source_format": cfg.bbox_source_format,
        "target_format": cfg.bbox_target_format,
        "invalid_bbox_count": bad_bbox,
        "clamped": False,
        "deleted": False,
        "fused": False,
        "nms_applied": False,
        "rounded": False,
    }
    area_validation = {
        "area_formula": "width * height",
        "area_rel_tol": cfg.area_rel_tol,
        "area_abs_tol": cfg.area_abs_tol,
        "area_mismatch_count": bad_area,
        "area_pass": bad_area == 0,
    }
    boundary_validation = {
        "rule": "x + width <= image_width AND y + height <= image_height",
        "boundary_abs_tol": cfg.boundary_abs_tol,
        "boundary_violation_count": bad_boundary,
        "boundary_pass": bad_boundary == 0,
    }

    # ---- Traceability: compare every annotation against the canonical source.
    src = bbox_df.set_index(bmap["canonical_ann_id"])
    coord_mismatch = 0
    image_map_mismatch = 0
    category_map_mismatch = 0
    for a in annotations:
        cid = a.get("canonical_ann_id")
        if cid is None or cid not in src.index:
            coord_mismatch += 1
            continue
        row = src.loc[cid]
        x0 = to_float(row[bmap["x_min"]])
        y0 = to_float(row[bmap["y_min"]])
        x1 = to_float(row[bmap["x_max"]])
        y1 = to_float(row[bmap["y_max"]])
        bx, by, bw, bh = (float(v) for v in a["bbox"])
        if not (
            math.isclose(bx, x0, abs_tol=cfg.coord_abs_tol)
            and math.isclose(by, y0, abs_tol=cfg.coord_abs_tol)
            and math.isclose(bw, x1 - x0, abs_tol=cfg.coord_abs_tol)
            and math.isclose(bh, y1 - y0, abs_tol=cfg.coord_abs_tol)
        ):
            coord_mismatch += 1
        # image mapping
        src_img = str(row[bmap["image_id"]])
        expect_coco_img = meta_by_imgid.get(src_img, {}).get("coco_image_id")
        if expect_coco_img != a["image_id"]:
            image_map_mismatch += 1
        # category mapping
        if bmap["canonical_class_id"] and pd.notna(row[bmap["canonical_class_id"]]):
            expect_cat = canon_to_cat.get(to_int(row[bmap["canonical_class_id"]]))
            if expect_cat != a["category_id"]:
                category_map_mismatch += 1

    traceability_validation = {
        "coordinate_mismatch_count": coord_mismatch,
        "image_mapping_mismatch_count": image_map_mismatch,
        "category_mapping_mismatch_count": category_map_mismatch,
        "coordinate_abs_tol": cfg.coord_abs_tol,
        "traceability_pass": (
            coord_mismatch == 0 and image_map_mismatch == 0 and category_map_mismatch == 0
        ),
        "fields_preserved": [
            "canonical_ann_id",
            "source_row_id",
            "original_image_id",
            "rad_id",
            "canonical_class_id",
            "class_id_original",
        ],
    }
    if coord_mismatch:
        hard_errors.append(f"TRACEABILITY: {coord_mismatch} coordinate mismatch(es) vs canonical.")
    if image_map_mismatch:
        hard_errors.append(f"TRACEABILITY: {image_map_mismatch} image mapping mismatch(es).")
    if category_map_mismatch:
        hard_errors.append(
            f"TRACEABILITY: {category_map_mismatch} category mapping mismatch(es)."
        )

    # ---- One-to-one preservation (evidence that no bbox was dropped or fused).
    canon_src_ids = set(to_int(v) for v in bbox_df[bmap["canonical_ann_id"]].tolist())
    coco_canon_ids_list = [a.get("canonical_ann_id") for a in annotations]
    coco_canon_ids = set(v for v in coco_canon_ids_list if v is not None)
    missing = canon_src_ids - coco_canon_ids
    extra = coco_canon_ids - canon_src_ids
    dup_count = len(coco_canon_ids_list) - len(coco_canon_ids)

    one_to_one = {
        "canonical_bbox_rows": len(bbox_df),
        "coco_annotation_rows": len(annotations),
        "canonical_ann_id_sets_equal": canon_src_ids == coco_canon_ids,
        "missing_canonical_annotation_count": len(missing),
        "duplicated_canonical_annotation_count": dup_count,
        "extra_coco_annotation_count": len(extra),
        "one_to_one_pass": (
            len(bbox_df) == len(annotations)
            and canon_src_ids == coco_canon_ids
            and dup_count == 0
        ),
        "note": (
            "Set equality of canonical_ann_id is the evidence that no bbox — including "
            "near-duplicate candidates — was deleted or fused. Phase 2D did NOT re-run "
            "near-duplicate detection; the candidate file is not an input here."
        ),
    }
    if not one_to_one["one_to_one_pass"]:
        hard_errors.append(
            f"ONE-TO-ONE: missing={len(missing)}, duplicated={dup_count}, extra={len(extra)}."
        )

    # ---- No Finding (determined from canonical METADATA, not zero-annotation).
    ann_count_by_img: Dict[int, int] = {}
    for a in annotations:
        ann_count_by_img[a["image_id"]] = ann_count_by_img.get(a["image_id"], 0) + 1

    nf_rows: List[Dict[str, Any]] = []
    nf_with_ann = 0
    for im in images:
        if not im.get("is_negative"):
            continue
        n = ann_count_by_img.get(im["id"], 0)
        if n != 0:
            nf_with_ann += 1
        nf_rows.append(
            {
                "coco_image_id": im["id"],
                "canonical_image_id": im.get("canonical_image_id"),
                "original_image_id": im.get("original_image_id"),
                "file_name": im["file_name"],
                "source_is_negative": True,
                "coco_annotation_count": n,
                "zero_annotation_pass": n == 0,
            }
        )
    no_finding_validation = {
        "no_finding_images": len(nf_rows),
        "no_finding_images_pass": len(nf_rows) == cfg.expect_no_finding_images,
        "determined_from": "canonical_image_metadata_not_zero_annotation",
        "no_finding_with_annotation_count": nf_with_ann,
        "all_zero_annotation": nf_with_ann == 0,
        "no_finding_in_categories": "no finding" in cat_names_lower,
        "negative_images_lost": cfg.expect_no_finding_images - len(nf_rows),
    }
    if not no_finding_validation["no_finding_images_pass"]:
        hard_errors.append(
            f"NO FINDING: {len(nf_rows)} negative images, expected {cfg.expect_no_finding_images}."
        )
    if nf_with_ann:
        hard_errors.append(f"NO FINDING: {nf_with_ann} negative image(s) carry annotations.")

    path_validation = {
        "file_name_source": "relative_dicom_path",
        "local_dicom_path_used": False,
        "separator": "/",
        "absolute_path_count": len(abs_paths),
        "strict_relative_pass": len(abs_paths) == 0,
        "resolved_against_filesystem": False,
        "image_root_env_var": cfg.path_root_variable,
    }

    id_validation = {
        "image_ids_unique": image_validation["image_ids_unique"],
        "image_ids_contiguous_from_1": image_validation["image_ids_contiguous_from_1"],
        "annotation_ids_unique": annotation_validation["annotation_ids_unique"],
        "annotation_ids_contiguous_from_1": annotation_validation[
            "annotation_ids_contiguous_from_1"
        ],
        "category_ids_contiguous_1_to_n": category_validation["ids_contiguous_1_to_n"],
        "category_id_zero_present": category_validation["category_id_zero_present"],
    }

    blocks = {
        "image_validation": image_validation,
        "annotation_validation": annotation_validation,
        "category_validation": category_validation,
        "relationship_validation": relationship_validation,
        "bbox_validation": bbox_validation,
        "area_validation": area_validation,
        "boundary_validation": boundary_validation,
        "no_finding_validation": no_finding_validation,
        "path_validation": path_validation,
        "traceability_validation": traceability_validation,
        "one_to_one_preservation": one_to_one,
        "id_validation": id_validation,
        "_no_finding_rows": nf_rows,
        "_ann_count_by_img": ann_count_by_img,
    }
    return blocks, invalid_rows


def run_pycocotools(json_path: str, hard_errors: List[str], warnings: List[str]) -> Dict[str, Any]:
    """Parse the annotation JSON with pycocotools. NEVER used to read images."""
    out: Dict[str, Any] = {
        "used_for": "annotation_json_parsing_only",
        "pycocotools_available": False,
        "pycocotools_load_pass": False,
        "exception": None,
        "images_loaded": None,
        "annotations_loaded": None,
        "categories_loaded": None,
    }
    try:
        from pycocotools.coco import COCO  # noqa: WPS433
    except ImportError as exc:
        out["exception"] = repr(exc)
        warnings.append(
            "pycocotools is not installed; COCO-load verification was skipped. "
            "Internal validation still ran in full."
        )
        return out

    out["pycocotools_available"] = True
    try:
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            c = COCO(json_path)
        out["images_loaded"] = len(c.imgs)
        out["annotations_loaded"] = len(c.anns)
        out["categories_loaded"] = len(c.cats)
        out["pycocotools_load_pass"] = True
    except Exception as exc:
        out["exception"] = repr(exc)
        hard_errors.append(
            f"PYCOCOTOOLS: installed but failed to load the COCO JSON: {exc!r}"
        )
    return out


# --- Markdown report ------------------------------------------------------


def write_report_md(path: str, p: Dict[str, Any], cfg: ProtocolConfig) -> None:
    L: List[str] = []
    A = L.append
    A("# Phase 2D — COCO Master Conversion & Validation")
    A("")
    A(f"_Generated {p['created_utc']}._")
    A("")
    A("## Objective and scope")
    A("")
    A(
        "Convert the Phase 2B canonical schema into a real COCO detection JSON at "
        f"`{p['output_path']}` and validate it exhaustively. This phase reorganizes "
        "METADATA ONLY."
    )
    A("")
    A(
        "**No image or DICOM access of any kind occurred.** No `.dicom` file was "
        "read, opened, or even checked for existence; `pydicom`, `cv2`, and `PIL` "
        "were never imported. `file_name`, `width`, and `height` come exclusively "
        "from `canonical_image_table.csv`."
    )
    A("")
    A("## Input / output")
    A("")
    A("| Role | Path | SHA-256 |")
    A("|---|---|---|")
    for k, v in p["input_paths"].items():
        A(f"| input: {k} | `{v}` | `{(p['input_sha256'].get(k) or 'n/a')[:16]}…` |")
    A(f"| output | `{p['output_path']}` | `{(p['output_sha256'] or 'n/a')[:16]}…` |")
    A("")
    A("## COCO schema")
    A("")
    A("- `images[]`: id, file_name, width, height (+ canonical_image_id, original_image_id, scope_label, is_negative)")
    A("- `annotations[]`: id, image_id, category_id, bbox, area, iscrowd (+ traceability fields)")
    A(f"- `categories[]`: id, name, supercategory=`{cfg.supercategory}`, canonical_class_id, class_id_original")
    A("")
    A("## ID policy")
    A("")
    A(f"- image id: contiguous 1..{p['counts']['images']}, sorted by `canonical_image_id`.")
    A(f"- annotation id: contiguous 1..{p['counts']['annotations']}, sorted by `canonical_ann_id`.")
    A(f"- category id: contiguous 1..{p['counts']['categories']}, following canonical_class_id order (never alphabetical). Category id 0 is never used.")
    A("")
    A("## BBox conversion policy")
    A("")
    A(f"- `{cfg.bbox_source_format}` → `{cfg.bbox_target_format}`")
    A("- x = x_min; y = y_min; width = x_max − x_min; height = y_max − y_min; area = width × height; iscrowd = 0.")
    A("- **No clamping, no deletion, no fusion, no NMS, no rounding.**")
    A("")
    A("## Count summary")
    A("")
    A("| Item | Value | Expected | Pass |")
    A("|---|---|---|---|")
    c = p["counts"]
    rows = [
        ("images", c["images"], cfg.expect_images),
        ("annotations", c["annotations"], cfg.expect_annotations),
        ("categories", c["categories"], cfg.expect_categories),
        ("abnormal images", c["abnormal_images"], cfg.expect_abnormal_images),
        ("No Finding images", c["no_finding_images"], cfg.expect_no_finding_images),
        ("No Finding annotations", c["no_finding_annotations"], 0),
    ]
    for name, got, want in rows:
        A(f"| {name} | {got} | {want} | {'PASS' if got == want else 'FAIL'} |")
    A("")
    A("## Validation results")
    A("")
    A("| Check | Result | Status |")
    A("|---|---|---|")
    for label, ok in p["dod_checklist"].items():
        A(f"| {label} | {ok} | {'PASS' if ok else 'FAIL'} |")
    A("")
    A("## No Finding audit")
    A("")
    nf = p["no_finding_validation"]
    A(f"- Negative images are determined from **canonical image metadata**, not from a zero-annotation count (`{nf['determined_from']}`).")
    A(f"- No Finding images: **{nf['no_finding_images']}** (expected {cfg.expect_no_finding_images}).")
    A(f"- All carry zero annotations: **{nf['all_zero_annotation']}**.")
    A(f"- No Finding present in categories: **{nf['no_finding_in_categories']}** (must be false).")
    A("- Per-image evidence: `phase2D_coco_no_finding_audit.csv` (500 rows).")
    A("")
    A("## Category summary")
    A("")
    A("| id | name | canonical_class_id | annotations | images |")
    A("|---|---|---|---|---|")
    for row in p["_category_rows"]:
        A(
            f"| {row['category_id']} | {row['name']} | {row['canonical_class_id']} | "
            f"{row['annotation_count']} | {row['unique_image_count']} |"
        )
    A("")
    A("## Traceability preservation")
    A("")
    t = p["traceability_validation"]
    o = p["one_to_one_preservation"]
    A(f"- Coordinate/image/category mismatches vs canonical: {t['coordinate_mismatch_count']} / {t['image_mapping_mismatch_count']} / {t['category_mapping_mismatch_count']}.")
    A(f"- canonical_ann_id sets equal: **{o['canonical_ann_id_sets_equal']}**; missing={o['missing_canonical_annotation_count']}, duplicated={o['duplicated_canonical_annotation_count']}, extra={o['extra_coco_annotation_count']}.")
    A(f"- {o['note']}")
    A("")
    A("## pycocotools result")
    A("")
    pc = p["pycocotools_validation"]
    A(f"- available: {pc['pycocotools_available']}; load pass: {pc['pycocotools_load_pass']}.")
    if pc["exception"]:
        A(f"- exception: `{pc['exception']}`")
    A(f"- Used for **{pc['used_for']}** — never to read images.")
    A("")
    A("## Protocol enforcement")
    A("")
    pv = p["protocol_validation"]
    A(f"- Protocol YAML: `{pv['yaml_path']}`")
    A(f"- Strict schema load: **{pv['strict_schema_pass']}** (no silent fallback; a missing/malformed/negative/non-finite value aborts the run).")
    A(f"- Phase 2B cross-check: **{pv['phase2b_crosscheck_pass']}**; protocol drift count: **{pv['protocol_drift_count']}**.")
    A("- Every expected count is reconciled three ways — protocol YAML ↔ Phase 2B validation JSON ↔ the actual canonical tables — so a YAML edited to legitimise a different dataset cannot pass.")
    A("")
    A("Effective protocol driving this run:")
    A("")
    A("```json")
    A(json.dumps(pv["effective_protocol"], indent=2))
    A("```")
    A("")
    A("## Atomic output validation")
    A("")
    ao = p["atomic_output_validation"]
    A("| Check | Value |")
    A("|---|---|")
    for k in (
        "validation_completed_before_promotion",
        "per_image_count_checked_before_promotion",
        "temporary_file_used",
        "temporary_written",
        "temporary_json_parse_pass" if "temporary_json_parse_pass" in ao else "final_reparse_pass",
        "pycocotools_checked_before_promotion",
        "all_pre_promotion_checks_pass",
        "final_output_replaced",
        "previous_valid_output_preserved_on_failure",
    ):
        if k in ao:
            A(f"| {k} | {ao[k]} |")
    A(f"| temporary_json_parse_pass | {p['json_validation']['temporary_json_parse_pass']} |")
    A("")
    A("The final `coco_master.json` is replaced with `os.replace()` **only after** every hard check — including per-image `canonical_bbox_count == coco_annotation_count` and the pycocotools load — has passed on a temporary file. On any failure the temporary file is removed in a `finally` block and any pre-existing final output is left byte-for-byte untouched.")
    A("")
    A("## Forbidden actions confirmation")
    A("")
    for k, v in p["forbidden_actions"].items():
        A(f"- {k}: {v}")
    A("")
    A("## Warnings")
    A("")
    if p["warnings"]:
        for w in p["warnings"]:
            A(f"- {w}")
    else:
        A("- none")
    A("")
    A("## Limitations")
    A("")
    A("- A valid COCO annotation file does **not** make the dataset training-ready.")
    A("- The DICOM loader is NOT validated; MMDetection's default `LoadImageFromFile` cannot read `.dicom`.")
    A("- Empty-image (No Finding) loading behaviour is NOT validated here; `filter_empty_gt` was deliberately not checked.")
    A("- Near-duplicate detection was NOT re-run in Phase 2D; one-to-one preservation is the evidence that nothing was dropped or fused.")
    A("")
    A(f"- **dataset_training_ready = {p['dataset_training_ready']}**")
    A("")
    A("## Next phase")
    A("")
    A("- Phase 2D.1 (DICOM loader / empty-image loading) remains **LOCKED** until GPT review PASS of this evidence.")
    A("- This script does not and cannot conclude a GPT review verdict.")
    A("")
    ensure_parent(path).write_text("\n".join(L), encoding="utf-8")


# --- Main -----------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2D — build and validate the COCO master (metadata only).",
    )
    p.add_argument("--canonical-image-table", type=str,
                   default="data/processed/canonical/canonical_image_table.csv")
    p.add_argument("--canonical-bbox-table", type=str,
                   default="data/processed/canonical/canonical_bbox_table.csv")
    p.add_argument("--canonical-class-mapping", type=str,
                   default="data/processed/canonical/canonical_class_mapping.csv")
    p.add_argument("--phase2b-validation-json", type=str,
                   default="reports/phase2B_canonical_schema_validation.json")
    p.add_argument("--protocol-yaml", type=str,
                   default="configs/protocol/phase2D_coco_master_validation.yaml")
    p.add_argument("--coco-master", type=str,
                   default="data/processed/coco/coco_master.json")
    p.add_argument("--report-md", type=str,
                   default="reports/phase2D_coco_master_validation.md")
    p.add_argument("--validation-json", type=str,
                   default="reports/phase2D_coco_master_validation.json")
    p.add_argument("--image-counts-csv", type=str,
                   default="reports/phase2D_coco_image_annotation_counts.csv")
    p.add_argument("--category-summary-csv", type=str,
                   default="reports/phase2D_coco_category_summary.csv")
    p.add_argument("--invalid-annotations-csv", type=str,
                   default="reports/phase2D_coco_invalid_annotations.csv")
    p.add_argument("--no-finding-audit-csv", type=str,
                   default="reports/phase2D_coco_no_finding_audit.csv")
    return p.parse_args(argv)


def promote_atomic(
    tmp_path: str,
    final_path: str | Path,
    all_pre_promotion_checks_pass: bool,
) -> Dict[str, Any]:
    """STEP 6/7 — promote temp -> final ONLY if every pre-promotion check passed.

    On failure the temp file is removed and any pre-existing final output is left
    byte-for-byte untouched. Returns the atomic_output_validation block.
    """
    final = Path(final_path)
    previous_existed = final.exists()
    previous_sha = sha256_file(final) if previous_existed else None

    # tmp_path may legitimately be empty/None when internal validation failed before
    # any temporary file was created. Path("") resolves to ".", so guard explicitly.
    has_tmp = bool(tmp_path) and str(tmp_path).strip() != ""
    tmp = Path(tmp_path) if has_tmp else None
    temp_exists = bool(tmp is not None and tmp.is_file())

    out: Dict[str, Any] = {
        "temporary_file_used": temp_exists,
        "temporary_written": temp_exists,
        "all_pre_promotion_checks_pass": bool(all_pre_promotion_checks_pass),
        "final_output_replaced": False,
        "previous_output_existed": previous_existed,
        "previous_valid_output_preserved": True,
        "previous_valid_output_preserved_on_failure": True,
        "final_output_sha256": None,
        "final_reparse_pass": None,
    }

    if not (all_pre_promotion_checks_pass and temp_exists):
        # Failure path: remove the temp file (if any) and never touch the final output.
        if temp_exists and tmp is not None:
            tmp.unlink(missing_ok=True)
        if previous_existed:
            still = sha256_file(final)
            out["previous_valid_output_preserved"] = still == previous_sha
            out["previous_valid_output_preserved_on_failure"] = still == previous_sha
        return out

    os.replace(tmp_path, final)  # atomic on the same filesystem
    out["final_output_replaced"] = final.exists()
    out["final_output_sha256"] = sha256_file(final)
    # Confirm the rename produced a parseable file.
    try:
        with final.open("r", encoding="utf-8") as fh:
            json.load(fh)
        out["final_reparse_pass"] = True
    except Exception:
        out["final_reparse_pass"] = False
    # On the success path the previous output is intentionally replaced.
    out["previous_valid_output_preserved"] = not previous_existed
    out["previous_valid_output_preserved_on_failure"] = True
    return out


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    hard_errors: List[str] = []
    warnings: List[str] = []
    tmp_name: Optional[str] = None

    try:
        # ================= STEP 1 — PREFLIGHT =================
        # 1a. Strict protocol load. No silent fallback; hard-fail on any defect.
        protocol_validation: Dict[str, Any] = {
            "yaml_path": str(args.protocol_yaml),
            "yaml_loaded": False,
            "strict_schema_pass": False,
            "phase2b_crosscheck_pass": False,
            "protocol_drift_count": 0,
            "protocol_drift": [],
            "effective_protocol": None,
        }
        try:
            cfg = load_protocol_strict(args.protocol_yaml)
            protocol_validation["yaml_loaded"] = True
            protocol_validation["strict_schema_pass"] = True
            protocol_validation["effective_protocol"] = cfg.as_dict()
        except ProtocolError as exc:
            print(f"ERROR: PROTOCOL: {exc}", file=sys.stderr)
            print(
                "\nPhase 2D aborted at preflight (strict protocol load failed). "
                "No COCO file was written or replaced.",
                file=sys.stderr,
            )
            return 2

        # 1b. Phase 2B validation JSON + input existence.
        phase2b, pf_errors, pf_warnings = preflight(args)
        hard_errors.extend(pf_errors)
        warnings.extend(pf_warnings)
        if hard_errors:
            for e in hard_errors:
                print(f"ERROR: {e}", file=sys.stderr)
            print(
                "\nPhase 2D aborted at preflight. No COCO file was written or replaced.",
                file=sys.stderr,
            )
            return 2

        # 1c. Read canonical tables (read-only) and resolve columns by NAME.
        image_df = pd.read_csv(args.canonical_image_table)
        bbox_df = pd.read_csv(args.canonical_bbox_table)
        class_df = pd.read_csv(args.canonical_class_mapping)

        try:
            imap = map_image_columns(image_df)
            bmap = map_bbox_columns(bbox_df)
            cmap = map_class_columns(class_df)
        except ValueError as exc:
            print(f"ERROR: column mapping failed.\n       {exc}", file=sys.stderr)
            return 2

        # Actual negativity counts straight from canonical METADATA.
        if imap["is_negative"]:
            actual_neg = int(image_df[imap["is_negative"]].fillna(False).astype(bool).sum())
        elif imap["is_abnormal"]:
            actual_neg = int(
                (~image_df[imap["is_abnormal"]].fillna(False).astype(bool)).sum()
            )
        else:
            lab = image_df[imap["scope_label"]].astype(str).str.strip().str.lower()
            actual_neg = int(lab.isin(["no_finding", "no finding", "negative"]).sum())
        actual_abn = int(len(image_df) - actual_neg)

        # 1d. THREE-WAY protocol reconciliation: YAML <-> Phase 2B <-> actual tables.
        drift = crosscheck_protocol(
            cfg,
            phase2b,
            actual_image_rows=len(image_df),
            actual_bbox_rows=len(bbox_df),
            actual_class_rows=len(class_df),
            actual_abnormal_images=actual_abn,
            actual_no_finding_images=actual_neg,
        )
        protocol_validation["protocol_drift"] = drift
        protocol_validation["protocol_drift_count"] = len(drift)
        protocol_validation["phase2b_crosscheck_pass"] = len(drift) == 0
        if drift:
            for d in drift:
                print(f"ERROR: {d}", file=sys.stderr)
            print(
                "\nPhase 2D aborted: the protocol YAML disagrees with Phase 2B and/or "
                "the canonical tables. No COCO file was written or replaced.",
                file=sys.stderr,
            )
            return 2

        # 1e. Duplicate identifier checks.
        if image_df[imap["image_id"]].duplicated().any():
            hard_errors.append("PREFLIGHT: duplicate image_id in canonical image table.")
        if imap["canonical_image_id"] and image_df[imap["canonical_image_id"]].duplicated().any():
            hard_errors.append("PREFLIGHT: duplicate canonical_image_id.")
        if bbox_df[bmap["canonical_ann_id"]].duplicated().any():
            hard_errors.append("PREFLIGHT: duplicate canonical_ann_id in canonical bbox table.")
        if class_df[cmap["canonical_class_id"]].duplicated().any():
            hard_errors.append("PREFLIGHT: duplicate canonical_class_id in class mapping.")
        if class_df[cmap["class_name"]].duplicated().any():
            hard_errors.append("PREFLIGHT: duplicate class_name in class mapping.")

        if bmap["bbox_format"]:
            fmts = sorted(set(bbox_df[bmap["bbox_format"]].astype(str)))
            if fmts != [cfg.bbox_source_format]:
                warnings.append(
                    f"bbox_format in source is {fmts}, protocol expects "
                    f"['{cfg.bbox_source_format}']."
                )

        if hard_errors:
            for e in hard_errors:
                print(f"ERROR: {e}", file=sys.stderr)
            print(
                "\nPhase 2D aborted. No COCO file was written or replaced.",
                file=sys.stderr,
            )
            return 2

        # ================= STEP 2 — BUILD IN MEMORY =================
        # Nothing is written to disk in this step.
        categories, canon_to_cat = build_categories(class_df, cmap, cfg, hard_errors)
        name_to_cat = {c["name"]: c["id"] for c in categories}
        images, imgid_to_cocoid, meta_by_imgid = build_images(
            image_df, imap, cfg, hard_errors
        )
        annotations = build_annotations(
            bbox_df, bmap, imgid_to_cocoid, canon_to_cat, name_to_cat, hard_errors
        )

        coco: Dict[str, Any] = {
            "info": {
                "description": (
                    "VinBigData Chest X-ray — controlled scope COCO master (Phase 2D)."
                ),
                "source_of_truth": "phase2B_canonical_detection_schema",
                "bbox_source_format": cfg.bbox_source_format,
                "bbox_target_format": cfg.bbox_target_format,
                "image_root_env_var": cfg.path_root_variable,
                "dataset_training_ready": False,
                "created_utc": datetime.now(timezone.utc).isoformat(),
            },
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

        # ================= STEP 3 — FULL INTERNAL VALIDATION =================
        # EVERY hard check runs here, BEFORE any file is written or promoted.
        blocks, invalid_rows = validate_coco(
            coco, bbox_df, bmap, meta_by_imgid, canon_to_cat, cfg, hard_errors
        )
        nf_rows = blocks.pop("_no_finding_rows")
        ann_count_by_img = blocks.pop("_ann_count_by_img")

        # 3b. PER-IMAGE bbox-count validation — mandated BEFORE promotion.
        img_rows: List[Dict[str, Any]] = []
        for im in images:
            oid = im.get("original_image_id")
            canon_bbox = meta_by_imgid.get(oid, {}).get("canonical_bbox_count")
            coco_n = ann_count_by_img.get(im["id"], 0)
            img_rows.append(
                {
                    "coco_image_id": im["id"],
                    "canonical_image_id": im.get("canonical_image_id"),
                    "original_image_id": oid,
                    "file_name": im["file_name"],
                    "is_negative": im.get("is_negative"),
                    "canonical_bbox_count": canon_bbox,
                    "coco_annotation_count": coco_n,
                    "count_match": (canon_bbox is None) or (int(canon_bbox) == coco_n),
                }
            )
        count_mismatches = sum(1 for r in img_rows if not r["count_match"])
        if count_mismatches:
            hard_errors.append(
                f"COUNTS: {count_mismatches} image(s) where canonical bbox_count != "
                "COCO annotation count."
            )

        # 3c. Category annotation counts (also pre-promotion).
        cat_rows: List[Dict[str, Any]] = []
        ann_by_cat: Dict[int, int] = {}
        imgs_by_cat: Dict[int, set] = {}
        for a in annotations:
            ann_by_cat[a["category_id"]] = ann_by_cat.get(a["category_id"], 0) + 1
            imgs_by_cat.setdefault(a["category_id"], set()).add(a["image_id"])
        for c in categories:
            cat_rows.append(
                {
                    "category_id": c["id"],
                    "name": c["name"],
                    "canonical_class_id": c.get("canonical_class_id"),
                    "class_id_original": c.get("class_id_original"),
                    "annotation_count": ann_by_cat.get(c["id"], 0),
                    "unique_image_count": len(imgs_by_cat.get(c["id"], set())),
                }
            )
        cat_total = sum(r["annotation_count"] for r in cat_rows)
        if cat_total != len(annotations):
            hard_errors.append(
                f"COUNTS: category annotation total {cat_total} != annotation count "
                f"{len(annotations)}."
            )

        internal_validation_pass = len(hard_errors) == 0

        # ================= STEP 4 — WRITE TEMPORARY JSON =================
        json_validation = {
            "strict_allow_nan": False,
            "temporary_json_parse_pass": False,
            "top_level_keys_present": False,
            "temporary_counts_match": False,
            "atomic_write": True,
        }
        pycocotools_validation: Dict[str, Any] = {
            "used_for": "annotation_json_parsing_only",
            "pycocotools_available": False,
            "pycocotools_load_pass": False,
            "exception": None,
            "images_loaded": None,
            "annotations_loaded": None,
            "categories_loaded": None,
        }

        out_path = ensure_parent(args.coco_master)
        if internal_validation_pass:
            tmp_fd, tmp_name = tempfile.mkstemp(
                dir=str(out_path.parent), prefix=".coco_master_", suffix=".tmp.json"
            )
            try:
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                    json.dump(coco, fh, allow_nan=False, ensure_ascii=False)
                    fh.flush()
                    os.fsync(fh.fileno())
            except (ValueError, TypeError) as exc:
                hard_errors.append(
                    f"JSON: serialization failed (NaN/Infinity or bad type): {exc!r}"
                )
                Path(tmp_name).unlink(missing_ok=True)
                tmp_name = None

            # ============ STEP 5 — VALIDATE THE TEMPORARY JSON ============
            if tmp_name and Path(tmp_name).exists():
                try:
                    reparsed = json.loads(Path(tmp_name).read_text(encoding="utf-8"))
                    json_validation["temporary_json_parse_pass"] = True
                    json_validation["top_level_keys_present"] = all(
                        k in reparsed for k in ("images", "annotations", "categories")
                    )
                    if not json_validation["top_level_keys_present"]:
                        hard_errors.append("JSON: missing a required top-level key.")
                    json_validation["temporary_counts_match"] = (
                        len(reparsed.get("images", [])) == cfg.expect_images
                        and len(reparsed.get("annotations", [])) == cfg.expect_annotations
                        and len(reparsed.get("categories", [])) == cfg.expect_categories
                    )
                    if not json_validation["temporary_counts_match"]:
                        hard_errors.append(
                            "JSON: temporary file counts do not match the protocol."
                        )
                except Exception as exc:
                    hard_errors.append(f"JSON: temporary file did not parse: {exc!r}")

                # pycocotools on the TEMP file (annotation JSON only, never images).
                pycocotools_validation = run_pycocotools(tmp_name, hard_errors, warnings)
        else:
            warnings.append(
                "Internal validation failed; no temporary COCO file was written and the "
                "existing final output (if any) was left untouched."
            )

        # ================= DoD CHECKLIST (pre-promotion) =================
        pycoco_ok = (not pycocotools_validation["pycocotools_available"]) or (
            pycocotools_validation["pycocotools_load_pass"]
        )
        dod_checklist = {
            "protocol_yaml_strict_load_pass": protocol_validation["strict_schema_pass"],
            "protocol_phase2b_crosscheck_pass": protocol_validation["phase2b_crosscheck_pass"],
            "images_expected": blocks["image_validation"]["image_count_pass"],
            "annotations_expected": blocks["annotation_validation"]["annotation_count_pass"],
            "categories_expected": blocks["category_validation"]["category_count_pass"],
            "abnormal_images_expected": blocks["image_validation"]["abnormal_images_pass"],
            "no_finding_images_expected": blocks["no_finding_validation"][
                "no_finding_images_pass"
            ],
            "no_finding_zero_annotations": blocks["no_finding_validation"][
                "all_zero_annotation"
            ],
            "no_finding_not_a_category": not blocks["no_finding_validation"][
                "no_finding_in_categories"
            ],
            "category_ids_contiguous_1_to_n": blocks["category_validation"][
                "ids_contiguous_1_to_n"
            ],
            "category_id_zero_absent": not blocks["category_validation"][
                "category_id_zero_present"
            ],
            "image_ids_unique": blocks["image_validation"]["image_ids_unique"],
            "annotation_ids_unique": blocks["annotation_validation"]["annotation_ids_unique"],
            "file_names_unique": blocks["image_validation"]["file_names_unique"],
            "relative_paths_only": blocks["image_validation"]["relative_path_only"],
            "all_references_valid": blocks["relationship_validation"]["all_image_ids_exist"]
            and blocks["relationship_validation"]["all_category_ids_exist"],
            "zero_invalid_annotations": len(invalid_rows) == 0,
            "boundary_pass": blocks["boundary_validation"]["boundary_pass"],
            "area_pass": blocks["area_validation"]["area_pass"],
            "traceability_pass": blocks["traceability_validation"]["traceability_pass"],
            "one_to_one_preservation_pass": blocks["one_to_one_preservation"][
                "one_to_one_pass"
            ],
            "per_image_bbox_count_pass": count_mismatches == 0,
            "category_annotation_total_pass": cat_total == len(annotations),
            "json_parse_pass": json_validation["temporary_json_parse_pass"],
            "json_top_level_keys_pass": json_validation["top_level_keys_present"],
            "pycocotools_pass_or_unavailable": pycoco_ok,
            "all_hard_checks_completed_before_output_replace": True,
            "per_image_bbox_count_checked_before_output_replace": True,
        }

        all_pre_promotion_checks_pass = bool(
            len(hard_errors) == 0 and all(dod_checklist.values())
        )

        # ================= STEP 6/7 — ATOMIC PROMOTE / PRESERVE =================
        atomic = promote_atomic(
            tmp_name or "", args.coco_master, all_pre_promotion_checks_pass
        )
        if tmp_name and not Path(tmp_name).is_file():
            tmp_name = None  # consumed by os.replace or already cleaned
        atomic["validation_completed_before_promotion"] = True
        atomic["per_image_count_checked_before_promotion"] = True
        atomic["temporary_json_parse_pass"] = json_validation["temporary_json_parse_pass"]
        atomic["pycocotools_checked_before_promotion"] = bool(
            pycocotools_validation["pycocotools_available"]
        )

        # Promotion counts as a pass only when the final file was actually replaced.
        dod_checklist["atomic_output_promotion_pass"] = bool(atomic["final_output_replaced"])

        output_written = bool(atomic["final_output_replaced"])
        dod_pass_candidate = bool(
            len(hard_errors) == 0
            and all(dod_checklist.values())
            and output_written
        )

        # ================= EVIDENCE OUTPUT =================
        pd.DataFrame(cat_rows).to_csv(ensure_parent(args.category_summary_csv), index=False)
        pd.DataFrame(img_rows).to_csv(ensure_parent(args.image_counts_csv), index=False)
        invalid_cols = [
            "annotation_id", "canonical_ann_id", "image_id", "category_id",
            "reason", "bbox", "area", "image_width", "image_height",
        ]
        pd.DataFrame(invalid_rows, columns=invalid_cols).to_csv(
            ensure_parent(args.invalid_annotations_csv), index=False
        )
        nf_cols = [
            "coco_image_id", "canonical_image_id", "original_image_id", "file_name",
            "source_is_negative", "coco_annotation_count", "zero_annotation_pass",
        ]
        pd.DataFrame(nf_rows, columns=nf_cols).to_csv(
            ensure_parent(args.no_finding_audit_csv), index=False
        )

        counts = {
            "images": len(images),
            "annotations": len(annotations),
            "categories": len(categories),
            "abnormal_images": blocks["image_validation"]["abnormal_images"],
            "no_finding_images": blocks["no_finding_validation"]["no_finding_images"],
            "no_finding_annotations": blocks["no_finding_validation"][
                "no_finding_with_annotation_count"
            ],
            "canonical_bbox_rows": len(bbox_df),
        }
        input_paths = {
            "canonical_image_table": args.canonical_image_table,
            "canonical_bbox_table": args.canonical_bbox_table,
            "canonical_class_mapping": args.canonical_class_mapping,
            "phase2b_validation_json": args.phase2b_validation_json,
            "protocol_yaml": args.protocol_yaml,
        }
        forbidden_actions = {
            "dicom_file_read": False,
            "dicom_file_existence_checked": False,
            "pydicom_used": False,
            "dicom_header_read": False,
            "pixel_array_read": False,
            "cv2_imread_used": False,
            "pil_image_open_used": False,
            "any_image_loader_used": False,
            "image_copied_or_converted": False,
            "train_val_test_split_created": False,
            "labeled_unlabeled_split_created": False,
            "mmdet_or_detectron2_dataset_loaded": False,
            "filter_empty_gt_checked": False,
            "training_started": False,
            "inference_run": False,
            "pseudo_label_generated": False,
            "threshold_tuned": False,
            "test_set_used": False,
            "ap_map_computed": False,
            "canonical_schema_modified": False,
            "source_annotation_modified": False,
            "bbox_deleted": False,
            "bbox_clamped": False,
            "bbox_fused": False,
            "nms_applied": False,
            "dataset_training_ready_claimed": False,
        }

        payload: Dict[str, Any] = {
            "phase": "phase2D_coco_master_conversion",
            "created_utc": coco["info"]["created_utc"],
            "input_paths": input_paths,
            "input_sha256": {k: sha256_file(v) for k, v in input_paths.items()},
            "output_path": str(args.coco_master),
            "output_written": output_written,
            "output_sha256": atomic.get("final_output_sha256"),
            "counts": counts,
            "protocol_validation": protocol_validation,
            "id_validation": blocks["id_validation"],
            "image_validation": blocks["image_validation"],
            "annotation_validation": blocks["annotation_validation"],
            "relationship_validation": blocks["relationship_validation"],
            "bbox_validation": blocks["bbox_validation"],
            "area_validation": blocks["area_validation"],
            "boundary_validation": blocks["boundary_validation"],
            "no_finding_validation": blocks["no_finding_validation"],
            "category_validation": blocks["category_validation"],
            "path_validation": blocks["path_validation"],
            "traceability_validation": blocks["traceability_validation"],
            "one_to_one_preservation": blocks["one_to_one_preservation"],
            "json_validation": json_validation,
            "pycocotools_validation": pycocotools_validation,
            "atomic_output_validation": atomic,
            "column_mapping": {
                "image_table": dict(imap),
                "bbox_table": dict(bmap),
                "class_mapping": dict(cmap),
            },
            "image_annotation_count_mismatches": count_mismatches,
            "forbidden_actions": forbidden_actions,
            "dataset_training_ready": False,
            "next_phase_locked_until_gpt_review": True,
            "hard_errors": hard_errors,
            "warnings": warnings,
            "dod_checklist": dod_checklist,
            "dod_pass_candidate": dod_pass_candidate,
        }
        ensure_parent(args.validation_json).write_text(
            json.dumps(payload, indent=2, sort_keys=False, allow_nan=False),
            encoding="utf-8",
        )

        md_payload = dict(payload)
        md_payload["_category_rows"] = cat_rows
        write_report_md(args.report_md, md_payload, cfg)

        # ================= CONSOLE SUMMARY =================
        pc = pycocotools_validation
        pc_status = (
            "unavailable (warning)"
            if not pc["pycocotools_available"]
            else ("load PASS" if pc["pycocotools_load_pass"] else "load FAIL (hard error)")
        )
        print("=" * 70)
        print("Phase 2D — COCO Master Conversion & Validation")
        print("=" * 70)
        print(f"images                     : {counts['images']}")
        print(f"annotations                : {counts['annotations']}")
        print(f"categories                 : {counts['categories']}")
        print(f"abnormal images            : {counts['abnormal_images']}")
        print(f"No Finding images          : {counts['no_finding_images']}")
        print(f"invalid annotations        : {len(invalid_rows)}")
        print(f"No Finding annotations     : {counts['no_finding_annotations']}")
        print(f"absolute paths             : {blocks['path_validation']['absolute_path_count']}")
        print(f"pycocotools                : {pc_status}")
        print(
            "protocol strict load       : "
            f"{'PASS' if protocol_validation['strict_schema_pass'] else 'FAIL'}"
        )
        print(
            "protocol / Phase 2B drift  : "
            f"{protocol_validation['protocol_drift_count']}"
        )
        print(
            "pre-promotion checks       : "
            f"{'PASS' if all_pre_promotion_checks_pass else 'FAIL'}"
        )
        print(
            "atomic promotion           : "
            f"{'PASS' if atomic['final_output_replaced'] else 'NOT PERFORMED'}"
        )
        print(f"hard errors                : {len(hard_errors)}")
        print(f"warnings                   : {len(warnings)}")
        print("-" * 70)
        for e in hard_errors:
            print(f"  ERROR: {e}")
        for w in warnings:
            print(f"  WARN : {w}")
        if hard_errors or warnings:
            print("-" * 70)
        if not atomic["final_output_replaced"] and atomic["previous_output_existed"]:
            print(
                "  NOTE : the previous coco_master.json was PRESERVED unchanged "
                f"(preserved={atomic['previous_valid_output_preserved_on_failure']})."
            )
            print("-" * 70)
        print("dataset_training_ready     : False  (DICOM loader = Phase 2D.1)")
        print(f"dod_pass_candidate         : {dod_pass_candidate}")
        print("-" * 70)
        print("Outputs:")
        print(
            f"  {args.coco_master}"
            + ("" if output_written else "   [NOT REPLACED — previous file preserved]")
        )
        print(f"  {args.validation_json}")
        print(f"  {args.report_md}")
        print(f"  {args.image_counts_csv}")
        print(f"  {args.category_summary_csv}")
        print(f"  {args.invalid_annotations_csv}")
        print(f"  {args.no_finding_audit_csv}")
        print(f"  {args.protocol_yaml}")
        print("=" * 70)
        print("No DICOM/image access occurred. Next phase LOCKED until GPT review PASS.")

        return 0 if dod_pass_candidate else 1

    finally:
        # STEP 7 — never leave a temp file behind, even on an exception.
        if tmp_name and str(tmp_name).strip():
            leftover = Path(tmp_name)
            if leftover.is_file():
                leftover.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
