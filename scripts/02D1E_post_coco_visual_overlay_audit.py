#!/usr/bin/env python3
r"""Post-COCO Visual Overlay Audit (technical label: 2D1E).

Supporting QA / evidence step with two clearly separated scopes:

  A. EXHAUSTIVE AUTOMATED VALIDATION — runs over ALL COCO images and ALL COCO
     annotations: JPEG resolution/availability, unreadable-JPEG detection,
     actual-JPEG-vs-COCO width/height consistency, bbox geometry validity,
     reference integrity and category mapping. This is the authoritative source
     of ``automated_checks_status``.
  B. REPRESENTATIVE VISUAL QA — a small deterministic stress/representative
     selection (e.g. ~16 images) that is rendered into individual overlays and a
     contact sheet for INDEPENDENT manual visual review. It NEVER covers the
     whole dataset and NEVER gates the automated verdict.

It reads image metadata, bounding boxes and category names DIRECTLY from the
COCO master derivative ``data/processed/coco/coco_master_jpg.json``, resolves the
corresponding JPEGs, and (for the representative selection only) overlays the
COCO boxes onto the JPEGs, producing a manifest CSV and a report JSON so that
GPT / the researcher can perform the manual visual review.

This is NOT a new preprocessing step, NOT an annotation-editing step, NOT a
training step, NOT a new research phase, and NOT a new method or algorithm. The
technical label "2D1E" is only a filename tag; it does not declare an official
research phase.

READ-ONLY guarantees (enforced by construction):
    * Never opens the COCO master, canonical tables, or any source JPEG for
      writing. Source JPEGs are opened read-only for size + rendering.
    * Never edits / deletes / fuses / dedups / clips / scales bounding boxes.
    * Never crops / resizes / rotates / flips / transposes / re-encodes the
      source images. Any thumbnail resize is a DISPLAY-only artifact for the
      contact sheet and never overwrites the dataset.
    * Overlapping / near-duplicate multi-radiologist boxes are NOT treated as
      errors or duplicates.

Fail-safe: on missing COCO, malformed JSON, ambiguous JPEG resolution, missing
selected JPEG, dimension mismatch, invalid reference, invalid bbox, or category
mapping error, the script fails loudly (records evidence, sets automated status
to FAIL, returns a non-zero exit code) and NEVER edits the input to "make it
pass".

The bounding boxes drawn are read directly from COCO ``bbox = [x, y, w, h]`` and
converted for display only as ``x1=x, y1=y, x2=x+w, y2=y+h`` without rounding
that would alter the stored annotation.

manual_visual_review_status is ALWAYS ``PENDING_GPT_REVIEW`` on a run; producing
overlays/contact sheets successfully does NOT constitute a manual-review PASS.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Sequence, Tuple

AUDIT_NAME = "Post-COCO Visual Overlay Audit"
TECHNICAL_LABEL = "2D1E"

# --- Defaults (project convention; overridable via CLI) --------------------- #
DEFAULT_REPO_ROOT = "."
DEFAULT_COCO_JSON = "data/processed/coco/coco_master_jpg.json"
DEFAULT_IMAGE_ROOT = "data/processed/images_jpg"
DEFAULT_REPORT_JSON = "reports/phase2D1E_post_coco_visual_overlay_audit.json"
DEFAULT_MANIFEST_CSV = "reports/phase2D1E_post_coco_visual_overlay_manifest.csv"
DEFAULT_PLOTS_DIR = "plots/phase2D1E_post_coco_visual_overlay"

# Expected locked counts (verified at runtime, never hard-coded to force PASS).
DEFAULT_EXPECTED_IMAGES = 4894
DEFAULT_EXPECTED_ANNOTATIONS = 36096
DEFAULT_EXPECTED_CATEGORIES = 14
DEFAULT_EXPECTED_EMPTY_IMAGES = 500

# Recorded COCO SHA-256 for the locked scope (compared, never enforced-by-edit).
RECORDED_COCO_SHA256 = (
    "f587152278f713460ff1e727a2912248a47052f6abc48de8f7bad6e8a63b94c0"
)

DEFAULT_SEED = 2026  # project-locked seed; selection is deterministic anyway
DEFAULT_NUM_ABNORMAL = 20
DEFAULT_NUM_ZERO_GT = 4
BBOX_BOUND_TOLERANCE = 1.0  # px tolerance for boundary/dimension checks
NEAR_BOUNDARY_PX = 3.0      # a bbox edge within this many px of an image edge


class AuditError(Exception):
    """Any hard failure specific to this visual overlay audit."""


# --------------------------------------------------------------------------- #
# Small utilities.                                                             #
# --------------------------------------------------------------------------- #
def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 ``Z`` string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute the SHA-256 hex digest of a file, streaming in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(repo_root: Path, maybe_relative: str) -> Path:
    """Resolve ``maybe_relative`` against ``repo_root`` unless already absolute."""
    candidate = Path(maybe_relative)
    if candidate.is_absolute():
        return candidate
    return (repo_root / candidate).resolve()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via a temp file + atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically write UTF-8 text."""
    atomic_write_bytes(path, text.encode("utf-8"))


def write_json_atomic(path: Path, obj: Any) -> None:
    """Atomically write ``obj`` as indented, sorted-key JSON."""
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str))


def write_csv_atomic(path: Path, header: Sequence[str],
                     rows: Sequence[Sequence[Any]]) -> None:
    """Atomically write a CSV; ``rows`` may be empty (header-only file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(list(header))
            for row in rows:
                writer.writerow(list(row))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


# --------------------------------------------------------------------------- #
# COCO structure (read directly from coco_master_jpg.json).                    #
# --------------------------------------------------------------------------- #
@dataclass
class CocoStructure:
    """Structural facts derived purely from the COCO JSON (no image decode)."""

    images_by_id: Dict[int, Dict[str, Any]]
    anns_by_image: Dict[int, List[Dict[str, Any]]]
    annotations: List[Dict[str, Any]]
    category_id_to_name: Dict[int, str]
    image_ids: List[int]
    annotation_ids: List[int]
    category_ids: List[int]
    duplicate_image_ids: List[int]
    duplicate_annotation_ids: List[int]
    duplicate_category_ids: List[int]
    invalid_image_refs: int
    invalid_category_refs: int
    zero_gt_image_ids: List[int]
    nonempty_image_ids: List[int]

    @property
    def num_images(self) -> int:
        return len(self.image_ids)

    @property
    def num_annotations(self) -> int:
        return len(self.annotation_ids)

    @property
    def num_categories(self) -> int:
        return len(self.category_ids)


def _duplicates(values: Sequence[int]) -> List[int]:
    """Return the sorted list of values that appear more than once."""
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def analyze_coco_structure(coco: Dict[str, Any]) -> CocoStructure:
    """Analyze the COCO dict structure without decoding any image.

    Raises ``AuditError`` if a required top-level list is missing/malformed.
    """
    for key in ("images", "annotations", "categories"):
        if key not in coco or not isinstance(coco[key], list):
            raise AuditError(f"COCO JSON missing/invalid list key: {key!r}")

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    images_by_id: Dict[int, Dict[str, Any]] = {}
    for img in images:
        images_by_id[int(img["id"])] = img

    category_id_to_name: Dict[int, str] = {
        int(cat["id"]): str(cat["name"]) for cat in categories
    }

    image_ids = [int(img["id"]) for img in images]
    annotation_ids = [int(ann["id"]) for ann in annotations]
    category_ids = [int(cat["id"]) for cat in categories]

    valid_image_ids = set(image_ids)
    valid_category_ids = set(category_ids)

    anns_by_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    invalid_image_refs = 0
    invalid_category_refs = 0
    for ann in annotations:
        img_id = int(ann["image_id"])
        cat_id = int(ann["category_id"])
        if img_id in valid_image_ids:
            anns_by_image[img_id].append(ann)
        else:
            invalid_image_refs += 1
        if cat_id not in valid_category_ids:
            invalid_category_refs += 1

    zero_gt_image_ids = sorted(
        iid for iid in image_ids if len(anns_by_image.get(iid, [])) == 0)
    nonempty_image_ids = sorted(
        iid for iid in image_ids if len(anns_by_image.get(iid, [])) > 0)

    return CocoStructure(
        images_by_id=images_by_id,
        anns_by_image=dict(anns_by_image),
        annotations=annotations,
        category_id_to_name=category_id_to_name,
        image_ids=image_ids,
        annotation_ids=annotation_ids,
        category_ids=category_ids,
        duplicate_image_ids=_duplicates(image_ids),
        duplicate_annotation_ids=_duplicates(annotation_ids),
        duplicate_category_ids=_duplicates(category_ids),
        invalid_image_refs=invalid_image_refs,
        invalid_category_refs=invalid_category_refs,
        zero_gt_image_ids=zero_gt_image_ids,
        nonempty_image_ids=nonempty_image_ids,
    )


# --------------------------------------------------------------------------- #
# JPEG resolution (safe; report ambiguity, never silently pick).               #
# --------------------------------------------------------------------------- #
def resolve_jpeg(image_root: Path, file_name: str) -> Dict[str, Any]:
    """Resolve one COCO ``file_name`` to a unique JPEG under ``image_root``.

    Handles both ``train/<id>.jpg`` and ``<id>.jpg`` conventions. Returns a dict
    with ``status`` in {"ok", "missing", "ambiguous"} and the resolved path(s).
    Ambiguity (basename matching >1 file) is a hard error, never auto-picked.
    """
    rel = PurePosixPath(str(file_name).replace("\\", "/"))
    direct = image_root / Path(*rel.parts)
    if direct.is_file():
        return {"status": "ok", "path": str(direct), "candidates": [str(direct)]}

    # Fallback: search by basename under image_root (report ambiguity).
    basename = rel.name
    matches = [p for p in image_root.rglob(basename) if p.is_file()]
    if len(matches) == 1:
        return {"status": "ok", "path": str(matches[0]),
                "candidates": [str(matches[0])], "note": "resolved_by_basename"}
    if len(matches) == 0:
        return {"status": "missing", "path": None,
                "candidates": [], "attempted": str(direct)}
    return {"status": "ambiguous", "path": None,
            "candidates": [str(m) for m in matches]}


# --------------------------------------------------------------------------- #
# BBox validation (xywh; never clip / never edit).                             #
# --------------------------------------------------------------------------- #
def validate_bbox_xywh(bbox: Sequence[float], img_w: float, img_h: float,
                       tolerance: float = BBOX_BOUND_TOLERANCE) -> Dict[str, Any]:
    """Validate a single COCO ``[x, y, w, h]`` box against image dimensions.

    Returns a dict of boolean checks + a scalar ``valid``. Never mutates input.
    """
    checks: Dict[str, Any] = {
        "four_values": False,
        "finite": False,
        "w_positive": False,
        "h_positive": False,
        "in_bounds": False,
    }
    if bbox is None or len(bbox) != 4:
        checks["valid"] = False
        return checks
    checks["four_values"] = True
    try:
        x, y, w, h = (float(bbox[0]), float(bbox[1]),
                      float(bbox[2]), float(bbox[3]))
    except (TypeError, ValueError):
        checks["valid"] = False
        return checks
    checks["finite"] = all(math.isfinite(v) for v in (x, y, w, h))
    checks["w_positive"] = w > 0.0
    checks["h_positive"] = h > 0.0
    if checks["finite"]:
        x2, y2 = x + w, y + h
        checks["in_bounds"] = (
            x >= -tolerance and y >= -tolerance
            and x2 <= img_w + tolerance and y2 <= img_h + tolerance)
    checks["valid"] = bool(
        checks["four_values"] and checks["finite"] and checks["w_positive"]
        and checks["h_positive"] and checks["in_bounds"])
    return checks


def bbox_is_near_boundary(bbox: Sequence[float], img_w: float, img_h: float,
                          margin: float = NEAR_BOUNDARY_PX) -> bool:
    """Return True if any edge of the box is within ``margin`` px of an edge."""
    x, y, w, h = (float(bbox[0]), float(bbox[1]),
                  float(bbox[2]), float(bbox[3]))
    x2, y2 = x + w, y + h
    return (x <= margin or y <= margin
            or x2 >= img_w - margin or y2 >= img_h - margin)


# --------------------------------------------------------------------------- #
# EXHAUSTIVE full-dataset automated validation (ALL images + ALL annotations). #
# --------------------------------------------------------------------------- #
# This scope is intentionally separate from the representative visual QA: it
# never renders overlays and never resizes/mutates any JPEG. It reads each JPEG
# header (PIL lazily reads size without decoding the full raster) to obtain the
# actual width/height and compares against the COCO metadata for EVERY image, and
# it runs the SAME ``validate_bbox_xywh`` validator on EVERY annotation. There is
# exactly one bbox-validity definition, reused here and in the visual scope.
_FAILURE_SAMPLE_CAP = 500  # counts are exact; recorded example rows are capped


def validate_full_dataset(structure: "CocoStructure",
                          image_root: Path) -> Dict[str, Any]:
    """Run exhaustive automated validation over ALL images and annotations.

    Returns exact counts plus per-invariant PASS/FAIL statuses and a capped list
    of example failures. Never mutates any input; JPEGs are opened read-only for
    their header size only. ``validate_bbox_xywh`` is reused (single definition,
    preserving its intended in-bounds tolerance semantics).
    """
    from PIL import Image  # lazy: only needed at runtime, not for import/compile

    images_checked = 0
    jpegs_resolved = 0
    missing_jpeg = 0
    ambiguous_jpeg = 0
    unreadable_jpeg = 0
    dimension_mismatch = 0

    annotations_checked = 0
    invalid_bbox = 0

    failures: List[Dict[str, Any]] = []

    def _record(kind: str, detail: Dict[str, Any]) -> None:
        if len(failures) < _FAILURE_SAMPLE_CAP:
            failures.append({"type": kind, **detail})

    # --- Every image: resolve JPEG, verify existence/ambiguity, read size --- #
    for iid in structure.image_ids:
        images_checked += 1
        img = structure.images_by_id[iid]
        file_name = str(img["file_name"])
        coco_w, coco_h = int(img["width"]), int(img["height"])

        res = resolve_jpeg(image_root, file_name)
        if res["status"] == "missing":
            missing_jpeg += 1
            _record("missing_jpeg", {"image_id": iid, "file_name": file_name})
            continue
        if res["status"] == "ambiguous":
            ambiguous_jpeg += 1
            _record("ambiguous_jpeg", {"image_id": iid, "file_name": file_name,
                                       "candidates": res.get("candidates")})
            continue

        jpegs_resolved += 1
        try:
            with Image.open(res["path"]) as im:  # read-only header
                actual_w, actual_h = im.size
        except Exception as exc:  # noqa: BLE001 - record, never hide/mutate
            unreadable_jpeg += 1
            _record("unreadable_jpeg", {"image_id": iid,
                                        "path": res["path"],
                                        "error": f"{type(exc).__name__}: {exc}"})
            continue

        if int(actual_w) != coco_w or int(actual_h) != coco_h:
            dimension_mismatch += 1
            _record("dimension_mismatch",
                    {"image_id": iid, "coco": [coco_w, coco_h],
                     "jpeg": [int(actual_w), int(actual_h)]})

    # --- EVERY annotation (ALL, including orphans): traverse exhaustively --- #
    # Iterate the raw annotation list so annotations whose image_id is invalid
    # (orphans) are STILL traversed and counted, not skipped. Orphans cannot be
    # geometry-checked (no image dimensions), so bbox geometry is evaluated ONLY
    # when the image reference resolves; orphans are counted separately and are
    # surfaced as failures by reference integrity. Geometry never crashes on an
    # orphan because the image lookup guards it.
    annotations_geometry_unchecked = 0
    for ann in structure.annotations:
        annotations_checked += 1
        img = structure.images_by_id.get(int(ann["image_id"]))
        if img is None:
            annotations_geometry_unchecked += 1  # orphan: invalid image_id
            _record("orphan_annotation_invalid_image_ref",
                    {"ann_id": int(ann.get("id", -1)),
                     "image_id": ann.get("image_id")})
            continue
        coco_w, coco_h = float(img["width"]), float(img["height"])
        vb = validate_bbox_xywh(ann.get("bbox"), coco_w, coco_h)
        if not vb["valid"]:
            invalid_bbox += 1
            _record("invalid_bbox",
                    {"image_id": int(ann["image_id"]),
                     "ann_id": int(ann.get("id", -1)),
                     "bbox": ann.get("bbox"), "checks": vb})

    invalid_image_refs = structure.invalid_image_refs
    invalid_category_refs = structure.invalid_category_refs

    coverage_ok = (images_checked == structure.num_images
                   and annotations_checked == structure.num_annotations)
    jpeg_availability_ok = (missing_jpeg == 0 and ambiguous_jpeg == 0
                            and unreadable_jpeg == 0)
    dimension_ok = dimension_mismatch == 0
    bbox_ok = invalid_bbox == 0
    reference_ok = (invalid_image_refs == 0
                    and not structure.duplicate_image_ids
                    and not structure.duplicate_annotation_ids
                    and not structure.duplicate_category_ids)
    category_mapping_ok = invalid_category_refs == 0

    return {
        "images_checked": images_checked,
        "total_images": structure.num_images,
        "annotations_checked": annotations_checked,
        "total_annotations": structure.num_annotations,
        "annotations_geometry_unchecked_count": annotations_geometry_unchecked,
        "jpegs_resolved": jpegs_resolved,
        "missing_jpeg_count": missing_jpeg,
        "ambiguous_jpeg_count": ambiguous_jpeg,
        "unreadable_jpeg_count": unreadable_jpeg,
        "dimension_mismatch_count": dimension_mismatch,
        "invalid_bbox_count": invalid_bbox,
        "invalid_image_reference_count": invalid_image_refs,
        "invalid_category_reference_count": invalid_category_refs,
        "duplicate_image_ids": structure.duplicate_image_ids,
        "duplicate_annotation_ids": structure.duplicate_annotation_ids,
        "duplicate_category_ids": structure.duplicate_category_ids,
        "coverage_status": "PASS" if coverage_ok else "FAIL",
        "jpeg_availability_status": "PASS" if jpeg_availability_ok else "FAIL",
        "dimension_consistency_status": "PASS" if dimension_ok else "FAIL",
        "bbox_validity_status": "PASS" if bbox_ok else "FAIL",
        "reference_integrity_status": "PASS" if reference_ok else "FAIL",
        "category_mapping_status": "PASS" if category_mapping_ok else "FAIL",
        "all_invariants_pass": bool(
            coverage_ok and jpeg_availability_ok and dimension_ok and bbox_ok
            and reference_ok and category_mapping_ok),
        "failure_samples": failures,
        "failure_samples_truncated": len(failures) >= _FAILURE_SAMPLE_CAP,
    }


# --------------------------------------------------------------------------- #
# Deterministic representative sample selection.                               #
# --------------------------------------------------------------------------- #
def build_selection(structure: CocoStructure, num_abnormal: int,
                    num_zero_gt: int) -> List[Dict[str, Any]]:
    """Deterministically select representative images with selection reasons.

    Strategy (all tie-breaks by ascending image_id, so it is reproducible with
    no randomness):
      * one image per category (category coverage across the 14 classes);
      * fewest / most bounding boxes;
      * smallest / largest single-bbox area;
      * most distinct categories in one image;
      * an image containing a near-boundary bbox;
      * the first ``num_zero_gt`` zero-GT images.
    A single image may carry multiple selection reasons. Abnormal selections are
    capped at ``num_abnormal`` while always preserving category coverage first.
    """
    reasons: Dict[int, List[str]] = defaultdict(list)

    # Per-image derived features (deterministic).
    feats: Dict[int, Dict[str, Any]] = {}
    for iid in structure.nonempty_image_ids:
        anns = structure.anns_by_image[iid]
        areas = [float(a["bbox"][2]) * float(a["bbox"][3]) for a in anns]
        cats = sorted({int(a["category_id"]) for a in anns})
        img = structure.images_by_id[iid]
        w, h = float(img["width"]), float(img["height"])
        near = any(bbox_is_near_boundary(a["bbox"], w, h) for a in anns)
        feats[iid] = {
            "num_bbox": len(anns),
            "num_cats": len(cats),
            "min_area": min(areas) if areas else None,
            "max_area": max(areas) if areas else None,
            "near_boundary": near,
        }

    # 1) Category coverage: smallest image_id containing each category.
    for cat_id in sorted(structure.category_id_to_name):
        name = structure.category_id_to_name[cat_id]
        for iid in structure.nonempty_image_ids:  # ascending order
            if any(int(a["category_id"]) == cat_id
                   for a in structure.anns_by_image[iid]):
                reasons[iid].append(f"category_coverage:{name}")
                break

    def _argmin(key, predicate=lambda v: True):
        best_iid, best_val = None, None
        for iid in structure.nonempty_image_ids:
            val = feats[iid][key]
            if val is None or not predicate(val):
                continue
            if best_val is None or val < best_val:
                best_val, best_iid = val, iid
        return best_iid

    def _argmax(key):
        best_iid, best_val = None, None
        for iid in structure.nonempty_image_ids:
            val = feats[iid][key]
            if val is None:
                continue
            if best_val is None or val > best_val:
                best_val, best_iid = val, iid
        return best_iid

    special = {
        "fewest_bboxes": _argmin("num_bbox"),
        "most_bboxes": _argmax("num_bbox"),
        "smallest_bbox_area": _argmin("min_area"),
        "largest_bbox_area": _argmax("max_area"),
        "most_categories": _argmax("num_cats"),
    }
    for iid in structure.nonempty_image_ids:
        if feats[iid]["near_boundary"]:
            special["near_boundary_bbox"] = iid
            break
    for reason, iid in special.items():
        if iid is not None:
            reasons[iid].append(reason)

    # Order abnormal selections: category-coverage images first (ascending id),
    # then special-property images, deduplicated, capped at num_abnormal.
    coverage_ids = [iid for iid in sorted(reasons)
                    if any(r.startswith("category_coverage")
                           for r in reasons[iid])]
    special_ids = [iid for iid in sorted(reasons) if iid not in coverage_ids]
    ordered_abnormal = coverage_ids + special_ids
    if len(ordered_abnormal) > num_abnormal:
        keep = set(coverage_ids)
        for iid in special_ids:
            if len(keep) >= num_abnormal:
                break
            keep.add(iid)
        ordered_abnormal = [iid for iid in ordered_abnormal if iid in keep]

    selection: List[Dict[str, Any]] = []
    for iid in ordered_abnormal:
        selection.append({
            "image_id": iid,
            "zero_gt": False,
            "selection_reasons": sorted(set(reasons[iid])),
        })

    # 2) Zero-GT images: first N by ascending image_id.
    for iid in structure.zero_gt_image_ids[:num_zero_gt]:
        selection.append({
            "image_id": iid,
            "zero_gt": True,
            "selection_reasons": ["zero_gt_no_finding"],
        })
    return selection


# --------------------------------------------------------------------------- #
# Rendering (matplotlib Agg; source images read-only).                         #
# --------------------------------------------------------------------------- #
# 14 visually distinct colors (one per category slot; cycles if more).
_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff",
    "#9a6324", "#800000",
]


def _color_for_category(cat_id: int, category_ids: Sequence[int]) -> str:
    """Return a stable color for a category id based on its sorted position."""
    ordered = sorted(category_ids)
    idx = ordered.index(cat_id) if cat_id in ordered else 0
    return _PALETTE[idx % len(_PALETTE)]


def render_overlay(image_path: Path, out_path: Path, image_id: int,
                   anns: List[Dict[str, Any]], category_id_to_name: Dict[int, str],
                   category_ids: Sequence[int], zero_gt: bool,
                   show_ann_id: bool, dpi: int) -> Dict[str, Any]:
    """Render one overlay PNG. Returns loaded (width, height) + draw summary.

    The source JPEG is opened READ-ONLY. Boxes are drawn as-is from COCO xywh;
    invalid boxes are drawn without clipping and flagged, never silently fixed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image

    with Image.open(image_path) as im:  # read-only
        loaded_w, loaded_h = im.size
        rgb = im.convert("RGB")

    fig_w = max(4.0, loaded_w / max(dpi, 1))
    fig_h = max(4.0, loaded_h / max(dpi, 1))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    try:
        ax.imshow(rgb)
        ax.set_axis_off()
        line_w = max(1.5, min(loaded_w, loaded_h) / 500.0)
        font_sz = max(8.0, min(loaded_w, loaded_h) / 90.0)

        drawn = 0
        for ann in anns:
            x, y, w, h = (float(ann["bbox"][0]), float(ann["bbox"][1]),
                          float(ann["bbox"][2]), float(ann["bbox"][3]))
            cat_id = int(ann["category_id"])
            color = _color_for_category(cat_id, category_ids)
            rect = patches.Rectangle(
                (x, y), w, h, linewidth=line_w, edgecolor=color,
                facecolor="none")
            ax.add_patch(rect)
            label = category_id_to_name.get(cat_id, f"UNKNOWN_CAT_{cat_id}")
            if show_ann_id:
                label = f"{label} #{int(ann['id'])}"
            ax.text(x, max(0.0, y - 2), label, color="white", fontsize=font_sz,
                    va="bottom", ha="left",
                    bbox=dict(facecolor=color, edgecolor="none",
                              alpha=0.7, pad=1.0))
            drawn += 1

        title = f"image_id={image_id}"
        if zero_gt:
            title = f"image_id={image_id}  |  ZERO-GT / NO COCO BBOX"
        ax.set_title(title, fontsize=max(10.0, font_sz), color="black")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    finally:
        plt.close(fig)

    return {"loaded_width": loaded_w, "loaded_height": loaded_h,
            "boxes_drawn": drawn}


def render_contact_sheet(overlay_paths: List[Tuple[int, bool, Path]],
                         out_path: Path, dpi: int,
                         cols: int = 4) -> Optional[str]:
    """Render a contact sheet from individual overlays (display-only thumbnails).

    Thumbnails are a DISPLAY artifact: the dataset JPEGs are not modified. Zero-GT
    tiles are visually marked. Returns the written path or None if nothing to do.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    tiles = [t for t in overlay_paths if t[2] is not None and t[2].is_file()]
    if not tiles:
        return None
    n = len(tiles)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), dpi=dpi)
    axes = axes.reshape(rows, cols) if hasattr(axes, "reshape") else \
        [[axes]]
    try:
        for idx in range(rows * cols):
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            ax.set_axis_off()
            if idx >= n:
                continue
            image_id, zero_gt, path = tiles[idx]
            with Image.open(path) as im:  # read overlay PNG (not dataset JPEG)
                ax.imshow(im.convert("RGB"))
            tag = f"id={image_id}" + ("  [ZERO-GT]" if zero_gt else "")
            ax.set_title(tag, fontsize=9,
                         color="crimson" if zero_gt else "black")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi)
    finally:
        plt.close(fig)
    return str(out_path)


# --------------------------------------------------------------------------- #
# CLI.                                                                         #
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser for the audit script."""
    parser = argparse.ArgumentParser(
        prog="02D1E_post_coco_visual_overlay_audit.py",
        description=(f"{AUDIT_NAME} (technical label {TECHNICAL_LABEL}) — "
                     "read-only post-COCO visual QA; no training, no edits."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-root", default=DEFAULT_REPO_ROOT)
    parser.add_argument("--coco-json", default=DEFAULT_COCO_JSON)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--report-json", default=DEFAULT_REPORT_JSON)
    parser.add_argument("--manifest-csv", default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--plots-dir", default=DEFAULT_PLOTS_DIR)
    parser.add_argument("--num-abnormal", type=int, default=DEFAULT_NUM_ABNORMAL)
    parser.add_argument("--num-zero-gt", type=int, default=DEFAULT_NUM_ZERO_GT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--contact-cols", type=int, default=4)
    parser.add_argument("--show-ann-id", action="store_true", default=True,
                        help="Draw annotation IDs on overlays (default on).")
    parser.add_argument("--no-show-ann-id", dest="show_ann_id",
                        action="store_false")
    parser.add_argument("--expected-images", type=int,
                        default=DEFAULT_EXPECTED_IMAGES)
    parser.add_argument("--expected-annotations", type=int,
                        default=DEFAULT_EXPECTED_ANNOTATIONS)
    parser.add_argument("--expected-categories", type=int,
                        default=DEFAULT_EXPECTED_CATEGORIES)
    parser.add_argument("--expected-empty-images", type=int,
                        default=DEFAULT_EXPECTED_EMPTY_IMAGES)
    parser.add_argument("--expected-coco-sha256", default=RECORDED_COCO_SHA256,
                        help="Recorded COCO SHA-256; discrepancy is reported.")
    parser.add_argument("--strict", action="store_true",
                        help="Treat expected-count/SHA discrepancies as FAIL.")
    return parser


# --------------------------------------------------------------------------- #
# Orchestration.                                                               #
# --------------------------------------------------------------------------- #
def run_audit(args: argparse.Namespace) -> Tuple[Dict[str, Any], bool]:
    """Execute the audit and return ``(report_dict, automated_pass)``.

    ``automated_pass`` reflects ONLY the automated invariants. Manual visual
    review remains PENDING regardless.
    """
    repo_root = Path(args.repo_root).resolve()
    coco_json = resolve_path(repo_root, args.coco_json)
    image_root = resolve_path(repo_root, args.image_root)
    report_json = resolve_path(repo_root, args.report_json)
    manifest_csv = resolve_path(repo_root, args.manifest_csv)
    plots_dir = resolve_path(repo_root, args.plots_dir)

    checks: Dict[str, Any] = {}
    errors: List[Dict[str, Any]] = []
    warnings: List[str] = []

    def record_error(name: str, detail: Any) -> None:
        errors.append({"check": name, "detail": detail})

    # --- A. COCO presence + parse ---------------------------------------- #
    checks["coco_exists"] = coco_json.is_file()
    if not checks["coco_exists"]:
        record_error("coco_exists", f"missing COCO JSON: {coco_json}")
        return _finalize_failure(
            args, repo_root, coco_json, image_root, report_json, manifest_csv,
            plots_dir, checks, errors, warnings, coco_sha256=None,
            structure=None, selection=[], manifest_rows=[], overlay_count=0,
            contact_sheet=None)

    coco_sha256 = sha256_of_file(coco_json)
    sha_match = coco_sha256 == args.expected_coco_sha256
    checks["coco_sha256_matches_recorded"] = sha_match
    if not sha_match:
        msg = (f"COCO SHA-256 {coco_sha256} != recorded "
               f"{args.expected_coco_sha256}")
        if args.strict:
            record_error("coco_sha256_matches_recorded", msg)
        else:
            warnings.append(msg)

    try:
        with open(coco_json, "r", encoding="utf-8") as handle:
            coco = json.load(handle)
        checks["coco_json_parse"] = True
    except (OSError, ValueError) as exc:
        checks["coco_json_parse"] = False
        record_error("coco_json_parse", str(exc))
        return _finalize_failure(
            args, repo_root, coco_json, image_root, report_json, manifest_csv,
            plots_dir, checks, errors, warnings, coco_sha256=coco_sha256,
            structure=None, selection=[], manifest_rows=[], overlay_count=0,
            contact_sheet=None)

    try:
        structure = analyze_coco_structure(coco)
        checks["coco_structure_analyzed"] = True
    except AuditError as exc:
        checks["coco_structure_analyzed"] = False
        record_error("coco_structure_analyzed", str(exc))
        return _finalize_failure(
            args, repo_root, coco_json, image_root, report_json, manifest_csv,
            plots_dir, checks, errors, warnings, coco_sha256=coco_sha256,
            structure=None, selection=[], manifest_rows=[], overlay_count=0,
            contact_sheet=None)

    # --- B. Structural integrity ----------------------------------------- #
    checks["images_present"] = structure.num_images > 0
    checks["annotations_present"] = structure.num_annotations > 0
    checks["categories_present"] = structure.num_categories > 0
    checks["no_duplicate_image_ids"] = not structure.duplicate_image_ids
    checks["no_duplicate_annotation_ids"] = not structure.duplicate_annotation_ids
    checks["no_duplicate_category_ids"] = not structure.duplicate_category_ids
    checks["no_invalid_image_refs"] = structure.invalid_image_refs == 0
    checks["no_invalid_category_refs"] = structure.invalid_category_refs == 0
    for name in ("no_duplicate_image_ids", "no_duplicate_annotation_ids",
                 "no_duplicate_category_ids", "no_invalid_image_refs",
                 "no_invalid_category_refs"):
        if not checks[name]:
            record_error(name, "structural integrity violation")

    # --- C. Dataset counts (computed; expected comparison) --------------- #
    observed_counts = {
        "images": structure.num_images,
        "annotations": structure.num_annotations,
        "categories": structure.num_categories,
        "zero_gt_images": len(structure.zero_gt_image_ids),
    }
    count_matches = {
        "images": structure.num_images == args.expected_images,
        "annotations": structure.num_annotations == args.expected_annotations,
        "categories": structure.num_categories == args.expected_categories,
        "zero_gt_images":
            len(structure.zero_gt_image_ids) == args.expected_empty_images,
    }
    checks["counts_match_expected"] = all(count_matches.values())
    if not checks["counts_match_expected"]:
        detail = {"observed": observed_counts, "expected": {
            "images": args.expected_images,
            "annotations": args.expected_annotations,
            "categories": args.expected_categories,
            "zero_gt_images": args.expected_empty_images}}
        if args.strict:
            record_error("counts_match_expected", detail)
        else:
            warnings.append(f"count discrepancy vs expected: {detail}")

    # --- Selection (REPRESENTATIVE visual QA scope ONLY) ----------------- #
    selection = build_selection(structure, args.num_abnormal, args.num_zero_gt)

    # ==================================================================== #
    # EXHAUSTIVE AUTOMATED VALIDATION over ALL images + ALL annotations.    #
    # This — not the representative loop below — is the authoritative source #
    # for the automated technical verdict.                                  #
    # ==================================================================== #
    full = validate_full_dataset(structure, image_root)
    for sample in full["failure_samples"]:
        record_error("full_dataset_" + sample["type"], sample)

    checks["full_dataset_coverage"] = full["coverage_status"] == "PASS"
    checks["full_dataset_jpeg_availability"] = \
        full["jpeg_availability_status"] == "PASS"
    checks["full_dataset_dimension_consistency"] = \
        full["dimension_consistency_status"] == "PASS"
    checks["full_dataset_bbox_validity"] = \
        full["bbox_validity_status"] == "PASS"
    checks["full_dataset_reference_integrity"] = \
        full["reference_integrity_status"] == "PASS"
    checks["full_dataset_category_mapping"] = \
        full["category_mapping_status"] == "PASS"

    # ==================================================================== #
    # REPRESENTATIVE VISUAL QA: render overlays for the SELECTED samples    #
    # ONLY. This never covers the whole dataset and never gates the         #
    # automated technical verdict; it exists for manual gross-misalignment  #
    # review. Per-sample geometry columns reuse the SAME validators.        #
    # ==================================================================== #
    manifest_rows: List[Dict[str, Any]] = []
    overlay_dir = plots_dir
    overlay_paths: List[Tuple[int, bool, Path]] = []
    categories_represented: set = set()

    for item in selection:
        iid = item["image_id"]
        img = structure.images_by_id[iid]
        file_name = str(img["file_name"])
        coco_w, coco_h = float(img["width"]), float(img["height"])
        anns = structure.anns_by_image.get(iid, [])
        present_cats = sorted({int(a["category_id"]) for a in anns})
        categories_represented.update(present_cats)
        cat_names = [structure.category_id_to_name.get(c, f"UNKNOWN_{c}")
                     for c in present_cats]

        res = resolve_jpeg(image_root, file_name)
        resolved_path = res.get("path")
        if res["status"] in ("missing", "ambiguous"):
            # Recorded exhaustively by validate_full_dataset already; here it
            # only means this selected sample cannot be rendered.
            warnings.append(
                f"representative sample image_id={iid} not renderable "
                f"(jpeg {res['status']})")

        loaded_w = loaded_h = None
        boxes_drawn = None
        bbox_valid_count = 0
        bbox_invalid_count = 0
        min_area = max_area = None
        near_boundary = False
        overlay_rel = None

        if resolved_path is not None:
            areas = []
            for ann in anns:
                vb = validate_bbox_xywh(ann["bbox"], coco_w, coco_h)
                if vb["valid"]:
                    bbox_valid_count += 1
                else:
                    bbox_invalid_count += 1  # exhaustively counted in `full`
                areas.append(float(ann["bbox"][2]) * float(ann["bbox"][3]))
                if bbox_is_near_boundary(ann["bbox"], coco_w, coco_h):
                    near_boundary = True
            min_area = min(areas) if areas else None
            max_area = max(areas) if areas else None

            overlay_name = (f"overlay_id{iid:05d}"
                            f"{'_zeroGT' if item['zero_gt'] else ''}.png")
            overlay_path = overlay_dir / overlay_name
            try:
                rr = render_overlay(
                    Path(resolved_path), overlay_path, iid, anns,
                    structure.category_id_to_name, structure.category_ids,
                    item["zero_gt"], args.show_ann_id, args.dpi)
                loaded_w, loaded_h = rr["loaded_width"], rr["loaded_height"]
                boxes_drawn = rr["boxes_drawn"]
                overlay_rel = overlay_name
                overlay_paths.append((iid, item["zero_gt"], overlay_path))
            except Exception as exc:  # noqa: BLE001 - record, never hide
                record_error("representative_render_failed",
                             {"image_id": iid, "error":
                              f"{type(exc).__name__}: {exc}"})

        manifest_rows.append({
            "image_id": iid,
            "file_name": file_name,
            "resolved_jpeg_path": resolved_path,
            "selection_reason": "|".join(item["selection_reasons"]),
            "width": int(coco_w),
            "height": int(coco_h),
            "loaded_width": loaded_w,
            "loaded_height": loaded_h,
            "num_annotations": len(anns),
            "bbox_count": len(anns),
            "num_categories_present": len(present_cats),
            "category_names": ";".join(cat_names),
            "zero_gt": item["zero_gt"],
            "near_boundary": near_boundary,
            "min_bbox_area": min_area,
            "max_bbox_area": max_area,
            "bbox_valid_count": bbox_valid_count,
            "bbox_invalid_count": bbox_invalid_count,
            "boxes_drawn": boxes_drawn,
            "jpeg_status": res["status"],
            "overlay_artifact": overlay_rel,
        })

    checks["representative_overlays_rendered"] = \
        len(overlay_paths) == len(selection) and len(selection) > 0

    # --- Contact sheet (representative artifact) -------------------------- #
    contact_sheet_path = None
    if overlay_paths:
        cs_out = overlay_dir / "contact_sheet_post_coco_overlay.png"
        try:
            contact_sheet_path = render_contact_sheet(
                overlay_paths, cs_out, args.dpi, cols=max(1, args.contact_cols))
        except Exception as exc:  # noqa: BLE001
            record_error("contact_sheet_failed",
                         {"error": f"{type(exc).__name__}: {exc}"})
    checks["contact_sheet_created"] = contact_sheet_path is not None

    # ==================================================================== #
    # AUTOMATED VERDICT — driven by the EXHAUSTIVE full-dataset invariants  #
    # (plus parse/structure), NOT by the representative visual sample.      #
    # ==================================================================== #
    required_checks = (
        "coco_exists", "coco_json_parse", "coco_structure_analyzed",
        "images_present", "annotations_present", "categories_present",
        "no_duplicate_image_ids", "no_duplicate_annotation_ids",
        "no_duplicate_category_ids", "no_invalid_image_refs",
        "no_invalid_category_refs",
        "full_dataset_coverage",
        "full_dataset_jpeg_availability", "full_dataset_dimension_consistency",
        "full_dataset_bbox_validity", "full_dataset_reference_integrity",
        "full_dataset_category_mapping")
    automated_pass = all(checks.get(k, False) for k in required_checks)
    if args.strict:
        automated_pass = automated_pass and checks.get(
            "counts_match_expected", False) and checks.get(
            "coco_sha256_matches_recorded", False)

    report_dict = _assemble_report(
        args, repo_root, coco_json, image_root, report_json, manifest_csv,
        plots_dir, coco_sha256, structure, observed_counts, selection,
        manifest_rows, len(overlay_paths), contact_sheet_path, checks, errors,
        warnings, categories_represented, full, automated_pass)

    _write_outputs(report_json, manifest_csv, report_dict, manifest_rows)
    return report_dict, automated_pass


def _assemble_report(args, repo_root, coco_json, image_root, report_json,
                     manifest_csv, plots_dir, coco_sha256, structure,
                     observed_counts, selection, manifest_rows, overlay_count,
                     contact_sheet_path, checks, errors, warnings,
                     categories_represented, full, automated_pass
                     ) -> Dict[str, Any]:
    """Assemble the machine-readable report dict.

    ``full`` is the exhaustive full-dataset validation result (or None when the
    run failed before it could execute). The report keeps the EXHAUSTIVE
    automated scope and the REPRESENTATIVE visual scope in clearly separate,
    non-overlapping blocks so a reader can never mistake a 16-sample result for
    a whole-dataset result.
    """
    num_abnormal = sum(1 for s in selection if not s["zero_gt"])
    num_zero_gt = sum(1 for s in selection if s["zero_gt"])
    total_cats = structure.num_categories if structure else None
    total_imgs = structure.num_images if structure else None
    total_anns = structure.num_annotations if structure else None
    full = full or {}
    exhaustive_block = {
        "scope": "ALL COCO images + ALL COCO annotations",
        "full_dataset_images_checked": full.get("images_checked"),
        "full_dataset_total_images": total_imgs,
        "full_dataset_annotations_checked": full.get("annotations_checked"),
        "full_dataset_total_annotations": total_anns,
        "full_dataset_annotations_geometry_unchecked_count":
            full.get("annotations_geometry_unchecked_count"),
        "full_dataset_coverage_status": full.get("coverage_status"),
        "full_dataset_jpegs_resolved": full.get("jpegs_resolved"),
        "full_dataset_missing_jpeg_count": full.get("missing_jpeg_count"),
        "full_dataset_ambiguous_jpeg_count": full.get("ambiguous_jpeg_count"),
        "full_dataset_unreadable_jpeg_count": full.get("unreadable_jpeg_count"),
        "full_dataset_dimension_mismatch_count":
            full.get("dimension_mismatch_count"),
        "full_dataset_invalid_bbox_count": full.get("invalid_bbox_count"),
        "full_dataset_invalid_image_reference_count":
            full.get("invalid_image_reference_count"),
        "full_dataset_invalid_category_reference_count":
            full.get("invalid_category_reference_count"),
        "full_dataset_jpeg_availability_status":
            full.get("jpeg_availability_status"),
        "full_dataset_dimension_consistency_status":
            full.get("dimension_consistency_status"),
        "full_dataset_bbox_validity_status": full.get("bbox_validity_status"),
        "full_dataset_reference_integrity_status":
            full.get("reference_integrity_status"),
        "full_dataset_category_mapping_status":
            full.get("category_mapping_status"),
        "full_dataset_all_invariants_pass": full.get("all_invariants_pass"),
        "full_dataset_bbox_tolerance_px": BBOX_BOUND_TOLERANCE,
        "full_dataset_bbox_tolerance_note": (
            "Intentional +/-1px in-bounds tolerance from the existing "
            "validate_bbox_xywh validator; preserved, not changed."),
        "full_dataset_images_fully_covered":
            (full.get("images_checked") == total_imgs
             if total_imgs is not None else None),
        "full_dataset_annotations_fully_covered":
            (full.get("annotations_checked") == total_anns
             if total_anns is not None else None),
        "full_dataset_failure_samples": full.get("failure_samples", []),
        "full_dataset_failure_samples_truncated":
            full.get("failure_samples_truncated"),
    }
    representative_block = {
        "scope": "deterministic stress/representative selection only",
        "not_statistical_proof": (
            "A representative visual sample does not prove the absence of errors "
            "across the entire dataset."),
        "selected_sample_count": len(selection),
        "images_with_gt": num_abnormal,
        "zero_gt_samples": num_zero_gt,
        "categories_represented": sorted(categories_represented),
        "category_coverage": {
            "represented": len(categories_represented),
            "total": total_cats,
            "complete": (total_cats is not None
                         and len(categories_represented) == total_cats),
        },
        "selection_strategy": (
            "deterministic; ascending-image_id tie-breaks; category coverage + "
            "fewest/most bboxes + smallest/largest bbox area + most categories + "
            "near-boundary + first-N zero-GT; no randomness"),
        "selection": selection,
        "num_overlay_artifacts": overlay_count,
        "overlays_rendered_ok": checks.get("representative_overlays_rendered"),
        "contact_sheet_path": contact_sheet_path,
        "manifest_path": str(manifest_csv),
    }
    return {
        "audit_name": AUDIT_NAME,
        "technical_label": TECHNICAL_LABEL,
        "note": ("Supporting QA evidence only; a visual sample does not prove "
                 "the absence of errors across the entire dataset. Not a new "
                 "phase, method, or algorithm."),
        "timestamp_utc": utc_now(),
        "seed": args.seed,
        "selection_strategy": (
            "deterministic; ascending-image_id tie-breaks; category coverage + "
            "fewest/most bboxes + smallest/largest bbox area + most categories + "
            "near-boundary + first-N zero-GT; no randomness"),
        "input_paths": {
            "repo_root": str(repo_root),
            "coco_json": str(coco_json),
            "image_root": str(image_root),
            "report_json": str(report_json),
            "manifest_csv": str(manifest_csv),
            "plots_dir": str(plots_dir),
        },
        "coco_sha256": coco_sha256,
        "coco_sha256_expected": args.expected_coco_sha256,
        "image_root_resolution": {
            "image_root": str(image_root),
            "file_name_convention_example":
                (manifest_rows[0]["file_name"] if manifest_rows else None),
        },
        "expected_counts": {
            "images": args.expected_images,
            "annotations": args.expected_annotations,
            "categories": args.expected_categories,
            "zero_gt_images": args.expected_empty_images,
        },
        "observed_counts": observed_counts,
        "total_coco_images": total_imgs,
        "total_coco_annotations": total_anns,
        "total_categories": total_cats,
        "total_zero_gt_images":
            len(structure.zero_gt_image_ids) if structure else None,
        "global_structural_precheck": {
            "no_duplicate_image_ids": checks.get("no_duplicate_image_ids"),
            "no_duplicate_annotation_ids":
                checks.get("no_duplicate_annotation_ids"),
            "no_duplicate_category_ids": checks.get("no_duplicate_category_ids"),
            "no_invalid_image_refs": checks.get("no_invalid_image_refs"),
            "no_invalid_category_refs": checks.get("no_invalid_category_refs"),
            "counts_match_expected": checks.get("counts_match_expected"),
        },
        # ----- EXHAUSTIVE AUTOMATED SCOPE (ALL images + ALL annotations) --- #
        "exhaustive_automated_validation": exhaustive_block,
        # ----- REPRESENTATIVE VISUAL SCOPE (selected samples only) --------- #
        "representative_visual_qa": representative_block,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "automated_checks_status": "PASS" if automated_pass else "FAIL",
        "automated_checks_scope":
            "exhaustive over ALL COCO images and ALL COCO annotations",
        "manual_visual_review_status": "PENDING_GPT_REVIEW",
        "manual_visual_review_scope":
            "representative selected samples only",
        "dataset_or_coco_modified": False,
        "scientific_scope": (
            "Exhaustive automated validation checks the code-verifiable technical "
            "invariants over the entire COCO/JPEG dataset. The representative "
            "visual overlay audit is supporting evidence to detect gross "
            "COCO-to-JPEG spatial/geometry misalignment on a purposeful "
            "stress/representative subset. A visual sample does not prove the "
            "absence of errors across the entire dataset, is not clinical ground "
            "truth, and implies no bbox fusion/dedup decision."),
    }


def _write_outputs(report_json: Path, manifest_csv: Path,
                   report_dict: Dict[str, Any],
                   manifest_rows: List[Dict[str, Any]]) -> None:
    """Write the report JSON and manifest CSV atomically."""
    write_json_atomic(report_json, report_dict)
    header = [
        "image_id", "file_name", "resolved_jpeg_path", "selection_reason",
        "width", "height", "loaded_width", "loaded_height", "num_annotations",
        "bbox_count", "num_categories_present", "category_names", "zero_gt",
        "near_boundary", "min_bbox_area", "max_bbox_area", "bbox_valid_count",
        "bbox_invalid_count", "boxes_drawn", "jpeg_status", "overlay_artifact",
    ]
    rows = [[r.get(k) for k in header] for r in manifest_rows]
    write_csv_atomic(manifest_csv, header, rows)


def _finalize_failure(args, repo_root, coco_json, image_root, report_json,
                      manifest_csv, plots_dir, checks, errors, warnings,
                      coco_sha256, structure, selection, manifest_rows,
                      overlay_count, contact_sheet) -> Tuple[Dict[str, Any], bool]:
    """Write a FAIL report early (e.g. missing/malformed COCO) and return it."""
    observed_counts = {
        "images": structure.num_images if structure else None,
        "annotations": structure.num_annotations if structure else None,
        "categories": structure.num_categories if structure else None,
        "zero_gt_images":
            len(structure.zero_gt_image_ids) if structure else None,
    }
    report_dict = _assemble_report(
        args, repo_root, coco_json, image_root, report_json, manifest_csv,
        plots_dir, coco_sha256, structure, observed_counts, selection,
        manifest_rows, overlay_count, contact_sheet, checks, errors, warnings,
        set(), None, False) if structure else _minimal_fail_report(
        args, repo_root, coco_json, image_root, report_json, manifest_csv,
        plots_dir, coco_sha256, checks, errors, warnings)
    _write_outputs(report_json, manifest_csv, report_dict, manifest_rows)
    return report_dict, False


def _minimal_fail_report(args, repo_root, coco_json, image_root, report_json,
                         manifest_csv, plots_dir, coco_sha256, checks, errors,
                         warnings) -> Dict[str, Any]:
    """Assemble a minimal FAIL report when structure could not be analyzed."""
    return {
        "audit_name": AUDIT_NAME,
        "technical_label": TECHNICAL_LABEL,
        "timestamp_utc": utc_now(),
        "seed": args.seed,
        "input_paths": {
            "repo_root": str(repo_root),
            "coco_json": str(coco_json),
            "image_root": str(image_root),
            "report_json": str(report_json),
            "manifest_csv": str(manifest_csv),
            "plots_dir": str(plots_dir),
        },
        "coco_sha256": coco_sha256,
        "coco_sha256_expected": args.expected_coco_sha256,
        "observed_counts": {"images": None, "annotations": None,
                            "categories": None, "zero_gt_images": None},
        "total_coco_images": None,
        "total_coco_annotations": None,
        "total_categories": None,
        "total_zero_gt_images": None,
        "exhaustive_automated_validation": {
            "scope": "ALL COCO images + ALL COCO annotations",
            "full_dataset_images_checked": None,
            "full_dataset_annotations_checked": None,
        },
        "representative_visual_qa": {
            "selected_sample_count": 0,
            "images_with_gt": 0,
            "zero_gt_samples": 0,
            "category_coverage": {"represented": 0, "total": None},
            "num_overlay_artifacts": 0,
            "contact_sheet_path": None,
            "manifest_path": str(manifest_csv),
        },
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "automated_checks_status": "FAIL",
        "automated_checks_scope":
            "exhaustive over ALL COCO images and ALL COCO annotations",
        "manual_visual_review_status": "PENDING_GPT_REVIEW",
        "dataset_or_coco_modified": False,
    }


def _status_str(value: Optional[str]) -> str:
    """Normalize a PASS/FAIL/None status string for console printing."""
    return value if value in ("PASS", "FAIL") else "N/A"


def print_console_summary(report: Dict[str, Any]) -> None:
    """Print the required audit-friendly console summary (two explicit scopes)."""
    full = report.get("exhaustive_automated_validation", {}) or {}
    rep = report.get("representative_visual_qa", {}) or {}
    cov = rep.get("category_coverage", {}) or {}

    print("=" * 50)
    print("POST-COCO VISUAL OVERLAY AUDIT")
    print("=" * 50)
    print(f"COCO source: {report.get('input_paths', {}).get('coco_json')}")
    print(f"COCO SHA-256: {report.get('coco_sha256')}")
    print(f"Total images: {report.get('total_coco_images')}")
    print(f"Total annotations: {report.get('total_coco_annotations')}")
    print(f"Categories: {report.get('total_categories')}")
    print(f"Zero-GT images: {report.get('total_zero_gt_images')}")
    print("")
    print("EXHAUSTIVE AUTOMATED VALIDATION")
    print(f"Images checked: {full.get('full_dataset_images_checked')}/"
          f"{report.get('total_coco_images')}")
    print(f"Annotations checked: {full.get('full_dataset_annotations_checked')}/"
          f"{report.get('total_coco_annotations')}")
    print(f"Annotations geometry-unchecked (orphans): "
          f"{full.get('full_dataset_annotations_geometry_unchecked_count')}")
    print(f"Coverage: "
          f"{_status_str(full.get('full_dataset_coverage_status'))}")
    print(f"JPEG availability: "
          f"{_status_str(full.get('full_dataset_jpeg_availability_status'))}")
    print(f"Dimension consistency: "
          f"{_status_str(full.get('full_dataset_dimension_consistency_status'))}")
    print(f"BBox validity: "
          f"{_status_str(full.get('full_dataset_bbox_validity_status'))}")
    print(f"Image/category reference integrity: "
          f"{_status_str(full.get('full_dataset_reference_integrity_status'))}")
    print(f"Category mapping: "
          f"{_status_str(full.get('full_dataset_category_mapping_status'))}")
    print(f"Missing JPEG: {full.get('full_dataset_missing_jpeg_count')}")
    print(f"Ambiguous JPEG: {full.get('full_dataset_ambiguous_jpeg_count')}")
    print(f"Unreadable JPEG: {full.get('full_dataset_unreadable_jpeg_count')}")
    print(f"Dimension mismatches: "
          f"{full.get('full_dataset_dimension_mismatch_count')}")
    print(f"Invalid bboxes: {full.get('full_dataset_invalid_bbox_count')}")
    print(f"Invalid image references: "
          f"{full.get('full_dataset_invalid_image_reference_count')}")
    print(f"Invalid category references: "
          f"{full.get('full_dataset_invalid_category_reference_count')}")
    print(f"Automated status: {report.get('automated_checks_status')}")
    print("")
    print("REPRESENTATIVE VISUAL QA")
    print(f"Selected visual samples: {rep.get('selected_sample_count')}")
    print(f"Images with GT: {rep.get('images_with_gt')}")
    print(f"Zero-GT samples: {rep.get('zero_gt_samples')}")
    print(f"Categories represented: {cov.get('represented')}/{cov.get('total')}")
    print(f"Overlay artifacts: {rep.get('num_overlay_artifacts')}")
    print(f"Contact sheet: {rep.get('contact_sheet_path')}")
    print(f"Manual visual review: {report.get('manual_visual_review_status')}")
    print(f"Dataset/COCO modified: "
          f"{'NO' if not report.get('dataset_or_coco_modified') else 'YES'}")
    print("=" * 50)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns process exit code (0 = automated PASS)."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    report_dict, automated_pass = run_audit(args)
    print_console_summary(report_dict)
    return 0 if automated_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
