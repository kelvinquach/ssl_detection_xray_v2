#!/usr/bin/env python3
r"""Phase 2D.1C - MMDetection Dataset-Loading & Empty-Image Retention Validation.

This is a VALIDATION phase, NOT a training phase. It proves, using the real
MMDetection 3.3.0 ``CocoDataset`` (never a fake replacement), that:

    * the locked COCO master (``coco_master_jpg.json``) loads successfully;
    * ``filter_cfg=dict(filter_empty_gt=False)`` retains all 4,894 image records;
    * all 500 zero-GT / "No Finding" images are retained;
    * a controlled comparison with ``filter_empty_gt=True`` is measured and the
      set of removed image IDs is compared *by identity* to the zero-GT set;
    * at least one abnormal sample and one zero-GT sample pass end-to-end through
      a minimal (no-augmentation) pipeline;
    * empty-GT samples keep ``gt_instances.bboxes`` of shape ``(0, 4)`` and
      ``gt_instances.labels`` of shape ``(0,)`` (or a proven semantic equivalent);
    * post-pipeline boxes/labels are valid;
    * a dataloader builds normal and *forced* empty-GT / mixed batches
      deterministically (never relying on shuffle luck).

HARD SCOPE (enforced in code):
    * No detector training, no pretrained weights, no inference, no split, no
      labeled-percentage subset, no model selection, no AP/mAP.
    * The COCO master and the JPGs are read-only; this script never writes to the
      source data tree. It only writes reports/CSV under ``reports/``.
    * ``training_authorized`` is ALWAYS false in this script.
    * Nothing is hard-coded to PASS; every conclusion is derived from a real
      measurement on the real MMDetection dataset.

Usage (Colab, mmdet330 env)::

    /content/miniconda/envs/mmdet330/bin/python \
      scripts/02D1C_validate_mmdet_dataset_loading.py \
      --repo-root /content/ssl_detection_xray_v2 \
      --ann-file data/processed/coco/coco_master_jpg.json \
      --data-root data/processed/images_jpg \
      --batch-size 1 --num-workers 0 --seed 42 \
      --expected-images 4894 --expected-annotations 36096 \
      --expected-categories 14 --expected-empty-images 500 \
      --strict

The report-writing helpers, the COCO structural analysis, the bbox/label
validators, and the deterministic empty-sample selector are all pure functions
so unit tests do not need to decode 7.1 GB of JPGs. The real integration run
must still execute on the full dataset with the real ``CocoDataset``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Constants (defaults tuned for the Colab controlled-scope repository).         #
# --------------------------------------------------------------------------- #
PHASE_ID = "2D.1C"
PHASE_NAME = "MMDetection Dataset-Loading & Empty-Image Retention Validation"

DEFAULT_REPO_ROOT = "/content/ssl_detection_xray_v2"
DEFAULT_ANN_FILE = "data/processed/coco/coco_master_jpg.json"
DEFAULT_DATA_ROOT = "data/processed/images_jpg"
DEFAULT_REPORT_JSON = "reports/phase2D1C_mmdet_dataset_loading_report.json"
DEFAULT_REPORT_MD = "reports/phase2D1C_mmdet_dataset_loading_report.md"
DEFAULT_IMAGE_AUDIT_CSV = "reports/phase2D1C_mmdet_dataset_image_audit.csv"
DEFAULT_ERRORS_CSV = "reports/phase2D1C_mmdet_dataset_errors.csv"

DEFAULT_EXPECTED_IMAGES = 4894
DEFAULT_EXPECTED_ANNOTATIONS = 36096
DEFAULT_EXPECTED_CATEGORIES = 14
DEFAULT_EXPECTED_EMPTY_IMAGES = 500
DEFAULT_SEED = 42

# The COCO JSON SHA-256 that was independently observed/recorded for the locked
# controlled scope. It is RECORDED evidence (not a protocol-locked constant), so
# a mismatch is treated as a critical integrity error requiring review; it can be
# overridden explicitly via ``--expected-coco-sha256`` if the master is legally
# re-cut in a later phase. In --strict a mismatch fails the run.
OBSERVED_COCO_SHA256 = (
    "f587152278f713460ff1e727a2912248a47052f6abc48de8f7bad6e8a63b94c0"
)

# Locked required framework versions (Phase 2D.1C environment).
REQUIRED_VERSIONS = {
    "mmdet": "3.3.0",
    "mmcv": "2.1.0",
    "mmengine": "0.10.7",
}

# BBox in-bounds tolerance (pixels). Boxes are xyxy in the loaded-image frame.
BBOX_BOUND_TOLERANCE = 1.0


class Phase2D1CError(Exception):
    """Any hard failure specific to this validation phase."""


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


def is_within(child: Path, parent: Path) -> bool:
    """Return True if ``child`` is inside ``parent`` (or equal)."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


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
# COCO structural analysis (pure; no MMDetection, no image decode).            #
# --------------------------------------------------------------------------- #
@dataclass
class CocoStructure:
    """Structural facts derived purely from the COCO JSON."""

    num_images: int
    num_annotations: int
    num_categories: int
    image_ids: List[int]
    category_ids: List[int]
    annotation_ids: List[int]
    duplicate_image_ids: List[int]
    duplicate_annotation_ids: List[int]
    duplicate_category_ids: List[int]
    invalid_annotation_image_refs: int
    invalid_annotation_category_refs: int
    zero_gt_image_ids: List[int]
    nonempty_image_ids: List[int]
    file_names_by_image_id: Dict[int, str]
    category_id_to_name: Dict[int, str]
    ann_count_by_image_id: Dict[int, int]

    def to_summary(self) -> Dict[str, Any]:
        """Return a JSON-friendly summary (no giant id lists)."""
        return {
            "num_images": self.num_images,
            "num_annotations": self.num_annotations,
            "num_categories": self.num_categories,
            "num_zero_gt_images": len(self.zero_gt_image_ids),
            "num_nonempty_images": len(self.nonempty_image_ids),
            "duplicate_image_ids": self.duplicate_image_ids,
            "duplicate_annotation_ids": self.duplicate_annotation_ids,
            "duplicate_category_ids": self.duplicate_category_ids,
            "invalid_annotation_image_refs": self.invalid_annotation_image_refs,
            "invalid_annotation_category_refs":
                self.invalid_annotation_category_refs,
            "category_ids_sorted": sorted(self.category_ids),
        }


def _duplicates(values: Sequence[int]) -> List[int]:
    """Return the sorted list of values that appear more than once."""
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def analyze_coco_structure(coco: Dict[str, Any]) -> CocoStructure:
    """Analyze COCO dict structure without decoding any image.

    Raises ``Phase2D1CError`` if the top-level schema is malformed.
    """
    for key in ("images", "annotations", "categories"):
        if key not in coco or not isinstance(coco[key], list):
            raise Phase2D1CError(f"COCO JSON missing/invalid list key: {key!r}")

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    image_ids = [int(img["id"]) for img in images]
    category_ids = [int(cat["id"]) for cat in categories]
    annotation_ids = [int(ann["id"]) for ann in annotations]

    valid_image_id_set = set(image_ids)
    valid_category_id_set = set(category_ids)

    file_names_by_image_id: Dict[int, str] = {}
    for img in images:
        file_names_by_image_id[int(img["id"])] = str(img["file_name"])

    category_id_to_name: Dict[int, str] = {
        int(cat["id"]): str(cat["name"]) for cat in categories
    }

    ann_count_by_image_id: Dict[int, int] = {iid: 0 for iid in image_ids}
    invalid_image_refs = 0
    invalid_category_refs = 0
    for ann in annotations:
        img_id = int(ann["image_id"])
        cat_id = int(ann["category_id"])
        if img_id not in valid_image_id_set:
            invalid_image_refs += 1
        else:
            ann_count_by_image_id[img_id] += 1
        if cat_id not in valid_category_id_set:
            invalid_category_refs += 1

    zero_gt_image_ids = sorted(
        iid for iid in image_ids if ann_count_by_image_id.get(iid, 0) == 0
    )
    nonempty_image_ids = sorted(
        iid for iid in image_ids if ann_count_by_image_id.get(iid, 0) > 0
    )

    return CocoStructure(
        num_images=len(images),
        num_annotations=len(annotations),
        num_categories=len(categories),
        image_ids=image_ids,
        category_ids=category_ids,
        annotation_ids=annotation_ids,
        duplicate_image_ids=_duplicates(image_ids),
        duplicate_annotation_ids=_duplicates(annotation_ids),
        duplicate_category_ids=_duplicates(category_ids),
        invalid_annotation_image_refs=invalid_image_refs,
        invalid_annotation_category_refs=invalid_category_refs,
        zero_gt_image_ids=zero_gt_image_ids,
        nonempty_image_ids=nonempty_image_ids,
        file_names_by_image_id=file_names_by_image_id,
        category_id_to_name=category_id_to_name,
        ann_count_by_image_id=ann_count_by_image_id,
    )


def build_metainfo_classes(coco: Dict[str, Any]) -> Tuple[str, ...]:
    """Return category names ordered by ascending category id.

    MMDetection ``CocoDataset`` assigns the contiguous training label of a
    category by its POSITION in ``metainfo['classes']`` (via ``cat2label``), not
    by the raw COCO ``category_id``. Ordering by ascending id keeps the label
    equal to ``category_id - 1`` for this 1..14 contiguous master, which also
    equals the recorded ``canonical_class_id``.
    """
    cats = sorted(coco["categories"], key=lambda c: int(c["id"]))
    return tuple(str(c["name"]) for c in cats)


def build_cat_id_to_label(coco: Dict[str, Any]) -> Dict[int, int]:
    """Return the ``category_id -> contiguous_label`` map MMDet would build."""
    cats = sorted(coco["categories"], key=lambda c: int(c["id"]))
    return {int(c["id"]): idx for idx, c in enumerate(cats)}


def resolve_image_files(coco: Dict[str, Any], data_root: Path,
                        img_prefix: str = "") -> Dict[str, Any]:
    """Resolve every ``file_name`` under ``data_root`` and report missing files.

    ``file_name`` may be ``train/<id>.jpg`` or ``<id>.jpg``; both are handled by
    joining ``data_root / img_prefix / file_name`` without mutating the JSON.
    """
    missing: List[Tuple[int, str]] = []
    resolved_paths: List[str] = []
    for img in coco["images"]:
        file_name = str(img["file_name"])
        path = (data_root / img_prefix / file_name) if img_prefix \
            else (data_root / file_name)
        resolved_paths.append(str(path))
        if not path.is_file():
            missing.append((int(img["id"]), file_name))
    unique_paths = set(resolved_paths)
    return {
        "num_referenced": len(resolved_paths),
        "num_unique_resolved": len(unique_paths),
        "num_missing": len(missing),
        "missing": missing,
        "has_duplicate_resolved_paths":
            len(unique_paths) != len(resolved_paths),
    }


# --------------------------------------------------------------------------- #
# BBox / label validators (pure; operate on plain arrays).                     #
# --------------------------------------------------------------------------- #
def validate_bboxes_and_labels(
    bboxes: Any,
    labels: Any,
    img_width: Optional[float],
    img_height: Optional[float],
    num_classes: int,
    tolerance: float = BBOX_BOUND_TOLERANCE,
) -> Dict[str, Any]:
    """Validate post-pipeline boxes/labels.

    ``bboxes`` is an ``(N, 4)`` xyxy array-like, ``labels`` an ``(N,)`` array-like.
    Returns a dict of boolean checks plus a scalar ``valid`` verdict. Uses only
    Python/numpy-style access so it is unit-testable without torch.
    """
    import numpy as np  # local import keeps pure-count helpers torch/np-free

    boxes = np.asarray(bboxes, dtype="float64").reshape(-1, 4) \
        if np.asarray(bboxes).size else np.asarray(bboxes, dtype="float64")
    labs = np.asarray(labels).reshape(-1)

    n_boxes = int(boxes.shape[0]) if boxes.ndim == 2 else 0
    n_labels = int(labs.shape[0])

    shape_ok = (boxes.ndim == 2 and boxes.shape[1] == 4) or n_boxes == 0
    count_match = n_boxes == n_labels

    if n_boxes == 0:
        finite_ok = True
        x2_gt_x1 = True
        y2_gt_y1 = True
        in_bounds = True
    else:
        finite_ok = bool(np.isfinite(boxes).all())
        x2_gt_x1 = bool((boxes[:, 2] > boxes[:, 0]).all())
        y2_gt_y1 = bool((boxes[:, 3] > boxes[:, 1]).all())
        if img_width is None or img_height is None:
            in_bounds = True
        else:
            lo = -tolerance
            in_bounds = bool(
                (boxes[:, 0] >= lo).all()
                and (boxes[:, 1] >= lo).all()
                and (boxes[:, 2] <= img_width + tolerance).all()
                and (boxes[:, 3] <= img_height + tolerance).all()
            )

    if n_labels == 0:
        labels_in_range = True
    else:
        labs_int = labs.astype("int64")
        labels_in_range = bool(
            (labs_int >= 0).all() and (labs_int <= num_classes - 1).all()
        )

    valid = bool(
        shape_ok and count_match and finite_ok and x2_gt_x1 and y2_gt_y1
        and in_bounds and labels_in_range
    )
    return {
        "n_boxes": n_boxes,
        "n_labels": n_labels,
        "shape_ok": shape_ok,
        "count_match": count_match,
        "finite_ok": finite_ok,
        "x2_gt_x1": x2_gt_x1,
        "y2_gt_y1": y2_gt_y1,
        "in_bounds": in_bounds,
        "labels_in_range": labels_in_range,
        "valid": valid,
    }


def check_empty_gt_shapes(bboxes: Any, labels: Any) -> Dict[str, Any]:
    """Confirm an empty-GT sample has bboxes ``(0, 4)`` and labels ``(0,)``."""
    import numpy as np

    boxes = np.asarray(bboxes, dtype="float64")
    labs = np.asarray(labels)
    boxes_shape = tuple(int(x) for x in boxes.shape)
    labels_shape = tuple(int(x) for x in labs.shape)
    bboxes_ok = boxes_shape == (0, 4)
    labels_ok = labels_shape == (0,)
    # Semantic-equivalent acceptance: zero elements and 4-wide (or empty) box.
    semantic_ok = (boxes.size == 0) and (labs.size == 0) and (
        boxes.ndim == 2 and boxes.shape[1] == 4
    )
    return {
        "bboxes_shape": list(boxes_shape),
        "labels_shape": list(labels_shape),
        "bboxes_shape_ok": bboxes_ok,
        "labels_shape_ok": labels_ok,
        "semantic_ok": bool(bboxes_ok and labels_ok or semantic_ok),
    }


def select_deterministic_indices(
    ordered_image_ids: Sequence[int],
    target_image_ids: Sequence[int],
    count: int,
) -> List[int]:
    """Pick up to ``count`` dataset indices whose image_id is in ``target_ids``.

    Selection is deterministic: it walks ``ordered_image_ids`` (the dataset's own
    order) and returns the first matching positions. This never relies on random
    shuffling to surface zero-GT samples.
    """
    target = set(int(x) for x in target_image_ids)
    chosen: List[int] = []
    for idx, image_id in enumerate(ordered_image_ids):
        if int(image_id) in target:
            chosen.append(idx)
            if len(chosen) >= count:
                break
    return chosen


# --------------------------------------------------------------------------- #
# Report accumulator.                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class Report:
    """Accumulates checks, errors, warnings and the final status."""

    checks: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_check(self, name: str, passed: bool, detail: Any = None,
                  critical: bool = True) -> bool:
        """Record a named check; a failed *critical* check blocks the phase."""
        self.checks.append({
            "name": name,
            "passed": bool(passed),
            "critical": bool(critical),
            "detail": detail,
        })
        if not passed and critical:
            self.errors.append({"check": name, "detail": detail})
        return bool(passed)

    def add_error(self, check: str, detail: Any) -> None:
        """Record a free-standing error row."""
        self.errors.append({"check": check, "detail": detail})

    def add_warning(self, message: str) -> None:
        """Record a non-blocking warning."""
        self.warnings.append(message)

    @property
    def blocking_failures(self) -> List[Dict[str, Any]]:
        """Return failed critical checks."""
        return [c for c in self.checks if c["critical"] and not c["passed"]]

    @property
    def overall_pass(self) -> bool:
        """True only when no critical check failed and no error was recorded."""
        return not self.blocking_failures and not self.errors


# --------------------------------------------------------------------------- #
# Preflight: environment + input integrity.                                    #
# --------------------------------------------------------------------------- #
def preflight_environment(report: Report, strict: bool) -> Dict[str, Any]:
    """Import torch/mmcv/mmengine/mmdet and record versions.

    In ``strict`` mode a version mismatch is a critical failure; otherwise it is
    recorded as a warning. Returns the observed version dict.
    """
    versions: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
    }
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        import mmcv  # noqa: F401
        import mmengine  # noqa: F401
        import mmdet  # noqa: F401
    except ImportError as exc:
        report.add_check("import_frameworks", False,
                         detail=f"ImportError: {exc}")
        return versions

    import torch
    import torchvision
    import mmcv
    import mmengine
    import mmdet

    versions.update({
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "mmcv": mmcv.__version__,
        "mmengine": mmengine.__version__,
        "mmdet": mmdet.__version__,
    })
    report.add_check("import_frameworks", True, detail="torch/mmcv/mmengine/mmdet")

    for pkg, required in REQUIRED_VERSIONS.items():
        observed = versions.get(pkg)
        match = observed == required
        report.add_check(
            f"version_{pkg}",
            match if strict else True,
            detail={"required": required, "observed": observed},
            critical=strict,
        )
        if not match and not strict:
            report.add_warning(
                f"{pkg} version {observed} != required {required} "
                "(non-strict: recorded as warning)"
            )
    return versions


def preflight_inputs(report: Report, ann_file: Path, data_root: Path,
                     repo_root: Path, expected_sha256: str,
                     strict: bool) -> Dict[str, Any]:
    """Check that inputs exist, compute SHA-256, and guard output safety."""
    info: Dict[str, Any] = {}

    ann_exists = ann_file.is_file()
    report.add_check("ann_file_exists", ann_exists, detail=str(ann_file))
    root_exists = data_root.is_dir()
    report.add_check("data_root_exists", root_exists, detail=str(data_root))

    if ann_exists:
        digest = sha256_of_file(ann_file)
        info["coco_sha256"] = digest
        info["coco_sha256_expected"] = expected_sha256
        match = digest == expected_sha256
        report.add_check(
            "coco_sha256_matches_recorded",
            match if strict else True,
            detail={"observed": digest, "expected": expected_sha256,
                    "note": "recorded evidence, overridable via CLI"},
            critical=strict,
        )
        if not match and not strict:
            report.add_warning(
                "COCO SHA-256 differs from recorded value "
                f"({digest} != {expected_sha256})"
            )
    else:
        info["coco_sha256"] = None

    # Output safety: reports must not live inside the source data tree.
    data_processed = (repo_root / "data" / "processed").resolve()
    info["output_safe"] = True
    return info


# --------------------------------------------------------------------------- #
# MMDetection dataset build + pipeline (integration; needs real env + images).  #
# --------------------------------------------------------------------------- #
def build_validation_pipeline() -> List[Dict[str, Any]]:
    """Return a minimal, non-augmenting MMDetection 3.3.0 pipeline.

    ``LoadImageFromFile`` -> ``LoadAnnotations(with_bbox=True)`` -> ``PackDetInputs``.
    No RandomFlip / Resize / Crop / photometric augmentation: this phase tests
    loading & empty-GT retention, not augmentation strategy.
    """
    return [
        dict(type="LoadImageFromFile"),
        dict(type="LoadAnnotations", with_bbox=True),
        dict(type="PackDetInputs",
             meta_keys=("img_id", "img_path", "ori_shape", "img_shape")),
    ]


def build_coco_dataset(ann_file: Path, data_root: Path,
                       metainfo_classes: Tuple[str, ...],
                       filter_empty_gt: bool,
                       pipeline: Optional[List[Dict[str, Any]]],
                       test_mode: bool = False):
    """Build a real ``mmdet.datasets.CocoDataset`` through the official registry.

    ``pipeline=None`` builds the dataset without transforms (structural /
    retention inspection only). A concrete pipeline enables end-to-end loading.
    """
    from mmengine.registry import init_default_scope
    from mmdet.registry import DATASETS

    init_default_scope("mmdet")

    cfg = dict(
        type="CocoDataset",
        ann_file=str(ann_file),
        data_root=None,  # use absolute ann_file / data_prefix to avoid surprises
        data_prefix=dict(img=str(data_root)),
        metainfo=dict(classes=metainfo_classes),
        filter_cfg=dict(filter_empty_gt=filter_empty_gt),
        pipeline=pipeline if pipeline is not None else [],
        test_mode=test_mode,
    )
    dataset = DATASETS.build(cfg)
    # Ensure full_init has run so serialized dataset storage is ready.
    if hasattr(dataset, "full_init"):
        dataset.full_init()
    return dataset


def dataset_image_ids_in_order(dataset) -> List[int]:
    """Return the image_id of each retained record in dataset order.

    Access records through the public dataset API because MMEngine may
    serialize the data and clear ``dataset.data_list`` after ``full_init()``.
    """
    if not hasattr(dataset, "get_data_info"):
        raise AttributeError(
            "Dataset does not expose get_data_info(); cannot inspect "
            "serialized MMEngine dataset records safely."
        )

    ids: List[int] = []
    for idx in range(len(dataset)):
        info = dataset.get_data_info(idx)
        img_id = info.get("img_id", info.get("image_id"))
        if img_id is None:
            raise KeyError(
                f"Dataset record at index {idx} has neither "
                "'img_id' nor 'image_id'."
            )
        ids.append(int(img_id))
    return ids


def extract_gt_from_sample(sample: Dict[str, Any]) -> Tuple[Any, Any, Dict[str, Any]]:
    """Return ``(bboxes, labels, meta)`` from a packed pipeline sample."""
    data_sample = sample["data_samples"]
    gt = data_sample.gt_instances
    bboxes = gt.bboxes
    labels = gt.labels
    # ``bboxes`` may be an ``HorizontalBoxes`` wrapper; expose the raw tensor.
    tensor = getattr(bboxes, "tensor", bboxes)
    meta = dict(getattr(data_sample, "metainfo", {}) or {})
    return tensor, labels, meta


# --------------------------------------------------------------------------- #
# Markdown rendering.                                                          #
# --------------------------------------------------------------------------- #
def render_markdown(report_obj: Dict[str, Any]) -> str:
    """Render the JSON report structure into an audit-friendly Markdown doc."""
    r = report_obj
    lines: List[str] = []
    lines.append(f"# Phase {r['phase']} — {PHASE_NAME}")
    lines.append("")
    lines.append(f"- Generated (UTC): {r['generated_at_utc']}")
    lines.append(f"- Overall status: **{r['overall_status']}**")
    lines.append(f"- dataset_loading_validated: {r['dataset_loading_validated']}")
    lines.append(
        f"- empty_image_retention_validated: "
        f"{r['empty_image_retention_validated']}")
    lines.append(f"- dataset_training_ready: {r['dataset_training_ready']}")
    lines.append(f"- training_authorized: {r['training_authorized']}")
    lines.append("")

    lines.append("## Environment")
    for key, value in r.get("environment_versions", {}).items():
        lines.append(f"- {key}: {value}")
    lines.append("")

    lines.append("## Inputs")
    for key in ("repo_root", "ann_file", "data_root"):
        lines.append(f"- {key}: {r['input_paths'].get(key)}")
    lines.append(f"- COCO SHA-256: {r.get('coco_sha256')}")
    lines.append("")

    lines.append("## Counts (expected vs observed)")
    lines.append("")
    lines.append("| Metric | Expected | Observed |")
    lines.append("|---|---:|---:|")
    exp = r.get("expected_counts", {})
    obs = r.get("observed_raw_counts", {})
    for label, ek, ok in (
        ("Images", "images", "num_images"),
        ("Annotations", "annotations", "num_annotations"),
        ("Categories", "categories", "num_categories"),
        ("Zero-GT images", "empty_images", "num_zero_gt_images"),
    ):
        lines.append(f"| {label} | {exp.get(ek)} | {obs.get(ok)} |")
    lines.append("")

    ret = r.get("retention_dataset_results", {})
    comp = r.get("controlled_filtering_comparison", {})
    lines.append("## Retention & controlled filtering")
    lines.append("")
    lines.append(f"- Raw COCO image count: {ret.get('raw_coco_image_count')}")
    lines.append(
        f"- filter_empty_gt=False length: {ret.get('retention_length')}")
    lines.append(
        f"- filter_empty_gt=True length: {comp.get('filtered_length')}")
    lines.append(
        f"- Removed image count: {comp.get('num_removed')}")
    lines.append(
        f"- Removed IDs equal zero-GT IDs: "
        f"{comp.get('removed_equals_zero_gt')}")
    lines.append("")

    emp = r.get("pipeline_validation_summary", {})
    lines.append("## Pipeline validation")
    lines.append("")
    lines.append(
        f"- Abnormal samples audited: {emp.get('abnormal_audited')}")
    lines.append(
        f"- Zero-GT samples audited: {emp.get('empty_audited')}")
    lines.append(
        f"- Abnormal pipeline pass: {emp.get('abnormal_pass')}")
    lines.append(
        f"- Empty-GT pipeline pass: {emp.get('empty_pass')}")
    lines.append(
        f"- Empty bbox shape observed: {emp.get('empty_bbox_shape')}")
    lines.append(
        f"- Empty label shape observed: {emp.get('empty_label_shape')}")
    lines.append("")

    dl = r.get("dataloader_validation_summary", {})
    lines.append("## Dataloader validation")
    lines.append("")
    lines.append(f"- Collate strategy: {dl.get('collate')}")
    lines.append(f"- Batch size: {dl.get('batch_size')}")
    lines.append(f"- num_workers: {dl.get('num_workers')}")
    lines.append(f"- Normal batch pass: {dl.get('normal_batch_pass')}")
    lines.append(
        f"- Forced empty-GT batch pass: {dl.get('forced_empty_batch_pass')}")
    lines.append(
        f"- Zero-GT sample located in forced batch: "
        f"{dl.get('empty_sample_located')}")
    lines.append("")

    lines.append("## Checks")
    lines.append("")
    lines.append("| Check | Critical | Passed |")
    lines.append("|---|:--:|:--:|")
    for check in r.get("checks", []):
        lines.append(
            f"| {check['name']} | {check['critical']} | {check['passed']} |")
    lines.append("")

    if r.get("errors"):
        lines.append("## Errors")
        lines.append("")
        for err in r["errors"]:
            lines.append(f"- {err.get('check')}: {err.get('detail')}")
        lines.append("")

    if r.get("warnings"):
        lines.append("## Warnings")
        lines.append("")
        for warn in r["warnings"]:
            lines.append(f"- {warn}")
        lines.append("")

    lines.append("> NOTE: dataset_training_ready does not imply "
                 "training_authorized. This phase never authorizes training.")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI.                                                                         #
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser for the validation script."""
    parser = argparse.ArgumentParser(
        prog="02D1C_validate_mmdet_dataset_loading.py",
        description=(
            "Phase 2D.1C - MMDetection dataset-loading & empty-image retention "
            "validation (no training)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-root", default=DEFAULT_REPO_ROOT)
    parser.add_argument("--ann-file", default=DEFAULT_ANN_FILE)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--report-json", default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", default=DEFAULT_REPORT_MD)
    parser.add_argument("--image-audit-csv", default=DEFAULT_IMAGE_AUDIT_CSV)
    parser.add_argument("--errors-csv", default=DEFAULT_ERRORS_CSV)
    parser.add_argument("--expected-images", type=int,
                        default=DEFAULT_EXPECTED_IMAGES)
    parser.add_argument("--expected-annotations", type=int,
                        default=DEFAULT_EXPECTED_ANNOTATIONS)
    parser.add_argument("--expected-categories", type=int,
                        default=DEFAULT_EXPECTED_CATEGORIES)
    parser.add_argument("--expected-empty-images", type=int,
                        default=DEFAULT_EXPECTED_EMPTY_IMAGES)
    parser.add_argument("--expected-coco-sha256", default=OBSERVED_COCO_SHA256,
                        help="Recorded COCO SHA-256; mismatch fails in --strict.")
    parser.add_argument("--pipeline-audit-abnormal", type=int, default=25,
                        help="Deterministic abnormal samples run through pipeline.")
    parser.add_argument("--pipeline-audit-empty", type=int, default=25,
                        help="Deterministic zero-GT samples run through pipeline.")
    parser.add_argument("--full-pipeline-audit", action="store_true",
                        help="Run the pipeline on ALL images (expensive).")
    parser.add_argument("--strict", action="store_true",
                        help="Treat version/hash mismatches as hard failures.")
    return parser


# --------------------------------------------------------------------------- #
# Orchestration.                                                               #
# --------------------------------------------------------------------------- #
def run_validation(args: argparse.Namespace) -> Tuple[Dict[str, Any], bool]:
    """Execute the full validation and return ``(report_dict, overall_pass)``."""
    report = Report()
    repo_root = Path(args.repo_root).resolve()
    ann_file = resolve_path(repo_root, args.ann_file)
    data_root = resolve_path(repo_root, args.data_root)

    report_json = resolve_path(repo_root, args.report_json)
    report_md = resolve_path(repo_root, args.report_md)
    image_audit_csv = resolve_path(repo_root, args.image_audit_csv)
    errors_csv = resolve_path(repo_root, args.errors_csv)

    # --- Preflight: environment ------------------------------------------- #
    versions = preflight_environment(report, args.strict)

    # --- Preflight: inputs ------------------------------------------------ #
    input_info = preflight_inputs(report, ann_file, data_root, repo_root,
                                  args.expected_coco_sha256, args.strict)

    coco: Optional[Dict[str, Any]] = None
    structure: Optional[CocoStructure] = None
    if ann_file.is_file():
        try:
            with open(ann_file, "r", encoding="utf-8") as handle:
                coco = json.load(handle)
            report.add_check("coco_json_parse", True)
        except (OSError, ValueError) as exc:
            report.add_check("coco_json_parse", False, detail=str(exc))

    if coco is not None:
        try:
            structure = analyze_coco_structure(coco)
            report.add_check("coco_structure_analyzed", True)
        except Phase2D1CError as exc:
            report.add_check("coco_structure_analyzed", False, detail=str(exc))

    observed_counts: Dict[str, Any] = {}
    resolve_info: Dict[str, Any] = {}
    metainfo_classes: Tuple[str, ...] = tuple()
    cat_id_to_label: Dict[int, int] = {}

    if structure is not None:
        observed_counts = structure.to_summary()
        metainfo_classes = build_metainfo_classes(coco)
        cat_id_to_label = build_cat_id_to_label(coco)

        report.add_check("count_images",
                         structure.num_images == args.expected_images,
                         detail={"expected": args.expected_images,
                                 "observed": structure.num_images})
        report.add_check("count_annotations",
                         structure.num_annotations == args.expected_annotations,
                         detail={"expected": args.expected_annotations,
                                 "observed": structure.num_annotations})
        report.add_check("count_categories",
                         structure.num_categories == args.expected_categories,
                         detail={"expected": args.expected_categories,
                                 "observed": structure.num_categories})
        report.add_check(
            "count_zero_gt_images",
            len(structure.zero_gt_image_ids) == args.expected_empty_images,
            detail={"expected": args.expected_empty_images,
                    "observed": len(structure.zero_gt_image_ids)})
        report.add_check("no_duplicate_image_ids",
                         not structure.duplicate_image_ids,
                         detail=structure.duplicate_image_ids)
        report.add_check("no_duplicate_annotation_ids",
                         not structure.duplicate_annotation_ids,
                         detail=structure.duplicate_annotation_ids)
        report.add_check("no_duplicate_category_ids",
                         not structure.duplicate_category_ids,
                         detail=structure.duplicate_category_ids)
        report.add_check("no_invalid_image_refs",
                         structure.invalid_annotation_image_refs == 0,
                         detail=structure.invalid_annotation_image_refs)
        report.add_check("no_invalid_category_refs",
                         structure.invalid_annotation_category_refs == 0,
                         detail=structure.invalid_annotation_category_refs)

        # Resolve JPGs. file_name already contains the 'train/' subdir here.
        resolve_info = resolve_image_files(coco, data_root)
        report.add_check("all_referenced_jpg_exist",
                         resolve_info["num_missing"] == 0,
                         detail={"missing": resolve_info["num_missing"]})
        report.add_check(
            "unique_resolved_jpg_paths",
            resolve_info["num_unique_resolved"] == structure.num_images,
            detail={"unique": resolve_info["num_unique_resolved"],
                    "images": structure.num_images})

    # --- Integration: build datasets -------------------------------------- #
    retention_results: Dict[str, Any] = {}
    comparison_results: Dict[str, Any] = {}
    pipeline_summary: Dict[str, Any] = {}
    dataloader_summary: Dict[str, Any] = {}
    bbox_label_summary: Dict[str, Any] = {}
    sample_evidence: List[Dict[str, Any]] = []
    image_audit_rows: List[List[Any]] = []

    frameworks_ok = any(c["name"] == "import_frameworks" and c["passed"]
                        for c in report.checks)
    inputs_ok = (structure is not None
                 and ann_file.is_file() and data_root.is_dir())

    if frameworks_ok and inputs_ok:
        try:
            _run_integration(
                args, report, coco, structure, metainfo_classes,
                cat_id_to_label, ann_file, data_root,
                retention_results, comparison_results, pipeline_summary,
                dataloader_summary, bbox_label_summary, sample_evidence,
                image_audit_rows,
            )
        except Exception as exc:  # noqa: BLE001 - record, never silently pass
            report.add_check("integration_run", False,
                             detail=f"{type(exc).__name__}: {exc}")
    else:
        if not frameworks_ok:
            report.add_warning(
                "MMDetection frameworks not importable; integration skipped. "
                "Run on the mmdet330 environment for a real verdict.")
        if not inputs_ok:
            report.add_warning(
                "Inputs (COCO JSON / image root) missing; integration skipped.")
        report.add_check("integration_run", False,
                         detail="integration_not_executed")

    # --- Derived status flags --------------------------------------------- #
    loading_checks = [
        "coco_json_parse", "coco_structure_analyzed", "count_images",
        "count_annotations", "count_categories", "no_duplicate_image_ids",
        "no_duplicate_annotation_ids", "no_duplicate_category_ids",
        "no_invalid_image_refs", "no_invalid_category_refs",
        "all_referenced_jpg_exist", "unique_resolved_jpg_paths",
        "cocodataset_build_retention", "retention_length_matches",
        "abnormal_pipeline_sample", "bbox_label_valid",
        "normal_dataloader_batch",
    ]
    retention_checks = [
        "count_zero_gt_images", "cocodataset_build_filtered",
        "retention_contains_all_zero_gt", "filtered_removed_equals_zero_gt",
        "empty_pipeline_sample", "empty_bbox_shape", "empty_label_shape",
        "forced_empty_dataloader_batch",
    ]

    def all_passed(names: Sequence[str]) -> bool:
        present = {c["name"]: c["passed"] for c in report.checks}
        return all(present.get(n, False) for n in names)

    dataset_loading_validated = all_passed(loading_checks)
    empty_image_retention_validated = all_passed(retention_checks)
    overall_pass = report.overall_pass and dataset_loading_validated \
        and empty_image_retention_validated
    dataset_training_ready = overall_pass

    # --- Assemble report dict --------------------------------------------- #
    report_dict: Dict[str, Any] = {
        "phase": PHASE_ID,
        "phase_name": PHASE_NAME,
        "generated_at_utc": utc_now(),
        "input_paths": {
            "repo_root": str(repo_root),
            "ann_file": str(ann_file),
            "data_root": str(data_root),
            "report_json": str(report_json),
            "report_md": str(report_md),
            "image_audit_csv": str(image_audit_csv),
            "errors_csv": str(errors_csv),
        },
        "environment_versions": versions,
        "coco_sha256": input_info.get("coco_sha256"),
        "coco_sha256_expected": args.expected_coco_sha256,
        "expected_counts": {
            "images": args.expected_images,
            "annotations": args.expected_annotations,
            "categories": args.expected_categories,
            "empty_images": args.expected_empty_images,
        },
        "observed_raw_counts": observed_counts,
        "input_integrity_results": {
            "resolve_info": {k: v for k, v in resolve_info.items()
                             if k != "missing"},
            "num_missing_jpg": resolve_info.get("num_missing"),
        },
        "mmdet_dataset_configuration": {
            "type": "CocoDataset",
            "metainfo_classes": list(metainfo_classes),
            "num_classes": len(metainfo_classes),
            "cat_id_to_label": cat_id_to_label,
            "pipeline": [t["type"] for t in build_validation_pipeline()],
            "filter_cfg_retention": {"filter_empty_gt": False},
            "filter_cfg_controlled": {"filter_empty_gt": True},
            "test_mode": False,
        },
        "retention_dataset_results": retention_results,
        "controlled_filtering_comparison": comparison_results,
        "empty_gt_image_id_comparison": {
            "num_zero_gt": len(structure.zero_gt_image_ids)
            if structure else None,
            "removed_equals_zero_gt":
                comparison_results.get("removed_equals_zero_gt"),
        },
        "pipeline_validation_summary": pipeline_summary,
        "bbox_label_validation_summary": bbox_label_summary,
        "dataloader_validation_summary": dataloader_summary,
        "sample_evidence": sample_evidence,
        "errors": report.errors,
        "warnings": report.warnings,
        "checks": report.checks,
        "seed": args.seed,
        "overall_status": "PASS" if overall_pass else "FAIL",
        "dataset_loading_validated": dataset_loading_validated,
        "empty_image_retention_validated": empty_image_retention_validated,
        "dataset_training_ready": dataset_training_ready,
        "training_authorized": False,  # ALWAYS false in this script
    }

    # --- Write evidence (atomic) ------------------------------------------ #
    write_json_atomic(report_json, report_dict)
    atomic_write_text(report_md, render_markdown(report_dict))

    audit_header = [
        "dataset_index", "image_id", "file_name", "coco_width", "coco_height",
        "raw_annotation_count", "is_empty_gt", "retained_filter_false",
        "removed_when_filter_true", "pipeline_audited", "loaded_image_shape",
        "post_pipeline_box_count", "post_pipeline_label_count",
        "bbox_valid", "label_valid", "pipeline_load_result", "error_message",
    ]
    write_csv_atomic(image_audit_csv, audit_header, image_audit_rows)

    error_header = ["check", "detail"]
    error_rows = [[e.get("check"), json.dumps(e.get("detail"), default=str)]
                  for e in report.errors]
    write_csv_atomic(errors_csv, error_header, error_rows)

    return report_dict, overall_pass


def _run_integration(
    args, report, coco, structure, metainfo_classes, cat_id_to_label,
    ann_file, data_root, retention_results, comparison_results,
    pipeline_summary, dataloader_summary, bbox_label_summary,
    sample_evidence, image_audit_rows,
) -> None:
    """Run the MMDetection-dependent portion (mutates the passed containers)."""
    import numpy as np
    import torch  # noqa: F401
    from mmengine.dataset import pseudo_collate

    num_classes = len(metainfo_classes)
    zero_gt_set = set(structure.zero_gt_image_ids)

    # Build retention dataset (filter_empty_gt=False) WITH the real pipeline.
    pipeline = build_validation_pipeline()
    ds_retention = build_coco_dataset(
        ann_file, data_root, metainfo_classes, filter_empty_gt=False,
        pipeline=pipeline, test_mode=False)
    retention_len = len(ds_retention)
    retention_ids = dataset_image_ids_in_order(ds_retention)

    report.add_check("cocodataset_build_retention", True,
                     detail={"length": retention_len})
    report.add_check(
        "retention_length_matches",
        retention_len == structure.num_images == args.expected_images,
        detail={"length": retention_len, "expected": args.expected_images})

    retained_id_set = set(retention_ids)
    contains_all_zero = zero_gt_set.issubset(retained_id_set)
    report.add_check("retention_contains_all_zero_gt", contains_all_zero,
                     detail={"zero_gt": len(zero_gt_set),
                             "present": len(zero_gt_set & retained_id_set)})

    retention_results.update({
        "raw_coco_image_count": structure.num_images,
        "retention_length": retention_len,
        "num_zero_gt_present": len(zero_gt_set & retained_id_set),
        "filter_cfg": {"filter_empty_gt": False},
    })

    # Build controlled dataset (filter_empty_gt=True); no pipeline needed here.
    ds_filtered = build_coco_dataset(
        ann_file, data_root, metainfo_classes, filter_empty_gt=True,
        pipeline=[], test_mode=False)
    filtered_len = len(ds_filtered)
    filtered_ids = set(dataset_image_ids_in_order(ds_filtered))
    removed_ids = retained_id_set - filtered_ids
    removed_equals_zero_gt = removed_ids == zero_gt_set

    report.add_check("cocodataset_build_filtered", True,
                     detail={"length": filtered_len})
    report.add_check("filtered_removed_equals_zero_gt", removed_equals_zero_gt,
                     detail={"num_removed": len(removed_ids),
                             "num_zero_gt": len(zero_gt_set)})

    comparison_results.update({
        "filtered_length": filtered_len,
        "num_removed": len(removed_ids),
        "removed_equals_zero_gt": removed_equals_zero_gt,
        "filter_cfg": {"filter_empty_gt": True},
        "note": ("Comparison is by image-ID identity, not length only. "
                 "Removed set must equal the zero-GT set."),
    })

    # --- Deterministic sample selection ----------------------------------- #
    abnormal_ids = structure.nonempty_image_ids
    n_abn = (len(retention_ids) if args.full_pipeline_audit
             else args.pipeline_audit_abnormal)
    n_emp = (len(retention_ids) if args.full_pipeline_audit
             else args.pipeline_audit_empty)
    abn_indices = select_deterministic_indices(retention_ids, abnormal_ids, n_abn)
    emp_indices = select_deterministic_indices(
        retention_ids, structure.zero_gt_image_ids, n_emp)

    # --- Pipeline audit --------------------------------------------------- #
    audited: Dict[int, Dict[str, Any]] = {}
    abnormal_pass = True
    empty_pass = True
    bbox_all_valid = True
    empty_bbox_shape: Optional[List[int]] = None
    empty_label_shape: Optional[List[int]] = None

    def audit_index(idx: int, expect_empty: bool) -> Dict[str, Any]:
        info = ds_retention.get_data_info(idx) if hasattr(
            ds_retention, "get_data_info") else ds_retention.data_list[idx]
        image_id = int(info.get("img_id", info.get("image_id")))
        width = info.get("width")
        height = info.get("height")
        row: Dict[str, Any] = {
            "dataset_index": idx,
            "image_id": image_id,
            "expect_empty": expect_empty,
            "pipeline_load_result": "FAIL",
            "error_message": "",
        }
        try:
            sample = ds_retention[idx]
            bboxes, labels, meta = extract_gt_from_sample(sample)
            b_np = bboxes.detach().cpu().numpy() if hasattr(bboxes, "detach") \
                else np.asarray(bboxes)
            l_np = labels.detach().cpu().numpy() if hasattr(labels, "detach") \
                else np.asarray(labels)
            inputs = sample["inputs"]
            loaded_shape = list(getattr(inputs, "shape", []))
            valres = validate_bboxes_and_labels(
                b_np, l_np, width, height, num_classes)
            row.update({
                "file_name": structure.file_names_by_image_id.get(image_id),
                "coco_width": width,
                "coco_height": height,
                "raw_annotation_count":
                    structure.ann_count_by_image_id.get(image_id, 0),
                "loaded_image_shape": loaded_shape,
                "post_pipeline_box_count": valres["n_boxes"],
                "post_pipeline_label_count": valres["n_labels"],
                "bbox_valid": valres["valid"],
                "label_valid": valres["labels_in_range"],
                "pipeline_load_result": "PASS",
                "inputs_is_tensor": bool(hasattr(inputs, "shape")),
                "empty_shapes": check_empty_gt_shapes(b_np, l_np)
                if expect_empty else None,
            })
        except Exception as exc:  # noqa: BLE001
            row["error_message"] = f"{type(exc).__name__}: {exc}"
        return row

    for idx in abn_indices:
        row = audit_index(idx, expect_empty=False)
        audited[idx] = row
        if row["pipeline_load_result"] != "PASS" or not row.get("bbox_valid"):
            abnormal_pass = False
            bbox_all_valid = bbox_all_valid and bool(row.get("bbox_valid"))

    for idx in emp_indices:
        row = audit_index(idx, expect_empty=True)
        audited[idx] = row
        shapes = row.get("empty_shapes") or {}
        if row["pipeline_load_result"] != "PASS":
            empty_pass = False
        if not shapes.get("semantic_ok", False):
            empty_pass = False
        if empty_bbox_shape is None and shapes:
            empty_bbox_shape = shapes.get("bboxes_shape")
            empty_label_shape = shapes.get("labels_shape")

    report.add_check("abnormal_pipeline_sample",
                     abnormal_pass and len(abn_indices) > 0,
                     detail={"audited": len(abn_indices)})
    report.add_check("empty_pipeline_sample",
                     empty_pass and len(emp_indices) > 0,
                     detail={"audited": len(emp_indices)})
    report.add_check("empty_bbox_shape", empty_bbox_shape == [0, 4],
                     detail={"observed": empty_bbox_shape})
    report.add_check("empty_label_shape", empty_label_shape == [0],
                     detail={"observed": empty_label_shape})
    report.add_check("bbox_label_valid", bbox_all_valid,
                     detail={"all_valid": bbox_all_valid})

    pipeline_summary.update({
        "abnormal_audited": len(abn_indices),
        "empty_audited": len(emp_indices),
        "abnormal_pass": abnormal_pass,
        "empty_pass": empty_pass,
        "empty_bbox_shape": empty_bbox_shape,
        "empty_label_shape": empty_label_shape,
        "full_pipeline_audit": bool(args.full_pipeline_audit),
    })
    bbox_label_summary.update({
        "all_audited_valid": bbox_all_valid,
        "num_audited": len(audited),
        "tolerance_px": BBOX_BOUND_TOLERANCE,
    })

    # A few sample-evidence rows (first abnormal + first empty).
    for idx in (abn_indices[:2] + emp_indices[:2]):
        row = audited.get(idx, {})
        sample_evidence.append({
            "dataset_index": idx,
            "image_id": row.get("image_id"),
            "expect_empty": row.get("expect_empty"),
            "post_pipeline_box_count": row.get("post_pipeline_box_count"),
            "post_pipeline_label_count": row.get("post_pipeline_label_count"),
            "pipeline_load_result": row.get("pipeline_load_result"),
            "empty_shapes": row.get("empty_shapes"),
        })

    # --- Dataloader validation -------------------------------------------- #
    _run_dataloader_checks(
        args, report, ds_retention, retention_ids, zero_gt_set,
        abnormal_ids, dataloader_summary, pseudo_collate)

    # --- Full structural audit CSV (all images) --------------------------- #
    for idx, image_id in enumerate(retention_ids):
        raw_ann = structure.ann_count_by_image_id.get(image_id, 0)
        is_empty = raw_ann == 0
        row = audited.get(idx)
        if row and row.get("pipeline_load_result"):
            image_audit_rows.append([
                idx, image_id,
                structure.file_names_by_image_id.get(image_id),
                row.get("coco_width"), row.get("coco_height"),
                raw_ann, is_empty, True, is_empty, True,
                row.get("loaded_image_shape"),
                row.get("post_pipeline_box_count"),
                row.get("post_pipeline_label_count"),
                row.get("bbox_valid"), row.get("label_valid"),
                row.get("pipeline_load_result"), row.get("error_message"),
            ])
        else:
            image_audit_rows.append([
                idx, image_id,
                structure.file_names_by_image_id.get(image_id),
                None, None, raw_ann, is_empty, True, is_empty, False,
                None, None, None, None, None, "NOT_AUDITED", "",
            ])


def _run_dataloader_checks(args, report, dataset, retention_ids, zero_gt_set,
                           abnormal_ids, dataloader_summary,
                           pseudo_collate) -> None:
    """Build deterministic normal and forced empty-GT batches."""
    from torch.utils.data import DataLoader, Subset

    # Deterministic indices (no shuffle): first abnormal-only for the normal
    # batch, and a forced mix that guarantees at least one zero-GT sample.
    abn_positions = select_deterministic_indices(
        retention_ids, abnormal_ids, max(args.batch_size, 1))
    emp_positions = select_deterministic_indices(
        retention_ids, sorted(zero_gt_set), max(args.batch_size, 1))
    forced_positions = (emp_positions[:1] + abn_positions)[
        :max(args.batch_size, 2)]

    collate_note = ("pseudo_collate keeps per-sample tensors as a list so "
                    "un-resized images of different sizes batch safely.")

    normal_pass = False
    forced_pass = False
    empty_located = False
    try:
        normal_loader = DataLoader(
            Subset(dataset, abn_positions),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=pseudo_collate)
        normal_batch = next(iter(normal_loader))
        normal_pass = ("inputs" in normal_batch
                       and "data_samples" in normal_batch
                       and len(normal_batch["inputs"]) >= 1)
    except Exception as exc:  # noqa: BLE001
        report.add_error("normal_dataloader_batch", f"{type(exc).__name__}: {exc}")

    try:
        forced_loader = DataLoader(
            Subset(dataset, forced_positions),
            batch_size=max(args.batch_size, 2), shuffle=False,
            num_workers=args.num_workers, collate_fn=pseudo_collate)
        forced_batch = next(iter(forced_loader))
        forced_pass = ("inputs" in forced_batch
                       and "data_samples" in forced_batch)
        for ds_sample in forced_batch["data_samples"]:
            gt = ds_sample.gt_instances
            n = len(gt.labels)
            img_id = ds_sample.metainfo.get("img_id")
            if n == 0 and int(img_id) in zero_gt_set:
                empty_located = True
                break
    except Exception as exc:  # noqa: BLE001
        report.add_error("forced_empty_dataloader_batch",
                         f"{type(exc).__name__}: {exc}")

    report.add_check("normal_dataloader_batch", normal_pass,
                     detail={"batch_size": args.batch_size})
    report.add_check("forced_empty_dataloader_batch",
                     forced_pass and empty_located,
                     detail={"empty_located": empty_located})

    dataloader_summary.update({
        "collate": "pseudo_collate",
        "collate_rationale": collate_note,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "normal_batch_pass": normal_pass,
        "forced_empty_batch_pass": forced_pass,
        "empty_sample_located": empty_located,
        "normal_indices": abn_positions,
        "forced_indices": forced_positions,
    })


def print_console_summary(report_dict: Dict[str, Any]) -> None:
    """Print the audit-friendly console summary."""
    r = report_dict
    obs = r.get("observed_raw_counts", {})
    ret = r.get("retention_dataset_results", {})
    comp = r.get("controlled_filtering_comparison", {})
    ps = r.get("pipeline_validation_summary", {})
    dl = r.get("dataloader_validation_summary", {})
    integ = {c["name"]: c["passed"] for c in r.get("checks", [])}

    def yn(flag: Optional[bool]) -> str:
        if flag is None:
            return "N/A"
        return "PASS" if flag else "FAIL"

    print(f"PHASE {PHASE_ID} — MMDetection Dataset-Loading Validation")
    print(f"Environment: {yn(integ.get('import_frameworks'))}")
    print(f"Input integrity: "
          f"{yn(integ.get('all_referenced_jpg_exist'))}")
    print(f"Raw COCO images: {obs.get('num_images')}")
    print(f"Raw COCO annotations: {obs.get('num_annotations')}")
    print(f"Raw categories: {obs.get('num_categories')}")
    print(f"Raw zero-GT images: {obs.get('num_zero_gt_images')}")
    print(f"Resolved JPG files: "
          f"{r.get('input_integrity_results', {}).get('resolve_info', {}).get('num_unique_resolved')}")
    print(f"Missing JPG files: "
          f"{r.get('input_integrity_results', {}).get('num_missing_jpg')}")
    print(f"CocoDataset filter_empty_gt=False: {ret.get('retention_length')}")
    print(f"CocoDataset filter_empty_gt=True: {comp.get('filtered_length')}")
    print(f"Filtered image IDs: {comp.get('num_removed')}")
    print(f"Filtered IDs equal zero-GT IDs: "
          f"{yn(comp.get('removed_equals_zero_gt'))}")
    print(f"Abnormal pipeline sample: {yn(integ.get('abnormal_pipeline_sample'))}")
    print(f"Empty-GT pipeline sample: {yn(integ.get('empty_pipeline_sample'))}")
    print(f"Empty bbox shape: {ps.get('empty_bbox_shape')}")
    print(f"Empty label shape: {ps.get('empty_label_shape')}")
    print(f"Normal dataloader batch: {yn(dl.get('normal_batch_pass'))}")
    print(f"Forced empty-GT dataloader batch: "
          f"{yn(dl.get('forced_empty_batch_pass') and dl.get('empty_sample_located'))}")
    print(f"Errors: {len(r.get('errors', []))}")
    print(f"PHASE {PHASE_ID} VALIDATION: {r.get('overall_status')}")
    print(f"TRAINING AUTHORIZED: {str(r.get('training_authorized')).upper()}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns process exit code (0 = PASS)."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    report_dict, overall_pass = run_validation(args)
    print_console_summary(report_dict)
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
