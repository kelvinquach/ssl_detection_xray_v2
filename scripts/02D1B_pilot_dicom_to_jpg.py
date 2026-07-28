#!/usr/bin/env python3
r"""Phase 2D.1B-Pilot - Representative DICOM-to-JPG Pilot (orchestrator).

Official pipeline:
    DICOM metadata-aware, standard-aligned reference representation pipeline
    (reference representation pipeline; NOT a novel preprocessing algorithm).

This script runs a representative, coverage-first, failure-seeking pilot on the
locked 4,894-image controlled scope. It inspects headers for ALL 4,894 images
but decodes pixels ONLY for the deterministically selected pilot images. It
emits paired JPEG q95/q100 candidates plus a lossless uint8 reference PNG and a
large body of machine-readable technical evidence for GPT/researcher review.

HARD SCOPE (enforced in code + tests):
    * No full conversion of 4,894 pixels; no data/processed/images_jpg/train.
    * No coco_master_jpg.json; no split; no training/inference/pseudo-labels.
    * No final JPEG quality selection; final_quality stays null.
    * No modification of coco_master.json / canonical tables / protocol YAML /
      Phase 2D.1A artifacts.
    * No readiness flag is ever set true; training_authorized stays false.
    * phase_status is never "PASS"; the best structural outcome is
      OPEN_REVIEW_REQUIRED with structural_dod_candidate=true.

The pure transformation + validation logic lives in
``src/utils/dicom_jpg_protocol.py`` (importable; digit-leading script names are
not importable, and the guardrail tests need those pure functions).

Usage (Windows CMD):
    set VINBIGDATA_DICOM_ROOT=D:\ssl_detection_xray\data\raw\vinbigdata\dicom_subset
    python scripts\02D1B_pilot_dicom_to_jpg.py
    python scripts\02D1B_pilot_dicom_to_jpg.py --jpeg2000-decoder pylibjpeg
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import dicom_jpg_protocol as P  # noqa: E402

LOG = logging.getLogger("phase2D1B_pilot")

# Canonical input paths (never use upload-suffixed names).
PROTOCOL_YAML = REPO_ROOT / "configs" / "protocol" / "phase2D1_jpg_representation.yaml"
COCO_MASTER = REPO_ROOT / "data" / "processed" / "coco" / "coco_master.json"
CANONICAL_BBOX = REPO_ROOT / "data" / "processed" / "canonical" / "canonical_bbox_table.csv"
CANONICAL_CLASS = REPO_ROOT / "data" / "processed" / "canonical" / "canonical_class_mapping.csv"
PHASE2A_META = REPO_ROOT / "reports" / "phase2A_image_metadata.csv"
PHASE2D_VALID = REPO_ROOT / "reports" / "phase2D_coco_master_validation.json"

REPORTS_DIR = REPO_ROOT / "reports"
PLOTS_DIR = REPO_ROOT / "plots" / "phase2D1B_pilot"
MAPPING_DIR = REPO_ROOT / "data" / "processed" / "image_mapping"
PILOT_OUT_DIR = REPO_ROOT / "data" / "processed" / "images_jpg_pilot"

# Forbidden full-conversion artifacts (must be absent before pixel decoding).
FORBIDDEN_ARTIFACTS = (
    REPO_ROOT / "data" / "processed" / "images_jpg" / "train",
    REPO_ROOT / "data" / "processed" / "coco" / "coco_master_jpg.json",
    REPO_ROOT / "scripts" / "02D1B_full_dicom_to_jpg.py",
)

REQUIRED_DEPS = ("numpy", "pandas", "pydicom", "PIL", "yaml", "skimage", "matplotlib")
OPTIONAL_DEPS = ("pylibjpeg", "openjpeg", "gdcm")
FORBIDDEN_IMPORTS = ("mmdet", "mmcv", "mmengine", "torch", "torchvision", "detectron2")

# Bounded, deterministic visual subset size cap (visual evidence only).
VISUAL_SUBSET_MAX = 40


# =========================================================================== #
# Dependency preflight (never auto-install; report clearly)                     #
# =========================================================================== #
def dependency_report() -> Dict[str, Any]:
    import importlib

    report: Dict[str, Any] = {"required": {}, "optional": {}, "forbidden_not_used": {}}
    for name in REQUIRED_DEPS:
        try:
            mod = importlib.import_module(name)
            report["required"][name] = {"import_ok": True,
                                        "version": getattr(mod, "__version__", None)}
        except Exception as exc:
            report["required"][name] = {"import_ok": False, "error": repr(exc)}
    for name in OPTIONAL_DEPS:
        try:
            mod = importlib.import_module(name)
            report["optional"][name] = {"import_ok": True,
                                        "version": getattr(mod, "__version__", None)}
        except Exception as exc:
            report["optional"][name] = {"import_ok": False, "error": repr(exc)}
    for name in FORBIDDEN_IMPORTS:
        report["forbidden_not_used"][name] = {"used_by_pilot": False}
    return report


# =========================================================================== #
# Strict JSON / CSV + atomic writes                                            #
# =========================================================================== #
def strict_json_dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
            writer.writeheader()
            for r in rows:
                writer.writerow({k: _csv_cell(r.get(k, "")) for k in fieldnames})
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(x) for x in value)
    return value


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# =========================================================================== #
# Stage 1: protocol_preflight                                                  #
# =========================================================================== #
def stage_protocol_preflight() -> Dict[str, Any]:
    import yaml

    with open(PROTOCOL_YAML, "r", encoding="utf-8") as fh:
        protocol = yaml.safe_load(fh)
    evidence = P.validate_protocol(protocol)
    LOG.info("protocol preflight PASS: version=%s fingerprint=%s",
             evidence["values"]["protocol_version"], evidence["protocol_sha256"])
    return {"protocol": protocol, "evidence": evidence}


# =========================================================================== #
# Stage 2: input_crosscheck                                                    #
# =========================================================================== #
def stage_input_crosscheck() -> Dict[str, Any]:
    with open(COCO_MASTER, "r", encoding="utf-8") as fh:
        coco = json.load(fh)

    coco_sha = file_sha256(COCO_MASTER)
    with open(PHASE2D_VALID, "r", encoding="utf-8") as fh:
        phase2d = json.load(fh)
    expected_sha = phase2d.get("output_sha256")
    if expected_sha != coco_sha or coco_sha != P.EXPECTED_COCO_MASTER_SHA256:
        raise P.CocoMasterDriftError(
            f"coco_master_drift_detected: file={coco_sha} phase2d={expected_sha} "
            f"locked={P.EXPECTED_COCO_MASTER_SHA256}"
        )

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    counts = {
        "images": len(images),
        "annotations": len(annotations),
        "categories": len(categories),
        "abnormal_images": sum(1 for im in images if not im.get("is_negative")),
        "no_finding_images": sum(1 for im in images if im.get("is_negative")),
    }
    for key, expected in P.LOCKED_INPUT_COUNTS.items():
        if key in counts and counts[key] != expected:
            raise P.UnsupportedInputError(f"count mismatch {key}: {counts[key]} != {expected}")

    neg_image_ids = {im["id"] for im in images if im.get("is_negative")}
    if sum(1 for a in annotations if a["image_id"] in neg_image_ids) != 0:
        raise P.UnsupportedInputError("no_finding_annotations != 0")

    if len({im["id"] for im in images}) != len(images):
        raise P.UnsupportedInputError("duplicate COCO image id")
    if len({im["original_image_id"] for im in images}) != len(images):
        raise P.UnsupportedInputError("duplicate original_image_id")
    file_names = [im["file_name"] for im in images]
    if len(set(file_names)) != len(file_names):
        raise P.UnsupportedInputError("duplicate COCO file_name")
    for fn in file_names:
        if Path(fn).is_absolute() or fn.startswith("/") or fn.startswith("\\"):
            raise P.UnsupportedInputError(f"absolute file_name: {fn}")
        if not fn.startswith("train/") or not fn.endswith(".dicom"):
            raise P.UnsupportedInputError(f"unexpected file_name: {fn}")

    cat_ids = sorted(c["id"] for c in categories)
    if cat_ids != list(range(1, P.NUM_ABNORMAL_CLASSES + 1)):
        raise P.UnsupportedInputError(f"category ids not contiguous 1..14: {cat_ids}")
    canon_ids = sorted(c["canonical_class_id"] for c in categories)
    if canon_ids != list(range(0, P.NUM_ABNORMAL_CLASSES)):
        raise P.UnsupportedInputError(f"canonical ids not contiguous 0..13: {canon_ids}")
    class_names: Dict[int, str] = {}
    for c in categories:
        if c["id"] != c["canonical_class_id"] + 1:
            raise P.UnsupportedInputError(f"category {c['id']} canonical mapping mismatch")
        if c["name"].strip().lower() in ("no finding", "background"):
            raise P.UnsupportedInputError(f"forbidden category: {c['name']}")
        class_names[c["canonical_class_id"]] = c["name"]

    canon_rows = invalid = boundary_bad = 0
    with open(CANONICAL_BBOX, "r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            canon_rows += 1
            if str(r.get("is_valid_bbox")).strip().lower() != "true":
                invalid += 1
            if str(r.get("boundary_valid")).strip().lower() != "true":
                boundary_bad += 1
    if canon_rows != P.LOCKED_INPUT_COUNTS["annotations"]:
        raise P.UnsupportedInputError(f"canonical bbox rows {canon_rows} != 36096")
    if invalid or boundary_bad:
        raise P.UnsupportedInputError(
            f"canonical bbox validity failure invalid={invalid} boundary={boundary_bad}"
        )

    # Rare-class ranking basis from the canonical class mapping (image_count).
    class_image_count: Dict[int, int] = {}
    with open(CANONICAL_CLASS, "r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            class_image_count[int(r["canonical_class_id"])] = int(r["image_count"])

    LOG.info("input cross-check PASS: %s", counts)
    return {
        "coco": coco, "counts": counts, "coco_sha256": coco_sha,
        "canonical_bbox_rows": canon_rows, "class_names": class_names,
        "class_image_count": class_image_count,
    }


# =========================================================================== #
# Stage 3: decoder_preflight                                                   #
# =========================================================================== #
def stage_decoder_preflight(jpeg2000_decoder: str) -> Dict[str, Any]:
    import importlib

    env: Dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "utc": utc_now(),
        "jpeg2000_decoder_requested": jpeg2000_decoder,
    }
    for name in ("numpy", "pydicom", "PIL", "skimage", "yaml", "matplotlib",
                 "pandas", "pylibjpeg"):
        try:
            mod = importlib.import_module(name)
            env[f"{name}_version"] = getattr(mod, "__version__", None)
        except Exception as exc:
            env[f"{name}_version"] = None
            env[f"{name}_error"] = repr(exc)
    try:
        import PIL
        env["pillow_features_jpg"] = True
    except Exception:
        env["pillow_features_jpg"] = None
    try:
        import pydicom
        env["pydicom_pixel_handlers"] = [
            getattr(h, "__name__", str(h))
            for h in getattr(pydicom.config, "pixel_data_handlers", [])
        ]
    except Exception as exc:
        env["pydicom_pixel_handlers_error"] = repr(exc)
    env["jpeg2000_backend_available"] = P.jpeg2000_backend_available(jpeg2000_decoder)
    return env


# =========================================================================== #
# Stage 4: header_inventory (headers only; NO pixel_array)                      #
# =========================================================================== #
def read_header(path: Path) -> Dict[str, Any]:
    import pydicom

    ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
    wc = ds.get("WindowCenter", None)
    ww = ds.get("WindowWidth", None)

    def as_list(x):
        if x is None:
            return []
        if isinstance(x, (list, tuple)) or "MultiValue" in type(x).__name__:
            return list(x)
        return [x]

    wc_list = as_list(wc)
    ww_list = as_list(ww)
    nframes_raw = ds.get("NumberOfFrames", None)
    return {
        "SOPClassUID": str(ds.get("SOPClassUID", "") or "") or "ABSENT",
        "Modality": str(ds.get("Modality", "") or "") or "ABSENT",
        "Rows": int(ds.Rows),
        "Columns": int(ds.Columns),
        "PhotometricInterpretation": str(ds.get("PhotometricInterpretation", "") or ""),
        "TransferSyntaxUID": str(getattr(ds.file_meta, "TransferSyntaxUID", "") or ""),
        "BitsAllocated": int(ds.get("BitsAllocated", 0) or 0),
        "BitsStored": int(ds.get("BitsStored", 0) or 0),
        "HighBit": int(ds.get("HighBit", -1)) if ds.get("HighBit", None) is not None else -1,
        "PixelRepresentation": int(ds.get("PixelRepresentation", 0) or 0),
        "SamplesPerPixel": int(ds.get("SamplesPerPixel", 1) or 1),
        "NumberOfFrames_raw": None if nframes_raw is None else int(nframes_raw),
        "NumberOfFrames_effective": 1 if nframes_raw is None else int(nframes_raw),
        "RescaleSlope": ds.get("RescaleSlope", None),
        "RescaleIntercept": ds.get("RescaleIntercept", None),
        "rescale_slope_present": "RescaleSlope" in ds,
        "rescale_intercept_present": "RescaleIntercept" in ds,
        "modality_lut_present": "ModalityLUTSequence" in ds,
        "modality_lut_count": len(ds.ModalityLUTSequence) if "ModalityLUTSequence" in ds else 0,
        "voi_lut_present": "VOILUTSequence" in ds,
        "voi_lut_count": len(ds.VOILUTSequence) if "VOILUTSequence" in ds else 0,
        "WindowCenter_all": ";".join(str(x) for x in wc_list),
        "WindowWidth_all": ";".join(str(x) for x in ww_list),
        "window_center_count": len(wc_list),
        "window_width_count": len(ww_list),
        "window_is_multivalued": len(wc_list) > 1 or len(ww_list) > 1,
        "VOILUTFunction": str(ds.get("VOILUTFunction", "") or "") or "ABSENT",
        "PresentationLUTShape": str(ds.get("PresentationLUTShape", "") or "") or "ABSENT",
        "presentation_lut_sequence_present": "PresentationLUTSequence" in ds,
        "presentation_lut_sequence_count": len(ds.PresentationLUTSequence) if "PresentationLUTSequence" in ds else 0,
        "PixelPaddingValue": ds.get("PixelPaddingValue", None),
        "PixelPaddingRangeLimit": ds.get("PixelPaddingRangeLimit", None),
        "pixel_padding_value_present": "PixelPaddingValue" in ds,
        "pixel_padding_range_present": "PixelPaddingRangeLimit" in ds,
    }


class StructuralHeaderError(P.UnsupportedInputError):
    """Structural header failure (dimension/bits/photometric) -> BLOCKED."""


def validate_header_structural(header, coco_w, coco_h, meta_w, meta_h) -> List[str]:
    """Raise StructuralHeaderError on any hard structural failure (no skipping).

    Returns a list of non-fatal warnings to record in the errors CSV.
    """
    warnings: List[str] = []
    if header["SamplesPerPixel"] != 1:
        raise StructuralHeaderError("SamplesPerPixel != 1")
    if header["PhotometricInterpretation"] not in P.ALLOWED_PHOTOMETRIC:
        raise StructuralHeaderError(f"PhotometricInterpretation {header['PhotometricInterpretation']}")
    if header["NumberOfFrames_effective"] != 1:
        raise StructuralHeaderError("NumberOfFrames_effective != 1")
    for k in ("Rows", "Columns", "BitsAllocated", "BitsStored"):
        if header[k] <= 0:
            raise StructuralHeaderError(f"{k} <= 0")
    if header["BitsStored"] > header["BitsAllocated"]:
        raise StructuralHeaderError("BitsStored > BitsAllocated")
    if not (0 <= header["HighBit"] < header["BitsAllocated"]):
        raise StructuralHeaderError("HighBit out of range")
    if header["PixelRepresentation"] not in (0, 1):
        raise StructuralHeaderError("PixelRepresentation not in {0,1}")
    if not (header["Rows"] == coco_h == meta_h and header["Columns"] == coco_w == meta_w):
        raise StructuralHeaderError(
            f"dimension mismatch DICOM({header['Rows']}x{header['Columns']}) "
            f"COCO({coco_h}x{coco_w}) meta({meta_h}x{meta_w})"
        )
    if header["HighBit"] != header["BitsStored"] - 1:
        warnings.append("high_bit_not_bitsstored_minus_1")
    return warnings


def header_transform_preflight(header: Dict[str, Any]) -> None:
    """Header-only transform preflight. Raises to BLOCK before any pixel decode.

    Detects (without pixels): modality-branch incompleteness, presentation gaps
    / conflicts / LUT sequence, and invalid (present-but-broken) windows.
    """
    # Modality branch (incomplete rescale -> ModalityBranchError -> structural).
    P.modality_branch_name(header["modality_lut_present"],
                           header["rescale_slope_present"],
                           header["rescale_intercept_present"])
    # Presentation gap/conflict -> ProtocolGapError.
    dec = P.presentation_polarity_decision(
        header["PhotometricInterpretation"],
        None if header["PresentationLUTShape"] == "ABSENT" else header["PresentationLUTShape"],
        header["presentation_lut_sequence_present"],
    )
    P.require_inversion_count(dec)
    # Window: present-but-invalid AND no VOI LUT -> ProtocolGapError. Width is
    # validated against the VOILUTFunction (LINEAR needs >=1; LINEAR_EXACT /
    # SIGMOID need >0) BEFORE any pixel decode.
    if not header["voi_lut_present"]:
        wdec = P.classify_window(header["WindowCenter_all"], header["WindowWidth_all"])
        if wdec.state == "invalid":
            P.require_valid_window(wdec)  # raises ProtocolGapError
        if wdec.state == "valid":
            voi_func = None if header["VOILUTFunction"] == "ABSENT" else header["VOILUTFunction"]
            P.validate_window_width_for_function(wdec.width, voi_func)


# =========================================================================== #
# Deterministic selection (coverage-first greedy + SHA tie-break)               #
# =========================================================================== #
def deterministic_selection(image_features, mandatory_ids, negative_ids, all_features):
    selected: List[str] = []
    selected_set: set = set()
    covered: set = set()
    records: List[Dict[str, Any]] = []

    def take(image_id: str, why: List[str]) -> None:
        newly = image_features.get(image_id, set()) - covered
        covered.update(image_features.get(image_id, set()))
        selected.append(image_id)
        selected_set.add(image_id)
        records.append({
            "image_id": image_id,
            "selection_order": len(selected),
            "selected_for_features": sorted(why) if why else sorted(newly),
            "newly_covered_feature_count": len(newly),
            "tie_break_rank": P.tie_break_rank(image_id),
        })

    for image_id in sorted(set(mandatory_ids), key=P.tie_break_rank):
        if image_id not in selected_set:
            take(image_id, ["mandatory_extremum"])

    def greedy_until(stop_predicate) -> None:
        while not stop_predicate():
            best_id = None
            best_gain = 0
            for image_id, feats in image_features.items():
                if image_id in selected_set:
                    continue
                gain = len(feats - covered)
                if gain > best_gain or (
                    gain == best_gain and gain > 0 and best_id is not None
                    and P.tie_break_rank(image_id) < P.tie_break_rank(best_id)
                ):
                    best_gain = gain
                    best_id = image_id
            if best_id is None or best_gain == 0:
                break
            take(best_id, [])
            if len(selected) > P.MAX_PILOT_IMAGES:
                raise P.PilotScopeExplosionError("pilot_scope_explosion")

    greedy_until(lambda: all_features.issubset(covered))

    def num_negative() -> int:
        return sum(1 for i in selected if i in negative_ids)

    neg_pool = sorted(negative_ids, key=P.tie_break_rank)
    while num_negative() < P.MIN_PILOT_NO_FINDING:
        candidate = None
        best_gain = -1
        for image_id in neg_pool:
            if image_id in selected_set:
                continue
            gain = len(image_features.get(image_id, set()) - covered)
            if gain > best_gain:
                best_gain = gain
                candidate = image_id
        if candidate is None:
            break
        take(candidate, ["no_finding_minimum"])
        if len(selected) > P.MAX_PILOT_IMAGES:
            raise P.PilotScopeExplosionError("pilot_scope_explosion")

    if len(selected) < P.MIN_PILOT_IMAGES:
        for image_id in sorted(image_features.keys(), key=P.tie_break_rank):
            if len(selected) >= P.MIN_PILOT_IMAGES:
                break
            if image_id not in selected_set:
                take(image_id, ["fill_to_minimum"])
                if len(selected) > P.MAX_PILOT_IMAGES:
                    raise P.PilotScopeExplosionError("pilot_scope_explosion")

    greedy_until(lambda: all_features.issubset(covered))
    if len(selected) > P.MAX_PILOT_IMAGES:
        raise P.PilotScopeExplosionError("pilot_scope_explosion")
    return records


# =========================================================================== #
# Baseline validation-JSON status (pre-review maximum)                          #
# =========================================================================== #
def baseline_validation_status() -> Dict[str, Any]:
    return {
        "phase_id": P.PHASE_ID,
        "mentor_approval_status": "approved",
        "pipeline_implementation_authorized": True,
        "technical_validation_authorized": True,
        "pipeline_name": "dicom_metadata_aware_standard_aligned_reference_representation_pipeline",
        "pipeline_display_name": "DICOM metadata-aware, standard-aligned reference representation pipeline",
        "method_positioning": "reference_representation_pipeline",
        "technical_basis": "dicom_grayscale_transformation_sequence",
        "applied_research_precedent": "cheng_et_al_2024",
        "research_precedent_scope": "metadata_use_precedent_not_algorithm_replication",
        "novel_algorithm_claimed": False,
        "full_dicom_standard_conformance_claimed": False,
        "clinical_validation_claimed": False,
        "downstream_superiority_evaluated": False,
        "controlled_downstream_ablation_status": "pending_mentor_confirmation",
        "controlled_downstream_ablation_authorized": False,
        "controlled_downstream_ablation_required": None,
        "master_representation_channel_count": 1,
        "master_representation_mode": "L",
        "model_input_channel_adaptation_status": "deferred_to_dataset_loading_or_training_phase",
        "model_input_channel_adaptation_authorized_in_phase2D1B": False,
        "patient_space_orientation_independently_validated": False,
        "pixel_matrix_order_unchanged": True,
        "presentation_metadata_conflict_detected": False,
        "presentation_lut_sequence_detected": False,
        "protocol_gap_detected": False,
        "protocol_review_required": False,
        "visual_review_status": "PENDING_GPT",
        "critical_visual_failure": None,
        "phase_status": "OPEN_REVIEW_REQUIRED",
        "structural_dod_candidate": True,
        "gpt_review_status": "pending",
        "final_jpeg_quality": None,
        "final_quality_status": "pending_gpt_pilot_review",
        "full_conversion_authorized": False,
        "jpg_training_representation_ready": False,
        "coco_jpg_training_annotation_ready": False,
        "mmdetection_dataset_loading_ready": False,
        "empty_image_retention_ready": False,
        "dataset_training_ready": False,
        "training_authorized": False,
    }


DECISION_TEMPLATE_JSON = {
    "decision_status": "pending_gpt_and_researcher_review",
    "final_jpeg_quality": None,
    "selected_candidate": None,
    "full_conversion_authorized": False,
    "decision_rationale": None,
    "reviewed_evidence": [],
    "reviewer_notes": None,
}


def write_decision_templates(reports: Path) -> None:
    atomic_write_text(reports / "phase2D1B_pilot_decision_template.json",
                      strict_json_dumps(DECISION_TEMPLATE_JSON) + "\n")
    atomic_write_text(
        reports / "phase2D1B_pilot_decision_template.md",
        "# Phase 2D.1B-Pilot - Decision Template (PENDING)\n\n"
        "This template is intentionally empty of any decision.\n\n"
        "- decision_status: pending_gpt_and_researcher_review\n"
        "- final_jpeg_quality: null\n"
        "- selected_candidate: null\n"
        "- full_conversion_authorized: false\n\n"
        "The final JPEG quality is selected only after GPT review and explicit\n"
        "researcher confirmation, in a separate authorized step.\n",
    )


# =========================================================================== #
# Forbidden-artifact snapshot (Section 6.1)                                     #
# =========================================================================== #
def snapshot_forbidden_artifacts() -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {"preexisting_forbidden_artifact": False, "entries": []}
    for path in FORBIDDEN_ARTIFACTS:
        if path.exists():
            snapshot["preexisting_forbidden_artifact"] = True
            entry = {
                "path": str(path.relative_to(REPO_ROOT)),
                "type": "dir" if path.is_dir() else "file",
                "existed_before_current_run": True,
                "created_by_current_run": False,
                "modified_by_current_run": False,
            }
            try:
                st = path.stat()
                entry["size"] = st.st_size
                entry["mtime"] = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)\
                    .strftime("%Y-%m-%dT%H:%M:%SZ")
                if path.is_file():
                    entry["sha256"] = file_sha256(path)
            except Exception as exc:
                entry["stat_error"] = repr(exc)
            snapshot["entries"].append(entry)
    return snapshot


# =========================================================================== #
# CLI                                                                          #
# =========================================================================== #
@dataclass
class Args:
    dicom_root: Optional[str]
    jpeg2000_decoder: str
    overwrite: bool


def parse_args(argv: Optional[Sequence[str]] = None) -> Args:
    ap = argparse.ArgumentParser(
        description="Phase 2D.1B-Pilot representative DICOM-to-JPG pilot (pilot-only)."
    )
    ap.add_argument("--dicom-root", default=None,
                    help="Override DICOM root. Conflicting CLI/env paths hard-fail.")
    ap.add_argument("--jpeg2000-decoder", default="pylibjpeg",
                    choices=["pylibjpeg", "gdcm", "pillow"],
                    help="Explicit JPEG2000 backend. No silent fallback.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite prior pilot evidence only after staging validation.")
    ns = ap.parse_args(argv)
    return Args(ns.dicom_root, ns.jpeg2000_decoder, ns.overwrite)


# =========================================================================== #
# Orchestrator entry                                                            #
# =========================================================================== #
def run(args: Args) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    deps = dependency_report()
    missing = [n for n, v in deps["required"].items() if not v.get("import_ok")]
    if missing:
        LOG.error("Missing required dependencies (not auto-installed): %s", missing)
        return 3

    forbidden = snapshot_forbidden_artifacts()
    if forbidden["preexisting_forbidden_artifact"]:
        status = baseline_validation_status()
        status.update({"phase_status": "BLOCKED", "structural_dod_candidate": False,
                       "protocol_review_required": True,
                       "preexisting_forbidden_artifact": True,
                       "forbidden_artifact_snapshot": forbidden})
        # Failure-preserving: write to the separate blocked directory only; do
        # NOT overwrite any prior valid validation.json / promoted evidence.
        out = write_blocked_report(status, "preexisting_forbidden_artifact")
        LOG.error("Preexisting forbidden artifact(s) present; refusing to run. "
                  "Report: %s (prior valid evidence untouched).", out)
        return 4

    pre = stage_protocol_preflight()
    xcheck = stage_input_crosscheck()
    decoder_env = stage_decoder_preflight(args.jpeg2000_decoder)
    resolution = P.resolve_dicom_root(args.dicom_root, os.environ.get(P.ENV_VAR_NAME))
    LOG.info("DICOM root resolved via %s -> %s", resolution.source, resolution.root)
    return run_pipeline(args, pre, xcheck, decoder_env, resolution, forbidden)


def run_pipeline(args, pre, xcheck, decoder_env, resolution, forbidden) -> int:
    import numpy as np  # noqa: F401

    coco = xcheck["coco"]
    images = coco["images"]
    annotations = coco["annotations"]
    dicom_root = resolution.root

    meta: Dict[str, Dict[str, int]] = {}
    with open(PHASE2A_META, "r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            meta[r["image_id"]] = {"w": int(float(r["image_width"])),
                                   "h": int(float(r["image_height"]))}

    resolved: Dict[str, Path] = {}
    for im in images:
        oid = im["original_image_id"]
        path = P.safe_resolve_under_root(dicom_root, im["file_name"])
        if not path.exists():
            raise P.UnsupportedInputError(f"missing DICOM: {im['file_name']}")
        if path.stem != oid:
            raise P.UnsupportedInputError(f"stem mismatch: {path.stem} != {oid}")
        resolved[oid] = path
    LOG.info("resolved %d/%d controlled DICOM paths", len(resolved), len(images))

    # ---- Stage 4: header inventory for ALL 4894 (headers only) ----
    id_to_coco = {im["original_image_id"]: im for im in images}
    headers: Dict[str, Dict[str, Any]] = {}
    header_rows: List[Dict[str, Any]] = []
    warnings_rows: List[Dict[str, Any]] = []

    for oid, path in resolved.items():
        im = id_to_coco[oid]
        try:
            h = read_header(path)
            warns = validate_header_structural(h, im["width"], im["height"],
                                               meta[oid]["w"], meta[oid]["h"])
            header_transform_preflight(h)
        except StructuralHeaderError as exc:
            return emit_blocked(args, pre, xcheck, decoder_env, forbidden,
                                reason=str(exc), image_id=oid)
        except P.ModalityBranchError as exc:
            return emit_blocked(args, pre, xcheck, decoder_env, forbidden,
                                reason=f"modality:{exc}", image_id=oid)
        except P.ProtocolGapError as gap:
            return emit_blocked_protocol_review(
                args, pre, xcheck, decoder_env, forbidden, gap.reason, oid)
        h["image_id"] = oid
        h["coco_image_id"] = im["id"]
        h["canonical_image_id"] = im["canonical_image_id"]
        h["dicom_relative_path"] = im["file_name"]
        headers[oid] = h
        header_rows.append({k: v for k, v in h.items() if not k.startswith("_")})
        for w in warns:
            warnings_rows.append({"image_id": oid, "severity": "warning",
                                  "code": w, "detail": ""})

    if len(headers) != P.LOCKED_INPUT_COUNTS["images"]:
        raise P.UnsupportedInputError(
            f"header inventory incomplete: {len(headers)}/4894")
    LOG.info("header inventory complete: %d/4894", len(headers))

    # ---- Stage 5-6: strata + coverage universe ----
    image_features: Dict[str, set] = {}
    all_features: set = set()
    for oid, h in headers.items():
        feats = set(P.metadata_stratum_keys(h))
        image_features[oid] = feats
        all_features.update(feats)

    ann_by_oid: Dict[str, List[Dict[str, Any]]] = {}
    for a in annotations:
        ann_by_oid.setdefault(a["original_image_id"], []).append(a)
    negative_ids = {im["original_image_id"] for im in images if im.get("is_negative")}

    for oid in headers:
        for a in ann_by_oid.get(oid, []):
            image_features[oid].add(f"class={a['canonical_class_id']}")
            all_features.add(f"class={a['canonical_class_id']}")
        if oid in negative_ids:
            image_features[oid].add("scope=no_finding")
            all_features.add("scope=no_finding")

    mandatory = compute_extrema_ids(images, annotations, id_to_coco)
    for feat in ("dim_min_w", "dim_max_w", "dim_min_h", "dim_max_h", "px_min",
                 "px_max", "bbox_abs_min", "bbox_abs_max", "bbox_rel_min", "bbox_rel_max"):
        all_features.add(f"extremum={feat}")
    for feat, oid in mandatory.items():
        if oid in image_features:
            image_features[oid].add(f"extremum={feat}")

    # ---- Stage 7: deterministic selection ----
    records = deterministic_selection(image_features, list(mandatory.values()),
                                      negative_ids, all_features)
    selected_ids = [r["image_id"] for r in records]
    selected_set = set(selected_ids)
    if len(selected_ids) < P.MIN_PILOT_IMAGES:
        raise P.UnsupportedInputError("selection below minimum images")
    if sum(1 for i in selected_ids if i in negative_ids) < P.MIN_PILOT_NO_FINDING:
        raise P.UnsupportedInputError("selection below minimum No Finding")
    if len(selected_set) >= P.LOCKED_INPUT_COUNTS["images"]:
        raise P.AccidentalFullConversionError("selection equals full scope")

    covered = set().union(*(image_features[i] for i in selected_ids))
    coverage_result = P.validate_full_coverage(covered, all_features)
    LOG.info("selected %d pilot images; coverage OK (%d/%d features)",
             len(selected_ids), coverage_result["covered_total"],
             coverage_result["all_features_total"])

    # ---- Decoder enforcement BEFORE any pixel decode (blocker 5) ----
    needs_jpeg2000 = any(P.is_jpeg2000(headers[oid]["TransferSyntaxUID"])
                         for oid in selected_ids)
    if needs_jpeg2000:
        P.ensure_jpeg2000_backend(args.jpeg2000_decoder)  # hard fail if unavailable

    # ---- Stages 8-16 in staging, then true atomic promotion ----
    with tempfile.TemporaryDirectory(prefix="phase2D1B_stage_") as tmp:
        staging = Path(tmp)
        ctx = {
            "staging": staging, "args": args, "headers": headers,
            "header_rows": header_rows, "resolved": resolved,
            "selected_ids": selected_ids, "id_to_coco": id_to_coco,
            "ann_by_oid": ann_by_oid, "records": records,
            "negative_ids": negative_ids, "image_features": image_features,
            "all_features": all_features, "mandatory": mandatory,
            "coverage_result": coverage_result, "decoder_env": decoder_env,
            "protocol_evidence": pre["evidence"], "coco_sha256": xcheck["coco_sha256"],
            "forbidden": forbidden, "resolution": resolution,
            "class_names": xcheck["class_names"],
            "class_image_count": xcheck["class_image_count"],
            "warnings_rows": warnings_rows,
            "coco_ann_by_cannid": {str(a["canonical_ann_id"]): a for a in annotations},
            "cat_by_canonical": {c["canonical_class_id"]: c["id"]
                                 for c in coco["categories"]},
        }
        decoded_count = decode_transform_encode(ctx)
        if decoded_count >= P.LOCKED_INPUT_COUNTS["images"]:
            raise P.AccidentalFullConversionError("accidental_full_conversion_detected")
        compute_bbox_roi(ctx)
        generate_visual_evidence(ctx)
        write_all_evidence(ctx, decoded_count)
        validate_staging(staging, len(selected_ids), ctx["n_selected_annotations"],
                         ctx["visual_expectations"])
        promote_atomic(staging, args.overwrite)

    LOG.info("Phase 2D.1B-Pilot structural run complete: OPEN_REVIEW_REQUIRED "
             "(pending GPT review). No PASS, no final quality selected.")
    return 0


def compute_extrema_ids(images, annotations, id_to_coco) -> Dict[str, str]:
    def rank(oid: str) -> str:
        return P.tie_break_rank(oid)

    by_w = sorted(images, key=lambda im: (im["width"], rank(im["original_image_id"])))
    by_h = sorted(images, key=lambda im: (im["height"], rank(im["original_image_id"])))
    by_px = sorted(images, key=lambda im: (im["width"] * im["height"], rank(im["original_image_id"])))
    extrema = {
        "dim_min_w": by_w[0]["original_image_id"], "dim_max_w": by_w[-1]["original_image_id"],
        "dim_min_h": by_h[0]["original_image_id"], "dim_max_h": by_h[-1]["original_image_id"],
        "px_min": by_px[0]["original_image_id"], "px_max": by_px[-1]["original_image_id"],
    }
    abs_areas, rel_areas = [], []
    for a in annotations:
        oid = a["original_image_id"]
        im = id_to_coco[oid]
        area = float(a["area"])
        abs_areas.append((area, oid))
        rel_areas.append((area / (im["width"] * im["height"]), oid))
    abs_areas.sort(key=lambda t: (t[0], rank(t[1])))
    rel_areas.sort(key=lambda t: (t[0], rank(t[1])))
    extrema["bbox_abs_min"] = abs_areas[0][1]
    extrema["bbox_abs_max"] = abs_areas[-1][1]
    extrema["bbox_rel_min"] = rel_areas[0][1]
    extrema["bbox_rel_max"] = rel_areas[-1][1]
    return extrema


# =========================================================================== #
# Stage 9: DICOM -> uint8 transform (one selected image)                        #
# =========================================================================== #
def transform_pixels(header: Dict[str, Any], stored, ds) -> Dict[str, Any]:
    import numpy as np

    padding_mask = P.build_padding_mask(
        stored,
        int(header["PixelPaddingValue"]) if header["pixel_padding_value_present"] else None,
        int(header["PixelPaddingRangeLimit"]) if header["pixel_padding_range_present"] else None,
    )
    t_low, t_high = P.theoretical_stored_range(header["BitsStored"], header["PixelRepresentation"])

    branch = P.modality_branch_name(header["modality_lut_present"],
                                    header["rescale_slope_present"],
                                    header["rescale_intercept_present"])
    if branch == "rescale":
        mod_values, m_low, m_high = P.apply_rescale(
            stored, float(header["RescaleSlope"]), float(header["RescaleIntercept"]),
            t_low, t_high)
        rescale_state = "applied"
    elif branch == "modality_lut":
        mod_values = _apply_modality_lut(stored, ds).astype("float64")
        m_low, m_high = P.modality_lut_output_bounds(ds.ModalityLUTSequence[0].LUTData)
        rescale_state = "present_not_applied" if header["rescale_slope_present"] else "absent"
    else:
        mod_values = stored.astype("float64")
        m_low, m_high = float(t_low), float(t_high)
        rescale_state = "absent"

    voi_func = None if header["VOILUTFunction"] == "ABSENT" else header["VOILUTFunction"]
    theoretical_voi_low, theoretical_voi_high = 0.0, 1.0
    if header["voi_lut_present"]:
        applied = _apply_voi_lut(mod_values, ds)
        lut_desc = ds.VOILUTSequence[0].LUTDescriptor
        nbits = int(lut_desc[2])
        fraction = P.voi_lut_normalize(applied, nbits)
        voi_branch = "voi_lut"
    else:
        wdec = P.classify_window(header["WindowCenter_all"], header["WindowWidth_all"])
        if wdec.state == "valid":
            c, w = P.require_valid_window(wdec)
            fraction = P.apply_windowing(mod_values, c, w, voi_func)
            voi_branch = "windowing"
        elif wdec.state == "absent":
            fraction = P.fallback_modality_fraction(mod_values, m_low, m_high)
            voi_branch = "theoretical_fallback"
        else:
            P.require_valid_window(wdec)  # invalid window -> ProtocolGapError

    dec = P.presentation_polarity_decision(
        header["PhotometricInterpretation"],
        None if header["PresentationLUTShape"] == "ABSENT" else header["PresentationLUTShape"],
        header["presentation_lut_sequence_present"])
    inv = P.require_inversion_count(dec)  # never None -> 0
    fraction = P.apply_presentation(fraction, inv)

    uint8 = P.fraction_to_uint8(fraction, padding_mask)
    return {
        "uint8": uint8, "modality_branch": branch, "rescale_state": rescale_state,
        "voi_branch": voi_branch, "presentation_inversion_count": inv,
        "presentation_inversion_applied": bool(inv),
        "theoretical_modality_low": m_low, "theoretical_modality_high": m_high,
        "theoretical_voi_low": theoretical_voi_low, "theoretical_voi_high": theoretical_voi_high,
        "padding_pixel_count": int(padding_mask.sum()), "padding_present": bool(padding_mask.any()),
    }


def decode_pixels(ds, header, backend, *, pixel_array_fn=None, native_getter=None):
    """Decode pixels using an EXPLICIT backend for JPEG2000 (no silent fallback).

    Returns ``(array, used_backend)``. For uncompressed transfer syntaxes the
    native pydicom decoder is used. For JPEG2000 the requested plugin is passed
    explicitly into ``pydicom.pixels.pixel_array`` (pydicom>=3); on older
    pydicom the handler list is constrained to the single requested handler.

    ``pixel_array_fn`` / ``native_getter`` are injectable for unit testing so
    the plugin-passing behaviour can be verified without real DICOM.
    """
    plugin = P.resolve_decoding_plugin(header["TransferSyntaxUID"], backend)
    if plugin is None:
        arr = native_getter(ds) if native_getter is not None else ds.pixel_array
        return arr, "pydicom_native"
    if pixel_array_fn is None:
        try:
            from pydicom.pixels import pixel_array as pixel_array_fn  # pydicom>=3
        except Exception:
            pixel_array_fn = None
    if pixel_array_fn is not None:
        arr = pixel_array_fn(ds, decoding_plugin=plugin)
        return arr, f"{backend}:{plugin}"
    arr = _decode_with_single_handler(ds, backend)
    return arr, f"{backend}:single_handler"


def _decode_with_single_handler(ds, backend):
    """pydicom<3 fallback: restrict handlers to the requested one (no fallback)."""
    import importlib
    import pydicom

    handler_module = {"pylibjpeg": "pydicom.pixel_data_handlers.pylibjpeg_handler",
                      "gdcm": "pydicom.pixel_data_handlers.gdcm_handler",
                      "pillow": "pydicom.pixel_data_handlers.pillow_handler"}.get(backend)
    if handler_module is None:
        raise P.UnsupportedInputError(f"no single-handler mapping for {backend}")
    handler = importlib.import_module(handler_module)
    original = list(pydicom.config.pixel_data_handlers)
    try:
        pydicom.config.pixel_data_handlers = [handler]
        return ds.pixel_array
    finally:
        pydicom.config.pixel_data_handlers = original


def _apply_modality_lut(stored, ds):
    try:
        from pydicom.pixels import apply_modality_lut
    except Exception:
        from pydicom.pixel_data_handlers.util import apply_modality_lut
    return apply_modality_lut(stored, ds)


def _apply_voi_lut(values, ds):
    import numpy as np
    try:
        from pydicom.pixels import apply_voi_lut
    except Exception:
        from pydicom.pixel_data_handlers.util import apply_voi_lut
    return apply_voi_lut(np.asarray(values).astype("int64"), ds)


# =========================================================================== #
# Stage 8/10/11: decode selected -> ref PNG + paired JPEG + whole-image metrics #
# =========================================================================== #
def decode_transform_encode(ctx: Dict[str, Any]) -> int:
    import numpy as np
    import pydicom
    from PIL import Image

    staging: Path = ctx["staging"]
    selected_ids: List[str] = ctx["selected_ids"]
    headers = ctx["headers"]
    resolved = ctx["resolved"]
    id_to_coco = ctx["id_to_coco"]
    args = ctx["args"]
    selected_set = set(selected_ids)

    ref_dir = staging / "images_jpg_pilot" / "reference_uint8" / "train"
    q95_dir = staging / "images_jpg_pilot" / "q95" / "train"
    q100_dir = staging / "images_jpg_pilot" / "q100" / "train"
    for d in (ref_dir, q95_dir, q100_dir):
        d.mkdir(parents=True, exist_ok=True)

    decoded_unique: set = set()
    attempt = success = error = 0
    ctx["transform_records"] = {}
    ctx["uint8_cache"] = {}
    ctx["fidelity_by_key"] = {}   # (oid, quality) -> row
    ctx["geometry_rows"] = []
    fidelity_rows: List[Dict[str, Any]] = []

    for oid in selected_ids:
        assert oid in selected_set, "decode restricted to selected pilot set"
        header = headers[oid]
        path = resolved[oid]
        attempt += 1
        # Enforce explicit JPEG2000 backend BEFORE decode (no silent fallback).
        if P.is_jpeg2000(header["TransferSyntaxUID"]):
            P.ensure_jpeg2000_backend(args.jpeg2000_decoder)
        try:
            ds = pydicom.dcmread(str(path), force=False)
            stored, decoder_backend = decode_pixels(ds, header, args.jpeg2000_decoder)
            if stored.ndim != 2 or stored.shape != (header["Rows"], header["Columns"]):
                raise P.UnsupportedInputError("decoded array shape invalid")
            tr = transform_pixels(header, stored, ds)
            uint8 = tr["uint8"]
            tr["uint8_zero_fraction"] = float((uint8 == 0).mean())
            tr["uint8_255_fraction"] = float((uint8 == 255).mean())
            success += 1
            decoded_unique.add(oid)
        except P.Phase2D1BError:
            error += 1
            raise
        except Exception as exc:
            error += 1
            raise P.UnsupportedInputError(f"decode/transform failed for {oid}: {exc!r}")

        ctx["uint8_cache"][oid] = uint8

        # Lossless reference PNG + exact round-trip.
        ref_path = ref_dir / f"{oid}.png"
        Image.fromarray(uint8, mode="L").save(ref_path, format="PNG", optimize=False)
        with Image.open(ref_path) as reopened:
            assert reopened.mode == "L"
            decoded_ref = np.array(reopened)
        if decoded_ref.dtype != np.uint8 or decoded_ref.shape != uint8.shape \
                or not np.array_equal(decoded_ref, uint8):
            raise P.UnsupportedInputError(f"reference PNG not exact for {oid}")

        pre_sha = P.pre_jpeg_sha256(uint8)
        ref_byte_sha = file_sha256(ref_path)
        ref_pixel_sha = P.pre_jpeg_sha256(decoded_ref)

        for quality, out_dir in ((95, q95_dir), (100, q100_dir)):
            out_path = out_dir / f"{oid}.jpg"
            Image.fromarray(uint8, mode="L").save(out_path, format="JPEG",
                                                  quality=quality, optimize=False,
                                                  progressive=False)
            sha_a = file_sha256(out_path)
            tmp2 = out_dir / f"{oid}.det.tmp"
            Image.fromarray(uint8, mode="L").save(tmp2, format="JPEG", quality=quality,
                                                  optimize=False, progressive=False)
            sha_b = file_sha256(tmp2)
            os.remove(tmp2)
            if sha_a != sha_b:
                raise P.NonDeterministicEncodingError(f"jpeg not deterministic {oid} q{quality}")
            with Image.open(out_path) as jp:
                if jp.format != "JPEG" or jp.mode != "L" \
                        or jp.size != (header["Columns"], header["Rows"]):
                    raise P.UnsupportedInputError(f"JPEG validation failed {oid} q{quality}")
                exif = jp.getexif()
                orientation = exif.get(0x0112) if exif else None
                if orientation not in (None, 1):
                    raise P.UnsupportedInputError(f"EXIF orientation {orientation}")
                decoded_jpg = np.array(jp)
            m = P.whole_image_error_metrics(uint8, decoded_jpg)
            ssim = P.whole_image_ssim(uint8, decoded_jpg)
            size_bytes = out_path.stat().st_size
            row = {
                "original_image_id": oid,
                "coco_image_id": id_to_coco[oid]["id"],
                "jpeg_quality": quality,
                "decoder_backend": decoder_backend,
                "mae": m["mae"], "rmse": m["rmse"], "psnr_db": m["psnr_db"],
                "psnr_is_infinite": m["psnr_is_infinite"],
                "ssim": ssim["ssim"],
                "max_absolute_error": m["max_absolute_error"],
                "p95_absolute_error": m["p95_absolute_error"],
                "p99_absolute_error": m["p99_absolute_error"],
                "percentile_method": m["percentile_method"],
                "jpg_file_size_bytes": size_bytes,
                "pre_jpeg_uint8_bytes": int(uint8.size),
                "compression_ratio": (uint8.size / size_bytes) if size_bytes else None,
                "jpg_bytes_per_pixel": (size_bytes / uint8.size) if uint8.size else None,
                "ssim_win_size_whole_image": None,
                "skimage_version": ssim["skimage_version"],
                "output_jpg_sha256": sha_a,
                "decoded_jpg_uint8_sha256": P.pre_jpeg_sha256(decoded_jpg),
            }
            fidelity_rows.append(row)
            ctx["fidelity_by_key"][(oid, quality)] = row

            # Geometry validation row per image x quality.
            ctx["geometry_rows"].append({
                "original_image_id": oid, "jpeg_quality": quality,
                "dicom_rows": header["Rows"], "dicom_columns": header["Columns"],
                "coco_height": id_to_coco[oid]["height"], "coco_width": id_to_coco[oid]["width"],
                "pre_jpeg_shape_unchanged": True,
                "reference_png_shape_unchanged": True,
                "decoded_jpg_shape_unchanged": decoded_jpg.shape == uint8.shape,
                "reference_png_mode_L": True, "jpg_mode_L": True,
                "reference_png_dtype_uint8": True, "decoded_jpg_dtype_uint8": True,
                "reference_png_exact_pixel_match": True,
                "exif_orientation_absent_or_1": True,
                "bbox_scaling_required": False,
                "pixel_matrix_order_unchanged": True,
                "rotation_applied": False, "flip_applied": False,
                "transpose_applied": False, "exif_orientation_transform_applied": False,
            })

        ctx["transform_records"][oid] = {
            **{k: v for k, v in tr.items() if k != "uint8"},
            "pre_jpeg_uint8_sha256": pre_sha,
            "reference_png_byte_sha256": ref_byte_sha,
            "reference_png_decoded_pixel_sha256": ref_pixel_sha,
            "reference_png_exact_pixel_match": True,
            "source_dicom_sha256": file_sha256(path),
            "decoder_backend": decoder_backend,
            "rows": header["Rows"], "columns": header["Columns"],
        }

    ctx["fidelity_rows"] = fidelity_rows
    ctx["decode_stats"] = {
        "pixel_decode_attempt_count": attempt,
        "pixel_decode_success_count": success,
        "pixel_decode_error_count": error,
        "unique_pixel_decoded_image_count": len(decoded_unique),
    }
    return len(decoded_unique)


# =========================================================================== #
# Stage 12: BBox ROI fidelity + summaries                                       #
# =========================================================================== #
def compute_bbox_roi(ctx: Dict[str, Any]) -> None:
    import numpy as np
    from PIL import Image

    staging: Path = ctx["staging"]
    selected_set = set(ctx["selected_ids"])
    class_names = ctx["class_names"]
    id_to_coco = ctx["id_to_coco"]

    # Load canonical bbox rows for selected images only.
    canon_by_oid: Dict[str, List[Dict[str, Any]]] = {}
    with open(CANONICAL_BBOX, "r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["image_id"] in selected_set:
                canon_by_oid.setdefault(r["image_id"], []).append(r)

    q95_dir = staging / "images_jpg_pilot" / "q95" / "train"
    q100_dir = staging / "images_jpg_pilot" / "q100" / "train"
    roi_rows: List[Dict[str, Any]] = []
    coco_ann = ctx["coco_ann_by_cannid"]
    cat_by_canonical = ctx["cat_by_canonical"]
    n_annotations = 0

    for oid in ctx["selected_ids"]:
        pre = ctx["uint8_cache"][oid]
        im = id_to_coco[oid]
        decoded = {}
        for quality, d in ((95, q95_dir), (100, q100_dir)):
            with Image.open(d / f"{oid}.jpg") as jp:
                decoded[quality] = np.array(jp)
        for r in canon_by_oid.get(oid, []):
            # Cross-check canonical <-> COCO (identifier, class, category mapping,
            # xyxy vs xywh). Any mismatch BLOCKS; we never repair the bbox.
            matched_ann = coco_ann.get(str(r["canonical_ann_id"]))
            P.crosscheck_canonical_coco_bbox(r, matched_ann, cat_by_canonical)
            # category_id sourced from the cross-checked metadata mapping,
            # NEVER computed as canonical_class_id + 1.
            category_id = int(cat_by_canonical[int(r["canonical_class_id"])])
            n_annotations += 1
            x_min, y_min = float(r["x_min"]), float(r["y_min"])
            x_max, y_max = float(r["x_max"]), float(r["y_max"])
            x0, y0, x1, y1 = P.roi_extraction_coords(x_min, y_min, x_max, y_max)
            # No silent clamping: extraction coords must be within bounds.
            P.assert_extraction_in_bounds(x0, y0, x1, y1, im["width"], im["height"])
            area = float(r["bbox_area"])
            rel = area / (im["width"] * im["height"])
            for quality in (95, 100):
                ref_roi = pre[y0:y1, x0:x1]
                tgt_roi = decoded[quality][y0:y1, x0:x1]
                m = P.whole_image_error_metrics(ref_roi, tgt_roi)
                s = P.roi_ssim(ref_roi, tgt_roi)
                roi_rows.append(_roi_row(
                    r, oid, im, quality, class_names, area, rel, (x0, y0, x1, y1),
                    m["mae"], m["psnr_db"], s["ssim"], s["evaluable"], s["reason"],
                    s["win_size"], m["psnr_is_infinite"], m["max_absolute_error"],
                    ref_roi.shape, category_id=category_id))

    ctx["roi_rows"] = roi_rows
    ctx["n_selected_annotations"] = n_annotations
    # Summaries.
    ctx["roi_summary_mae"] = P.summarize_roi_metrics(roi_rows, "ROI_MAE")
    ctx["roi_summary_ssim"] = P.summarize_roi_metrics(roi_rows, "ROI_SSIM")
    ctx["roi_worst"] = P.worst_roi_cases(roi_rows, "ROI_MAE", top=5, largest_is_worst=True)
    ctx["roi_pairwise"] = P.pairwise_q100_minus_q95(roi_rows)


def _roi_row(r, oid, im, quality, class_names, area, rel, ext,
             mae, psnr, ssim, evaluable, reason, win, psnr_inf=None, maxerr=None,
             roi_shape=None, category_id=None) -> Dict[str, Any]:
    x0, y0, x1, y1 = ext
    ccid = int(r["canonical_class_id"])
    if category_id is None:
        raise P.UnsupportedInputError(
            "category_id must be supplied from metadata (never canonical+1)")
    return {
        "annotation_id": r["canonical_ann_id"], "canonical_ann_id": r["canonical_ann_id"],
        "source_row_id": r.get("source_row_id", ""), "rad_id": r.get("rad_id", ""),
        "image_id": oid, "coco_image_id": im["id"],
        "category_id": int(category_id), "canonical_class_id": ccid,
        "class_id_original": r.get("class_id_original", ""),
        "class_name": class_names.get(ccid, r.get("class_name", "")),
        "canonical_x_min": r["x_min"], "canonical_y_min": r["y_min"],
        "canonical_x_max": r["x_max"], "canonical_y_max": r["y_max"],
        "bbox_width": r["bbox_width"], "bbox_height": r["bbox_height"],
        "bbox_area": area, "relative_bbox_area": rel,
        "extraction_x0": x0, "extraction_y0": y0, "extraction_x1": x1, "extraction_y1": y1,
        "roi_width": (roi_shape[1] if roi_shape else 0),
        "roi_height": (roi_shape[0] if roi_shape else 0),
        "ROI_MAE": mae, "ROI_PSNR": psnr, "ROI_PSNR_is_infinite": psnr_inf,
        "ROI_SSIM": ssim, "ROI_SSIM_evaluable": evaluable, "ROI_SSIM_reason": reason,
        "ROI_SSIM_win_size": win, "ROI_maximum_absolute_error": maxerr,
        "jpeg_quality": quality,
    }


# =========================================================================== #
# Stage 14: visual evidence (deterministic subset)                              #
# =========================================================================== #
def _stratum_signature(ctx: Dict[str, Any], oid: str) -> str:
    """Metadata-stratum signature of an image (excludes class/extremum/scope)."""
    feats = ctx["image_features"].get(oid, set())
    return "|".join(sorted(f for f in feats
                           if not f.startswith(("class=", "extremum=", "scope="))))


def select_visual_subset(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic visual subset.

    Returns image-level reasons, annotation-level crop requests (each tied to
    the SPECIFIC canonical_ann_id that activated the reason), and the chosen No
    Finding images with their metadata-stratum signatures.

    Annotation-triggered reasons (smallest overall / smallest per class / worst
    q95 ROI / rare class) are recorded per canonical_ann_id so the bbox crop
    shows the exact activating annotation. No Finding coverage requires >=4
    UNIQUE images with >=4 DISTINCT metadata-stratum signatures, else hard fail.
    """
    from collections import defaultdict
    selected_set = set(ctx["selected_ids"])
    image_reasons: Dict[str, List[str]] = defaultdict(list)
    ann_requests: List[Dict[str, Any]] = []
    nf_images: Dict[str, str] = {}

    def add_img(oid: str, reason: str) -> None:
        if oid in selected_set and reason not in image_reasons[oid]:
            image_reasons[oid].append(reason)

    def add_ann(image_id: str, ann_id: Any, reason: str) -> None:
        ann_requests.append({"image_id": image_id, "canonical_ann_id": str(ann_id),
                             "reason": reason})

    # image-level: dimension / pixel / bbox extrema
    for feat, oid in ctx["mandatory"].items():
        add_img(oid, f"extremum:{feat}")

    q95_roi = [r for r in ctx["roi_rows"] if int(r["jpeg_quality"]) == 95]

    # annotation-level: smallest relative bbox overall + per class
    sl = P.small_lesion_ranking(ctx["roi_rows"])
    if sl["smallest_overall"]:
        r = sl["smallest_overall"]
        add_ann(r["image_id"], r["canonical_ann_id"], "smallest_relative_bbox_overall")
    for cid, r in sl["smallest_per_class"].items():
        add_ann(r["image_id"], r["canonical_ann_id"], f"smallest_relative_bbox_class{cid}")

    # annotation-level: rare classes -> one representative annotation each
    classes_present = sorted({r["canonical_class_id"] for r in q95_roi})
    rare = P.rare_class_ranking(ctx["class_image_count"], classes_present)
    for cid in rare["rare_classes"]:
        cand = sorted([r for r in q95_roi if r["canonical_class_id"] == cid],
                      key=lambda r: (float(r["relative_bbox_area"]),
                                     P.tie_break_rank(str(r["canonical_ann_id"]))))
        if cand:
            add_ann(cand[0]["image_id"], cand[0]["canonical_ann_id"], f"rare_class{cid}")

    # annotation-level: worst q95 ROI (top 3 by ROI_MAE)
    worst = sorted([r for r in q95_roi if r.get("ROI_MAE") is not None],
                   key=lambda r: (r["ROI_MAE"], P.tie_break_rank(str(r["canonical_ann_id"]))),
                   reverse=True)[:3]
    for r in worst:
        add_ann(r["image_id"], r["canonical_ann_id"], "worst_q95_roi")

    # image-level: worst q95 whole-image
    q95_wi = sorted([r for r in ctx["fidelity_rows"] if r["jpeg_quality"] == 95],
                    key=lambda r: (r["mae"], P.tie_break_rank(r["original_image_id"])),
                    reverse=True)[:3]
    for r in q95_wi:
        add_img(r["original_image_id"], "worst_q95_whole_image")

    # image-level: No Finding with DISTINCT strata (unique images + signatures)
    n_nf = len(ctx["negative_ids"] & selected_set)
    if n_nf > 0:
        seen: Dict[str, str] = {}  # signature -> oid (first, deterministic)
        for oid in sorted(ctx["negative_ids"] & selected_set, key=P.tie_break_rank):
            sig = _stratum_signature(ctx, oid)
            if sig not in seen:
                seen[sig] = oid
            if len(seen) >= 4:
                break
        if len(seen) < 4:
            raise P.UnsupportedInputError(
                f"insufficient_distinct_no_finding_strata: only {len(seen)} distinct "
                "metadata strata among selected No Finding images (need >=4); "
                "refusing to accept duplicate strata")
        for sig, oid in seen.items():
            add_img(oid, "no_finding_strata_diverse")
            nf_images[oid] = sig

    # image-level: padding-unusual + saturation-unusual
    for oid, tr in ctx["transform_records"].items():
        if tr.get("padding_present"):
            add_img(oid, "padding_unusual")
    sat = sorted(ctx["transform_records"].items(),
                 key=lambda kv: (kv[1].get("uint8_255_fraction", 0.0)
                                 + kv[1].get("uint8_zero_fraction", 0.0)),
                 reverse=True)[:2]
    for oid, _ in sat:
        add_img(oid, "saturation_unusual")

    # image-level: all warning cases
    for w in ctx.get("warnings_rows", []):
        add_img(w["image_id"], f"warning:{w.get('code', '')}")

    union = set(image_reasons) | {a["image_id"] for a in ann_requests}
    if len(union) > VISUAL_SUBSET_MAX:
        raise P.UnsupportedInputError(
            f"visual_subset_exceeds_cap: {len(union)} required visual images > "
            f"cap {VISUAL_SUBSET_MAX}; refusing to truncate required cases")

    return {"image_reasons": dict(image_reasons), "ann_requests": ann_requests,
            "nf_images": nf_images}


def visual_expectations(ctx: Dict[str, Any], subset: Dict[str, Any]) -> Dict[str, Any]:
    """Expected visual coverage the validator checks independently."""
    n_nf = len(ctx["negative_ids"] & set(ctx["selected_ids"]))
    ann_req = [{"canonical_ann_id": a["canonical_ann_id"], "image_id": a["image_id"],
                "reason": a["reason"]} for a in subset["ann_requests"]]
    return {
        "expect_extremum": True,
        "expect_worst_q95_whole_image": any(r["jpeg_quality"] == 95
                                            for r in ctx["fidelity_rows"]),
        "expect_padding": any(t.get("padding_present")
                              for t in ctx["transform_records"].values()),
        "expect_saturation": bool(ctx["transform_records"]),
        "warning_image_ids": sorted({w["image_id"] for w in ctx.get("warnings_rows", [])}),
        "min_no_finding_unique": 4 if n_nf > 0 else 0,
        "min_no_finding_distinct_strata": 4 if n_nf > 0 else 0,
        "annotation_requests": ann_req,
    }


def _open_gray(path: Path):
    import numpy as np
    from PIL import Image
    with Image.open(path) as im:
        return np.array(im)


def generate_visual_evidence(ctx: Dict[str, Any]) -> None:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    staging: Path = ctx["staging"]
    plots = staging / "plots" / "phase2D1B_pilot"
    for sub in ("full_image", "bbox_crops", "difference_heatmaps", "contact_sheets"):
        (plots / sub).mkdir(parents=True, exist_ok=True)

    q95_dir = staging / "images_jpg_pilot" / "q95" / "train"
    q100_dir = staging / "images_jpg_pilot" / "q100" / "train"
    subset = select_visual_subset(ctx)
    image_reasons = subset["image_reasons"]
    ann_requests = subset["ann_requests"]
    nf_images = subset["nf_images"]
    manifest: List[Dict[str, Any]] = []

    # ROI (q95) indexed by canonical_ann_id for annotation-specific crops.
    roi_by_ann: Dict[str, Dict[str, Any]] = {}
    for r in ctx["roi_rows"]:
        if int(r["jpeg_quality"]) == 95:
            roi_by_ann[str(r["canonical_ann_id"])] = r

    def sig(oid: str) -> str:
        return nf_images.get(oid, _stratum_signature(ctx, oid))

    thumbs = []
    # ---- image-level panels (full_image + difference heatmap) ----
    for oid, reasons in image_reasons.items():
        pre = ctx["uint8_cache"][oid]
        a95 = _open_gray(q95_dir / f"{oid}.jpg")
        a100 = _open_gray(q100_dir / f"{oid}.jpg")
        d95 = np.abs(pre.astype(np.int16) - a95.astype(np.int16)).astype(np.uint8)
        d100 = np.abs(pre.astype(np.int16) - a100.astype(np.int16)).astype(np.uint8)
        tr = ctx["transform_records"][oid]
        reason_str = ";".join(reasons)

        fig, ax = plt.subplots(1, 5, figsize=(15, 3))
        for a, img, title in zip(ax, (pre, a95, a100, d95, d100),
                                 ("pre-JPEG ref", "q95", "q100", "q95 |diff|", "q100 |diff|")):
            a.imshow(img, cmap="gray", vmin=0, vmax=255)
            a.set_title(title, fontsize=8)
            a.axis("off")
        fig.suptitle(f"{oid} | mod={tr['modality_branch']} voi={tr['voi_branch']} "
                     f"inv={tr['presentation_inversion_count']} | {reason_str}", fontsize=7)
        fig.savefig(plots / "full_image" / f"{oid}.png", dpi=72, bbox_inches="tight")
        plt.close(fig)
        manifest.append({"image_id": oid, "artifact_type": "full_image",
                         "artifact_path": f"full_image/{oid}.png", "canonical_ann_id": "",
                         "reason": reason_str, "stratum_signature": sig(oid),
                         "review_status": "PENDING_GPT", "critical_visual_failure": "",
                         "review_notes": ""})

        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        for a, img, t in zip(ax, (d95, d100), ("q95 |diff|", "q100 |diff|")):
            im0 = a.imshow(img, cmap="magma", vmin=0, vmax=255)
            a.set_title(t, fontsize=8)
            a.axis("off")
        fig.colorbar(im0, ax=ax.ravel().tolist(), fraction=0.046)
        fig.savefig(plots / "difference_heatmaps" / f"{oid}.png", dpi=72, bbox_inches="tight")
        plt.close(fig)
        manifest.append({"image_id": oid, "artifact_type": "difference_heatmap",
                         "artifact_path": f"difference_heatmaps/{oid}.png",
                         "canonical_ann_id": "", "reason": reason_str,
                         "stratum_signature": sig(oid), "review_status": "PENDING_GPT",
                         "critical_visual_failure": "", "review_notes": ""})
        thumbs.append((oid, pre))

    # ---- annotation-level bbox crops (one per required canonical_ann_id) ----
    from collections import defaultdict
    ann_group: Dict[Any, Dict[str, Any]] = defaultdict(
        lambda: {"reasons": [], "image_id": None})
    for req in ann_requests:
        g = ann_group[req["canonical_ann_id"]]
        g["image_id"] = req["image_id"]
        if req["reason"] not in g["reasons"]:
            g["reasons"].append(req["reason"])

    for ann_id, g in ann_group.items():
        oid = g["image_id"]
        roi = roi_by_ann.get(str(ann_id))
        if roi is None:
            raise P.UnsupportedInputError(f"visual crop request for unknown ann {ann_id}")
        pre = ctx["uint8_cache"][oid]
        a95 = _open_gray(q95_dir / f"{oid}.jpg")
        a100 = _open_gray(q100_dir / f"{oid}.jpg")
        x0, y0 = int(roi["extraction_x0"]), int(roi["extraction_y0"])
        x1, y1 = int(roi["extraction_x1"]), int(roi["extraction_y1"])
        cpre, c95, c100 = pre[y0:y1, x0:x1], a95[y0:y1, x0:x1], a100[y0:y1, x0:x1]
        fig, ax = plt.subplots(1, 3, figsize=(9, 3))
        for a, img, t in zip(ax, (cpre, c95, c100),
                             (f"ref {roi.get('class_name', '')}", "q95", "q100")):
            a.imshow(img, cmap="gray", vmin=0, vmax=255)
            a.set_title(t, fontsize=8)
            a.axis("off")
        rel = f"bbox_crops/{oid}__{ann_id}.png"
        fig.suptitle(f"{oid} ann={ann_id} | {';'.join(g['reasons'])}", fontsize=7)
        fig.savefig(plots / rel, dpi=72, bbox_inches="tight")
        plt.close(fig)
        manifest.append({"image_id": oid, "artifact_type": "bbox_crop",
                         "artifact_path": rel, "canonical_ann_id": str(ann_id),
                         "reason": ";".join(g["reasons"]),
                         "stratum_signature": sig(oid), "review_status": "PENDING_GPT",
                         "critical_visual_failure": "", "review_notes": ""})
        if oid not in dict(thumbs):
            thumbs.append((oid, pre))

    # ---- contact sheet ----
    if thumbs:
        n = len(thumbs)
        cols = min(6, n)
        rows_n = (n + cols - 1) // cols
        fig, ax = plt.subplots(rows_n, cols, figsize=(2 * cols, 2 * rows_n))
        axes = np.array(ax).reshape(-1)
        for a in axes:
            a.axis("off")
        for a, (oid, img) in zip(axes, thumbs):
            a.imshow(img, cmap="gray", vmin=0, vmax=255)
            a.set_title(oid[:8], fontsize=6)
        fig.suptitle("contact sheet (display resized only; all metrics computed at "
                     "native resolution)", fontsize=8)
        fig.savefig(plots / "contact_sheets" / "contact_sheet.png", dpi=72, bbox_inches="tight")
        plt.close(fig)
        manifest.append({"image_id": "*", "artifact_type": "contact_sheet",
                         "artifact_path": "contact_sheets/contact_sheet.png",
                         "canonical_ann_id": "", "reason": "deterministic_subset_overview",
                         "stratum_signature": "", "review_status": "PENDING_GPT",
                         "critical_visual_failure": "", "review_notes": ""})

    ctx["visual_manifest"] = manifest
    ctx["visual_expectations"] = visual_expectations(ctx, subset)


# =========================================================================== #
# Stage 15: write all mandatory evidence artifacts                              #
# =========================================================================== #
def write_all_evidence(ctx: Dict[str, Any], decoded_count: int) -> None:
    staging: Path = ctx["staging"]
    reports = staging / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    mapping_dir = staging / "image_mapping"
    mapping_dir.mkdir(parents=True, exist_ok=True)

    atomic_write_text(reports / "phase2D1B_pilot_environment.json",
                      strict_json_dumps(ctx["decoder_env"]) + "\n")
    write_decision_templates(reports)

    # ---- validation.json / .md ----
    status = baseline_validation_status()
    status.update({
        "decoded_pixel_image_count": decoded_count,
        "pilot_selected_image_count": len(ctx["selected_ids"]),
        "no_finding_selected_count": sum(1 for i in ctx["selected_ids"] if i in ctx["negative_ids"]),
        "decode_stats": ctx["decode_stats"],
        "protocol_sha256": ctx["protocol_evidence"]["protocol_sha256"],
        "coco_master_sha256": ctx["coco_sha256"],
        "coverage_result": ctx["coverage_result"],
        "forbidden_artifact_snapshot": ctx["forbidden"],
    })
    atomic_write_text(reports / "phase2D1B_pilot_validation.json",
                      strict_json_dumps(status) + "\n")
    atomic_write_text(
        reports / "phase2D1B_pilot_validation.md",
        "# Phase 2D.1B-Pilot - Validation\n\n"
        "## Scientific Positioning and Scope\n\n"
        "This is a DICOM metadata-aware, standard-aligned *reference "
        "representation* pipeline. Phase 2D.1B does **not** prove that this "
        "representation produces the best downstream detection performance; "
        "claiming superiority over alternative preprocessing strategies would "
        "require controlled downstream evidence. Whether a controlled "
        "downstream ablation is included remains pending mentor confirmation "
        "and is not part of the Phase 2D.1B Definition of Done.\n\n"
        "Geometry preservation here refers to unchanged pixel-matrix "
        "dimensions and ordering (no crop/resize/rotation/flip/transpose/EXIF "
        "orientation). It does not claim independent validation of "
        "patient-space orientation.\n\n"
        f"- phase_status: {status['phase_status']}\n"
        f"- structural_dod_candidate: {status['structural_dod_candidate']}\n"
        f"- final_jpeg_quality: null\n"
        f"- pilot images decoded: {decoded_count}\n",
    )

    # ---- header inventory: ALL 4894 rows ----
    hdr_rows = ctx["header_rows"]
    hdr_keys: List[str] = []
    for row in hdr_rows:
        for k in row:
            if k not in hdr_keys:
                hdr_keys.append(k)
    write_csv(reports / "phase2D1B_pilot_header_inventory.csv", hdr_keys, hdr_rows)

    # ---- metadata strata (from all headers) ----
    write_metadata_strata(ctx, reports)

    # ---- selection + selection_coverage ----
    write_selection(ctx, reports)
    write_selection_coverage(ctx, reports)

    # ---- fidelity metrics (whole image) ----
    if ctx["fidelity_rows"]:
        write_csv(reports / "phase2D1B_pilot_fidelity_metrics.csv",
                  list(ctx["fidelity_rows"][0].keys()), ctx["fidelity_rows"])

    # ---- bbox ROI metrics ----
    if ctx["roi_rows"]:
        write_csv(reports / "phase2D1B_pilot_bbox_roi_metrics.csv",
                  list(ctx["roi_rows"][0].keys()), ctx["roi_rows"])

    # ---- quality summary + pairwise ----
    write_quality_summary(ctx, reports)
    write_quality_pairwise(ctx, reports)

    # ---- geometry validation ----
    if ctx["geometry_rows"]:
        write_csv(reports / "phase2D1B_pilot_geometry_validation.csv",
                  list(ctx["geometry_rows"][0].keys()), ctx["geometry_rows"])

    # ---- visual audit manifest ----
    write_csv(reports / "phase2D1B_pilot_visual_audit_manifest.csv",
              ["image_id", "artifact_type", "artifact_path", "canonical_ann_id",
               "reason", "stratum_signature", "review_status",
               "critical_visual_failure", "review_notes"],
              ctx.get("visual_manifest", []))

    # ---- multi-window audit ----
    write_multi_window_audit(ctx, reports)

    # ---- reference renderer concordance + viewer manifest ----
    write_reference_renderer(ctx, reports)

    # ---- synthetic conformance ----
    write_synthetic_conformance(reports)

    # ---- errors CSV (actual warnings/errors) ----
    err_rows = list(ctx.get("warnings_rows", []))
    for oid in ctx["selected_ids"]:
        for r in ctx.get("roi_rows", []):
            if r["image_id"] == oid and r.get("ROI_SSIM_evaluable") is False:
                err_rows.append({"image_id": oid, "severity": "info",
                                 "code": "roi_ssim_not_evaluable",
                                 "detail": f"ann={r['canonical_ann_id']} "
                                           f"reason={r['ROI_SSIM_reason']}"})
    write_csv(reports / "phase2D1B_pilot_errors.csv",
              ["image_id", "severity", "code", "detail"], err_rows)

    # ---- mapping (full schema, joined hashes) ----
    write_mapping(ctx, mapping_dir)


def write_metadata_strata(ctx, reports) -> None:
    from collections import defaultdict
    scope_counts: Dict[str, set] = defaultdict(set)
    for oid, feats in ctx["image_features"].items():
        for f in feats:
            scope_counts[f].add(oid)
    selected_set = set(ctx["selected_ids"])
    rows = []
    for feat in sorted(ctx["all_features"]):
        kind = feat.split("=", 1)[0]
        stype = ("class" if kind == "class" else "extremum" if kind == "extremum"
                 else "scope" if kind == "scope" else "stratum")
        sel_ids = sorted(scope_counts[feat] & selected_set)
        rows.append({
            "stratum_type": stype,
            "stratum_name": kind,
            "stratum_value": feat.split("=", 1)[1] if "=" in feat else feat,
            "scope_image_count": len(scope_counts[feat]),
            "selected_image_count": len(sel_ids),
            "covered": len(sel_ids) > 0,
            "selected_image_ids": ";".join(sel_ids[:20]),
        })
    write_csv(reports / "phase2D1B_pilot_metadata_strata.csv",
              ["stratum_type", "stratum_name", "stratum_value", "scope_image_count",
               "selected_image_count", "covered", "selected_image_ids"], rows)


def write_selection(ctx, reports) -> None:
    rows = []
    for r in ctx["records"]:
        oid = r["image_id"]
        im = ctx["id_to_coco"][oid]
        anns = ctx["ann_by_oid"].get(oid, [])
        rows.append({
            "selection_order": r["selection_order"], "original_image_id": oid,
            "coco_image_id": im["id"], "canonical_image_id": im["canonical_image_id"],
            "scope_label": im.get("scope_label", ""), "is_negative": im.get("is_negative", False),
            "class_ids_coco": ";".join(sorted({str(a["category_id"]) for a in anns})),
            "canonical_class_ids": ";".join(sorted({str(a["canonical_class_id"]) for a in anns})),
            "class_names": ";".join(sorted({ctx["class_names"].get(a["canonical_class_id"], "")
                                            for a in anns})),
            "width": im["width"], "height": im["height"],
            "pixel_count": im["width"] * im["height"], "bbox_count": len(anns),
            "minimum_bbox_area": min((float(a["area"]) for a in anns), default=""),
            "minimum_relative_bbox_area": min(
                (float(a["area"]) / (im["width"] * im["height"]) for a in anns), default=""),
            "selected_for_features": ";".join(r["selected_for_features"]),
            "newly_covered_feature_count": r["newly_covered_feature_count"],
            "tie_break_rank": r["tie_break_rank"],
        })
    write_csv(reports / "phase2D1B_pilot_selection.csv", list(rows[0].keys()), rows)


def write_selection_coverage(ctx, reports) -> None:
    from collections import defaultdict
    feat_to_selected: Dict[str, set] = defaultdict(set)
    selected_set = set(ctx["selected_ids"])
    for oid in selected_set:
        for f in ctx["image_features"][oid]:
            feat_to_selected[f].add(oid)
    rows = []
    for feat in sorted(ctx["all_features"]):
        kind = feat.split("=", 1)[0]
        sel = sorted(feat_to_selected.get(feat, set()))
        rows.append({
            "feature": feat,
            "feature_kind": kind,
            "covered": len(sel) > 0,
            "selected_count": len(sel),
            "selected_image_ids": ";".join(sel[:20]),
        })
    cov = ctx["coverage_result"]
    rows.append({"feature": "__SUMMARY__", "feature_kind": "summary",
                 "covered": cov["fully_covered"],
                 "selected_count": cov["covered_total"],
                 "selected_image_ids": f"classes={cov['classes_covered']}/{cov['classes_expected']};"
                                       f"extrema={cov['extrema_covered']}/{cov['extrema_expected']}"})
    write_csv(reports / "phase2D1B_pilot_selection_coverage.csv",
              ["feature", "feature_kind", "covered", "selected_count", "selected_image_ids"], rows)


def write_quality_summary(ctx, reports) -> None:
    rows = []
    fidelity = ctx["fidelity_rows"]
    roi = ctx["roi_rows"]

    def add(scope, q, metric, micro, image_macro, class_macro, worst, extra=""):
        rows.append({"scope": scope, "jpeg_quality": q, "metric": metric,
                     "annotation_micro_mean": micro, "image_macro_mean": image_macro,
                     "class_macro_mean": class_macro, "worst_value": worst, "detail": extra})

    # ---- whole-image micro + image-macro + worst ----
    for q in (95, 100):
        wr = [r for r in fidelity if r["jpeg_quality"] == q]
        if not wr:
            continue

        def mean(key):
            vals = [r[key] for r in wr if r.get(key) is not None]
            return sum(vals) / len(vals) if vals else None
        # whole-image "image-macro" == micro (one row per image already).
        add("whole_image", q, "MAE", mean("mae"), mean("mae"), "",
            max((r["mae"] for r in wr), default=""))
        add("whole_image", q, "SSIM", mean("ssim"), mean("ssim"), "",
            min((r["ssim"] for r in wr if r["ssim"] is not None), default=""))
        add("whole_image", q, "PSNR", mean("psnr_db"), mean("psnr_db"), "",
            min((r["psnr_db"] for r in wr if r["psnr_db"] is not None), default=""))
        # worst whole-image case (largest MAE).
        worst_wi = sorted(wr, key=lambda r: (r["mae"], P.tie_break_rank(r["original_image_id"])),
                          reverse=True)[:1]
        if worst_wi:
            add("whole_image_worst", q, "MAE", "", "", "", worst_wi[0]["mae"],
                worst_wi[0]["original_image_id"])

    # ---- ROI micro / image-macro / class-macro + worst ----
    for metric, summ in (("ROI_MAE", ctx["roi_summary_mae"]),
                         ("ROI_SSIM", ctx["roi_summary_ssim"])):
        for q, s in summ.items():
            worst = ctx["roi_worst"].get(q, [])
            add("roi", q, metric, s["annotation_micro_mean"], s["image_macro_mean"],
                s["class_macro_mean"], (worst[0].get(metric) if worst else ""),
                (worst[0].get("annotation_id") if worst else ""))

    # ---- per-class distributions ----
    for d in P.per_class_distribution(roi, "ROI_MAE"):
        add("per_class_distribution", d["jpeg_quality"], f"ROI_MAE_class{d['canonical_class_id']}",
            d["mean"], d["min"], d["max"], d["max"], f"n={d['n']}")

    # ---- small-lesion summary (deterministic, relative_bbox_area ascending) ----
    sl = P.small_lesion_ranking(roi)
    if sl["smallest_overall"]:
        add("small_lesion", 95, "smallest_overall_relative_area",
            sl["smallest_overall"]["relative_bbox_area"], "", "",
            sl["smallest_overall"]["ROI_MAE"], sl["ranking_basis"])
    for cid, r in sorted(sl["smallest_per_class"].items(), key=lambda kv: kv[0]):
        add("small_lesion_per_class", 95, f"class{cid}_smallest_relative_area",
            r["relative_bbox_area"], "", "", r["ROI_MAE"], r["canonical_ann_id"])

    # ---- rare-class summary (canonical image_count ascending) ----
    classes_present = sorted({r["canonical_class_id"] for r in roi})
    rare = P.rare_class_ranking(ctx["class_image_count"], classes_present)
    for cid in rare["rare_classes"]:
        add("rare_class", "", f"class{cid}", rare["counts"][cid], "", "",
            rare["counts"][cid], rare["ranking_basis"])

    write_csv(reports / "phase2D1B_pilot_quality_summary.csv",
              ["scope", "jpeg_quality", "metric", "annotation_micro_mean",
               "image_macro_mean", "class_macro_mean", "worst_value", "detail"], rows)


def write_quality_pairwise(ctx, reports) -> None:
    # ROI pairwise (per annotation) + whole-image pairwise (per image).
    roi_rows = ctx.get("roi_pairwise", [])
    wi_rows = P.pairwise_q100_minus_q95(
        ctx["fidelity_rows"], metric_keys=("mae", "psnr_db", "ssim"),
        key_field="original_image_id")
    rows = []
    for r in roi_rows:
        rows.append({"scope": "roi", "unit_id": r["canonical_ann_id"],
                     "MAE_q100_minus_q95": r.get("ROI_MAE_q100_minus_q95"),
                     "PSNR_q100_minus_q95": r.get("ROI_PSNR_q100_minus_q95"),
                     "SSIM_q100_minus_q95": r.get("ROI_SSIM_q100_minus_q95")})
    for r in wi_rows:
        rows.append({"scope": "whole_image", "unit_id": r["original_image_id"],
                     "MAE_q100_minus_q95": r.get("mae_q100_minus_q95"),
                     "PSNR_q100_minus_q95": r.get("psnr_db_q100_minus_q95"),
                     "SSIM_q100_minus_q95": r.get("ssim_q100_minus_q95")})
    write_csv(reports / "phase2D1B_pilot_quality_pairwise.csv",
              ["scope", "unit_id", "MAE_q100_minus_q95", "PSNR_q100_minus_q95",
               "SSIM_q100_minus_q95"], rows)


def write_multi_window_audit(ctx, reports) -> None:
    rows = []
    for oid in ctx["selected_ids"]:
        h = ctx["headers"][oid]
        wdec = P.classify_window(h["WindowCenter_all"], h["WindowWidth_all"])
        if wdec.state == "valid" and (len(wdec.centers) > 1 or len(wdec.widths) > 1):
            rows.append({
                "image_id": oid, "window_state": wdec.state,
                "all_window_centers": ";".join(str(x) for x in wdec.centers),
                "all_window_widths": ";".join(str(x) for x in wdec.widths),
                "selected_window_index": 0,
                "selected_center": wdec.center, "selected_width": wdec.width,
                "voi_lut_function": h["VOILUTFunction"],
            })
    write_csv(reports / "phase2D1B_pilot_multi_window_audit.csv",
              ["image_id", "window_state", "all_window_centers", "all_window_widths",
               "selected_window_index", "selected_center", "selected_width",
               "voi_lut_function"], rows)


def write_reference_renderer(ctx, reports) -> None:
    # Status is chosen from ACTUAL evidence, not hard-coded. A rendering
    # dependency (pydicom apply_voi_lut) IS available, but no controlled,
    # comparable renderer configuration (matched VOI index, window, function,
    # polarity, padding, bit depth) is defined in this pilot -> the correct
    # status is NOT_COMPARABLE_CONFIGURATION_UNCONTROLLED (not dependency-missing).
    dependency_available = ctx.get("renderer_dependency_available", True)
    controlled_configuration = ctx.get("renderer_controlled_configuration", False)
    status = P.reference_renderer_status(dependency_available, controlled_configuration)
    conc_rows = []
    for oid in ctx["selected_ids"]:
        conc_rows.append({
            "image_id": oid,
            "independent_renderer_status": status,
            "comparable": status == "PASS",
            "note": "renderer dependency present but comparison configuration is "
                    "not controlled in this pilot",
        })
    write_csv(reports / "phase2D1B_pilot_reference_renderer_concordance.csv",
              ["image_id", "independent_renderer_status", "comparable", "note"], conc_rows)

    man_rows = []
    for oid in ctx["selected_ids"]:
        h = ctx["headers"][oid]
        tr = ctx["transform_records"][oid]
        man_rows.append({
            "image_id": oid, "photometric_interpretation": h["PhotometricInterpretation"],
            "presentation_lut_shape": h["PresentationLUTShape"],
            "modality_branch": tr["modality_branch"], "voi_branch": tr["voi_branch"],
            "reference_png_relative_path": f"images_jpg_pilot/reference_uint8/train/{oid}.png",
            "review_status": "PENDING_EXPERT_REVIEW",
        })
    write_csv(reports / "phase2D1B_pilot_reference_viewer_manifest.csv",
              ["image_id", "photometric_interpretation", "presentation_lut_shape",
               "modality_branch", "voi_branch", "reference_png_relative_path",
               "review_status"], man_rows)


def write_synthetic_conformance(reports) -> None:
    """Run synthetic (in-memory) transformation cases; record actual vs locked.

    These are SYNTHETIC arrays, not real DICOM. Expected outputs are defined
    INDEPENDENTLY (hand-computed constants / explicit locked formulas), never by
    calling the same production helper under test. They validate implementation
    conformity to the locked protocol; they do NOT certify formal, complete
    DICOM Standard conformance.
    """
    import numpy as np
    cases: List[Dict[str, Any]] = []

    def case(name: str, ok: bool, status: str = "PASS", **extra):
        cases.append({"case": name, "pass": bool(ok), "test_status": status, **extra})

    # --- stored ranges (unsigned / signed) ---
    case("unsigned_stored_range_bits12",
         P.theoretical_stored_range(12, 0) == (0, 4095), expected="(0,4095)")
    case("signed_stored_range_bits12",
         P.theoretical_stored_range(12, 1) == (-2048, 2047), expected="(-2048,2047)")

    # --- modality branches ---
    case("identity_modality_branch",
         P.modality_branch_name(False, False, False) == "identity", expected="identity")
    _, plo, phi = P.apply_rescale(np.array([0, 4095]), 1.0, -1024.0, 0, 4095)
    case("positive_rescale_bounds", (plo, phi) == (-1024.0, 3071.0),
         expected="(-1024.0,3071.0)")
    _, nlo, nhi = P.apply_rescale(np.array([0, 100]), -1.0, 100.0, 0, 100)
    case("negative_rescale_bounds_sorted", (nlo, nhi) == (0.0, 100.0),
         expected="(0.0,100.0)")
    case("modality_lut_output_bounds_from_data",
         P.modality_lut_output_bounds(list(range(100, 356))) == (100.0, 355.0),
         expected="(100.0,355.0)")

    # --- window functions (independent formulas) ---
    fl = P.window_linear(np.array([800.0, 1000.0, 1200.0]), 1000.0, 400.0)
    # independent LINEAR at center: (1000-999.5)/399 + 0.5
    exp_center = (1000.0 - 999.5) / 399.0 + 0.5
    case("window_linear", bool(np.isclose(fl[0], 0.0) and np.isclose(fl[2], 1.0)
                               and np.isclose(fl[1], exp_center)),
         expected=f"edges 0/1, center {exp_center:.6f}")
    fle = P.window_linear_exact(np.array([1000.0]), 1000.0, 200.0)
    case("window_linear_exact_center_half", bool(np.isclose(fle[0], 0.5)), expected="0.5")
    fs = P.window_sigmoid(np.array([1000.0]), 1000.0, 200.0)
    case("window_sigmoid_center_half", bool(np.isclose(fs[0], 0.5)), expected="0.5")

    # --- VOI LUT normalization (bit depth, not observed) ---
    vf = P.voi_lut_normalize(np.array([0, 255]), 8)
    case("voi_lut_normalize_8bit", bool(np.allclose(vf, [0.0, 1.0])), expected="[0,1]")

    # --- presentation action table (all combos + conflicts + LUT sequence) ---
    presentation = [
        ("MONOCHROME1", None, False, "invert_once", 1, False),
        ("MONOCHROME2", None, False, "no_inversion", 0, False),
        ("MONOCHROME1", "IDENTITY", False, "conflict", None, True),
        ("MONOCHROME2", "IDENTITY", False, "no_inversion", 0, False),
        ("MONOCHROME1", "INVERSE", False, "invert_once", 1, False),
        ("MONOCHROME2", "INVERSE", False, "conflict", None, True),
        ("MONOCHROME2", None, True, "gap", None, True),
    ]
    for pi, shape, seq, exp_action, exp_inv, exp_gap in presentation:
        dec = P.presentation_polarity_decision(pi, shape, seq)
        cases.append({
            "case": f"presentation:{pi}+{shape or 'ABSENT'}+seq{seq}",
            "sop_class_uid": "SYNTHETIC",
            "photometric_interpretation": pi,
            "presentation_lut_shape": shape or "ABSENT",
            "presentation_lut_sequence_present": seq,
            "expected_project_action": exp_action,
            "expected_inversion_count": exp_inv,
            "actual_inversion_count": dec.inversion_count,
            "metadata_presentation_conflict": dec.metadata_conflict,
            "protocol_gap_detected": dec.protocol_gap,
            "test_status": "BLOCKED_PROTOCOL_REVIEW" if exp_gap else "PASS",
            "pass": (dec.action == exp_action and dec.inversion_count == exp_inv
                     and dec.protocol_gap == exp_gap),
        })

    # --- pixel padding ---
    m1 = P.build_padding_mask(np.array([[0, 5], [5, 9]]), 5, None)
    case("pixel_padding_value_mask",
         bool(np.array_equal(m1, [[False, True], [True, False]])), expected="==5")
    m2 = P.build_padding_mask(np.array([[0, 3], [7, 11]]), 2, 8)
    case("pixel_padding_range_mask_inclusive",
         bool(np.array_equal(m2, [[False, True], [True, False]])), expected="[2..8]")

    # --- multi-valued windows use index 0 ---
    wdec = P.classify_window("1000;2000", "400;800")
    case("multi_valued_window_index0",
         wdec.state == "valid" and wdec.center == 1000.0 and len(wdec.centers) == 2,
         expected="valid,index0=1000")

    # --- numpy.rint uint8 (round half to even) ---
    u = P.fraction_to_uint8(np.array([0.0, 0.5, 1.0]))
    case("numpy_rint_uint8_half_even", list(u) == [0, 128, 255], expected="[0,128,255]")

    out = {
        "phase_id": P.PHASE_ID,
        "disclaimer": "Synthetic conformity to the locked project protocol only; "
                      "NOT formal certification of complete DICOM Standard conformance.",
        "n_cases": len(cases),
        "all_pass": all(c["pass"] for c in cases),
        "cases": cases,
    }
    atomic_write_text(reports / "phase2D1B_pilot_synthetic_conformance.json",
                      strict_json_dumps(out) + "\n")
    md = ["# Phase 2D.1B-Pilot - Synthetic Transformation Conformance\n",
          "These synthetic tests validate implementation conformity to the "
          "locked project protocol. They do **not** constitute formal "
          "certification of complete DICOM Standard conformance.\n",
          f"- cases: {len(cases)}", f"- all_pass: {out['all_pass']}\n"]
    atomic_write_text(reports / "phase2D1B_pilot_synthetic_conformance.md",
                      "\n".join(md) + "\n")


def write_mapping(ctx, mapping_dir) -> None:
    rows = []
    for oid in ctx["selected_ids"]:
        tr = ctx["transform_records"][oid]
        im = ctx["id_to_coco"][oid]
        h = ctx["headers"][oid]
        for quality in (95, 100):
            fid = ctx["fidelity_by_key"][(oid, quality)]
            rows.append({
                "original_image_id": oid, "canonical_image_id": im["canonical_image_id"],
                "coco_image_id": im["id"], "dicom_relative_path": im["file_name"],
                "pilot_jpg_relative_path": f"images_jpg_pilot/q{quality}/train/{oid}.jpg",
                "source_dicom_sha256": tr["source_dicom_sha256"],
                "pre_jpeg_uint8_sha256": tr["pre_jpeg_uint8_sha256"],
                "reference_png_relative_path": f"images_jpg_pilot/reference_uint8/train/{oid}.png",
                "reference_png_byte_sha256": tr["reference_png_byte_sha256"],
                "reference_png_decoded_pixel_sha256": tr["reference_png_decoded_pixel_sha256"],
                "reference_png_exact_pixel_match": tr["reference_png_exact_pixel_match"],
                "output_jpg_sha256": fid["output_jpg_sha256"],
                "decoded_jpg_uint8_sha256": fid["decoded_jpg_uint8_sha256"],
                "protocol_version": P.EXPECTED_PROTOCOL_VERSION,
                "protocol_sha256": P.EXPECTED_PROTOCOL_SHA256,
                "jpeg_quality": quality, "decoder_backend": tr["decoder_backend"],
                "transfer_syntax_uid": h["TransferSyntaxUID"],
                "sop_class_uid": h["SOPClassUID"], "modality": h["Modality"],
                "rows": h["Rows"], "columns": h["Columns"],
                "bits_allocated": h["BitsAllocated"], "bits_stored": h["BitsStored"],
                "high_bit": h["HighBit"], "pixel_representation": h["PixelRepresentation"],
                "samples_per_pixel": h["SamplesPerPixel"],
                "photometric_interpretation": h["PhotometricInterpretation"],
                "number_of_frames_effective": h["NumberOfFrames_effective"],
                "modality_branch": tr["modality_branch"], "voi_branch": tr["voi_branch"],
                "presentation_lut_shape": h["PresentationLUTShape"],
                "presentation_lut_sequence_present": h["presentation_lut_sequence_present"],
                "presentation_inversion_applied": tr["presentation_inversion_applied"],
                "presentation_inversion_count": tr["presentation_inversion_count"],
                "presentation_metadata_conflict": False,
                "padding_present": tr["padding_present"],
                "padding_pixel_count": tr["padding_pixel_count"],
                "pre_jpeg_channel_count": 1, "pre_jpeg_mode": "L",
                "model_input_channel_adaptation_applied": False,
                "pixel_matrix_order_unchanged": True,
                "rotation_applied": False, "flip_applied": False,
                "transpose_applied": False, "exif_orientation_transform_applied": False,
            })
    write_csv(mapping_dir / "phase2D1B_pilot_dicom_to_jpg_mapping.csv",
              list(rows[0].keys()), rows)


# =========================================================================== #
# Stage 16: staging validation + TRUE atomic promotion                          #
# =========================================================================== #
def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def validate_staging(staging: Path, n_selected: int, n_annotations: int,
                     visual_expectations: Optional[Dict[str, Any]] = None) -> None:
    reports = staging / "reports"
    required = [
        "reports/phase2D1B_pilot_environment.json",
        "reports/phase2D1B_pilot_synthetic_conformance.json",
        "reports/phase2D1B_pilot_synthetic_conformance.md",
        "reports/phase2D1B_pilot_multi_window_audit.csv",
        "reports/phase2D1B_pilot_reference_renderer_concordance.csv",
        "reports/phase2D1B_pilot_reference_viewer_manifest.csv",
        "reports/phase2D1B_pilot_header_inventory.csv",
        "reports/phase2D1B_pilot_metadata_strata.csv",
        "reports/phase2D1B_pilot_selection.csv",
        "reports/phase2D1B_pilot_selection_coverage.csv",
        "reports/phase2D1B_pilot_fidelity_metrics.csv",
        "reports/phase2D1B_pilot_bbox_roi_metrics.csv",
        "reports/phase2D1B_pilot_quality_summary.csv",
        "reports/phase2D1B_pilot_quality_pairwise.csv",
        "reports/phase2D1B_pilot_geometry_validation.csv",
        "reports/phase2D1B_pilot_visual_audit_manifest.csv",
        "reports/phase2D1B_pilot_errors.csv",
        "reports/phase2D1B_pilot_validation.json",
        "reports/phase2D1B_pilot_validation.md",
        "reports/phase2D1B_pilot_decision_template.json",
        "reports/phase2D1B_pilot_decision_template.md",
        "image_mapping/phase2D1B_pilot_dicom_to_jpg_mapping.csv",
    ]
    for rel in required:
        if not (staging / rel).exists():
            raise P.UnsupportedInputError(f"staging missing artifact: {rel}")

    # ---- strict JSON: reject NaN / Infinity / -Infinity everywhere ----
    for jp in staging.rglob("*.json"):
        P.strict_json_loads(jp.read_text(encoding="utf-8"))

    # ---- row-count hard checks ----
    def nrows(name: str) -> int:
        return len(_read_csv_rows(reports / name))

    if nrows("phase2D1B_pilot_header_inventory.csv") != P.LOCKED_INPUT_COUNTS["images"]:
        raise P.UnsupportedInputError("header inventory rows != 4894")
    if nrows("phase2D1B_pilot_fidelity_metrics.csv") != n_selected * 2:
        raise P.UnsupportedInputError("fidelity rows != selected*2")
    if nrows("phase2D1B_pilot_geometry_validation.csv") != n_selected * 2:
        raise P.UnsupportedInputError("geometry rows != selected*2")
    if len(_read_csv_rows(staging / "image_mapping" /
                          "phase2D1B_pilot_dicom_to_jpg_mapping.csv")) != n_selected * 2:
        raise P.UnsupportedInputError("mapping rows != selected*2")
    if nrows("phase2D1B_pilot_bbox_roi_metrics.csv") != n_annotations * 2:
        raise P.UnsupportedInputError("roi rows != annotations*2")

    # ---- image files count ----
    q95 = {p.stem for p in (staging / "images_jpg_pilot" / "q95" / "train").glob("*.jpg")}
    q100 = {p.stem for p in (staging / "images_jpg_pilot" / "q100" / "train").glob("*.jpg")}
    ref = {p.stem for p in (staging / "images_jpg_pilot" / "reference_uint8" / "train").glob("*.png")}
    if len(q95) != n_selected or q95 != q100 or ref != q95:
        raise P.UnsupportedInputError("paired q95/q100/reference incomplete")

    # ---- required columns + non-empty hashes ----
    map_rows = _read_csv_rows(staging / "image_mapping" /
                              "phase2D1B_pilot_dicom_to_jpg_mapping.csv")
    for col in ("output_jpg_sha256", "decoded_jpg_uint8_sha256", "pre_jpeg_uint8_sha256",
                "reference_png_byte_sha256", "bits_stored", "sop_class_uid",
                "decoder_backend"):
        if col not in map_rows[0]:
            raise P.UnsupportedInputError(f"mapping missing column {col}")
    for r in map_rows:
        for col in ("output_jpg_sha256", "decoded_jpg_uint8_sha256",
                    "pre_jpeg_uint8_sha256", "reference_png_byte_sha256"):
            if not str(r.get(col, "")).strip():
                raise P.UnsupportedInputError(f"empty hash in mapping: {col}")
    fid_rows = _read_csv_rows(reports / "phase2D1B_pilot_fidelity_metrics.csv")
    for col in ("ssim", "jpg_bytes_per_pixel", "decoder_backend"):
        if col not in fid_rows[0]:
            raise P.UnsupportedInputError(f"fidelity missing column {col}")

    # ---- geometry checks all true (never empty), bbox scaling false ----
    for r in _read_csv_rows(reports / "phase2D1B_pilot_geometry_validation.csv"):
        if str(r.get("bbox_scaling_required")).strip().lower() not in ("false", "0"):
            raise P.UnsupportedInputError("bbox_scaling_required not false")
        for k, v in r.items():
            if k.endswith(("_unchanged", "_L", "_uint8", "_match", "absent_or_1")):
                # Boolean geometry flags MUST be true/1 - empty is not accepted.
                if str(v).strip().lower() not in ("true", "1"):
                    raise P.UnsupportedInputError(f"geometry flag not true: {k}={v!r}")

    # ---- synthetic conformance all_pass ----
    sc = P.strict_json_loads((reports / "phase2D1B_pilot_synthetic_conformance.json")
                             .read_text(encoding="utf-8"))
    if not sc.get("all_pass"):
        raise P.UnsupportedInputError("synthetic_conformance all_pass false")

    # ---- visual artifacts + manifest paths exist, in all 4 subdirs ----
    plots = staging / "plots" / "phase2D1B_pilot"
    man = _read_csv_rows(reports / "phase2D1B_pilot_visual_audit_manifest.csv")
    if not man:
        raise P.UnsupportedInputError("visual manifest empty")
    for r in man:
        ap = r.get("artifact_path", "")
        if not ap or not (plots / ap).exists():
            raise P.UnsupportedInputError(f"visual artifact missing: {ap}")
    for sub in ("full_image", "bbox_crops", "difference_heatmaps", "contact_sheets"):
        if not any((plots / sub).glob("*.png")):
            raise P.UnsupportedInputError(f"no visual files in {sub}")

    # ---- no hard errors in errors CSV ----
    for r in _read_csv_rows(reports / "phase2D1B_pilot_errors.csv"):
        if str(r.get("severity", "")).strip().lower() == "error":
            raise P.UnsupportedInputError("hard error present in errors CSV")

    # ---- decision template pending + validation invariants ----
    dt = P.strict_json_loads((reports / "phase2D1B_pilot_decision_template.json")
                             .read_text(encoding="utf-8"))
    if dt.get("final_jpeg_quality") is not None or dt.get("full_conversion_authorized"):
        raise P.UnsupportedInputError("decision template is not pending")
    val = P.strict_json_loads((reports / "phase2D1B_pilot_validation.json")
                              .read_text(encoding="utf-8"))
    if val.get("final_jpeg_quality") is not None:
        raise P.UnsupportedInputError("final_jpeg_quality not null")
    if val.get("phase_status") == "PASS":
        raise P.UnsupportedInputError("phase_status must never be PASS")
    for flag in ("jpg_training_representation_ready", "coco_jpg_training_annotation_ready",
                 "mmdetection_dataset_loading_ready", "empty_image_retention_ready",
                 "dataset_training_ready", "training_authorized"):
        if val.get(flag) is not False:
            raise P.UnsupportedInputError(f"readiness flag {flag} not false")

    # ---- visual coverage: required categories must not be silently dropped ----
    if visual_expectations is not None:
        roi_ann_map = {str(r.get("canonical_ann_id")): str(r.get("image_id"))
                       for r in _read_csv_rows(reports / "phase2D1B_pilot_bbox_roi_metrics.csv")}
        check_visual_coverage(man, visual_expectations, plots, roi_ann_map)


def check_visual_coverage(man: List[Dict[str, Any]], req: Dict[str, Any],
                          plots: Path, roi_ann_map: Dict[str, str]) -> None:
    """Independently verify the visual manifest satisfies required coverage.

    Enforces: image-level reason tokens present; No Finding coverage counted by
    UNIQUE image_id with DISTINCT metadata-stratum signatures; all warning
    images present; and each required annotation-level reason has a bbox crop
    whose (canonical_ann_id, image_id) agree across the expectation, the ROI
    evidence, and the manifest row, with the crop file on disk.

    ``roi_ann_map`` maps canonical_ann_id -> image_id from the ROI evidence.
    """
    reasons_all = " ".join(r.get("reason", "") for r in man)
    for expected, token in [
        (req.get("expect_extremum"), "extremum:"),
        (req.get("expect_worst_q95_whole_image"), "worst_q95_whole_image"),
        (req.get("expect_padding"), "padding_unusual"),
        (req.get("expect_saturation"), "saturation_unusual"),
    ]:
        if expected and token not in reasons_all:
            raise P.UnsupportedInputError(f"visual coverage missing: {token}")

    # No Finding: UNIQUE images + DISTINCT metadata-stratum signatures.
    nf_rows = [r for r in man if "no_finding_strata_diverse" in r.get("reason", "")]
    nf_images = {r.get("image_id", "") for r in nf_rows}
    nf_sigs = {r.get("stratum_signature", "") for r in nf_rows}
    if len(nf_images) < int(req.get("min_no_finding_unique", 0)):
        raise P.UnsupportedInputError(
            f"No Finding unique-image coverage {len(nf_images)} < "
            f"{req.get('min_no_finding_unique')} (rows must not be double-counted)")
    if len(nf_sigs) < int(req.get("min_no_finding_distinct_strata", 0)):
        raise P.UnsupportedInputError(
            f"No Finding distinct-strata coverage {len(nf_sigs)} < "
            f"{req.get('min_no_finding_distinct_strata')}")

    man_images = {r.get("image_id", "") for r in man}
    for wid in req.get("warning_image_ids", []):
        if wid not in man_images:
            raise P.UnsupportedInputError(f"visual coverage missing warning image {wid}")

    crops = [r for r in man if r.get("artifact_type") == "bbox_crop"]
    for ar in req.get("annotation_requests", []):
        aid = str(ar["canonical_ann_id"])
        token = ar["reason"]
        exp_img = str(ar.get("image_id"))
        # canonical_ann_id must exist in ROI evidence AND its image must agree.
        roi_img = roi_ann_map.get(aid)
        if roi_img is None:
            raise P.UnsupportedInputError(f"crop ann {aid} not present in ROI evidence")
        if exp_img != roi_img:
            raise P.UnsupportedInputError(
                f"annotation image mismatch: expected {exp_img} but ROI has {roi_img} "
                f"for ann {aid}")
        # Manifest crop must match BOTH canonical_ann_id and image_id.
        match = [r for r in crops
                 if str(r.get("canonical_ann_id")) == aid
                 and str(r.get("image_id")) == exp_img
                 and token in r.get("reason", "")]
        if not match:
            raise P.UnsupportedInputError(
                f"missing annotation crop for reason {token} ann {aid} image {exp_img}")
        if not (plots / match[0].get("artifact_path", "")).exists():
            raise P.UnsupportedInputError(f"crop file missing for ann {aid}")


def promote_atomic(staging: Path, overwrite: bool, _fail_hook=None, _mid_hook=None) -> None:
    """Transactional promotion with per-destination backup + full rollback.

    Phase A: copy every staged file to a sibling '<dest>.promote'.
    Phase B: for each destination, move any existing file to '<dest>.backup',
             then os.replace('<dest>.promote' -> dest).

    Rollback covers a failure at ANY point, including the exact gap BETWEEN
    ``os.replace(dest, backup)`` and ``os.replace(tmp, dest)``. The in-flight
    item's transaction state is tracked separately from the list of fully
    completed items so that a crash after the backup move but before the
    replacement still restores the prior destination byte-for-byte. On failure
    every completed replacement is also rolled back, and no '.promote'/'.backup'
    temporaries are left behind on either success or failure.

    ``_fail_hook(index)`` raises AFTER an item completes; ``_mid_hook(index,
    dest)`` raises in the gap between the backup move and the replacement. Both
    are test seams.
    """
    targets: List[Tuple[Path, Path]] = []
    for src in (staging / "reports").glob("*"):
        targets.append((src, REPORTS_DIR / src.name))
    for src in (staging / "image_mapping").glob("*"):
        targets.append((src, MAPPING_DIR / src.name))
    img_root = staging / "images_jpg_pilot"
    for src in img_root.rglob("*"):
        if src.is_file():
            targets.append((src, PILOT_OUT_DIR / src.relative_to(img_root)))
    plots_root = staging / "plots" / "phase2D1B_pilot"
    if plots_root.exists():
        for src in plots_root.rglob("*"):
            if src.is_file():
                targets.append((src, PLOTS_DIR / src.relative_to(plots_root)))

    if not overwrite:
        for _, dest in targets:
            if dest.exists():
                raise P.UnsupportedInputError(f"exists (use --overwrite): {dest}")

    # Phase A: copy to '<dest>.promote'.
    promoted: List[Tuple[Path, Path]] = []
    try:
        for src, dest in targets:
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".promote")
            shutil.copy2(src, tmp)
            promoted.append((tmp, dest))
    except Exception:
        for tmp, _ in promoted:
            if tmp.exists():
                tmp.unlink()
        raise

    # Phase B: backup + replace, with full rollback on failure.
    done: List[Tuple[Path, Path, bool]] = []  # completed (dest, backup, had_prior)
    current: Optional[Dict[str, Any]] = None   # in-flight item transaction state
    try:
        for idx, (tmp, dest) in enumerate(promoted):
            backup = dest.with_name(dest.name + ".backup")
            had_prior = dest.exists()
            # Begin the in-flight transaction BEFORE moving anything.
            current = {"dest": dest, "backup": backup, "had_prior": had_prior,
                       "backup_moved": False, "replaced": False}
            if had_prior:
                os.replace(dest, backup)            # (1) move prior aside
                current["backup_moved"] = True
            if _mid_hook is not None:
                _mid_hook(idx, dest)                # crash exactly in the gap
            os.replace(tmp, dest)                   # (2) place new output
            current["replaced"] = True
            done.append((dest, backup, had_prior))
            current = None
            if _fail_hook is not None:
                _fail_hook(idx)                     # crash after completion
    except Exception:
        # 1) Restore the in-flight item (covers failure between (1) and (2)).
        if current is not None:
            dest = current["dest"]
            backup = current["backup"]
            if current["replaced"] and dest.exists():
                dest.unlink()
            if current["backup_moved"] and backup.exists():
                os.replace(backup, dest)
        # 2) Roll back every fully completed replacement.
        for dest, backup, had_prior in reversed(done):
            if dest.exists():
                dest.unlink()
            if had_prior and backup.exists():
                os.replace(backup, dest)
        # 3) Remove any remaining '.promote' temporaries.
        for tmp, _ in promoted:
            if tmp.exists():
                tmp.unlink()
        raise
    # Success: drop all backups.
    for _, backup, had_prior in done:
        if had_prior and backup.exists():
            backup.unlink()


# =========================================================================== #
# Blocked-status emitters (no pixel decoding)                                    #
# =========================================================================== #
BLOCKED_DIR_NAME = "phase2D1B_pilot_blocked"


def write_blocked_report(status: Dict[str, Any], kind: str) -> Path:
    """Write a blocked-run report to a SEPARATE failure directory.

    Failure runs must never overwrite a prior VALID validation.json or any other
    promoted evidence. Each blocked report is written to
    ``reports/phase2D1B_pilot_blocked/<utc>_<kind>.json`` via an atomic write.
    """
    blocked_dir = REPORTS_DIR / BLOCKED_DIR_NAME
    blocked_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = blocked_dir / f"{ts}_{kind}.json"
    atomic_write_text(out, strict_json_dumps(status) + "\n")
    return out


def _base_blocked(pre, xcheck, decoder_env, forbidden) -> Dict[str, Any]:
    status = baseline_validation_status()
    status["protocol_sha256"] = pre["evidence"]["protocol_sha256"]
    status["coco_master_sha256"] = xcheck["coco_sha256"]
    status["forbidden_artifact_snapshot"] = forbidden
    return status


def emit_blocked(args, pre, xcheck, decoder_env, forbidden, reason, image_id) -> int:
    status = _base_blocked(pre, xcheck, decoder_env, forbidden)
    status.update({"phase_status": "BLOCKED", "structural_dod_candidate": False,
                   "protocol_review_required": True,
                   "structural_failure_reason": reason, "structural_failure_image_id": image_id})
    out = write_blocked_report(status, "structural_blocked")
    LOG.error("BLOCKED (structural): %s [%s]; no pixels decoded. Report: %s "
              "(prior valid evidence untouched).", reason, image_id, out)
    return 6


def emit_blocked_protocol_review(args, pre, xcheck, decoder_env, forbidden, reason, image_id) -> int:
    status = _base_blocked(pre, xcheck, decoder_env, forbidden)
    status.update({
        "phase_status": "BLOCKED_PROTOCOL_REVIEW", "structural_dod_candidate": False,
        "protocol_gap_detected": True, "protocol_review_required": True,
        "presentation_metadata_conflict_detected": "presentation_metadata_conflict" in reason,
        "presentation_lut_sequence_detected": "presentation_lut_sequence" in reason,
        "protocol_gap_reason": reason, "protocol_gap_image_id": image_id,
    })
    out = write_blocked_report(status, "protocol_review_blocked")
    LOG.error("BLOCKED_PROTOCOL_REVIEW: %s [%s]; no pixels decoded. Report: %s "
              "(prior valid evidence untouched).", reason, image_id, out)
    return 5


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except P.ProtocolGapError as gap:
        LOG.error("protocol_gap_detected (%s): %s", gap.reason, gap)
        return 5
    except P.Phase2D1BError as exc:
        LOG.error("hard fail (%s): %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
