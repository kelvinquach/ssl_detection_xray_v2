#!/usr/bin/env python3
"""Phase 2D.1A - Image Representation Protocol Decision.

DECISION-ONLY phase. This script does NOT touch any medical image, DICOM byte,
or produce any training artifact. It emits a single authoritative protocol
specification (PROTOCOL_SPEC) and renders three consistent, non-drifting views
of it (YAML + JSON + Markdown), after cross-checking the locked Phase 2D counts
against already-existing JSON / COCO evidence.

HARD CONSTRAINT - NO IMAGE / RAW-MEDICAL ACCESS OF ANY KIND:
    This script never opens, decodes, stats, or converts a raw medical image.
    It imports only the Python standard library plus PyYAML. It does not import
    any raw-medical-image reader, any computer-vision library, or any imaging
    library, and it never reads stored pixel data. `file_name`, `width` and
    `height` are read from existing COCO/JSON evidence ONLY, for count
    cross-checking.

Also forbidden here (and asserted false in every output): full conversion,
JPG dataset creation, coco_master_jpg.json creation, any split, training,
inference, pseudo-labels, threshold tuning, AP/mAP computation, test-set use,
and any edit of the canonical bboxes or coco_master.json.

The final JPEG quality is deliberately left `null` (PENDING PILOT). No numeric
fidelity threshold is locked in this phase.

Usage (Windows CMD):
    python scripts\\02D1A_image_representation_protocol.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except Exception as exc:  # pragma: no cover
    print(f"ERROR: PyYAML required but not importable: {exc!r}", file=sys.stderr)
    raise SystemExit(2)


# --------------------------------------------------------------------------- #
# Repository layout                                                            #
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[1]

EVIDENCE_COCO_MASTER = REPO_ROOT / "data" / "processed" / "coco" / "coco_master.json"
EVIDENCE_PHASE2D = REPO_ROOT / "reports" / "phase2D_coco_master_validation.json"
EVIDENCE_PHASE2A = REPO_ROOT / "reports" / "phase2A_dicom_bbox_validation.json"
EVIDENCE_PHASE2B = REPO_ROOT / "reports" / "phase2B_canonical_schema_validation.json"

OUT_YAML = REPO_ROOT / "configs" / "protocol" / "phase2D1_jpg_representation.yaml"
OUT_JSON = REPO_ROOT / "reports" / "phase2D1_image_representation_decision.json"
OUT_MD = REPO_ROOT / "reports" / "phase2D1_image_representation_decision.md"


# --------------------------------------------------------------------------- #
# Locked evidence (Phase 2D, CLOSED / PASS)                                    #
# --------------------------------------------------------------------------- #
LOCKED_COUNTS: Dict[str, int] = {
    "images": 4894,
    "abnormal_images": 4394,
    "no_finding_images": 500,
    "annotations": 36096,
    "categories": 14,
    "no_finding_annotations": 0,
}

# The 20 authoritative policy items that MUST be documented in this phase.
# Each entry is a top-level key of PROTOCOL_SPEC. Coverage == 20/20 is required.
REQUIRED_POLICY_ITEMS: Tuple[str, ...] = (
    "protocol_metadata",
    "artifact_roles",
    "dicom_decoding_policy",
    "pixel_padding_policy",
    "modality_transformation_policy",
    "voi_windowing_policy",
    "presentation_polarity_policy",
    "uint8_conversion_policy",
    "output_channel_policy",
    "jpeg_encoding",
    "geometry_bbox_policy",
    "filename_path_policy",
    "traceability_policy",
    "pilot_selection_policy",
    "fidelity_validation_policy",
    "final_quality_decision_rule",
    "thresholds_not_locked",
    "readiness_flags",
    "forbidden_actions",
    "locked_input_counts",
)

READINESS_FLAGS: Dict[str, bool] = {
    "jpg_training_representation_ready": False,
    "coco_jpg_training_annotation_ready": False,
    "mmdetection_dataset_loading_ready": False,
    "empty_image_retention_ready": False,
    "dataset_training_ready": False,
    "training_authorized": False,
}

FORBIDDEN_ACTIONS: Dict[str, bool] = {
    "full_conversion_run": False,
    "full_jpg_dataset_created": False,
    "coco_master_jpg_created": False,
    "split_created": False,
    "labeled_unlabeled_split_created": False,
    "training_started": False,
    "inference_run": False,
    "pseudo_labels_generated": False,
    "threshold_tuned": False,
    "ap_map_computed": False,
    "test_set_used": False,
    "canonical_bbox_modified": False,
    "coco_master_modified": False,
}


# --------------------------------------------------------------------------- #
# Single source of truth                                                       #
# --------------------------------------------------------------------------- #
def build_protocol_spec() -> Dict[str, Any]:
    """Return the authoritative PROTOCOL_SPEC.

    THIS DICTIONARY IS THE ONLY SOURCE OF TRUTH. YAML, JSON and Markdown are all
    rendered from it. Nothing downstream is written independently, which is what
    prevents protocol drift. All values are plain JSON/YAML-safe scalars, lists
    and dicts (no tuples), so the document round-trips through safe YAML/JSON.
    """
    spec: Dict[str, Any] = {
        # 7.1 --------------------------------------------------------------- #
        "protocol_metadata": {
            "phase_id": "2D.1A",
            "protocol_name": "image_representation_protocol_decision",
            "protocol_version": "1.0.0",
            "status": "decision_locked_pilot_pending",
            "seed": 2026,
            "gpt_review_status": "pending",
        },
        # 7.2 --------------------------------------------------------------- #
        "artifact_roles": {
            "DICOM": "immutable_raw_medical_source",
            "JPG": "processed_training_representation",
            "coco_master.json": "official_annotation_master",
            "coco_master_jpg.json": "path-only training derivative",
        },
        # 7.3 --------------------------------------------------------------- #
        "dicom_decoding_policy": {
            "documented_only": True,
            "executed_in_this_phase": False,
            "applies_to_phase": "2D.1B",
            "force_read": False,
            "single_frame_only": True,
            "samples_per_pixel_must_equal": 1,
            "allowed_photometric_interpretation": ["MONOCHROME1", "MONOCHROME2"],
            "unsupported_inputs": "hard_fail",
            "required_future_recording_fields": [
                "TransferSyntaxUID",
                "decoder_backend",
                "Rows",
                "Columns",
                "BitsAllocated",
                "BitsStored",
                "HighBit",
                "PixelRepresentation",
                "SamplesPerPixel",
                "PhotometricInterpretation",
                "NumberOfFrames",
            ],
        },
        # 7.4 --------------------------------------------------------------- #
        "pixel_padding_policy": {
            "build_padding_mask_from_stored_pixels": True,
            "use_fields_when_present": [
                "PixelPaddingValue",
                "PixelPaddingRangeLimit",
            ],
            "padding_must_not_influence_intensity_statistics": True,
            "final_padding_value_after_monochrome2_normalization": 0,
        },
        # 7.5 --------------------------------------------------------------- #
        "modality_transformation_policy": {
            "branch": {
                "if_modality_lut_sequence_present": "apply_modality_lut",
                "elif_rescale_slope_and_intercept_present": "apply_rescale",
                "else": "identity",
            },
            "do_not_apply_both_lut_and_rescale_sequentially": True,
            "only_one_of_rescale_slope_intercept_present": "hard_fail",
            "conflicting_or_ambiguous_modality_metadata": "hard_fail",
            "occurs_before_voi_windowing": True,
        },
        # 7.6 --------------------------------------------------------------- #
        "voi_windowing_policy": {
            "branch": {
                "if_voi_lut_sequence_exists": "prefer_voi_lut",
                "elif_valid_window_center_and_width_exist": "use_windowing",
                "else": "use_theoretical_modality_domain_range_fallback",
            },
            "selected_index": 0,
            "record_all_available_values": True,
            "respect_voi_lut_function": True,
            "direct_observed_per_image_min_max": "forbidden",
            "automatic_percentile_clipping": "forbidden",
            "fallback_basis": "theoretical_stored_or_modality_range",
            "fallback_must_not_use_per_image_array_min_max": True,
        },
        # 7.7 --------------------------------------------------------------- #
        "presentation_polarity_policy": {
            "branch": {
                "if_presentation_lut_shape_inverse": "invert_once",
                "elif_shape_absent_and_photometric_monochrome1": "invert_once",
                "else": "no_inversion",
            },
            "output_target": "MONOCHROME2_equivalent_polarity",
            "low_value": "dark",
            "high_value": "bright",
        },
        # 7.8 --------------------------------------------------------------- #
        "uint8_conversion_policy": {
            "steps": [
                "clip_using_theoretical_output_bounds",
                "linear_mapping_to_0_255",
                "round_using_numpy_rint",
                "final_clip_0_255",
                "cast_uint8",
            ],
            "nan_or_inf": "hard_fail",
        },
        # 7.9 --------------------------------------------------------------- #
        "output_channel_policy": {
            "jpg_storage": {
                "jpeg_mode": "L",
                "channels": 1,
                "dtype": "uint8",
            },
            "mmdetection_model_input": {
                "channels": 3,
                "replicate_grayscale_in_loader": True,
                "actual_validation_deferred_to": "2D.1C",
            },
        },
        # 7.10 -------------------------------------------------------------- #
        "jpeg_encoding": {
            "encoder": "Pillow",
            "quality_candidates": [95, 100],
            "final_quality": None,
            "final_quality_status": "pending_phase2D1B_pilot",
            "optimize": False,
            "progressive": False,
            "lossless_claim": "forbidden",
            "required_future_encoder_environment_recording": [
                "Python",
                "Pillow",
                "libjpeg",
                "pydicom",
                "numpy",
            ],
        },
        # 7.11 -------------------------------------------------------------- #
        "geometry_bbox_policy": {
            "resize": False,
            "crop": False,
            "rotation": False,
            "flip": False,
            "transpose": False,
            "preserve_width_and_height": True,
            "bbox_scaling_expected": False,
            "bbox_scaling_validated": False,
            "on_dimension_or_orientation_change": [
                "hard_fail",
                "do_not_automatically_scale_bbox",
            ],
        },
        # 7.12 -------------------------------------------------------------- #
        "filename_path_policy": {
            "jpg_root": "data/processed/images_jpg",
            "jpg_relative_file_name": "train/<image_id>.jpg",
            "coco_jpg_file_name": "train/<image_id>.jpg",
            "absolute_path_in_coco_jpg": "forbidden",
        },
        # 7.13 -------------------------------------------------------------- #
        "traceability_policy": {
            "future_mapping_target": "data/processed/image_mapping/dicom_to_jpg_mapping.csv",
            "required_future_fields": [
                "original_image_id",
                "canonical_image_id",
                "coco_image_id",
                "dicom_relative_path",
                "jpg_relative_path",
                "source_dicom_sha256",
                "pre_jpeg_uint8_sha256",
                "output_jpg_sha256",
                "protocol_version",
                "protocol_sha256",
                "transfer_syntax_uid",
                "decoder_backend",
                "rows",
                "columns",
                "jpeg_quality",
                "modality_branch",
                "voi_branch",
                "presentation_inversion_applied",
            ],
        },
        # 7.14 -------------------------------------------------------------- #
        "pilot_selection_policy": {
            "minimum_images": 64,
            "minimum_no_finding_images": 16,
            "tie_break_seed": 2026,
            "selection_unit": "image_id",
            "selection": "deterministic_coverage_first",
            "actual_image_ids_selected_in_this_phase": False,
            "must_cover": [
                "all_14_abnormal_classes",
                "minimum_and_maximum_dimensions_and_pixel_count",
                "smallest_and_largest_bbox",
                "all_photometric_interpretation_values",
                "all_transfer_syntax_patterns",
                "all_bits_stored_and_pixel_representation_patterns",
                "rescale_slope_intercept_patterns",
                "modality_lut_presence_and_absence",
                "voi_lut_presence_and_absence",
                "window_center_width_presence_and_absence",
                "single_and_multi_valued_windows",
                "presentation_lut_shape_patterns",
                "pixel_padding_value_presence_and_absence",
            ],
            "expansion_rule": (
                "if 64 images are insufficient, expand until all observed "
                "metadata strata are represented"
            ),
        },
        # 7.15 -------------------------------------------------------------- #
        "fidelity_validation_policy": {
            "jpeg_fidelity_reference": "pre_jpeg_uint8_image",
            "comparison_target": "decoded_jpg_image",
            "never_describe_raw_dicom_to_jpg_difference_as_jpeg_compression_error": True,
            "required_whole_image_metrics": [
                "MAE",
                "RMSE",
                "PSNR",
                "SSIM",
                "maximum_absolute_error",
                "p95_absolute_error",
                "p99_absolute_error",
            ],
            "required_bbox_roi_metrics": [
                "ROI_MAE",
                "ROI_PSNR",
                "ROI_SSIM",
                "ROI_maximum_absolute_error",
            ],
            "also_required": [
                "file_size",
                "compression_ratio_relative_to_pre_jpeg_uint8_bytes",
                "full_image_visual_audit",
                "bbox_crop_visual_audit",
                "difference_heatmap",
            ],
            "numeric_pass_thresholds_locked_in_this_phase": False,
        },
        # 7.16 -------------------------------------------------------------- #
        "final_quality_decision_rule": {
            "pilot_candidates": [95, 100],
            "quality_95_selectable_only_after": [
                "structural_checks_pass",
                "whole_image_metrics_pass",
                "bbox_roi_metrics_pass",
                "visual_audit_pass",
            ],
            "quality_100_selected_only_if": (
                "quality_95_shows_meaningful_lesion_region_degradation "
                "and quality_100_resolves_it"
            ),
            "if_both_candidates_fail": [
                "block_full_conversion",
                "reopen_image_format_decision",
                "do_not_silently_continue",
            ],
            "final_quality_must_remain_null_in_this_phase": True,
        },
        # Decision-only vs pilot-dependent thresholds --------------------- #
        "thresholds_not_locked": {
            "psnr_pass_threshold": None,
            "ssim_pass_threshold": None,
            "mae_pass_threshold": None,
            "roi_psnr_pass_threshold": None,
            "roi_ssim_pass_threshold": None,
            "note": "No numeric fidelity pass threshold is locked in Phase 2D.1A.",
        },
        # Section 8 ------------------------------------------------------- #
        "readiness_flags": dict(READINESS_FLAGS),
        "forbidden_actions": dict(FORBIDDEN_ACTIONS),
        # Section 3 ------------------------------------------------------- #
        "locked_input_counts": dict(LOCKED_COUNTS),
    }
    return spec


# --------------------------------------------------------------------------- #
# Canonicalization / fingerprinting                                            #
# --------------------------------------------------------------------------- #
def canonical_json(obj: Any) -> str:
    """Deterministic JSON serialization for hashing/comparison."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def protocol_sha256(spec: Dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Policy coverage                                                              #
# --------------------------------------------------------------------------- #
def check_policy_coverage(spec: Dict[str, Any]) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    documented = 0
    for key in REQUIRED_POLICY_ITEMS:
        present = key in spec
        value = spec.get(key)
        # "documented" means present and non-empty.
        is_documented = bool(present and value not in (None, {}, [], ""))
        if is_documented:
            documented += 1
        items.append({"item": key, "present": present, "documented": is_documented})
    return {
        "required_policy_items_total": len(REQUIRED_POLICY_ITEMS),
        "required_policy_items_documented": documented,
        "items": items,
    }


# --------------------------------------------------------------------------- #
# Evidence cross-check                                                         #
# --------------------------------------------------------------------------- #
def _load_json(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def crosscheck_locked_counts() -> Dict[str, Any]:
    """Cross-check LOCKED_COUNTS against existing JSON/COCO evidence.

    Reads only structured JSON/COCO evidence. Never touches an image. Any
    mismatch is a hard error; a missing non-critical source is a warning.
    """
    hard_errors: List[str] = []
    warnings: List[str] = []
    per_source: Dict[str, Any] = {}

    def compare(source: str, provided: Dict[str, Any]) -> None:
        mism: Dict[str, Any] = {}
        clean: Dict[str, Any] = {}
        for key, val in provided.items():
            if val is None:
                continue
            clean[key] = int(val)
            expected = LOCKED_COUNTS.get(key)
            if expected is not None and int(val) != int(expected):
                mism[key] = {"expected": expected, "found": int(val)}
        per_source[source] = {"provided": clean, "mismatches": mism}
        for key, d in mism.items():
            hard_errors.append(
                f"{source}: {key} mismatch (expected {d['expected']}, found {d['found']})"
            )

    # --- coco_master.json (critical) --------------------------------------
    coco = _load_json(EVIDENCE_COCO_MASTER)
    if coco is None:
        hard_errors.append(f"missing critical evidence: {EVIDENCE_COCO_MASTER}")
    else:
        images = coco.get("images", [])
        anns = coco.get("annotations", [])
        cats = coco.get("categories", [])
        abnormal = sum(1 for im in images if im.get("scope_label") == "abnormal")
        no_finding = len(images) - abnormal
        compare(
            "coco_master.json",
            {
                "images": len(images),
                "annotations": len(anns),
                "categories": len(cats),
                "abnormal_images": abnormal,
                "no_finding_images": no_finding,
            },
        )
        cat_names = {str(c.get("name", "")).strip().lower() for c in cats}
        for banned in ("no finding", "nofinding", "background", "normal", "__background__"):
            if banned in cat_names:
                hard_errors.append(f"coco_master.json: forbidden category present: {banned!r}")

    # --- phase2D validation json (critical) -------------------------------
    p2d = _load_json(EVIDENCE_PHASE2D)
    if p2d is None:
        hard_errors.append(f"missing critical evidence: {EVIDENCE_PHASE2D}")
    else:
        c = p2d.get("counts", {})
        compare(
            "phase2D_coco_master_validation.json",
            {
                "images": c.get("images"),
                "annotations": c.get("annotations"),
                "categories": c.get("categories"),
                "abnormal_images": c.get("abnormal_images"),
                "no_finding_images": c.get("no_finding_images"),
                "no_finding_annotations": c.get("no_finding_annotations"),
            },
        )

    # --- phase2A / phase2B (checked if present) ---------------------------
    p2a = _load_json(EVIDENCE_PHASE2A)
    if p2a is None:
        warnings.append(f"prior evidence not found (skipped): {EVIDENCE_PHASE2A}")
    else:
        compare(
            "phase2A_dicom_bbox_validation.json",
            {
                "images": p2a.get("expected_image_count"),
                "annotations": p2a.get("abnormal_bbox_rows_checked"),
                "abnormal_images": p2a.get("abnormal_images"),
                "no_finding_images": p2a.get("no_finding_images"),
            },
        )

    p2b = _load_json(EVIDENCE_PHASE2B)
    if p2b is None:
        warnings.append(f"prior evidence not found (skipped): {EVIDENCE_PHASE2B}")
    else:
        compare(
            "phase2B_canonical_schema_validation.json",
            {
                "images": p2b.get("canonical_image_rows"),
                "annotations": p2b.get("canonical_bbox_rows"),
                "categories": p2b.get("canonical_class_count"),
                "abnormal_images": p2b.get("abnormal_images"),
                "no_finding_images": p2b.get("no_finding_images"),
            },
        )

    return {"per_source": per_source, "hard_errors": hard_errors, "warnings": warnings}


# --------------------------------------------------------------------------- #
# Rendering                                                                    #
# --------------------------------------------------------------------------- #
def render_yaml_text(spec: Dict[str, Any]) -> str:
    """Render the protocol as safe YAML (no arbitrary Python objects)."""
    return yaml.safe_dump(spec, sort_keys=False, allow_unicode=True, default_flow_style=False)


def build_report(spec: Dict[str, Any], crosscheck: Dict[str, Any]) -> Dict[str, Any]:
    coverage = check_policy_coverage(spec)
    sha = protocol_sha256(spec)

    hard_errors: List[str] = list(crosscheck["hard_errors"])
    warnings: List[str] = list(crosscheck["warnings"])

    # Structural invariants that must hold in this decision-only phase.
    jpeg = spec["jpeg_encoding"]
    final_jpeg_quality_is_pending = bool(
        jpeg["final_quality"] is None
        and jpeg["final_quality_status"] == "pending_phase2D1B_pilot"
    )
    if not final_jpeg_quality_is_pending:
        hard_errors.append("final JPEG quality must remain null / pending in Phase 2D.1A")
    if jpeg["quality_candidates"] != [95, 100]:
        hard_errors.append("jpeg quality_candidates must equal [95, 100]")

    if coverage["required_policy_items_documented"] != coverage["required_policy_items_total"]:
        hard_errors.append("required policy coverage incomplete")

    if any(spec["readiness_flags"].values()):
        hard_errors.append("a readiness flag is not false")
    if any(spec["forbidden_actions"].values()):
        hard_errors.append("a forbidden action is not false")

    geo = spec["geometry_bbox_policy"]
    for k in ("resize", "crop", "rotation", "flip", "transpose"):
        if geo[k] is not False:
            hard_errors.append(f"geometry policy {k} must be false")
    if geo["bbox_scaling_validated"] is not False:
        hard_errors.append("bbox_scaling_validated must be false")

    if spec["voi_windowing_policy"]["direct_observed_per_image_min_max"] != "forbidden":
        hard_errors.append("direct per-image min-max must be forbidden")
    if spec["voi_windowing_policy"]["automatic_percentile_clipping"] != "forbidden":
        hard_errors.append("automatic percentile clipping must be forbidden")

    dod_pass_candidate = len(hard_errors) == 0
    phase_status = "OPEN_REVIEW_REQUIRED"  # never PASS in this phase

    report: Dict[str, Any] = {
        "phase_id": "2D.1A",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha,
        "locked_counts": dict(LOCKED_COUNTS),
        "protocol_summary": {
            "protocol_version": spec["protocol_metadata"]["protocol_version"],
            "status": spec["protocol_metadata"]["status"],
            "seed": spec["protocol_metadata"]["seed"],
            "jpeg_quality_candidates": jpeg["quality_candidates"],
            "final_jpeg_quality": jpeg["final_quality"],
            "final_jpeg_quality_status": jpeg["final_quality_status"],
        },
        "required_policy_coverage": coverage,
        "cross_output_consistency": {
            "protocol_sha256": sha,
            "cross_output_drift_count": None,  # filled in after rendering
            "yaml_reload_matches": None,
            "json_reload_matches": None,
            "markdown_embeds_fingerprint": None,
        },
        "evidence_crosscheck": crosscheck["per_source"],
        "readiness_flags": dict(spec["readiness_flags"]),
        "forbidden_actions": dict(spec["forbidden_actions"]),
        "final_jpeg_quality_is_pending": final_jpeg_quality_is_pending,
        "required_policy_items_total": coverage["required_policy_items_total"],
        "required_policy_items_documented": coverage["required_policy_items_documented"],
        "hard_errors": hard_errors,
        "warnings": warnings,
        "dod_pass_candidate": dod_pass_candidate,
        "gpt_review_status": "pending",
        "phase_status": phase_status,
        # Full spec embedded so JSON is a faithful, non-drifting view.
        "protocol_spec": spec,
    }
    return report


def render_json_text(report: Dict[str, Any]) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _fmt(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def render_markdown(spec: Dict[str, Any], report: Dict[str, Any]) -> str:
    sha = report["protocol_sha256"]
    jpeg = spec["jpeg_encoding"]
    L: List[str] = []
    A = L.append

    A("# Phase 2D.1A - Image Representation Protocol Decision\n")
    A("> **Final JPEG quality has not been selected.**  ")
    A("> **Full DICOM-to-JPG conversion remains locked.**  ")
    A("> **coco_master_jpg.json has not been created.**  ")
    A("> **Dataset is not training-ready.**  ")
    A("> **Training is not authorized.**\n")
    A(f"Protocol fingerprint (sha256): `{sha}`\n")

    A("## 1. Executive decision\n")
    A(
        "Phase 2D.1A locks the *image representation protocol* for converting the "
        "immutable raw DICOM source into a processed JPG training representation. "
        "This is a decision-only phase: no image is read, decoded, or written, and "
        "no numeric fidelity threshold is set. Two JPEG quality candidates (95 and "
        "100) are carried forward to a Phase 2D.1B pilot; the final quality remains "
        f"`null`. Protocol version `{spec['protocol_metadata']['protocol_version']}`, "
        f"status `{spec['protocol_metadata']['status']}`.\n"
    )

    A("## 2. Current gate and readiness\n")
    A(f"Phase status: `{report['phase_status']}` (GPT review pending).\n")
    A("| Readiness flag | Value |")
    A("| --- | --- |")
    for k, v in spec["readiness_flags"].items():
        A(f"| {k} | {_fmt(v)} |")
    A("")

    A("## 3. Locked input evidence\n")
    A("| Count | Value |")
    A("| --- | --- |")
    for k, v in report["locked_counts"].items():
        A(f"| {k} | {v} |")
    A("")
    A("Sources cross-checked: `data/processed/coco/coco_master.json`, "
      "`reports/phase2D_coco_master_validation.json`, "
      "`reports/phase2A_dicom_bbox_validation.json`, "
      "`reports/phase2B_canonical_schema_validation.json`.\n")

    A("## 4. Artifact roles\n")
    A("| Artifact | Role |")
    A("| --- | --- |")
    for k, v in spec["artifact_roles"].items():
        A(f"| `{k}` | {v} |")
    A("")

    A("## 5. Ordered pixel-transformation pipeline\n")
    A("The following order is authoritative for Phase 2D.1B and MUST NOT be reordered:\n")
    A("1. DICOM decode (single frame, MONOCHROME1/2 only)")
    A("2. Pixel padding mask build")
    A("3. Modality transformation (Modality LUT **or** rescale **or** identity)")
    A("4. VOI LUT / windowing (or theoretical modality-domain fallback)")
    A("5. Presentation polarity normalization to MONOCHROME2-equivalent")
    A("6. uint8 conversion (clip -> linear map -> rint -> clip -> cast)")
    A("7. Output channel handling (store 1-channel L; replicate to 3 at model load)")
    A("8. JPEG encoding (Pillow; quality pending pilot)\n")

    A("## 6. DICOM decoding policy\n")
    d = spec["dicom_decoding_policy"]
    A(f"Documented only, executed in phase `{d['applies_to_phase']}`. "
      f"`force_read = {_fmt(d['force_read'])}`, `single_frame_only = {_fmt(d['single_frame_only'])}`, "
      f"`SamplesPerPixel must equal {d['samples_per_pixel_must_equal']}`. "
      f"Allowed PhotometricInterpretation: {', '.join(d['allowed_photometric_interpretation'])}. "
      f"Unsupported inputs = {d['unsupported_inputs']}.\n")
    A("Required future recording: " + ", ".join(d["required_future_recording_fields"]) + ".\n")

    A("## 7. Modality LUT / Rescale policy\n")
    A("Branch: if a Modality LUT sequence is present, apply the Modality LUT; "
      "elif both RescaleSlope and RescaleIntercept are present, apply rescale; "
      "else identity. Do not apply both LUT and rescale sequentially. Exactly one "
      "of RescaleSlope/Intercept present = hard fail. Conflicting/ambiguous "
      "modality metadata = hard fail. Modality transformation occurs before "
      "VOI/windowing.\n")

    A("## 8. VOI LUT / Windowing policy\n")
    A("Branch: if a VOI LUT sequence exists, prefer the VOI LUT; elif valid "
      "WindowCenter and WindowWidth exist, use windowing; else use the theoretical "
      "modality-domain range fallback. Selected index = 0. Record all available "
      "values and respect VOILUTFunction. **Direct observed per-image min-max is "
      "forbidden.** **Automatic percentile clipping is forbidden.** The fallback is "
      "based on the theoretical stored/modality range, never per-image "
      "`arr.min()`/`arr.max()`.\n")

    A("## 9. Pixel padding and clipping\n")
    A("Build a padding mask from stored pixels using PixelPaddingValue and "
      "PixelPaddingRangeLimit when present. Padding pixels must not influence "
      "intensity statistics. Final padding value after MONOCHROME2 normalization = "
      "0. uint8 conversion clips using theoretical output bounds only.\n")

    A("## 10. Presentation LUT / MONOCHROME1 policy\n")
    A("If PresentationLUTShape == INVERSE, invert once; elif PresentationLUTShape "
      "is absent and PhotometricInterpretation == MONOCHROME1, invert once; else no "
      "inversion. Output target: MONOCHROME2-equivalent polarity (low = dark, "
      "high = bright).\n")

    A("## 11. uint8 conversion\n")
    A("Steps: " + " -> ".join(spec["uint8_conversion_policy"]["steps"]) +
      ". NaN/Inf = hard fail.\n")

    A("## 12. Output channel policy\n")
    A("JPG storage: JPEG mode L, one grayscale channel, uint8. MMDetection model "
      "input: three channels via grayscale replication in the loader; actual "
      "validation deferred to Phase 2D.1C.\n")

    A("## 13. JPEG candidates and pending decision\n")
    A(f"Encoder: {jpeg['encoder']}. Quality candidates: {jpeg['quality_candidates']}. "
      f"`final_quality = {_fmt(jpeg['final_quality'])}` "
      f"(`{jpeg['final_quality_status']}`). "
      f"`optimize = {_fmt(jpeg['optimize'])}`, `progressive = {_fmt(jpeg['progressive'])}`. "
      "Any lossless claim is forbidden. Future encoder environment to record: " +
      ", ".join(jpeg["required_future_encoder_environment_recording"]) + ".\n")

    A("## 14. Geometry and bbox preservation\n")
    g = spec["geometry_bbox_policy"]
    A(f"`resize = {_fmt(g['resize'])}`, `crop = {_fmt(g['crop'])}`, "
      f"`rotation = {_fmt(g['rotation'])}`, `flip = {_fmt(g['flip'])}`, "
      f"`transpose = {_fmt(g['transpose'])}`. Preserve width and height = "
      f"{_fmt(g['preserve_width_and_height'])}. `bbox_scaling_expected = "
      f"{_fmt(g['bbox_scaling_expected'])}`, `bbox_scaling_validated = "
      f"{_fmt(g['bbox_scaling_validated'])}`. Any dimension or orientation change = "
      "hard fail; do not automatically scale bbox.\n")

    A("## 15. Filename and path policy\n")
    fp2 = spec["filename_path_policy"]
    A(f"JPG root: `{fp2['jpg_root']}`. JPG relative file name: `{fp2['jpg_relative_file_name']}`. "
      f"COCO-JPG file_name: `{fp2['coco_jpg_file_name']}`. Absolute path in COCO-JPG: "
      f"{fp2['absolute_path_in_coco_jpg']}.\n")

    A("## 16. Traceability and hashes\n")
    t = spec["traceability_policy"]
    A(f"Future mapping target: `{t['future_mapping_target']}`.\n")
    A("Required future fields: " + ", ".join(t["required_future_fields"]) + ".\n")
    A(f"This protocol's fingerprint (sha256): `{sha}`.\n")

    A("## 17. Pilot selection protocol\n")
    p = spec["pilot_selection_policy"]
    A(f"Minimum images: {p['minimum_images']}; minimum No Finding images: "
      f"{p['minimum_no_finding_images']}; tie-break seed: {p['tie_break_seed']}; "
      f"selection unit: {p['selection_unit']}; selection: {p['selection']}. "
      f"Actual image IDs are NOT selected in Phase 2D.1A.\n")
    A("Coverage required across: " + ", ".join(p["must_cover"]) + ".\n")
    A(p["expansion_rule"] + ".\n")

    A("## 18. Fidelity validation\n")
    fv = spec["fidelity_validation_policy"]
    A("JPEG fidelity reference: pre-JPEG uint8 image; comparison: decoded JPG image. "
      "The raw DICOM-to-JPG difference must NEVER be described as JPEG compression "
      "error.\n")
    A("Whole-image metrics: " + ", ".join(fv["required_whole_image_metrics"]) + ".\n")
    A("BBox-ROI metrics: " + ", ".join(fv["required_bbox_roi_metrics"]) + ".\n")
    A("Also required: " + ", ".join(fv["also_required"]) + ".\n")
    A("No numeric PSNR/SSIM/MAE pass threshold is set in Phase 2D.1A.\n")

    A("## 19. Decision-only versus pilot-dependent fields\n")
    A("Locked now (decision-only): artifact roles, transformation branch logic and "
      "ordering, geometry/bbox preservation, path policy, quality candidates. "
      "Deferred to the pilot (2D.1B): final JPEG quality, observed metadata strata, "
      "numeric fidelity outcomes, and the selected pilot image IDs.\n")

    A("## 20. Thresholds not locked\n")
    A("No numeric PSNR, SSIM or MAE pass threshold is locked in this phase. "
      "`final_quality` must remain `null`.\n")

    A("## 21. Definition of Done\n")
    A(f"`required_policy_items_total = {report['required_policy_items_total']}`, "
      f"`required_policy_items_documented = {report['required_policy_items_documented']}`, "
      f"`cross_output_drift_count = {report['cross_output_consistency']['cross_output_drift_count']}`, "
      f"`final_jpeg_quality_is_pending = {_fmt(report['final_jpeg_quality_is_pending'])}`, "
      f"`hard_errors = {len(report['hard_errors'])}`, "
      f"`dod_pass_candidate = {_fmt(report['dod_pass_candidate'])}`.\n")

    A("## 22. Forbidden actions\n")
    A("| Forbidden action | Executed |")
    A("| --- | --- |")
    for k, v in spec["forbidden_actions"].items():
        A(f"| {k} | {_fmt(v)} |")
    A("")

    A("## 23. Remaining risks\n")
    A("Residual risks deferred to the pilot: unobserved TransferSyntax/Photometric "
      "strata, multi-valued windowing edge cases, lesion-region degradation at "
      "quality 95, and encoder-environment (libjpeg) variance. Each is mitigated by "
      "the coverage-first pilot and the fidelity metric suite before any full "
      "conversion.\n")

    A("## 24. Next gate\n")
    A("Phase 2D.1B (locked): implement the documented decoder + encoder on the "
      "coverage-first pilot, record all metadata strata, compute the fidelity "
      "metrics, and select the final JPEG quality. Full conversion stays blocked "
      "until GPT review concludes Phase 2D.1A PASS.\n")

    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# Drift detection                                                              #
# --------------------------------------------------------------------------- #
def compute_drift(
    spec: Dict[str, Any], yaml_text: str, json_text: str, md_text: str
) -> Dict[str, Any]:
    """All three views must be derived from one spec. Count mismatches."""
    canonical = protocol_sha256(spec)

    yaml_reload = yaml.safe_load(yaml_text)
    yaml_matches = protocol_sha256(yaml_reload) == canonical

    json_reload = json.loads(json_text)
    json_matches = protocol_sha256(json_reload.get("protocol_spec", {})) == canonical

    md_matches = canonical in md_text

    drift = sum(1 for ok in (yaml_matches, json_matches, md_matches) if not ok)
    return {
        "cross_output_drift_count": drift,
        "yaml_reload_matches": yaml_matches,
        "json_reload_matches": json_matches,
        "markdown_embeds_fingerprint": md_matches,
    }


# --------------------------------------------------------------------------- #
# Atomic multi-file write                                                      #
# --------------------------------------------------------------------------- #
def _validate_rendered(targets: Dict[Path, str]) -> List[str]:
    """Re-parse rendered payloads before promotion. Returns list of errors."""
    errors: List[str] = []
    for path, text in targets.items():
        suffix = path.suffix.lower()
        try:
            if suffix in (".yaml", ".yml"):
                doc = yaml.safe_load(text)
                if not isinstance(doc, dict):
                    errors.append(f"{path.name}: YAML did not parse to a mapping")
            elif suffix == ".json":
                json.loads(text)
            elif suffix == ".md":
                if not text.strip():
                    errors.append(f"{path.name}: empty markdown")
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"{path.name}: re-parse failed: {exc!r}")
    return errors


def atomic_write_all(
    targets: Dict[Path, str], simulate_validation_failure: bool = False
) -> Tuple[bool, List[str]]:
    """Write every target via temp file -> validate -> atomic replace.

    On any validation failure: no valid previous output is replaced, every
    temporary file is removed, and (False, errors) is returned. Only when ALL
    targets validate are they promoted with os.replace().
    """
    tmp_paths: Dict[Path, Path] = {}
    errors: List[str] = []
    try:
        for path, text in targets.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                dir=str(path.parent), prefix=f".{path.stem}_", suffix=path.suffix + ".tmp"
            )
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
            tmp_paths[path] = Path(tmp_name)

        errors = _validate_rendered({path: targets[path] for path in targets})
        if simulate_validation_failure:
            errors.append("simulated validation failure")

        if errors:
            # Do NOT promote. Previous outputs are left untouched.
            return False, errors

        # All good: promote atomically.
        for path, tmp in tmp_paths.items():
            os.replace(str(tmp), str(path))
        tmp_paths.clear()
        return True, []
    finally:
        # Always clean up any temp file that was not promoted.
        for tmp in tmp_paths.values():
            try:
                Path(tmp).unlink(missing_ok=True)
            except Exception:  # pragma: no cover
                pass


# --------------------------------------------------------------------------- #
# Console summary                                                              #
# --------------------------------------------------------------------------- #
def print_console_summary(report: Dict[str, Any], spec: Dict[str, Any]) -> None:
    jpeg = spec["jpeg_encoding"]
    lc = report["locked_counts"]
    rows = [
        ("Phase", "2D.1A"),
        ("Status", report["phase_status"]),
        ("Protocol version", spec["protocol_metadata"]["protocol_version"]),
        ("Locked images", lc["images"]),
        ("Locked annotations", lc["annotations"]),
        ("Locked categories", lc["categories"]),
        ("JPEG candidates", jpeg["quality_candidates"]),
        ("Final JPEG quality", "PENDING PILOT"),
        ("Direct min-max allowed", False),
        (
            "Resize/crop/rotation",
            f"{spec['geometry_bbox_policy']['resize']} / "
            f"{spec['geometry_bbox_policy']['crop']} / "
            f"{spec['geometry_bbox_policy']['rotation']}",
        ),
        ("Full conversion run", spec["forbidden_actions"]["full_conversion_run"]),
        ("COCO-JPG created", spec["forbidden_actions"]["coco_master_jpg_created"]),
        ("Dataset training-ready", spec["readiness_flags"]["dataset_training_ready"]),
        ("Training authorized", spec["readiness_flags"]["training_authorized"]),
        ("Hard errors", len(report["hard_errors"])),
        ("Warnings", len(report["warnings"])),
        ("DoD pass candidate", report["dod_pass_candidate"]),
        ("GPT review status", "PENDING"),
    ]
    for label, value in rows:
        print(f"{label:<28}: {value}")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2D.1A - Image Representation Protocol Decision (decision-only)."
    )
    p.add_argument(
        "--repo-root",
        default=None,
        help="Override repository root (defaults to the script's parent's parent).",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    parse_args(argv)  # accepted for parity; paths are module-level constants.

    spec = build_protocol_spec()

    # 1) Cross-check locked counts against existing evidence (no image access).
    crosscheck = crosscheck_locked_counts()

    # 2) Build the report (structural invariants + hard-error accumulation).
    report = build_report(spec, crosscheck)

    # 3) Render the three views from the SAME spec/report.
    yaml_text = render_yaml_text(spec)
    md_text = render_markdown(spec, report)
    # JSON is rendered after drift so the drift result is embedded.
    provisional_json = render_json_text(report)

    # 4) Drift detection across the three views.
    drift = compute_drift(spec, yaml_text, provisional_json, md_text)
    report["cross_output_consistency"].update(drift)
    if drift["cross_output_drift_count"] != 0:
        report["hard_errors"].append("cross-output drift detected")
        report["dod_pass_candidate"] = False

    # Re-render JSON and Markdown now that drift is known.
    json_text = render_json_text(report)
    md_text = render_markdown(spec, report)

    # 5) Atomic multi-file write: temp -> validate -> replace.
    targets: Dict[Path, str] = {
        OUT_YAML: yaml_text,
        OUT_JSON: json_text,
        OUT_MD: md_text,
    }
    ok, write_errors = atomic_write_all(targets)
    if not ok:
        for e in write_errors:
            print(f"ERROR: output validation failed: {e}", file=sys.stderr)
        print(
            "ERROR: previous valid outputs left untouched; temporary files removed.",
            file=sys.stderr,
        )
        return 1

    # 6) Console summary.
    print_console_summary(report, spec)

    # Exit non-zero if any hard error was recorded (outputs are still written).
    return 0 if not report["hard_errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
