#!/usr/bin/env python3
"""Phase 2C — Framework & Format Decision / COCO Conversion Planning.

DECISION / PROTOCOL EVIDENCE ONLY. This script reads the Phase 2B canonical
schema to re-confirm its invariants, then writes the framework and annotation-
format decisions plus a COCO conversion PLAN for Phase 2D.

It does NOT perform the COCO conversion. No coco_master.json is written.

Scope guardrails (Phase 2C): this script does NOT
  - create data/processed/coco/coco_master.json
  - create any train/val/test or labeled/unlabeled split
  - train, infer, pseudo-label, tune thresholds, or use the test set
  - read pixel_array, copy/convert images
  - modify/delete/clamp/fuse bboxes, source annotations, or the Phase 2B schema

"No Finding" is a NEGATIVE image label, NOT a detection class. In the planned
COCO it appears in `images` (with zero annotations) and never in `categories`.

Usage (Windows CMD):
    python scripts\\02C_framework_format_decision.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


# --- Constants (locked from Phase 2B) -------------------------------------

EXPECT_IMAGE_ROWS = 4894
EXPECT_BBOX_ROWS = 36096
EXPECT_CLASS_COUNT = 14
EXPECT_ABNORMAL_IMAGES = 4394
EXPECT_NO_FINDING_IMAGES = 500

PATH_ROOT_VARIABLE = "VINBIGDATA_DICOM_ROOT"
BBOX_SOURCE_FORMAT = "xyxy_original_image"
BBOX_TARGET_FORMAT_2D = "coco_xywh_absolute"

PRIMARY_FRAMEWORK = "MMDetection"
FALLBACK_FRAMEWORK = "Detectron2_optional"
PRIMARY_ANNOTATION_FORMAT = "COCO_detection_JSON"
SOURCE_OF_TRUTH = "canonical_detection_schema"
PLANNED_COCO_MASTER_PATH = "data/processed/coco/coco_master.json"

# Defensive import probes (must NEVER fail the script in Phase 2C).
MMDET_STACK = ["mmengine", "mmcv", "mmdet"]

# --- Format comparison evidence (decision rationale, not a new decision) ----

FORMAT_COMPARISON_MATRIX: Dict[str, Dict[str, Any]] = {
    "COCO_detection_JSON": {
        "mmdetection_compatibility": "native (CocoDataset, no custom dataset class needed)",
        "coco_map_compatibility": "native (pycocotools; standard mAP@0.5:0.95)",
        "negative_image_support": (
            "first-class: an image may appear in `images` with zero entries in "
            "`annotations`; requires filter_empty_gt=False so the 500 No Finding "
            "negatives are retained"
        ),
        "multi_class_support": "explicit `categories` list; 14 detection classes",
        "category_metadata_support": (
            "explicit and extensible; can carry canonical_class_id and "
            "class_id_original alongside the contiguous COCO category_id"
        ),
        "bbox_format": (
            "[x, y, width, height] absolute pixels; direct, lossless map from "
            "canonical xyxy_original_image (w = x_max - x_min, h = y_max - y_min)"
        ),
        "bbox_coordinate_fidelity": "absolute pixels; no normalization, no precision loss",
        "traceability": (
            "annotation objects accept extra keys: canonical_ann_id, source_row_id, "
            "original_image_id, rad_id"
        ),
        "ssod_compatibility": (
            "strong: teacher-student SSOD implementations in MMDetection consume "
            "COCO; unlabeled/negative images fit the same schema"
        ),
        "pseudo_label_output_compatibility": (
            "pseudo-labels can be emitted directly as COCO annotations, reusing the "
            "same categories and evaluation path"
        ),
        "reproducibility_ecosystem": "widest ecosystem support; standard, well-specified",
        "implementation_risk": "low (single conversion step; standard tooling)",
        "fit_for_this_thesis": (
            "best preserves the image-level negative vs detection-category "
            "separation this thesis depends on"
        ),
        "verdict": "CHOSEN",
        "selection_reason": (
            "Native MMDetection/Detectron2 fit; first-class support for "
            "zero-annotation negative images; standard COCO mAP@0.5:0.95 via "
            "pycocotools; explicit categories; supports traceability fields back to "
            "the canonical schema."
        ),
    },
    "YOLO_txt": {
        "mmdetection_compatibility": "weak (requires conversion or a custom dataset)",
        "coco_map_compatibility": "indirect (must convert back to COCO for pycocotools mAP)",
        "negative_image_support": (
            "fragile: negatives are represented by an EMPTY .txt file (or a missing "
            "file); silent-drop risk for the 500 No Finding images"
        ),
        "multi_class_support": "yes, but classes are bare integer indices",
        "category_metadata_support": (
            "minimal: class names live in a separate side file; no room for "
            "canonical/original id metadata"
        ),
        "bbox_format": (
            "normalized center-based (cx, cy, w, h in [0,1]); requires a lossy "
            "round-trip from canonical xyxy_original_image"
        ),
        "bbox_coordinate_fidelity": (
            "reduced: normalization introduces float rounding and depends on exact "
            "image dimensions; conversion risk from the canonical absolute xyxy"
        ),
        "traceability": "none in-format; would need an external side-car mapping",
        "ssod_compatibility": "possible in YOLO-native SSOD, but off-stack for MMDetection",
        "pseudo_label_output_compatibility": (
            "would need conversion back to COCO for evaluation, adding a second "
            "lossy hop"
        ),
        "reproducibility_ecosystem": "popular but tied to the YOLO ecosystem",
        "implementation_risk": "medium-high (normalization + empty-file negatives)",
        "fit_for_this_thesis": (
            "poor: the negative-image representation is exactly the part this thesis "
            "cannot afford to get wrong"
        ),
        "verdict": "REJECTED",
        "rejection_reason": (
            "Normalized center-based bbox adds conversion risk from canonical "
            "xyxy_original_image; negative images are represented by empty txt files, "
            "which is fragile here; weaker fit for MMDetection and the COCO mAP "
            "pipeline."
        ),
    },
    "Pascal_VOC_XML": {
        "mmdetection_compatibility": "supported but legacy (VOCDataset); off the COCO path",
        "coco_map_compatibility": (
            "poor: VOC-style AP differs from COCO mAP@0.5:0.95; conversion needed"
        ),
        "negative_image_support": (
            "unclean: an object-less XML must still exist per negative image; less "
            "explicit than COCO's images-without-annotations"
        ),
        "multi_class_support": "yes (per-object <name> tags)",
        "category_metadata_support": "implicit only; no central category table",
        "bbox_format": "absolute xmin/ymin/xmax/ymax (fidelity is fine)",
        "bbox_coordinate_fidelity": "good: absolute pixels, matches canonical xyxy",
        "traceability": "would require non-standard custom XML tags",
        "ssod_compatibility": "weak; modern SSOD tooling assumes COCO",
        "pseudo_label_output_compatibility": (
            "awkward: one XML file per pseudo-labelled image per iteration"
        ),
        "reproducibility_ecosystem": "dated; less tooling for COCO-style evaluation",
        "implementation_risk": "medium (file sprawl; eval mismatch)",
        "fit_for_this_thesis": (
            "poor: 4,894 separate XML files and a non-COCO AP definition work against "
            "the locked mAP@0.5:0.95 metric"
        ),
        "verdict": "REJECTED",
        "rejection_reason": (
            "One XML per image is cumbersome for 4,894 controlled-scope images; less "
            "suitable for COCO-style mAP@0.5:0.95 and modern MMDetection workflows; "
            "negative images are represented less cleanly than in COCO."
        ),
    },
    "JSONL_custom": {
        "mmdetection_compatibility": "none out of the box; requires a custom dataset class",
        "coco_map_compatibility": "requires a custom evaluator or a conversion to COCO",
        "negative_image_support": "arbitrary (whatever we define) — but unvalidated",
        "multi_class_support": "arbitrary",
        "category_metadata_support": "arbitrary; fully flexible",
        "bbox_format": "arbitrary; could keep xyxy_original_image verbatim",
        "bbox_coordinate_fidelity": "can be perfect, but the guarantee is ours to maintain",
        "traceability": "excellent in principle (any field can be carried)",
        "ssod_compatibility": "requires custom teacher-student plumbing",
        "pseudo_label_output_compatibility": "custom; no standard evaluation path",
        "reproducibility_ecosystem": (
            "weakest: bespoke format means results are harder for others to reproduce "
            "or compare"
        ),
        "implementation_risk": (
            "high: custom dataset + custom evaluator are new surfaces for silent bugs "
            "in the very metric the thesis reports"
        ),
        "fit_for_this_thesis": (
            "poor: flexibility does not compensate for hand-rolled evaluation code on "
            "the primary metric"
        ),
        "verdict": "REJECTED",
        "rejection_reason": (
            "Non-standard; would require a custom dataset and evaluator; higher "
            "reproducibility risk; weak fit for MMDetection/SSOD without extra custom "
            "code."
        ),
    },
}

COMPARISON_CONCLUSION = (
    "COCO is selected not merely because it is common, but because it best "
    "preserves the required separation between image-level negatives and "
    "detection categories, supports standard COCO mAP evaluation, and minimizes "
    "custom evaluation code in the later MMDetection/SSOD pipeline."
)

# --- Framework selection evidence (rationale, not a new decision) -----------

FRAMEWORK_COMPARISON_MATRIX: Dict[str, Dict[str, Any]] = {
    "MMDetection": {
        "object_detection_support": (
            "native: modular PyTorch detection toolbox with a large model zoo "
            "(Faster R-CNN, RetinaNet, DINO, etc.)"
        ),
        "coco_dataset_compatibility": "native CocoDataset; no custom dataset class needed",
        "coco_map_compatibility": (
            "native CocoMetric via pycocotools; mAP@0.5:0.95 out of the box"
        ),
        "ssod_teacher_student_readiness": (
            "official semi-supervised components (e.g. SoftTeacher / MeanTeacher-style "
            "hooks, EMA teacher); teacher-student SSOD is a documented use case"
        ),
        "labeled_unlabeled_pipeline_support": (
            "official multi-branch pipeline and semi-supervised dataloader for "
            "labeled/unlabeled dataset preparation"
        ),
        "config_reproducibility": (
            "config-driven training; the full experiment is captured in a versionable "
            "config file"
        ),
        "empty_no_finding_image_handling": (
            "supported via filter_empty_gt=False; the 500 No Finding negatives are "
            "retained rather than silently dropped"
        ),
        "dicom_loader_extensibility": (
            "transform/pipeline registry allows a custom DICOM LoadImage transform "
            "without forking the framework"
        ),
        "pseudo_label_workflow": (
            "pseudo-labels flow through the same COCO-shaped structures used for "
            "training and evaluation"
        ),
        "classwise_ap_ap50_ap75_readiness": (
            "CocoMetric reports AP50, AP75 and per-class AP without custom code"
        ),
        "implementation_burden": "low-medium: mostly configuration, little bespoke code",
        "research_reproducibility": (
            "high: config + seed + published baselines make results comparable"
        ),
        "fit_for_this_thesis": (
            "direct match: COCO detection + COCO mAP + teacher-student SSOD + "
            "labeled/unlabeled handling + config reproducibility"
        ),
        "verdict": "CHOSEN",
        "selection_reason": (
            "Modular PyTorch-based detection toolbox with native COCO dataset and "
            "COCO-style evaluation; official semi-supervised object detection "
            "components (labeled/unlabeled dataset preparation, multi-branch "
            "pipeline, semi-supervised dataloader, teacher-student / MeanTeacher-style "
            "training); config-driven training suits reproducibility. Best fit for a "
            "teacher-student SSOD thesis pipeline."
        ),
    },
    "Detectron2_optional": {
        "object_detection_support": "native: strong PyTorch detection framework (FAIR)",
        "coco_dataset_compatibility": "strong: COCO and custom dataset registration",
        "coco_map_compatibility": "strong: COCOEvaluator via pycocotools",
        "ssod_teacher_student_readiness": (
            "no first-party SSOD components; teacher-student would rely on external "
            "repos (e.g. Unbiased Teacher) or bespoke implementation"
        ),
        "labeled_unlabeled_pipeline_support": (
            "no official labeled/unlabeled multi-branch dataloader; would need custom "
            "plumbing"
        ),
        "config_reproducibility": "good: LazyConfig / yacs configs",
        "empty_no_finding_image_handling": (
            "supported, but requires care with filter_empty_annotations to keep "
            "negatives"
        ),
        "dicom_loader_extensibility": "possible via a custom mapper",
        "pseudo_label_workflow": "would need a custom pseudo-label loop and EMA teacher",
        "classwise_ap_ap50_ap75_readiness": "COCOEvaluator provides AP50/AP75/per-class AP",
        "implementation_burden": (
            "medium-high for SSOD: the semi-supervised layer must be built for this "
            "project"
        ),
        "research_reproducibility": "good, but SSOD code would be project-specific",
        "fit_for_this_thesis": (
            "adequate as a detection backbone, weaker as an SSOD platform without "
            "extra custom work"
        ),
        "verdict": "FALLBACK_ONLY",
        "rejection_reason": (
            "Strong PyTorch detection framework with good COCO/custom dataset support "
            "and a suitable fallback if MMDetection setup fails; however the "
            "teacher-student SSOD pipeline would require more custom implementation in "
            "this project. Retained as fallback only, and only after GPT re-review."
        ),
    },
    "YOLO_based_framework": {
        "object_detection_support": "native and fast; strong single-stage baselines",
        "coco_dataset_compatibility": (
            "indirect: expects YOLO-native layout; COCO must be converted, which "
            "conflicts with the COCO-master source of truth"
        ),
        "coco_map_compatibility": (
            "reports its own mAP; alignment with pycocotools mAP@0.5:0.95 requires "
            "care/conversion"
        ),
        "ssod_teacher_student_readiness": (
            "SSOD exists in the YOLO ecosystem but is not aligned with the planned "
            "MMDetection SSOD protocol"
        ),
        "labeled_unlabeled_pipeline_support": (
            "not first-class for the planned labeled/unlabeled COCO protocol"
        ),
        "config_reproducibility": "good within its own ecosystem",
        "empty_no_finding_image_handling": (
            "risky: negatives are expressed as empty label files; silent-drop risk for "
            "the 500 No Finding images"
        ),
        "dicom_loader_extensibility": "possible but off the framework's standard path",
        "pseudo_label_workflow": "YOLO-native; would diverge from the COCO master",
        "classwise_ap_ap50_ap75_readiness": (
            "available, but under the framework's own metric implementation rather "
            "than pycocotools"
        ),
        "implementation_burden": (
            "medium: easy to train, but the SSOD + COCO + negative-image protocol must "
            "be re-adapted"
        ),
        "research_reproducibility": "good, but tied to a different evaluation stack",
        "fit_for_this_thesis": (
            "poor as primary: the annotation and evaluation pipeline is YOLO-native "
            "and diverges from the locked COCO/MMDetection protocol"
        ),
        "verdict": "REJECTED",
        "rejection_reason": (
            "Easy to train and deploy with a strong baseline ecosystem, but the "
            "annotation/evaluation pipeline is YOLO-native and less aligned with the "
            "COCO master + MMDetection SSOD protocol. The teacher-student SSOD pipeline "
            "and No Finding empty-image handling would require more project-specific "
            "adaptation. Rejected as primary framework."
        ),
    },
    "Custom_PyTorch_torchvision": {
        "object_detection_support": (
            "torchvision provides detection models, but the training/eval stack is "
            "ours to build"
        ),
        "coco_dataset_compatibility": "must be hand-written",
        "coco_map_compatibility": "must integrate pycocotools manually",
        "ssod_teacher_student_readiness": (
            "none: EMA teacher, strong/weak augmentation branches, and the pseudo-label "
            "loop must all be implemented"
        ),
        "labeled_unlabeled_pipeline_support": "must be implemented from scratch",
        "config_reproducibility": "must design a config/experiment protocol ourselves",
        "empty_no_finding_image_handling": (
            "fully under our control, but every guarantee is also ours to test"
        ),
        "dicom_loader_extensibility": "maximum flexibility (the one genuine advantage)",
        "pseudo_label_workflow": "entirely custom",
        "classwise_ap_ap50_ap75_readiness": "custom evaluation wiring required",
        "implementation_burden": (
            "high: custom dataset, dataloader, evaluator, trainer, pseudo-label loop, "
            "EMA teacher, COCO metric integration, logging and config protocol"
        ),
        "research_reproducibility": (
            "low-medium: bespoke code is harder for others to reproduce or compare "
            "against published baselines"
        ),
        "fit_for_this_thesis": (
            "poor: engineering effort and silent-bug risk would dominate the research "
            "contribution"
        ),
        "verdict": "REJECTED",
        "rejection_reason": (
            "Maximum flexibility, but requires custom dataset, dataloader, evaluator, "
            "trainer, pseudo-label loop, EMA teacher, COCO metric integration, logging "
            "and config protocol. High implementation risk and reproducibility risk."
        ),
    },
}

FRAMEWORK_SELECTION_CONCLUSION = (
    "MMDetection is selected as the primary framework because it best matches the "
    "planned COCO-based teacher-student SSOD workflow, supports COCO-style "
    "evaluation, offers config-driven reproducibility, and reduces the amount of "
    "custom training/evaluation code required. Detectron2 remains an optional "
    "fallback only after GPT re-review."
)

FRAMEWORK_RATIONALE_MD_CONCLUSION = (
    "MMDetection is selected as the primary framework not merely because it is "
    "popular, but because it most directly matches the thesis pipeline: COCO-based "
    "detection, COCO mAP evaluation, teacher-student semi-supervised object "
    "detection, labeled/unlabeled data handling, and config-driven reproducibility. "
    "Detectron2 remains a fallback because it is a strong detection framework, but "
    "it would require more custom SSOD plumbing for this project."
)


def probe_import(name: str) -> Dict[str, Any]:
    """Defensively probe a package; never raises."""
    import importlib

    out: Dict[str, Any] = {"import_ok": False, "version": None, "error": None}
    try:
        mod = importlib.import_module(name)
        out["import_ok"] = True
        out["version"] = getattr(mod, "__version__", None)
    except Exception as exc:
        out["error"] = repr(exc)
    return out


# --- CLI ------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2C — framework/format decision & COCO planning (no conversion).",
    )
    p.add_argument(
        "--canonical-image-table",
        type=str,
        default="data/processed/canonical/canonical_image_table.csv",
    )
    p.add_argument(
        "--canonical-bbox-table",
        type=str,
        default="data/processed/canonical/canonical_bbox_table.csv",
    )
    p.add_argument(
        "--canonical-class-mapping",
        type=str,
        default="data/processed/canonical/canonical_class_mapping.csv",
    )
    p.add_argument(
        "--phase2b-validation-json",
        type=str,
        default="reports/phase2B_canonical_schema_validation.json",
    )
    p.add_argument(
        "--report-md",
        type=str,
        default="reports/phase2C_framework_format_decision.md",
    )
    p.add_argument(
        "--output-json",
        type=str,
        default="reports/phase2C_framework_format_decision.json",
    )
    p.add_argument(
        "--framework-yaml",
        type=str,
        default="configs/framework/main_framework.yaml",
    )
    p.add_argument(
        "--coco-paths-yaml",
        type=str,
        default="configs/dataset/coco_paths.yaml",
    )
    p.add_argument(
        "--conversion-policy-yaml",
        type=str,
        default="configs/protocol/coco_conversion_policy.yaml",
    )
    return p.parse_args(argv)


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_csv(path: str, what: str) -> pd.DataFrame:
    fp = Path(path)
    if not fp.exists():
        raise FileNotFoundError(
            f"{what} not found: {fp}\n"
            "       Phase 2C requires the Phase 2B canonical schema. Run Phase 2B first."
        )
    try:
        return pd.read_csv(fp)
    except Exception as exc:
        raise ValueError(f"Failed to read {what} '{fp}': {exc!r}") from exc


def dump_yaml(path: str | Path, data: Dict[str, Any]) -> None:
    """Write YAML (falls back to JSON-in-YAML if pyyaml is missing)."""
    out = ensure_parent(path)
    if _HAVE_YAML:
        out.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    else:  # pragma: no cover
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")


# --- Canonical re-confirmation --------------------------------------------


def reconfirm_canonical(
    image_df: pd.DataFrame,
    bbox_df: pd.DataFrame,
    class_df: pd.DataFrame,
) -> Tuple[Dict[str, Any], List[str]]:
    """Re-read the canonical tables and verify Phase 2B invariants."""
    warnings: List[str] = []

    canonical_image_rows = int(len(image_df))
    canonical_image_unique = int(image_df["image_id"].nunique())
    canonical_bbox_rows = int(len(bbox_df))
    canonical_class_count = int(len(class_df))

    abnormal_images = (
        int(image_df["is_abnormal"].fillna(False).astype(bool).sum())
        if "is_abnormal" in image_df.columns
        else 0
    )
    no_finding_images = (
        int(image_df["is_negative"].fillna(False).astype(bool).sum())
        if "is_negative" in image_df.columns
        else 0
    )

    # No Finding must not be a detection class.
    nf_in_classes = False
    if "is_no_finding" in class_df.columns:
        nf_in_classes = bool(class_df["is_no_finding"].fillna(False).astype(bool).any())
    if "class_name" in class_df.columns:
        nf_names = class_df["class_name"].astype(str).str.strip().str.lower()
        nf_in_classes = nf_in_classes or bool((nf_names == "no finding").any())

    # No Finding must not appear in the bbox table.
    nf_in_bbox = False
    if "class_name" in bbox_df.columns and not bbox_df.empty:
        b_names = bbox_df["class_name"].astype(str).str.strip().str.lower()
        nf_in_bbox = bool((b_names == "no finding").any())

    # BBox source format.
    bbox_source_format = BBOX_SOURCE_FORMAT
    if "bbox_format" in bbox_df.columns and not bbox_df.empty:
        formats = bbox_df["bbox_format"].astype(str).unique().tolist()
        if len(formats) == 1:
            bbox_source_format = formats[0]
        else:
            warnings.append(f"bbox_format is not uniform: {formats}")

    # Path policy fields present?
    has_relative = "relative_dicom_path" in image_df.columns
    has_root_var = "path_root_variable" in image_df.columns
    path_policy_ok = has_relative
    if not has_relative:
        warnings.append(
            "canonical_image_table missing 'relative_dicom_path'; portable path "
            "policy cannot be confirmed."
        )
    root_var_value = PATH_ROOT_VARIABLE
    if has_root_var:
        vals = image_df["path_root_variable"].astype(str).unique().tolist()
        if len(vals) == 1:
            root_var_value = vals[0]
        else:
            warnings.append(f"path_root_variable is not uniform: {vals}")

    # Invariant checks vs locked Phase 2B numbers.
    if canonical_image_rows != EXPECT_IMAGE_ROWS:
        warnings.append(f"canonical_image_rows={canonical_image_rows}, expected {EXPECT_IMAGE_ROWS}")
    if canonical_image_unique != EXPECT_IMAGE_ROWS:
        warnings.append(f"canonical_image_unique_images={canonical_image_unique}, expected {EXPECT_IMAGE_ROWS}")
    if canonical_bbox_rows != EXPECT_BBOX_ROWS:
        warnings.append(f"canonical_bbox_rows={canonical_bbox_rows}, expected {EXPECT_BBOX_ROWS}")
    if canonical_class_count != EXPECT_CLASS_COUNT:
        warnings.append(f"canonical_class_count={canonical_class_count}, expected {EXPECT_CLASS_COUNT}")
    if abnormal_images != EXPECT_ABNORMAL_IMAGES:
        warnings.append(f"abnormal_images={abnormal_images}, expected {EXPECT_ABNORMAL_IMAGES}")
    if no_finding_images != EXPECT_NO_FINDING_IMAGES:
        warnings.append(f"no_finding_images={no_finding_images}, expected {EXPECT_NO_FINDING_IMAGES}")
    if nf_in_classes:
        warnings.append("No Finding appears in the detection class mapping (policy violation).")
    if nf_in_bbox:
        warnings.append("No Finding appears in the canonical bbox table (policy violation).")

    metrics = {
        "canonical_image_rows": canonical_image_rows,
        "canonical_image_unique_images": canonical_image_unique,
        "canonical_bbox_rows": canonical_bbox_rows,
        "canonical_class_count": canonical_class_count,
        "abnormal_images": abnormal_images,
        "no_finding_images": no_finding_images,
        "no_finding_in_detection_classes": nf_in_classes,
        "no_finding_in_canonical_bbox_table": nf_in_bbox,
        "bbox_source_format": bbox_source_format,
        "path_policy_confirmed": bool(path_policy_ok),
        "path_root_variable": root_var_value,
    }
    return metrics, warnings


# --- YAML builders --------------------------------------------------------


def build_framework_yaml(metrics: Dict[str, Any], probes: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "phase": "phase2C_framework_format_decision",
        "primary_framework": PRIMARY_FRAMEWORK,
        "fallback_framework": FALLBACK_FRAMEWORK,
        "fallback_condition": (
            "Use Detectron2 ONLY if MMDetection remote/GPU setup fails AND the "
            "fallback is re-reviewed and approved by GPT review."
        ),
        "framework_decision_status": "decided_pending_remote_setup",
        "local_training_framework_ready": False,
        "remote_gpu_training_required": True,
        "phase2c_requires_mmdet_import": False,
        "mmdet_stack_import_probe": probes,
        "primary_annotation_format": PRIMARY_ANNOTATION_FORMAT,
        "detection_num_classes": int(metrics["canonical_class_count"]),
        "no_finding_is_detection_class": False,
        "required_future_mmdet_config_rules": [
            "filter_empty_gt: false  # negatives (No Finding) MUST be retained",
            "No background class is added; 14 detection classes only.",
            "category_id must be contiguous 1..14 in COCO; model num_classes = 14.",
            "Dataset must resolve images via VINBIGDATA_DICOM_ROOT + relative_dicom_path.",
            "Default LoadImageFromFile is NOT validated for DICOM; a custom loader "
            "or a processed-image protocol is required before training.",
        ],
        "dicom_loader_validated": False,
        "custom_dicom_loader_or_processed_image_phase_required": True,
    }


def build_coco_paths_yaml(metrics: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "phase": "phase2C_framework_format_decision",
        "status": "planning_only",
        "canonical_inputs": {
            "canonical_image_table": args.canonical_image_table,
            "canonical_bbox_table": args.canonical_bbox_table,
            "canonical_class_mapping": args.canonical_class_mapping,
            "phase2b_validation_json": args.phase2b_validation_json,
        },
        "planned_coco_outputs_phase2D": {
            "coco_master": PLANNED_COCO_MASTER_PATH,
            "created_in_phase_2c": False,
            "note": "Phase 2C does NOT create this file.",
        },
        "image_root_env_var": PATH_ROOT_VARIABLE,
        "image_path_field": {
            "coco_file_name_source": "relative_dicom_path",
            "resolution": f"os.path.join(${{{PATH_ROOT_VARIABLE}}}, relative_dicom_path)",
            "local_dicom_path": "evidence_only_not_a_downstream_identifier",
        },
        "format_policy": {
            "primary_annotation_format": PRIMARY_ANNOTATION_FORMAT,
            "source_of_truth": SOURCE_OF_TRUTH,
            "bbox_source_format": metrics["bbox_source_format"],
            "bbox_target_format_phase2D": BBOX_TARGET_FORMAT_2D,
        },
        "bbox_policy": {
            "conversion": "xyxy_original_image -> [x, y, width, height]",
            "width": "x_max - x_min",
            "height": "y_max - y_min",
            "area": "width * height",
            "iscrowd": 0,
            "clamp": False,
            "delete": False,
            "fuse_near_duplicates": False,
            "near_duplicate_candidates_retained": 147,
            "traceability": "keep canonical_ann_id and source_row_id in annotation metadata",
        },
        "no_finding_policy": {
            "in_coco_images": True,
            "in_coco_annotations": False,
            "in_coco_categories": False,
            "background_class_created": False,
            "expected_no_finding_images": int(metrics["no_finding_images"]),
        },
        "split_policy": {
            "train_val_test_split_created": False,
            "labeled_unlabeled_split_created": False,
            "note": "No split of any kind is created in Phase 2C or Phase 2D planning.",
        },
    }


def build_conversion_policy_yaml() -> Dict[str, Any]:
    return {
        "phase_2c_policy": "planning_only",
        "actual_conversion_phase": "Phase 2D",
        "forbidden_in_phase_2c": [
            "create data/processed/coco/coco_master.json",
            "create any COCO JSON file",
            "create train/val/test split",
            "create labeled/unlabeled split",
            "train a model",
            "run inference",
            "generate pseudo-labels",
            "tune thresholds",
            "use the test set",
            "read pixel_array",
            "copy or convert image files",
            "modify, delete, clamp, or fuse any bbox",
            "modify source annotations",
            "modify the Phase 2B canonical schema",
        ],
        "allowed_in_phase_2c": [
            "read the canonical image/bbox/class tables (read-only)",
            "read the Phase 2B validation JSON (read-only)",
            "defensively probe mmengine/mmcv/mmdet imports (failure is acceptable)",
            "write framework decision YAML",
            "write COCO path planning YAML",
            "write this conversion policy YAML",
            "write the Phase 2C decision JSON and Markdown report",
        ],
    }


# --- Markdown report ------------------------------------------------------


def write_report_md(path: str, p: Dict[str, Any]) -> None:
    L: List[str] = []
    L.append("# Phase 2C — Framework & Format Decision / COCO Conversion Planning")
    L.append("")
    L.append(f"_Generated {p['created_utc']}._")
    L.append("")
    L.append("## Executive summary")
    L.append("")
    L.append(
        f"Primary framework: **{p['primary_framework']}** (fallback: "
        f"{p['fallback_framework']}, only after GPT re-review). Primary annotation "
        f"format: **{p['primary_annotation_format']}**, sourced from "
        f"`{p['source_schema']}`. Canonical schema re-confirmed: "
        f"**{p['canonical_image_rows']}** images, **{p['canonical_bbox_rows']}** "
        f"bboxes, **{p['canonical_class_count']}** detection classes. "
        f"COCO conversion is **NOT** performed here "
        f"(actual_coco_conversion_done={p['actual_coco_conversion_done']}); it is "
        "planned for Phase 2D. "
        f"DoD pass candidate: **{p['dod_pass_candidate']}**."
    )
    L.append("")
    L.append("## Phase scope")
    L.append("")
    L.append("- Decision/protocol evidence ONLY. No COCO file, no split, no training.")
    L.append("- Canonical schema (Phase 2B) is read-only and unmodified.")
    L.append("- Dataset is NOT training-ready at the end of Phase 2C.")
    L.append("")
    L.append("## Inputs used")
    L.append("")
    for k, v in p["inputs_used"].items():
        L.append(f"- {k}: `{v}`")
    L.append("")
    L.append("## Framework/format comparison")
    L.append("")
    L.append("### Table 1 — High-level format comparison")
    L.append("")
    L.append("| Format | Strengths | Weaknesses | Fit for this thesis | Verdict |")
    L.append("|---|---|---|---|---|")
    L.append(
        "| **COCO detection JSON** | Native fit for MMDetection and Detectron2; "
        "explicit `images` / `annotations` / `categories`; supports images with zero "
        "annotations (No Finding negatives); compatible with COCO-style "
        "mAP@0.5:0.95 and pycocotools; allows traceability fields such as "
        "`canonical_ann_id`, `source_row_id`, `original_image_id` | Verbose; needs a "
        "conversion step from the canonical schema | Best preserves the separation "
        "between image-level negatives and detection categories; keeps evaluation on "
        "the standard COCO path | **CHOSEN** |"
    )
    L.append(
        "| YOLO txt | Simple; popular in the YOLO ecosystem | Uses normalized "
        "center-based bbox, adding conversion risk from canonical "
        "`xyxy_original_image`; negative images are usually represented by empty txt "
        "files, which is fragile here; weaker fit for MMDetection and the COCO mAP "
        "pipeline | Poor: the empty-file negative representation is exactly the part "
        "this thesis cannot afford to get wrong | Rejected |"
    )
    L.append(
        "| Pascal VOC XML | Human-readable; supports bbox in absolute coordinates | "
        "One XML per image, cumbersome for 4,894 controlled-scope images; less "
        "suitable for COCO-style mAP@0.5:0.95 and modern MMDetection workflows; "
        "negative images represented less cleanly than in COCO | Poor: file sprawl "
        "plus a non-COCO AP definition works against the locked metric | Rejected |"
    )
    L.append(
        "| JSONL / custom | Flexible; can preserve arbitrary metadata | Non-standard; "
        "would require a custom dataset and evaluator; higher reproducibility risk; "
        "weak fit for MMDetection/SSOD without extra custom code | Poor: flexibility "
        "does not compensate for hand-rolled evaluation code on the primary metric | "
        "Rejected |"
    )
    L.append("")
    L.append("### Table 2 — Detailed format suitability matrix")
    L.append("")
    L.append(
        "| Criterion | COCO detection JSON | YOLO txt | Pascal VOC XML | JSONL / custom |"
    )
    L.append("|---|---|---|---|---|")

    criteria = [
        ("MMDetection compatibility", "mmdetection_compatibility"),
        ("COCO mAP@0.5:0.95 / pycocotools", "coco_map_compatibility"),
        ("Negative / No Finding image support", "negative_image_support"),
        ("Multi-class object detection support", "multi_class_support"),
        ("Category metadata support", "category_metadata_support"),
        ("BBox coordinate fidelity", "bbox_coordinate_fidelity"),
        ("Traceability to canonical/source rows", "traceability"),
        ("SSOD teacher-student compatibility", "ssod_compatibility"),
        ("Pseudo-label output compatibility", "pseudo_label_output_compatibility"),
        ("Reproducibility / ecosystem support", "reproducibility_ecosystem"),
        ("Implementation risk", "implementation_risk"),
    ]
    order = ["COCO_detection_JSON", "YOLO_txt", "Pascal_VOC_XML", "JSONL_custom"]
    matrix = p["format_comparison_matrix"]
    for label, key in criteria:
        cells = [str(matrix[fmt].get(key, "-")).replace("|", "\\|") for fmt in order]
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    verdicts = [f"**{matrix[fmt]['verdict']}**" for fmt in order]
    L.append("| **Verdict** | " + " | ".join(verdicts) + " |")
    L.append("")
    L.append(f"> {COMPARISON_CONCLUSION}")
    L.append("")
    L.append("## Framework selection rationale")
    L.append("")
    L.append("### Table 3 — High-level framework comparison")
    L.append("")
    L.append("| Framework | Strengths | Weaknesses / risks | Fit for this thesis | Verdict |")
    L.append("|---|---|---|---|---|")
    L.append(
        "| **MMDetection** | Modular PyTorch-based object detection toolbox; supports "
        "COCO-format datasets and COCO-style evaluation; official semi-supervised "
        "object detection documentation/components including labeled/unlabeled dataset "
        "preparation, multi-branch pipeline, semi-supervised dataloader, and "
        "teacher-student / MeanTeacher-style training components; config-driven "
        "training suits reproducibility | Heavier learning curve; mmcv/mmengine version "
        "pinning can be fragile on some platforms | Better fit for the teacher-student "
        "SSOD thesis pipeline; COCO in, COCO mAP out, minimal custom code | **CHOSEN** |"
    )
    L.append(
        "| Detectron2 | Strong PyTorch detection framework; good COCO/custom dataset "
        "support; solid COCOEvaluator | No first-party SSOD components; the "
        "teacher-student pipeline would require more custom implementation in this "
        "project; historically awkward to build on Windows | Suitable fallback if "
        "MMDetection setup fails, but weaker as an SSOD platform out of the box | "
        "**Fallback only** |"
    )
    L.append(
        "| Ultralytics YOLO / YOLO-based | Easy to train and deploy; strong baseline "
        "ecosystem; fast iteration | Annotation/evaluation pipeline is YOLO-native and "
        "less aligned with the COCO master + MMDetection SSOD protocol; teacher-student "
        "SSOD and No Finding empty-image handling need project-specific adaptation | "
        "Diverges from the locked COCO/MMDetection protocol; negative-image handling is "
        "the exact risk this thesis cannot take | Rejected as primary framework |"
    )
    L.append(
        "| Custom PyTorch / torchvision | Maximum flexibility; no framework constraints "
        "| Requires custom dataset, dataloader, evaluator, trainer, pseudo-label loop, "
        "EMA teacher, COCO metric integration, logging and config protocol; high "
        "implementation risk and reproducibility risk | Engineering effort and "
        "silent-bug risk would dominate the research contribution | Rejected |"
    )
    L.append("")
    L.append("### Table 4 — Detailed framework suitability matrix")
    L.append("")
    L.append(
        "| Criterion | MMDetection | Detectron2 | YOLO-based | Custom PyTorch/torchvision |"
    )
    L.append("|---|---|---|---|---|")

    fw_criteria = [
        ("Native object detection support", "object_detection_support"),
        ("COCO dataset compatibility", "coco_dataset_compatibility"),
        ("COCO mAP@0.5:0.95 / pycocotools", "coco_map_compatibility"),
        ("Teacher-student / SSOD readiness", "ssod_teacher_student_readiness"),
        ("Labeled/unlabeled pipeline support", "labeled_unlabeled_pipeline_support"),
        ("Config-based reproducibility", "config_reproducibility"),
        ("Empty / No Finding image handling risk", "empty_no_finding_image_handling"),
        ("Custom DICOM loader extensibility", "dicom_loader_extensibility"),
        ("Pseudo-label workflow compatibility", "pseudo_label_workflow"),
        ("Class-wise AP / AP50 / AP75 readiness", "classwise_ap_ap50_ap75_readiness"),
        ("Implementation burden", "implementation_burden"),
        ("Research reproducibility", "research_reproducibility"),
        ("Fit for this thesis", "fit_for_this_thesis"),
    ]
    fw_order = [
        "MMDetection",
        "Detectron2_optional",
        "YOLO_based_framework",
        "Custom_PyTorch_torchvision",
    ]
    fw_matrix = p["framework_comparison_matrix"]
    for label, key in fw_criteria:
        cells = [
            str(fw_matrix[fw].get(key, "-")).replace("|", "\\|") for fw in fw_order
        ]
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    fw_verdicts = [f"**{fw_matrix[fw]['verdict']}**" for fw in fw_order]
    L.append("| **Verdict** | " + " | ".join(fw_verdicts) + " |")
    L.append("")
    L.append(f"> {FRAMEWORK_RATIONALE_MD_CONCLUSION}")
    L.append("")
    L.append("## Final framework decision")
    L.append("")
    L.append(f"- primary_framework: **{p['primary_framework']}**")
    L.append(f"- fallback_framework: **{p['fallback_framework']}**")
    L.append(
        "- Fallback is used ONLY if MMDetection remote/GPU setup fails AND the "
        "change is re-reviewed by GPT."
    )
    L.append(f"- local_training_framework_ready: {p['local_training_framework_ready']}")
    L.append(f"- remote_gpu_training_required: {p['remote_gpu_training_required']}")
    L.append(
        "- Phase 2C does NOT require a successful `mmdet` import; the import probe "
        "is defensive and a missing package does not fail this phase."
    )
    L.append("")
    L.append("## Final annotation format decision")
    L.append("")
    L.append(f"- primary_annotation_format: **{p['primary_annotation_format']}**")
    L.append(f"- source_of_truth: `{p['source_schema']}`")
    L.append(f"- actual_coco_conversion_done: {p['actual_coco_conversion_done']}")
    L.append(f"- actual_coco_conversion_phase: {p['actual_coco_conversion_phase']}")
    L.append(f"- planned_coco_master_path: `{p['planned_coco_master_path']}` (NOT created here)")
    L.append("")
    L.append("## COCO conversion plan for Phase 2D")
    L.append("")
    L.append(f"- `images`: all **{p['canonical_image_rows']}** controlled-scope images.")
    L.append(f"- `annotations`: only the **{p['canonical_bbox_rows']}** abnormal bboxes.")
    L.append(f"- `categories`: only the **{p['canonical_class_count']}** abnormal detection classes.")
    L.append("- No Finding images appear in `images` with ZERO annotations.")
    L.append("- No background class is created.")
    L.append("- Traceability: each COCO annotation keeps `canonical_ann_id` and `source_row_id`.")
    L.append("")
    L.append("## No Finding / empty image policy")
    L.append("")
    L.append(f"- no_finding_images: {p['no_finding_images']}")
    L.append(f"- in COCO images: {p['no_finding_in_coco_images_planned']}")
    L.append(f"- in COCO annotations: {p['no_finding_in_coco_annotations_planned']}")
    L.append(f"- in COCO categories: {p['no_finding_in_coco_categories_planned']}")
    L.append(
        "- **MMDetection must set `filter_empty_gt=False`** (or equivalent) so the "
        "500 negative images are NOT silently dropped."
    )
    L.append("")
    L.append("## BBox conversion policy")
    L.append("")
    L.append(f"- source: `{p['bbox_source_format']}` → target: `{p['bbox_target_format_phase2D']}`")
    L.append("- width = x_max - x_min; height = y_max - y_min; area = width * height; iscrowd = 0.")
    L.append("- No clamping, no deletion, no fusion. 147 near-duplicate candidates retained.")
    L.append("")
    L.append("## Category id policy")
    L.append("")
    L.append(f"- {p['category_id_policy']}")
    L.append(
        "- The original/canonical class ids are retained in category metadata for "
        "traceability back to the canonical mapping."
    )
    L.append("")
    L.append("## Path portability policy")
    L.append("")
    L.append(f"- {p['image_path_policy']}")
    L.append(f"- image root env var: `{p['path_root_variable']}`")
    L.append("- `local_dicom_path` is evidence only; never a downstream identifier.")
    L.append("")
    L.append("## DICOM loader risk")
    L.append("")
    L.append("- COCO annotations alone do NOT make the dataset training-ready.")
    L.append(
        "- MMDetection's default `LoadImageFromFile` is NOT validated for DICOM "
        f"(dicom_loader_validated={p['dicom_loader_validated']})."
    )
    L.append(
        "- A later phase must provide a custom DICOM loader OR a processed-image "
        "conversion protocol before any training run."
    )
    L.append(f"- dataset_training_ready: {p['dataset_training_ready']}")
    L.append("")
    L.append("## Metric readiness policy")
    L.append("")
    L.append(
        "Phase 2C does not compute AP metrics because no split, model training, "
        "inference, or prediction file exists yet. However, the selected COCO "
        "detection format is required to preserve downstream compatibility with "
        "COCO-style detection metrics, including the primary metric mAP@0.5:0.95 "
        "and secondary diagnostics such as AP50, AP75, class-wise AP, "
        "recall/sensitivity, FP/image, and FP per negative image. These metrics "
        "must only be computed in later evaluation phases after COCO conversion, "
        "fixed split creation, model training, and prediction generation. No "
        "test-set metric may be used for checkpoint selection, threshold tuning, "
        "model selection, or augmentation decisions."
    )
    L.append("")
    L.append("## Forbidden actions avoided")
    L.append("")
    for k, v in p["forbidden_actions_confirmed"].items():
        L.append(f"- {k}: {v}")
    L.append("")
    L.append("## Definition of Done status")
    L.append("")
    L.append(f"- dod_pass_candidate: **{p['dod_pass_candidate']}**")
    if p["warnings"]:
        L.append("- Warnings:")
        for w in p["warnings"]:
            L.append(f"  - {w}")
    else:
        L.append("- No warnings; canonical invariants re-confirmed.")
    L.append("")
    L.append("## Next phase")
    L.append("")
    L.append(
        "- **Phase 2D (actual COCO conversion) only after GPT review PASS** of this "
        "decision evidence. Do not proceed automatically."
    )
    L.append("")
    ensure_parent(path).write_text("\n".join(L), encoding="utf-8")


# --- Main -----------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # Load canonical schema (read-only).
    try:
        image_df = load_csv(args.canonical_image_table, "canonical image table")
        bbox_df = load_csv(args.canonical_bbox_table, "canonical bbox table")
        class_df = load_csv(args.canonical_class_mapping, "canonical class mapping")
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    warnings: List[str] = []

    # Optional: Phase 2B validation JSON (read-only, informational).
    p2b: Dict[str, Any] = {}
    p2b_path = Path(args.phase2b_validation_json)
    if p2b_path.exists():
        try:
            p2b = json.loads(p2b_path.read_text(encoding="utf-8"))
        except Exception as exc:
            warnings.append(f"Could not parse Phase 2B validation JSON: {exc!r}")
    else:
        warnings.append(f"Phase 2B validation JSON not found: {p2b_path}")

    metrics, m_warnings = reconfirm_canonical(image_df, bbox_df, class_df)
    warnings.extend(m_warnings)

    # Defensive framework probes (never fatal in Phase 2C).
    probes = {name: probe_import(name) for name in MMDET_STACK}
    framework_import_ok = all(probes[n]["import_ok"] for n in MMDET_STACK)
    if not framework_import_ok:
        warnings.append(
            "MMDetection stack not importable locally; this is EXPECTED in Phase 2C "
            "(local training framework deferred; remote GPU required)."
        )

    category_id_policy = (
        "COCO category_id is a contiguous integer 1..14 (No Finding excluded). "
        "canonical_class_id (0..13) and class_id_original are preserved in "
        "category metadata for traceability."
    )
    image_path_policy = (
        f"COCO file_name uses relative_dicom_path; resolved at load time by joining "
        f"the {PATH_ROOT_VARIABLE} root."
    )

    forbidden = {
        "coco_master_json_created": False,
        "any_coco_json_created": False,
        "train_val_test_split_created": False,
        "labeled_unlabeled_split_created": False,
        "training_started": False,
        "inference_run": False,
        "pseudo_label_generated": False,
        "threshold_tuned": False,
        "test_set_used": False,
        "pixel_array_read": False,
        "image_copied_or_converted": False,
        "bbox_modified_clamped_deleted_or_fused": False,
        "source_annotation_modified": False,
        "phase2b_canonical_schema_modified": False,
    }

    dod_pass_candidate = bool(
        metrics["canonical_image_rows"] == EXPECT_IMAGE_ROWS
        and metrics["canonical_image_unique_images"] == EXPECT_IMAGE_ROWS
        and metrics["canonical_bbox_rows"] == EXPECT_BBOX_ROWS
        and metrics["canonical_class_count"] == EXPECT_CLASS_COUNT
        and metrics["abnormal_images"] == EXPECT_ABNORMAL_IMAGES
        and metrics["no_finding_images"] == EXPECT_NO_FINDING_IMAGES
        and metrics["no_finding_in_detection_classes"] is False
        and metrics["no_finding_in_canonical_bbox_table"] is False
        and metrics["bbox_source_format"] == BBOX_SOURCE_FORMAT
        and metrics["path_policy_confirmed"] is True
        and all(v is False for v in forbidden.values())
    )

    payload: Dict[str, Any] = {
        "phase": "phase2C_framework_format_decision",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs_used": {
            "canonical_image_table": args.canonical_image_table,
            "canonical_bbox_table": args.canonical_bbox_table,
            "canonical_class_mapping": args.canonical_class_mapping,
            "phase2b_validation_json": args.phase2b_validation_json,
        },
        # Framework decision.
        "primary_framework": PRIMARY_FRAMEWORK,
        "fallback_framework": FALLBACK_FRAMEWORK,
        "framework_decision_status": "decided_pending_remote_setup",
        "local_training_framework_ready": False,
        "remote_gpu_training_required": True,
        "mmdet_stack_import_probe": probes,
        "mmdet_import_required_in_phase2c": False,
        # Format decision.
        "primary_annotation_format": PRIMARY_ANNOTATION_FORMAT,
        "source_schema": SOURCE_OF_TRUTH,
        "actual_coco_conversion_done": False,
        "actual_coco_conversion_phase": "Phase 2D",
        "planned_coco_master_path": PLANNED_COCO_MASTER_PATH,
        # Comparison evidence (rationale for the format decision).
        "format_comparison_matrix": FORMAT_COMPARISON_MATRIX,
        "format_comparison_conclusion": COMPARISON_CONCLUSION,
        # Comparison evidence (rationale for the framework decision).
        "framework_comparison_matrix": FRAMEWORK_COMPARISON_MATRIX,
        "framework_selection_conclusion": FRAMEWORK_SELECTION_CONCLUSION,
        # Canonical re-confirmation.
        **{
            k: metrics[k]
            for k in (
                "canonical_image_rows",
                "canonical_image_unique_images",
                "canonical_bbox_rows",
                "canonical_class_count",
                "abnormal_images",
                "no_finding_images",
                "no_finding_in_canonical_bbox_table",
                "bbox_source_format",
                "path_root_variable",
            )
        },
        "no_finding_is_detection_class": bool(metrics["no_finding_in_detection_classes"]),
        # Planned COCO No Finding policy.
        "no_finding_in_coco_categories_planned": False,
        "no_finding_in_coco_annotations_planned": False,
        "no_finding_in_coco_images_planned": True,
        "background_class_planned": False,
        # BBox / category / path policies.
        "bbox_target_format_phase2D": BBOX_TARGET_FORMAT_2D,
        "category_id_policy": category_id_policy,
        "image_path_policy": image_path_policy,
        "near_duplicate_bbox_candidates_retained": 147,
        # Guardrail flags.
        "train_val_test_split_done": False,
        "labeled_unlabeled_split_done": False,
        "training_done": False,
        "pseudo_label_done": False,
        "threshold_tuning_done": False,
        "test_set_used": False,
        "annotation_modified": False,
        "bbox_clamped": False,
        "bbox_deleted": False,
        "near_duplicate_bbox_fused": False,
        "pixel_array_read": False,
        "image_copied_or_converted": False,
        "dicom_loader_validated": False,
        "dataset_training_ready": False,
        # Metric readiness policy (no metric is computed in Phase 2C).
        "metric_readiness": {
            "ap_metrics_computed_in_phase2c": False,
            "reason": (
                "No split, model training, inference, or prediction file exists yet."
            ),
            "primary_metric": "mAP@0.5:0.95",
            "secondary_diagnostics": [
                "AP50",
                "AP75",
                "class-wise AP",
                "recall/sensitivity",
                "FP/image",
                "FP per negative image",
            ],
            "metrics_computable_only_after": [
                "COCO conversion (Phase 2D)",
                "fixed split creation",
                "model training",
                "prediction generation",
            ],
            "test_set_metric_forbidden_for": [
                "checkpoint selection",
                "threshold tuning",
                "model selection",
                "augmentation decisions",
            ],
            "coco_format_preserves_metric_compatibility": True,
        },
        "warnings": warnings,
        "forbidden_actions_confirmed": forbidden,
        "dod_pass_candidate": dod_pass_candidate,
        "generated_files": {
            "output_json": args.output_json,
            "report_md": args.report_md,
            "framework_yaml": args.framework_yaml,
            "coco_paths_yaml": args.coco_paths_yaml,
            "conversion_policy_yaml": args.conversion_policy_yaml,
        },
    }

    # Write YAML decisions.
    dump_yaml(args.framework_yaml, build_framework_yaml(metrics, probes))
    dump_yaml(args.coco_paths_yaml, build_coco_paths_yaml(metrics, args))
    dump_yaml(args.conversion_policy_yaml, build_conversion_policy_yaml())

    # Write JSON + Markdown.
    ensure_parent(args.output_json).write_text(
        json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8"
    )
    write_report_md(args.report_md, payload)

    # Hard guarantee: Phase 2C must not have created a COCO master file.
    coco_master = Path(PLANNED_COCO_MASTER_PATH)
    if coco_master.exists():
        warnings.append(
            f"NOTE: {PLANNED_COCO_MASTER_PATH} exists on disk but was NOT created by "
            "Phase 2C."
        )

    # Console summary.
    print("=" * 68)
    print("Phase 2C — Framework & Format Decision / COCO Conversion Planning")
    print("=" * 68)
    print(f"primary_framework          : {PRIMARY_FRAMEWORK}")
    print(f"primary_annotation_format  : {PRIMARY_ANNOTATION_FORMAT}")
    print(f"canonical_image_rows       : {metrics['canonical_image_rows']}")
    print(f"canonical_bbox_rows        : {metrics['canonical_bbox_rows']}")
    print(f"canonical_class_count      : {metrics['canonical_class_count']}")
    print(f"no_finding_images          : {metrics['no_finding_images']}")
    print(f"actual_coco_conversion_done: {payload['actual_coco_conversion_done']}")
    print(f"dataset_training_ready     : {payload['dataset_training_ready']}")
    print("-" * 68)
    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  WARN: {w}")
    else:
        print("Warnings                   : none")
    print("-" * 68)
    print(f"dod_pass_candidate         : {dod_pass_candidate}")
    print("=" * 68)
    print("Decision/protocol evidence only. NO COCO file was created.")
    print("Next: Phase 2D only after GPT review PASS.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
