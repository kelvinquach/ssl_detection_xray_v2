#!/usr/bin/env python3
"""Phase 2B — Build Canonical Detection Annotation Schema (CLI).

Reorganizes Phase 1C/2A metadata into canonical tables (image, bbox, class
mapping) with full traceability. Report-only transformation: no COCO, no split,
no training, no image reads/copies, and no bbox edit/clamp/fuse/drop.

Usage (Windows CMD):
    python scripts\\02B_build_canonical_schema.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make `src` importable when run from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    print(f"ERROR: pandas required but not importable: {exc!r}", file=sys.stderr)
    raise SystemExit(2)

from src.data.build_canonical_schema import (  # noqa: E402
    EXPECT_ABNORMAL_IMAGES,
    EXPECT_ABNORMAL_ROWS,
    EXPECT_DETECTION_CLASSES,
    EXPECT_NO_FINDING_IMAGES,
    EXPECT_TOTAL_IMAGES,
    build_bbox_table,
    build_class_mapping,
    build_image_table,
    build_no_finding_audit,
    load_image_dimensions,
    validate_schema,
)

BBOX_COLUMNS = ["x_min", "y_min", "x_max", "y_max"]
REQUIRED_ANN_COLUMNS = ["image_id", "class_name", *BBOX_COLUMNS]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2B — build canonical detection annotation schema.",
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
        "--image-metadata-csv",
        type=str,
        default="reports/phase2A_image_metadata.csv",
    )
    p.add_argument(
        "--bbox-boundary-csv",
        type=str,
        default="reports/phase2A_bbox_boundary_validation.csv",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/canonical",
    )
    p.add_argument(
        "--report-md",
        type=str,
        default="reports/phase2B_canonical_schema_report.md",
    )
    p.add_argument(
        "--validation-json",
        type=str,
        default="reports/phase2B_canonical_schema_validation.json",
    )
    p.add_argument(
        "--no-finding-audit-csv",
        type=str,
        default="reports/phase2B_no_finding_policy_audit.csv",
    )
    p.add_argument(
        "--schema-errors-csv",
        type=str,
        default="reports/phase2B_schema_consistency_errors.csv",
    )
    return p.parse_args(argv)


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_csv(path: str, required: List[str], what: str, optional: bool = False):
    fp = Path(path)
    if not fp.exists():
        if optional:
            return None
        raise FileNotFoundError(
            f"{what} not found: {fp}\n       Check the path."
        )
    try:
        df = pd.read_csv(fp)
    except Exception as exc:
        raise ValueError(f"Failed to read {what} '{fp}': {exc!r}") from exc
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{what} missing required column(s): {missing}\n"
            f"       Found: {list(df.columns)}"
        )
    return df


def build_boundary_map(bbox_boundary: Optional[pd.DataFrame]) -> Optional[Dict[int, bool]]:
    """Map source_row_index -> boundary_valid from Phase 2A, if available."""
    if bbox_boundary is None or bbox_boundary.empty:
        return None
    if "source_row_index" not in bbox_boundary.columns or "boundary_valid" not in bbox_boundary.columns:
        return None
    out: Dict[int, bool] = {}
    for _, row in bbox_boundary.iterrows():
        try:
            out[int(row["source_row_index"])] = bool(row["boundary_valid"])
        except Exception:
            continue
    return out


def write_report_md(path: str, p: Dict[str, Any]) -> None:
    L: List[str] = []
    L.append("# Phase 2B — Canonical Detection Annotation Schema")
    L.append("")
    L.append(f"_Generated {p['created_utc']}._")
    L.append("")
    L.append("## Executive summary")
    L.append("")
    L.append(
        f"Built canonical schema for **{p['canonical_image_rows']}** images and "
        f"**{p['canonical_bbox_rows']}** abnormal bboxes across "
        f"**{p['canonical_class_count']}** detection classes. "
        f"No Finding policy pass: **{p['no_finding_policy_pass']}**. "
        f"Schema errors: **{p['schema_error_count']}**. "
        f"DoD pass candidate: **{p['dod_pass_candidate']}**."
    )
    L.append("")
    L.append("## Inputs")
    L.append("")
    L.append(f"- annotations_csv: `{p['annotations_csv']}`")
    L.append(f"- manifest_csv: `{p['manifest_csv']}`")
    L.append(f"- image_metadata_csv: `{p['image_metadata_csv']}`")
    L.append("")
    L.append("## Outputs")
    L.append("")
    L.append("- `canonical_image_table.csv` — one row per unique image_id.")
    L.append("- `canonical_bbox_table.csv` — one row per abnormal bbox (xyxy original).")
    L.append("- `canonical_class_mapping.csv` — 14 abnormal detection classes.")
    L.append("- No Finding audit, validation JSON, and schema-error CSV.")
    L.append("")
    L.append("## Canonical image table schema")
    L.append("")
    L.append(
        "canonical_image_id, image_id, dicom_filename, relative_dicom_path, "
        "dicom_path (= relative_dicom_path), local_dicom_path, "
        "local_dicom_path_is_absolute, path_root_variable, image_width, "
        "image_height, scope_label, is_abnormal, is_negative, has_bbox, "
        "bbox_count, no_finding_bbox_count, abnormal_class_count, "
        "abnormal_class_names, source_row_count, abnormal_row_count, "
        "no_finding_row_count. `local_dicom_path_is_absolute` is True when "
        "`local_dicom_path` is an absolute path (local evidence only)."
    )
    L.append("")
    L.append("## Canonical bbox table schema")
    L.append("")
    L.append(
        "canonical_ann_id, image_id, source_row_id, rad_id, class_id_original, "
        "class_name, canonical_class_id, x_min, y_min, x_max, y_max, bbox_width, "
        "bbox_height, bbox_area, image_width, image_height, bbox_format, "
        "is_valid_bbox, boundary_valid. Format is xyxy on the ORIGINAL image; "
        "no bbox is clamped, modified, fused, or dropped."
    )
    L.append("")
    L.append("## Canonical class mapping schema")
    L.append("")
    L.append(
        "canonical_class_id, class_id_original, class_name, is_detection_class, "
        "is_no_finding, row_count, image_count, bbox_count. canonical_class_id is "
        "deterministic: classes sorted by (class_id_original, class_name), "
        "enumerated from 0. No Finding is excluded from detection classes."
    )
    L.append("")
    L.append("## No Finding policy audit")
    L.append("")
    L.append(f"- no_finding_images: {p['no_finding_images']}")
    L.append(f"- no_finding_policy_pass: {p['no_finding_policy_pass']}")
    L.append(f"- no_finding_in_detection_classes: {p['no_finding_in_detection_classes']}")
    L.append("- Audit file: `phase2B_no_finding_policy_audit.csv`.")
    L.append("")
    L.append("## Consistency validation")
    L.append("")
    L.append(f"- bbox_without_image_count: {p['bbox_without_image_count']}")
    L.append(f"- image_without_metadata_count: {p['image_without_metadata_count']}")
    L.append(f"- bbox_missing_dimension_count: {p['bbox_missing_dimension_count']}")
    L.append(f"- bbox_invalid_count: {p['bbox_invalid_count']}")
    L.append(f"- class_mapping_issue_count: {p['class_mapping_issue_count']}")
    L.append(f"- schema_error_count: {p['schema_error_count']}")
    L.append("")
    L.append("## Portable path policy")
    L.append("")
    L.append(f"- portable_path_policy_pass: {p['portable_path_policy_pass']}")
    L.append(f"- relative_dicom_path_missing_count: {p['relative_dicom_path_missing_count']}")
    L.append(f"- relative_dicom_path_absolute_count: {p['relative_dicom_path_absolute_count']} (expected 0)")
    L.append(f"- local_dicom_path_absolute_count: {p['local_dicom_path_absolute_count']}")
    L.append(f"- path_root_variable: `{p['path_root_variable']}`")
    L.append("")
    L.append(
        "- Canonical schema uses `image_id` and `relative_dicom_path` as portable "
        "identifiers."
    )
    L.append(
        "- `local_dicom_path` is retained only as Phase 2A/2B local evidence."
    )
    L.append(
        "- Downstream COCO conversion or dataloader must resolve image files by "
        "joining an environment/config root such as `VINBIGDATA_DICOM_ROOT` with "
        "`relative_dicom_path`."
    )
    L.append("- No image file was copied or converted.")
    L.append("")
    L.append("## Traceability guarantees")
    L.append("")
    L.append(
        "- Every bbox row keeps `source_row_id` (index into the Phase 1C scope "
        "annotations) and `class_id_original`, so any canonical row can be traced "
        "back to the exact source annotation."
    )
    L.append(
        "- `canonical_class_id` is a deterministic re-index of the original "
        "class_id; the mapping table records both."
    )
    L.append("- No source annotation is edited; near-duplicate bboxes are retained.")
    L.append("")
    L.append("## Forbidden actions confirmed")
    L.append("")
    for k, v in p["forbidden_actions_confirmed"].items():
        L.append(f"- {k}: {v}")
    L.append("")
    L.append("## Limitations")
    L.append("")
    L.append(
        "- This schema is not a COCO dataset and not a split. It is a canonical "
        "intermediate for downstream conversion."
    )
    L.append(
        "- 147 near-duplicate bbox candidates (Phase 1B) remain present; fusion is "
        "a later research decision, not performed here."
    )
    L.append(
        "- Moving to a remote/GPU environment requires setting "
        "`VINBIGDATA_DICOM_ROOT` or an equivalent data-root config."
    )
    L.append(
        "- Absolute local paths must not be used as canonical downstream "
        "identifiers."
    )
    L.append("")
    L.append("## Next allowed phase")
    L.append("")
    L.append(
        "- **Phase 2C / COCO conversion only after GPT review PASS** of these "
        "outputs. Do not proceed automatically."
    )
    L.append("")
    ensure_parent(path).write_text("\n".join(L), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # Load inputs.
    try:
        ann = load_csv(args.annotations_csv, REQUIRED_ANN_COLUMNS, "annotations CSV")
        manifest = load_csv(args.manifest_csv, ["image_id"], "manifest CSV")
        image_metadata = load_csv(
            args.image_metadata_csv, ["image_id"], "image metadata CSV", optional=True
        )
        bbox_boundary = load_csv(
            args.bbox_boundary_csv, [], "bbox boundary CSV", optional=True
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if image_metadata is None:
        print(
            f"ERROR: image metadata CSV not found: {args.image_metadata_csv}\n"
            "       Phase 2B needs Phase 2A image dimensions. Run Phase 2A first.",
            file=sys.stderr,
        )
        return 1

    warnings: List[str] = []

    # Core metadata.
    dims_by_image, path_by_image = load_image_dimensions(image_metadata)
    boundary_by_row = build_boundary_map(bbox_boundary)

    total_annotation_rows = int(len(ann))
    unique_annotation_images = int(ann["image_id"].nunique())
    manifest_rows = int(len(manifest))
    manifest_unique_images = int(manifest["image_id"].nunique())

    # Build canonical tables.
    mapping_df, class_issue_count, mapping_warnings = build_class_mapping(ann)
    warnings.extend(mapping_warnings)
    image_df = build_image_table(ann, manifest, dims_by_image, path_by_image)
    bbox_df = build_bbox_table(ann, mapping_df, dims_by_image, boundary_by_row)
    nf_audit_df, nf_policy_pass = build_no_finding_audit(image_df)

    # Validate.
    metrics, errors_df, val_warnings = validate_schema(
        ann, manifest, image_df, bbox_df, mapping_df,
        dims_by_image, nf_policy_pass, class_issue_count,
    )
    warnings.extend(val_warnings)

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
        "image_files_copied": False,
        "image_files_converted": False,
        "processed_training_images_created": False,
    }

    dod_pass_candidate = bool(
        metrics["canonical_image_rows"] == EXPECT_TOTAL_IMAGES
        and metrics["canonical_image_unique_images"] == EXPECT_TOTAL_IMAGES
        and metrics["abnormal_images"] == EXPECT_ABNORMAL_IMAGES
        and metrics["no_finding_images"] == EXPECT_NO_FINDING_IMAGES
        and metrics["canonical_bbox_rows"] == EXPECT_ABNORMAL_ROWS
        and metrics["canonical_class_count"] == EXPECT_DETECTION_CLASSES
        and metrics["no_finding_policy_pass"] is True
        and metrics["no_finding_in_detection_classes"] is False
        and metrics["bbox_without_image_count"] == 0
        and metrics["image_without_metadata_count"] == 0
        and metrics["bbox_missing_dimension_count"] == 0
        and metrics["bbox_invalid_count"] == 0
        and metrics["class_mapping_issue_count"] == 0
        and metrics["schema_error_count"] == 0
        and metrics["portable_path_policy_pass"] is True
        and metrics["relative_dicom_path_absolute_count"] == 0
        and metrics["relative_dicom_path_missing_count"] == 0
        and all(v is False for v in forbidden.values())
    )

    payload: Dict[str, Any] = {
        "phase": "phase2B_canonical_schema",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "annotations_csv": str(args.annotations_csv),
        "manifest_csv": str(args.manifest_csv),
        "image_metadata_csv": str(args.image_metadata_csv),
        "total_annotation_rows": total_annotation_rows,
        "unique_annotation_images": unique_annotation_images,
        "manifest_rows": manifest_rows,
        "manifest_unique_images": manifest_unique_images,
        **metrics,
        "forbidden_actions_confirmed": forbidden,
        "warnings": warnings,
        "dod_pass_candidate": dod_pass_candidate,
    }

    # Write outputs.
    out_dir = ensure_dir(args.output_dir)
    image_df.to_csv(out_dir / "canonical_image_table.csv", index=False, encoding="utf-8")
    bbox_df.to_csv(out_dir / "canonical_bbox_table.csv", index=False, encoding="utf-8")
    mapping_df.to_csv(out_dir / "canonical_class_mapping.csv", index=False, encoding="utf-8")

    nf_audit_df.to_csv(ensure_parent(args.no_finding_audit_csv), index=False, encoding="utf-8")
    errors_df.to_csv(ensure_parent(args.schema_errors_csv), index=False, encoding="utf-8")
    ensure_parent(args.validation_json).write_text(
        json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8"
    )
    write_report_md(args.report_md, payload)

    # Console summary.
    print("=" * 66)
    print("Phase 2B — Canonical Detection Annotation Schema")
    print("=" * 66)
    print(f"canonical_image_rows        : {metrics['canonical_image_rows']}")
    print(f"canonical_bbox_rows         : {metrics['canonical_bbox_rows']}")
    print(f"canonical_class_count       : {metrics['canonical_class_count']}")
    print(f"no_finding_images           : {metrics['no_finding_images']}")
    print(f"no_finding_policy_pass      : {metrics['no_finding_policy_pass']}")
    print(f"bbox_without_image_count    : {metrics['bbox_without_image_count']}")
    print(f"image_without_metadata_count: {metrics['image_without_metadata_count']}")
    print(f"schema_error_count          : {metrics['schema_error_count']}")
    print(f"portable_path_policy_pass   : {metrics['portable_path_policy_pass']}")
    print(f"relative_dicom_path_abs_cnt : {metrics['relative_dicom_path_absolute_count']}")
    print(f"local_dicom_path_abs_cnt    : {metrics['local_dicom_path_absolute_count']}")
    print("-" * 66)
    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for w in warnings[:20]:
            print(f"  WARN: {w}")
    else:
        print("Warnings                    : none")
    print("-" * 66)
    print(f"dod_pass_candidate          : {dod_pass_candidate}")
    print("=" * 66)
    print("Outputs ->", str(out_dir))
    print("NOTE: Not COCO, not a split. Send to GPT review before Phase 2C.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
