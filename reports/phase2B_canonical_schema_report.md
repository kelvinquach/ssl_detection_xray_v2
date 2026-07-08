# Phase 2B — Canonical Detection Annotation Schema

_Generated 2026-07-08T16:38:01.958636+00:00._

## Executive summary

Built canonical schema for **4894** images and **36096** abnormal bboxes across **14** detection classes. No Finding policy pass: **True**. Schema errors: **0**. DoD pass candidate: **True**.

## Inputs

- annotations_csv: `data\interim\vinbigdata_phase1C_scope_annotations.csv`
- manifest_csv: `data\manifests\phase1C_selected_images_manifest.csv`
- image_metadata_csv: `reports\phase2A_image_metadata.csv`

## Outputs

- `canonical_image_table.csv` — one row per unique image_id.
- `canonical_bbox_table.csv` — one row per abnormal bbox (xyxy original).
- `canonical_class_mapping.csv` — 14 abnormal detection classes.
- No Finding audit, validation JSON, and schema-error CSV.

## Canonical image table schema

canonical_image_id, image_id, dicom_filename, relative_dicom_path, dicom_path (= relative_dicom_path), local_dicom_path, local_dicom_path_is_absolute, path_root_variable, image_width, image_height, scope_label, is_abnormal, is_negative, has_bbox, bbox_count, no_finding_bbox_count, abnormal_class_count, abnormal_class_names, source_row_count, abnormal_row_count, no_finding_row_count. `local_dicom_path_is_absolute` is True when `local_dicom_path` is an absolute path (local evidence only).

## Canonical bbox table schema

canonical_ann_id, image_id, source_row_id, rad_id, class_id_original, class_name, canonical_class_id, x_min, y_min, x_max, y_max, bbox_width, bbox_height, bbox_area, image_width, image_height, bbox_format, is_valid_bbox, boundary_valid. Format is xyxy on the ORIGINAL image; no bbox is clamped, modified, fused, or dropped.

## Canonical class mapping schema

canonical_class_id, class_id_original, class_name, is_detection_class, is_no_finding, row_count, image_count, bbox_count. canonical_class_id is deterministic: classes sorted by (class_id_original, class_name), enumerated from 0. No Finding is excluded from detection classes.

## No Finding policy audit

- no_finding_images: 500
- no_finding_policy_pass: True
- no_finding_in_detection_classes: False
- Audit file: `phase2B_no_finding_policy_audit.csv`.

## Consistency validation

- bbox_without_image_count: 0
- image_without_metadata_count: 0
- bbox_missing_dimension_count: 0
- bbox_invalid_count: 0
- class_mapping_issue_count: 0
- schema_error_count: 0

## Portable path policy

- portable_path_policy_pass: True
- relative_dicom_path_missing_count: 0
- relative_dicom_path_absolute_count: 0 (expected 0)
- local_dicom_path_absolute_count: 4894
- path_root_variable: `VINBIGDATA_DICOM_ROOT`

- Canonical schema uses `image_id` and `relative_dicom_path` as portable identifiers.
- `local_dicom_path` is retained only as Phase 2A/2B local evidence.
- Downstream COCO conversion or dataloader must resolve image files by joining an environment/config root such as `VINBIGDATA_DICOM_ROOT` with `relative_dicom_path`.
- No image file was copied or converted.

## Traceability guarantees

- Every bbox row keeps `source_row_id` (index into the Phase 1C scope annotations) and `class_id_original`, so any canonical row can be traced back to the exact source annotation.
- `canonical_class_id` is a deterministic re-index of the original class_id; the mapping table records both.
- No source annotation is edited; near-duplicate bboxes are retained.

## Forbidden actions confirmed

- split_created: False
- coco_created: False
- training_started: False
- pseudo_label_generated: False
- threshold_tuned: False
- test_set_used: False
- annotations_deleted_or_edited: False
- bbox_clamped_or_modified: False
- near_duplicate_bbox_deleted_or_fused: False
- image_files_copied: False
- image_files_converted: False
- processed_training_images_created: False

## Limitations

- This schema is not a COCO dataset and not a split. It is a canonical intermediate for downstream conversion.
- 147 near-duplicate bbox candidates (Phase 1B) remain present; fusion is a later research decision, not performed here.
- Moving to a remote/GPU environment requires setting `VINBIGDATA_DICOM_ROOT` or an equivalent data-root config.
- Absolute local paths must not be used as canonical downstream identifiers.

## Next allowed phase

- **Phase 2C / COCO conversion only after GPT review PASS** of these outputs. Do not proceed automatically.
