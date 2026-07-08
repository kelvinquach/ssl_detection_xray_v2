# Phase 2A — DICOM Availability & BBox Boundary Validation

_Generated 2026-07-08T09:41:23.892857+00:00._

## Executive summary

Checked availability of **4894** images and validated **36096** abnormal bboxes against true DICOM dimensions. DICOM missing: **0**, read errors: **0**. Boundary-invalid bboxes: **0**. DoD pass candidate: **True**.

## Scope

- Phase 1C controlled scope only (expected 4,894 images). Report-only: no bbox edited/clamped/deleted/fused; no image copied/converted.
- DICOM read for metadata/dimensions only (header via stop_before_pixels).

## Inputs

- annotations_csv: `data\interim\vinbigdata_phase1C_scope_annotations.csv`
- manifest_csv: `data\manifests\phase1C_selected_images_manifest.csv`
- dicom_root: `D:\ssl_detection_xray\data\raw\vinbigdata\dicom_subset\train`

## Image availability summary

- selected_scope_expected_images: 4894
- availability_checked_image_count: 4894
- dicom_available_count: 4894
- dicom_missing_count: 0

## DICOM metadata summary

- dicom_read_success_count: 4894
- dicom_read_error_count: 0
- pixel_array_checked: False
- pixel_array_check_count: 0
- pixel_array_error_count: 0

## Image dimension summary

- image_dimension_available_count: 4894
- image_dimension_missing_count: 0
- width/height distribution: {'width_min': 1320, 'width_max': 3320, 'width_mean': 2491.66, 'height_min': 1416, 'height_max': 3408, 'height_mean': 2835.09, 'distinct_wh_pairs': 2186}

## BBox boundary validation summary

- abnormal_bbox_rows_checked: 36096
- bbox_boundary_valid_count: 36096
- bbox_boundary_invalid_count: 0

| reason | count |
|---|---|
| missing_coordinate | 0 |
| non_numeric_coordinate | 0 |
| x_min_negative | 0 |
| y_min_negative | 0 |
| x_max_negative | 0 |
| y_max_negative | 0 |
| x_min_ge_x_max | 0 |
| y_min_ge_y_max | 0 |
| bbox_width_le_0 | 0 |
| bbox_height_le_0 | 0 |
| x_max_gt_image_width | 0 |
| y_max_gt_image_height | 0 |
| image_dimension_missing | 0 |
| dicom_missing_or_read_error | 0 |

## No Finding policy check

- no_finding_images: 500
- no_finding_rows: 1500
- no_finding_with_bbox_count: 0
- abnormal_missing_bbox_count: 0
- No Finding is a negative image label; excluded from detection classes.

## Invalid bbox / DICOM error details

- Invalid bbox candidates: `phase2A_invalid_bbox_candidates.csv` (review only, not auto-fixed).
- DICOM read errors: `phase2A_dicom_read_errors.csv`.

## Decision candidates

- None; all checks clean.

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
- processed_training_images_created: False
- image_files_copied: False
- image_files_converted: False
- png_or_jpg_created: False

## Recommended next action

- **Send these outputs to GPT review BEFORE ticking the Phase 2A checklist.** Do not auto-fix boxes or images; corrections are research decisions.
