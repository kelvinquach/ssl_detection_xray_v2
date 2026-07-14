# Phase 2D — COCO Master Conversion & Validation

_Generated 2026-07-14T13:38:09.036754+00:00._

## Objective and scope

Convert the Phase 2B canonical schema into a real COCO detection JSON at `data/processed/coco/coco_master.json` and validate it exhaustively. This phase reorganizes METADATA ONLY.

**No image or DICOM access of any kind occurred.** No `.dicom` file was read, opened, or even checked for existence; `pydicom`, `cv2`, and `PIL` were never imported. `file_name`, `width`, and `height` come exclusively from `canonical_image_table.csv`.

## Input / output

| Role | Path | SHA-256 |
|---|---|---|
| input: canonical_image_table | `data/processed/canonical/canonical_image_table.csv` | `8b3a2368eaf806a1…` |
| input: canonical_bbox_table | `data/processed/canonical/canonical_bbox_table.csv` | `d04d11025eda6038…` |
| input: canonical_class_mapping | `data/processed/canonical/canonical_class_mapping.csv` | `443f79cbadc7ab9a…` |
| input: phase2b_validation_json | `reports/phase2B_canonical_schema_validation.json` | `8c5c57acc60d4a61…` |
| input: protocol_yaml | `configs/protocol/phase2D_coco_master_validation.yaml` | `d841a34e6633e1c3…` |
| output | `data/processed/coco/coco_master.json` | `36f09d1b1477ea4a…` |

## COCO schema

- `images[]`: id, file_name, width, height (+ canonical_image_id, original_image_id, scope_label, is_negative)
- `annotations[]`: id, image_id, category_id, bbox, area, iscrowd (+ traceability fields)
- `categories[]`: id, name, supercategory=`chest_abnormality`, canonical_class_id, class_id_original

## ID policy

- image id: contiguous 1..4894, sorted by `canonical_image_id`.
- annotation id: contiguous 1..36096, sorted by `canonical_ann_id`.
- category id: contiguous 1..14, following canonical_class_id order (never alphabetical). Category id 0 is never used.

## BBox conversion policy

- `xyxy_original_image` → `coco_xywh_absolute`
- x = x_min; y = y_min; width = x_max − x_min; height = y_max − y_min; area = width × height; iscrowd = 0.
- **No clamping, no deletion, no fusion, no NMS, no rounding.**

## Count summary

| Item | Value | Expected | Pass |
|---|---|---|---|
| images | 4894 | 4894 | PASS |
| annotations | 36096 | 36096 | PASS |
| categories | 14 | 14 | PASS |
| abnormal images | 4394 | 4394 | PASS |
| No Finding images | 500 | 500 | PASS |
| No Finding annotations | 0 | 0 | PASS |

## Validation results

| Check | Result | Status |
|---|---|---|
| protocol_yaml_strict_load_pass | True | PASS |
| protocol_phase2b_crosscheck_pass | True | PASS |
| images_expected | True | PASS |
| annotations_expected | True | PASS |
| categories_expected | True | PASS |
| abnormal_images_expected | True | PASS |
| no_finding_images_expected | True | PASS |
| no_finding_zero_annotations | True | PASS |
| no_finding_not_a_category | True | PASS |
| category_ids_contiguous_1_to_n | True | PASS |
| category_id_zero_absent | True | PASS |
| image_ids_unique | True | PASS |
| annotation_ids_unique | True | PASS |
| file_names_unique | True | PASS |
| relative_paths_only | True | PASS |
| all_references_valid | True | PASS |
| zero_invalid_annotations | True | PASS |
| boundary_pass | True | PASS |
| area_pass | True | PASS |
| traceability_pass | True | PASS |
| one_to_one_preservation_pass | True | PASS |
| per_image_bbox_count_pass | True | PASS |
| category_annotation_total_pass | True | PASS |
| json_parse_pass | True | PASS |
| json_top_level_keys_pass | True | PASS |
| pycocotools_pass_or_unavailable | True | PASS |
| all_hard_checks_completed_before_output_replace | True | PASS |
| per_image_bbox_count_checked_before_output_replace | True | PASS |
| atomic_output_promotion_pass | True | PASS |

## No Finding audit

- Negative images are determined from **canonical image metadata**, not from a zero-annotation count (`canonical_image_metadata_not_zero_annotation`).
- No Finding images: **500** (expected 500).
- All carry zero annotations: **True**.
- No Finding present in categories: **False** (must be false).
- Per-image evidence: `phase2D_coco_no_finding_audit.csv` (500 rows).

## Category summary

| id | name | canonical_class_id | annotations | images |
|---|---|---|---|---|
| 1 | Aortic enlargement | 0 | 7162 | 3067 |
| 2 | Atelectasis | 1 | 279 | 186 |
| 3 | Calcification | 2 | 960 | 452 |
| 4 | Cardiomegaly | 3 | 5427 | 2300 |
| 5 | Consolidation | 4 | 556 | 353 |
| 6 | ILD | 5 | 1000 | 386 |
| 7 | Infiltration | 6 | 1247 | 613 |
| 8 | Lung Opacity | 7 | 2483 | 1322 |
| 9 | Nodule/Mass | 8 | 2580 | 826 |
| 10 | Other lesion | 9 | 2203 | 1134 |
| 11 | Pleural effusion | 10 | 2476 | 1032 |
| 12 | Pleural thickening | 11 | 4842 | 1981 |
| 13 | Pneumothorax | 12 | 226 | 96 |
| 14 | Pulmonary fibrosis | 13 | 4655 | 1617 |

## Traceability preservation

- Coordinate/image/category mismatches vs canonical: 0 / 0 / 0.
- canonical_ann_id sets equal: **True**; missing=0, duplicated=0, extra=0.
- Set equality of canonical_ann_id is the evidence that no bbox — including near-duplicate candidates — was deleted or fused. Phase 2D did NOT re-run near-duplicate detection; the candidate file is not an input here.

## pycocotools result

- available: True; load pass: True.
- Used for **annotation_json_parsing_only** — never to read images.

## Protocol enforcement

- Protocol YAML: `configs/protocol/phase2D_coco_master_validation.yaml`
- Strict schema load: **True** (no silent fallback; a missing/malformed/negative/non-finite value aborts the run).
- Phase 2B cross-check: **True**; protocol drift count: **0**.
- Every expected count is reconciled three ways — protocol YAML ↔ Phase 2B validation JSON ↔ the actual canonical tables — so a YAML edited to legitimise a different dataset cannot pass.

Effective protocol driving this run:

```json
{
  "expected_counts": {
    "images": 4894,
    "annotations": 36096,
    "categories": 14,
    "abnormal_images": 4394,
    "no_finding_images": 500,
    "no_finding_annotations": 0
  },
  "tolerance": {
    "area_rel_tol": 1e-09,
    "area_abs_tol": 1e-06,
    "boundary_abs_tol": 1e-06,
    "coordinate_abs_tol": 1e-09
  },
  "bbox_source_format": "xyxy_original_image",
  "bbox_target_format": "coco_xywh_absolute",
  "supercategory": "chest_abnormality",
  "image_root_env_var": "VINBIGDATA_DICOM_ROOT",
  "forbidden_category_names": [
    "__background__",
    "background",
    "no finding",
    "nofinding",
    "normal"
  ]
}
```

## Atomic output validation

| Check | Value |
|---|---|
| validation_completed_before_promotion | True |
| per_image_count_checked_before_promotion | True |
| temporary_file_used | True |
| temporary_written | True |
| temporary_json_parse_pass | True |
| pycocotools_checked_before_promotion | True |
| all_pre_promotion_checks_pass | True |
| final_output_replaced | True |
| previous_valid_output_preserved_on_failure | True |
| temporary_json_parse_pass | True |

The final `coco_master.json` is replaced with `os.replace()` **only after** every hard check — including per-image `canonical_bbox_count == coco_annotation_count` and the pycocotools load — has passed on a temporary file. On any failure the temporary file is removed in a `finally` block and any pre-existing final output is left byte-for-byte untouched.

## Forbidden actions confirmation

- dicom_file_read: False
- dicom_file_existence_checked: False
- pydicom_used: False
- dicom_header_read: False
- pixel_array_read: False
- cv2_imread_used: False
- pil_image_open_used: False
- any_image_loader_used: False
- image_copied_or_converted: False
- train_val_test_split_created: False
- labeled_unlabeled_split_created: False
- mmdet_or_detectron2_dataset_loaded: False
- filter_empty_gt_checked: False
- training_started: False
- inference_run: False
- pseudo_label_generated: False
- threshold_tuned: False
- test_set_used: False
- ap_map_computed: False
- canonical_schema_modified: False
- source_annotation_modified: False
- bbox_deleted: False
- bbox_clamped: False
- bbox_fused: False
- nms_applied: False
- dataset_training_ready_claimed: False

## Warnings

- none

## Limitations

- A valid COCO annotation file does **not** make the dataset training-ready.
- The DICOM loader is NOT validated; MMDetection's default `LoadImageFromFile` cannot read `.dicom`.
- Empty-image (No Finding) loading behaviour is NOT validated here; `filter_empty_gt` was deliberately not checked.
- Near-duplicate detection was NOT re-run in Phase 2D; one-to-one preservation is the evidence that nothing was dropped or fused.

- **dataset_training_ready = False**

## Next phase

- Phase 2D.1 (DICOM loader / empty-image loading) remains **LOCKED** until GPT review PASS of this evidence.
- This script does not and cannot conclude a GPT review verdict.
