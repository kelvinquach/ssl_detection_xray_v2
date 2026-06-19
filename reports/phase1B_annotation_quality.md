# Phase 1B — Annotation Quality Report

_Generated 2026-06-19T15:07:22.437157+00:00 from `data\raw\vinbigdata\annotations\train.csv`._

## Executive summary

Analyzed **67914** annotation rows across **15000** images. Invalid bbox flags: **0**. Exact duplicate candidates: **0**, near-duplicate candidates (IoU ≥ 0.95): **147**. Class-mapping issues: **0**.

## Scope

- Reads the **full source metadata `train.csv` only**. It does NOT build the downstream 4,894-image controlled subset.
- CSV-only: no split, no COCO, no image/DICOM/PNG reads, no training, no pseudo-labelling, no threshold tuning, no test-set access.
- Report-only: no annotation is deleted or modified.

## Checks performed

- Bbox coordinate sanity (missing, non-numeric, negative, degenerate geometry, non-positive area).
- Image-boundary checks (only if the CSV carries image dimensions).
- No Finding policy (negative label, must carry no bbox).
- Abnormal-annotation completeness (full bbox required).
- Exact and near-duplicate bbox candidates within (image_id, class).
- class_id <-> class_name mapping bijection.

## Key findings

- Abnormal rows: 36096; No Finding rows: 31818.
- Abnormal images: 4394; No Finding images: 10606.
- Abnormal detection classes (excl. No Finding): 14.
- No Finding rows carrying bbox: 0.
- Abnormal rows missing bbox: 0.
- Images mixing No Finding + abnormal: 0.
- Negative-coordinate rows: 0.
- Zero/negative-area rows: 0.

## Invalid bbox summary (by reason)

| reason | count |
|---|---|
| missing_coordinate_abnormal | 0 |
| non_numeric_coordinate | 0 |
| x_min_negative | 0 |
| y_min_negative | 0 |
| x_max_negative | 0 |
| y_max_negative | 0 |
| x_min_ge_x_max | 0 |
| y_min_ge_y_max | 0 |
| width_le_0 | 0 |
| height_le_0 | 0 |
| area_le_0 | 0 |
| no_finding_with_bbox | 0 |
| x_max_gt_image_width | 0 |
| y_max_gt_image_height | 0 |
| x_min_gt_image_width | 0 |
| y_min_gt_image_height | 0 |

## Duplicate / near-duplicate summary

- Exact duplicate candidates: 0.
- Near-duplicate candidates (IoU ≥ 0.95): 147.
- **Interpretation:** VinBigData is multi-radiologist. Duplicate and near-duplicate boxes on the same image+class are very likely independent annotations from different readers, NOT confirmed errors. They are recorded as *candidates* for review only.

## Class mapping summary

- Mapping issues detected: 0.
- See `phase1B_class_mapping.csv` for the full table.

## No Finding policy summary

- 'No Finding' is treated as a negative image label, excluded from detection classes.
- No Finding rows with bbox (policy violation candidates): 0.
- Images mixing No Finding and abnormal labels: 0.

## Boundary check status

- `not_evaluable_without_image_dimensions`
- The CSV does not carry image dimensions; boundary checks are deliberately skipped. No image files are read in Phase 1B.

## Research risk interpretation

- Coordinate-sanity violations would directly corrupt detection targets and must be resolved before any COCO conversion.
- Duplicate/near-duplicate candidates affect how multi-reader boxes are fused later; they are not necessarily defects.
- Mixed No Finding + abnormal images, if any, would break the negative-image assumption used for semi-supervised negatives.

## Recommended next action

- **Send these outputs to GPT review BEFORE ticking the Phase 1B checklist.** Do not auto-correct annotations; decisions on duplicate fusion and any flagged rows are research decisions.
