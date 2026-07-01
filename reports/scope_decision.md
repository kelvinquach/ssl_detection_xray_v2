# Phase 1C — Dataset Scope Decision

_Generated 2026-07-01T07:50:54.304427+00:00._

## Executive summary

Locked the controlled working scope of **4894** images (**4394** abnormal + **500** No Finding), cross-validated across train.csv, package manifests, and the DICOM filename inventory. DoD pass candidate: **True**.

## Source metadata vs controlled working scope

- Source rows: 67914; source images: 15000.
- Source abnormal images: 4394; No Finding images: 10606.
- Controlled scope: 4894 images (subset of source).

## Evidence from package manifests

- Manifest parts found: 35.
- Manifest rows: 4894; unique image_id: 4894; duplicates: 0.
- image_type abnormal: 4394; normal: 500 (not trusted blindly; reconciled with train.csv).

## Evidence from DICOM filename inventory

- DICOM files listed (*.dicom): 4894; unique image_id: 4894; duplicates: 0.
- Filenames only: no DICOM header, pixel, or dimension was read.

## Cross-check manifest vs DICOM filenames

- manifest_not_in_dicom_count: 0.
- dicom_not_in_manifest_count: 0.

## Cross-check selected scope vs train.csv

- unknown_manifest_image_id_count: 0.
- selected_abnormal_images: 4394; selected_no_finding_images: 500; selected_mixed_images: 0.
- image_type_label_mismatch_count: 0.

## Abnormal retention proof

- lost_abnormal_image_count: 0.
- abnormal_retention_rate: 1.0 (selected abnormal / source abnormal).
- All abnormal source images are retained in the controlled scope.

## No Finding image-level proof

- selected_no_finding_images: 500 unique image_id (image-level, not row-level).
- no_finding_row_level_sampling_used: False.
- See `phase1C_no_finding_selection_audit.csv` for the 500 audited image_ids.

## Class distribution summary

- abnormal_detection_classes_excluding_no_finding: 14.
- No Finding is absent from the detection-class distribution file by design.

## No Finding policy

- no_finding_is_detection_class: False.
- No Finding remains a negative image label without bounding boxes.

## Near-duplicate bbox candidates

- 147 near-duplicate bbox candidates (from Phase 1B) are **retained, not deleted**. Fusion of multi-radiologist boxes is a later research decision.

## Limitation

- Selected normal images are 500 out of 10606 available No Finding images; the negative pool is deliberately capped for the controlled scope.

## Boundary validation

- boundary_check_status: deferred_to_phase2A (image dimensions not read in Phase 1C).

## Forbidden actions confirmation

- split_created: False
- coco_created: False
- training_started: False
- pseudo_label_generated: False
- threshold_tuned: False
- test_set_used: False
- images_copied: False
- pixel_read: False
- dicom_header_read: False
- image_dimensions_read: False
- annotations_deleted_or_edited: False
- near_duplicate_bbox_deleted: False

## Next action

- **Send these outputs to GPT review BEFORE ticking the Phase 1C checklist.** This script does not auto-tick anything.
