# Phase 2D.1A - Image Representation Protocol Decision

> **Final JPEG quality has not been selected.**  
> **Full DICOM-to-JPG conversion remains locked.**  
> **coco_master_jpg.json has not been created.**  
> **Dataset is not training-ready.**  
> **Training is not authorized.**

Protocol fingerprint (sha256): `1528da27758d35786847141c37d0ddb754dddb146aff116a8f3a9a7b07221229`

## 1. Executive decision

Phase 2D.1A locks the *image representation protocol* for converting the immutable raw DICOM source into a processed JPG training representation. This is a decision-only phase: no image is read, decoded, or written, and no numeric fidelity threshold is set. Two JPEG quality candidates (95 and 100) are carried forward to a Phase 2D.1B pilot; the final quality remains `null`. Protocol version `1.0.0`, status `decision_locked_pilot_pending`.

## 2. Current gate and readiness

Phase status: `OPEN_REVIEW_REQUIRED` (GPT review pending).

| Readiness flag | Value |
| --- | --- |
| jpg_training_representation_ready | false |
| coco_jpg_training_annotation_ready | false |
| mmdetection_dataset_loading_ready | false |
| empty_image_retention_ready | false |
| dataset_training_ready | false |
| training_authorized | false |

## 3. Locked input evidence

| Count | Value |
| --- | --- |
| images | 4894 |
| abnormal_images | 4394 |
| no_finding_images | 500 |
| annotations | 36096 |
| categories | 14 |
| no_finding_annotations | 0 |

Sources cross-checked: `data/processed/coco/coco_master.json`, `reports/phase2D_coco_master_validation.json`, `reports/phase2A_dicom_bbox_validation.json`, `reports/phase2B_canonical_schema_validation.json`.

## 4. Artifact roles

| Artifact | Role |
| --- | --- |
| `DICOM` | immutable_raw_medical_source |
| `JPG` | processed_training_representation |
| `coco_master.json` | official_annotation_master |
| `coco_master_jpg.json` | path-only training derivative |

## 5. Ordered pixel-transformation pipeline

The following order is authoritative for Phase 2D.1B and MUST NOT be reordered:

1. DICOM decode (single frame, MONOCHROME1/2 only)
2. Pixel padding mask build
3. Modality transformation (Modality LUT **or** rescale **or** identity)
4. VOI LUT / windowing (or theoretical modality-domain fallback)
5. Presentation polarity normalization to MONOCHROME2-equivalent
6. uint8 conversion (clip -> linear map -> rint -> clip -> cast)
7. Output channel handling (store 1-channel L; replicate to 3 at model load)
8. JPEG encoding (Pillow; quality pending pilot)

## 6. DICOM decoding policy

Documented only, executed in phase `2D.1B`. `force_read = false`, `single_frame_only = true`, `SamplesPerPixel must equal 1`. Allowed PhotometricInterpretation: MONOCHROME1, MONOCHROME2. Unsupported inputs = hard_fail.

Required future recording: TransferSyntaxUID, decoder_backend, Rows, Columns, BitsAllocated, BitsStored, HighBit, PixelRepresentation, SamplesPerPixel, PhotometricInterpretation, NumberOfFrames.

## 7. Modality LUT / Rescale policy

Branch: if a Modality LUT sequence is present, apply the Modality LUT; elif both RescaleSlope and RescaleIntercept are present, apply rescale; else identity. Do not apply both LUT and rescale sequentially. Exactly one of RescaleSlope/Intercept present = hard fail. Conflicting/ambiguous modality metadata = hard fail. Modality transformation occurs before VOI/windowing.

## 8. VOI LUT / Windowing policy

Branch: if a VOI LUT sequence exists, prefer the VOI LUT; elif valid WindowCenter and WindowWidth exist, use windowing; else use the theoretical modality-domain range fallback. Selected index = 0. Record all available values and respect VOILUTFunction. **Direct observed per-image min-max is forbidden.** **Automatic percentile clipping is forbidden.** The fallback is based on the theoretical stored/modality range, never per-image `arr.min()`/`arr.max()`.

## 9. Pixel padding and clipping

Build a padding mask from stored pixels using PixelPaddingValue and PixelPaddingRangeLimit when present. Padding pixels must not influence intensity statistics. Final padding value after MONOCHROME2 normalization = 0. uint8 conversion clips using theoretical output bounds only.

## 10. Presentation LUT / MONOCHROME1 policy

If PresentationLUTShape == INVERSE, invert once; elif PresentationLUTShape is absent and PhotometricInterpretation == MONOCHROME1, invert once; else no inversion. Output target: MONOCHROME2-equivalent polarity (low = dark, high = bright).

## 11. uint8 conversion

Steps: clip_using_theoretical_output_bounds -> linear_mapping_to_0_255 -> round_using_numpy_rint -> final_clip_0_255 -> cast_uint8. NaN/Inf = hard fail.

## 12. Output channel policy

JPG storage: JPEG mode L, one grayscale channel, uint8. MMDetection model input: three channels via grayscale replication in the loader; actual validation deferred to Phase 2D.1C.

## 13. JPEG candidates and pending decision

Encoder: Pillow. Quality candidates: [95, 100]. `final_quality = null` (`pending_phase2D1B_pilot`). `optimize = false`, `progressive = false`. Any lossless claim is forbidden. Future encoder environment to record: Python, Pillow, libjpeg, pydicom, numpy.

## 14. Geometry and bbox preservation

`resize = false`, `crop = false`, `rotation = false`, `flip = false`, `transpose = false`. Preserve width and height = true. `bbox_scaling_expected = false`, `bbox_scaling_validated = false`. Any dimension or orientation change = hard fail; do not automatically scale bbox.

## 15. Filename and path policy

JPG root: `data/processed/images_jpg`. JPG relative file name: `train/<image_id>.jpg`. COCO-JPG file_name: `train/<image_id>.jpg`. Absolute path in COCO-JPG: forbidden.

## 16. Traceability and hashes

Future mapping target: `data/processed/image_mapping/dicom_to_jpg_mapping.csv`.

Required future fields: original_image_id, canonical_image_id, coco_image_id, dicom_relative_path, jpg_relative_path, source_dicom_sha256, pre_jpeg_uint8_sha256, output_jpg_sha256, protocol_version, protocol_sha256, transfer_syntax_uid, decoder_backend, rows, columns, jpeg_quality, modality_branch, voi_branch, presentation_inversion_applied.

This protocol's fingerprint (sha256): `1528da27758d35786847141c37d0ddb754dddb146aff116a8f3a9a7b07221229`.

## 17. Pilot selection protocol

Minimum images: 64; minimum No Finding images: 16; tie-break seed: 2026; selection unit: image_id; selection: deterministic_coverage_first. Actual image IDs are NOT selected in Phase 2D.1A.

Coverage required across: all_14_abnormal_classes, minimum_and_maximum_dimensions_and_pixel_count, smallest_and_largest_bbox, all_photometric_interpretation_values, all_transfer_syntax_patterns, all_bits_stored_and_pixel_representation_patterns, rescale_slope_intercept_patterns, modality_lut_presence_and_absence, voi_lut_presence_and_absence, window_center_width_presence_and_absence, single_and_multi_valued_windows, presentation_lut_shape_patterns, pixel_padding_value_presence_and_absence.

if 64 images are insufficient, expand until all observed metadata strata are represented.

## 18. Fidelity validation

JPEG fidelity reference: pre-JPEG uint8 image; comparison: decoded JPG image. The raw DICOM-to-JPG difference must NEVER be described as JPEG compression error.

Whole-image metrics: MAE, RMSE, PSNR, SSIM, maximum_absolute_error, p95_absolute_error, p99_absolute_error.

BBox-ROI metrics: ROI_MAE, ROI_PSNR, ROI_SSIM, ROI_maximum_absolute_error.

Also required: file_size, compression_ratio_relative_to_pre_jpeg_uint8_bytes, full_image_visual_audit, bbox_crop_visual_audit, difference_heatmap.

No numeric PSNR/SSIM/MAE pass threshold is set in Phase 2D.1A.

## 19. Decision-only versus pilot-dependent fields

Locked now (decision-only): artifact roles, transformation branch logic and ordering, geometry/bbox preservation, path policy, quality candidates. Deferred to the pilot (2D.1B): final JPEG quality, observed metadata strata, numeric fidelity outcomes, and the selected pilot image IDs.

## 20. Thresholds not locked

No numeric PSNR, SSIM or MAE pass threshold is locked in this phase. `final_quality` must remain `null`.

## 21. Definition of Done

`required_policy_items_total = 20`, `required_policy_items_documented = 20`, `cross_output_drift_count = 0`, `final_jpeg_quality_is_pending = true`, `hard_errors = 0`, `dod_pass_candidate = true`.

## 22. Forbidden actions

| Forbidden action | Executed |
| --- | --- |
| full_conversion_run | false |
| full_jpg_dataset_created | false |
| coco_master_jpg_created | false |
| split_created | false |
| labeled_unlabeled_split_created | false |
| training_started | false |
| inference_run | false |
| pseudo_labels_generated | false |
| threshold_tuned | false |
| ap_map_computed | false |
| test_set_used | false |
| canonical_bbox_modified | false |
| coco_master_modified | false |

## 23. Remaining risks

Residual risks deferred to the pilot: unobserved TransferSyntax/Photometric strata, multi-valued windowing edge cases, lesion-region degradation at quality 95, and encoder-environment (libjpeg) variance. Each is mitigated by the coverage-first pilot and the fidelity metric suite before any full conversion.

## 24. Next gate

Phase 2D.1B (locked): implement the documented decoder + encoder on the coverage-first pilot, record all metadata strata, compute the fidelity metrics, and select the final JPEG quality. Full conversion stays blocked until GPT review concludes Phase 2D.1A PASS.

