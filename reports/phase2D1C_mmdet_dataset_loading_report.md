# Phase 2D.1C — MMDetection Dataset-Loading & Empty-Image Retention Validation

- Generated (UTC): 2026-07-30T11:57:48Z
- Overall status: **PASS**
- dataset_loading_validated: True
- empty_image_retention_validated: True
- dataset_training_ready: True
- training_authorized: False

## Environment
- python: 3.10.16
- python_executable: /content/miniconda/envs/mmdet330/bin/python
- torch: 2.1.0+cu118
- torchvision: 0.16.0+cu118
- cuda_runtime: 11.8
- cuda_available: True
- mmcv: 2.1.0
- mmengine: 0.10.7
- mmdet: 3.3.0

## Inputs
- repo_root: /content/ssl_detection_xray_v2
- ann_file: /content/ssl_detection_xray_v2/data/processed/coco/coco_master_jpg.json
- data_root: /content/ssl_detection_xray_v2/data/processed/images_jpg
- COCO SHA-256: f587152278f713460ff1e727a2912248a47052f6abc48de8f7bad6e8a63b94c0

## Counts (expected vs observed)

| Metric | Expected | Observed |
|---|---:|---:|
| Images | 4894 | 4894 |
| Annotations | 36096 | 36096 |
| Categories | 14 | 14 |
| Zero-GT images | 500 | 500 |

## Retention & controlled filtering

- Raw COCO image count: 4894
- filter_empty_gt=False length: 4894
- filter_empty_gt=True length: 4394
- Removed image count: 500
- Removed IDs equal zero-GT IDs: True

## Pipeline validation

- Abnormal samples audited: 4394
- Zero-GT samples audited: 500
- Abnormal pipeline pass: True
- Empty-GT pipeline pass: True
- Empty bbox shape observed: [0, 4]
- Empty label shape observed: [0]

## Dataloader validation

- Collate strategy: pseudo_collate
- Batch size: 1
- num_workers: 0
- Normal batch pass: True
- Forced empty-GT batch pass: True
- Zero-GT sample located in forced batch: True

## Checks

| Check | Critical | Passed |
|---|:--:|:--:|
| import_frameworks | True | True |
| version_mmdet | True | True |
| version_mmcv | True | True |
| version_mmengine | True | True |
| ann_file_exists | True | True |
| data_root_exists | True | True |
| coco_sha256_matches_recorded | True | True |
| coco_json_parse | True | True |
| coco_structure_analyzed | True | True |
| count_images | True | True |
| count_annotations | True | True |
| count_categories | True | True |
| count_zero_gt_images | True | True |
| no_duplicate_image_ids | True | True |
| no_duplicate_annotation_ids | True | True |
| no_duplicate_category_ids | True | True |
| no_invalid_image_refs | True | True |
| no_invalid_category_refs | True | True |
| all_referenced_jpg_exist | True | True |
| unique_resolved_jpg_paths | True | True |
| cocodataset_build_retention | True | True |
| retention_length_matches | True | True |
| retention_contains_all_zero_gt | True | True |
| cocodataset_build_filtered | True | True |
| filtered_removed_equals_zero_gt | True | True |
| abnormal_pipeline_sample | True | True |
| empty_pipeline_sample | True | True |
| empty_bbox_shape | True | True |
| empty_label_shape | True | True |
| bbox_label_valid | True | True |
| normal_dataloader_batch | True | True |
| forced_empty_dataloader_batch | True | True |

> NOTE: dataset_training_ready does not imply training_authorized. This phase never authorizes training.
