# Phase 1A — VinBigData Dataset Overview

_Generated 2026-06-19T13:04:56.929964+00:00 from `data\raw\vinbigdata\annotations\train.csv`._

> Scope: CSV-only statistics. No split, no COCO, no image reads, no training. 'No Finding' is a negative image label, not a detection class.

## Summary

- Total rows: **67914**
- Unique images: **15000**
- Abnormal images: **4394**
- No Finding images: **10606**
- Abnormal classes (excluding No Finding): **14**
- Bbox invalid count (total flags): **0**

## Class distribution

| class_name | No Finding? | rows | images |
|---|---|---|---|
| Aortic enlargement | False | 7162 | 3067 |
| Atelectasis | False | 279 | 186 |
| Calcification | False | 960 | 452 |
| Cardiomegaly | False | 5427 | 2300 |
| Consolidation | False | 556 | 353 |
| ILD | False | 1000 | 386 |
| Infiltration | False | 1247 | 613 |
| Lung Opacity | False | 2483 | 1322 |
| No finding | True | 31818 | 10606 |
| Nodule/Mass | False | 2580 | 826 |
| Other lesion | False | 2203 | 1134 |
| Pleural effusion | False | 2476 | 1032 |
| Pleural thickening | False | 4842 | 1981 |
| Pneumothorax | False | 226 | 96 |
| Pulmonary fibrosis | False | 4655 | 1617 |

## Bbox quality

- Has bbox columns: True
- Bbox rows considered: 36096
- Missing coordinate: 0
- x_min >= x_max: 0
- y_min >= y_max: 0
- Non-positive width/height: 0
- Valid bboxes: 36096

| dim | min | mean | max |
|---|---|---|---|
| width | 11.00 | 440.94 | 2938.00 |
| height | 3.00 | 391.40 | 2803.00 |
| area | 180.00 | 218415.71 | 4575318.00 |

## No Finding policy check

- No Finding labels recognized: ['no finding']
- No Finding rows with bbox: 0
- Abnormal rows missing bbox: 0
- Images with both No Finding and abnormal: 0

## Warnings

- None.

## Generated files

- `overview_json`: `reports/phase1A_dataset_overview.json`
- `class_distribution_csv`: `reports/phase1A_class_distribution.csv`
- `image_level_summary_csv`: `reports/phase1A_image_level_summary.csv`
- `bbox_quality_csv`: `reports/phase1A_bbox_quality_summary.csv`
- `report_md`: `reports/phase1A_dataset_overview.md`
