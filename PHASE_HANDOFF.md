# PHASE HANDOFF — `ssl_detection_xray_v2`

Ngày cập nhật: 2026-07-28

Dự án: **Nghiên cứu học bán giám sát cho dò tìm bất thường trên X-quang phổi**

Bài toán: **Semi-supervised object detection trên VinBigData Chest X-ray**

---

## 1. Vai trò làm việc

- Người nghiên cứu: quyết định hướng nghiên cứu, protocol, phạm vi thí nghiệm.
- GPT: thiết kế quy trình, phản biện logic, review evidence, quyết định pass/fail DoD.
- Claude: viết code trong repo theo prompt được giao.
- Python: chạy script, kiểm tra dữ liệu, train/evaluate và tạo evidence.

Quy trình bắt buộc:

```text
script → output → DoD → GPT review → người nghiên cứu tick checklist
```

Không tick checklist nếu chưa có evidence.

---

## 2. Nguyên tắc nghiên cứu đã khóa

- Không nhảy phase.
- Không train khi data/split/COCO/No Finding/seed/checkpoint criterion chưa pass DoD.
- Không dùng test set để tune threshold.
- Không dùng test set để chọn checkpoint.
- Không dùng test set để chọn model/backbone.
- Không dùng test set để quyết định augmentation.
- `No Finding` là ảnh âm tính không có bbox, không phải detection class.
- Metric chính: `mAP@0.5:0.95`.
- Metric phụ: `AP50`, `AP75`, class-wise AP, recall/sensitivity, FP/image, FP per negative image.
- Supervised và SSL phải dùng cùng labeled split, cùng `split_seed`, cùng fixed test set.
- Stability phải dùng nhiều `training_seed`.
- Local environment hiện tại không training-ready.
- DICOM là nguồn ảnh y khoa gốc, bất biến và không bị thay thế.
- JPG chất lượng cao là processed training image representation dự kiến cho MMDetection.
- Việc chuyển DICOM sang JPG phải dùng protocol cố định, có version và có validation.
- Không được chuyển trực tiếp `pixel_array` sang `uint8` nếu chưa khóa:
  - `RescaleSlope` / `RescaleIntercept`;
  - modality LUT;
  - VOI LUT hoặc windowing;
  - `MONOCHROME1` inversion;
  - intensity clipping;
  - JPEG quality;
  - channel policy.
- Không resize, crop hoặc rotate trong bước DICOM-to-JPG nếu chưa có protocol biến đổi bbox tương ứng.
- `coco_master.json` tiếp tục là annotation master.
- `coco_master_jpg.json` sẽ là training derivative, chỉ thay đổi representation path, không thay đổi annotation semantics.

---

## 3. Trạng thái hiện tại

## 3. Trạng thái hiện tại

```text
Current subphase:
Phase 2D.1B-Full —
Full Controlled-Scope DICOM-to-JPG Conversion & Validation

Overall phase:
Phase 2D.1 — JPG Training Representation &
MMDetection Empty-Image Loading Validation: IN PROGRESS

Previous subphase:
Phase 2D.1B-Pilot — Representative DICOM-to-JPG Pilot:
CLOSED / PASS

Final JPEG quality:
95 / LOCKED

Phase 0 core: PASS
Phase 0 local training framework: DEFERRED
Phase 1A — Dataset Overview: PASS
Phase 1B — Annotation Quality: PASS
Phase 1C — Dataset Scope Decision: PASS
Phase 1D — Label Reliability & Kappa Feasibility: PASS
Phase 2A — Data Standardization / Image-Boundary Validation: PASS
Phase 2B — Canonical Detection Annotation Schema: PASS
Phase 2C — Framework & Format Decision / COCO Conversion Planning: PASS
Phase 2D — COCO Master Conversion & Validation: CLOSED / PASS

Phase 2D.1A: CLOSED / PASS
Phase 2D.1B: IN PROGRESS
Phase 2D.1B-Pilot: CLOSED / PASS
Phase 2D.1B-Full: OPEN / CURRENT
Phase 2D.1C: LOCKED until Phase 2D.1B-Full PASS
Phase 2D.1D: LOCKED until Phase 2D.1C PASS

Pilot implementation:
V6_FROZEN

Pilot guardrail tests:
139/139 PASS

Pilot image count:
64

Pilot No Finding image count:
16

Metadata/features coverage:
54/54 PASS

Abnormal class coverage:
14/14 PASS

Pixel decoding:
64/64 PASS

Geometry preservation:
PASS

BBox invariance:
PASS

Final JPEG quality:
95

Full conversion authorized:
TRUE

Git status:
Phase 2D committed and pushed to origin/main.
Phase 2D commit: 1a3f7a7.
Phase 2D.1A and Phase 2D.1B-Pilot evidence reviewed locally.
Synchronized documentation update and commit/push: PENDING.

jpg_training_representation_ready: FALSE
coco_jpg_training_annotation_ready: FALSE
mmdetection_dataset_loading_ready: FALSE
empty_image_retention_ready: FALSE
dataset_training_ready: FALSE
training_authorized: FALSE
```

Được mở / hiện tại:

```text
Phase 2D.1B-Full —
Full Controlled-Scope DICOM-to-JPG Conversion & Validation

Environment:
Local
```

Next gated phase:

```text
Phase 2D.1C —
MMDetection Dataset / Empty-Image Loading Validation

Environment:
Google Colab
```

Chưa được làm:

```text
MMDetection dataset loading validation
MMDetection empty-GT retention validation
Train/val/test split
Labeled/unlabeled split
Supervised detector training
SSL detector training
Detector inference
Pseudo-label generation
Threshold tuning
AP/mAP computation
Test-set usage
Checkpoint/model/backbone selection
```

Phase 2D.1 structure:

```text
2D.1A — Image Representation Protocol Decision
Environment: Local
Status: CLOSED / PASS

2D.1B — DICOM-to-JPG Conversion & Validation
Environment: Local
Status: IN PROGRESS

2D.1B-Pilot — Representative DICOM-to-JPG Pilot
Status: CLOSED / PASS

Final JPEG quality:
95 / LOCKED

2D.1B-Full — Full 4,894-image conversion
Status: OPEN / CURRENT

2D.1C — MMDetection Dataset / Empty-Image Loading Validation
Environment: Google Colab
Status: LOCKED until 2D.1B-Full PASS

2D.1D — Evidence Consolidation, GPT Review & Closure
Environment: Local
Status: LOCKED until 2D.1C PASS
```

Representation roles:

```text
DICOM:
Immutable raw medical source and source evidence.

JPG:
Processed training image representation generated by a fixed,
versioned and reproducible protocol.

coco_master.json:
Official annotation master linked to the original DICOM representation.

coco_master_jpg.json:
Training derivative linked to JPG file_name values and permitted
to change only the image representation path.

MMDetection:
Downstream framework for JPG dataset loading, detector training,
evaluation and later SSOD experiments.
```

Ghi chú trạng thái:

```text
Phase 2D.1A đã khóa protocol DICOM-to-JPG version 1.0.0.

Phase 2D.1B-Pilot đã:
- đọc header 4,894/4,894 DICOM;
- chọn 64 representative pilot images;
- bao gồm 16 No Finding images;
- bao phủ 54/54 metadata/features;
- bao phủ 14/14 abnormal classes;
- decode pixel thành công 64/64;
- tạo paired JPEG quality 95 và 100;
- tính whole-image fidelity;
- tính bbox-ROI fidelity trên 402 bbox;
- validate geometry và bbox invariance;
- thực hiện visual review;
- khóa final JPEG quality = 95.

Quality 100 có numerical fidelity cao hơn.

Quality 95 được chọn vì vẫn giữ fidelity cao nhưng giảm projected
storage khoảng 48.79% so với quality 100.

Phase 2D.1B-Pilot chỉ cho phép mở full conversion.

Dataset vẫn chưa training-ready.

Training vẫn chưa được authorize.
```

---


## 4. Phase 0 — Kết quả bàn giao

### 4.1 Phase 0A — Repo structure

Trạng thái: **PASS**

Đã có các thư mục chính:

```text
configs/protocol/
data/raw/
data/interim/
data/processed/
data/manifests/
src/utils/
scripts/
experiments/
reports/
plots/
models/
logs/
tests/
draft/
```

Đã có tài liệu:

```text
README.md
CLAUDE.md
STRUCTURE.md
RESEARCH_CHECKLIST.md
repository_structure.md
research_log.md
PHASE_HANDOFF.md
```

Đã có protocol:

```text
configs/protocol/checkpoint_policy.yaml
```

### 4.2 Phase 0B — Core environment

Trạng thái: **PASS core / DEFER training framework**

Evidence đã có:

```text
reports/phase0_environment_check.json
reports/phase0_pip_freeze.txt
reports/reproducibility_settings.md
data/manifests/seed_state_manifest.json
```

Kết quả chính:

```text
Python: 3.10.20
Conda env: sslxray
PyTorch: 2.3.1
torchvision: 0.18.1
numpy: 1.24.3
pandas: 2.3.3
OpenCV/cv2: 4.11.0
pydicom: 3.0.2
pycocotools: 2.0.11
pip check: No broken requirements found
pytest tests/test_phase0.py -q: 5 passed
CUDA available: false
mmengine/mmcv/mmdet: not installed
framework_import_ok: false
```

Quyết định:

```text
Local environment chỉ dùng cho repo/data/script/report validation.
MMDetection/GPU training environment sẽ setup riêng ở remote/GPU.
```

---

## 5. Seed và reproducibility

Seed mặc định Phase 0:

```text
seed = 2026
```

Đã ghi nhận trong:

```text
data/manifests/seed_state_manifest.json
reports/phase0_environment_check.json
reports/reproducibility_settings.md
```

Deterministic flags:

```text
PYTHONHASHSEED = 2026
Python random seed: enabled
NumPy seed: enabled
PyTorch CPU seed: enabled
PyTorch CUDA seed: not applied because CUDA unavailable
torch.use_deterministic_algorithms = true
torch.backends.cudnn.deterministic = true
torch.backends.cudnn.benchmark = false
CUBLAS_WORKSPACE_CONFIG = :4096:8
```

---

## 6. Checkpoint/evaluation protocol đã khóa

File:

```text
configs/protocol/checkpoint_policy.yaml
```

Protocol:

```text
Primary metric: mAP@0.5:0.95
Checkpoint selection split: val
Test usage: final_evaluation_only
```

Cấm:

```text
Dùng test set để tune threshold.
Dùng test set để chọn checkpoint.
Dùng test set để chọn model/backbone.
Dùng test set để quyết định augmentation.
```

---

## 7. Phase progress summary

### Phase 1A — Dataset Overview

Status: **PASS**

Date: 2026-06-19

Scripts run:

```cmd
python scripts\01A_dataset_overview.py --train-csv data\raw\vinbigdata\annotations\train.csv
```

Key findings:

```text
Total images: 15,000
Annotation rows: 67,914
Abnormal images: 4,394
No Finding images: 10,606
Abnormal bbox rows: 36,096
Abnormal detection classes excluding No Finding: 14
Invalid bbox count: 0
No Finding rows with bbox: 0
Mixed No Finding + abnormal images: 0
```

Research decision:

```text
Full 15,000-image CSV is source metadata only.
Downstream controlled working scope will be locked later to 4,894 images:
4,394 abnormal images + 500 No Finding images.
```

Forbidden actions avoided:

```text
No split.
No COCO conversion.
No image read/copy.
No training.
No pseudo-labeling.
No threshold tuning.
No test-set usage.
```

---

### Phase 1B — Annotation Quality

Status: **PASS**

Date: 2026-06-19

Scripts run:

```cmd
python scripts\01B_annotation_quality.py --train-csv data\raw\vinbigdata\annotations\train.csv
```

Outputs generated:

```text
reports/phase1B_annotation_quality.json
reports/phase1B_annotation_quality.md
reports/annotation_sanity_report.md
reports/invalid_bbox_rows.csv
reports/duplicate_bbox_candidates.csv
reports/phase1B_class_mapping.csv
reports/phase1B_bbox_quality_by_class.csv
reports/phase1B_image_label_consistency.csv
```

DoD result:

```text
Annotation-level bbox sanity: PASS
No Finding policy: PASS
Abnormal bbox completeness: PASS
Class mapping consistency: PASS
Duplicate/near-duplicate candidates reported: PASS
Boundary check: DEFERRED to image-level validation because train.csv has no image dimensions
Forbidden actions avoided: PASS
```

Key findings:

```text
Total rows: 67,914
Unique images: 15,000
Abnormal rows: 36,096
No Finding rows: 31,818
Abnormal images: 4,394
No Finding images: 10,606
Abnormal detection classes excluding No Finding: 14
Invalid bbox total: 0
No Finding rows with bbox: 0
Abnormal rows missing bbox: 0
Mixed No Finding + abnormal images: 0
Class mapping issues: 0
Exact duplicate candidates: 0
Near-duplicate candidates IoU >= 0.95: 147
Boundary check status: not_evaluable_without_image_dimensions
```

Research decisions:

```text
Do not delete or modify near-duplicate bbox candidates in Phase 1B.
Treat near-duplicate boxes as multi-radiologist annotation candidates requiring later fusion/handling decision.
Defer image-boundary validation to Phase 2A because Phase 1B is CSV-only.
```

---

### Phase 1C — Dataset Scope Decision

Status: **PASS**

Date: 2026-07-01

Scripts run:

```cmd
python scripts\01C_dataset_scope_decision.py ^
  --train-csv D:\ssl_detection_xray_v2\data\raw\vinbigdata\annotations\train.csv ^
  --manifest-glob "D:\ssl_detection_xray\data\raw\vinbigdata\dicom_subset_chunks\dicom_package_manifest_part_*.csv" ^
  --dicom-root D:\ssl_detection_xray\data\raw\vinbigdata\dicom_subset\train ^
  --chunk-summary D:\ssl_detection_xray\data\raw\vinbigdata\dicom_subset_chunks\dicom_chunk_summary.csv
```

Outputs generated:

```text
reports/scope_decision.md
reports/phase1C_dataset_scope_decision.json
reports/phase1C_scope_class_distribution.csv
reports/phase1C_image_level_scope_summary.csv
reports/phase1C_no_finding_selection_audit.csv
data/manifests/phase1C_selected_images_manifest.csv
data/manifests/phase1C_downloaded_image_inventory.csv
data/manifests/phase1C_combined_package_manifest.csv
data/interim/vinbigdata_phase1C_scope_annotations.csv
```

DoD result:

```text
Dataset scope decision: PASS
Image-level manifest: PASS
DICOM filename inventory cross-check: PASS
Package manifest cross-check: PASS
train.csv metadata cross-check: PASS
Abnormal retention: PASS
No Finding image-level handling: PASS
Class distribution report: PASS
Forbidden actions avoided: PASS
```

Key findings:

```text
Selected total images: 4,894
Selected abnormal images: 4,394
Selected No Finding images: 500
Selected mixed No Finding + abnormal images: 0
Lost abnormal image count: 0
Abnormal retention rate: 1.0
Selected subset rows: 37,596
Selected abnormal rows: 36,096
Selected No Finding rows: 1,500
Abnormal detection classes excluding No Finding: 14
No Finding is detection class: false
No Finding row-level sampling used: false
```

Research decisions:

```text
Controlled working scope is officially locked to 4,894 image-level samples:
4,394 abnormal images + 500 No Finding images.

The 500 No Finding samples are selected and verified at image_id level, not row level.
The metadata-only subset annotation CSV is created for selected image_id values only.
No Finding remains a negative image label, not a detection class.
```

Issues / risks:

```text
The controlled scope uses 500 out of 10,606 No Finding images, not the full No Finding pool.
This is a deliberate controlled-scope design decision and must be stated as a limitation.
Boundary validity is not concluded in Phase 1C because image dimensions were not read.
147 near-duplicate bbox candidates from Phase 1B are retained, not deleted or fused.
Fusion/handling of multi-radiologist boxes is deferred to a later phase.
```

---

### Phase 1D — Label Reliability & Kappa Feasibility

Status: **PASS**

Date: 2026-07-01

Scripts run:

```cmd
python scripts\01D_kappa_feasibility.py
```

Outputs generated:

```text
reports/phase1D_kappa_feasibility.md
reports/phase1D_kappa_feasibility.json
reports/phase1D_classwise_agreement_feasibility.csv
reports/phase1D_radiologist_per_image.csv
reports/phase1D_rare_class_kappa_instability.csv
```

DoD result:

```text
rad_id availability: PASS
radiologists per image: PASS
image-class-radiologist binary matrix feasibility: PASS
Cohen's Kappa feasibility: PASS
Fleiss' Kappa feasibility: PASS
class-wise image-level agreement: PASS
rare-class kappa instability risk: PASS
label-level agreement vs bbox-level consistency: PASS
forbidden actions avoided: PASS
```

Key findings:

```text
rad_id_available: true
rad_id_missing_count: 0
total_images: 4894
total_rows: 37596
radiologists_total: 17
radiologists_per_image_distribution: {'3': 4894}
uniform_rater_count_per_image: true
same_rater_identity_panel_across_images: false
binary_matrix_feasible: true
cohen_kappa_feasible: false
fleiss_kappa_feasible: true
overall_fleiss_kappa_mean: 0.4879
rare_class_instability_summary: 5/14 classes carry kappa_instability_risk
label_level_agreement_status: evaluable_fleiss_computed
bbox_level_consistency_status: evaluated_descriptive_only
```

Research decisions:

```text
Fleiss' Kappa is computed at image-level class agreement.
Cohen's Kappa is not used as the main agreement statistic because each image has 3 radiologist ratings.
Kappa/agreement is used only as data-quality evidence and limitation evidence.
Kappa is not a model metric and is not used for split/model/threshold/training/pseudo-labeling.
BBox-level consistency is kept separate from label-level agreement and remains descriptive only.
```

---

### Phase 2A — Data Standardization / Image-Boundary Validation

Status: **PASS**

Date: 2026-07-08

Scripts run:

```cmd
python scripts\02A_dicom_bbox_boundary_validation.py ^
  --annotations-csv data\interim\vinbigdata_phase1C_scope_annotations.csv ^
  --manifest-csv data\manifests\phase1C_selected_images_manifest.csv ^
  --dicom-root D:\ssl_detection_xray\data\raw\vinbigdata\dicom_subset\train
```

Outputs generated:

```text
reports/phase2A_dicom_bbox_validation.md
reports/phase2A_dicom_bbox_validation.json
reports/phase2A_image_metadata.csv
reports/phase2A_image_availability.csv
reports/phase2A_bbox_boundary_validation.csv
reports/phase2A_invalid_bbox_candidates.csv
reports/phase2A_dicom_read_errors.csv
```

DoD result:

```text
DICOM availability: PASS
DICOM metadata/header read: PASS
Image dimension extraction: PASS
BBox boundary validation: PASS
No Finding policy: PASS
Forbidden actions avoided: PASS
```

Key findings:

```text
DICOM files indexed under root: 4,894
DICOM available/missing: 4,894 / 0
DICOM read success/error: 4,894 / 0
Image dimensions available/missing: 4,894 / 0
Abnormal bbox rows checked: 36,096
BBox boundary valid/invalid: 36,096 / 0
No Finding rows with bbox: 0
Abnormal rows missing bbox: 0
dod_pass_candidate: true
```

Research decisions:

```text
All 4,894 controlled-scope DICOM files are available and readable at metadata/header level.
All 36,096 abnormal bbox rows are valid within original image boundaries under xyxy convention.
No Finding remains a negative image label without bbox and is not a detection class.
No bbox was edited, clamped, deleted or fused.
No image was copied, converted, normalized or saved as processed training data.
```

Issues / risks:

```text
Pixel array decoding was not checked in the main run because pixel_array_checked=false.
Framework dataloader / empty image loading is not checked yet.
Dataset is not training-ready until schema/COCO/split/loading phases pass DoD.
```

---

### Phase 2B — Canonical Detection Annotation Schema

Status: **PASS**

Date: 2026-07-08

Scripts run:

```cmd
python scripts\02B_build_canonical_schema.py ^
  --annotations-csv data\interim\vinbigdata_phase1C_scope_annotations.csv ^
  --manifest-csv data\manifests\phase1C_selected_images_manifest.csv ^
  --image-metadata-csv reports\phase2A_image_metadata.csv ^
  --bbox-boundary-csv reports\phase2A_bbox_boundary_validation.csv ^
  --output-dir data\processed\canonical ^
  --report-md reports\phase2B_canonical_schema_report.md ^
  --validation-json reports\phase2B_canonical_schema_validation.json ^
  --no-finding-audit-csv reports\phase2B_no_finding_policy_audit.csv ^
  --schema-errors-csv reports\phase2B_schema_consistency_errors.csv
```

Outputs generated:

```text
data/processed/canonical/canonical_image_table.csv
data/processed/canonical/canonical_bbox_table.csv
data/processed/canonical/canonical_class_mapping.csv
reports/phase2B_canonical_schema_report.md
reports/phase2B_canonical_schema_validation.json
reports/phase2B_no_finding_policy_audit.csv
reports/phase2B_schema_consistency_errors.csv
```

DoD result:

```text
Canonical image table: PASS
Canonical bbox table: PASS
Canonical class mapping: PASS
No Finding policy audit: PASS
Schema consistency validation: PASS
Portable path policy: PASS
Forbidden actions avoided: PASS
```

Key findings:

```text
canonical_image_rows: 4894
canonical_image_unique_images: 4894
canonical_bbox_rows: 36096
canonical_class_count: 14
abnormal_images: 4394
no_finding_images: 500
no_finding_policy_pass: true
no_finding_in_detection_classes: false
bbox_without_image_count: 0
image_without_metadata_count: 0
bbox_missing_dimension_count: 0
bbox_invalid_count: 0
class_mapping_issue_count: 0
schema_error_count: 0
portable_path_policy_pass: true
relative_dicom_path_absolute_count: 0
path_root_variable: VINBIGDATA_DICOM_ROOT
dod_pass_candidate: true
```

Research decisions:

```text
Canonical schema is accepted as the intermediate detection annotation schema.
Canonical image table keeps all 4,894 controlled-scope images.
Canonical bbox table keeps all 36,096 abnormal bbox rows.
Canonical class mapping contains exactly 14 abnormal detection classes.
No Finding remains a negative image-level sample with no bbox and is not a detection class.
No Finding is excluded from canonical bbox annotations and detection class mapping.
BBox format remains xyxy_original_image.
No bbox was edited, clamped, deleted, fused, or converted.
147 near-duplicate bbox candidates remain retained; fusion/handling remains deferred.
Portable path policy is adopted: downstream should resolve image files using VINBIGDATA_DICOM_ROOT + relative_dicom_path.
Phase 2B is not a COCO dataset and not a train/val/test split.
Dataset is still not training-ready until COCO/split/loading phases pass DoD.
```

Issues / risks:

```text
local_dicom_path stores absolute local evidence paths but must not be used as canonical downstream identifiers.
Remote/GPU environments must set VINBIGDATA_DICOM_ROOT or an equivalent data-root config.
Framework dataloader validation has not been performed.
Empty image loading check has not been performed.
COCO conversion has not been performed.
Train/val/test split has not been created.
Near-duplicate bbox handling is still deferred.
```

---

### Phase 2C — Framework & Format Decision / COCO Conversion Planning

Status: **PASS**

Date: 2026-07-14

Scripts run:

```cmd
python scripts\02C_framework_format_decision.py
```

Outputs generated:

```text
reports/phase2C_framework_format_decision.md
reports/phase2C_framework_format_decision.json
configs/framework/main_framework.yaml
configs/dataset/coco_paths.yaml
configs/protocol/coco_conversion_policy.yaml
```

DoD result:

```text
Framework decision: PASS
Framework rationale: PASS
Format comparison: PASS
COCO planning: PASS
No Finding / empty image policy: PASS
BBox conversion policy: PASS
Category id policy: PASS
Path portability policy: PASS
DICOM loader risk: PASS
Metric readiness policy: PASS
Forbidden actions avoided: PASS
No COCO master created: PASS
Dataset training-ready claim avoided: PASS
```

Key findings:

```text
primary_framework: MMDetection
fallback_framework: Detectron2_optional
primary_annotation_format: COCO_detection_JSON
source_schema: canonical_detection_schema
canonical_image_rows: 4894
canonical_image_unique_images: 4894
canonical_bbox_rows: 36096
canonical_class_count: 14
abnormal_images: 4394
no_finding_images: 500
actual_coco_conversion_done: false
dataset_training_ready: false
dod_pass_candidate: true
```

#### Format comparison decision

```text
COCO detection JSON: CHOSEN
YOLO txt: REJECTED
Pascal VOC XML: REJECTED
JSONL/custom: REJECTED
```

COCO được chọn vì phù hợp nhất với MMDetection/SSOD, hỗ trợ ảnh No Finding không annotation, tương thích COCO mAP@0.5:0.95 / pycocotools, hỗ trợ category metadata và traceability.

#### Framework selection rationale

```text
MMDetection: CHOSEN
Detectron2: FALLBACK_ONLY
YOLO-based framework: REJECTED
Custom PyTorch / torchvision: REJECTED
```

MMDetection được chọn làm framework chính không chỉ vì phổ biến, mà vì khớp trực tiếp nhất với pipeline luận văn:

```text
COCO-based object detection
COCO mAP@0.5:0.95 evaluation
teacher-student semi-supervised object detection
labeled/unlabeled data handling
config-driven reproducibility
pseudo-label workflow compatibility
class-wise AP / AP50 / AP75 readiness
lower custom training/evaluation code burden
```

Detectron2 được giữ làm fallback vì là framework detection mạnh và có COCO/custom dataset support, nhưng teacher-student SSOD pipeline trong project này sẽ cần nhiều custom implementation hơn.

YOLO-based framework bị loại khỏi vai trò framework chính vì annotation/evaluation pipeline thiên về YOLO-native, lệch khỏi COCO master + MMDetection SSOD protocol, và negative-image handling dựa trên empty label file có rủi ro với 500 ảnh No Finding.

Custom PyTorch/torchvision bị loại vì đòi hỏi tự viết dataset, dataloader, evaluator, trainer, pseudo-label loop, EMA teacher, COCO metric integration, logging và config protocol; rủi ro engineering và reproducibility sẽ lấn át đóng góp nghiên cứu.

#### Research decisions

```text
Primary framework is MMDetection.
Detectron2 is fallback optional only after GPT re-review.
Primary annotation format is COCO_detection_JSON.
Canonical schema Phase 2B remains the source of truth.
Actual COCO conversion belongs to Phase 2D.
COCO images should include all 4,894 controlled-scope images.
COCO annotations should include only 36,096 abnormal bboxes.
COCO categories should include only 14 abnormal detection classes.
No Finding remains a negative image with zero annotations and is not a detection class.
No background class is created.
BBox conversion for Phase 2D is xyxy_original_image → coco_xywh_absolute.
COCO category_id should be contiguous integer 1..14.
Path resolution should use VINBIGDATA_DICOM_ROOT + relative_dicom_path.
```

#### Metric readiness policy

```text
Phase 2C does not compute AP metrics.
Primary metric remains mAP@0.5:0.95.
Secondary diagnostics planned: AP50, AP75, class-wise AP, recall/sensitivity, FP/image, FP per negative image.
Metrics are computable only after COCO conversion, fixed split creation, model training and prediction generation.
Test-set metric is forbidden for checkpoint selection, threshold tuning, model selection and augmentation decisions.
```

#### Issues / risks

```text
MMDetection stack is not importable locally; this is expected because local training framework is deferred.
Remote/GPU environment is still required for detector training.
DICOM loader is not validated.
Empty image loading is not validated.
If filter_empty_gt is configured incorrectly, 500 No Finding images may be silently dropped.
COCO annotation format alone does not make the dataset training-ready.
```

Forbidden actions confirmed:

```text
No COCO master JSON created.
No train/val/test split created.
No labeled/unlabeled split created.
No training started.
No inference run.
No pseudo-label generated.
No threshold tuned.
No test set used.
No pixel_array read.
No image copied or converted.
No bbox modified, clamped, deleted or fused.
No source annotation modified.
No Phase 2B canonical schema modified.
No AP metrics computed.
```

Next phase:

```text
Phase 2D — COCO Master Conversion & Validation
```

Phase 2D chỉ được mở sau khi Phase 2C evidence đã commit và push GitHub.

---

### Phase 2D — COCO Master Conversion & Validation

Status: **PASS**

Date: 2026-07-14

Scripts run:

```cmd
python scripts\02D_build_coco_master.py
python -m unittest discover -s tests -p "test_phase2D_guardrails.py" -v
python -m json.tool data\processed\coco\coco_master.json > NUL
python -c "from pycocotools.coco import COCO; c=COCO(r'data\processed\coco\coco_master.json'); print(len(c.imgs), len(c.anns), len(c.cats))"
```

Outputs generated:

```text
data/processed/coco/coco_master.json
configs/protocol/phase2D_coco_master_validation.yaml
reports/phase2D_coco_master_validation.json
reports/phase2D_coco_master_validation.md
reports/phase2D_coco_image_annotation_counts.csv
reports/phase2D_coco_category_summary.csv
reports/phase2D_coco_invalid_annotations.csv
reports/phase2D_coco_no_finding_audit.csv
tests/test_phase2D_guardrails.py
```

DoD result:

```text
COCO master conversion: PASS
COCO image coverage: PASS
COCO annotation preservation: PASS
COCO category policy: PASS
BBox xyxy-to-xywh conversion: PASS
BBox boundary validation: PASS
Area validation: PASS
No Finding policy: PASS
Reference integrity: PASS
Traceability preservation: PASS
One-to-one bbox preservation: PASS
Strict JSON validation: PASS
pycocotools loading: PASS
Strict YAML protocol validation: PASS
Protocol drift protection: PASS
Atomic output promotion: PASS
Guardrail unit tests: PASS
Forbidden actions avoided: PASS
GPT review: PASS
```

Key findings:

```text
COCO images: 4,894
COCO annotations: 36,096
COCO categories: 14
Abnormal images: 4,394
No Finding images: 500
No Finding annotations: 0

Invalid annotations: 0
Boundary violations: 0
Area mismatches: 0
Broken image references: 0
Broken category references: 0
Absolute paths: 0

Image IDs unique and contiguous: true
Annotation IDs unique and contiguous: true
Category IDs contiguous 1..14: true
Category ID 0 present: false

No Finding in categories: false
Background in categories: false

Coordinate mismatches: 0
Image mapping mismatches: 0
Category mapping mismatches: 0
Missing canonical annotations: 0
Duplicated canonical annotations: 0
Extra COCO annotations: 0
One-to-one preservation: true

JSON parse: PASS
pycocotools load: PASS
Protocol strict load: PASS
Protocol drift count: 0
Pre-promotion checks: PASS
Atomic promotion: PASS
Guardrail unit tests: 22/22 PASS
Hard errors: 0
Warnings: 0
dod_pass_candidate: true
dataset_training_ready: false
```

Research decisions:

```text
The Phase 2B canonical detection schema is accepted as the source of truth for COCO conversion.

data/processed/coco/coco_master.json is accepted as the official controlled-scope COCO master.

All 4,894 controlled-scope images are retained in COCO images.

All 36,096 abnormal canonical bbox rows are retained one-to-one in COCO annotations.

COCO categories contain exactly 14 abnormal detection classes.

No Finding remains an image-level negative:
- retained in COCO images;
- zero annotations;
- excluded from COCO categories.

No background category is created.

BBox format is converted from xyxy_original_image to coco_xywh_absolute:
[x, y, width, height].

Area is calculated as width * height.
iscrowd is fixed to 0.

No bbox was clamped, deleted, fused, rounded, or processed using NMS.

COCO file_name uses relative_dicom_path and does not contain absolute local paths.

The Phase 2D YAML protocol is strict-loaded and cross-checked against Phase 2B validation and the canonical tables.

The final COCO output is atomically promoted only after all required hard validations pass.
```

Issues / risks:

```text
A valid COCO annotation file does not make the dataset training-ready.

DICOM pixel decoding has not been validated in a framework-compatible loading pipeline.

MMDetection default LoadImageFromFile must not be assumed to support .dicom files.

No Finding / empty-image loading has not been validated using the framework dataset pipeline.

filter_empty_gt=False or an equivalent mechanism has not been validated.

Train/val/test split has not been created.

Labeled/unlabeled SSL subsets have not been created.

Training, inference, pseudo-labeling, threshold tuning, AP/mAP computation, and test-set usage remain locked.
```

Forbidden actions confirmed:

```text
No DICOM file read.
No DICOM file existence check.
No DICOM header read.
No pixel_array read.
No pydicom, cv2, or PIL image loading.
No image copying or conversion.
No train/val/test split created.
No labeled/unlabeled split created.
No MMDetection or Detectron2 dataset loaded.
No filter_empty_gt validation performed.
No training started.
No inference run.
No pseudo-label generated.
No threshold tuned.
No test set used.
No AP/mAP computed.
No canonical schema modified.
No source annotation modified.
No bbox clamped, deleted, fused, rounded, or processed using NMS.
No dataset training-ready claim made.
```

Next phase:

```text
Phase 2D.1A — Image Representation Protocol Decision
```

Overall phase:

```text
Phase 2D.1 — JPG Training Representation &
MMDetection Empty-Image Loading Validation
```

Opening condition:

```text
Phase 2D committed and pushed: true
Phase 2D commit: 1a3f7a7
Phase 2D.1A may begin: true
```

---

### Phase 2D.1 — JPG Training Representation & MMDetection Empty-Image Loading Validation

Status: **IN PROGRESS**

Date opened: 2026-07-15

Current subphase:

```text
Phase 2D.1B-Full —
Full Controlled-Scope DICOM-to-JPG Conversion & Validation

Environment: Local
Status: OPEN / CURRENT
```

Subphase status:

```text
Phase 2D.1A — Image Representation Protocol Decision:
CLOSED / PASS

Phase 2D.1B-Pilot — Representative DICOM-to-JPG Pilot:
CLOSED / PASS

Phase 2D.1B-Full — Full Controlled-Scope Conversion:
OPEN / CURRENT

Phase 2D.1C — MMDetection Dataset / Empty-Image Loading Validation:
LOCKED

Phase 2D.1D — Evidence Consolidation, GPT Review & Closure:
LOCKED
```

Phase 2D.1A result:

```text
Protocol version: 1.0.0
JPEG quality candidates: [95, 100]
Required policy coverage: 20/20
Guardrail tests: 31/31 PASS
Status: CLOSED / PASS
```

Phase 2D.1B-Pilot implementation:

```text
Implementation version: V6_FROZEN
Guardrail tests: 139/139 PASS
```

Phase 2D.1B-Pilot result:

```text
DICOM paths resolved: 4,894/4,894
DICOM headers read: 4,894/4,894

Pilot images: 64
No Finding pilot images: 16

Metadata/features coverage: 54/54
Abnormal class coverage: 14/14

Pixel decode attempts: 64
Pixel decode successes: 64
Pixel decode errors: 0

Geometry preservation: PASS
BBox invariance: PASS
Critical visual failure: false

Final JPEG quality: 95
Full conversion authorized: true

Pilot status: CLOSED / PASS
```

#### So sánh fidelity và dung lượng

| Tiêu chí | JPEG quality 95 | JPEG quality 100 | Kết luận |
|---|---:|---:|---|
| Pilot images | 64 | 64 | Bằng nhau |
| Whole-image MAE trung bình | 0.873271 | 0.085074 | q100 tốt hơn |
| Whole-image RMSE trung bình | 1.235387 | 0.291270 | q100 tốt hơn |
| Whole-image PSNR trung bình | 47.2414 dB | 58.8577 dB | q100 tốt hơn |
| Whole-image SSIM trung bình | 0.981217 | 0.998981 | q100 tốt hơn |
| Whole-image maximum absolute error lớn nhất | 12 | 2 | q100 tốt hơn |
| BBox-ROI được đánh giá | 402 | 402 | Bằng nhau |
| BBox-ROI MAE trung bình | 0.848022 | 0.087567 | q100 tốt hơn |
| BBox-ROI PSNR trung bình | 47.9413 dB | 58.7247 dB | q100 tốt hơn |
| BBox-ROI SSIM trung bình | 0.996632 | 0.999820 | q100 tốt hơn |
| ROI maximum absolute error lớn nhất | 10 | 2 | q100 tốt hơn |
| Kích thước trung bình mỗi ảnh | 1.619 MB | 3.162 MB | q95 nhỏ hơn |
| Tổng dung lượng 64 ảnh pilot | 98.82 MiB | 192.97 MiB | q95 nhỏ hơn |
| Compression ratio trung bình | 5.04:1 | 2.47:1 | q95 hiệu quả hơn |
| Projected storage cho 4,894 ảnh | 7.38 GiB | 14.41 GiB | q95 tiết kiệm khoảng 7.03 GiB |

Storage result:

```text
Quality 95 reduces projected storage by approximately 48.79%
relative to quality 100.

Quality 100 is approximately 1.95 times larger
than quality 95.
```

Small-bbox evidence at quality 95:

```text
20 smallest relative-area bbox cases

Mean ROI MAE: 0.4165
Mean ROI PSNR: 52.29 dB
Mean ROI SSIM: 0.995884
Largest ROI maximum absolute error: 5
```

Research decisions:

```text
Quality 100 is the numerical-fidelity winner.

Quality 95 is selected as the fidelity–storage/I/O trade-off.

Final JPEG quality is locked to 95.

Quality 95 must be used consistently for all 4,894 images
during Phase 2D.1B-Full.

No mixed JPEG quality is permitted across images or subsets.

Phase 2D.1B-Pilot is CLOSED / PASS.

Phase 2D.1B-Full is authorized and becomes OPEN / CURRENT.
```

Decision rationale:

```text
Quality 95 preserves high whole-image and bbox-ROI fidelity,
passes geometry and bbox invariance validation and reduces
projected storage by approximately 48.79% relative to quality 100.

Quality 100 has higher numerical fidelity but is approximately
1.95 times larger.

The pilot provides no evidence that the additional numerical fidelity
of quality 100 is required for downstream detector performance.
```

Important claim guardrails:

```text
Do not claim quality 95 has better detector performance than quality 100.

Do not claim clinical equivalence between JPG and source DICOM.

Do not claim preservation of every possible diagnostic feature.

Do not claim full DICOM-standard conformance.

Do not claim dataset training readiness.
```

Issues / risks:

```text
The full 4,894-image JPG dataset has not yet been created.

Full-scope decode error count is unknown.

Full-scope geometry and hash validation have not yet been completed.

coco_master_jpg.json has not yet been created.

MMDetection loading has not yet been validated.

filter_empty_gt=False has not yet been validated.

Retention of all 500 No Finding images inside MMDetection
has not yet been confirmed.

No downstream q95-versus-q100 detector ablation has been performed.

Dataset training readiness remains false.

Training authorization remains false.
```

Next phase:

```text
Phase 2D.1B-Full —
Full Controlled-Scope DICOM-to-JPG Conversion & Validation

Environment: Local
Status: OPEN / CURRENT
```

---

## 8. Phase 2D.1 — JPG Training Representation & MMDetection Empty-Image Loading Validation

Status: **IN PROGRESS**

Current subphase:

```text
Phase 2D.1B-Full —
Full Controlled-Scope DICOM-to-JPG Conversion & Validation

Environment:
Local

Status:
OPEN / CURRENT
```

Overall gate:

```text
Phase 2D.1A: CLOSED / PASS
Phase 2D.1B-Pilot: CLOSED / PASS
Phase 2D.1B-Full: OPEN / CURRENT
Phase 2D.1C: LOCKED
Phase 2D.1D: LOCKED

final_jpeg_quality: 95
full_conversion_authorized: true

jpg_training_representation_ready: false
coco_jpg_training_annotation_ready: false
mmdetection_dataset_loading_ready: false
empty_image_retention_ready: false
dataset_training_ready: false
training_authorized: false
```

### 8.1 Phase 2D.1A — Image Representation Protocol Decision

Status: **CLOSED / PASS**

Environment:

```text
Local
```

Protocol:

```text
configs/protocol/phase2D1_jpg_representation.yaml
```

Protocol version:

```text
1.0.0
```

Locked transformation order:

```text
DICOM decode
→ pixel-padding mask
→ modality transformation
→ VOI LUT/windowing
→ presentation-polarity normalization
→ deterministic uint8 conversion
→ JPEG encoding
```

Locked guardrails:

```text
No observed per-image min-max normalization.

No automatic percentile clipping.

No resize.

No crop.

No rotation.

No flip.

No transpose.

No automatic bbox scaling.
```

Pilot JPEG candidates:

```text
95
100
```

Phase 2D.1A result:

```text
Required policy coverage: 20/20
Cross-output drift: 0
Guardrail tests: 31/31 PASS
JSON parse: PASS
YAML strict-load: PASS
GPT review: PASS
Status: CLOSED / PASS
```

---

### 8.2 Phase 2D.1B — DICOM-to-JPG Conversion & Validation

Status: **IN PROGRESS**

Environment:

```text
Local
```

Internal gates:

```text
2D.1B-Pilot:
CLOSED / PASS

2D.1B-Full:
OPEN / CURRENT
```

#### 8.2.1 Phase 2D.1B-Pilot

Status: **CLOSED / PASS**

Implementation:

```text
scripts/02D1B_pilot_dicom_to_jpg.py
src/utils/dicom_jpg_protocol.py
tests/test_phase2D1B_pilot_guardrails.py
```

Implementation result:

```text
Implementation version: V6_FROZEN
Guardrail tests: 139/139 PASS
```

DICOM source:

```text
D:\ssl_detection_xray\data\raw\vinbigdata\dicom_subset\train
```

Configured root:

```text
D:\ssl_detection_xray\data\raw\vinbigdata\dicom_subset
```

Decoder backend evidence:

```text
pylibjpeg: unavailable
gdcm: unavailable
pillow: available
```

The initial explicit `pylibjpeg` run was blocked:

```text
jpeg2000_backend_unavailable:pylibjpeg
EXIT_CODE=1
```

This was expected guardrail behavior.

No silent decoder fallback occurred.

The successful pilot explicitly used:

```text
--jpeg2000-decoder pillow
```

Pilot result:

```text
Controlled DICOM paths resolved: 4,894/4,894
DICOM header inventory: 4,894/4,894

Pilot selected images: 64
No Finding pilot images: 16

Metadata/features expected: 54
Metadata/features covered: 54
Missing features: 0

Abnormal classes expected: 14
Abnormal classes covered: 14

Pixel decode attempts: 64
Pixel decode successes: 64
Pixel decode errors: 0
```

Geometry validation:

```text
Geometry records:
128 = 64 images × 2 JPEG candidates

Pre-JPEG shape unchanged: PASS
Reference PNG shape unchanged: PASS
Decoded JPG shape unchanged: PASS

Reference PNG mode L: PASS
JPEG mode L: PASS

Reference PNG dtype uint8: PASS
JPEG dtype uint8: PASS

Pixel matrix order unchanged: PASS

Unexpected EXIF orientation: 0

Resize applied: false
Crop applied: false
Rotation applied: false
Flip applied: false
Transpose applied: false

BBox scaling required: false
BBox modification performed: false
```

##### So sánh fidelity và dung lượng

Fidelity reference:

```text
pre-JPEG uint8 image
```

Compared representation:

```text
decoded JPEG image
```

BBox-ROI count:

```text
402 pilot annotations
```

| Tiêu chí | JPEG quality 95 | JPEG quality 100 | Kết luận |
|---|---:|---:|---|
| Pilot images | 64 | 64 | Bằng nhau |
| Whole-image MAE trung bình | 0.873271 | 0.085074 | q100 tốt hơn |
| Whole-image RMSE trung bình | 1.235387 | 0.291270 | q100 tốt hơn |
| Whole-image PSNR trung bình | 47.2414 dB | 58.8577 dB | q100 tốt hơn |
| Whole-image SSIM trung bình | 0.981217 | 0.998981 | q100 tốt hơn |
| Whole-image maximum absolute error lớn nhất | 12 | 2 | q100 tốt hơn |
| BBox-ROI MAE trung bình | 0.848022 | 0.087567 | q100 tốt hơn |
| BBox-ROI PSNR trung bình | 47.9413 dB | 58.7247 dB | q100 tốt hơn |
| BBox-ROI SSIM trung bình | 0.996632 | 0.999820 | q100 tốt hơn |
| ROI maximum absolute error lớn nhất | 10 | 2 | q100 tốt hơn |
| Kích thước trung bình mỗi ảnh | 1.619 MB | 3.162 MB | q95 nhỏ hơn |
| Tổng dung lượng 64 ảnh pilot | 98.82 MiB | 192.97 MiB | q95 nhỏ hơn |
| Compression ratio trung bình | 5.04:1 | 2.47:1 | q95 hiệu quả hơn |
| Projected storage cho 4,894 ảnh | 7.38 GiB | 14.41 GiB | q95 tiết kiệm khoảng 7.03 GiB |

Pairwise evidence:

```text
Quality 100 has better whole-image MAE, PSNR and SSIM
for 64/64 pilot images.

Quality 100 has better ROI MAE, ROI PSNR and ROI SSIM
for 402/402 pilot bbox regions.
```

Storage evidence:

```text
Quality 95 reduces projected storage by approximately 48.79%
relative to quality 100.

Quality 100 is approximately 1.95 times larger
than quality 95.
```

Small-bbox evidence at quality 95:

```text
20 smallest relative-area bbox cases

Mean ROI MAE: 0.4165
Mean ROI PSNR: 52.29 dB
Mean ROI SSIM: 0.995884
Largest ROI maximum absolute error: 5
```

Visual review:

```text
No global polarity inversion observed.

No unexpected crop observed.

No rotation observed.

No flip observed.

No transpose observed.

No geometry deformation observed.

No conversion-induced anatomical truncation observed.

Critical visual failure: false
```

Visual review is representation-pipeline evidence only.

It is not clinical validation.

##### Final JPEG quality decision

Selected quality:

```text
95
```

Decision status:

```text
approved_after_gpt_and_researcher_pilot_review
```

Full conversion authorization:

```text
true
```

Decision rationale:

```text
Quality 100 provides the highest numerical fidelity.

Quality 95 still preserves high whole-image and bbox-ROI fidelity,
passes geometry and bbox invariance validation and reduces projected
storage by approximately 48.79% relative to quality 100.

Quality 100 is approximately 1.95 times larger.

No pilot evidence demonstrates that the additional numerical fidelity
of quality 100 is required for downstream detector performance.

Quality 95 is therefore selected as the controlled
fidelity–storage/I/O trade-off.
```

Quality lock:

```text
All 4,894 images in Phase 2D.1B-Full must use JPEG quality 95.

Mixed JPEG qualities are forbidden.
```

Claim guardrails:

```text
Do not claim quality 95 has better detector performance than quality 100.

Do not claim clinical equivalence between JPG and source DICOM.

Do not claim preservation of every possible diagnostic feature.

Do not claim full DICOM-standard conformance.

Do not claim dataset training readiness.
```

Historical evidence rule:

```text
reports/phase2D1B_pilot_validation.json
was generated before final GPT/researcher review.

It may therefore retain:
phase_status = OPEN_REVIEW_REQUIRED
final_jpeg_quality = null
full_conversion_authorized = false

Do not edit this generated structural evidence retroactively.
```

Final decision ownership:

```text
reports/phase2D1B_pilot_decision_template.json
```

Expected final decision fields:

```text
decision_status:
approved_after_gpt_and_researcher_pilot_review

final_jpeg_quality:
95

selected_candidate:
95

full_conversion_authorized:
true
```

Pilot evidence:

```text
reports/phase2D1B_pilot_unit_tests_output_v6.txt
reports/phase2D1B_pilot_run_output.txt
reports/phase2D1B_pilot_run_output_pillow.txt
reports/phase2D1B_pilot_environment.json
reports/phase2D1B_pilot_header_inventory.csv
reports/phase2D1B_pilot_metadata_strata.csv
reports/phase2D1B_pilot_selection.csv
reports/phase2D1B_pilot_selection_coverage.csv
reports/phase2D1B_pilot_fidelity_metrics.csv
reports/phase2D1B_pilot_bbox_roi_metrics.csv
reports/phase2D1B_pilot_quality_summary.csv
reports/phase2D1B_pilot_quality_pairwise.csv
reports/phase2D1B_pilot_geometry_validation.csv
reports/phase2D1B_pilot_visual_audit_manifest.csv
reports/phase2D1B_pilot_validation.json
reports/phase2D1B_pilot_validation.md
reports/phase2D1B_pilot_decision_template.json
```

Pilot image evidence:

```text
data/processed/images_jpg_pilot/reference_uint8/
data/processed/images_jpg_pilot/q95/
data/processed/images_jpg_pilot/q100/
```

Visual evidence:

```text
plots/phase2D1B_pilot/full_image/
plots/phase2D1B_pilot/bbox_crops/
plots/phase2D1B_pilot/difference_heatmaps/
plots/phase2D1B_pilot/contact_sheets/
```

Pilot closure:

```text
Phase 2D.1B-Pilot:
CLOSED / PASS

Final JPEG quality:
95 / LOCKED

Full conversion:
AUTHORIZED
```

---

#### 8.2.2 Phase 2D.1B-Full

Status: **OPEN / CURRENT**

Environment:

```text
Local
```

Opening requirements:

```text
Phase 2D.1B-Pilot PASS: satisfied

Final JPEG quality selected: satisfied

Final JPEG quality = 95: locked

GPT/researcher review PASS: satisfied

Full conversion authorized: true
```

Full conversion policy:

```text
Protocol version:
1.0.0

JPEG quality:
95

Decoder backend:
must be explicitly selected

Silent decoder fallback:
forbidden

Resize:
false

Crop:
false

Rotation:
false

Flip:
false

Transpose:
false

BBox scaling:
false
```

Full-conversion targets:

```text
data/processed/images_jpg/train/<image_id>.jpg

data/processed/coco/coco_master_jpg.json

data/processed/image_mapping/dicom_to_jpg_mapping.csv
```

Full validation targets:

```text
JPG files: 4,894
Missing JPG: 0
Duplicate image IDs: 0
Decode errors: 0
JPG dimension mismatches: 0
Orientation changes: 0

COCO-JPG images: 4,894
COCO-JPG annotations: 36,096
COCO-JPG categories: 14

Abnormal images: 4,394
No Finding images: 500
No Finding annotations: 0

BBox mismatches versus coco_master: 0
Area mismatches: 0
Category mismatches: 0
Traceability mismatches: 0
Boundary violations: 0
Absolute COCO-JPG paths: 0
```

`coco_master_jpg.json` may change only:

```text
images[].file_name

from:
train/<image_id>.dicom

to:
train/<image_id>.jpg
```

The following must remain unchanged:

```text
image id
annotation id
category id
width
height
bbox
area
iscrowd
categories
canonical_ann_id
source_row_id
traceability fields
```

Full-conversion restrictions:

```text
Do not modify source DICOM files.

Do not modify canonical annotations.

Do not modify coco_master.json.

Do not resize, crop, rotate, flip or transpose.

Do not scale, clamp, delete or fuse bbox.

Do not create train/val/test split.

Do not create labeled/unlabeled split.

Do not train.

Do not run detector inference.

Do not generate pseudo-labels.

Do not tune thresholds.

Do not compute AP/mAP.

Do not use the test set.
```

Git rule:

```text
Do not commit 4,894 JPG files to ordinary Git.

Commit only scripts, protocols, reports, mappings,
small audit artifacts and documentation that belong in source control.
```

Full PASS rule:

```text
Full DICOM inventory: PASS

All 4,894 images converted using quality 95.

Decode errors: 0

Missing JPG: 0

Duplicate image IDs: 0

Width/height mismatches: 0

Orientation changes: 0

Full traceability mapping: PASS

coco_master_jpg.json created: PASS

COCO-JPG structural validation: PASS

BBox/category/area/traceability mismatches: 0

No Finding images retained: 500

No Finding annotations: 0

GPT review: PASS
```

---

### 8.3 Phase 2D.1C — MMDetection Dataset / Empty-Image Loading Validation

Status: **LOCKED until Phase 2D.1B-Full PASS**

Environment:

```text
Google Colab
```

Inputs:

```text
Full JPG training representation
coco_master_jpg.json
MMDetection dataset config
Validation scripts
```

Required checks:

```text
MMDetection import: PASS
MMEngine/MMCV/MMDetection versions recorded
COCO-JPG parse: PASS
Dataset build: PASS
dataset.full_init(): PASS

Dataset length: 4,894
Abnormal images retained: 4,394
No Finding images retained: 500

Empty-GT samples: exactly 500
Unexpected empty abnormal images: 0
Annotated No Finding images: 0

filter_empty_gt=False effective: PASS

Abnormal sample loading: PASS
No Finding loading with empty GT: PASS

Dataloader smoke test with num_workers=0: PASS
Dataloader smoke test with num_workers>0: PASS

All image IDs seen: 4,894
```

No Finding sample expectations:

```text
gt_instances.bboxes shape = (0, 4)
gt_instances.labels shape = (0,)
sample remains present in the dataset
```

Failure rule:

```text
If dataset length = 4,394, Phase 2D.1C FAILS because
500 No Finding images were filtered out.
```

Restrictions:

```text
Do not train.

Do not run model inference.

Do not create split.

Do not generate pseudo-labels.

Do not tune thresholds.

Do not compute AP/mAP.

Do not use the test set.
```

---

### 8.4 Phase 2D.1D — Evidence Consolidation, GPT Review & Closure

Status: **LOCKED until Phase 2D.1C PASS**

Environment:

```text
Local
```

Required actions:

```text
Download Colab evidence to the local repo.

Verify report files and hashes.

Request GPT final review.

Update PROJECT_CONTEXT.md after PASS.

Update PHASE_HANDOFF.md after PASS.

Update research_log.md after PASS.

Update CHECKLIST_TRIEN_KHAI_FULL.xlsx after PASS.

Stage reviewed evidence only.

Commit and push Phase 2D.1.
```

Do not commit:

```text
4,894 JPG files to ordinary Git.

Temporary conversion files.

Colab caches.

MMDetection checkpoints.

Training logs unrelated to loading validation.
```

---

### 8.5 Training-readiness rule

```text
Passing Phase 2D.1A confirms only that the image-representation
protocol has been locked.

Passing Phase 2D.1B-Pilot confirms only that:
- representative DICOM decoding succeeded;
- geometry and bbox invariance passed;
- JPEG quality 95 was selected;
- full conversion may proceed.

Passing Phase 2D.1B-Pilot does not confirm that the full JPG
training representation exists.

Passing Phase 2D.1B-Full confirms:
- the complete JPG representation exists;
- full mapping and geometry validation pass;
- coco_master_jpg.json is structurally ready.

Passing Phase 2D.1C confirms:
- MMDetection dataset loading readiness;
- retention of all 500 No Finding images;
- empty-GT samples are represented correctly.

Passing Phase 2D.1 overall does not create a fixed
train/val/test split.

Passing Phase 2D.1 overall does not authorize detector training.

dataset_training_ready remains false.

training_authorized remains false until later split and downstream
protocol phases pass their own Definition of Done.
```

---