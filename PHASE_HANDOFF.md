# PHASE HANDOFF — `ssl_detection_xray_v2`

Ngày cập nhật: 2026-07-14

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

---

## 3. Trạng thái hiện tại

```text
Current phase: Phase 2D — COCO Master Conversion & Validation
Previous phase: Phase 2C — Framework & Format Decision / COCO Conversion Planning

Phase 0 core: PASS
Phase 0 local training framework: DEFERRED
Phase 1A — Dataset Overview: PASS
Phase 1B — Annotation Quality: PASS
Phase 1C — Dataset Scope Decision: PASS
Phase 1D — Label Reliability & Kappa Feasibility: PASS
Phase 2A — Data Standardization / Image-Boundary Validation: PASS
Phase 2B — Canonical Detection Annotation Schema: PASS
Phase 2C — Framework & Format Decision / COCO Conversion Planning: PASS

Git status: Phase 2C completed; pending commit/push after documentation update
```

Được mở / tiếp theo:

```text
Phase 2D — COCO Master Conversion & Validation
```

Chưa được làm:

```text
Train/val/test split
Labeled/unlabeled split
Train supervised detector
Train SSL detector
Generate pseudo-label
Tune threshold
Use test set
Framework dataloader validation
Empty image loading check
Pixel array reading
Image copy/convert
```

Ghi chú trạng thái:

```text
Phase 2A đã xác nhận 4,894 DICOM files tồn tại và đọc được metadata/header.
Image dimensions available cho toàn bộ 4,894 images.
Toàn bộ 36,096 abnormal bbox hợp lệ trong image boundary.

Phase 2B đã tạo portable canonical detection schema.
Canonical image table có 4,894 images.
Canonical bbox table có 36,096 abnormal bbox rows.
Canonical class mapping có đúng 14 abnormal detection classes.
No Finding vẫn là negative image không có bbox, không phải detection class.
No Finding không nằm trong canonical bbox table hoặc detection class mapping.
Path policy đã portable: downstream dùng VINBIGDATA_DICOM_ROOT + relative_dicom_path.

Phase 2C đã chốt:
Primary framework: MMDetection.
Fallback framework: Detectron2_optional, chỉ dùng nếu được GPT review lại.
Primary annotation format: COCO_detection_JSON.
COCO conversion thật deferred sang Phase 2D.
No Finding phải nằm trong COCO images, không nằm trong annotations/categories.
BBox conversion Phase 2D: xyxy_original_image → coco_xywh_absolute.
Metric readiness đã ghi cho mAP@0.5:0.95, AP50, AP75 và class-wise AP.
Dataset vẫn chưa training-ready vì DICOM loader và empty image loading chưa được validate.

Không được tạo split, train, pseudo-label hoặc tune threshold khi chưa mở đúng phase.
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

## 8. Phase 2D — next-phase guardrail

Phase 2D chỉ được làm:

```text
COCO Master Conversion & Validation
```

Phase 2D được phép:

```text
Read canonical_image_table.csv.
Read canonical_bbox_table.csv.
Read canonical_class_mapping.csv.
Create data/processed/coco/coco_master.json.
Validate COCO image/annotation/category counts.
Validate bbox conversion xyxy_original_image → coco_xywh_absolute.
Validate No Finding images appear in images with zero annotations.
Validate No Finding is absent from annotations and categories.
Validate category_id contiguous 1..14.
Validate traceability fields.
Generate reports/phase2D_coco_validation_report.md.
Generate reports/phase2D_coco_validation_report.json.
```

Phase 2D chưa được làm:

```text
Train/val/test split.
Labeled/unlabeled split.
Training.
Inference.
Pseudo-labeling.
Threshold tuning.
Test-set usage.
Framework dataloader validation.
Empty image loading check.
Pixel array reading.
Image copy/convert.
```

Definition of Done Phase 2D dự kiến:

```text
coco_master.json created: true
image_count = 4,894
annotation_count = 36,096
category_count = 14
No Finding category count = 0
No Finding annotations count = 0
No Finding images retained = 500
All bbox converted to valid COCO xywh absolute
All areas valid
iscrowd = 0 for all annotations
category_id contiguous 1..14
All annotation image_id exist in images
All annotation category_id exist in categories
No split/train/pseudo-label/test actions performed
GPT review PASS
```

---

## 9. Nguyên tắc review bắt buộc

Khi đưa code/log/output, GPT phải kiểm tra:

1. Output đã đủ chưa?
2. Có đạt Definition of Done chưa?
3. Có lỗi logic nghiên cứu không?
4. Có rủi ro leakage không?
5. Có sai split/seed/metric không?
6. Có xử lý đúng No Finding không?
7. Có pseudo-bbox generation đúng bản chất SSOD không?
8. Có NMS và box quality filtering không?
9. Có rủi ro confirmation bias không?
10. Có rủi ro threshold làm rare class biến mất không?
11. Có test set bị dùng để tune không?
12. Có được tick checklist chưa?
13. Nếu chưa đạt, viết prompt sửa lỗi cho Claude.

---

## 10. Những lỗi nguy hiểm cần luôn nhắc

### 10.1 Data / COCO / No Finding

- No Finding bị đưa nhầm thành detection class.
- Ảnh No Finding bị framework lọc khỏi dataloader.
- Ảnh No Finding không nằm trong test set nên không đo được FP/negative.
- Bbox nhầm `xyxy` / `xywh`.
- Bbox bị lệch sau resize, flip, crop, scale.
- Split leakage.
- Unlabeled vô tình dùng ground truth.

### 10.2 Training / SSL

- Supervised và SSL không cùng labeled split.
- So sánh SSL vs supervised nhưng không cùng `split_seed`.
- Khóa nhầm `training_seed` giống nhau cho mọi run làm variance giả thấp.
- Chỉ chạy 1 seed rồi kết luận.
- Checkpoint chọn bằng test set.
- Threshold tune bằng test set.
- SSL gain nhỏ hơn std nhưng vẫn over-claim.
- Teacher bật pseudo-label quá sớm, gây confirmation bias.
- Không có burn-in.
- Không log λ của unlabeled loss.

### 10.3 SSOD-specific

- Pseudo-label chỉ ghi class/confidence mà quên bbox.
- Không dùng NMS trước khi lấy pseudo-bbox.
- Quá nhiều bbox trùng lặp làm student học nhiễu.
- Không transform bbox từ weak view sang strong view.
- Không kiểm tra box quality.
- Threshold cao làm rare class biến mất.
- Threshold thấp làm ảnh negative sinh nhiều pseudo-box sai.
- Không theo dõi pseudo-box trên unlabeled negative images.
- Không có fallback khi SSL không cải thiện.

### 10.4 Compute

- Teacher–Student OOM do batch quá lớn.
- Không có AMP / gradient accumulation / checkpoint resume.
- Chạy full training trước khi smoke test.
- Không ghi compute budget, GPU, VRAM, thời gian train.

---

## 11. Prompt mở chat mới

Dán đoạn này khi mở chat mới:

```text
Tôi đang tiếp tục đề tài đã khóa:
“Nghiên cứu học bán giám sát cho dò tìm bất thường trên X-quang phổi”.

Hãy đọc PROJECT_CONTEXT.md và PHASE_HANDOFF.md trước.

Trạng thái hiện tại:
- Phase 2C — Framework & Format Decision / COCO Conversion Planning: PASS.
- Current phase: Phase 2D — COCO Master Conversion & Validation.
- Không train, không split, không pseudo-label, không tune threshold, không dùng test set.

Làm việc theo quy trình:
script → output → DoD → GPT review → tôi tick checklist.

Không nhảy phase.
Không train khi data/split/COCO/No Finding/seed/checkpoint criterion chưa pass DoD.
Nếu cần code, hãy viết prompt rõ ràng để tôi giao cho Claude.
Nếu tôi đưa output/log, hãy review theo DoD và chỉ ra lỗi logic nếu có.
```
