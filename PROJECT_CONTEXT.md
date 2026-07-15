# PROJECT_CONTEXT — SSL Object Detection trên X-quang phổi

> File này là **bộ nhớ trung tâm của dự án**.  
> Khi mở chat mới với GPT/Claude, hãy upload file này trước để giữ đúng ngữ cảnh, quy trình và các quyết định đã khóa.

---

## 1. Đề tài đã khóa

**Tên đề tài:**  
“Nghiên cứu học bán giám sát cho dò tìm bất thường trên X-quang phổi”

**Trọng tâm:**  
Semi-supervised object detection trên ảnh X-quang phổi.

**Dataset chính:**  
VinBigData Chest X-ray.

**Task:**  
Object detection bằng bounding box, **không phải classification**, **không phải segmentation**.

**Bản chất kỹ thuật:**  
Đây là bài toán **Semi-Supervised Object Detection (SSOD)**. Vì vậy pseudo-label không chỉ là nhãn lớp, mà gồm:

- `class_id`
- `confidence score`
- `bbox` theo format `[x, y, width, height]`
- thông tin lọc bbox như NMS, box size, aspect ratio, boundary validity

---

## 2. Quy ước nghiên cứu đã khóa

### 2.1 Dữ liệu và No Finding

- No Finding / normal **không phải detection class**.
- No Finding là **ảnh âm tính không có bbox**.
- Ảnh No Finding phải nằm trong `images` của COCO JSON.
- Ảnh No Finding **không có dòng annotation** trong `annotations`.
- `No Finding` **không được nằm trong `categories`**.
- Framework detection không được tự lọc ảnh empty / ảnh không có bbox.
- Cần có kiểm tra riêng: `filter_empty_gt=False` hoặc cấu hình tương đương.

### 2.2 Metric và đánh giá

- Metric chính để kết luận: **mAP@0.5:0.95**.
- Metric phụ: AP50 (`mAP@0.5`), AP75, class-wise AP, recall/sensitivity, FP/image, FP per negative image.
- FP per negative image chỉ có ý nghĩa khi **test set có đủ ảnh No Finding**.
- Test set chỉ dùng để đánh giá cuối, **không dùng để chọn checkpoint, threshold, model tốt nhất, backbone hoặc hyperparameter**.
- Checkpoint được chọn bằng **validation mAP@0.5:0.95**.

### 2.3 Split, seed và reproducibility

- Labeled split: 1%, 5%, 10%, 20%.
- Nested sampling: **1% ⊂ 5% ⊂ 10% ⊂ 20%**.
- Supervised low-label và SSL phải dùng:
  - cùng labeled split;
  - cùng `split_seed`;
  - cùng fixed train/val/test;
  - cùng fixed test set.
- Stability dùng nhiều `training_seed`, thường 3–5 seeds.
- Phân biệt rõ:
  - `split_seed`: tạo train/val/test và labeled/unlabeled split;
  - `training_seed`: khởi tạo mô hình, dataloader shuffle, augmentation và training.
- Không khóa cùng một `training_seed` cho mọi run vì sẽ tạo variance giả thấp.
- Lưu đầy đủ:
  - seed number;
  - Python RNG state;
  - NumPy RNG state;
  - PyTorch CPU RNG state;
  - PyTorch CUDA RNG state;
  - deterministic flags;
  - config snapshot;
  - checkpoint;
  - log;
  - report.

### 2.4 Class imbalance

- Class imbalance được phân tích và báo cáo như một đặc trưng tự nhiên của VinBigData Chest X-ray.
- Không áp dụng kỹ thuật xử lý mất cân bằng độc lập như:
  - oversampling;
  - undersampling;
  - class weighting;
  - reweighting loss.
- Ảnh hưởng của class imbalance được phân tích thông qua:
  - class-wise AP;
  - pseudo-label count theo class;
  - retained pseudo-label ratio;
  - rare-class pseudo-label survival rate.
- Nếu có xử lý liên quan rare classes, nó chỉ được thực hiện trong phạm vi SSL / pseudo-label filtering, không được trình bày như một đóng góp xử lý imbalance độc lập.

---

## 3. Quy ước SSOD đã khóa

### 3.1 Teacher–Student SSL Detection

Pipeline chính:

```text
Supervised detector
→ Teacher model
→ pseudo-label unlabeled X-ray
→ confidence filtering
→ class-wise / dynamic thresholding
→ NMS
→ box quality filtering
→ transform pseudo-bbox từ weak view sang strong view
→ train Student trên labeled + pseudo-labeled data
→ EMA update Teacher
→ evaluate SSL gain/loss
```

### 3.2 Pseudo-BBox Generation

Trong SSOD, pseudo-label phải gồm bbox. Vì vậy Phase SSL phải kiểm tra:

- Teacher sinh prediction trên weakly augmented image.
- Pseudo-label gồm `class_id`, confidence và bbox `[x, y, w, h]`.
- Áp dụng confidence threshold trước hoặc cùng lúc với NMS.
- Áp dụng **NMS** để loại bbox trùng lặp.
- Áp dụng box quality filters:
  - box size;
  - aspect ratio;
  - boundary validity;
  - loại bbox quá nhỏ/quá lớn bất thường;
  - loại pseudo-box đáng ngờ trên ảnh No Finding nếu cần phân tích.
- Sau khi lọc, bbox phải được transform đúng sang strong view cho Student.
- Cần visualize một số mẫu để kiểm tra pseudo-box không bị lệch.

### 3.3 Anti-confirmation-bias safeguards

Confirmation bias có dạng:

```text
Teacher đoán sai
→ sinh pseudo-bbox sai
→ Student học sai
→ EMA cập nhật lại Teacher
→ lỗi được củng cố
```

Các cơ chế bắt buộc phải có hoặc phải được ghi rõ nếu không dùng:

- supervised burn-in / warm-up trước khi bật pseudo-label;
- EMA teacher;
- confidence threshold sweep;
- class-wise hoặc dynamic thresholding;
- NMS và box-quality filtering;
- theo dõi pseudo-box trên ảnh No Finding;
- theo dõi rare-class pseudo-label survival rate;
- không bật SSL quá sớm khi teacher chưa ổn định;
- log pseudo-label count theo class và theo threshold.

### 3.4 Class-wise / Dynamic Threshold

- Không mặc định dùng một threshold cố định cho cả 14 class mà không phân tích.
- Cần sweep threshold chung và/hoặc thử class-wise threshold.
- Threshold cho rare class chỉ được điều chỉnh trong phạm vi phân tích SSL/pseudo-label filtering.
- Không khóa cứng các giá trị như 0.9 hoặc 0.6 nếu chưa có validation evidence.
- Quyết định threshold cuối phải dựa trên validation / analysis split được định nghĩa trước, không dùng test set.

### 3.5 Positive/Negative mini-batch monitoring

- Không giả định ảnh normal chiếm đa số; phải kiểm tra tỷ lệ thật trong subset/split/batch.
- Cần log tỷ lệ positive/negative trong labeled batch.
- Cần log tỷ lệ positive/negative trong unlabeled batch nếu có metadata để phân tích.
- Cần theo dõi pseudo-box sinh ra trên unlabeled negative images.
- Nếu batch quá lệch, có thể dùng sampler kiểm soát nhẹ như một biện pháp ổn định training, nhưng không trình bày như đóng góp xử lý imbalance độc lập.

### 3.6 Compute / OOM fallback

Teacher–Student SSOD tốn GPU vì có labeled branch, unlabeled weak branch, unlabeled strong branch, teacher forward và student forward.

Fallback khi OOM:

- giảm batch size;
- dùng gradient accumulation;
- dùng mixed precision AMP;
- giảm image size có kiểm soát;
- giảm số unlabeled images per batch;
- chạy smoke test trước full training;
- chạy threshold sweep trên subset nhỏ trước;
- bật checkpoint/resume;
- chuyển sang Vast.ai nếu Colab/Kaggle không đủ.

---

## 4. Vai trò làm việc

- **Tôi** = người quyết định nghiên cứu.
- **GPT** = người thiết kế, phản biện và viết học thuật.
- **Claude** = người viết code trong repo.
- **Python** = công cụ chạy dữ liệu, train, evaluate và tạo bằng chứng.

Quy trình bắt buộc:

```text
script → output → DoD → GPT review → tôi tick checklist
```

Nguyên tắc:

- Không nhảy phase.
- Không train khi data/split/COCO/No Finding/seed/checkpoint criterion chưa pass DoD.
- Không kết luận khi chưa có output thật.
- Không để Claude tự đổi protocol nghiên cứu.

---

## 5. Các file điều phối chính

- `PROJECT_CONTEXT.md`: bộ nhớ trung tâm khi chuyển chat.
- `PHASE_HANDOFF.md`: trạng thái bàn giao phase hiện tại, dùng khi chuyển chat hoặc giao việc cho Claude.
- `STRUCTURE.md`: khung thesis/paper.
- `repository_structure.md`: cấu trúc repo.
- `RESEARCH_CHECKLIST.md`: checklist nghiên cứu tổng thể.
- `CHECKLIST_TRIEN_KHAI_FULL.md`: checklist triển khai theo phase.
- `CHECKLIST_TRIEN_KHAI_FULL.xlsx`: checklist trực quan có Dashboard, Checklist, Phase Summary, Lists.

Lưu ý thống nhất tên file:

- Dùng `STRUCTURE.md`, không dùng lẫn `RESEARCH_STRUCTURE.md` nếu repo đã chốt tên `STRUCTURE.md`.

---

## 6. Phase triển khai đã khóa

### PHASE 0 — Setup repo và môi trường

- Tạo repo.
- Tạo `.gitignore`.
- Tạo requirements/environment.
- Khóa seed, RNG state và deterministic flags.
- Tạo `README.md`, `CLAUDE.md`, `research_log.md`.

### PHASE 1 — Data & Medical Feasibility

- **1A:** Dataset overview.
- **1B:** Annotation quality.
- **1C:** Dataset scope decision.
- **1D:** Kappa feasibility / limitation-aware analysis.

### PHASE 2 — Data Standardization & Master Format

- **2A:** DICOM & bbox validation.
- **2B:** Canonical schema.
- **2C:** Framework & format decision.
- **2D:** COCO master conversion and validation.
- **2D.1:** JPG Training Representation & MMDetection Empty-Image Loading Validation.
  - **2D.1A:** Image Representation Protocol Decision.
  - **2D.1B:** DICOM-to-JPG Conversion & Validation.
  - **2D.1C:** MMDetection Dataset / Empty-Image Loading Validation.
  - **2D.1D:** Evidence Consolidation, GPT Review & Closure.
- **2E:** Fixed train/val/test split.
- **2F:** Labeled/unlabeled split for SSL.

### PHASE 3 — Pre-training Diagnostics

- **3A:** Dataset diagnostics before training.

### PHASE 4 — Supervised Baselines

- **4A.0:** Full-label supervised upper bound.
- **4A.1:** Low-label supervised baseline.
- **4B:** Attention / ViT-oriented supervised extension, optional.

### PHASE 5 — SSL Detection

- **5.1A:** Teacher–Student SSL pipeline.
- **5.1B:** Pseudo-BBox Generation.
- **5.1C:** Pseudo-label filtering with NMS and box quality filters.
- **5.1D:** Anti-confirmation-bias safeguards.
- **5.1E:** SSL main experiments.
- **5.1F:** Positive/negative mini-batch monitoring.
- **5.2 / 5.3:** Optional SSL extension, only if compute allows.

### PHASE 6 — Analysis

- Threshold sweep.
- Main evaluation.
- Error analysis.
- Negative image false positive analysis.
- Pseudo-label bias analysis.
- Seed stability analysis.
- Failure/fallback analysis.

### PHASE 7 — Contribution & Paper

- Final reports.
- Thesis/paper writing.
- Paper framing.
- Figures/tables.
- Limitations and claim guardrails.

---


## 7. Trạng thái hiện tại

Current phase: Phase 2D.1A — Image Representation Protocol Decision
Overall phase: Phase 2D.1 — JPG Training Representation & MMDetection Empty-Image Loading Validation: IN PROGRESS
Previous phase: Phase 2D — COCO Master Conversion & Validation: CLOSED / PASS
Next gated phase: Phase 2D.1B — DICOM-to-JPG Conversion & Validation
Git status: Phase 2D committed and pushed to origin/main at commit 1a3f7a7.
Dataset training-ready: false
Training authorized: false

### 7.1 Current gate

```text
Phase 0 core: PASS
Phase 0 committed/pushed: PASS
Phase 0 training framework: DEFERRED

Phase 1A — Dataset Overview: PASS
Phase 1B — Annotation Quality: PASS
Phase 1C — Dataset Scope Decision: PASS
Phase 1D — Label Reliability & Kappa Feasibility: PASS

Phase 2A — Data Standardization / Image-Boundary Validation: PASS
Phase 2B — Canonical Detection Annotation Schema: PASS
Phase 2C — Framework & Format Decision / COCO Conversion Planning: PASS
Phase 2D — COCO Master Conversion & Validation: CLOSED / PASS

Phase 2D.1 — JPG Training Representation & MMDetection
Empty-Image Loading Validation: IN PROGRESS

Phase 2D.1A — Image Representation Protocol Decision:
OPEN / CURRENT

Phase 2D.1B — DICOM-to-JPG Conversion & Validation:
LOCKED until Phase 2D.1A GPT review PASS

Phase 2D.1C — MMDetection Dataset / Empty-Image Loading Validation:
LOCKED until Phase 2D.1B GPT review PASS

Phase 2D.1D — Evidence Consolidation, GPT Review & Closure:
LOCKED until Phase 2D.1C PASS

Split train/val/test: LOCKED
Labeled/unlabeled split: LOCKED
Training: LOCKED
Inference: LOCKED
Pseudo-labeling: LOCKED
Threshold tuning: LOCKED
AP/mAP computation: LOCKED
Test-set usage: LOCKED

jpg_training_representation_ready: FALSE
coco_jpg_training_annotation_ready: FALSE
mmdetection_dataset_loading_ready: FALSE
empty_image_retention_ready: FALSE
dataset_training_ready: FALSE
training_authorized: FALSE
```


### 7.2 Controlled working scope locked by Phase 1C

```text
Controlled working scope: 4,894 images
- Abnormal images: 4,394 / 4,394 retained
- No Finding images: 500 / 10,606 selected
- Selection unit: image_id
- No Finding row-level sampling used: false
- Source metadata: full VinBigData train.csv with 15,000 images and 67,914 rows
```

The controlled scope was validated using three evidence sources:

```text
1. Full train.csv metadata
2. DICOM package manifests: dicom_package_manifest_part_001.csv to dicom_package_manifest_part_035.csv
3. DICOM filename inventory from dicom_subset/train/*.dicom
```

Phase 1C confirmed:

```text
selected_total_images = 4,894
selected_abnormal_images = 4,394
selected_no_finding_images = 500
lost_abnormal_image_count = 0
abnormal_retention_rate = 1.0
manifest_unique_images = 4,894
dicom_unique_image_ids = 4,894
unknown_manifest_image_id_count = 0
selected_mixed_images = 0
image_type_label_mismatch_count = 0
abnormal_detection_classes_excluding_no_finding = 14
```

### 7.3 What remains locked after Phase 1C

```text
Do not create train/val/test split yet.
Do not convert COCO yet.
Do not train yet.
Do not generate pseudo-labels yet.
Do not tune thresholds yet.
Do not use test set yet.
Do not claim image-boundary bbox validity yet.
Do not delete/fuse near-duplicate bbox candidates yet.
```

Boundary validation is deferred to Phase 2A because Phase 1C did not read DICOM headers, image pixels, or image dimensions.


---

### 7.4 Phase 1D locked evidence

Phase 1D — Label Reliability & Kappa Feasibility: **PASS**

```text
Status: PASS_agreement_computed_and_documented
Date: 2026-07-01
Script: scripts/01D_kappa_feasibility.py
Input: data/interim/vinbigdata_phase1C_scope_annotations.csv
```

Outputs generated:

```text
reports/phase1D_kappa_feasibility.md
reports/phase1D_kappa_feasibility.json
reports/phase1D_classwise_agreement_feasibility.csv
reports/phase1D_radiologist_per_image.csv
reports/phase1D_rare_class_kappa_instability.csv
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
classwise_feasibility_summary: 14 abnormal classes assessed; 14 with feasible Fleiss’ Kappa; mean kappa=0.4879
rare_class_instability_summary: 5/14 classes carry kappa_instability_risk (severe=2, moderate=3, low=9); risk is prevalence/rarity-driven, not measured instability
label_level_agreement_status: evaluable_fleiss_computed
bbox_level_consistency_status: evaluated_descriptive_only
```

Research decisions:

```text
Fleiss’ Kappa is computed at image-level class agreement.
Cohen’s Kappa is not used as the main agreement statistic because each image has 3 radiologist ratings.
Kappa/agreement is used only as data-quality evidence and limitation evidence.
Kappa is not a model metric.
Kappa is not used for split/model/threshold selection.
Kappa is not used for training or pseudo-labeling.
Kappa is not used to delete, fuse, or edit annotations.
BBox-level consistency is kept separate from label-level agreement and remains descriptive only.
```

Issues / risks:

```text
Negative class decisions are inferred from read-coverage according to VinBigData labelling convention.
Kappa can be affected by prevalence imbalance.
Some rare classes carry kappa_instability_risk.
BBox-level consistency is not a bbox fusion policy.
Near-duplicate bbox handling is still deferred to a later annotation standardization decision.
```

Forbidden actions confirmed:

```text
No train/val/test split created.
No COCO conversion created.
No training started.
No pseudo-label generated.
No threshold tuned.
No test set used.
No pixel read.
No DICOM/header read.
No image dimensions read.
No boundary validation performed.
No annotation deleted or edited.
No near-duplicate bbox deleted or fused.
No Kappa used as model metric.
No Kappa used for split/model/threshold.
```

Next phase:

```text
Phase 2A — Data Standardization / Image-Boundary Validation
```

### 7.5 Phase 2C locked evidence

Phase 2C — Framework & Format Decision / COCO Conversion Planning: **PASS**

```text
Status: PASS
Date: 2026-07-14
Script: scripts/02C_framework_format_decision.py
```

Outputs generated:

```text
reports/phase2C_framework_format_decision.md
reports/phase2C_framework_format_decision.json
configs/framework/main_framework.yaml
configs/dataset/coco_paths.yaml
configs/protocol/coco_conversion_policy.yaml
```

Key decisions:

```text
Primary framework: MMDetection
Fallback framework: Detectron2_optional, only after GPT re-review
Primary annotation format: COCO_detection_JSON
Source schema: canonical_detection_schema from Phase 2B
Actual COCO conversion: deferred to Phase 2D
No Finding: negative image with zero annotations, not a detection class
No Finding in planned COCO images: true
No Finding in planned COCO annotations: false
No Finding in planned COCO categories: false
BBox target format for Phase 2D: coco_xywh_absolute
COCO category_id policy: contiguous integer 1..14
Path policy: VINBIGDATA_DICOM_ROOT + relative_dicom_path
Dataset training-ready: false
```

Framework rationale:

```text
MMDetection is selected because it best matches the planned COCO-based teacher-student SSOD workflow:
- native object detection support;
- native COCO dataset and COCO-style mAP evaluation;
- official/semi-official SSOD-oriented components and configuration patterns;
- labeled/unlabeled pipeline compatibility;
- config-driven reproducibility;
- lower custom training/evaluation implementation burden than Detectron2/custom PyTorch for this project.

Detectron2 remains a fallback because it is a strong detection framework, but the teacher-student SSOD layer would require more project-specific custom implementation.
YOLO-based frameworks are rejected as primary because their annotation/evaluation stack diverges from the locked COCO/MMDetection protocol and represents negatives differently.
Custom PyTorch/torchvision is rejected because dataset/evaluator/trainer/pseudo-label loop/EMA teacher/COCO metric integration would create excessive engineering and silent-bug risk.
```

Format rationale:

```text
COCO detection JSON is selected because it best preserves:
- image-level negatives separate from detection categories;
- COCO mAP@0.5:0.95 / pycocotools evaluation compatibility;
- AP50, AP75 and class-wise AP readiness;
- traceability to canonical/source rows;
- pseudo-label output compatibility for later SSOD phases.
```

Metric readiness:

```text
Phase 2C does not compute AP metrics.
Primary metric remains mAP@0.5:0.95.
Secondary diagnostics planned: AP50, AP75, class-wise AP, recall/sensitivity, FP/image, FP per negative image.
Metrics are computable only after COCO conversion, fixed split creation, model training, and prediction generation.
Test-set metric is forbidden for checkpoint selection, threshold tuning, model selection, and augmentation decisions.
```

Issues / risks:

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

### 7.6 Phase 2D locked evidence

Phase 2D — COCO Master Conversion & Validation: **PASS**

```text
Status: PASS
Date: 2026-07-14
Script: scripts/02D_build_coco_master.py
Protocol: configs/protocol/phase2D_coco_master_validation.yaml
Unit tests: tests/test_phase2D_guardrails.py
```

Primary output:

```text
data/processed/coco/coco_master.json
```

Validation evidence:

```text
reports/phase2D_coco_master_validation.json
reports/phase2D_coco_master_validation.md
reports/phase2D_coco_image_annotation_counts.csv
reports/phase2D_coco_category_summary.csv
reports/phase2D_coco_invalid_annotations.csv
reports/phase2D_coco_no_finding_audit.csv
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
Broken image/category references: 0
Absolute paths: 0

Image IDs: unique and contiguous
Annotation IDs: unique and contiguous
Category IDs: contiguous 1..14
Category ID 0: absent
No Finding category: absent
Background category: absent

Traceability mismatches: 0
Missing canonical annotations: 0
Duplicated canonical annotations: 0
Extra COCO annotations: 0
One-to-one preservation: PASS

JSON parse: PASS
pycocotools load: PASS
Protocol strict load: PASS
Protocol / Phase 2B drift count: 0
Pre-promotion validation: PASS
Atomic output promotion: PASS
Guardrail unit tests: 22/22 PASS
Warnings: 0
Hard errors: 0
dod_pass_candidate: true
```

Research decisions:

```text
The Phase 2B canonical schema has been converted to the official COCO master.

All 4,894 controlled-scope images are retained in COCO images.

All 36,096 abnormal bbox rows are preserved one-to-one in COCO annotations.

COCO categories contain exactly 14 abnormal detection classes.

No Finding remains an image-level negative:
- retained in COCO images;
- zero annotations;
- excluded from categories.

No background category is created.

BBox format is converted from xyxy_original_image to coco_xywh_absolute:
[x, y, width, height].

Area is calculated as width * height and iscrowd is fixed to 0.

No bbox was clamped, deleted, fused, rounded, or processed using NMS.

COCO file_name uses relative_dicom_path and does not contain an absolute local path.

The protocol YAML is strict-loaded and cross-checked against Phase 2B validation and canonical tables.

The final COCO output is atomically replaced only after all hard validations pass.
```

Issues / risks:

```text
A valid COCO annotation file does not make the dataset training-ready.

The processed training image representation has not yet been created.

The DICOM-to-JPG intensity transformation protocol has not yet been locked.

RescaleSlope/RescaleIntercept, VOI LUT/windowing, MONOCHROME1 inversion,
intensity clipping, uint8 conversion, JPEG quality and channel policy
have not yet been validated for the training representation.

coco_master_jpg.json has not yet been created.

JPG dimensions and bbox invariance have not yet been validated.

MMDetection has not yet loaded the JPG + COCO-JPG dataset.

No Finding / empty-GT retention has not yet been validated using MMDetection.

filter_empty_gt=False or an equivalent configuration has not yet been
proven effective.

Train/val/test split has not been created.

Labeled/unlabeled SSL subsets have not been created.

Training, inference, pseudo-labeling, threshold tuning, AP/mAP computation
and test-set usage remain locked.
```


Forbidden actions confirmed:

```text
No DICOM file read.
No DICOM existence check.
No DICOM header read.
No pixel_array read.
No pydicom, cv2, or PIL image loading.
No image copying or conversion.
No train/val/test split.
No labeled/unlabeled split.
No MMDetection/Detectron2 dataset loading.
No filter_empty_gt validation.
No training.
No inference.
No pseudo-label generation.
No threshold tuning.
No test-set usage.
No AP/mAP computation.
No canonical annotation modification.
No bbox clamp/delete/fusion/NMS.
No dataset training-ready claim.
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
Opening condition: SATISFIED

Phase 2D documents updated: true
Phase 2D evidence committed: true
Phase 2D pushed to origin/main: true
Phase 2D commit: 1a3f7a7
Phase 2D.1A may begin: true
```
### 7.7 Phase 2D.1 current planning

Phase 2D.1 — JPG Training Representation & MMDetection Empty-Image Loading Validation

Status: **IN PROGRESS**

Current subphase:

```text
Phase 2D.1A — Image Representation Protocol Decision
Environment: Local
Status: OPEN / CURRENT
```

Subphase structure:

```text
Phase 2D.1A — Image Representation Protocol Decision
Environment: Local
Status: OPEN / CURRENT

Phase 2D.1B — DICOM-to-JPG Conversion & Validation
Environment: Local
Status: LOCKED until Phase 2D.1A GPT review PASS

Phase 2D.1B-Pilot — Representative DICOM-to-JPG pilot
Status: LOCKED

Phase 2D.1B-Full — Full controlled-scope conversion
Status: LOCKED until pilot PASS

Phase 2D.1C — MMDetection Dataset / Empty-Image Loading Validation
Environment: Google Colab
Status: LOCKED until Phase 2D.1B PASS

Phase 2D.1D — Evidence Consolidation, GPT Review & Closure
Environment: Local
Status: LOCKED until Phase 2D.1C PASS
```

Representation roles:

```text
DICOM:
Immutable raw medical source and source evidence.

JPG:
Processed training image representation generated by a fixed,
versioned and reproducible conversion protocol.

coco_master.json:
Annotation master linked to the original DICOM representation.

coco_master_jpg.json:
Training derivative linked to JPG file_name values.

MMDetection:
Downstream framework for dataset loading, detector training,
evaluation and later SSOD experiments.
```

Phase 2D.1A must lock:

```text
DICOM pixel decoding policy.
RescaleSlope and RescaleIntercept policy.
Modality LUT policy.
VOI LUT or windowing policy.
MONOCHROME1 inversion policy.
Intensity clipping policy.
uint8 [0,255] conversion policy.
JPEG quality candidate protocol.
Final JPEG quality selection rule.
Output channel policy.
No-resize policy.
No-crop policy.
No-rotation policy.
BBox scaling requirement.
JPG filename convention.
COCO-JPG path convention.
Traceability policy.
Pilot selection policy.
Fidelity validation protocol.
```

Current preliminary direction:

```text
Do not resize during DICOM-to-JPG conversion.
Do not crop.
Do not rotate.
Preserve original width and height.
Do not scale bbox when dimensions and orientation are proven unchanged.
Evaluate JPEG quality 95 and 100 during the pilot.
Lock one final JPEG quality value before full conversion.
```

Required phase outputs:

```text
Phase 2D.1A:
configs/protocol/phase2D1_jpg_representation.yaml
reports/phase2D1_image_representation_decision.md
reports/phase2D1_image_representation_decision.json

Phase 2D.1B:
data/processed/images_jpg/train/<image_id>.jpg
data/processed/coco/coco_master_jpg.json
data/processed/image_mapping/dicom_to_jpg_mapping.csv
reports/phase2D1_jpg_conversion_report.md
reports/phase2D1_jpg_conversion_validation.json
reports/phase2D1_jpg_image_metadata.csv
reports/phase2D1_jpg_fidelity_audit.csv
reports/phase2D1_jpg_bbox_validation.csv
reports/phase2D1_jpg_conversion_errors.csv
reports/phase2D1_no_finding_jpg_audit.csv

Phase 2D.1C:
configs/dataset/mmdet_coco_jpg_loading.py
scripts/02D1C_mmdet_loading_validation.py
reports/phase2D1_mmdet_loading_check.md
reports/phase2D1_mmdet_loading_check.json
reports/phase2D1_mmdet_environment_versions.txt
reports/phase2D1_mmdet_pip_freeze.txt
reports/phase2D1_empty_image_retention_audit.csv
reports/phase2D1_sample_loading_audit.csv
reports/phase2D1_dataloader_iteration_audit.csv
```

Current restrictions:

```text
Do not run full DICOM-to-JPG conversion before Phase 2D.1A PASS.

Do not run Phase 2D.1B-Full before the pilot and final JPG quality
decision pass review.

Do not open MMDetection validation before JPG conversion and COCO-JPG
validation pass.

Do not create train/val/test split.

Do not create labeled/unlabeled split.

Do not train a detector.

Do not run detector inference.

Do not generate pseudo-labels.

Do not tune confidence thresholds.

Do not compute AP/mAP.

Do not use the test set.

Do not modify canonical bbox or COCO master annotations.

Do not claim the dataset is training-ready.
```

Training-readiness rule:

```text
Passing Phase 2D.1 confirms JPG representation and MMDetection loading
readiness only.

It does not authorize training.

dataset_training_ready remains false until the required split and
downstream dataset protocol phases pass their own Definition of Done.
```

Next action:

```text
Design and review Phase 2D.1A — Image Representation Protocol Decision.
```


## 8. Nguyên tắc review bắt buộc

Khi tôi đưa code/log/output, GPT phải kiểm tra:

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

## 9. Những lỗi nguy hiểm cần luôn nhắc

### 9.1 Data / COCO / No Finding

- No Finding bị đưa nhầm thành detection class.
- Ảnh No Finding bị framework lọc khỏi dataloader.
- Ảnh No Finding không nằm trong test set nên không đo được FP/negative.
- Bbox nhầm `xyxy` / `xywh`.
- Bbox bị lệch sau resize, flip, crop, scale.
- Split leakage.
- Unlabeled vô tình dùng ground truth.
- DICOM-to-JPG conversion dùng min-max normalization không được khóa.
- Bỏ qua RescaleSlope hoặc RescaleIntercept.
- Bỏ qua modality LUT khi DICOM yêu cầu.
- VOI LUT/windowing policy không được ghi nhận.
- MONOCHROME1 không được đảo đúng.
- Chuyển trực tiếp pixel_array 12/16-bit sang uint8 gây mất thông tin không kiểm soát.
- JPEG quality thay đổi giữa các lần conversion.
- Full conversion được chạy trước khi pilot PASS.
- JPG bị resize, crop hoặc rotate nhưng bbox không được biến đổi tương ứng.
- JPG width/height không khớp COCO metadata.
- So sánh fidelity sai giữa raw DICOM và JPG mà không tách ảnh hưởng của windowing/uint8 quantization.
- coco_master_jpg.json làm thay đổi bbox, area, category hoặc annotation ID.
- COCO-JPG file_name không resolve đúng dưới MMDetection data_root.
- JPG trên local và JPG upload lên Colab không cùng hash.
- 500 ảnh No Finding bị MMDetection lọc do filter_empty_gt.

### 9.2 Training / SSL

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

### 9.3 SSOD-specific

- Pseudo-label chỉ ghi class/confidence mà quên bbox.
- Không dùng NMS trước khi lấy pseudo-bbox.
- Quá nhiều bbox trùng lặp làm student học nhiễu.
- Không transform bbox từ weak view sang strong view.
- Không kiểm tra box quality.
- Threshold cao làm rare class biến mất.
- Threshold thấp làm ảnh negative sinh nhiều pseudo-box sai.
- Không theo dõi pseudo-box trên unlabeled negative images.
- Không có fallback khi SSL không cải thiện.

### 9.4 Compute

- Teacher–Student OOM do batch quá lớn.
- Không có AMP / gradient accumulation / checkpoint resume.
- Chạy full training trước khi smoke test.
- Không ghi compute budget, GPU, VRAM, thời gian train.

---

## 10. Quy tắc kết luận kết quả

- SSL gain/loss phải tính bằng metric chính `mAP@0.5:0.95`.
- SSL gain phải so với supervised baseline cùng labeled split và cùng `split_seed`.
- SSL gain phải báo cáo kèm mean ± std theo nhiều `training_seed`.
- Nếu SSL gain nhỏ hơn hoặc tương đương std giữa các seed, chỉ nói: “chưa đủ bằng chứng ổn định”.
- Nếu SSL tăng mAP nhưng FP per negative image tăng mạnh, phải thảo luận trade-off y khoa.
- Nếu threshold làm rare-class survival rate giảm mạnh, không được chỉ báo cáo mAP tổng.
- Nếu SSL không cải thiện, phải phân tích failure mode trước khi thay đổi method.

---

## 11. Quy tắc cập nhật PROJECT_CONTEXT.md sau mỗi phase

Sau mỗi phase pass DoD, cập nhật ngắn gọn:

```md
### Phase X — Tên phase
Status: DONE / IN PROGRESS / BLOCKED
Date:
Scripts run:
Outputs generated:
DoD result:
Key findings:
Research decisions:
Issues / risks:
Next phase:
```

Không nhét toàn bộ log vào `PROJECT_CONTEXT.md`.  
Log chi tiết để trong `reports/`, `logs/`, `research_log.md` hoặc checklist Excel.

---

## 12. Prompt mở chat mới

Dán đoạn này khi mở chat mới:

```text
Tôi đang tiếp tục đề tài đã khóa:
“Nghiên cứu học bán giám sát cho dò tìm bất thường trên X-quang phổi”.

Hãy đọc PROJECT_CONTEXT.md trước.
Làm việc theo quy trình:
script → output → DoD → GPT review → tôi tick checklist.

Không nhảy phase.
Không train khi data/split/COCO/No Finding/seed/checkpoint criterion chưa pass DoD.
Nếu cần code, hãy viết prompt rõ ràng để tôi giao cho Claude.
Nếu tôi đưa output/log, hãy review theo DoD và chỉ ra lỗi logic nếu có.
```

---

## 13. Phase Progress Log

### Phase 0 — Setup repo và môi trường

Status: **DONE / CORE PASS**

Date: 2026-06-18

Scripts run:

```cmd
python scripts/00_check_environment.py --seed 2026 --output reports/phase0_environment_check.json --seed-manifest data/manifests/seed_state_manifest.json --freeze-output reports/phase0_pip_freeze.txt
python -m pytest tests	est_phase0.py -q
pip check
```

Outputs generated:

- `reports/phase0_environment_check.json`
- `reports/phase0_pip_freeze.txt`
- `reports/reproducibility_settings.md`
- `data/manifests/seed_state_manifest.json`
- `configs/protocol/checkpoint_policy.yaml`
- `research_log.md`
- `PHASE_HANDOFF.md`

DoD result:

- Repo structure: **PASS**
- Core environment: **PASS**
- Reproducibility evidence: **PASS**
- Checkpoint/evaluation policy: **PASS**
- Tests: **PASS**
- Local MMDetection/GPU training-ready: **DEFERRED**

Key findings:

- `core_import_ok = true`
- `framework_import_ok = false`
- `cuda_available = false`
- `pip check = No broken requirements found`
- `pytest = 5 passed`

Research decisions:

- Local environment is for validation/reporting only.
- Training-related setup will be done on remote/GPU.
- Test set policy is locked: final evaluation only.
- Primary metric is locked: `mAP@0.5:0.95`.

Issues / risks:

- Local environment cannot be used for detector training.
- MMDetection stack is not installed locally.
- Must not start split/COCO/training before their phase DoD.

Git:

```text
commit: b5127fd phase0: setup reproducible core environment
branch: main
remote: origin/main
```

---

### Phase 1A — Dataset Overview

Date: 2026-06-19
Status: **PASS**

Completed focus:

- tạo `scripts/01A_dataset_overview.py`;
- đọc annotation CSV;
- thống kê dataset overview;
- kiểm tra No Finding policy;
- kiểm tra bbox validity sơ bộ;
- tạo evidence report.

Expected script:

```text
scripts/01A_dataset_overview.py
```

Expected outputs:

- `reports/phase1A_dataset_overview.json`
- `reports/phase1A_dataset_overview.md`
- `reports/phase1A_class_distribution.csv`
- `reports/phase1A_image_level_summary.csv`
- `reports/phase1A_bbox_quality_summary.csv`

DoD:

- Có total rows.
- Có unique images.
- Có No Finding images.
- Có abnormal images.
- Có abnormal classes excluding No Finding.
- Có bbox missing/invalid summary.
- Có warning nếu No Finding có bbox.
- Có warning nếu abnormal class thiếu bbox.
- Không split.
- Không COCO.
- Không train.
- GPT review pass.

Next phase after pass:

- Phase 1B — Annotation Quality (kiểm tra chất lượng Annotation).

### Phase 1B — Annotation Quality (kiểm tra chất lượng Annotation).

Status: PASS

Date: 2026-06-19

Scripts run:

```cmd
python scripts\01B_annotation_quality.py --train-csv data\raw\vinbigdata\annotations\train.csv
```
Outputs generated:

reports/phase1B_annotation_quality.json
reports/phase1B_annotation_quality.md
reports/annotation_sanity_report.md
reports/invalid_bbox_rows.csv
reports/duplicate_bbox_candidates.csv
reports/phase1B_class_mapping.csv
reports/phase1B_bbox_quality_by_class.csv
reports/phase1B_image_label_consistency.csv

DoD result:

Annotation-level bbox sanity: PASS
No Finding policy: PASS
Abnormal bbox completeness: PASS
Class mapping consistency: PASS
Duplicate/near-duplicate candidates reported: PASS
Boundary check: DEFERRED to image-level validation because train.csv has no image dimensions
Forbidden actions avoided: PASS

Key findings:

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

Research decisions:

Do not delete or modify near-duplicate bbox candidates in Phase 1B.
Treat near-duplicate boxes as multi-radiologist annotation candidates requiring later fusion/handling decision.
Defer image-boundary validation to Phase 2A because Phase 1B is CSV-only.

Issues / risks:

Boundary validity cannot be concluded from CSV alone.
No Finding rows are repeated reader-level rows; future processing must operate at image level for negative images.


Lệnh commit:

```cmd
git status
git add scripts/01B_annotation_quality.py reports/phase1B_annotation_quality.json reports/phase1B_annotation_quality.md reports/annotation_sanity_report.md reports/invalid_bbox_rows.csv reports/duplicate_bbox_candidates.csv reports/phase1B_class_mapping.csv reports/phase1B_bbox_quality_by_class.csv reports/phase1B_image_label_consistency.csv PROJECT_CONTEXT.md research_log.md CHECKLIST_TRIEN_KHAI_FULL.xlsx
git commit -m "phase1B: validate annotation quality"
git push
```

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
Source total rows: 67,914
Source unique images: 15,000
Source abnormal images: 4,394
Source No Finding images: 10,606

Manifest rows: 4,894
Manifest unique image_id: 4,894
DICOM files listed: 4,894
DICOM unique image_id: 4,894

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
Image type / train.csv label mismatch count: 0
Unknown manifest image_id count: 0
Chunk summary match: true
```

Research decisions:

```text
Controlled working scope is officially locked to 4,894 image-level samples:
4,394 abnormal images + 500 No Finding images.

The 500 No Finding samples are selected and verified at image_id level, not row level.
The controlled scope is based on the already-downloaded DICOM package manifests and validated against train.csv and DICOM filename inventory.
No Finding remains a negative image label, not a detection class.
The metadata-only subset annotation CSV is created for selected image_id values only.
```

Issues / risks:

```text
The controlled scope uses 500 out of 10,606 No Finding images, not the full No Finding pool.
This is a deliberate controlled-scope design decision and must be stated as a limitation.
Boundary validity is not concluded in Phase 1C because image dimensions were not read.
147 near-duplicate bbox candidates from Phase 1B are retained, not deleted or fused.
Fusion/handling of multi-radiologist boxes is deferred to a later phase.
```

Forbidden actions confirmed:

```text
No train/val/test split created.
No COCO conversion created.
No training started.
No pseudo-label generated.
No threshold tuned.
No test set used.
No image copied.
No DICOM header read.
No pixel read.
No image dimension read.
No original annotation deleted or edited.
No near-duplicate bbox candidate deleted.
```

Next phase:

```text
Phase 1D — Kappa feasibility / limitation-aware analysis
```
### Phase 1D — Label Reliability & Kappa Feasibility

Status: **PASS**

Date: 2026-07-01

Scripts run:

```cmd
python scripts/01D_kappa_feasibility.py
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
Label reliability / Kappa feasibility: PASS
rad_id availability: PASS
radiologists per image: PASS
binary matrix feasibility: PASS
Cohen’s Kappa feasibility: PASS
Fleiss’ Kappa feasibility: PASS
class-wise image-level agreement: PASS
rare-class kappa instability risk: PASS
label-level vs bbox-level separation: PASS
forbidden actions avoided: PASS
```

Key findings:
```text
rad_id_available = true
rad_id_missing_count = 0
radiologists_total = 17
radiologists_per_image_distribution = {'3': 4894}
uniform_rater_count_per_image = true
same_rater_identity_panel_across_images = false
binary_matrix_feasible = true
cohen_kappa_feasible = false
fleiss_kappa_feasible = true
overall_fleiss_kappa_mean = 0.4879
rare_class_instability_summary = 5/14 classes carry kappa_instability_risk
```

Research decisions:
```text
Fleiss’ Kappa is used as image-level class agreement evidence.
Cohen’s Kappa is not used as the main agreement statistic because the per-image panel size is 3.
Kappa is data-quality evidence only, not a model metric.
Kappa is not used for split, threshold tuning, training, pseudo-labeling, or annotation editing.
BBox-level consistency remains descriptive only.
```

Issues / risks:
```text
Negative class decisions are reconstructed from read-coverage under VinBigData labelling convention.
Kappa may be affected by prevalence imbalance.
Rare classes need careful interpretation.
BBox near-duplicate handling remains deferred.
```

Next phase:
```text
Phase 2A — Data Standardization / Image-Boundary Validation
```

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
Image dimensions extraction: PASS
BBox boundary validation: PASS
No Finding bbox policy: PASS
Forbidden actions avoided: PASS
```

Key findings:

```text
DICOM files indexed under root: 4,894
Selected scope expected images: 4,894
Availability checked images: 4,894
DICOM available/missing: 4,894 / 0
DICOM read success/error: 4,894 / 0
Image dimensions available/missing: 4,894 / 0
Abnormal bbox rows checked: 36,096
BBox boundary valid/invalid: 36,096 / 0
No Finding rows with bbox: 0
Abnormal rows missing bbox: 0
Warnings: none
dod_pass_candidate: true
```

Research decisions:

```text
All controlled-scope DICOM files are available and readable at metadata/header level.
All abnormal bbox coordinates are valid within original image boundaries under xyxy convention.
No Finding remains a negative image label with no bbox and is not a detection class.
No bbox was edited, clamped, deleted or fused.
No image was copied, converted or normalized.
```

Issues / risks:

```text
Pixel array decoding was not checked in the main Phase 2A run.
Canonical schema, COCO conversion, split, dataloader loading and training remain locked for later phases.
Dataset is not yet training-ready until schema/COCO/split/loading phases pass DoD.
```

Next phase:

```text
Phase 2B — Canonical Schema
```


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
canonical_image_rows = 4894
canonical_image_unique_images = 4894
canonical_bbox_rows = 36096
canonical_class_count = 14
abnormal_images = 4394
no_finding_images = 500
no_finding_policy_pass = true
no_finding_in_detection_classes = false
bbox_without_image_count = 0
image_without_metadata_count = 0
bbox_missing_dimension_count = 0
bbox_invalid_count = 0
class_mapping_issue_count = 0
schema_error_count = 0
portable_path_policy_pass = true
relative_dicom_path_missing_count = 0
relative_dicom_path_absolute_count = 0
local_dicom_path_absolute_count = 4894
path_root_variable = VINBIGDATA_DICOM_ROOT
warnings = []
dod_pass_candidate = true
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
source_row_id traces to the Phase 1C controlled-scope annotation file, not necessarily the original full VinBigData train.csv row index.
Framework dataloader validation and empty-image loading checks have not been performed.
COCO conversion has not been performed.
Train/val/test split has not been created.
Near-duplicate bbox handling is still deferred.
```

Next phase:

```text
Phase 2C — Framework & Format Decision / COCO Conversion Planning
```

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
Format rationale: PASS
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
primary_framework = MMDetection
fallback_framework = Detectron2_optional
primary_annotation_format = COCO_detection_JSON
actual_coco_conversion_done = false
canonical_image_rows = 4894
canonical_bbox_rows = 36096
canonical_class_count = 14
abnormal_images = 4394
no_finding_images = 500
dataset_training_ready = false
dod_pass_candidate = true
```

Framework comparison decision:

```text
MMDetection: CHOSEN
Detectron2: FALLBACK_ONLY
Ultralytics YOLO / YOLO-based framework: REJECTED as primary framework
Custom PyTorch / torchvision: REJECTED
```

Framework rationale:

```text
MMDetection is selected as the primary framework because it most directly matches the thesis pipeline:
COCO-based detection, COCO mAP evaluation, teacher-student semi-supervised object detection, labeled/unlabeled data handling, and config-driven reproducibility.

Detectron2 remains a fallback because it is a strong PyTorch detection framework with COCO support, but the teacher-student SSOD pipeline would require more project-specific custom implementation.

YOLO-based frameworks are rejected as primary because their annotation/evaluation pipeline is YOLO-native and less aligned with the locked COCO master + MMDetection SSOD protocol.

Custom PyTorch / torchvision is rejected because the project would need custom dataset, dataloader, evaluator, trainer, pseudo-label loop, EMA teacher, COCO metric integration, logging, and config protocol. This would increase implementation risk and reproducibility risk.
```

Format comparison decision:

```text
COCO detection JSON: CHOSEN
YOLO txt: REJECTED
Pascal VOC XML: REJECTED
JSONL/custom: REJECTED
```

Format rationale:

```text
COCO detection JSON is selected as the downstream annotation format because it supports:
- explicit images / annotations / categories;
- images with zero annotations for No Finding negatives;
- standard COCO mAP@0.5:0.95 / pycocotools evaluation;
- AP50, AP75 and class-wise AP readiness;
- traceability fields such as canonical_ann_id and source_row_id;
- pseudo-label output compatibility in later SSOD phases.
```

Research decisions:

```text
COCO detection JSON is selected as the downstream annotation format.
MMDetection is selected as the primary framework.
Detectron2 is retained only as optional fallback after GPT re-review.
COCO conversion is deferred to Phase 2D.
No Finding remains a negative image with zero annotations and is not a detection class.
No background class is created.
BBox conversion for Phase 2D is xyxy_original_image -> coco_xywh_absolute.
COCO category_id should be contiguous integer 1..14.
Path resolution should use VINBIGDATA_DICOM_ROOT + relative_dicom_path.
Metric readiness is documented for mAP@0.5:0.95, AP50, AP75 and class-wise AP.
```

Issues / risks:

```text
MMDetection stack is not importable locally, which is expected because local training framework is deferred.
Remote/GPU environment is still required for detector training.
DICOM loader is not validated.
Empty image loading is not validated.
If filter_empty_gt is configured incorrectly, 500 No Finding images may be silently dropped.
COCO annotation format alone does not make the dataset training-ready.
AP50, AP75 and class-wise AP are not computed in Phase 2C.
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


### Phase 2D — COCO Master Conversion & Validation

Status: **PASS**

Date: 2026-07-14

Scripts run:

```cmd
python scripts\02D_build_coco_master.py
python -m unittest discover -s tests -p "test_phase2D_guardrails.py" -v
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
COCO conversion: PASS
COCO structure and relationships: PASS
BBox xyxy-to-xywh conversion: PASS
Area and boundary validation: PASS
No Finding policy: PASS
Category policy: PASS
Traceability and one-to-one preservation: PASS
JSON parse: PASS
pycocotools load: PASS
Strict YAML protocol: PASS
Protocol drift protection: PASS
Atomic output promotion: PASS
Guardrail tests 22/22: PASS
Forbidden actions avoided: PASS
GPT review: PASS
```

Key findings:

```text
images = 4894
annotations = 36096
categories = 14
abnormal_images = 4394
no_finding_images = 500
no_finding_annotations = 0
invalid_annotations = 0
absolute_paths = 0
hard_errors = 0
warnings = 0
dataset_training_ready = false
```

Research decisions:

```text
coco_master.json is accepted as the official controlled-scope COCO master.
No Finding remains an image-level negative with zero annotations.
No Finding and background are excluded from categories.
All canonical bbox rows are preserved one-to-one.
The final COCO output is promoted only after all hard validation passes.
```

Issues / risks:

```text
DICOM loader and pixel decoding are not validated.
Framework empty-image loading is not validated.
filter_empty_gt=False is not validated.
Split, training, inference, and pseudo-labeling remain locked.
```

Next phase:

```text
Phase 2D.1A — Image Representation Protocol Decision
```

### Phase 2D.1 — JPG Training Representation & MMDetection Empty-Image Loading Validation

Status: **IN PROGRESS**

Date opened: 2026-07-15

Current subphase:

```text
Phase 2D.1A — Image Representation Protocol Decision
Environment: Local
Status: OPEN / CURRENT
```

Planned subphases:

```text
Phase 2D.1A — Image Representation Protocol Decision
Phase 2D.1B — DICOM-to-JPG Conversion & Validation
Phase 2D.1C — MMDetection Dataset / Empty-Image Loading Validation
Phase 2D.1D — Evidence Consolidation, GPT Review & Closure
```

Representation decision:

```text
DICOM remains the immutable raw medical source.
JPG is selected as the processed training image representation.
coco_master.json remains the annotation master.
coco_master_jpg.json will be created as a training derivative.
MMDetection will use JPG + COCO-JPG for downstream loading and training.
```

Current protocol requirements:

```text
Explicit DICOM intensity transformation.
RescaleSlope/RescaleIntercept handling.
Modality LUT handling where applicable.
VOI LUT/windowing handling.
MONOCHROME1 inversion.
Intensity clipping.
uint8 conversion.
JPEG quality pilot.
No resize, crop or rotation.
Original width and height preservation.
BBox invariance validation.
DICOM-to-JPG traceability.
MMDetection empty-GT retention validation.
```

Current gate:

```text
Phase 2D.1A: OPEN / CURRENT
Phase 2D.1B: LOCKED
Phase 2D.1C: LOCKED
Phase 2D.1D: LOCKED

jpg_training_representation_ready: false
coco_jpg_training_annotation_ready: false
mmdetection_dataset_loading_ready: false
empty_image_retention_ready: false
dataset_training_ready: false
training_authorized: false
```

Forbidden actions:

```text
No full DICOM-to-JPG conversion before Phase 2D.1A and pilot review PASS.
No train/val/test split.
No labeled/unlabeled split.
No detector training.
No inference.
No pseudo-labeling.
No threshold tuning.
No AP/mAP computation.
No test-set usage.
```

Next action:

```text
Create and review the Phase 2D.1A representation protocol.
```
### Phase 2D.1 — JPG Training Representation & MMDetection Empty-Image Loading Validation

Status: **IN PROGRESS**

Date opened: 2026-07-15

Current subphase:

```text
Phase 2D.1A — Image Representation Protocol Decision
Environment: Local
Status: OPEN / CURRENT
```

Planned subphases:

```text
Phase 2D.1A — Image Representation Protocol Decision
Phase 2D.1B — DICOM-to-JPG Conversion & Validation
Phase 2D.1C — MMDetection Dataset / Empty-Image Loading Validation
Phase 2D.1D — Evidence Consolidation, GPT Review & Closure
```

Representation decision:

```text
DICOM remains the immutable raw medical source.
JPG is selected as the processed training image representation.
coco_master.json remains the annotation master.
coco_master_jpg.json will be created as a training derivative.
MMDetection will use JPG + COCO-JPG for downstream loading and training.
```

Current protocol requirements:

```text
Explicit DICOM intensity transformation.
RescaleSlope/RescaleIntercept handling.
Modality LUT handling where applicable.
VOI LUT/windowing handling.
MONOCHROME1 inversion.
Intensity clipping.
uint8 conversion.
JPEG quality pilot.
No resize, crop or rotation.
Original width and height preservation.
BBox invariance validation.
DICOM-to-JPG traceability.
MMDetection empty-GT retention validation.
```

Current gate:

```text
Phase 2D.1A: OPEN / CURRENT
Phase 2D.1B: LOCKED
Phase 2D.1C: LOCKED
Phase 2D.1D: LOCKED

jpg_training_representation_ready: false
coco_jpg_training_annotation_ready: false
mmdetection_dataset_loading_ready: false
empty_image_retention_ready: false
dataset_training_ready: false
training_authorized: false
```

Forbidden actions:

```text
No full DICOM-to-JPG conversion before Phase 2D.1A and pilot review PASS.
No train/val/test split.
No labeled/unlabeled split.
No detector training.
No inference.
No pseudo-labeling.
No threshold tuning.
No AP/mAP computation.
No test-set usage.
```

Next action:

```text
Create and review the Phase 2D.1A representation protocol.
```
