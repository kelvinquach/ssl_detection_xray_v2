# Nhật ký nghiên cứu `ssl_detection_xray_v2`

Ghi theo thứ tự thời gian. Mỗi entry cần có: mục tiêu, việc đã làm, evidence, kết quả review, quyết định tiếp theo.

---

## 2026-06-18 — PHASE 0: Khởi tạo repo & môi trường

### Mục tiêu

Dựng cấu trúc repo, tài liệu, môi trường Python cơ bản và utility tái lập cho đề tài:

**“Nghiên cứu học bán giám sát cho dò tìm bất thường trên X-quang phổi.”**

Trọng tâm: semi-supervised object detection trên VinBigData Chest X-ray.

Phase 0 chỉ phục vụ setup và reproducibility.

Không đọc dataset, không kiểm tra annotation, không convert COCO, không tạo split, không train.

---

### Đã làm

#### 1. Tạo cấu trúc repo

Đã tạo các thư mục chính:

* `configs/protocol/`
* `data/raw/`
* `data/interim/`
* `data/processed/`
* `data/manifests/`
* `src/utils/`
* `scripts/`
* `experiments/`
* `reports/`
* `plots/`
* `models/`
* `logs/`
* `tests/`
* `draft/`

#### 2. Tạo tài liệu dự án

Đã tạo:

* `README.md`
* `CLAUDE.md`
* `STRUCTURE.md`
* `RESEARCH_CHECKLIST.md`
* `repository_structure.md`
* `research_log.md`

#### 3. Tạo file môi trường

Đã tạo:

* `requirements.txt`
* `requirements_phase0_base.txt`
* `environment.yml`
* `scripts/00_setup_environment.sh`

Framework chính được định hướng là **MMDetection**.

Detectron2 chỉ được ghi nhận là optional, không dùng làm framework mặc định.

#### 4. Tạo utility tái lập

Đã tạo:

* `src/utils/seed.py`
* `src/utils/env.py`

`src/utils/seed.py` hỗ trợ:

* `set_global_seed()`
* `get_rng_state_summary()`
* `save_seed_manifest()`

`src/utils/env.py` hỗ trợ:

* ghi nhận Python/platform
* kiểm tra import package
* kiểm tra CUDA
* ghi nhận trạng thái framework detection

#### 5. Tạo script kiểm tra môi trường

Đã tạo và chạy:

* `scripts/00_check_environment.py`

Script có nhiệm vụ:

* set seed mặc định `2026`
* ghi environment report
* ghi seed manifest
* ghi pip freeze
* không crash nếu `mmengine/mmcv/mmdet` chưa cài
* ghi rõ `framework_import_ok: false` nếu MMDetection stack chưa import được

#### 6. Khóa protocol checkpoint/evaluation

Đã tạo:

* `configs/protocol/checkpoint_policy.yaml`

Nội dung protocol đã khóa:

* primary metric: `mAP@0.5:0.95`
* checkpoint selection split: `val`
* test set chỉ dùng cho final evaluation
* không dùng test set để tune threshold
* không dùng test set để chọn checkpoint
* không dùng test set để chọn model/backbone
* không dùng test set để quyết định augmentation

#### 7. Tạo `.gitignore`

`.gitignore` đã loại trừ dữ liệu nặng, ảnh, checkpoint, logs, virtual environment và cache.

---

### Evidence đã tạo

Đã chạy lệnh:

```cmd
python scripts/00_check_environment.py --seed 2026 --output reports/phase0_environment_check.json --seed-manifest data/manifests/seed_state_manifest.json --freeze-output reports/phase0_pip_freeze.txt
```

Các file evidence đã sinh:

* `reports/phase0_environment_check.json`
* `reports/phase0_pip_freeze.txt`
* `data/manifests/seed_state_manifest.json`

Đã kiểm tra dependency bằng:

```cmd
pip check
```

Kết quả:

```text
No broken requirements found.
```

---

### Kết quả environment check

Kết quả chính từ `reports/phase0_environment_check.json`:

* Python: `3.10.20`
* Platform: `Windows-10-10.0.26200-SP0`
* Conda environment: `sslxray`
* Python executable: `C:\Users\USER\anaconda3\envs\sslxray\python.exe`
* Seed: `2026`

Core imports:

* `torch`: OK, version `2.3.1`
* `torchvision`: OK, version `0.18.1`
* `numpy`: OK, version `1.24.3`
* `pandas`: OK, version `2.3.3`
* `cv2`: OK, version `4.11.0`
* `pydicom`: OK, version `3.0.2`
* `pycocotools`: OK, version `2.0.11`

Detection framework imports:

* `mmengine`: FAIL / not installed
* `mmcv`: FAIL / not installed
* `mmdet`: FAIL / not installed

CUDA:

* `torch.cuda.is_available()`: `False`
* `torch.version.cuda`: `null`
* GPU device count: `0`

Summary:

* `core_import_ok`: `true`
* `framework_import_ok`: `false`
* primary framework: `mmdetection`
* detectron2: `optional`

---

### Seed và deterministic settings

Seed manifest đã được tạo tại:

* `data/manifests/seed_state_manifest.json`

Seed settings:

* Global seed: `2026`
* `PYTHONHASHSEED`: `2026`
* Python random seed: enabled
* NumPy seed: enabled
* PyTorch CPU seed: enabled
* PyTorch CUDA seed: not applied because CUDA is unavailable

Deterministic flags:

* `torch.use_deterministic_algorithms`: `true`
* `torch.backends.cudnn.deterministic`: `true`
* `torch.backends.cudnn.benchmark`: `false`
* `CUBLAS_WORKSPACE_CONFIG`: `:4096:8`

---

### Vấn đề đã gặp

Trong quá trình thử cài OpenMIM/MMDetection local, `openmim` kéo thêm nhiều dependency phụ và làm môi trường bị lệch nhẹ. Sau đó đã gỡ `openmim` và repair môi trường.

Sau khi repair:

```cmd
pip check
```

cho kết quả:

```text
No broken requirements found.
```

Quyết định: không ép cài MMDetection trên Windows CPU-only local ở Phase 0.

---

### Review GPT

Phase 0A — Repository structure: **PASS**

Phase 0B — Core Python environment: **PASS**

Phase 0B — Local training framework: **DEFERRED**

Lý do:

* Core packages import được.
* Seed và deterministic settings đã được ghi nhận.
* Environment report, pip freeze và seed manifest đã được sinh.
* Local CUDA không khả dụng.
* MMDetection stack chưa import được.
* Local environment chưa được xem là training-ready.

---

### Quyết định

Local environment được chấp nhận cho:

* kiểm tra repo
* kiểm tra script
* kiểm tra metadata
* kiểm tra annotation
* tạo report
* tạo split sau khi Phase 1/2 pass DoD
* kiểm tra COCO format sau khi đến đúng phase

Local environment không được dùng cho:

* detector training
* checkpoint selection
* SSL pseudo-label training
* final evaluation
* threshold tuning
* model/backbone selection

MMDetection/GPU training environment sẽ được setup riêng ở môi trường remote/GPU sau.

---

### Ràng buộc tuân thủ

Trong Phase 0 đã tuân thủ:

* Không đọc dataset.
* Không đọc `train.csv`.
* Không convert COCO.
* Không tạo split.
* Không train.
* Không pseudo-label.
* Không tune threshold.
* Không dùng test set.

---

### Trạng thái checklist

Được tick:

* Phase 0A repo structure
* Phase 0B core environment
* pip dependency check
* PyTorch/torchvision import
* numpy/pandas/cv2/pydicom/pycocotools import
* seed manifest
* deterministic flags
* environment report
* pip freeze
* checkpoint policy

Chưa tick:

* MMDetection import OK
* `mmengine` import OK
* `mmcv` import OK
* `mmdet` import OK
* CUDA/GPU ready
* Local training-ready environment
* Full detection framework setup

---

### Việc cần làm tiếp trước khi mở Phase 1

Cần bổ sung:

* `reports/reproducibility_settings.md`

Cần chạy:

```cmd
python -m pytest tests\test_phase0.py -q
```

Sau khi `reproducibility_settings.md` tồn tại và test Phase 0 pass, có thể đóng **Phase 0 core** và xin review để mở **Phase 1A — Data Overview**.

---
## 2026-06-19 — PHASE 1A: Dataset Overview

### Dataset scope clarification after Phase 1A

Phase 1A dataset overview was performed on the full VinBigData `train.csv` source metadata.

Observed source metadata:

```text
Total images: 15,000
Abnormal images: 4,394
No Finding images: 10,606
Annotation rows: 67,914
Abnormal bbox rows: 36,096
Abnormal detection classes: 14
```

Research decision:

```text
Downstream controlled working scope is locked to 4,894 images:
- 4,394 abnormal images
- 500 No Finding images
```

Important distinction:

```text
The full 15,000-image CSV is source metadata only.
It is not the downstream working dataset.
The 4,894-image subset has not been constructed in Phase 1A.
Subset manifest creation belongs to a later dataset scope / manifest phase.
```

Scope lock:

```text
No split was created in Phase 1A.
No COCO conversion was performed in Phase 1A.
No image files were copied or read in Phase 1A.
No training, pseudo-labeling, or threshold tuning was performed in Phase 1A.
```

## 2026-06-19 — PHASE 1B: Annotation Quality

Status: PASS

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

Next phase:

Phase 1C — Dataset Scope Decision


Lệnh commit:

```cmd
git status
git add scripts/01B_annotation_quality.py reports/phase1B_annotation_quality.json reports/phase1B_annotation_quality.md reports/annotation_sanity_report.md reports/invalid_bbox_rows.csv reports/duplicate_bbox_candidates.csv reports/phase1B_class_mapping.csv reports/phase1B_bbox_quality_by_class.csv reports/phase1B_image_label_consistency.csv PROJECT_CONTEXT.md research_log.md CHECKLIST_TRIEN_KHAI_FULL.xlsx
git commit -m "phase1B: validate annotation quality"
git push
```
---

## 2026-07-01 — PHASE 1C: Dataset Scope Decision

### Mục tiêu

Chính thức hóa downstream controlled working scope cho đề tài:

**“Nghiên cứu học bán giám sát cho dò tìm bất thường trên X-quang phổi.”**

Mục tiêu Phase 1C:

```text
Controlled working scope = 4,894 image-level samples
= 4,394 abnormal images
+ 500 No Finding images
```

Phase 1C chỉ khóa phạm vi dữ liệu ở mức metadata/image-level. Phase này không tạo split, không convert COCO, không train, không pseudo-label, không tune threshold, không dùng test set, không đọc pixel ảnh, không đọc DICOM header và không kiểm tra image-boundary validity.

---

### Đã làm

#### 1. Tạo script Phase 1C

Đã tạo:

```text
scripts/01C_dataset_scope_decision.py
```

Script có nhiệm vụ:

* đọc full VinBigData `train.csv`;
* đọc DICOM package manifests `dicom_package_manifest_part_*.csv`;
* scan DICOM filename inventory từ `dicom_subset/train/*.dicom`;
* đọc `dicom_chunk_summary.csv`;
* đối chiếu `image_id` giữa package manifest, DICOM filename và `train.csv`;
* tạo selected image-level manifest;
* tạo metadata-only subset annotation CSV;
* tạo class distribution trong controlled scope;
* tạo No Finding audit;
* xác nhận các hành động bị cấm không xảy ra.

#### 2. Khóa scope theo package manifest đã tải

Phase 1C không random sample lại 500 No Finding từ `train.csv`.

Scope chính thức được lấy từ package manifests đã tải trước đó:

```text
D:\ssl_detection_xray\data\raw\vinbigdata\dicom_subset_chunks\dicom_package_manifest_part_*.csv
```

Nguồn kiểm tra chéo:

```text
D:\ssl_detection_xray\data\raw\vinbigdata\dicom_subset\train\*.dicom
```

Nguồn metadata gốc:

```text
D:\ssl_detection_xray_v2\data\raw\vinbigdata\annotations\train.csv
```

#### 3. Đối chiếu package manifest với DICOM filename inventory

Kết quả xác nhận:

```text
Manifest parts found: 35
Manifest rows: 4,894
Manifest unique image_id: 4,894
Manifest duplicate image_id count: 0

DICOM files listed: 4,894
DICOM unique image_id: 4,894
DICOM duplicate image_id count: 0

manifest_not_in_dicom_count: 0
dicom_not_in_manifest_count: 0
```

DICOM inventory chỉ dùng filename/path để lấy `image_id`. Không đọc DICOM header, không đọc pixel và không đọc image dimensions.

#### 4. Đối chiếu selected scope với train.csv

Kết quả xác nhận:

```text
Source total rows: 67,914
Source unique images: 15,000
Source abnormal images: 4,394
Source No Finding images: 10,606
Source mixed No Finding + abnormal images: 0

Selected total images: 4,894
Selected abnormal images: 4,394
Selected No Finding images: 500
Selected mixed images: 0
Lost abnormal image count: 0
Abnormal retention rate: 1.0
Unknown manifest image_id count: 0
Image type / train.csv label mismatch count: 0
```

#### 5. Tạo selected image-level manifest

Đã tạo:

```text
data/manifests/phase1C_selected_images_manifest.csv
```

Manifest này chứa 4,894 dòng, mỗi dòng là một `image_id`.

Các cột chính:

```text
image_id
scope_label
is_abnormal
is_no_finding
source_row_count
abnormal_row_count
no_finding_row_count
bbox_row_count
abnormal_class_count
abnormal_class_names
package_image_type
chunk_id
zip_name
source_path
source_size_bytes
selected_from
selection_reason
```

#### 6. Tạo metadata-only subset annotation CSV

Đã tạo:

```text
data/interim/vinbigdata_phase1C_scope_annotations.csv
```

File này gồm toàn bộ annotation rows trong `train.csv` có `image_id` thuộc selected manifest.

Kết quả:

```text
Selected subset rows: 37,596
Selected abnormal rows: 36,096
Selected No Finding rows: 1,500
```

Điểm quan trọng:

```text
36,096 abnormal rows + 1,500 No Finding rows = 37,596 selected rows
```

Điều này xác nhận subset annotation CSV lấy toàn bộ rows thuộc selected image_id, không sampling theo row.

#### 7. Chứng minh No Finding xử lý theo image-level

Kết quả:

```text
Selected No Finding images: 500
Selected No Finding rows: 1,500
No Finding row-level sampling used: false
```

No Finding có nhiều row do reader-level annotation. Phase 1C đã chọn 500 unique No Finding `image_id`, không chọn 500 No Finding rows.

#### 8. Tạo class distribution trong controlled scope

Đã tạo:

```text
reports/phase1C_scope_class_distribution.csv
```

Kết quả:

```text
Abnormal detection classes excluding No Finding: 14
No Finding is detection class: false
```

Class distribution trong controlled scope:

```text
Aortic enlargement: row_count=7162, image_count=3067, bbox_count=7162
Atelectasis: row_count=279, image_count=186, bbox_count=279
Calcification: row_count=960, image_count=452, bbox_count=960
Cardiomegaly: row_count=5427, image_count=2300, bbox_count=5427
Consolidation: row_count=556, image_count=353, bbox_count=556
ILD: row_count=1000, image_count=386, bbox_count=1000
Infiltration: row_count=1247, image_count=613, bbox_count=1247
Lung Opacity: row_count=2483, image_count=1322, bbox_count=2483
Nodule/Mass: row_count=2580, image_count=826, bbox_count=2580
Other lesion: row_count=2203, image_count=1134, bbox_count=2203
Pleural effusion: row_count=2476, image_count=1032, bbox_count=2476
Pleural thickening: row_count=4842, image_count=1981, bbox_count=4842
Pneumothorax: row_count=226, image_count=96, bbox_count=226
Pulmonary fibrosis: row_count=4655, image_count=1617, bbox_count=4655
```

No Finding không nằm trong detection class distribution.

---

### Evidence đã tạo

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

Key JSON evidence:

```text
dod_pass_candidate: true
selected_scope_source: package_manifest_validated_by_train_csv_and_dicom_inventory
chunk_summary_match: true
warnings: []
```

---

### Review GPT

Phase 1C — Dataset Scope Decision: **PASS**

DoD review:

```text
selected_total_images = 4,894: PASS
selected_abnormal_images = 4,394: PASS
selected_no_finding_images = 500: PASS
lost_abnormal_image_count = 0: PASS
abnormal_retention_rate = 1.0: PASS
manifest_total_rows = 4,894: PASS
manifest_unique_images = 4,894: PASS
dicom_file_count = 4,894: PASS
dicom_unique_image_ids = 4,894: PASS
manifest_not_in_dicom_count = 0: PASS
dicom_not_in_manifest_count = 0: PASS
unknown_manifest_image_id_count = 0: PASS
selected_mixed_images = 0: PASS
image_type_label_mismatch_count = 0: PASS
abnormal_detection_classes_excluding_no_finding = 14: PASS
No Finding is not detection class: PASS
No Finding row-level sampling used = false: PASS
No split: PASS
No COCO: PASS
No train: PASS
No pseudo-label: PASS
No threshold tuning: PASS
No test set: PASS
No pixel read: PASS
No DICOM header read: PASS
No image dimension read: PASS
```

---

### Quyết định

Controlled working scope chính thức được khóa:

```text
4,894 image-level samples
= 4,394 abnormal images
+ 500 No Finding images
```

Quyết định nghiên cứu:

* Giữ toàn bộ 4,394 abnormal images.
* Chỉ dùng 500/10,606 No Finding images trong controlled working scope.
* 500 No Finding được xác nhận ở `image_id` level, không phải row-level.
* Full 15,000-image `train.csv` vẫn là source metadata.
* 4,894-image scope là downstream controlled working dataset.
* No Finding tiếp tục là ảnh âm tính không có bbox, không phải detection class.
* Metadata-only subset annotation CSV được tạo để dùng cho các phase chuẩn hóa tiếp theo.
* 147 near-duplicate bbox candidates từ Phase 1B được giữ nguyên, không tự động xóa hoặc fuse.
* Boundary validation vẫn deferred sang Phase 2A vì Phase 1C không đọc image dimensions.

---

### Vấn đề / rủi ro

* Controlled scope chỉ dùng 500/10,606 No Finding images, không phải toàn bộ normal pool.
* Việc giới hạn negative pool là quyết định thiết kế controlled scope và cần ghi rõ như limitation trong thesis/paper.
* Phase 1C chưa kết luận bbox có nằm trong biên ảnh hay không.
* Chưa được claim image-boundary validity cho bbox.
* Fusion/handling near-duplicate bbox candidates là quyết định ở phase sau.
* Chưa có train/val/test split; không được dùng 4,894-image scope như training split.

---

### Ràng buộc tuân thủ

Trong Phase 1C đã tuân thủ:

```text
Không split train/val/test.
Không convert COCO.
Không train.
Không pseudo-label.
Không tune threshold.
Không dùng test set.
Không copy ảnh.
Không đọc DICOM header.
Không đọc pixel ảnh.
Không đọc image dimensions.
Không xóa/sửa annotation gốc.
Không xóa/fuse near-duplicate bbox candidates.
```

---

### Trạng thái checklist

Được tick:

* Phase 1C — Dataset Scope Decision
* Controlled working scope 4,894 images
* Retain all abnormal images
* Select/validate 500 No Finding images
* Image-level selected manifest
* Metadata-only subset annotation CSV
* No Finding image-level handling
* Controlled-scope class distribution
* Forbidden actions avoided

---

### Quyết định tiếp theo

Phase tiếp theo:

```text
Phase 1D — Kappa feasibility / limitation-aware analysis
```

Chưa được làm ở thời điểm này:

```text
Split train/val/test
COCO conversion
DICOM/image-boundary validation
Training
Pseudo-labeling
Threshold tuning
Test-set usage
```

Phase 2A chỉ được mở sau khi Phase 1D hoàn tất hoặc sau khi có quyết định chính thức bỏ/khóa Phase 1D theo protocol.

---

## 2026-07-01 — PHASE 1D: Label Reliability & Kappa Feasibility

### Mục tiêu

Kiểm tra tính khả thi của inter-radiologist agreement / Kappa analysis từ metadata hiện có trong controlled working scope của đề tài:

**“Nghiên cứu học bán giám sát cho dò tìm bất thường trên X-quang phổi.”**

Phase 1D chỉ dùng metadata annotation đã khóa từ Phase 1C. Không tạo split, không convert COCO, không train, không pseudo-label, không tune threshold, không dùng test set, không đọc pixel ảnh, không đọc DICOM/header/image dimensions và không sửa annotation gốc.

Mục tiêu chính:

```text
Kiểm tra rad_id availability.
Đếm số radiologist trên mỗi image_id.
Kiểm tra khả năng xây image-class-radiologist binary matrix.
Đánh giá Cohen’s Kappa feasibility.
Đánh giá Fleiss’ Kappa feasibility.
Báo cáo class-wise image-level agreement.
Phân tích rare-class kappa instability risk.
Tách label-level agreement khỏi bbox-level consistency.
Ghi rõ Kappa chỉ là data-quality evidence, không phải model metric.
```

---

### Đã làm

#### 1. Tạo script Phase 1D

Đã tạo:

```text
scripts/01D_kappa_feasibility.py
```

Script đọc metadata từ:

```text
data/interim/vinbigdata_phase1C_scope_annotations.csv
```

Không đọc ảnh, không đọc DICOM, không đọc DICOM header, không đọc image dimensions.

#### 2. Kiểm tra `rad_id`

Kết quả:

```text
rad_id_available: True
rad_id_column_used: rad_id
rad_id_missing_count: 0
radiologists_total: 17
```

#### 3. Đếm số radiologist trên mỗi image

Kết quả:

```text
total_images: 4894
radiologists_per_image_distribution: {'3': 4894}
uniform_rater_count_per_image: True
rater_panel_size: 3
same_rater_identity_panel_across_images: False
```

Diễn giải:

```text
Mỗi ảnh có đúng 3 radiologist ratings.
Toàn dataset có 17 radiologists khác nhau.
Số lượng rater trên mỗi ảnh là cố định, nhưng danh tính bộ 3 radiologists có thể thay đổi giữa các ảnh.
```

#### 4. Kiểm tra image-class-radiologist binary matrix

Kết quả:

```text
binary_matrix_feasible: True
```

Giả định/logic được ghi rõ:

```text
Nếu một radiologist có record trên image_id, radiologist đó được xem là đã đọc ảnh.
Nếu radiologist đó không đánh dấu class C trên ảnh đã đọc, class C được suy ra là negative cho radiologist đó.
No Finding rows có rad_id và được dùng như read-coverage signal.
```

Đây là reconstructed image-level class decision matrix, không phải metric mô hình.

#### 5. Đánh giá Cohen’s Kappa feasibility

Kết quả:

```text
cohen_kappa_feasible: False
```

Lý do:

```text
Mỗi ảnh có 3 raters.
Cohen’s Kappa là thống kê cho 2 raters.
Do đó Cohen’s Kappa không phải lựa chọn tự nhiên cho protocol chính.
```

Pairwise Cohen có thể tính trong phân tích phụ nếu sau này thật sự cần, nhưng không dùng trong Phase 1D.

#### 6. Đánh giá Fleiss’ Kappa feasibility

Kết quả:

```text
fleiss_kappa_feasible: True
overall_fleiss_kappa_mean: 0.4879
```

Lý do:

```text
Mỗi ảnh có uniform rater count = 3.
Fleiss’ Kappa cho phép nhiều raters trên mỗi item và không bắt buộc cùng danh tính raters cho mọi item.
Negative class decisions có thể suy ra từ read-coverage theo VinBigData labelling convention.
```

#### 7. Báo cáo class-wise image-level agreement

Đã báo cáo Fleiss’ Kappa cho 14 abnormal detection classes:

```text
Aortic enlargement: 0.6393
Atelectasis: 0.3568
Calcification: 0.3960
Cardiomegaly: 0.7065
Consolidation: 0.3397
ILD: 0.4604
Infiltration: 0.4119
Lung Opacity: 0.3414
Nodule/Mass: 0.4946
Other lesion: 0.3024
Pleural effusion: 0.6711
Pleural thickening: 0.3583
Pneumothorax: 0.7402
Pulmonary fibrosis: 0.6126
```

Tóm tắt:

```text
14 abnormal classes assessed.
14 classes with feasible Fleiss’ Kappa.
Mean Fleiss’ Kappa = 0.4879.
```

#### 8. Phân tích rare-class Kappa instability risk

Kết quả:

```text
rare_class_instability_summary:
5/14 classes carry kappa_instability_risk
severe = 2
moderate = 3
low = 9
```

Diễn giải:

```text
Đây là prevalence/rarity-driven risk, không phải measured instability.
Một class có thể có Kappa cao nhưng vẫn có risk nếu positive count thấp hoặc prevalence quá lệch.
```

Các class đáng chú ý:

```text
Severe risk:
- Pneumothorax
- Atelectasis

Moderate risk:
- Consolidation
- ILD
- Calcification
```

#### 9. Tách label-level agreement khỏi bbox-level consistency

Label-level agreement:

```text
label_level_agreement_status: evaluable_fleiss_computed
```

BBox-level consistency:

```text
bbox_level_consistency_status: evaluated_descriptive_only
pairs_compared: 35521
near_duplicate_pairs_iou_ge_threshold: 78
iou_threshold: 0.95
```

Nguyên tắc:

```text
Label-level agreement dùng present/absent class decisions.
BBox-level consistency chỉ là descriptive spatial proximity.
Không xóa bbox.
Không fuse bbox.
Không sửa annotation.
Không xem near-duplicate bbox là lỗi chắc chắn.
```

#### 10. Ghi chú về khác biệt Phase 1B và Phase 1D near-duplicate count

Phase 1B báo:

```text
Near-duplicate candidates IoU >= 0.95: 147
```

Phase 1D báo:

```text
Near-duplicate bbox pairs IoU >= 0.95: 78
```

Hai con số này không mâu thuẫn vì khác đơn vị đếm và khác mục tiêu:

```text
Phase 1B đếm candidate bbox records/rows liên quan đến near-duplicate để phục vụ annotation-quality review.
Phase 1D đếm bbox pairs có IoU >= 0.95 để mô tả inter-rater spatial consistency.
```

Nếu 78 pairs hoàn toàn rời nhau, chúng có thể tương ứng tối đa 156 bbox records. Việc Phase 1B có 147 candidate records là hợp lý vì một số bbox có thể tham gia nhiều pair và được tính một lần ở mức candidate record.

Phase 1D vẫn giữ bbox-level consistency là descriptive only. Không bbox nào bị xóa, fuse hoặc xem là annotation error chắc chắn.

---

### Evidence đã tạo

Script run:

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

Key JSON evidence:

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
dod_status: PASS_agreement_computed_and_documented
```

Forbidden actions confirmed:

```text
split_created: false
coco_created: false
training_started: false
pseudo_label_generated: false
threshold_tuned: false
test_set_used: false
pixel_read: false
dicom_or_header_read: false
image_dimensions_read: false
boundary_validation: false
annotations_deleted_or_edited: false
near_duplicate_bbox_deleted_or_fused: false
kappa_used_as_model_metric: false
kappa_used_for_split_model_threshold: false
```

---

### Review GPT

Phase 1D — Label Reliability & Kappa Feasibility: **PASS**

DoD review:

```text
rad_id availability checked: PASS
radiologists per image reported: PASS
image-class-radiologist binary matrix feasibility checked: PASS
Cohen’s Kappa feasibility documented: PASS
Fleiss’ Kappa feasibility documented: PASS
Kappa unit/assumption/limitation documented: PASS
Class-wise image-level agreement reported: PASS
Rare-class kappa instability risk analyzed: PASS
Label-level agreement separated from bbox-level consistency: PASS
No annotation deletion/fusion/editing: PASS
Kappa not used as model metric: PASS
Kappa not used for split/tune/train/pseudo-label: PASS
Markdown report saved: PASS
JSON metrics saved: PASS
Forbidden actions avoided: PASS
```

---

### Quyết định

Phase 1D được khóa với trạng thái:

```text
PASS_agreement_computed_and_documented
```

Quyết định nghiên cứu:

* Fleiss’ Kappa được tính ở mức image-level class agreement.
* Cohen’s Kappa không dùng làm metric chính vì mỗi ảnh có 3 raters.
* Mean Fleiss’ Kappa across abnormal classes = 0.4879.
* Kappa/agreement chỉ dùng làm data-quality evidence và limitation evidence.
* Kappa không dùng làm model metric.
* Kappa không dùng để chọn split/model/threshold.
* Kappa không dùng để train hoặc pseudo-label.
* Kappa không dùng để sửa/xóa/fuse annotation.
* Rare-class risk được ghi nhận để giải thích độ tin cậy annotation theo class.
* BBox-level consistency được giữ riêng với label-level agreement và chỉ dùng mô tả.

---

### Vấn đề / rủi ro

* Negative decisions được suy ra từ read-coverage theo VinBigData labelling convention; đây là giả định cần ghi rõ trong thesis.
* Mean Fleiss’ Kappa ở mức trung bình, cho thấy annotation reliability không đồng đều tuyệt đối.
* Một số class có Kappa thấp, ví dụ Other lesion, Consolidation, Lung Opacity, Pleural thickening và Atelectasis.
* Một số class có rare/prevalence risk, đặc biệt Pneumothorax và Atelectasis.
* Kappa chịu ảnh hưởng bởi prevalence imbalance; không nên diễn giải như chất lượng annotation tuyệt đối.
* BBox-level consistency chưa phải bbox fusion policy.
* Việc xử lý multi-radiologist boxes / near-duplicate boxes vẫn cần quyết định ở phase chuẩn hóa annotation sau.

---

### Ràng buộc tuân thủ

Trong Phase 1D đã tuân thủ:

```text
Không split train/val/test.
Không convert COCO.
Không train.
Không pseudo-label.
Không tune threshold.
Không dùng test set.
Không đọc pixel ảnh.
Không đọc DICOM/PNG.
Không đọc DICOM header.
Không đọc image dimensions.
Không boundary validation.
Không xóa/sửa annotation gốc.
Không xóa/fuse near-duplicate bbox candidates.
Không dùng Kappa làm model metric.
Không dùng Kappa để split/model/threshold.
```

---

### Trạng thái checklist

Được tick:

* Phase 1D — Label Reliability & Kappa Feasibility
* rad_id availability
* radiologists per image
* image-class-radiologist binary matrix feasibility
* Cohen’s Kappa feasibility
* Fleiss’ Kappa feasibility
* class-wise image-level agreement
* rare-class kappa instability risk
* label-level agreement vs bbox-level consistency
* Kappa as data-quality evidence only
* reports/phase1D_kappa_feasibility.md
* reports/phase1D_kappa_feasibility.json
* forbidden actions avoided

---

### Quyết định tiếp theo

Phase tiếp theo:

```text
Phase 2A — Data Standardization / Image-Boundary Validation
```

Chưa được làm trước khi mở Phase 2A:

```text
Train/val/test split
COCO conversion
Training
Pseudo-labeling
Threshold tuning
Test-set usage
```

Phase 2A chỉ được mở sau khi Phase 1D evidence đã commit và push GitHub.

---

## 2026-07-08 — PHASE 2A: Data Standardization / Image-Boundary Validation

### Mục tiêu

Kiểm tra image availability, DICOM metadata/image dimensions và bbox boundary validity trong controlled working scope đã khóa của đề tài:

**“Nghiên cứu học bán giám sát cho dò tìm bất thường trên X-quang phổi.”**

Phase 2A chỉ thực hiện data standardization / image-boundary validation ở mức DICOM metadata và annotation boundary.

Phase 2A không tạo split, không convert COCO, không train, không pseudo-label, không tune threshold, không dùng test set, không sửa annotation gốc, không xóa/clamp/fuse bbox và không tạo processed training images.

Mục tiêu chính:

```text
Kiểm tra đủ 4,894 DICOM files trong controlled scope.
Đọc DICOM metadata/header để lấy image_width/image_height.
Xác nhận bbox abnormal nằm trong biên ảnh gốc.
Xác nhận No Finding là ảnh âm tính không có bbox.
Ghi evidence trước khi sang canonical schema / COCO / split.
```

---

### Đã làm

#### 1. Tạo script Phase 2A

Đã tạo:

```text
scripts/02A_dicom_bbox_boundary_validation.py
```

Script có nhiệm vụ:

* đọc controlled-scope annotation CSV;
* đọc selected image manifest;
* index DICOM files trong local DICOM root;
* kiểm tra image availability cho toàn bộ 4,894 image_id;
* đọc DICOM header bằng `pydicom.dcmread(..., stop_before_pixels=True, force=True)`;
* lấy `Rows`, `Columns` làm `image_height`, `image_width`;
* validate bbox theo image boundary;
* tạo báo cáo Markdown, JSON và các CSV evidence;
* xác nhận các forbidden actions không xảy ra.

#### 2. Chạy script với đúng DICOM root

Lệnh đã chạy:

```cmd
python scripts\02A_dicom_bbox_boundary_validation.py ^
  --annotations-csv data\interim\vinbigdata_phase1C_scope_annotations.csv ^
  --manifest-csv data\manifests\phase1C_selected_images_manifest.csv ^
  --dicom-root D:\ssl_detection_xray\data\raw\vinbigdata\dicom_subset\train
```

#### 3. Kiểm tra image availability

Kết quả:

```text
dicom files indexed under root: 4,894
total selected images: 4,894
availability checked images: 4,894
DICOM available: 4,894
DICOM missing: 0
```

Điều này xác nhận toàn bộ 4,894 image_id trong controlled scope đều có file DICOM local tương ứng.

#### 4. Đọc DICOM metadata / image dimensions

Kết quả:

```text
DICOM read success: 4,894
DICOM read error: 0
image_dimension_available_count: 4,894
image_dimension_missing_count: 0
```

Image dimension summary:

```text
width_min: 1320
width_max: 3320
width_mean: 2491.66
height_min: 1416
height_max: 3408
height_mean: 2835.09
distinct_wh_pairs: 2186
```

Phase 2A đọc DICOM metadata/header để lấy dimensions. Pixel array không được đọc trong run chính:

```text
pixel_array_checked: false
pixel_array_check_count: 0
pixel_array_error_count: 0
```

#### 5. Validate bbox boundary

Quy ước bbox:

```text
xyxy trên ảnh gốc:
x_min, y_min, x_max, y_max
```

Điều kiện hợp lệ:

```text
0 <= x_min < x_max <= image_width
0 <= y_min < y_max <= image_height
```

Kết quả:

```text
abnormal_bbox_rows_checked: 36,096
bbox_boundary_valid_count: 36,096
bbox_boundary_invalid_count: 0
```

Invalid bbox by reason:

```text
missing_coordinate: 0
non_numeric_coordinate: 0
x_min_negative: 0
y_min_negative: 0
x_max_negative: 0
y_max_negative: 0
x_min_ge_x_max: 0
y_min_ge_y_max: 0
bbox_width_le_0: 0
bbox_height_le_0: 0
x_max_gt_image_width: 0
y_max_gt_image_height: 0
image_dimension_missing: 0
dicom_missing_or_read_error: 0
```

#### 6. Kiểm tra No Finding policy

Kết quả:

```text
no_finding_images: 500
no_finding_rows: 1,500
no_finding_with_bbox_count: 0
abnormal_missing_bbox_count: 0
```

Diễn giải:

```text
No Finding tiếp tục được xử lý là ảnh âm tính không có bbox.
No Finding không phải detection class.
Không có No Finding row nào có bbox.
```

---

### Evidence đã tạo

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

Key JSON evidence:

```text
dicom_files_indexed_under_root: 4894
total_annotation_rows: 37596
unique_annotation_images: 4894
manifest_rows: 4894
manifest_unique_images: 4894
selected_scope_expected_images: 4894
availability_checked_image_count: 4894
abnormal_images: 4394
no_finding_images: 500
abnormal_rows: 36096
no_finding_rows: 1500
dicom_available_count: 4894
dicom_missing_count: 0
dicom_read_success_count: 4894
dicom_read_error_count: 0
image_dimension_available_count: 4894
image_dimension_missing_count: 0
abnormal_bbox_rows_checked: 36096
bbox_boundary_valid_count: 36096
bbox_boundary_invalid_count: 0
no_finding_with_bbox_count: 0
abnormal_missing_bbox_count: 0
annotation_not_in_manifest_count: 0
manifest_not_in_annotation_count: 0
duplicate_manifest_image_id_count: 0
warnings: []
dod_pass_candidate: true
```

Forbidden actions confirmed:

```text
split_created: false
coco_created: false
training_started: false
pseudo_label_generated: false
threshold_tuned: false
test_set_used: false
annotations_deleted_or_edited: false
bbox_clamped_or_modified: false
near_duplicate_bbox_deleted_or_fused: false
processed_training_images_created: false
image_files_copied: false
image_files_converted: false
png_or_jpg_created: false
```

---

### Review GPT

Phase 2A — Data Standardization / Image-Boundary Validation: **PASS**

DoD review:

```text
Controlled scope expected images = 4,894: PASS
Manifest unique images = 4,894: PASS
Annotation unique images = 4,894: PASS
DICOM files indexed under root = 4,894: PASS
Availability checked images = 4,894: PASS
DICOM available/missing = 4,894 / 0: PASS
DICOM read success/error = 4,894 / 0: PASS
Image dimensions available/missing = 4,894 / 0: PASS
Abnormal bbox rows checked = 36,096: PASS
BBox valid/invalid = 36,096 / 0: PASS
No Finding rows with bbox = 0: PASS
Abnormal rows missing bbox = 0: PASS
Annotation not in manifest = 0: PASS
Manifest not in annotation = 0: PASS
Duplicate manifest image_id = 0: PASS
Forbidden actions avoided: PASS
dod_pass_candidate = true: PASS
```

---

### Quyết định

Phase 2A được khóa với trạng thái:

```text
PASS
```

Quyết định nghiên cứu:

* Toàn bộ 4,894 DICOM files trong controlled working scope tồn tại trên local disk.
* Toàn bộ 4,894 DICOM files đọc được metadata/header thành công.
* Image dimensions được lấy thành công cho toàn bộ 4,894 images.
* Toàn bộ 36,096 abnormal bbox nằm hợp lệ trong image boundary theo quy ước xyxy.
* Không có bbox nào bị missing coordinate, non-numeric, negative, zero-area hoặc vượt biên ảnh.
* No Finding tiếp tục là ảnh âm tính không có bbox, không phải detection class.
* Không có No Finding row nào có bbox.
* Không sửa, clamp, xóa hoặc fuse bbox.
* Không tạo split, COCO, training data, pseudo-label hoặc threshold.
* Pixel array chưa được kiểm tra trong run chính; Phase 2A chỉ xác nhận metadata/header và bbox boundary.

---

### Vấn đề / rủi ro

* Phase 2A chưa kiểm tra pixel array decoding toàn bộ ảnh vì `pixel_array_checked = false`.
* Phase 2A chưa tạo canonical schema.
* Phase 2A chưa convert COCO.
* Phase 2A chưa tạo train/val/test split.
* Phase 2A chưa kiểm tra framework dataloader / empty image loading.
* Việc xử lý near-duplicate bbox candidates vẫn chưa được quyết định; Phase 2A chỉ xác nhận bbox nằm trong biên ảnh.
* Dataset chưa được xem là training-ready cho detector cho đến khi các phase schema/COCO/split/loading pass DoD.

---

### Ràng buộc tuân thủ

Trong Phase 2A đã tuân thủ:

```text
Không split train/val/test.
Không convert COCO.
Không train.
Không pseudo-label.
Không tune threshold.
Không dùng test set.
Không sửa annotation gốc.
Không xóa bbox.
Không clamp bbox.
Không fuse near-duplicate bbox.
Không xóa/sửa 147 near-duplicate bbox candidates.
Không copy ảnh.
Không convert ảnh sang PNG/JPG.
Không tạo processed training dataset.
Không dùng Kappa làm model metric.
Không dùng Kappa để split/model/threshold.
```

---

### Trạng thái checklist

Được tick:

* Phase 2A — Data Standardization / Image-Boundary Validation
* DICOM availability check
* DICOM metadata/header read
* Image dimension extraction
* BBox boundary validation
* No Finding bbox policy check
* reports/phase2A_dicom_bbox_validation.md
* reports/phase2A_dicom_bbox_validation.json
* reports/phase2A_image_metadata.csv
* reports/phase2A_image_availability.csv
* reports/phase2A_bbox_boundary_validation.csv
* reports/phase2A_invalid_bbox_candidates.csv
* reports/phase2A_dicom_read_errors.csv
* forbidden actions avoided

---

### Quyết định tiếp theo

Phase tiếp theo:

```text
Phase 2B — Canonical Schema
```

Chưa được làm trước khi mở Phase 2B:

```text
Canonical annotation schema
COCO conversion
Train/val/test split
Labeled/unlabeled split
Training
Pseudo-labeling
Threshold tuning
Test-set usage
```

Phase 2B chỉ được mở sau khi Phase 2A evidence đã commit và push GitHub.

## 2026-07-08 — PHASE 2B: Canonical Detection Annotation Schema

### Mục tiêu

Tạo canonical detection annotation schema cho controlled working scope đã khóa của đề tài:

**“Nghiên cứu học bán giám sát cho dò tìm bất thường trên X-quang phổi.”**

Phase 2B chỉ chuẩn hóa annotation thành schema canonical trung gian. Phase này không convert COCO, không tạo train/val/test split, không train, không pseudo-label, không tune threshold, không dùng test set, không sửa annotation gốc, không clamp bbox, không xóa bbox và không fuse near-duplicate bbox.

Mục tiêu chính:

```text
Tạo canonical image table.
Tạo canonical bbox / detection annotation table.
Tạo canonical class mapping.
Audit No Finding policy.
Validate schema consistency.
Thiết kế path portable, không phụ thuộc tuyệt đối vào ổ D:\.
```

---

### Đã làm

#### 1. Tạo script Phase 2B

Đã tạo:

```text
scripts/02B_build_canonical_schema.py
```

Script có nhiệm vụ:

* đọc controlled-scope annotation CSV từ Phase 1C;
* đọc selected image manifest;
* đọc image metadata từ Phase 2A;
* đọc bbox boundary validation từ Phase 2A;
* tạo canonical image table;
* tạo canonical bbox table;
* tạo canonical class mapping;
* audit No Finding policy;
* validate consistency/schema;
* xác nhận các forbidden actions không xảy ra.

#### 2. Chạy script Phase 2B

Lệnh đã chạy:

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

#### 3. Tạo canonical image table

Đã tạo:

```text
data/processed/canonical/canonical_image_table.csv
```

Kết quả:

```text
canonical_image_rows: 4894
canonical_image_unique_images: 4894
abnormal_images: 4394
no_finding_images: 500
```

Mỗi dòng tương ứng một `image_id` trong controlled working scope.

Các cột chính gồm:

```text
canonical_image_id
image_id
dicom_filename
relative_dicom_path
dicom_path
local_dicom_path
local_dicom_path_is_absolute
path_root_variable
image_width
image_height
scope_label
is_abnormal
is_negative
has_bbox
bbox_count
no_finding_bbox_count
abnormal_class_count
abnormal_class_names
source_row_count
abnormal_row_count
no_finding_row_count
```

#### 4. Tạo canonical bbox table

Đã tạo:

```text
data/processed/canonical/canonical_bbox_table.csv
```

Kết quả:

```text
canonical_bbox_rows: 36096
bbox_without_image_count: 0
bbox_missing_dimension_count: 0
bbox_invalid_count: 0
```

Mỗi dòng tương ứng một abnormal bbox row.

BBox format được giữ là:

```text
xyxy_original_image
```

Không bbox nào bị sửa, clamp, xóa, fuse hoặc convert sang COCO trong Phase 2B.

#### 5. Tạo canonical class mapping

Đã tạo:

```text
data/processed/canonical/canonical_class_mapping.csv
```

Kết quả:

```text
canonical_class_count: 14
no_finding_in_detection_classes: false
class_mapping_issue_count: 0
```

Canonical class mapping gồm 14 abnormal detection classes:

```text
Aortic enlargement
Atelectasis
Calcification
Cardiomegaly
Consolidation
ILD
Infiltration
Lung Opacity
Nodule/Mass
Other lesion
Pleural effusion
Pleural thickening
Pneumothorax
Pulmonary fibrosis
```

No Finding không nằm trong detection class mapping.

#### 6. Audit No Finding policy

Đã tạo:

```text
reports/phase2B_no_finding_policy_audit.csv
```

Kết quả:

```text
no_finding_images: 500
no_finding_policy_pass: true
no_finding_in_detection_classes: false
```

Quyết định tiếp tục giữ:

```text
No Finding là ảnh âm tính ở image-level.
No Finding không có bbox.
No Finding không phải detection class.
No Finding không xuất hiện trong canonical bbox table.
```

#### 7. Thiết kế portable path policy

Phase 2B đã sửa canonical path schema để không phụ thuộc tuyệt đối vào ổ `D:\`.

Kết quả validation:

```text
portable_path_policy_pass: true
relative_dicom_path_missing_count: 0
relative_dicom_path_absolute_count: 0
local_dicom_path_absolute_count: 4894
path_root_variable: VINBIGDATA_DICOM_ROOT
```

Quy ước path:

```text
relative_dicom_path = train/<image_id>.dicom
dicom_path = train/<image_id>.dicom
local_dicom_path = local evidence path trên máy hiện tại
local_dicom_path_is_absolute = true nếu local_dicom_path là absolute path
path_root_variable = VINBIGDATA_DICOM_ROOT
```

Downstream phase phải resolve image path bằng:

```text
VINBIGDATA_DICOM_ROOT + relative_dicom_path
```

Không dùng absolute `D:\...` làm canonical downstream identifier.

---

### Evidence đã tạo

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

Key JSON evidence:

```text
total_annotation_rows: 37596
unique_annotation_images: 4894
manifest_rows: 4894
manifest_unique_images: 4894
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
portable_path_policy_pass: true
relative_dicom_path_missing_count: 0
relative_dicom_path_absolute_count: 0
local_dicom_path_absolute_count: 4894
path_root_variable: VINBIGDATA_DICOM_ROOT
schema_error_count: 0
warnings: []
dod_pass_candidate: true
```

Forbidden actions confirmed:

```text
split_created: false
coco_created: false
training_started: false
pseudo_label_generated: false
threshold_tuned: false
test_set_used: false
annotations_deleted_or_edited: false
bbox_clamped_or_modified: false
near_duplicate_bbox_deleted_or_fused: false
image_files_copied: false
image_files_converted: false
processed_training_images_created: false
```

---

### Review GPT

Phase 2B — Canonical Detection Annotation Schema: **PASS**

DoD review:

```text
canonical_image_rows = 4,894: PASS
canonical_image_unique_images = 4,894: PASS
canonical_bbox_rows = 36,096: PASS
canonical_class_count = 14: PASS
No Finding policy audit: PASS
No Finding excluded from detection classes: PASS
No Finding excluded from bbox table: PASS
bbox_without_image_count = 0: PASS
image_without_metadata_count = 0: PASS
bbox_missing_dimension_count = 0: PASS
bbox_invalid_count = 0: PASS
class_mapping_issue_count = 0: PASS
schema_error_count = 0: PASS
portable_path_policy_pass = true: PASS
relative_dicom_path_absolute_count = 0: PASS
Forbidden actions avoided: PASS
dod_pass_candidate = true: PASS
```

---

### Quyết định

Phase 2B được khóa với trạng thái:

```text
PASS
```

Quyết định nghiên cứu:

* Canonical schema được chấp nhận là intermediate detection annotation schema.
* Canonical image table giữ toàn bộ 4,894 images.
* Canonical bbox table giữ toàn bộ 36,096 abnormal bbox rows.
* Canonical class mapping gồm đúng 14 abnormal detection classes.
* No Finding tiếp tục là ảnh âm tính không có bbox, không phải detection class.
* No Finding không nằm trong canonical bbox table.
* No Finding không nằm trong detection class mapping.
* BBox format tiếp tục là `xyxy_original_image`.
* Không bbox nào bị sửa, clamp, xóa, fuse hoặc convert.
* 147 near-duplicate bbox candidates vẫn được giữ nguyên; fusion/handling vẫn deferred.
* Path schema đã portable: downstream dùng `VINBIGDATA_DICOM_ROOT + relative_dicom_path`, không dùng absolute local path làm canonical key.
* Phase 2B chưa phải COCO dataset.
* Phase 2B chưa phải train/val/test split.
* Dataset vẫn chưa được xem là training-ready cho detector.

---

### Vấn đề / rủi ro

* `local_dicom_path` vẫn lưu absolute local path như một evidence path, nhưng không được dùng làm canonical downstream identifier.
* Khi chuyển sang GPU/remote/Linux/Kaggle/Vast.ai, cần set `VINBIGDATA_DICOM_ROOT` hoặc data-root config tương đương.
* `source_row_id` trace về file Phase 1C controlled-scope annotation, không nhất thiết là row index gốc của full VinBigData `train.csv`.
* Phase 2B chưa kiểm tra framework dataloader.
* Phase 2B chưa kiểm tra empty-image loading.
* Phase 2B chưa convert COCO.
* Phase 2B chưa tạo split.
* Phase 2B chưa quyết định xử lý near-duplicate bbox candidates.

---

### Ràng buộc tuân thủ

Trong Phase 2B đã tuân thủ:

```text
Không split train/val/test.
Không convert COCO.
Không train.
Không pseudo-label.
Không tune threshold.
Không dùng test set.
Không đọc pixel_array.
Không copy ảnh.
Không convert ảnh.
Không tạo processed training images.
Không sửa annotation gốc.
Không xóa bbox.
Không clamp bbox.
Không fuse near-duplicate bbox.
Không xóa/sửa 147 near-duplicate bbox candidates.
No Finding không được đưa thành detection class.
No Finding không được đưa vào canonical bbox table.
```

---

### Trạng thái checklist

Được tick:

* Phase 2B — Canonical Detection Annotation Schema
* Canonical image table
* Canonical bbox table
* Canonical class mapping
* No Finding policy audit
* Schema consistency validation
* Portable path policy
* No Finding excluded from detection classes
* No Finding excluded from bbox annotations
* Traceability preserved
* reports/phase2B_canonical_schema_report.md
* reports/phase2B_canonical_schema_validation.json
* reports/phase2B_no_finding_policy_audit.csv
* reports/phase2B_schema_consistency_errors.csv
* forbidden actions avoided
* GPT review PASS

---

### Quyết định tiếp theo

Phase tiếp theo:

```text
Phase 2C — Framework & Format Decision / COCO conversion planning
```

Chưa được làm trước khi mở Phase 2C:

```text
COCO conversion
Train/val/test split
Labeled/unlabeled split
Training
Pseudo-labeling
Threshold tuning
Test-set usage
Framework dataloader validation
Empty image loading check
```

Phase 2C chỉ được mở sau khi Phase 2B evidence đã commit và push GitHub.

## 2026-07-14 — PHASE 2C: Framework & Format Decision / COCO Conversion Planning

### Mục tiêu

Chốt framework chính, annotation format chính và protocol lập kế hoạch COCO conversion cho đề tài:

**“Nghiên cứu học bán giám sát cho dò tìm bất thường trên X-quang phổi.”**

Phase 2C chỉ ra quyết định framework/format và tạo decision/protocol evidence. Phase này không convert COCO master thật, không tạo train/val/test split, không tạo labeled/unlabeled split, không train, không pseudo-label, không tune threshold, không dùng test set, không đọc pixel array, không copy/convert ảnh và không sửa annotation/canonical schema.

Mục tiêu chính:

```text
Chọn framework chính cho downstream detection/SSOD.
So sánh MMDetection / Detectron2 / YOLO-based / Custom PyTorch-torchvision.
Chọn annotation format chính.
So sánh COCO / YOLO / Pascal VOC / JSONL-custom.
Lập kế hoạch COCO conversion cho Phase 2D.
Ghi rõ No Finding / empty image policy.
Ghi rõ bbox conversion policy.
Ghi rõ category_id policy.
Ghi rõ path portability policy.
Ghi rõ metric readiness cho mAP@0.5:0.95, AP50, AP75 và class-wise AP.
Ghi rõ DICOM loader risk và dataset chưa training-ready.
```

---

### Đã làm

#### 1. Tạo script Phase 2C

Đã tạo:

```text
scripts/02C_framework_format_decision.py
```

Script có nhiệm vụ:

* đọc canonical schema từ Phase 2B;
* xác nhận lại số lượng image, bbox và class;
* defensive import probe cho `mmengine`, `mmcv`, `mmdet`;
* chọn framework chính;
* chọn annotation format chính;
* tạo framework/config protocol;
* tạo report Markdown và JSON evidence;
* xác nhận các forbidden actions không xảy ra.

#### 2. Chạy script Phase 2C

Lệnh đã chạy:

```cmd
python scripts\02C_framework_format_decision.py
```

Console summary:

```text
primary_framework          : MMDetection
primary_annotation_format  : COCO_detection_JSON
canonical_image_rows       : 4894
canonical_bbox_rows        : 36096
canonical_class_count      : 14
no_finding_images          : 500
actual_coco_conversion_done: False
dataset_training_ready     : False
dod_pass_candidate         : True
```

Warning được ghi nhận:

```text
MMDetection stack not importable locally; this is EXPECTED in Phase 2C.
Local training framework remains deferred.
Remote/GPU environment is required for detector training.
```

#### 3. Chốt framework và bổ sung framework selection rationale

Quyết định:

```text
Primary framework: MMDetection
Fallback framework: Detectron2_optional
```

Detectron2 chỉ được dùng nếu MMDetection remote/GPU setup thất bại và phải được GPT review lại.

Phase 2C không yêu cầu `mmdet` import thành công trên local vì local environment chỉ dùng cho validation/reporting, chưa training-ready.

Đã bổ sung framework rationale evidence gồm:

```text
Table 3 — High-level framework comparison
Table 4 — Detailed framework suitability matrix
```

Các framework được so sánh:

```text
MMDetection
Detectron2
Ultralytics YOLO / YOLO-based framework
Custom PyTorch / torchvision
```

Các tiêu chí so sánh:

```text
Native object detection support
COCO dataset compatibility
COCO mAP@0.5:0.95 / pycocotools compatibility
Teacher-student / SSOD readiness
Labeled/unlabeled pipeline support
Config-based reproducibility
Empty / No Finding image handling risk
Custom DICOM loader extensibility
Pseudo-label workflow compatibility
Class-wise AP / AP50 / AP75 evaluation readiness
Implementation burden
Research reproducibility
Fit for this thesis
```

Kết luận framework:

```text
MMDetection được chọn không chỉ vì phổ biến, mà vì khớp trực tiếp với pipeline luận văn:
COCO-based detection
COCO mAP evaluation
Teacher-student semi-supervised object detection
Labeled/unlabeled data handling
Config-driven reproducibility
Giảm lượng custom training/evaluation code cần tự viết
```

Detectron2 được giữ làm fallback vì vẫn là framework detection mạnh, có hỗ trợ COCO/custom dataset và COCOEvaluator, nhưng pipeline teacher-student SSOD trong project này sẽ cần nhiều custom implementation hơn MMDetection.

YOLO-based framework bị loại khỏi vai trò framework chính vì annotation/evaluation pipeline thiên về YOLO-native, lệch khỏi COCO master + MMDetection SSOD protocol, và biểu diễn negative image bằng empty label files làm tăng rủi ro với 500 ảnh No Finding.

Custom PyTorch/torchvision bị loại vì tuy linh hoạt nhất, nhưng sẽ phải tự viết dataset, dataloader, evaluator, trainer, pseudo-label loop, EMA teacher, COCO metric integration, logging và config protocol. Rủi ro engineering và silent bug quá cao so với mục tiêu nghiên cứu của luận văn.

#### 4. Chốt annotation format

Quyết định:

```text
Primary annotation format: COCO_detection_JSON
Source of truth: canonical_detection_schema từ Phase 2B
Actual COCO conversion phase: Phase 2D
```

Phase 2C không tạo file COCO thật.

Planned COCO output cho Phase 2D:

```text
data/processed/coco/coco_master.json
```

#### 5. So sánh COCO / YOLO / Pascal VOC / JSONL-custom

Đã bổ sung comparison evidence gồm:

```text
Table 1 — High-level format comparison
Table 2 — Detailed format suitability matrix
```

Các tiêu chí so sánh:

```text
MMDetection compatibility
COCO mAP@0.5:0.95 / pycocotools compatibility
Negative / No Finding image support
Multi-class object detection support
Category metadata support
BBox coordinate fidelity
Traceability to canonical/source rows
SSOD teacher-student compatibility
Pseudo-label output compatibility
Reproducibility / ecosystem support
Implementation risk
```

Kết luận:

```text
COCO detection JSON được chọn vì phù hợp nhất với MMDetection/SSOD, hỗ trợ ảnh No Finding không annotation, tương thích COCO mAP@0.5:0.95 / pycocotools, hỗ trợ category metadata và traceability.
```

YOLO txt, Pascal VOC XML và JSONL/custom bị loại vì tăng rủi ro conversion/evaluation/custom pipeline hoặc biểu diễn negative images không sạch bằng COCO.

#### 6. COCO conversion plan cho Phase 2D

Protocol Phase 2D được lập kế hoạch như sau:

```text
COCO images: toàn bộ 4,894 controlled-scope images
COCO annotations: chỉ 36,096 abnormal bbox
COCO categories: chỉ 14 abnormal detection classes
No Finding: có trong images, không có annotation, không nằm trong categories
Không tạo background class
```

Traceability:

```text
Mỗi COCO annotation cần giữ canonical_ann_id và source_row_id.
```

#### 7. No Finding / empty image policy

Quyết định:

```text
No Finding là ảnh âm tính không có bbox.
No Finding không phải detection class.
No Finding không nằm trong COCO categories.
No Finding không có dòng trong COCO annotations.
500 No Finding images phải được giữ trong COCO images.
```

Risk bắt buộc cho MMDetection:

```text
MMDetection phải set filter_empty_gt=False hoặc cấu hình tương đương để không lọc mất 500 ảnh No Finding.
```

#### 8. BBox conversion policy

Quyết định:

```text
Source format: xyxy_original_image
Target Phase 2D format: coco_xywh_absolute
width = x_max - x_min
height = y_max - y_min
area = width * height
iscrowd = 0
```

Ràng buộc:

```text
Không clamp bbox.
Không xóa bbox.
Không fuse bbox.
147 near-duplicate bbox candidates tiếp tục được giữ nguyên.
```

#### 9. Category id policy

Quyết định:

```text
COCO category_id là số nguyên liên tục 1..14.
No Finding bị loại khỏi category_id.
canonical_class_id và class_id_original được giữ trong category metadata để traceability.
```

#### 10. Path portability policy

Quyết định:

```text
COCO file_name dùng relative_dicom_path.
Path được resolve bằng VINBIGDATA_DICOM_ROOT + relative_dicom_path.
local_dicom_path chỉ là evidence path, không phải downstream identifier.
```

#### 11. DICOM loader risk

Phase 2C ghi rõ:

```text
COCO annotation format chưa làm dataset training-ready.
MMDetection default LoadImageFromFile chưa được validate cho DICOM.
Phase sau cần custom DICOM loader hoặc processed-image conversion protocol trước khi train.
```

#### 12. Metric readiness policy

Đã bổ sung metric readiness:

```text
Phase 2C không tính AP metrics vì chưa có split, model training, inference hoặc prediction file.
COCO được chọn để bảo toàn khả năng tính metric downstream.
```

Metric chính và phụ được chuẩn bị ở mức protocol:

```text
Primary metric: mAP@0.5:0.95
Secondary diagnostics:
- AP50
- AP75
- class-wise AP
- recall/sensitivity
- FP/image
- FP per negative image
```

Ràng buộc:

```text
Các metric này chỉ được tính sau COCO conversion, fixed split creation, model training và prediction generation.
Không dùng test-set metric để chọn checkpoint, tune threshold, chọn model hoặc quyết định augmentation.
```

---

### Evidence đã tạo

Outputs generated:

```text
reports/phase2C_framework_format_decision.md
reports/phase2C_framework_format_decision.json
configs/framework/main_framework.yaml
configs/dataset/coco_paths.yaml
configs/protocol/coco_conversion_policy.yaml
```

Key JSON evidence:

```text
primary_framework: MMDetection
fallback_framework: Detectron2_optional
framework_comparison_matrix: present
framework_selection_conclusion: present
primary_annotation_format: COCO_detection_JSON
source_schema: canonical_detection_schema
canonical_image_rows: 4894
canonical_image_unique_images: 4894
canonical_bbox_rows: 36096
canonical_class_count: 14
abnormal_images: 4394
no_finding_images: 500
no_finding_is_detection_class: false
no_finding_in_coco_categories_planned: false
no_finding_in_coco_annotations_planned: false
no_finding_in_coco_images_planned: true
background_class_planned: false
bbox_source_format: xyxy_original_image
bbox_target_format_phase2D: coco_xywh_absolute
path_root_variable: VINBIGDATA_DICOM_ROOT
near_duplicate_bbox_candidates_retained: 147
actual_coco_conversion_done: false
dicom_loader_validated: false
dataset_training_ready: false
dod_pass_candidate: true
```

Metric readiness evidence:

```text
ap_metrics_computed_in_phase2c: false
primary_metric: mAP@0.5:0.95
secondary_diagnostics: AP50, AP75, class-wise AP, recall/sensitivity, FP/image, FP per negative image
coco_format_preserves_metric_compatibility: true
```

Forbidden actions confirmed:

```text
coco_master_json_created: false
any_coco_json_created: false
train_val_test_split_created: false
labeled_unlabeled_split_created: false
training_started: false
inference_run: false
pseudo_label_generated: false
threshold_tuned: false
test_set_used: false
pixel_array_read: false
image_copied_or_converted: false
bbox_modified_clamped_deleted_or_fused: false
source_annotation_modified: false
phase2b_canonical_schema_modified: false
```

---

### Review GPT

Phase 2C — Framework & Format Decision / COCO Conversion Planning: **PASS**

DoD review:

```text
Script chạy được: PASS
Framework decision: PASS
Framework selection rationale: PASS
Format decision: PASS
Detailed format comparison: PASS
COCO planning: PASS
No Finding policy: PASS
BBox policy: PASS
Category id policy: PASS
Path portability policy: PASS
DICOM loader risk documented: PASS
Metric readiness policy: PASS
Forbidden actions avoided: PASS
No COCO master created: PASS
Dataset training-ready claim avoided: PASS
dod_pass_candidate = true: PASS
```

---

### Quyết định

Phase 2C được khóa với trạng thái:

```text
PASS
```

Quyết định nghiên cứu:

* Framework chính: MMDetection.
* MMDetection được chọn vì phù hợp nhất với COCO-based detection, COCO mAP evaluation, teacher-student SSOD, labeled/unlabeled pipeline và config-driven reproducibility.
* Detectron2 chỉ là fallback optional, cần GPT review lại nếu dùng.
* YOLO-based framework và Custom PyTorch/torchvision không được chọn làm framework chính vì tăng rủi ro custom pipeline/evaluation và lệch khỏi protocol COCO/MMDetection đã khóa.
* Format chính: COCO detection JSON.
* Canonical schema Phase 2B là source of truth.
* COCO conversion thật thuộc Phase 2D, không làm trong Phase 2C.
* COCO sau này phải chứa toàn bộ 4,894 images.
* COCO annotations chỉ chứa 36,096 abnormal bboxes.
* COCO categories chỉ chứa 14 abnormal detection classes.
* No Finding là ảnh âm tính không bbox, không phải detection class.
* No Finding phải nằm trong images, không nằm trong annotations/categories.
* Không tạo background class.
* BBox conversion Phase 2D: xyxy_original_image → coco_xywh_absolute.
* COCO category_id dùng contiguous integer 1..14.
* Path downstream dùng VINBIGDATA_DICOM_ROOT + relative_dicom_path.
* Dataset chưa training-ready vì DICOM loader và empty image loading chưa được validate.
* Phase sau cần kiểm tra custom DICOM loader hoặc processed-image protocol trước training.

---

### Vấn đề / rủi ro

* MMDetection stack chưa import được local; đây là expected vì local training framework deferred.
* Remote/GPU environment vẫn cần setup riêng.
* COCO annotation format chưa đảm bảo MMDetection đọc được DICOM pixel.
* Empty image loading chưa được kiểm tra thật.
* Nếu `filter_empty_gt` cấu hình sai, 500 ảnh No Finding có thể bị lọc khỏi dataloader.
* Metrics AP50/AP75/class-wise AP chưa được tính và không được tính ở Phase 2C.
* Metric evaluation chỉ được thực hiện sau COCO conversion, fixed split, training và prediction generation.
* Test set vẫn bị cấm dùng cho checkpoint/threshold/model/augmentation decisions.

---

### Ràng buộc tuân thủ

Trong Phase 2C đã tuân thủ:

```text
Không tạo COCO master thật.
Không tạo bất kỳ COCO JSON thật nào.
Không split train/val/test.
Không tạo labeled/unlabeled split.
Không train.
Không inference.
Không pseudo-label.
Không tune threshold.
Không dùng test set.
Không đọc pixel_array.
Không copy ảnh.
Không convert ảnh.
Không sửa annotation gốc.
Không sửa canonical schema.
Không xóa bbox.
Không clamp bbox.
Không fuse near-duplicate bbox.
Không xóa/sửa 147 near-duplicate bbox candidates.
Không claim dataset training-ready.
Không tính AP metrics.
```

---

### Trạng thái checklist

Được tick:

* Phase 2C — Framework & Format Decision / COCO Conversion Planning
* Framework comparison: MMDetection / Detectron2 / YOLO-based / Custom PyTorch-torchvision
* Framework selection rationale
* Format comparison: COCO / YOLO / VOC / JSONL-custom
* Primary framework decision: MMDetection
* Primary format decision: COCO detection JSON
* COCO conversion planning
* No Finding / empty image policy
* BBox conversion policy
* Category id policy
* Path portability policy
* DICOM loader risk
* Metric readiness policy
* reports/phase2C_framework_format_decision.md
* reports/phase2C_framework_format_decision.json
* configs/framework/main_framework.yaml
* configs/dataset/coco_paths.yaml
* configs/protocol/coco_conversion_policy.yaml
* forbidden actions avoided
* GPT review PASS

---

### Quyết định tiếp theo

Phase tiếp theo:

```text
Phase 2D — COCO Master Conversion & Validation
```

Chưa được làm trước khi mở Phase 2D:

```text
Train/val/test split
Labeled/unlabeled split
Training
Inference
Pseudo-labeling
Threshold tuning
Test-set usage
Framework dataloader validation
Empty image loading check
Pixel array reading
Image copy/convert
```

Phase 2D chỉ được mở sau khi Phase 2C evidence đã commit và push GitHub.
