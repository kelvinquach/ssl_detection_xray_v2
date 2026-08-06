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

## 2026-07-14 — PHASE 2D: COCO Master Conversion & Validation

### Mục tiêu

Chuyển canonical detection schema đã khóa ở Phase 2B thành COCO detection JSON chính thức cho controlled working scope của đề tài:

**“Nghiên cứu học bán giám sát cho dò tìm bất thường trên X-quang phổi.”**

Mục tiêu Phase 2D:

```text
Tạo COCO master từ canonical image table, bbox table và class mapping.
Giữ toàn bộ 4,894 images trong controlled scope.
Giữ toàn bộ 36,096 abnormal bbox annotations.
Chỉ tạo 14 abnormal detection categories.
Chuyển bbox từ xyxy_original_image sang coco_xywh_absolute.
Giữ 500 ảnh No Finding trong images nhưng không có annotation.
Không tạo No Finding hoặc background thành detection category.
Validate COCO structure, geometry, relationships và traceability.
Không đọc DICOM, header hoặc pixel array.
Không chạy framework dataloader, training, inference hoặc pseudo-labeling.
```

---

### Đã làm

#### 1. Tạo script Phase 2D

Đã tạo:

```text
scripts/02D_build_coco_master.py
```

Script đọc các input canonical từ Phase 2B:

```text
data/processed/canonical/canonical_image_table.csv
data/processed/canonical/canonical_bbox_table.csv
data/processed/canonical/canonical_class_mapping.csv
reports/phase2B_canonical_schema_validation.json
```

Script không đọc file DICOM, không đọc DICOM header, không đọc pixel array và không import `pydicom`, `cv2` hoặc `PIL`.

#### 2. Tạo protocol YAML Phase 2D

Đã tạo:

```text
configs/protocol/phase2D_coco_master_validation.yaml
```

Protocol khóa:

```text
Expected images: 4,894
Expected annotations: 36,096
Expected categories: 14
Expected abnormal images: 4,394
Expected No Finding images: 500
BBox source format: xyxy_original_image
BBox target format: coco_xywh_absolute
Category IDs: contiguous 1..14
No Finding category: forbidden
Background category: forbidden
Path root variable: VINBIGDATA_DICOM_ROOT
```

Sau maintenance patch, YAML được strict-load và được đối chiếu với Phase 2B validation cùng canonical tables.

Nếu có protocol drift, script hard-fail và không thay thế output COCO hiện có.

#### 3. Tạo COCO master

Đã tạo:

```text
data/processed/coco/coco_master.json
```

COCO structure:

```text
images[]
annotations[]
categories[]
```

Count:

```text
images: 4,894
annotations: 36,096
categories: 14
abnormal images: 4,394
No Finding images: 500
```

#### 4. Chính sách COCO images

`images[]` chứa toàn bộ 4,894 images trong controlled scope.

Các field chính:

```text
id
file_name
width
height
original_image_id
canonical_image_id
scope_label
is_negative
```

Path policy:

```text
file_name lấy từ relative_dicom_path.
Không dùng absolute local path.
Path separator được chuẩn hóa thành "/".
absolute_path_count = 0.
```

#### 5. Chính sách COCO annotations

`annotations[]` chỉ chứa abnormal bbox.

Kết quả:

```text
COCO annotation rows: 36,096
Canonical bbox rows: 36,096
No Finding annotations: 0
```

BBox conversion:

```text
x = x_min
y = y_min
width = x_max - x_min
height = y_max - y_min
area = width * height
iscrowd = 0
```

Không thực hiện:

```text
Không clamp bbox.
Không delete bbox.
Không fuse bbox.
Không NMS.
Không rounding bbox.
```

#### 6. Chính sách COCO categories

`categories[]` chỉ chứa 14 abnormal detection classes.

Category IDs:

```text
1..14
```

Category metadata giữ:

```text
canonical_class_id
class_id_original
supercategory = chest_abnormality
```

Kết quả:

```text
No Finding category: absent
Normal category: absent
Background category: absent
Category ID 0: absent
```

#### 7. Audit No Finding

No Finding được xác định từ canonical image metadata, không suy ra đơn thuần từ zero annotation.

Kết quả:

```text
No Finding images: 500
No Finding images with annotations: 0
No Finding category present: false
Negative images lost: 0
```

500 ảnh No Finding vẫn nằm trong `images[]`, nhưng không có record trong `annotations[]`.

#### 8. Validation bbox và relationship

Kết quả:

```text
Invalid bbox count: 0
Boundary violation count: 0
Area mismatch count: 0
iscrowd violation count: 0
Broken image reference count: 0
Broken category reference count: 0
Absolute path count: 0
```

Validation rule:

```text
x >= 0
y >= 0
width > 0
height > 0
x + width <= image_width
y + height <= image_height
area == width * height
```

#### 9. Traceability và one-to-one preservation

Mỗi annotation giữ các field:

```text
canonical_ann_id
source_row_id
original_image_id
rad_id
canonical_class_id
class_id_original
```

Kết quả:

```text
Coordinate mismatch count: 0
Image mapping mismatch count: 0
Category mapping mismatch count: 0
Missing canonical annotations: 0
Duplicated canonical annotations: 0
Extra COCO annotations: 0
canonical_ann_id sets equal: true
```

Điều này chứng minh toàn bộ 36,096 canonical bbox được giữ one-to-one trong COCO.

147 near-duplicate bbox candidates không bị xóa hoặc fuse.

Phase 2D không chạy lại near-duplicate detection vì đây không phải mục tiêu của phase.

#### 10. Strict protocol guardrail

Maintenance patch đã bổ sung strict YAML validation:

```text
protocol strict load: PASS
protocol / Phase 2B drift: 0
```

Guardrail kiểm tra:

```text
YAML file phải tồn tại.
YAML phải parse được.
Required sections và keys phải đầy đủ.
Count phải là integer không âm.
Tolerance phải finite và không âm.
YAML expected counts phải khớp Phase 2B validation.
YAML expected counts phải khớp canonical tables.
```

Không còn silent fallback về empty dictionary.

#### 11. Atomic output promotion

Maintenance patch đã sửa thứ tự output:

```text
Build COCO in memory.
Chạy toàn bộ internal validation.
Kiểm tra per-image annotation count.
Ghi temporary JSON.
Parse temporary JSON.
Chạy pycocotools trên temporary JSON.
Chỉ atomic replace final COCO khi toàn bộ hard checks PASS.
```

Kết quả:

```text
pre-promotion checks: PASS
atomic promotion: PASS
```

Nếu run thất bại, output COCO hợp lệ từ lần trước được giữ nguyên.

#### 12. Unit tests guardrail

Đã tạo:

```text
tests/test_phase2D_guardrails.py
```

Lệnh chạy:

```cmd
python -m unittest discover -s tests -p "test_phase2D_guardrails.py" -v
```

Kết quả:

```text
Ran 22 tests
OK
```

Các test bao gồm:

```text
Strict YAML loading.
Missing/malformed YAML.
Missing required sections/keys.
Invalid count/tolerance.
Protocol drift detection.
Output preservation on failure.
Successful atomic promotion.
Temporary-file cleanup.
```

---

### Evidence đã tạo

Outputs:

```text
data/processed/coco/coco_master.json
reports/phase2D_coco_master_validation.json
reports/phase2D_coco_master_validation.md
reports/phase2D_coco_image_annotation_counts.csv
reports/phase2D_coco_category_summary.csv
reports/phase2D_coco_invalid_annotations.csv
reports/phase2D_coco_no_finding_audit.csv
configs/protocol/phase2D_coco_master_validation.yaml
tests/test_phase2D_guardrails.py
```

Lệnh chạy chính:

```cmd
python scripts\02D_build_coco_master.py
```

Console result:

```text
images                     : 4894
annotations                : 36096
categories                 : 14
abnormal images            : 4394
No Finding images          : 500
invalid annotations        : 0
No Finding annotations     : 0
absolute paths             : 0
pycocotools                : load PASS
protocol strict load       : PASS
protocol / Phase 2B drift  : 0
pre-promotion checks       : PASS
atomic promotion           : PASS
hard errors                : 0
warnings                   : 0
dataset_training_ready     : False
dod_pass_candidate         : True
```

JSON syntax validation:

```cmd
python -m json.tool data\processed\coco\coco_master.json > NUL
```

Result:

```text
JSON_PARSE_PASS
```

pycocotools validation:

```cmd
python -c "from pycocotools.coco import COCO; c=COCO(r'data\processed\coco\coco_master.json'); print('images=',len(c.imgs)); print('annotations=',len(c.anns)); print('categories=',len(c.cats))"
```

Result:

```text
images=4894
annotations=36096
categories=14
```

Dependency check:

```cmd
python -m pip check
```

Result:

```text
No broken requirements found.
```

Environment:

```text
Conda environment: ssl
Python executable: C:\Users\USER\anaconda3\envs\ssl\python.exe
pycocotools: 2.0.11
```

---

### Review GPT

Phase 2D — COCO Master Conversion & Validation: **PASS**

DoD review:

```text
Canonical schema converted to COCO master: PASS
COCO images contain all 4,894 scope images: PASS
COCO annotations contain 36,096 abnormal bbox only: PASS
COCO categories contain 14 abnormal classes only: PASS
BBox format xywh: PASS
Area calculation: PASS
No Finding retained in images: PASS
No Finding annotations = 0: PASS
No Finding excluded from categories: PASS
Background class excluded: PASS
Negative image count = 500: PASS
Annotation count matches canonical bbox table: PASS
Image IDs unique and contiguous: PASS
Annotation IDs unique and contiguous: PASS
Category IDs contiguous 1..14: PASS
Category ID 0 absent: PASS
Invalid annotations = 0: PASS
Boundary violations = 0: PASS
Area mismatches = 0: PASS
Broken references = 0: PASS
Absolute paths = 0: PASS
Traceability preservation: PASS
One-to-one annotation preservation: PASS
JSON parse: PASS
pycocotools load: PASS
Strict protocol loading: PASS
Protocol drift count = 0: PASS
Pre-promotion checks: PASS
Atomic output promotion: PASS
Guardrail tests 22/22: PASS
Warnings = 0: PASS
Hard errors = 0: PASS
Forbidden actions avoided: PASS
```

---

### Quyết định

Phase 2D được khóa với trạng thái:

```text
PASS
```

Quyết định nghiên cứu:

* `coco_master.json` là COCO master chính thức cho controlled working scope.
* COCO chứa toàn bộ 4,894 images.
* COCO giữ toàn bộ 36,096 abnormal bbox annotations.
* COCO categories chỉ gồm 14 abnormal detection classes.
* No Finding là ảnh âm tính trong `images[]`, không có annotation và không phải category.
* Không tạo background class.
* BBox được chuyển từ `xyxy_original_image` sang `coco_xywh_absolute`.
* Area được tính bằng `width * height`.
* Không bbox nào bị clamp, delete, fuse, NMS hoặc rounding.
* Traceability từ canonical schema được bảo toàn.
* YAML protocol được strict-load và cross-check với Phase 2B.
* Final COCO chỉ được atomic promote sau khi toàn bộ validation PASS.
* COCO JSON hợp lệ chưa làm dataset training-ready.
* Dataset vẫn có trạng thái `dataset_training_ready = false`.

---

### Vấn đề / rủi ro còn lại

* DICOM loader chưa được triển khai hoặc validate trong MMDetection.
* MMDetection default `LoadImageFromFile` không được giả định là đọc được `.dicom`.
* Pixel decoding chưa được kiểm tra trong Phase 2D.
* Empty-image loading chưa được kiểm tra thật bằng framework dataloader.
* `filter_empty_gt=False` hoặc cơ chế tương đương chưa được validate.
* COCO master chưa được chia train/val/test.
* Labeled/unlabeled subsets chưa được tạo.
* Dataset chưa được phép dùng để train detector.

---

### Ràng buộc tuân thủ

Trong Phase 2D đã tuân thủ:

```text
Không đọc DICOM file.
Không kiểm tra DICOM file existence.
Không đọc DICOM header.
Không đọc pixel_array.
Không import pydicom, cv2 hoặc PIL.
Không copy hoặc convert image.
Không tạo train/val/test split.
Không tạo labeled/unlabeled split.
Không load MMDetection/Detectron2 dataset.
Không kiểm tra filter_empty_gt.
Không train.
Không inference.
Không pseudo-label.
Không tune threshold.
Không dùng test set.
Không tính AP/mAP.
Không sửa canonical schema.
Không sửa source annotation.
Không clamp bbox.
Không delete bbox.
Không fuse bbox.
Không NMS.
Không claim dataset training-ready.
```

---

### Trạng thái checklist

Được tick:

* Phase 2D — COCO Master Conversion & Validation
* Convert canonical schema sang COCO master
* COCO images chứa toàn bộ ảnh trong scope
* COCO annotations chỉ chứa abnormal bbox
* COCO categories chỉ chứa 14 abnormal classes
* BBox format `[x, y, width, height]`
* Area calculation
* No Finding retained in images
* No Finding zero annotations
* No Finding excluded from categories
* Background class excluded
* Negative image count matches controlled scope
* Annotation count matches canonical bbox table
* COCO validator pass
* JSON parse pass
* pycocotools load pass
* Traceability preservation
* One-to-one annotation preservation
* Strict protocol YAML validation
* Protocol drift detection
* Atomic output promotion
* Guardrail unit tests
* Forbidden actions avoided
* GPT review PASS

Checklist output path được chuẩn hóa thành:

```text
reports/phase2D_coco_master_validation.json
```

Tên cũ:

```text
reports/coco_validation_report.json
```

không còn được dùng.

---

### Quyết định tiếp theo

Phase 2D đã được commit và push thành công:

```text
Commit: 1a3f7a7
Branch: main
Remote: origin/main
Status: CLOSED / PASS
```

Phase tiếp theo:

```text
Current subphase:
Phase 2D.1A — Image Representation Protocol Decision

Overall phase:
Phase 2D.1 — JPG Training Representation &
MMDetection Empty-Image Loading Validation
```

Quyết định chuyển phase:

```text
DICOM không còn được dự kiến dùng trực tiếp làm training image representation.

DICOM tiếp tục là immutable raw medical source và source evidence.

JPG chất lượng cao được chọn làm processed training image representation
cho MMDetection.

coco_master.json tiếp tục là annotation master chính thức.

coco_master_jpg.json sẽ được tạo ở Phase 2D.1B như một training derivative,
chỉ thay đổi image file_name/path representation và không thay đổi
annotation semantics.
```

Trạng thái mở phase:

```text
Phase 2D.1A: OPEN / CURRENT
Phase 2D.1B: LOCKED until Phase 2D.1A GPT review PASS
Phase 2D.1C: LOCKED until Phase 2D.1B GPT review PASS
Phase 2D.1D: LOCKED until Phase 2D.1C PASS

jpg_training_representation_ready: false
coco_jpg_training_annotation_ready: false
mmdetection_dataset_loading_ready: false
empty_image_retention_ready: false
dataset_training_ready: false
training_authorized: false
```

Chưa được làm:

```text
Không chạy full DICOM-to-JPG conversion.
Không tạo train/val/test split.
Không tạo labeled/unlabeled split.
Không train detector.
Không inference.
Không pseudo-label.
Không tune threshold.
Không tính AP/mAP.
Không dùng test set.
```

---

## 2026-07-15 — PHASE 2D.1 Planning: JPG Training Representation Redesign

### Mục tiêu

Thiết kế lại Phase 2D.1 sau khi quyết định sử dụng JPG chất lượng cao làm training image representation cho MMDetection.

Tên phase được khóa lại thành:

```text
Phase 2D.1 — JPG Training Representation &
MMDetection Empty-Image Loading Validation
```

Phase 2D.1 không chỉ kiểm tra empty-image loading mà phải chứng minh hai claim:

```text
1. JPG training representation được tạo đúng, có kiểm soát và có thể
   truy vết từ DICOM gốc.

2. MMDetection đọc được JPG + COCO-JPG và không làm mất 500 ảnh
   No Finding có zero annotations.
```

---

### Quyết định thiết kế phase

Phase 2D.1 được chia thành bốn tiểu giai đoạn:

```text
Phase 2D.1A — Image Representation Protocol Decision
Environment: Local

Phase 2D.1B — DICOM-to-JPG Conversion & Validation
Environment: Local

Phase 2D.1C — MMDetection Dataset / Empty-Image Loading Validation
Environment: Google Colab

Phase 2D.1D — Evidence Consolidation, GPT Review & Closure
Environment: Local
```

Phase 2D.1B có hai gate nội bộ:

```text
2D.1B-Pilot — Representative DICOM-to-JPG pilot

2D.1B-Full — Full controlled-scope conversion,
chỉ được chạy sau khi pilot và final JPEG quality decision PASS.
```

---

### Representation policy

Vai trò của các artifact được khóa như sau:

```text
DICOM:
Immutable raw medical source và source evidence.

JPG:
Processed training image representation được tạo bằng một protocol
cố định, có version và có thể tái lập.

coco_master.json:
Official annotation master gắn với representation DICOM gốc.

coco_master_jpg.json:
Training derivative gắn với các file JPG.

MMDetection:
Framework downstream dùng JPG + COCO-JPG để load dataset,
train detector, evaluate và triển khai SSOD ở các phase sau.
```

DICOM gốc không bị thay thế, chỉnh sửa hoặc xóa.

---

### Phase 2D.1A — Protocol requirements

Phase 2D.1A phải khóa toàn bộ DICOM-to-JPG transformation:

```text
DICOM pixel decoding policy.
RescaleSlope / RescaleIntercept policy.
Modality LUT policy.
VOI LUT hoặc windowing policy.
MONOCHROME1 inversion policy.
Intensity clipping policy.
uint8 [0,255] conversion policy.
Output channel policy.
JPEG quality candidates.
Final JPEG quality selection rule.
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

Preliminary direction:

```text
Không resize trong bước conversion.
Không crop.
Không rotate.
Giữ nguyên width và height gốc.
Không scale bbox nếu dimensions và orientation được chứng minh không đổi.
Đánh giá JPEG quality 95 và 100 trong pilot.
Chỉ khóa một final JPEG quality sau khi pilot được review.
```

Không được mặc định chọn `quality=95` hoặc `quality=100` trước khi có pilot evidence.

---

### DICOM intensity transformation guardrail

Việc chuyển DICOM sang JPG không được thực hiện đơn giản bằng:

```text
pixel_array
→ min-max normalize không kiểm soát
→ uint8
→ JPG
```

Protocol phải xem xét và ghi nhận:

```text
RescaleSlope
RescaleIntercept
modality LUT
VOI LUT
WindowCenter
WindowWidth
PhotometricInterpretation
MONOCHROME1 inversion
intensity clipping
uint8 quantization
JPEG compression
```

Fidelity evaluation phải phân biệt:

```text
1. Sai khác do DICOM windowing / normalization / uint8 quantization.

2. Sai khác riêng do JPEG encoding.
```

Đánh giá ảnh hưởng JPEG phải so sánh:

```text
pre-JPEG uint8 image
vs
decoded JPG image
```

không được diễn giải toàn bộ sai khác giữa raw DICOM 12/16-bit và JPG 8-bit là JPEG compression error.

---

### Phase 2D.1B planning

Target processed images:

```text
data/processed/images_jpg/train/<image_id>.jpg
```

Target training COCO derivative:

```text
data/processed/coco/coco_master_jpg.json
```

Target traceability mapping:

```text
data/processed/image_mapping/dicom_to_jpg_mapping.csv
```

`coco_master_jpg.json` chỉ được thay đổi representation path:

```text
train/<image_id>.dicom
→
train/<image_id>.jpg
```

Các field sau phải được giữ nguyên so với `coco_master.json`:

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

Full validation targets:

```text
JPG files: 4,894
Missing JPG: 0
Duplicate image IDs: 0
Decode errors: 0
Width/height mismatches: 0
Orientation changes: 0

COCO-JPG images: 4,894
COCO-JPG annotations: 36,096
COCO-JPG categories: 14
Abnormal images: 4,394
No Finding images: 500
No Finding annotations: 0

BBox mismatches: 0
Area mismatches: 0
Category mismatches: 0
Traceability mismatches: 0
Boundary violations: 0
```

4,894 JPG files không được commit vào ordinary Git. Chúng sẽ được quản lý bằng local storage, Google Drive hoặc storage mechanism riêng.

---

### Phase 2D.1C planning

Phase 2D.1C chạy trên Google Colab sau khi Phase 2D.1B PASS.

Inputs:

```text
JPG training representation
coco_master_jpg.json
MMDetection dataset config
MMDetection validation script
```

Environment evidence phải ghi:

```text
Python version
PyTorch version
CUDA version
MMEngine version
MMCV version
MMDetection version
pycocotools version
pip freeze
```

Validation targets:

```text
MMDetection import: PASS
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
No Finding sample loading: PASS
Dataloader smoke test: PASS
All image IDs seen: 4,894
```

No Finding sample expectation:

```text
gt_instances.bboxes shape = (0, 4)
gt_instances.labels shape = (0,)
sample remains present in the dataset
```

Failure rule:

```text
Nếu dataset length = 4,394 thì Phase 2D.1C FAIL,
vì 500 No Finding images đã bị framework lọc mất.
```

---

### Current phase gate

```text
Phase 2D: CLOSED / PASS

Phase 2D.1: IN PROGRESS

Phase 2D.1A:
OPEN / CURRENT

Phase 2D.1B:
LOCKED until Phase 2D.1A GPT review PASS

Phase 2D.1C:
LOCKED until Phase 2D.1B GPT review PASS

Phase 2D.1D:
LOCKED until Phase 2D.1C PASS
```

Readiness flags:

```text
jpg_training_representation_ready: false
coco_jpg_training_annotation_ready: false
mmdetection_dataset_loading_ready: false
empty_image_retention_ready: false
dataset_training_ready: false
training_authorized: false
```

---

### Ràng buộc hiện tại

```text
Không full-convert 4,894 DICOM trước khi Phase 2D.1A PASS.

Không chạy Phase 2D.1B-Full trước khi pilot và final JPEG quality
decision PASS.

Không mở Phase 2D.1C trước khi JPG và COCO-JPG validation PASS.

Không tạo train/val/test split.

Không tạo labeled/unlabeled split.

Không train supervised detector.

Không train SSL detector.

Không inference.

Không pseudo-label.

Không tune threshold.

Không tính AP/mAP.

Không dùng test set.

Không sửa canonical bbox.

Không sửa annotation semantics trong coco_master.json.

Không claim dataset training-ready.
```

---

### Evidence hiện tại

Phase 2D.1 mới ở mức planning/protocol redesign.

Chưa có:

```text
DICOM-to-JPG conversion script.
JPG pilot output.
Full JPG dataset.
coco_master_jpg.json.
MMDetection dataset loading evidence.
Empty-image retention evidence.
Phase 2D.1 PASS evidence.
```

Do đó chưa được tick:

```text
Phase 2D.1A PASS
Phase 2D.1B PASS
Phase 2D.1C PASS
Phase 2D.1D PASS
Phase 2D.1 CLOSED / PASS
```

---

### Quyết định tiếp theo

```text
Thiết kế và review Phase 2D.1A —
Image Representation Protocol Decision.
```

---

## 2026-07-15 — PHASE 2D.1A: Image Representation Protocol Decision

### Mục tiêu

Khóa protocol kỹ thuật để chuyển nguồn DICOM y khoa gốc thành JPG training representation cho MMDetection trước khi đọc pixel hoặc thực hiện chuyển đổi ảnh thật.

Phase 2D.1A là decision-only phase. Mục tiêu của phase này là:

```text
Khóa vai trò của DICOM, JPG, coco_master.json và coco_master_jpg.json.

Khóa thứ tự DICOM pixel transformation.

Khóa Modality LUT / RescaleSlope / RescaleIntercept policy.

Khóa VOI LUT / WindowCenter / WindowWidth policy.

Khóa PhotometricInterpretation, PresentationLUTShape
và MONOCHROME1 inversion policy.

Khóa intensity clipping và uint8 conversion policy.

Khóa output-channel policy.

Khóa no-resize, no-crop, no-rotation, no-flip
và no-transpose policy.

Khóa JPG filename, COCO-JPG path và traceability policy.

Định nghĩa pilot so sánh JPEG quality 95 và 100.

Định nghĩa fidelity metrics và visual-audit requirements.

Không chọn final JPEG quality trước pilot evidence.
```

Phase này không đọc DICOM pixel array, không tạo JPG và không chạy full conversion.

---

### Đã làm

#### 1. Tạo script protocol Phase 2D.1A

Đã tạo:

```text
scripts/02D1A_image_representation_protocol.py
```

Script dùng một protocol specification có kiểm soát để sinh đồng bộ YAML, Markdown và JSON evidence.

Script không import hoặc sử dụng:

```text
pydicom
PIL
cv2
pixel_array
DICOM decoding
image encoding
```

Do đó không có ảnh nào được đọc, giải mã hoặc chuyển đổi trong Phase 2D.1A.

#### 2. Tạo protocol YAML

Đã tạo:

```text
configs/protocol/phase2D1_jpg_representation.yaml
```

Protocol metadata:

```text
phase_id: 2D.1A
protocol_version: 1.0.0
protocol_status: decision_locked_pilot_pending
final_jpeg_quality: null
```

JPEG quality candidates:

```text
95
100
```

Final JPEG quality vẫn chưa được lựa chọn.

#### 3. Tạo decision reports

Đã tạo:

```text
reports/phase2D1_image_representation_decision.md
reports/phase2D1_image_representation_decision.json
```

Các report khóa đầy đủ:

```text
Artifact roles
DICOM decoding policy
Modality transformation policy
VOI/windowing policy
Pixel-padding policy
Presentation-polarity policy
uint8 conversion policy
Output-channel policy
JPEG candidate policy
Geometry-preservation policy
Path policy
Traceability policy
Pilot-selection protocol
Fidelity-validation protocol
Forbidden actions
Readiness flags
Definition of Done
```

#### 4. Tạo guardrail tests

Đã tạo:

```text
tests/test_phase2D1A_protocol_guardrails.py
```

Các test kiểm tra:

```text
Output files tồn tại và parse được.
YAML strict-load thành công.
JSON parse thành công.
Locked counts khớp evidence Phase 2D.
Protocol specification là single source of truth.
Cross-output drift bằng 0.
Final JPEG quality vẫn là null.
JPEG quality candidates đúng [95, 100].
Không khóa numeric fidelity thresholds.
Không cho phép direct per-image min-max.
Không cho phép automatic percentile clipping.
Không có geometry transformation.
BBox scaling chưa được đánh dấu là validated.
Tất cả readiness flags vẫn false.
Tất cả forbidden actions vẫn false.
Không có banned imports hoặc pixel decoding.
Atomic output preservation hoạt động khi failure.
Atomic promotion hoạt động khi success.
```

---

### Protocol đã khóa

#### Artifact roles

```text
DICOM:
Immutable raw medical source.

JPG:
Processed training image representation.

coco_master.json:
Official annotation master.

coco_master_jpg.json:
Path-only training derivative linked to JPG representation.
```

#### Authoritative pixel-transformation order

```text
DICOM decode
→ pixel-padding mask
→ modality transformation
→ VOI LUT/windowing
→ presentation-polarity normalization
→ deterministic uint8 conversion
→ one-channel JPG storage
→ JPEG encoding
```

Thứ tự này không được tự ý thay đổi trong Phase 2D.1B.

#### DICOM decoding policy

```text
force_read: false
single_frame_only: true
SamplesPerPixel: 1
Allowed PhotometricInterpretation:
- MONOCHROME1
- MONOCHROME2

Unsupported input:
hard fail
```

#### Modality transformation policy

```text
Nếu có Modality LUT Sequence:
    apply Modality LUT

Ngược lại, nếu có đầy đủ RescaleSlope và RescaleIntercept:
    apply rescale

Ngược lại:
    identity transformation
```

Guardrail:

```text
Không áp dụng Modality LUT và rescale tuần tự trên cùng ảnh.

Nếu chỉ tồn tại RescaleSlope hoặc chỉ tồn tại RescaleIntercept:
hard fail.

Metadata modality xung đột hoặc mơ hồ:
hard fail.
```

#### VOI LUT / Windowing policy

```text
Nếu có VOI LUT Sequence:
    ưu tiên VOI LUT

Ngược lại, nếu có WindowCenter và WindowWidth hợp lệ:
    dùng windowing

Ngược lại:
    dùng theoretical modality-domain range fallback
```

Các hành động bị cấm:

```text
Observed per-image min-max normalization
Automatic percentile clipping
```

Không được dùng trực tiếp:

```python
arr.min()
arr.max()
```

để thiết lập range riêng cho từng ảnh.

#### Presentation polarity policy

```text
Nếu PresentationLUTShape == INVERSE:
    invert đúng một lần

Ngược lại, nếu PresentationLUTShape không tồn tại
và PhotometricInterpretation == MONOCHROME1:
    invert đúng một lần

Ngược lại:
    không invert
```

Output target:

```text
MONOCHROME2-equivalent polarity
low value = dark
high value = bright
```

#### uint8 conversion policy

```text
clip bằng theoretical output bounds
→ linear map sang [0,255]
→ numpy.rint
→ final clip [0,255]
→ cast uint8
```

NaN hoặc Inf phải hard fail.

#### Channel policy

```text
JPG storage:
mode L
one grayscale channel
uint8

MMDetection input:
three identical channels created by grayscale replication
during framework loading
```

Việc MMDetection thực sự tạo đúng ba channel được trì hoãn tới Phase 2D.1C.

#### Geometry and bbox policy

```text
resize: false
crop: false
rotation: false
flip: false
transpose: false

preserve width: true
preserve height: true

bbox_scaling_expected: false
bbox_scaling_validated: false
```

Nếu dimensions hoặc orientation thay đổi:

```text
hard fail
```

Không được tự động scale bbox để che giấu geometry mismatch.

#### Path policy

```text
JPG root:
data/processed/images_jpg

JPG relative path:
train/<image_id>.jpg

COCO-JPG file_name:
train/<image_id>.jpg

Absolute COCO-JPG path:
forbidden
```

#### Traceability policy

Mapping dự kiến:

```text
data/processed/image_mapping/dicom_to_jpg_mapping.csv
```

Các hash bắt buộc trong phase chuyển đổi sau:

```text
source_dicom_sha256
pre_jpeg_uint8_sha256
output_jpg_sha256
protocol_version
protocol_sha256
```

---

### Pilot protocol đã khóa

Pilot Phase 2D.1B phải dùng:

```text
Selection strategy: deterministic_coverage_first
Selection unit: image_id
Minimum images: 64
Minimum No Finding images: 16
Tie-break seed: 2026
```

Pilot phải bao phủ:

```text
Tất cả 14 abnormal classes
No Finding images
Minimum/maximum dimensions
Minimum/maximum pixel counts
Smallest/largest bbox
PhotometricInterpretation patterns
Transfer Syntax patterns
BitsStored / PixelRepresentation patterns
Rescale patterns
Modality LUT presence/absence
VOI LUT presence/absence
WindowCenter/WindowWidth presence/absence
Single/multi-valued windows
PresentationLUTShape patterns
PixelPaddingValue presence/absence
```

Nếu 64 ảnh không đủ bao phủ toàn bộ metadata strata đã quan sát, pilot phải mở rộng cho tới khi coverage đầy đủ.

---

### Fidelity protocol đã khóa

JPEG fidelity phải được đánh giá bằng:

```text
pre-JPEG uint8 image
versus
decoded JPG image
```

Không được mô tả toàn bộ sai khác giữa raw DICOM và JPG là JPEG compression error.

Whole-image metrics:

```text
MAE
RMSE
PSNR
SSIM
maximum absolute error
p95 absolute error
p99 absolute error
file size
compression ratio
```

BBox-ROI metrics:

```text
ROI MAE
ROI PSNR
ROI SSIM
ROI maximum absolute error
```

Visual evidence:

```text
Full-image visual audit
BBox-crop visual audit
Difference heatmap
```

Phase 2D.1A không khóa ngưỡng định lượng cho PSNR, SSIM hoặc MAE.

---

### Evidence đã tạo và lệnh đã chạy

Lệnh chạy protocol:

```cmd
python scripts\02D1A_image_representation_protocol.py
```

Console result:

```text
Phase: 2D.1A
Status: OPEN_REVIEW_REQUIRED
Protocol version: 1.0.0
Locked images: 4894
Locked annotations: 36096
Locked categories: 14
JPEG candidates: [95, 100]
Final JPEG quality: PENDING PILOT
Direct min-max allowed: False
Resize/crop/rotation: False / False / False
Full conversion run: False
COCO-JPG created: False
Dataset training-ready: False
Training authorized: False
Hard errors: 0
Warnings: 0
DoD pass candidate: True
GPT review status: PENDING
```

Lệnh chạy tests:

```cmd
python -m unittest discover -s tests -p "test_phase2D1A_protocol_guardrails.py" -v
```

Kết quả:

```text
Ran 31 tests
OK
```

JSON validation:

```cmd
python -m json.tool reports\phase2D1_image_representation_decision.json > NUL
```

Kết quả:

```text
JSON_PARSE_PASS
```

YAML validation:

```cmd
python -c "import yaml; p=yaml.safe_load(open(r'configs\protocol\phase2D1_jpg_representation.yaml', encoding='utf-8')); print('phase=',p['protocol_metadata']['phase_id']); print('candidates=',p['jpeg_encoding']['quality_candidates']); print('final_quality=',p['jpeg_encoding']['final_quality'])"
```

Kết quả:

```text
phase= 2D.1A
candidates= [95, 100]
final_quality= None
```

---

### Review GPT

Phase 2D.1A — Image Representation Protocol Decision: **PASS**

DoD review:

```text
Script execution: PASS
Locked image count = 4,894: PASS
Locked annotation count = 36,096: PASS
Locked category count = 14: PASS
Required protocol items documented = 20/20: PASS
JPEG candidates = [95, 100]: PASS
Final JPEG quality remains null: PASS
No numeric fidelity threshold locked: PASS
Direct per-image min-max forbidden: PASS
Automatic percentile clipping forbidden: PASS
No geometry transforms: PASS
BBox scaling not falsely validated: PASS
Single-source protocol generation: PASS
Cross-output drift count = 0: PASS
Atomic output preservation: PASS
Atomic output promotion: PASS
Guardrail tests = 31/31: PASS
JSON parse: PASS
YAML strict-load: PASS
Hard errors = 0: PASS
Warnings = 0: PASS
Forbidden actions avoided: PASS
```

---

### Quyết định

Phase 2D.1A được khóa với trạng thái:

```text
CLOSED / PASS
```

Quyết định nghiên cứu:

```text
Protocol version 1.0.0 được chấp nhận làm protocol chính thức
cho Phase 2D.1B-Pilot.

JPEG quality 95 và 100 là hai ứng viên pilot.

Final JPEG quality chưa được lựa chọn.

Không được chạy full conversion trước pilot evidence và GPT review.

No-resize, no-crop, no-rotation, no-flip và no-transpose
tiếp tục là hard geometry guardrails.

BBox scaling không được kỳ vọng, nhưng bbox invariance
phải được kiểm chứng bằng evidence thật trong pilot.

DICOM vẫn là immutable raw source.

JPG chưa được xem là training-ready representation.

coco_master_jpg.json chưa được tạo.

MMDetection loading chưa được validate.
```

---

### Vấn đề / rủi ro còn lại

```text
Chưa đọc pixel array DICOM theo protocol 1.0.0.

Chưa xác định metadata strata thực tế ở mức pixel-decoding pilot.

Chưa tạo pilot JPG quality 95 hoặc 100.

Chưa tính fidelity metrics.

Chưa thực hiện visual audit.

Chưa lựa chọn final JPEG quality.

Chưa xác nhận width/height/orientation preservation bằng ảnh thật.

Chưa xác nhận bbox invariance bằng pilot.

Chưa tạo full JPG dataset.

Chưa tạo coco_master_jpg.json.

Chưa kiểm tra MMDetection giữ đủ 500 No Finding images.
```

---

### Ràng buộc tuân thủ

Trong Phase 2D.1A đã tuân thủ:

```text
Không đọc DICOM pixel array.
Không tạo JPG.
Không chạy pilot conversion.
Không chạy full conversion.
Không tạo coco_master_jpg.json.
Không tạo train/val/test split.
Không tạo labeled/unlabeled split.
Không train.
Không inference.
Không pseudo-label.
Không tune threshold.
Không tính AP/mAP.
Không dùng test set.
Không sửa canonical bbox.
Không sửa coco_master.json.
Không claim JPG representation ready.
Không claim dataset training-ready.
Không authorize training.
```

---

### Trạng thái checklist

Được tick:

```text
Khóa vai trò DICOM và JPG.
Khóa vai trò coco_master.json và coco_master_jpg.json.
Khóa DICOM decoding và modality transformation policy.
Khóa RescaleSlope / RescaleIntercept và Modality LUT policy.
Khóa VOI LUT / windowing và intensity clipping policy.
Khóa PhotometricInterpretation / MONOCHROME1 inversion policy.
Khóa uint8 [0,255] và output-channel policy.
Khóa no-resize/no-crop/no-rotation và bbox-scaling policy.
Định nghĩa pilot JPEG quality 95 và 100.
Khóa filename/path/traceability policy.
Khóa pilot subset và fidelity metrics.
Tạo protocol YAML.
Tạo decision Markdown report.
Tạo decision JSON report.
Tạo guardrail tests.
Forbidden actions avoided.
GPT review PASS.
```

Chưa được tick:

```text
Phase 2D.1B-Pilot PASS.
Final JPEG quality selected.
Phase 2D.1B-Full PASS.
Full JPG dataset created.
coco_master_jpg.json created.
JPG representation ready.
MMDetection loading ready.
Empty-image retention ready.
Phase 2D.1 overall PASS.
Dataset training-ready.
Training authorized.
```

---

### Trạng thái gate sau Phase 2D.1A

```text
Phase 2D.1A:
CLOSED / PASS

Phase 2D.1B:
IN PROGRESS

Phase 2D.1B-Pilot:
OPEN / CURRENT

Phase 2D.1B-Full:
LOCKED until pilot evidence, final JPEG quality decision
and GPT review PASS

Phase 2D.1C:
LOCKED until Phase 2D.1B-Full PASS

Phase 2D.1D:
LOCKED until Phase 2D.1C PASS
```

Readiness flags:

```text
jpg_training_representation_ready: false
coco_jpg_training_annotation_ready: false
mmdetection_dataset_loading_ready: false
empty_image_retention_ready: false
dataset_training_ready: false
training_authorized: false
```

---

### Quyết định tiếp theo

Phase tiếp theo được phép mở:

```text
Phase 2D.1B-Pilot — Representative DICOM-to-JPG Pilot
Environment: Local
```

Mục tiêu tiếp theo:

```text
Chọn pilot theo deterministic coverage-first protocol.

Đọc và giải mã pixel DICOM thật theo protocol version 1.0.0.

Tạo paired pilot JPG quality 95 và 100.

Kiểm tra geometry và bbox invariance.

Tính whole-image và bbox-ROI fidelity metrics.

Thực hiện visual audit và difference heatmaps.

Lựa chọn một final JPEG quality.

Yêu cầu GPT review trước khi mở full conversion.
```

Full conversion 4.894 ảnh vẫn bị khóa.

---

---

## 2026-07-28 — PHASE 2D.1B-Pilot: Representative DICOM-to-JPG Pilot & Final JPEG Quality Decision

### Mục tiêu

Thực hiện representative DICOM-to-JPG pilot trên dữ liệu DICOM thật theo protocol version `1.0.0` đã khóa tại Phase 2D.1A.

Mục tiêu chính:

```text
Đọc DICOM header cho toàn bộ 4,894 ảnh trong controlled scope.

Chọn representative pilot subset theo deterministic
coverage-first protocol.

Chỉ decode pixel cho pilot subset.

Tạo pre-JPEG uint8 reference PNG.

Tạo paired JPEG quality 95 và quality 100.

Đánh giá whole-image fidelity.

Đánh giá bbox-ROI fidelity.

Kiểm tra geometry và bbox invariance.

Thực hiện visual audit.

So sánh fidelity với storage/I/O cost.

Khóa một final JPEG quality cho Phase 2D.1B-Full.
```

Phase này không thực hiện full conversion 4,894 ảnh và không tạo `coco_master_jpg.json`.

---

### Đã làm

#### 1. Kiểm tra DICOM inventory

Lệnh kiểm tra:

```cmd
dir /b D:\ssl_detection_xray\data\raw\vinbigdata\dicom_subset\train\*.dicom | find /c /v ""
```

Kết quả:

```text
4,894 DICOM files
```

DICOM root được cấu hình:

```cmd
set VINBIGDATA_DICOM_ROOT=D:\ssl_detection_xray\data\raw\vinbigdata\dicom_subset
```

DICOM thực tế nằm tại:

```text
D:\ssl_detection_xray\data\raw\vinbigdata\dicom_subset\train
```

`VINBIGDATA_DICOM_ROOT` trỏ đến thư mục cha của `train` vì canonical/COCO path có dạng:

```text
train/<image_id>.dicom
```

#### 2. Thử pilot với backend `pylibjpeg`

Lệnh chạy:

```cmd
python scripts\02D1B_pilot_dicom_to_jpg.py --jpeg2000-decoder pylibjpeg > reports\phase2D1B_pilot_run_output.txt 2>&1
```

Kết quả:

```text
Protocol preflight: PASS
Input cross-check: PASS
DICOM paths resolved: 4,894/4,894
Header inventory: 4,894/4,894
Pilot selection: 64 images
Coverage: 54/54 features
Hard fail: jpeg2000_backend_unavailable:pylibjpeg
EXIT_CODE=1
```

Đây là expected guardrail behavior.

Pipeline dừng trước pixel decoding vì backend được yêu cầu không khả dụng. Không có silent fallback sang decoder khác.

#### 3. Kiểm tra các JPEG2000 backend đang có

Lệnh kiểm tra:

```cmd
python -c "from src.utils import dicom_jpg_protocol as P; print('pylibjpeg=', P.jpeg2000_backend_available('pylibjpeg')); print('gdcm=', P.jpeg2000_backend_available('gdcm')); print('pillow=', P.jpeg2000_backend_available('pillow'))"
```

Kết quả:

```text
pylibjpeg = false
gdcm      = false
pillow    = true
```

Do đó pilot được chạy lại bằng backend `pillow`, không cài thêm dependency và không sửa pipeline.

#### 4. Chạy representative pilot bằng Pillow

Lệnh chạy:

```cmd
python scripts\02D1B_pilot_dicom_to_jpg.py --jpeg2000-decoder pillow > reports\phase2D1B_pilot_run_output_pillow.txt 2>&1
set PILOT_EXIT=%ERRORLEVEL%
echo EXIT_CODE=%PILOT_EXIT% >> reports\phase2D1B_pilot_run_output_pillow.txt
```

Kết quả:

```text
INFO protocol preflight PASS:
version=1.0.0
fingerprint=1528da27758d35786847141c37d0ddb754dddb146aff116a8f3a9a7b07221229

INFO input cross-check PASS:
images=4,894
annotations=36,096
categories=14
abnormal_images=4,394
no_finding_images=500

INFO DICOM root resolved via env:
D:\ssl_detection_xray\data\raw\vinbigdata\dicom_subset

INFO resolved 4,894/4,894 controlled DICOM paths

INFO header inventory complete: 4,894/4,894

INFO selected 64 pilot images;
coverage OK: 54/54 features

INFO Phase 2D.1B-Pilot structural run complete:
OPEN_REVIEW_REQUIRED

EXIT_CODE=0
```

#### 5. Pilot selection và metadata coverage

Pilot được chọn theo:

```text
Selection strategy: deterministic_coverage_first
Tie-break seed: 2026
Selected images: 64
Selected No Finding images: 16
```

Coverage result:

```text
Metadata/features expected: 54
Metadata/features covered: 54
Missing features: 0

Abnormal classes expected: 14
Abnormal classes covered: 14

Extrema expected: 10
Extrema covered: 10

Fully covered: true
```

#### 6. Pixel decoding

Kết quả pixel decoding:

```text
Pixel decode attempts: 64
Pixel decode success: 64
Pixel decode errors: 0
Unique decoded images: 64
```

Chỉ 64 pilot images được decode pixel.

4,894 DICOM headers được đọc để xây inventory và chọn representative subset, nhưng full controlled-scope pixel conversion chưa được thực hiện.

#### 7. Geometry và bbox invariance

Geometry validation gồm:

```text
64 images × 2 JPEG candidates = 128 validation records
```

Kết quả:

```text
Pre-JPEG shape unchanged: PASS
Reference PNG shape unchanged: PASS
Decoded JPG shape unchanged: PASS

Reference PNG mode L: PASS
JPEG mode L: PASS

Reference PNG dtype uint8: PASS
Decoded JPEG dtype uint8: PASS

Reference PNG exact pixel match: PASS
EXIF orientation absent or 1: PASS

Pixel matrix order unchanged: PASS
Rotation applied: false
Flip applied: false
Transpose applied: false
EXIF orientation transform applied: false

BBox scaling required: false
```

Không resize, crop, rotation, flip hoặc transpose được thực hiện.

Không bbox nào bị scale, clamp hoặc sửa.

---

### So sánh fidelity và dung lượng

Whole-image fidelity được tính giữa:

```text
pre-JPEG uint8 reference image
versus
decoded JPEG image
```

BBox-ROI fidelity được tính trên 402 canonical bbox thuộc pilot subset.

| Tiêu chí | JPEG quality 95 | JPEG quality 100 | Nhận xét |
|---|---:|---:|---|
| Pilot images | 64 | 64 | Bằng nhau |
| Whole-image MAE trung bình | 0.873271 | 0.085074 | q100 thấp hơn |
| Whole-image RMSE trung bình | 1.235387 | 0.291270 | q100 thấp hơn |
| Whole-image PSNR trung bình | 47.2414 dB | 58.8577 dB | q100 cao hơn |
| Whole-image SSIM trung bình | 0.981217 | 0.998981 | q100 cao hơn |
| Whole-image max absolute error lớn nhất | 12 | 2 | q100 thấp hơn |
| BBox ROI được đánh giá | 402 | 402 | Bằng nhau |
| BBox-ROI MAE trung bình | 0.848022 | 0.087567 | q100 thấp hơn |
| BBox-ROI PSNR trung bình | 47.9413 dB | 58.7247 dB | q100 cao hơn |
| BBox-ROI SSIM trung bình | 0.996632 | 0.999820 | q100 cao hơn |
| ROI max absolute error lớn nhất | 10 | 2 | q100 thấp hơn |
| Kích thước trung bình mỗi ảnh, decimal MB | 1.619 MB | 3.162 MB | q100 lớn hơn khoảng 1.95 lần |
| Tổng dung lượng 64 ảnh pilot | 98.82 MiB | 192.97 MiB | q100 gần gấp đôi |
| Compression ratio trung bình | 5.04:1 | 2.47:1 | q95 nén hiệu quả hơn |
| Ước tính 4,894 ảnh | 7.38 GiB | 14.41 GiB | q100 tăng khoảng 7.03 GiB |

Storage comparison:

```text
JPEG quality 95 giảm khoảng 48.79% dung lượng
so với JPEG quality 100.

JPEG quality 100 lớn hơn JPEG quality 95
khoảng 1.95 lần.
```

Pairwise comparison:

```text
Whole-image:
q100 có MAE, PSNR và SSIM tốt hơn q95 trên 64/64 ảnh.

BBox-ROI:
q100 có ROI MAE, ROI PSNR và ROI SSIM tốt hơn q95
trên 402/402 bbox.
```

Kết luận định lượng:

```text
Quality 100 là ứng viên có numerical fidelity cao nhất.

Quality 95 là ứng viên có trade-off tốt hơn giữa
fidelity và storage/I/O cost.
```

---

### Kiểm tra các bbox nhỏ

Để kiểm tra rủi ro mất chi tiết ở các tổn thương nhỏ, 20 bbox có diện tích tương đối nhỏ nhất trong pilot được xem xét riêng ở quality 95.

Kết quả:

```text
Mean ROI MAE: 0.4165
Mean ROI PSNR: 52.29 dB
Mean ROI SSIM: 0.995884
Largest ROI maximum absolute error: 5
```

Không quan sát thấy suy giảm fidelity bất thường tập trung ở nhóm bbox nhỏ trong pilot.

Kết quả này chỉ là representation fidelity evidence, không phải bằng chứng về detection performance.

---

### Visual audit

Visual evidence bao gồm:

```text
Full-image contact sheet
BBox-specific crops
Difference heatmaps
Small-lesion cases
Rare-class cases
Dimension extrema
BBox extrema
Worst q95 whole-image cases
Worst q95 ROI cases
No Finding metadata strata
```

Contact-sheet review không phát hiện lỗi nghiêm trọng rõ ràng như:

```text
Global polarity inversion
Unexpected crop
Rotation
Flip
Transpose
Geometry deformation
Anatomical truncation do conversion
```

Visual audit được dùng để kiểm tra representation pipeline và không được diễn giải là clinical validation.

---

### Lý do chọn JPEG quality 95

Final JPEG quality được khóa là:

```text
95
```

Lý do:

1. Quality 95 giữ mức whole-image fidelity cao:

   ```text
   Mean MAE < 1 gray level trên thang uint8 [0,255]
   Mean PSNR = 47.24 dB
   Mean SSIM = 0.981217
   ```

2. Quality 95 giữ mức bbox-ROI fidelity cao:

   ```text
   Mean ROI MAE = 0.848022
   Mean ROI PSNR = 47.94 dB
   Mean ROI SSIM = 0.996632
   ```

3. Nhóm 20 bbox nhỏ nhất vẫn có:

   ```text
   Mean ROI PSNR = 52.29 dB
   Mean ROI SSIM = 0.995884
   ```

4. Geometry và bbox invariance đều PASS:

   ```text
   Width/height unchanged
   Pixel matrix order unchanged
   No rotation
   No flip
   No transpose
   BBox scaling required = false
   ```

5. Quality 95 giảm khoảng 48.79% projected storage so với quality 100:

   ```text
   q95 projected full scope: 7.38 GiB
   q100 projected full scope: 14.41 GiB
   projected saving: khoảng 7.03 GiB
   ```

6. Quality 100 có numerical fidelity cao hơn nhưng gần gấp đôi dung lượng. Hiện chưa có evidence cho thấy phần fidelity tăng thêm này là cần thiết cho downstream detector.

Quyết định được mô tả là:

```text
Fidelity–storage/I/O trade-off decision
```

Không được mô tả là:

```text
Quality 95 có model performance tốt hơn quality 100.
Quality 95 tương đương lâm sàng với DICOM.
Quality 95 không làm mất bất kỳ thông tin chẩn đoán nào.
Quality 95 đã được clinical validation.
```

---

### Evidence đã tạo

Implementation và unit-test evidence:

```text
scripts/02D1B_pilot_dicom_to_jpg.py
src/utils/dicom_jpg_protocol.py
tests/test_phase2D1B_pilot_guardrails.py
reports/phase2D1B_pilot_unit_tests_output_v6.txt
```

Pilot run logs:

```text
reports/phase2D1B_pilot_run_output.txt
reports/phase2D1B_pilot_run_output_pillow.txt
```

Pilot validation evidence:

```text
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

Pilot mapping:

```text
data/processed/image_mapping/
phase2D1B_pilot_dicom_to_jpg_mapping.csv
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

---

### Review GPT và researcher

Structural pipeline result:

```text
phase_status: OPEN_REVIEW_REQUIRED
structural_dod_candidate: true
gpt_review_status at generation time: pending
final_jpeg_quality at generation time: null
full_conversion_authorized at generation time: false
```

Sau khi review quantitative, geometry, bbox-ROI, storage và visual evidence:

```text
Pilot execution: COMPLETED
Pixel decoding: PASS 64/64
Coverage: PASS 54/54
Abnormal class coverage: PASS 14/14
No Finding pilot count: PASS 16
Geometry preservation: PASS
BBox invariance: PASS
Quantitative fidelity: PASS
Critical visual failure: false
Selected candidate: JPEG quality 95
```

Decision artifact được cập nhật:

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

---

### Quyết định

Phase 2D.1B-Pilot được khóa với trạng thái:

```text
CLOSED / PASS
```

Quyết định chính thức:

```text
Final JPEG quality: 95

Phase 2D.1B-Full:
OPEN / NEXT

Full controlled-scope conversion:
AUTHORIZED
```

Quality 95 được khóa cho toàn bộ Phase 2D.1B-Full.

Không được thay đổi quality giữa các ảnh hoặc giữa các subset.

---

### Các chú ý và giới hạn diễn giải

1. `phase2D1B_pilot_validation.json` là structural evidence được sinh trước review nên vẫn có thể ghi:

   ```text
   OPEN_REVIEW_REQUIRED
   final_jpeg_quality: null
   full_conversion_authorized: false
   ```

   Không sửa thủ công generated validation evidence để làm mất lịch sử.

   Quyết định closure được ghi trong:

   ```text
   reports/phase2D1B_pilot_decision_template.json
   ```

2. Phase 2D.1A protocol evidence không nên bị sửa ngược sau pilot. Final quality decision được ghi ở Phase 2D.1B-Pilot.

3. Quality 95 được chọn dựa trên:

   ```text
   Representation fidelity
   Geometry preservation
   BBox-ROI preservation
   Small-lesion diagnostic checks
   Storage/I/O trade-off
   ```

   Không dựa trên downstream mAP hoặc model ablation.

4. Chưa được claim:

   ```text
   Downstream detector superiority
   Clinical equivalence
   Full DICOM standard conformance
   Diagnostic safety
   Dataset training readiness
   ```

5. `full_conversion_authorized = true` chỉ cho phép chạy DICOM-to-JPG conversion trên controlled scope.

   Nó không cho phép:

   ```text
   Training
   Inference
   Pseudo-labeling
   Threshold tuning
   Test-set evaluation
   ```

6. Full conversion phải tiếp tục dùng:

   ```text
   Protocol version: 1.0.0
   Final JPEG quality: 95
   Decoder policy: explicit backend, no silent fallback
   No resize
   No crop
   No rotation
   No flip
   No transpose
   No bbox scaling
   ```

7. 4,894 JPG files không được commit vào ordinary Git.

8. `coco_master_jpg.json` chưa được tạo ở pilot phase.

9. MMDetection empty-image loading và việc giữ đủ 500 No Finding images chưa được validate.

---

### Vấn đề / rủi ro còn lại

```text
Full 4,894-image DICOM-to-JPG conversion chưa chạy.

Full JPG inventory chưa được validate.

Full-scope decode error count chưa biết.

Full-scope width/height mismatch count chưa biết.

Full-scope traceability mapping chưa được tạo.

coco_master_jpg.json chưa được tạo.

MMDetection dataset build chưa được kiểm tra.

filter_empty_gt=False chưa được kiểm tra thật.

500 No Finding images chưa được xác nhận giữ nguyên
trong MMDetection dataset.

Dataset chưa training-ready.

Training chưa được authorize.
```

---

### Ràng buộc tuân thủ

Trong Phase 2D.1B-Pilot đã tuân thủ:

```text
Không chạy full conversion trước pilot decision.

Không tạo full JPG dataset.

Không tạo coco_master_jpg.json.

Không resize ảnh.

Không crop ảnh.

Không rotate ảnh.

Không flip ảnh.

Không transpose ảnh.

Không scale bbox.

Không clamp bbox.

Không sửa canonical annotations.

Không sửa coco_master.json.

Không tạo train/val/test split.

Không tạo labeled/unlabeled split.

Không train.

Không inference.

Không pseudo-label.

Không tune threshold.

Không tính AP/mAP.

Không dùng test set.

Không claim dataset training-ready.

Không authorize training.
```

---

### Trạng thái checklist

Được tick:

```text
Phase 2D.1B-Pilot implementation complete.

Guardrail tests PASS 139/139.

DICOM inventory 4,894 verified.

Header inventory 4,894/4,894.

Representative pilot selected.

Pilot coverage 54/54.

All 14 abnormal classes covered.

16 No Finding images selected.

Pixel decode PASS 64/64.

Pre-JPEG reference PNG created.

JPEG quality 95 pilot created.

JPEG quality 100 pilot created.

Whole-image fidelity metrics computed.

BBox-ROI fidelity metrics computed.

Geometry validation PASS.

BBox invariance PASS.

Visual audit completed.

Fidelity and storage comparison completed.

Final JPEG quality 95 selected.

Pilot decision artifact updated.

Phase 2D.1B-Pilot GPT/researcher review PASS.

Phase 2D.1B-Pilot CLOSED / PASS.

Full conversion authorized.
```

Chưa được tick:

```text
Phase 2D.1B-Full conversion executed.

4,894 JPG files created.

Full decode validation PASS.

Full geometry validation PASS.

Full mapping created.

coco_master_jpg.json created.

JPG training representation ready.

MMDetection dataset loading PASS.

500 No Finding images retained by MMDetection.

Empty-image retention ready.

Dataset training-ready.

Training authorized.
```

---

### Trạng thái gate

```text
Phase 2D.1A:
CLOSED / PASS

Phase 2D.1B-Pilot:
CLOSED / PASS

Final JPEG quality:
95 / LOCKED

Phase 2D.1B-Full:
OPEN / CURRENT

Phase 2D.1C:
LOCKED until Phase 2D.1B-Full PASS

Phase 2D.1D:
LOCKED until Phase 2D.1C PASS
```

Readiness flags:

```text
final_jpeg_quality: 95
full_conversion_authorized: true

jpg_training_representation_ready: false
coco_jpg_training_annotation_ready: false
mmdetection_dataset_loading_ready: false
empty_image_retention_ready: false
dataset_training_ready: false
training_authorized: false
```

---

### Quyết định tiếp theo

Phase tiếp theo:

```text
Phase 2D.1B-Full —
Full Controlled-Scope DICOM-to-JPG Conversion & Validation

Environment:
Local
```

Mục tiêu tiếp theo:

```text
Convert đủ 4,894 DICOM thành JPG quality 95.

Không tạo q100 trong full conversion.

Validate 4,894 JPG files.

Validate decode errors = 0.

Validate width/height mismatches = 0.

Validate geometry changes = 0.

Tạo full DICOM-to-JPG traceability mapping.

Tạo coco_master_jpg.json dưới dạng path-only derivative.

Validate COCO-JPG:
images = 4,894
annotations = 36,096
categories = 14
No Finding images = 500
No Finding annotations = 0

Yêu cầu GPT review trước khi mở Phase 2D.1C.
```

Dataset vẫn chưa được phép dùng để train cho đến khi Phase 2D.1C MMDetection loading và empty-image retention PASS.

---

### Resolution update after Phase 2D.1B-Full

```text
Phase 2D.1B-Full was subsequently executed, reviewed and closed
with PASS on 2026-07-29.

The full 4,894-image JPG representation now exists.

coco_master_jpg.json now exists and has passed full structural,
geometry, bbox, category and No Finding validation.

The statements above describing the full conversion, full JPG inventory
and coco_master_jpg.json as incomplete are retained as historical
Phase 2D.1B-Pilot records. They are not the current project state.

MMDetection dataset loading and retention of all 500 No Finding
empty-GT images remain unvalidated.

Dataset training readiness remains false.

Training authorization remains false.
```

---

## 2026-07-29 — PHASE 2D.1B-Full: Full Controlled-Scope DICOM-to-JPG Conversion & Validation

### Mục tiêu

Thực hiện full controlled-scope DICOM-to-JPG conversion sau khi Phase 2D.1B-Pilot đã:

```text
CLOSED / PASS
```

và JPEG quality cuối đã được khóa:

```text
Final JPEG quality: 95
```

Mục tiêu chính:

```text
Convert đủ 4,894 DICOM thuộc controlled scope thành JPG quality 95.

Áp dụng thống nhất protocol version 1.0.0 cho toàn bộ ảnh.

Không resize, crop, rotate, flip hoặc transpose ảnh.

Không thay đổi geometry hoặc scale bounding box.

Tạo full DICOM-to-JPG traceability mapping.

Tạo coco_master_jpg.json dưới dạng path-only JPG training derivative.

Validate đầy đủ image inventory, geometry, bbox boundaries,
category mapping và No Finding policy.

Promote validated outputs vào vị trí chính thức.

Kiểm tra cleanup và final output integrity.

Không thực hiện MMDetection loading, training, inference
hoặc downstream detector evaluation trong phase này.
```

Phase này sử dụng:

```text
DICOM metadata-aware, standard-aligned reference representation pipeline
```

Đây là reference representation pipeline, không được mô tả là phương pháp mới, thuật toán mới hoặc đóng góp thuật toán.

---

### Đã làm

#### 1. Chạy preflight trước full conversion

Đã kiểm tra:

```text
Controlled-scope DICOM inventory.

Canonical image and bbox inputs.

Official coco_master.json.

Protocol version và final JPEG quality.

Output paths và promotion conditions.

Required decoder availability.

No Finding image inventory.

Guardrail conditions trước khi execute full conversion.
```

Preflight result:

```text
PASS
```

#### 2. Chạy full controlled-scope conversion

Đã chạy implementation:

```text
scripts/02D1B_full_dicom_to_jpg.py
```

với chế độ execute full conversion:

```cmd
python scripts\02D1B_full_dicom_to_jpg.py --execute-full
```

Fixed conversion policy:

```text
Protocol version:
1.0.0

JPEG quality:
95

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

Full conversion result:

```text
Controlled-scope images processed: 4,894
Output JPG files: 4,894
Conversion errors: 0
```

Không tạo quality 100 trong full conversion.

Toàn bộ 4,894 ảnh sử dụng cùng JPEG quality 95.

#### 3. Kiểm tra pixel decoder

Decoder audit:

```text
Native decoder images: 2,776
pylibjpeg JPEG 2000 decoder images: 2,118
Total decoded images: 4,894
Decode errors: 0
```

Không sử dụng silent decoder fallback.

Decoder được ghi nhận trong metadata audit và traceability evidence.

#### 4. Áp dụng intensity-transformation protocol

Transformation audit:

```text
VOI/windowing branch images: 4,536
Theoretical fallback branch images: 358
Presentation-polarity inversions: 1,562
Pixel-padding processing required: 0
```

Pipeline tiếp tục tuân thủ transformation order đã khóa tại Phase 2D.1A.

Không sửa ngược protocol evidence của Phase 2D.1A.

#### 5. Validate geometry

Đã kiểm tra:

```text
Output width.

Output height.

Orientation preservation.

Resize status.

Crop status.

Rotation status.

Flip status.

Transpose status.

Geometry consistency giữa DICOM source, JPG output
và canonical image metadata.
```

Kết quả:

```text
Missing JPG files: 0
Duplicate image IDs: 0
Width/height mismatches: 0
Orientation changes: 0
Geometry validation: PASS
```

#### 6. Validate bounding boxes

Đã đối chiếu toàn bộ canonical bounding boxes với kích thước JPG tương ứng.

Kết quả:

```text
Canonical annotations: 36,096
BBox scaling performed: false
BBox boundary validation: PASS
Invalid bbox after conversion: 0
Out-of-bound bbox after conversion: 0
```

Bounding box coordinates không bị thay đổi do pipeline không resize, crop, rotate, flip hoặc transpose ảnh.

#### 7. Tạo full traceability mapping

Đã tạo:

```text
reports/phase2D1B_full_mapping.csv
reports/phase2D1B_full_mapping.jsonl
```

Mapping bảo toàn quan hệ:

```text
image_id
DICOM source path
JPG output path
decoder branch
intensity-transformation branch
presentation-polarity handling
source width/height
output width/height
JPEG quality
conversion status
```

Kết quả:

```text
Mapped images: 4,894
Missing mappings: 0
Duplicate image IDs: 0
Conversion errors: 0
```

#### 8. Tạo COCO-JPG training derivative

Đã tạo:

```text
data/processed/coco/coco_master_jpg.json
```

Vai trò:

```text
coco_master.json:
Official annotation master.

coco_master_jpg.json:
Path-only JPG training derivative.

coco_master_jpg.json không thay thế coco_master.json.
```

Chỉ trường image representation path được chuyển sang JPG.

Không thay đổi:

```text
image IDs
annotation IDs
category IDs
bounding box coordinates
annotation areas
iscrowd values
category semantics
No Finding policy
```

#### 9. Validate COCO-JPG structure

Validation result:

```text
COCO-JPG images: 4,894
COCO-JPG annotations: 36,096
COCO-JPG categories: 14

Missing JPG files referenced by COCO: 0
Duplicate image IDs: 0
Invalid image references: 0
Invalid category references: 0

Category mapping validation: PASS
COCO-JPG structural validation: PASS
```

#### 10. Validate No Finding policy

Kết quả:

```text
Abnormal images with bbox: 4,394
No Finding images: 500
No Finding annotations: 0
No Finding category: absent
Background category: absent
```

No Finding policy validation:

```text
PASS
```

Phase này chỉ xác nhận 500 No Finding images tồn tại đúng trong COCO-JPG với zero annotations.

Phase này chưa xác nhận MMDetection thực sự giữ lại đủ 500 ảnh đó trong dataset pipeline.

#### 11. Promote validated outputs

Sau khi full validation PASS, các validated outputs được promote vào vị trí chính thức:

```text
data/processed/images_jpg/train/<image_id>.jpg
data/processed/coco/coco_master_jpg.json
```

Promotion result:

```text
PASS
```

#### 12. Cleanup và final output integrity

Đã kiểm tra:

```text
Backup cleanup.

Temporary output cleanup.

Final JPG inventory.

Final COCO-JPG references.

Final mapping integrity.

Missing output detection.

Unexpected output detection.
```

Kết quả:

```text
Backup cleanup: PASS
Final output integrity: PASS
Final JPG files: 4,894
Missing JPG files referenced by COCO: 0
```

#### 13. Chạy guardrail tests

Đã sử dụng:

```text
tests/test_phase2D1B_full_guardrails.py
```

Guardrail tests kiểm tra:

```text
Protocol version lock.

Final JPEG quality lock.

No mixed JPEG quality.

No geometry-changing transformation.

No bbox scaling.

Canonical annotation preservation.

coco_master.json immutability.

COCO-JPG path-only derivation.

No Finding preservation.

Preflight and promotion conditions.

Forbidden-action enforcement.
```

Kết quả:

```text
PASS
```

---

### Evidence đã tạo

Primary implementation:

```text
scripts/02D1B_full_dicom_to_jpg.py
tests/test_phase2D1B_full_guardrails.py
```

Preflight evidence:

```text
reports/phase2D1B_full_preflight.json
reports/phase2D1B_full_preflight.md
```

Full validation evidence:

```text
reports/phase2D1B_full_validation.json
reports/phase2D1B_full_validation.md
```

Promotion và cleanup evidence:

```text
reports/phase2D1B_full_promotion.json
reports/phase2D1B_full_cleanup_audit.json
```

Audit evidence:

```text
reports/phase2D1B_full_metadata_audit.csv
reports/phase2D1B_full_bbox_audit.csv
reports/phase2D1B_full_no_finding_audit.csv
reports/phase2D1B_full_errors.csv
```

Traceability mapping:

```text
reports/phase2D1B_full_mapping.csv
reports/phase2D1B_full_mapping.jsonl
```

Final COCO-JPG derivative:

```text
data/processed/coco/coco_master_jpg.json
```

Final processed representation:

```text
data/processed/images_jpg/train/<image_id>.jpg
```

Lưu ý:

```text
4,894 JPG files trong data/processed/images_jpg/train/
không được commit vào ordinary Git.
```

---

### Kết quả validation

```text
Full DICOM inventory: PASS

Images processed: 4,894/4,894
Output JPG files: 4,894
Uniform JPEG quality 95: PASS

Decode errors: 0
Missing JPG files: 0
Duplicate image IDs: 0

Width/height mismatches: 0
Orientation changes: 0
Geometry validation: PASS

BBox boundary validation: PASS
Category mapping validation: PASS
No Finding validation: PASS

COCO-JPG images: 4,894
COCO-JPG annotations: 36,096
COCO-JPG categories: 14
COCO-JPG structural validation: PASS

Output promotion: PASS
Backup cleanup: PASS
Final output integrity: PASS

Missing JPG files referenced by COCO: 0
```

---

### Review GPT và researcher

Review scope:

```text
Full conversion completeness.

Uniform JPEG quality.

Decoder audit.

Intensity-transformation audit.

Geometry preservation.

BBox boundary preservation.

Category mapping preservation.

No Finding policy.

COCO-JPG structural validity.

Traceability mapping.

Output promotion.

Cleanup and final output integrity.

Claim guardrails.

Next-phase gate.
```

Review result:

```text
Phase 2D.1B-Full implementation: PASS
Full controlled-scope conversion: PASS
Full validation: PASS
Promotion: PASS
Cleanup: PASS
Final output integrity: PASS
GPT/researcher review: PASS
```

Không phát hiện bằng chứng cho phép kết luận:

```text
MMDetection dataset loading ready.

Empty-GT image retention ready.

Dataset training-ready.

Training authorized.
```

---

### Quyết định

Phase 2D.1B-Full được khóa với trạng thái:

```text
CLOSED / PASS
```

Quyết định chính thức:

```text
Final JPEG quality:
95 / LOCKED

Full conversion completed:
true

Full validation passed:
true

Output promotion passed:
true

Backup cleanup passed:
true

Final output integrity passed:
true

JPG training representation ready:
true

COCO-JPG training annotation ready:
true
```

Artifact roles được khóa:

```text
DICOM:
Immutable raw medical source.

JPG quality 95:
Processed training representation generated by the fixed,
versioned and reproducible DICOM metadata-aware,
standard-aligned reference representation pipeline.

coco_master.json:
Official annotation master.

coco_master_jpg.json:
Validated path-only JPG training derivative.
It does not replace coco_master.json.
```

Phase tiếp theo được phép mở:

```text
Phase 2D.1C —
MMDetection Dataset / Empty-Image Loading Validation
```

Tuy nhiên, Phase 2D.1C hiện mới có trạng thái:

```text
NOT STARTED / NEXT
```

---

### Các chú ý và giới hạn diễn giải

1. Phase 2D.1B-Full PASS xác nhận full JPG representation và `coco_master_jpg.json` đã được tạo, vượt qua validation về inventory, geometry, bbox boundaries, category mapping, No Finding policy, promotion và final integrity.

2. Phase 2D.1B-Full PASS không xác nhận:

   ```text
   MMDetection loading readiness.

   Retention of empty-GT images trong MMDetection.

   Dataset training readiness.

   Training authorization.
   ```

3. Không được diễn giải quality 95 là có hiệu năng detector tốt hơn quality 100. Chưa có downstream q95-versus-q100 detector ablation.

4. Quality 100 là numerical-fidelity winner trong pilot. Quality 95 được chọn như một fidelity–storage/I/O trade-off.

5. Không được khẳng định JPG tương đương lâm sàng với DICOM hoặc bảo toàn mọi đặc trưng có thể phục vụ chẩn đoán.

6. Không được mô tả representation pipeline là phương pháp mới, thuật toán mới hoặc đóng góp thuật toán.

7. Controlled downstream image-representation ablation không phải yêu cầu bắt buộc ở trạng thái hiện tại và chưa được xác nhận là đã được giảng viên hướng dẫn phê duyệt.

8. Không dùng `full_conversion_authorized: true` để mô tả trạng thái hiện hành sau khi conversion hoàn tất. Trạng thái hiện hành là `full_conversion_completed: true`.

---

### Vấn đề / rủi ro còn lại

```text
MMDetection chưa load JPG + COCO-JPG dataset.

filter_empty_gt=False hoặc cấu hình tương đương
chưa được kiểm tra trong MMDetection.

Retention của toàn bộ 500 No Finding images
trong MMDetection dataset pipeline chưa được xác nhận.

Three-channel replication trong actual MMDetection pipeline
chưa được xác nhận.

Dataset length trong MMDetection chưa được kiểm tra.

Annotation loading và empty-GT behavior
chưa được kiểm tra bằng framework dataloader.

Không có downstream q95-versus-q100 detector ablation.

Controlled downstream image-representation ablation
chưa được xác nhận là bắt buộc hoặc đã được phê duyệt.

Train/validation/test split chưa được tạo.

Labeled/unlabeled SSL subsets chưa được tạo.

Dataset training readiness vẫn false.

Training authorization vẫn false.
```

---

### Ràng buộc tuân thủ

Trong Phase 2D.1B-Full đã tuân thủ:

```text
Không sửa source DICOM files.

Không sửa canonical image table.

Không sửa canonical bbox table.

Không sửa canonical class mapping.

Không sửa coco_master.json.

Không resize, crop, rotate, flip hoặc transpose ảnh.

Không scale hoặc clamp bbox.

Không tạo train/val/test split.

Không tạo labeled/unlabeled split.

Không train detector.

Không chạy detector inference.

Không tạo pseudo-label.

Không tune confidence threshold.

Không tính AP/mAP.

Không dùng test set.

Không claim MMDetection loading readiness.

Không claim empty-image retention readiness.

Không claim dataset training-ready.

Không authorize training.

Không commit 4,894 JPG files vào ordinary Git.
```

Các ràng buộc trên tiếp tục có hiệu lực sau khi Phase 2D.1B-Full đóng.

---

### Trạng thái checklist

Được tick:

```text
Phase 2D.1B-Pilot CLOSED / PASS.
Final JPEG quality 95 LOCKED.
Phase 2D.1B-Full implementation complete.
Full preflight PASS.
Full controlled-scope conversion executed.
4,894/4,894 DICOM images processed.
4,894 JPG files created.
Uniform JPEG quality 95 validated.
Full pixel decoding PASS.
Decode errors = 0.
Full metadata audit created.
Full geometry validation PASS.
Width/height mismatches = 0.
Orientation changes = 0.
Full bbox boundary validation PASS.
Full No Finding validation PASS.
Full traceability mapping created.
coco_master_jpg.json created.
COCO-JPG images = 4,894.
COCO-JPG annotations = 36,096.
COCO-JPG categories = 14.
No Finding images = 500.
No Finding annotations = 0.
COCO-JPG structural validation PASS.
Output promotion PASS.
Backup cleanup PASS.
Final output integrity PASS.
Missing JPG files referenced by COCO = 0.
Phase 2D.1B-Full GPT/researcher review PASS.
Phase 2D.1B-Full CLOSED / PASS.
JPG training representation ready.
COCO-JPG training annotation ready.
```

Chưa được tick:

```text
Phase 2D.1C started.
MMDetection JPG loading PASS.
MMDetection COCO-JPG loading PASS.
filter_empty_gt=False validated.
Dataset length = 4,894 in MMDetection.
All 500 No Finding images retained by MMDetection.
Empty-image retention ready.
Phase 2D.1C CLOSED / PASS.
Phase 2D.1D opened.
Phase 2D.1 overall CLOSED / PASS.
Train/validation/test split created.
Labeled/unlabeled split created.
Dataset training-ready.
Training authorized.
```

---

### Trạng thái gate

```text
Phase 2D.1:
IN PROGRESS

Phase 2D.1A:
CLOSED / PASS

Phase 2D.1B-Pilot:
CLOSED / PASS

Phase 2D.1B-Full:
CLOSED / PASS

Final JPEG quality:
95 / LOCKED

Phase 2D.1C:
NOT STARTED / NEXT

Phase 2D.1D:
LOCKED until Phase 2D.1C PASS
```

Readiness flags:

```text
final_jpeg_quality: 95

full_conversion_completed: true
full_validation_passed: true
promotion_passed: true
cleanup_passed: true
final_output_integrity_passed: true

jpg_training_representation_ready: true
coco_jpg_training_annotation_ready: true

mmdetection_dataset_loading_ready: false
empty_image_retention_ready: false
dataset_training_ready: false
training_authorized: false
```

---

### Quyết định tiếp theo

Phase tiếp theo:

```text
Phase 2D.1C —
MMDetection Dataset / Empty-Image Loading Validation

Environment:
Google Colab

Status:
NOT STARTED / NEXT
```

Mục tiêu tiếp theo:

```text
Cài đặt và ghi nhận MMDetection environment trên Google Colab.

Load JPG representation bằng MMDetection image pipeline.

Load coco_master_jpg.json bằng MMDetection dataset implementation.

Xác nhận dataset length = 4,894 trước split.

Áp dụng và validate filter_empty_gt=False
hoặc cấu hình tương đương.

Xác nhận đủ 4,394 abnormal images được giữ.

Xác nhận đủ 500 No Finding images với zero annotations được giữ.

Kiểm tra empty-GT sample behavior.

Kiểm tra image tensor shape và channel behavior.

Kiểm tra annotation loading trên abnormal samples.

Không train detector trong Phase 2D.1C.

Yêu cầu evidence review trước khi claim dataset training readiness
hoặc mở Phase 2D.1D.
```

Dataset vẫn chưa được phép dùng để train tại thời điểm đóng Phase 2D.1B-Full.

---

## 2026-07-30 — PHASE 2D.1C: MMDetection Dataset / Empty-Image Loading Validation

### Mục tiêu

Xác nhận bằng actual MMDetection/MMEngine pipeline rằng JPG representation và
`coco_master_jpg.json` của controlled scope có thể được load đúng trước khi tạo
split hoặc training.

Các gate bắt buộc:

```text
Load đủ 4,894 ảnh khi filter_empty_gt=False.

Giữ đủ 4,394 abnormal images.

Giữ đủ 500 No Finding images có zero ground-truth boxes.

Khi filter_empty_gt=True, chỉ loại đúng 500 zero-GT images.

Decode đúng JPG thành tensor ba kênh trong actual pipeline.

Load đúng bbox và class label của abnormal images.

Standard dataloader và forced empty-GT dataloader đều hoạt động.

Full pipeline audit phải bao phủ 4,894/4,894 ảnh.

Không train detector trong phase này.
```

Phase 2D.1C chỉ kiểm định dataset loading và empty-image behavior. Phase này
không tạo train/validation/test split, không tạo labeled/unlabeled SSL subsets,
không train detector và không đánh giá AP/mAP.

---

### Đã làm

#### 1. Tạo cấu hình validation MMDetection

Đã tạo:

* `configs/validation/phase2D1C_mmdet_dataset_loading.py`

Cấu hình validation khóa các thành phần cần thiết để kiểm tra:

* JPG training representation;
* COCO-JPG annotation;
* MMDetection dataset construction;
* `filter_empty_gt=False`;
* đối chứng với `filter_empty_gt=True`;
* image loading và annotation loading;
* dataloader behavior đối với standard samples và empty-GT samples.

#### 2. Tạo validator Phase 2D.1C

Đã tạo:

* `scripts/02D1C_validate_mmdet_dataset_loading.py`

Validator thực hiện:

* dựng MMDetection dataset từ `coco_master_jpg.json`;
* kiểm tra dataset length và image ID order;
* đối chiếu abnormal/zero-GT membership;
* kiểm tra hành vi của `filter_empty_gt=False`;
* kiểm tra hành vi đối chứng của `filter_empty_gt=True`;
* decode ảnh qua actual pipeline;
* kiểm tra image tensor shape/channel;
* kiểm tra bbox và label sau pipeline;
* chạy standard dataloader batch;
* ép lấy empty-GT dataloader batch;
* xuất image-level audit, error table và báo cáo tổng hợp.

#### 3. Tạo test suite guardrail

Đã tạo:

* `tests/test_phase2D1C_mmdet_dataset_loading_guardrails.py`

Test suite bao phủ:

* cấu hình và input guardrails;
* COCO/JPG membership;
* abnormal và zero-GT counting;
* dataset loading assertions;
* bbox/label validation;
* dataloader validation;
* report-state semantics;
* regression guard cho MMEngine `serialize_data=True`.

#### 4. Sửa lỗi audit dưới MMEngine `serialize_data=True`

Trong lần kiểm định đầu, MMEngine có thể serialize dataset và làm
`dataset.data_list` trở thành danh sách rỗng dù `len(dataset) > 0`. Nếu validator
đọc trực tiếp `data_list`, audit có thể duyệt 0 ảnh nhưng không phản ánh dataset
thật.

Validator đã được sửa để lấy record theo public indexed API:

```text
dataset.get_data_info(index)
```

thay vì giả định `dataset.data_list` luôn chứa toàn bộ records.

Đã bổ sung regression test chứng minh:

```text
len(dataset) > 0
dataset.data_list == []
dataset.get_data_info(index) trả record hợp lệ
dataset_image_ids_in_order(dataset) vẫn lấy đủ ID theo đúng thứ tự
```

Regression test mới đã PASS và khóa lỗi này cho các lần chạy sau.

#### 5. Chạy full pipeline audit

Đã chạy validator với full-audit mode trên toàn bộ controlled scope:

```text
4,394 abnormal images
500 No Finding / zero-GT images
4,894 images total
```

Không dùng subset audit để đưa ra kết luận cuối.

#### 6. Bảo toàn evidence

Đã kiểm tra lại SHA-256 của bốn report sau khi bổ sung regression test. Các hash
không thay đổi, xác nhận evidence full audit không bị sửa bởi bước cập nhật test.

File backup tạm dùng trong quá trình sửa lỗi đã được xóa và không đưa vào Git.

---

### Evidence đã tạo

Các file implementation và test:

* `configs/validation/phase2D1C_mmdet_dataset_loading.py`
* `scripts/02D1C_validate_mmdet_dataset_loading.py`
* `tests/test_phase2D1C_mmdet_dataset_loading_guardrails.py`

Các report:

* `reports/phase2D1C_mmdet_dataset_errors.csv`
* `reports/phase2D1C_mmdet_dataset_image_audit.csv`
* `reports/phase2D1C_mmdet_dataset_loading_report.json`
* `reports/phase2D1C_mmdet_dataset_loading_report.md`

SHA-256:

```text
0780595f5ff69c36329f05d69f7bb353fd095f32a0df3f76b16f039143a5f2cf  reports/phase2D1C_mmdet_dataset_errors.csv
00df8ed311e6de0ba863fa8e5a90551d34ef080b12cdd0063b6397fdfd76e474  reports/phase2D1C_mmdet_dataset_image_audit.csv
dabb3dbf27373c5271cdb3137406b583a9d3b7ee607ca2faabe18033ab772ca8  reports/phase2D1C_mmdet_dataset_loading_report.json
fb0170cadee8b7b66d81be4681af0b8955ba3c4e6b584fb5faf35d8e054b9ce9  reports/phase2D1C_mmdet_dataset_loading_report.md
```

Regression recheck:

```text
35 passed in 7.58s
PHASE 2D.1C REGRESSION RECHECK: PASS
```

Evidence preservation check:

```text
PHASE 2D.1C EVIDENCE PRESERVATION CHECK: PASS
```

---

### Kết quả validation

Full pipeline audit:

```text
full_pipeline_audit: true

images_audited: 4,894/4,894
abnormal_audited: 4,394/4,394
empty_audited: 500/500

bbox_label_num_audited: 4,894
bbox_label_all_audited_valid: true

errors: 0
```

MMDetection dataset behavior:

```text
filter_empty_gt=False:
retained 4,894/4,894 images
retained 4,394/4,394 abnormal images
retained 500/500 zero-GT images

filter_empty_gt=True:
excluded exactly 500 zero-GT images
retained the 4,394 abnormal images
```

Pipeline/dataloader behavior:

```text
JPG decoding through actual MMDetection pipeline: PASS
Three-channel image behavior: PASS
Abnormal bbox/label loading: PASS
Empty-GT sample handling: PASS
Standard dataloader batch: PASS
Forced empty-GT dataloader batch: PASS
```

Readiness flags:

```text
dataset_training_ready: true
training_authorized: false
```

Hai flag trên không đồng nghĩa. Kết quả đầu tiên xác nhận dataset đã vượt qua
technical loading gate của Phase 2D.1C. Kết quả thứ hai tiếp tục khóa training
cho đến khi các phase/gate tiếp theo hoàn tất.

---

### Review GPT và researcher

Kết quả review:

```text
Phase 2D.1C implementation: PASS

MMDetection JPG loading: PASS

MMDetection COCO-JPG loading: PASS

filter_empty_gt=False retention: PASS

500/500 No Finding image retention: PASS

filter_empty_gt=True control: PASS

Full 4,894-image pipeline audit: PASS

BBox/label validation: PASS

Standard and empty-GT dataloader validation: PASS

Errors: 0

Regression tests: 35 passed

Evidence preservation: PASS

Dataset training readiness: TRUE

Training authorization: FALSE

Phase 2D.1C: CLOSED / PASS
```

Lỗi `serialize_data=True` được xem là lỗi implementation có ý nghĩa vì bộ 34
test trước đó không phát hiện trường hợp dataset có length hợp lệ nhưng
`data_list` bị clear. Việc bổ sung test thứ 35 là guardrail bắt buộc để ngăn
audit 0 ảnh tái diễn.

---

### Quyết định

1. Đóng Phase 2D.1C với trạng thái:

   ```text
   CLOSED / PASS
   ```

2. Chấp nhận JPG quality 95 representation và `coco_master_jpg.json` là
   technically loadable trong MMDetection cho toàn bộ controlled scope.

3. Khóa cấu hình giữ ảnh âm:

   ```text
   filter_empty_gt=False
   ```

   hoặc cấu hình framework tương đương trong các dataset/dataloader dùng cho
   nghiên cứu.

4. Xác nhận 500 No Finding images là valid zero-GT detection samples và phải
   được giữ trong các bước tạo split và SSL protocol theo thiết kế nghiên cứu.

5. Nâng trạng thái:

   ```text
   mmdetection_dataset_loading_ready: true
   empty_image_retention_ready: true
   dataset_training_ready: true
   ```

6. Tiếp tục giữ:

   ```text
   training_authorized: false
   ```

7. Không dùng kết quả Phase 2D.1C để tuyên bố model performance, detector
   accuracy hoặc clinical equivalence.

8. Không cần chạy lại full audit chỉ vì bổ sung regression test, vì script và
   dữ liệu không thay đổi sau full audit PASS và bốn report hash đã được xác
   nhận không đổi.

---

### Các chú ý và giới hạn diễn giải

1. `dataset_training_ready: true` chỉ có nghĩa dataset đã vượt qua technical
   loading/retention gate trong phạm vi Phase 2D.1C.

2. `training_authorized: false` có nghĩa chưa được bắt đầu training tại thời
   điểm đóng phase này.

3. Full pipeline audit không chứng minh mô hình sẽ hội tụ, đạt AP/mAP mong muốn
   hoặc cải thiện nhờ SSL.

4. Kết quả không chứng minh JPEG quality 95 tốt hơn quality 100 về downstream
   detector performance.

5. Kết quả không chứng minh JPG tương đương lâm sàng với source DICOM.

6. Phase 2D.1C không kiểm tra train/validation/test leakage vì split chưa được
   tạo.

7. Phase 2D.1C không kiểm tra nested labeled fractions `1% ⊂ 5% ⊂ 10% ⊂ 20%`
   vì labeled/unlabeled subsets chưa được tạo.

8. Controlled downstream q95-versus-q100 detector ablation vẫn không phải yêu
   cầu bắt buộc và chưa được xem là đã được giảng viên hướng dẫn phê duyệt.

---

### Vấn đề / rủi ro còn lại

```text
Train/validation/test split chưa được tạo và khóa.

Split stratification/multilabel distribution chưa được validation.

Patient/image leakage guard chưa được xác nhận ở split phase.

Nested labeled fractions 1% ⊂ 5% ⊂ 10% ⊂ 20% chưa được tạo.

Labeled/unlabeled membership chưa được khóa.

Seed và RNG evidence cho split chưa được tạo.

Training configuration chưa được tạo và kiểm định.

Không có downstream q95-versus-q100 detector ablation.

Detector performance chưa được đánh giá.

Training authorization vẫn false.
```

---

### Ràng buộc tuân thủ

Trong Phase 2D.1C đã tuân thủ:

```text
Không sửa source DICOM files.

Không sửa 4,894 JPG representation files.

Không sửa canonical image table.

Không sửa canonical bbox table.

Không sửa canonical class mapping.

Không sửa coco_master.json.

Không thay đổi locked JPEG quality 95.

Không resize, crop, rotate, flip hoặc transpose ảnh.

Không scale hoặc clamp bbox.

Không tạo train/validation/test split.

Không tạo labeled/unlabeled split.

Không train detector.

Không chạy performance inference.

Không tạo pseudo-label.

Không tune confidence threshold.

Không tính AP/mAP.

Không dùng test set.

Không authorize training.

Không commit 4,894 JPG files vào ordinary Git.
```

---

### Trạng thái checklist

Được tick:

```text
Phase 2D.1C started.
MMDetection JPG loading PASS.
MMDetection COCO-JPG loading PASS.
filter_empty_gt=False validated.
Dataset length = 4,894 in MMDetection.
All 4,394 abnormal images retained.
All 500 No Finding images retained by MMDetection.
filter_empty_gt=True excludes exactly 500 zero-GT images.
Three-channel pipeline behavior PASS.
Abnormal bbox/label loading PASS.
Empty-GT sample behavior PASS.
Standard dataloader batch PASS.
Forced empty-GT dataloader batch PASS.
Full pipeline audit = 4,894/4,894.
Full bbox/label audit PASS.
Errors = 0.
Regression tests = 35 passed.
serialize_data=True regression guard added.
Evidence preservation hash check PASS.
MMDetection dataset loading ready.
Empty-image retention ready.
Dataset training-ready.
Phase 2D.1C GPT/researcher review PASS.
Phase 2D.1C CLOSED / PASS.
```

Chưa được tick:

```text
Phase 2D.1D opened.
Phase 2D.1 overall CLOSED / PASS.
Train/validation/test split created.
Train/validation/test split locked.
Split leakage validation PASS.
Labeled/unlabeled split created.
Nested labeled fractions created.
Training authorized.
Detector training started.
```

---

### Trạng thái gate

```text
Phase 2D.1:
IN PROGRESS

Phase 2D.1A:
CLOSED / PASS

Phase 2D.1B-Pilot:
CLOSED / PASS

Phase 2D.1B-Full:
CLOSED / PASS

Final JPEG quality:
95 / LOCKED

Phase 2D.1C:
CLOSED / PASS

Phase 2D.1D:
NEXT / NOT STARTED
```

Readiness flags:

```text
final_jpeg_quality: 95

full_conversion_completed: true
full_validation_passed: true
promotion_passed: true
cleanup_passed: true
final_output_integrity_passed: true

jpg_training_representation_ready: true
coco_jpg_training_annotation_ready: true

mmdetection_dataset_loading_ready: true
empty_image_retention_ready: true
dataset_training_ready: true
training_authorized: false
```

---

### Quyết định tiếp theo

Phase tiếp theo:

```text
Phase 2D.1D —
Split Locking / Training-Protocol Readiness

Status:
NOT STARTED / NEXT
```

Mục tiêu tiếp theo:

```text
Xác định và khóa train/validation/test split.

Validate image membership, disjointness và leakage guard.

Kiểm tra class distribution và zero-GT distribution giữa các split.

Tạo nested labeled fractions:
1% ⊂ 5% ⊂ 10% ⊂ 20%.

Khóa labeled/unlabeled membership và seed/RNG evidence.

Giữ No Finding images theo protocol đã phê duyệt.

Không dùng test set để tune threshold, chọn checkpoint,
chọn model hoặc quyết định augmentation.

Chỉ xem xét training authorization sau khi Phase 2D.1D
và các gate bắt buộc liên quan PASS.
```

Tại thời điểm đóng Phase 2D.1C:

```text
Dataset technical training readiness: TRUE
Training authorization: FALSE
```
---

## 2026-07-31 — PHASE 2D.1D: Scope Correction and Evidence-Review Opening

### Lý do đính chính

Ở cuối bản ghi Phase 2D.1C ngày 2026-07-30, Phase 2D.1D đã được mô tả
nhầm là:

```text
Split Locking / Training-Protocol Readiness
```

Mô tả trên không còn là định nghĩa hiện hành của Phase 2D.1D. Các công việc
split locking thuộc Phase 2E và các phase chuẩn bị experimental protocol tiếp
theo, không thuộc Phase 2D.1D.

Đính chính này chỉ sửa phase scope và project-state documentation. Không thay
đổi dữ liệu, protocol biểu diễn ảnh, JPEG quality, COCO annotation, kết quả
validation hoặc các evidence đã khóa.

---

### Định nghĩa chính thức của Phase 2D.1D

```text
Phase 2D.1D —
Evidence Consolidation, GPT Review & Closure
```

Mục tiêu:

1. Tổng hợp evidence xuyên suốt Phase 2D.1A, 2D.1B-Pilot, 2D.1B-Full và
   2D.1C.
2. Đối chiếu protocol YAML, decision reports, conversion reports, audit CSV,
   COCO master, COCO-JPG derivative, scripts, tests, notebook tái lập và các
   tài liệu quản trị dự án.
3. Phát hiện số liệu hoặc trạng thái mâu thuẫn, đường dẫn lỗi thời, kết luận
   vượt quá bằng chứng và nhầm lẫn giữa training readiness với training
   authorization.
4. Sửa các bất nhất tài liệu đã được review.
5. Chỉ đóng Phase 2D.1 sau khi evidence inventory và consistency review hoàn
   tất.

Phase 2D.1D không tạo train/validation/test split, không tạo labeled/unlabeled
subsets và không chạy detector training.

---

### Trạng thái chính thức sau đính chính

```text
Phase 2D.1A: CLOSED / PASS
Phase 2D.1B-Pilot: CLOSED / PASS
Phase 2D.1B-Full: CLOSED / PASS
Phase 2D.1C: CLOSED / PASS
Phase 2D.1D: OPEN / CURRENT

Final JPEG quality: 95 / LOCKED

jpg_training_representation_ready: true
coco_jpg_training_annotation_ready: true
mmdetection_dataset_loading_ready: true
empty_image_retention_ready: true
dataset_training_ready: true
training_authorized: false
```

`dataset_training_ready=true` chỉ xác nhận controlled-scope JPG + COCO-JPG
dataset đã vượt qua technical conversion, integrity, MMDetection loading và
empty-image retention gates.

Giá trị này không chứng minh detector sẽ train thành công, không chứng minh
model performance và không cấp quyền bắt đầu training.

---

### Phân chia phase đúng

```text
Phase 2D.1D:
Evidence Consolidation, GPT Review & Closure

Phase 2E:
Fixed Train/Validation/Test Split

Phase 2F:
Labeled/Unlabeled Split for SSL

Phase 2F.1:
Seed Protocol — split_seed versus training_seed
```

Do đó, các công việc sau không phải Definition of Done của Phase 2D.1D:

```text
Tạo train/validation/test split
Validate split leakage
Tạo nested labeled fractions
Khóa labeled/unlabeled membership
Tạo split-seed/RNG evidence
Authorize detector training
```

Các công việc này vẫn chưa bắt đầu và phải được thực hiện trong đúng phase sau.

---

### Training authorization

Tại Phase 2D.1D:

```text
training_authorized: false
```

Việc đóng Phase 2D.1D không tự động chuyển cờ này thành `true`.

Training chỉ được xem xét sau khi các gate liên quan đến split, leakage,
labeled/unlabeled membership, seed protocol và training configuration đã được
thực hiện, review và PASS trong các phase tiếp theo.

---

### Research-scope guardrails

Kết quả Phase 2D.1 không được dùng để tuyên bố:

```text
Training chắc chắn thành công
Detector đạt hiệu quả tốt
JPEG quality 95 tốt hơn quality 100 về detector performance
JPG tương đương lâm sàng với source DICOM
Mọi đặc trưng chẩn đoán đều được bảo toàn tuyệt đối
Representation là optimal preprocessing
Pipeline là một phương pháp hoặc thuật toán mới
```

Controlled downstream Q95-versus-Q100 detector ablation không phải yêu cầu bắt
buộc ở thời điểm hiện tại.

---

### Trạng thái sau correction

```text
Phase 2D.1 technical evidence inventory: COMPLETED
Phase 2D.1 documentation consistency review: IN PROGRESS
Phase 2D.1D final closure decision: PENDING
Training authorized: FALSE
```

Không chạy lại Phase 2D.1A, Phase 2D.1B-Pilot, Phase 2D.1B-Full hoặc Phase
2D.1C nếu không xuất hiện lỗi kỹ thuật hoặc bằng chứng mâu thuẫn mới.

---

## 2026-07-31 — PHASE 2D.1D: Evidence Consolidation, GPT Review & Closure

### Closure decision

```text
Phase 2D.1D: CLOSED / PASS
Phase 2D.1 overall: CLOSED / PASS

Technical evidence inventory: COMPLETED
Documentation consistency review: COMPLETED
Final closure decision: PASS

dataset_training_ready: true
training_authorized: false
```

Phase 2D.1D được đóng sau khi đối chiếu trực tiếp evidence của Phase 2D.1A,
2D.1B-Pilot, 2D.1B-Full và 2D.1C với protocol, scripts, tests, reports,
COCO master, COCO-JPG derivative, notebook tái lập và các tài liệu quản trị
dự án.

### Closure basis

```text
Controlled-scope images: 4,894
Abnormal images: 4,394
No Finding / zero-GT images: 500
Abnormal bbox annotations: 36,096
Abnormal detection classes: 14

Final JPEG quality: 95 / LOCKED
Full conversion and validation: PASS
Geometry and bbox invariance: PASS
Full-conversion errors: 0
COCO-JPG path-only derivative: PASS

MMDetection full-scope loading: PASS
Full pipeline audit: 4,894/4,894 PASS
filter_empty_gt=False retention: 4,894/4,894
filter_empty_gt=True control: excluded exactly 500 zero-GT images
Phase 2D.1C errors: 0
Guardrail tests: 35 passed
```

Phase 2D.1C dataloader evidence chỉ áp dụng cho `num_workers=0`. Multi-worker
loading không được kiểm định và không được tuyên bố PASS.

### Documentation consistency resolution

Các bất nhất đã được xử lý:

```text
Phase 2D.1D scope corrected to Evidence Consolidation, GPT Review & Closure.
Split locking assigned to Phase 2E, not Phase 2D.1D.
Labeled/unlabeled split assigned to Phase 2F.
Seed protocol assigned to Phase 2F.1.
The unsupported num_workers>0 PASS claim was removed.
Phase 2D.1B-Full completed checklist flags were synchronized.
Phase 2D.1C lifecycle/readiness fields were synchronized in the protocol YAML.
Phase 2D.1D and Phase 2D.1 closure states were synchronized across project documents.
```

Consistency result:

```text
research_log.md: UPDATED; consistency check PASS
PHASE_HANDOFF.md: UPDATED; consistency check PASS
PROJECT_CONTEXT.md: UPDATED; consistency check PASS
README.md: UPDATED; consistency check PASS
CHECKLIST_TRIEN_KHAI_FULL.xlsx: UPDATED; consistency check PASS
configs/protocol/phase2D1_jpg_representation.yaml: UPDATED; consistency check PASS
```

Các trạng thái `OPEN`, `PENDING_GPT` hoặc completion flag cũ bên trong
generated evidence lịch sử tiếp tục được giữ nguyên. Chúng phản ánh thời điểm
artifact được sinh và phải được đọc cùng decision/closure records; không sửa
ngược generated evidence.

### Claim boundary

Closure này chỉ cho phép kết luận:

```text
Phase 2D.1 dataset representation and loading validation: PASS
Controlled-scope dataset technical training readiness: TRUE
```

Closure này không chứng minh:

```text
Detector training chắc chắn thành công
Detector có hiệu quả tốt
JPEG quality 95 tốt hơn quality 100 về detector performance
Representation là optimal preprocessing
Pipeline là phương pháp hoặc thuật toán mới
Training đã được cấp quyền
```

### Handoff

```text
Next phase: Phase 2E — Fixed Train/Validation/Test Split
Status: NOT STARTED / NEXT

Train/validation/test split created: false
Labeled/unlabeled split created: false
Training started: false
training_authorized: false
```

Phase 2D.1D closure package phải được lưu trong một commit riêng, không amend
các commit Phase 2D.1C đã khóa. Hash của closure commit được tra từ Git history
và không được ghi ngược vào chính closure package này.

---

## Ghi chú bổ sung — Evidence về annotation QA và kiểm chứng sau chuyển đổi

> **Loại cập nhật:** supporting note only. Ghi chú này không mở lại, đổi trạng thái,
> reclassify hoặc thay đổi bất kỳ gate/authorization/quyết định nào đã được khóa trước đó.

Đối chiếu evidence hiện có cho thấy góp ý về kiểm tra bounding box, phát hiện
annotation trùng lặp và trực quan hóa sau chuẩn hóa đã được xử lý trong phạm vi
kỹ thuật của đề tài:

* **Bounding-box validity:** tính hợp lệ của bounding box được kiểm tra trên toàn
  bộ dữ liệu liên quan, bao gồm giới hạn tọa độ và tính nhất quán với kích thước
  ảnh. Trong phạm vi thực nghiệm có **36,096 abnormal bbox annotations** và không
  phát hiện bounding box không hợp lệ; do đó không phát sinh trường hợp cần loại bỏ.
* **Duplicate / near-duplicate annotation QA:** Phase 1B đã kiểm tra exact duplicate
  và near-duplicate trong cùng `(image_id, class)`. Kết quả ghi nhận **0 exact
  duplicate candidates** và **147 near-duplicate bbox records** tại ngưỡng
  `IoU >= 0.95`. Các near-duplicate này được giữ dưới dạng candidate vì dữ liệu
  có nguồn annotation multi-radiologist; bbox rất giống nhau không mặc nhiên được
  xem là lỗi dữ liệu và không bị xóa máy móc.
* **Post-conversion geometry/annotation consistency:** kiểm tra tự động trên
  **4,894 ảnh** và **36,096 bounding box** xác nhận tính nhất quán về kích thước,
  geometry và hệ tọa độ sau quá trình xây dựng biểu diễn ảnh.
* **Visual QA:** kiểm tra trực quan trên **16 mẫu representative/stress**, bao phủ
  **14/14 lớp bất thường** và gồm **4 ảnh zero-GT**, không phát hiện sai lệch không
  gian rõ rệt giữa ảnh JPEG và annotation COCO.
* **JPEG fidelity:** sai khác do mã hóa **JPEG Q95** đã được đánh giá định lượng
  so với biểu diễn lossless trước JPEG trong pilot (**64 ảnh**, với đánh giá
  bbox-ROI trên **402 annotations**). Đây là fidelity evidence ở phạm vi pilot,
  không phải full-dataset MAE/PSNR/SSIM audit trên 4,894 ảnh.

### Giới hạn diễn giải của ghi chú

Các evidence trên hỗ trợ kết luận rằng tính hợp lệ của annotation, geometry và
hệ tọa độ bounding box được duy trì nhất quán cho mục tiêu object detection, và
sai khác do JPEG Q95 đã được định lượng trong pilot. Chúng **không** được diễn
giải thành bằng chứng rằng DICOM và JPEG giống hệt ở mức pixel, rằng toàn bộ
thông tin lâm sàng của DICOM gốc được bảo toàn tuyệt đối, hoặc rằng JPEG Q95 là
lossless/clinically equivalent với DICOM gốc.

Ghi chú này chỉ tổng hợp evidence đã có; **không thay đổi các trạng thái phase,
gate, readiness hoặc authorization đã được ghi nhận trước đó trong research log**.
