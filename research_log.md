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
