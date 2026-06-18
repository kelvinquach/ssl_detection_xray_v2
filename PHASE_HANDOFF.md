# PHASE HANDOFF — `ssl_detection_xray_v2`

Ngày cập nhật: 2026-06-18

Dự án: **Nghiên cứu học bán giám sát cho dò tìm bất thường trên X-quang phổi**

Bài toán: **Semi-supervised object detection trên VinBigData Chest X-ray**

---

## 1. Vai trò làm việc

* Người nghiên cứu: người quyết định hướng nghiên cứu, protocol, phạm vi thí nghiệm.
* GPT: thiết kế quy trình, phản biện logic, review evidence, quyết định pass/fail DoD.
* Claude: viết code trong repo theo prompt được giao.
* Python: chạy script, kiểm tra dữ liệu, train/evaluate, tạo evidence.

Quy trình bắt buộc:

```text
script → output → DoD → GPT review → người nghiên cứu tick checklist
```

Không tick checklist nếu chưa có evidence.

---

## 2. Nguyên tắc nghiên cứu đã khóa

* Không nhảy phase.
* Không train khi data/split/COCO/No Finding/seed/checkpoint criterion chưa pass DoD.
* Không dùng test set để tune threshold.
* Không dùng test set để chọn checkpoint.
* Không dùng test set để chọn model/backbone.
* Không dùng test set để quyết định augmentation.
* `No Finding` là ảnh âm tính không có bbox, không phải detection class.
* Metric chính: `mAP@0.5:0.95`.
* Supervised và SSL phải dùng cùng labeled split, cùng split_seed, cùng fixed test set.
* Stability phải dùng nhiều `training_seed`.
* Local environment hiện tại không training-ready.

---

## 3. Trạng thái hiện tại

```text
Current phase: Phase 1A — Dataset Overview
Previous phase: Phase 0 — Core setup & reproducibility
Phase 0 core: PASS
Phase 0 local training framework: DEFERRED
```

Được mở:

```text
Phase 1A — Dataset Overview
```

Chưa được làm:

```text
Split train/val/test
Convert COCO
Train supervised detector
Train SSL detector
Generate pseudo-label
Tune threshold
Use test set
```

---

## 4. Phase 0 — Kết quả bàn giao

### 4.1. Phase 0A — Repo structure

Trạng thái: **PASS**

Đã có các thư mục chính:

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

Đã có tài liệu:

* `README.md`
* `CLAUDE.md`
* `STRUCTURE.md`
* `RESEARCH_CHECKLIST.md`
* `repository_structure.md`
* `research_log.md`
* `PHASE_HANDOFF.md`

Đã có protocol:

* `configs/protocol/checkpoint_policy.yaml`

---

### 4.2. Phase 0B — Core environment

Trạng thái: **PASS core / DEFER training framework**

Evidence đã có:

* `reports/phase0_environment_check.json`
* `reports/phase0_pip_freeze.txt`
* `reports/reproducibility_settings.md`
* `data/manifests/seed_state_manifest.json`

Kết quả chính:

* Python: `3.10.20`
* Conda env: `sslxray`
* Platform: `Windows-10-10.0.26200-SP0`
* PyTorch: `2.3.1`
* torchvision: `0.18.1`
* numpy: `1.24.3`
* pandas: `2.3.3`
* OpenCV/cv2: `4.11.0`
* pydicom: `3.0.2`
* pycocotools: `2.0.11`
* `pip check`: `No broken requirements found.`
* `pytest tests/test_phase0.py -q`: `5 passed`

CUDA:

* `torch.cuda.is_available()`: `False`
* `torch.version.cuda`: `null`
* GPU device count: `0`

Detection framework:

* `mmengine`: not installed
* `mmcv`: not installed
* `mmdet`: not installed
* `framework_import_ok`: `false`

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

* `data/manifests/seed_state_manifest.json`
* `reports/phase0_environment_check.json`
* `reports/reproducibility_settings.md`

Deterministic flags:

* `PYTHONHASHSEED = 2026`
* Python random seed: enabled
* NumPy seed: enabled
* PyTorch CPU seed: enabled
* PyTorch CUDA seed: not applied because CUDA unavailable
* `torch.use_deterministic_algorithms = true`
* `torch.backends.cudnn.deterministic = true`
* `torch.backends.cudnn.benchmark = false`
* `CUBLAS_WORKSPACE_CONFIG = :4096:8`

---

## 6. Checkpoint/evaluation protocol đã khóa

File:

```text
configs/protocol/checkpoint_policy.yaml
```

Protocol:

* Primary metric: `mAP@0.5:0.95`
* Checkpoint selection split: `val`
* Test usage: `final_evaluation_only`

Cấm:

* Dùng test set để tune threshold.
* Dùng test set để chọn checkpoint.
* Dùng test set để chọn model/backbone.
* Dùng test set để quyết định augmentation.

---

## 7. Checklist tick được sau Phase 0

Được tick:

* Phase 0A repo structure
* Phase 0B core environment
* pip dependency check
* PyTorch/torchvision import
* numpy/pandas/cv2/pydicom/pycocotools import
* seed manifest
* deterministic flags
* environment report
* reproducibility report
* pip freeze
* checkpoint policy
* Phase 0 pytest pass

Chưa được tick:

* MMDetection import OK
* `mmengine` import OK
* `mmcv` import OK
* `mmdet` import OK
* CUDA/GPU ready
* Local training-ready environment
* Full detection framework setup

---

## 8. Phase hiện tại: Phase 1A — Dataset Overview

Mục tiêu Phase 1A:

Đọc metadata/annotation CSV của VinBigData để tạo báo cáo tổng quan dataset.

Phase 1A chỉ được phép:

* đọc annotation CSV
* thống kê số dòng annotation
* thống kê số unique image_id
* thống kê class distribution
* kiểm tra sơ bộ bbox validity
* kiểm tra No Finding policy
* tạo evidence report

Phase 1A không được phép:

* tạo split
* convert COCO
* copy ảnh
* đọc DICOM/PNG
* train
* tạo pseudo-label
* tune threshold
* dùng test set

---

## 9. Câu hỏi Phase 1A phải trả lời

* Dataset có bao nhiêu annotation rows?
* Dataset có bao nhiêu unique `image_id`?
* Có bao nhiêu class?
* Có bao nhiêu abnormal class nếu loại `No Finding`?
* `No Finding` xuất hiện dưới dạng nào?
* `No Finding` có bbox không?
* `No Finding` có được xử lý như ảnh âm tính không?
* Có bao nhiêu No Finding images?
* Có bao nhiêu abnormal images?
* Có image nào vừa `No Finding` vừa abnormal label không?
* Có bbox thiếu coordinate không?
* Có bbox lỗi không?

  * `x_min >= x_max`
  * `y_min >= y_max`
  * `width <= 0`
  * `height <= 0`
* Phân bố bbox theo class như thế nào?
* Có class imbalance nghiêm trọng không?

---

## 10. Script cần tạo trong Phase 1A

Claude cần tạo:

```text
scripts/01A_dataset_overview.py
```

Script phải nhận tham số:

```text
--train-csv
--output-json
--class-csv
--image-summary-csv
--bbox-quality-csv
--report-md
```

Output mặc định:

```text
reports/phase1A_dataset_overview.json
reports/phase1A_class_distribution.csv
reports/phase1A_image_level_summary.csv
reports/phase1A_bbox_quality_summary.csv
reports/phase1A_dataset_overview.md
```

---

## 11. DoD Phase 1A

Phase 1A chỉ pass nếu có đủ:

* Script chạy được bằng Python.
* Có JSON report.
* Có Markdown report.
* Có class distribution CSV.
* Có image-level summary CSV.
* Có bbox quality summary CSV.
* Có thống kê total rows.
* Có thống kê unique images.
* Có thống kê No Finding images.
* Có thống kê abnormal images.
* Có danh sách abnormal classes excluding No Finding.
* Có kiểm tra bbox invalid/missing.
* Có cảnh báo nếu No Finding có bbox.
* Có cảnh báo nếu abnormal class thiếu bbox.
* Không tạo split.
* Không tạo COCO.
* Không train.
* Không dùng test set.
* GPT review pass.

Nếu No Finding có bbox thật, hoặc abnormal labels thiếu bbox hàng loạt, Phase 1A không pass cho đến khi làm rõ rule.

---

## 12. Prompt giao Claude cho Phase 1A

```text
Bạn đang ở repo D:\ssl_detection_xray_v2.

Phase hiện tại: Phase 1A — Dataset Overview.

Bối cảnh nghiên cứu:
- Đề tài: “Nghiên cứu học bán giám sát cho dò tìm bất thường trên X-quang phổi”.
- Dataset: VinBigData Chest X-ray.
- Bài toán: semi-supervised object detection.
- No Finding là ảnh âm tính không có bbox, không phải detection class.
- Metric chính sau này là mAP@0.5:0.95.
- Không được split, không convert COCO, không train trong Phase 1A.

Hãy tạo script:

scripts/01A_dataset_overview.py

Yêu cầu script:
1. Nhận tham số:
   --train-csv
   --output-json default reports/phase1A_dataset_overview.json
   --class-csv default reports/phase1A_class_distribution.csv
   --image-summary-csv default reports/phase1A_image_level_summary.csv
   --bbox-quality-csv default reports/phase1A_bbox_quality_summary.csv
   --report-md default reports/phase1A_dataset_overview.md

2. Đọc annotation CSV VinBigData.
   Script phải tự detect các cột phổ biến:
   - image_id
   - class_name
   - class_id
   - x_min, y_min, x_max, y_max
   Nếu thiếu cột bắt buộc thì báo lỗi rõ ràng.

3. Tính thống kê:
   - total_rows
   - unique_images
   - class_name list
   - class_id list nếu có
   - số row theo class_name
   - số image theo class_name
   - số bbox theo abnormal class
   - số No Finding rows
   - số No Finding images
   - số abnormal images
   - số image có cả No Finding và abnormal label nếu có
   - số bbox missing coordinate
   - số bbox có x_min >= x_max
   - số bbox có y_min >= y_max
   - số bbox width <= 0 hoặc height <= 0
   - min/mean/max width, height, area nếu có bbox hợp lệ

4. No Finding policy:
   - Treat No Finding / no finding / No finding as negative image label.
   - Không đưa No Finding vào detection class.
   - Nếu No Finding có bbox coordinates không null, ghi cảnh báo.
   - Nếu abnormal class thiếu bbox, ghi cảnh báo.

5. Output:
   - phase1A_dataset_overview.json
   - phase1A_class_distribution.csv
   - phase1A_image_level_summary.csv
   - phase1A_bbox_quality_summary.csv
   - phase1A_dataset_overview.md

6. Script phải in console summary:
   - total rows
   - unique images
   - abnormal images
   - No Finding images
   - number of abnormal classes excluding No Finding
   - bbox invalid count
   - warnings

7. Tuyệt đối không:
   - tạo split
   - tạo COCO json
   - copy ảnh
   - đọc DICOM/PNG
   - train
   - tune threshold
   - dùng test set

8. Viết code rõ ràng, có hàm main(), type hints cơ bản, error message dễ hiểu.

Sau khi tạo script, in ra lệnh chạy mẫu với đường dẫn:
python scripts/01A_dataset_overview.py --train-csv data/raw/vinbigdata/annotations/train.csv
```

---

## 13. Lệnh chạy Phase 1A

Nếu annotation nằm ở:

```text
data\raw\vinbigdata\annotations\train.csv
```

chạy:

```cmd
python scripts/01A_dataset_overview.py --train-csv data\raw\vinbigdata\annotations\train.csv
```

Nếu annotation nằm chỗ khác, dùng đúng path thực tế.

---

## 14. Output cần gửi GPT review sau Phase 1A

Gửi các output sau:

```cmd
type reports\phase1A_dataset_overview.md
```

```cmd
type reports\phase1A_dataset_overview.json
```

```cmd
type reports\phase1A_class_distribution.csv
```

```cmd
type reports\phase1A_bbox_quality_summary.csv
```

```cmd
type reports\phase1A_image_level_summary.csv
```

Nếu CSV quá dài, gửi 20 dòng đầu bằng PowerShell:

```cmd
powershell -Command "Get-Content reports\phase1A_class_distribution.csv -TotalCount 20"
```

```cmd
powershell -Command "Get-Content reports\phase1A_bbox_quality_summary.csv -TotalCount 20"
```

```cmd
powershell -Command "Get-Content reports\phase1A_image_level_summary.csv -TotalCount 20"
```

---

## 15. Gate hiện tại

```text
Phase 0 core: PASS
Phase 0 training framework: DEFERRED
Phase 1A: OPEN
Split: LOCKED
COCO conversion: LOCKED
Training: LOCKED
Pseudo-labeling: LOCKED
Threshold tuning: LOCKED
Test-set usage: LOCKED
```

---
