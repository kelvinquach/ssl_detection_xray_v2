# PHASE HANDOFF — `ssl_detection_xray_v2`

Ngày cập nhật: 2026-07-01

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
Current phase: Phase 1D — Kappa feasibility / limitation-aware analysis
Previous phase: Phase 1C — Dataset Scope Decision
Phase 0 core: PASS
Phase 0 local training framework: DEFERRED
Phase 1A — Dataset Overview: PASS
Phase 1B — Annotation Quality: PASS
Phase 1C — Dataset Scope Decision: PASS
Git status: Phase 1C PASS, waiting commit/push confirmation if not committed yet
```

Được mở / tiếp theo:

```text
Phase 1D — Kappa feasibility / limitation-aware analysis
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
DICOM/image-boundary validation
COCO master conversion
```

Ghi chú:

```text
Phase 1C đã khóa controlled working scope 4,894 image-level samples.
Phase 2A chưa được mở.
Không được đọc DICOM header / pixel / image dimensions cho đến Phase 2A.
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

## 8. Phase 1A — Dataset Overview

Ngày cập nhật: 2026-06-18
Status: PASS

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

## 15. Gate sau Phase 1C

```text
Phase 0 core: PASS
Phase 0 training framework: DEFERRED
Phase 1A — Dataset Overview: PASS
Phase 1B — Annotation Quality: PASS
Phase 1C — Dataset Scope Decision: PASS
Phase 1D — Kappa feasibility / limitation-aware analysis: OPEN / NEXT

Controlled working scope: LOCKED
Controlled scope size: 4,894 images
Abnormal images retained: 4,394 / 4,394
No Finding images selected: 500 / 10,606
Selection unit: image_id
No Finding row-level sampling used: false

Split train/val/test: LOCKED
COCO conversion: LOCKED
Training: LOCKED
Pseudo-labeling: LOCKED
Threshold tuning: LOCKED
Test-set usage: LOCKED
DICOM/image-boundary validation: DEFERRED to Phase 2A
```

---

### Phase 1B — Annotation Quality (kiểm tra chất lượng Annotation).

Status: PASS

Date: 2026-06-19

# Prompt giao Claude cho Phase 1B
Bạn đang ở repo:

D:\ssl_detection_xray_v2

Phase hiện tại: Phase 1B — Annotation Quality.

Bối cảnh nghiên cứu:

* Đề tài: “Nghiên cứu học bán giám sát cho dò tìm bất thường trên X-quang phổi”.
* Dataset: VinBigData Chest X-ray.
* Bài toán: semi-supervised object detection.
* Phase 0 core: PASS.
* Phase 1A — Dataset Overview: PASS.
* Phase 1A đã chạy trên full VinBigData train.csv source metadata:

  * 15,000 images.
  * 67,914 annotation rows.
  * 4,394 abnormal images.
  * 10,606 No Finding images.
  * 36,096 abnormal bbox rows.
  * 14 abnormal detection classes.
  * Invalid bbox count: 0.
  * No Finding rows with bbox: 0.
  * Images with both No Finding and abnormal labels: 0.
* Downstream controlled working scope đã khóa sau này là 4,894 images = 4,394 abnormal + 500 No Finding.
* Tuy nhiên Phase 1B vẫn chỉ chạy trên full source metadata train.csv, không tạo subset 4,894.

Nguyên tắc bắt buộc:

* Chỉ làm Phase 1B — Annotation Quality.
* Không split train/val/test.
* Không tạo subset 4,894.
* Không convert COCO.
* Không train.
* Không pseudo-label.
* Không tune threshold.
* Không dùng test set.
* Không đọc pixel ảnh.
* Không đọc DICOM/PNG.
* Chỉ đọc annotation-level metadata từ CSV.
* No Finding là ảnh âm tính không có bbox, không phải detection class.
* Không tự động xóa/sửa annotation, chỉ report lỗi/candidate.

Hãy tạo script:

scripts/01B_annotation_quality.py

Yêu cầu script:

1. Nhận tham số:
   --train-csv
   --output-json default reports/phase1B_annotation_quality.json
   --report-md default reports/phase1B_annotation_quality.md
   --annotation-sanity-md default reports/annotation_sanity_report.md
   --invalid-bbox-csv default reports/invalid_bbox_rows.csv
   --duplicate-csv default reports/duplicate_bbox_candidates.csv
   --class-mapping-csv default reports/phase1B_class_mapping.csv
   --bbox-quality-by-class-csv default reports/phase1B_bbox_quality_by_class.csv
   --image-label-consistency-csv default reports/phase1B_image_label_consistency.csv
   --near-duplicate-iou default 0.95

2. Đọc annotation CSV VinBigData.
   Script phải tự detect hoặc kiểm tra các cột:

   * image_id
   * class_name
   * class_id
   * x_min
   * y_min
   * x_max
   * y_max
   * rad_id nếu có

3. Kiểm tra bbox coordinate sanity:

   * missing coordinate trên abnormal rows
   * non-numeric coordinate
   * x_min < 0
   * y_min < 0
   * x_max < 0
   * y_max < 0
   * x_min >= x_max
   * y_min >= y_max
   * width <= 0
   * height <= 0
   * area <= 0

4. Kiểm tra bbox vượt biên ảnh:

   * Nếu CSV có cột image_width/image_height hoặc width/height đại diện kích thước ảnh, kiểm tra:

     * x_max > image_width
     * y_max > image_height
     * x_min > image_width
     * y_min > image_height
   * Nếu CSV không có image dimensions, không đọc ảnh.
   * Khi không có dimensions, report rõ:
     boundary_check_status = not_evaluable_without_image_dimensions
   * Không đọc DICOM/PNG để lấy shape trong Phase 1B.

5. Kiểm tra No Finding policy:

   * Treat No Finding / no finding / No finding as negative image label.
   * No Finding không phải detection class.
   * No Finding rows phải không có bbox coordinates.
   * Nếu No Finding có bbox, liệt kê vào invalid_bbox_rows.csv.
   * Kiểm tra image_id nào vừa có No Finding vừa có abnormal class.

6. Kiểm tra abnormal annotation consistency:

   * Abnormal class phải có bbox đầy đủ.
   * Nếu abnormal row thiếu bbox, liệt kê.
   * Tạo summary số lỗi theo class_name và class_id.

7. Kiểm tra duplicate / near-duplicate bbox:

   * Exact duplicate: cùng image_id, cùng class_id hoặc class_name, cùng x_min/y_min/x_max/y_max.
   * Near duplicate: cùng image_id, cùng class, IoU >= --near-duplicate-iou.
   * Nếu có rad_id, giữ rad_id trong output.
   * Không xóa duplicate.
   * Chỉ ghi duplicate_bbox_candidates.csv.
   * Trong report phải nói rõ duplicate candidates có thể là multi-radiologist annotations, chưa được xem là lỗi chắc chắn.

8. Kiểm tra class mapping:

   * Mỗi class_id map tới đúng một class_name.
   * Mỗi class_name map tới đúng một class_id.
   * No Finding không được đưa vào abnormal detection class.
   * Xuất phase1B_class_mapping.csv.

9. Output JSON phải có tối thiểu:

   * phase
   * train_csv
   * total_rows
   * unique_images
   * abnormal_rows
   * no_finding_rows
   * abnormal_images
   * no_finding_images
   * abnormal_detection_classes_excluding_no_finding
   * invalid_bbox_total
   * invalid_bbox_by_reason
   * no_finding_with_bbox_count
   * abnormal_missing_bbox_count
   * mixed_no_finding_abnormal_image_count
   * negative_coordinate_count
   * zero_or_negative_area_count
   * exact_duplicate_candidate_count
   * near_duplicate_candidate_count
   * class_mapping_issue_count
   * boundary_check_status
   * warnings
   * forbidden_actions_confirmed

10. Markdown report phải có:

* Executive summary.
* Scope: full source metadata train.csv only, not downstream 4,894 subset.
* Checks performed.
* Key findings.
* Invalid bbox summary.
* Duplicate / near-duplicate summary.
* Class mapping summary.
* No Finding policy summary.
* Boundary check status.
* Research risk interpretation.
* Recommended next action: send outputs to GPT review before ticking checklist.

11. Console summary phải in:

* total rows
* unique images
* invalid bbox total
* no_finding_with_bbox_count
* abnormal_missing_bbox_count
* exact duplicate candidates
* near duplicate candidates
* class mapping issues
* boundary_check_status
* warnings

12. Tuyệt đối không:

* tạo split
* tạo COCO json
* tạo subset 4,894
* copy ảnh
* đọc DICOM/PNG
* train
* pseudo-label
* tune threshold
* dùng test set
* tự động xóa/sửa annotation

13. Code cần có:

* main()
* argparse
* type hints cơ bản
* error message rõ ràng
* tạo thư mục reports nếu chưa có
* xử lý CSV lớn ổn định bằng pandas
* không phụ thuộc GPU/MMDetection

Sau khi tạo script, in ra lệnh chạy mẫu:
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

Next phase at that time:

Phase 1C — Dataset Scope Decision, now completed PASS.


---

## 16. Phase 1C — Dataset Scope Decision

Status: **PASS**

Date: 2026-07-01

### Mục tiêu

Chính thức hóa downstream controlled working scope ở mức image-level:

```text
4,894 images
= 4,394 abnormal images
+ 500 No Finding images
```

Phase 1C chỉ làm metadata/image-level scope decision.

Không tạo split, không convert COCO, không train, không pseudo-label, không tune threshold, không dùng test set, không đọc DICOM header, không đọc pixel và không đọc image dimensions.

---

### Scripts run

```cmd
python scripts\01C_dataset_scope_decision.py ^
  --train-csv D:\ssl_detection_xray_v2\data\raw\vinbigdata\annotations\train.csv ^
  --manifest-glob "D:\ssl_detection_xray\data\raw\vinbigdata\dicom_subset_chunks\dicom_package_manifest_part_*.csv" ^
  --dicom-root D:\ssl_detection_xray\data\raw\vinbigdata\dicom_subset\train ^
  --chunk-summary D:\ssl_detection_xray\data\raw\vinbigdata\dicom_subset_chunks\dicom_chunk_summary.csv
```

---

### Outputs generated

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

---

### DoD result

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

---

### Key findings

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

---

### Research decisions

```text
Controlled working scope is officially locked to 4,894 image-level samples:
4,394 abnormal images + 500 No Finding images.

The 500 No Finding samples are selected and verified at image_id level, not row level.
The controlled scope is based on the already-downloaded DICOM package manifests and validated against train.csv and DICOM filename inventory.
No Finding remains a negative image label, not a detection class.
The metadata-only subset annotation CSV is created for selected image_id values only.
```

---

### Issues / risks

```text
The controlled scope uses 500 out of 10,606 No Finding images, not the full No Finding pool.
This is a deliberate controlled-scope design decision and must be stated as a limitation.
Boundary validity is not concluded in Phase 1C because image dimensions were not read.
147 near-duplicate bbox candidates from Phase 1B are retained, not deleted or fused.
Fusion/handling of multi-radiologist boxes is deferred to a later phase.
```

---

### Forbidden actions confirmed

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

---

### Next phase

```text
Phase 1D — Kappa feasibility / limitation-aware analysis
```
