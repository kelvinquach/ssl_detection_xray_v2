# PHASE HANDOFF — `ssl_detection_xray_v2`

Ngày cập nhật: 2026-07-08

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
Current phase: Phase 2C — Framework & Format Decision / COCO Conversion Planning
Previous phase: Phase 2B — Canonical Detection Annotation Schema
Phase 0 core: PASS
Phase 0 local training framework: DEFERRED
Phase 1A — Dataset Overview: PASS
Phase 1B — Annotation Quality: PASS
Phase 1C — Dataset Scope Decision: PASS
Phase 1D — Label Reliability & Kappa Feasibility: PASS
Phase 2A — Data Standardization / Image-Boundary Validation: PASS
Phase 2B — Canonical Detection Annotation Schema: PASS
Git status: Phase 2B completed; pending rebase/commit/push after evidence update
```

Được mở / tiếp theo:

```text
Phase 2C — Framework & Format Decision / COCO Conversion Planning
```

Chưa được làm:

```text
Split train/val/test
COCO master conversion
Train supervised detector
Train SSL detector
Generate pseudo-label
Tune threshold
Use test set
Labeled/unlabeled split
Framework dataloader validation
Empty image loading check
```

Ghi chú:

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
Không được tạo split, COCO master, train, pseudo-label hoặc tune threshold khi chưa mở đúng phase.
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

## 15. Gate sau Phase 1D

```text
Phase 0 core: PASS
Phase 0 training framework: DEFERRED
Phase 1A — Dataset Overview: PASS
Phase 1B — Annotation Quality: PASS
Phase 1C — Dataset Scope Decision: PASS
Phase 1D — Label Reliability & Kappa Feasibility: PASS

Controlled working scope: LOCKED
Controlled scope size: 4,894 images
Abnormal images retained: 4,394 / 4,394
No Finding images selected: 500 / 10,606
Selection unit: image_id
No Finding row-level sampling used: false

Label reliability:
rad_id available: true
rad_id missing count: 0
radiologists_total: 17
radiologists_per_image_distribution: {'3': 4894}
uniform_rater_count_per_image: true
same_rater_identity_panel_across_images: false
binary_matrix_feasible: true
cohen_kappa_feasible: false
fleiss_kappa_feasible: true
overall_fleiss_kappa_mean: 0.4879

Split train/val/test: LOCKED
COCO conversion: LOCKED
Training: LOCKED
Pseudo-labeling: LOCKED
Threshold tuning: LOCKED
Test-set usage: LOCKED

Next phase:
Phase 2A — Data Standardization / Image-Boundary Validation
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

:::writing{variant="document" id="12370"}
### Next phase at that time

```text
Phase 1D — Kappa feasibility / limitation-aware analysis
Update:
Phase 1D has now been completed with PASS_agreement_computed_and_documented.
The current next phase is Phase 2A — Data Standardization / Image-Boundary Validation.
```
:::

---

## 17. Phase 1D — Label Reliability & Kappa Feasibility

Status: **PASS**

Date: 2026-07-01

### Mục tiêu

Kiểm tra tính khả thi của inter-radiologist agreement / Kappa analysis từ metadata trong controlled working scope đã khóa ở Phase 1C.

Phase 1D chỉ dùng metadata annotation. Không tạo split, không convert COCO, không train, không pseudo-label, không tune threshold, không dùng test set, không đọc pixel ảnh, không đọc DICOM/header/image dimensions và không sửa annotation gốc.

---

### Scripts run

```cmd
python scripts/01D_kappa_feasibility.py
```

---

### Outputs generated

```text
reports/phase1D_kappa_feasibility.md
reports/phase1D_kappa_feasibility.json
reports/phase1D_classwise_agreement_feasibility.csv
reports/phase1D_radiologist_per_image.csv
reports/phase1D_rare_class_kappa_instability.csv
```

---

### DoD result

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

---

### Key findings

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
classwise_feasibility_summary: 14 abnormal classes assessed; 14 with feasible Fleiss' Kappa; mean kappa=0.4879
rare_class_instability_summary: 5/14 classes carry kappa_instability_risk (severe=2, moderate=3, low=9); risk is prevalence/rarity-driven, not measured instability
label_level_agreement_status: evaluable_fleiss_computed
bbox_level_consistency_status: evaluated_descriptive_only
```

---

### Research decisions

```text
Fleiss' Kappa is computed at image-level class agreement.
Cohen's Kappa is not used as the main agreement statistic because each image has 3 radiologist ratings.
Kappa/agreement is used only as data-quality evidence and limitation evidence.
Kappa is not a model metric.
Kappa is not used for split/model/threshold selection.
Kappa is not used for training or pseudo-labeling.
Kappa is not used to delete, fuse, or edit annotations.
BBox-level consistency is kept separate from label-level agreement and remains descriptive only.
```

---

### Issues / risks

```text
Negative class decisions are inferred from read-coverage according to VinBigData labelling convention.
Kappa can be affected by prevalence imbalance.
5/14 abnormal classes carry kappa_instability_risk.
BBox-level consistency is not a bbox fusion policy.
Near-duplicate bbox handling is still deferred to a later annotation standardization decision.
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
No pixel read.
No DICOM/header read.
No image dimensions read.
No boundary validation performed.
No annotation deleted or edited.
No near-duplicate bbox deleted or fused.
No Kappa used as model metric.
No Kappa used for split/model/threshold.
```

---

### Next phase

```text
Phase 2A — Data Standardization / Image-Boundary Validation
```

:::writing{variant="document" id="93016"}
---

## 18. Phase 2A — Data Standardization / Image-Boundary Validation

Status: **PASS**

Date: 2026-07-08

### Mục tiêu

Kiểm tra DICOM availability, image dimensions và bbox boundary validity trong controlled working scope 4,894 images.

Phase 2A chỉ đọc DICOM metadata/header để lấy dimensions và validate bbox. Không tạo split, không convert COCO, không train, không pseudo-label, không tune threshold, không dùng test set, không sửa annotation và không tạo processed training images.

---

### Scripts run

```cmd
python scripts\02A_dicom_bbox_boundary_validation.py ^
  --annotations-csv data\interim\vinbigdata_phase1C_scope_annotations.csv ^
  --manifest-csv data\manifests\phase1C_selected_images_manifest.csv ^
  --dicom-root D:\ssl_detection_xray\data\raw\vinbigdata\dicom_subset\train

  ### Outputs generated

```text
reports/phase2A_dicom_bbox_validation.md
reports/phase2A_dicom_bbox_validation.json
reports/phase2A_image_metadata.csv
reports/phase2A_image_availability.csv
reports/phase2A_bbox_boundary_validation.csv
reports/phase2A_invalid_bbox_candidates.csv
reports/phase2A_dicom_read_errors.csv
```
---

  ### DoD result

```text
DICOM availability: PASS
DICOM metadata/header read: PASS
Image dimension extraction: PASS
BBox boundary validation: PASS
No Finding policy: PASS
Forbidden actions avoided: PASS
```
---

  ### Key findings

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
---

  ### Image dimension summary

```text
width_min: 1320
width_max: 3320
width_mean: 2491.66
height_min: 1416
height_max: 3408
height_mean: 2835.09
distinct_wh_pairs: 2186
```
---

  ### Research decisions

```text
All 4,894 controlled-scope DICOM files are available on local disk.
All 4,894 DICOM files are readable at metadata/header level.
Image dimensions are available for all controlled-scope images.
All 36,096 abnormal bbox rows are valid within original image boundaries.
BBox convention is treated as xyxy on original image coordinates.
No Finding remains a negative image label without bbox and is not a detection class.
No bbox was edited, clamped, deleted or fused.
No image was copied, converted, normalized or saved as processed training data.
```
---

  ### Issues / risks

```text
Pixel array decoding was not checked in the main run because pixel_array_checked=false.
This is acceptable for Phase 2A because the phase objective is metadata/header dimension and bbox boundary validation.
Canonical schema is not created yet.
COCO conversion is not created yet.
Train/val/test split is not created yet.
Framework dataloader / empty image loading is not checked yet.
Training is still locked.
Pseudo-labeling is still locked.
Threshold tuning is still locked.
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
No annotation deleted or edited.
No bbox clamped or modified.
No near-duplicate bbox deleted or fused.
No processed training images created.
No image files copied.
No image files converted.
No PNG/JPG created.
```
---

  ### Next phase

```text
Phase 2B — Canonical Schema
```
---
:::

## Phase 2B — Canonical Detection Annotation Schema

Status: **PASS**

Date: 2026-07-08

### Mục tiêu

Tạo canonical detection annotation schema cho controlled working scope 4,894 images.

Phase 2B chỉ tạo schema canonical trung gian. Không convert COCO, không tạo split, không train, không pseudo-label, không tune threshold, không dùng test set, không sửa annotation gốc, không clamp bbox, không xóa bbox và không fuse near-duplicate bbox.

### Scripts run

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

 ### Outputs generated

```text
data/processed/canonical/canonical_image_table.csv
data/processed/canonical/canonical_bbox_table.csv
data/processed/canonical/canonical_class_mapping.csv
reports/phase2B_canonical_schema_report.md
reports/phase2B_canonical_schema_validation.json
reports/phase2B_no_finding_policy_audit.csv
reports/phase2B_schema_consistency_errors.csv
```
---

 ### DoD result

```text
Canonical image table: PASS
Canonical bbox table: PASS
Canonical class mapping: PASS
No Finding policy audit: PASS
Schema consistency validation: PASS
Portable path policy: PASS
Forbidden actions avoided: PASS
```
---

 ### Key findings

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
relative_dicom_path_missing_count: 0
relative_dicom_path_absolute_count: 0
local_dicom_path_absolute_count: 4894
path_root_variable: VINBIGDATA_DICOM_ROOT
warnings: []
dod_pass_candidate: true
```
---

 ### Research decisions

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
---

 ### Issues / risks

```text
local_dicom_path stores absolute local evidence paths but must not be used as canonical downstream identifiers.
Remote/GPU environments must set VINBIGDATA_DICOM_ROOT or an equivalent data-root config.
source_row_id traces to the Phase 1C controlled-scope annotation file, not necessarily the original full VinBigData train.csv row index.
Framework dataloader validation has not been performed.
Empty image loading check has not been performed.
COCO conversion has not been performed.
Train/val/test split has not been created.
Near-duplicate bbox handling is still deferred.
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
No pixel_array read.
No image copied.
No image converted.
No processed training images created.
No annotation deleted or edited.
No bbox clamped or modified.
No near-duplicate bbox deleted or fused.
No Finding was not added as a detection class.
```
---

 ### Next phase

```text
Phase 2C — Framework & Format Decision / COCO Conversion Planning
```
---