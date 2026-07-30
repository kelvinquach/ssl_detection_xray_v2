# ssl_detection_xray_v2

Semi-supervised object detection for anomaly detection on chest X-rays.

**Đề tài:** Nghiên cứu học bán giám sát cho dò tìm bất thường trên X-quang phổi.  
**Trọng tâm:** Semi-supervised object detection trên **VinBigData Chest X-ray**.  
**Framework chính:** [MMDetection](https://github.com/open-mmlab/mmdetection) (OpenMMLab). Detectron2 là *optional fallback*.

> **Trạng thái hiện tại: Phase 2D.1C — CLOSED / PASS**
>
> - Phase 2D.1C — MMDetection Dataset / Empty-Image Loading Validation: **CLOSED / PASS**
> - Phase 2D.1B-Full — Full Controlled-Scope DICOM-to-JPG Conversion & Validation: **CLOSED / PASS**
> - Full controlled-scope conversion: **COMPLETED**
> - Final output integrity check: **PASS**
> - Final JPEG quality: **95 / LOCKED**
> - Phase 2D.1B-Pilot — Representative DICOM-to-JPG Pilot: **CLOSED / PASS**
> - Phase 2D.1A — Image Representation Protocol Decision: **CLOSED / PASS**
> - Phase 2D — COCO Master Conversion & Validation: **CLOSED / PASS**
> - Phase 2C — Framework & Format Decision / COCO Conversion Planning: **PASS**
> - Phase 2B — Canonical Detection Annotation Schema: **PASS**
> - Phase 2A — Data Standardization / Image-Boundary Validation: **PASS**
> - Phase 1D — Label Reliability & Kappa Feasibility: **PASS**
> - Phase 1C — Dataset Scope Decision: **PASS**
> - Phase 1B — Annotation Quality: **PASS**
> - Phase 1A — Dataset Overview: **PASS**
> - Phase 0 — Setup Environment: **CORE PASS**
> - Current phase: **Phase 2D.1D — Evidence Consolidation, GPT Review & Closure**
>
> Phase 2D.1C xác nhận dataset đã **training-ready về mặt kỹ thuật**, nhưng training vẫn **chưa được phép**.

## Trạng thái Phase 2D.1

| Hạng mục | Trạng thái |
|---|---|
| Phase 2D.1A — Image Representation Protocol Decision | CLOSED / PASS |
| Phase 2D.1B-Pilot — Representative DICOM-to-JPG Pilot | CLOSED / PASS |
| Final JPEG quality | 95 / LOCKED |
| Phase 2D.1B-Full — Full Controlled-Scope Conversion | CLOSED / PASS |
| Full conversion completed | TRUE |
| Full validation | PASS |
| Output promotion | PASS |
| Backup cleanup | PASS |
| Final output integrity | PASS |
| Phase 2D.1C — MMDetection Dataset / Empty-Image Loading Validation | CLOSED / PASS |
| Phase 2D.1D — Evidence Consolidation, GPT Review & Closure | OPEN / CURRENT |
| JPG training representation ready | TRUE |
| COCO-JPG training annotation ready | TRUE |
| MMDetection dataset loading ready | TRUE |
| Empty-image retention ready | TRUE |
| Dataset training-ready | TRUE |
| Training authorized | FALSE |

### Kết quả representative pilot

```text
Controlled DICOM paths resolved: 4,894/4,894
DICOM header inventory: 4,894/4,894

Pilot images selected: 64
No Finding pilot images: 16

Metadata/features coverage: 54/54
Abnormal class coverage: 14/14

Pixel decode success: 64/64
Pixel decode errors: 0

Geometry preservation: PASS
BBox invariance: PASS
Critical visual failure: false

Final JPEG quality: 95
Full conversion authorized: true
```

### Kết quả full controlled-scope conversion

```text
Full-scope images processed: 4,894
Output JPG files: 4,894
Conversion errors: 0

Native decoder images: 2,776
pylibjpeg JPEG 2000 decoder images: 2,118

VOI/windowing branch images: 4,536
Theoretical fallback branch images: 358
Presentation-polarity inversions: 1,562
Pixel-padding processing required: 0

COCO images: 4,894
COCO annotations: 36,096
COCO categories: 14

Abnormal images with bbox: 4,394
No Finding images: 500
No Finding annotations: 0

Geometry validation: PASS
BBox boundary validation: PASS
Category mapping validation: PASS
No Finding validation: PASS
Promotion: PASS
Cleanup: PASS
Final output integrity: PASS
Missing JPG referenced by COCO: 0
```

### Kết quả MMDetection dataset / empty-image loading validation

```text
Phase 2D.1C full pipeline audit: PASS
Images audited: 4,894/4,894
Abnormal images audited: 4,394/4,394
Zero-GT images audited: 500/500

BBox/label validation: PASS
Errors: 0
Regression/unit tests: 35 passed

filter_empty_gt=False: retained 4,894/4,894 images
filter_empty_gt=True: excluded exactly 500 zero-GT images
Standard empty-GT dataloader batches: PASS
Forced empty-GT dataloader batches: PASS

Dataset training-ready: TRUE
Training authorized: FALSE
```

Kết quả này chứng minh COCO-JPG derivative có thể được MMDetection nạp đúng, toàn bộ
500 ảnh No Finding được giữ lại khi dùng cấu hình chính thức
`filter_empty_gt=False`, và empty-GT samples đi qua dataloader mà không gây lỗi.
Kết quả không cấp quyền bắt đầu training và không thay thế các gate còn lại của
Phase 2D.1D.

### Quyết định JPEG quality

JPEG quality 100 có numerical fidelity cao hơn quality 95 trên whole-image và bbox-ROI metrics.

JPEG quality 95 được chọn vì:

- vẫn giữ whole-image và bbox-ROI fidelity cao;
- geometry và bbox invariance đều PASS;
- không phát hiện critical visual failure trong representative pilot;
- giảm projected storage khoảng **48.79%** so với quality 100;
- projected storage cho 4,894 ảnh khoảng **7.38 GiB**, so với khoảng **14.41 GiB** ở quality 100.

Quyết định này là:

```text
fidelity–storage/I/O trade-off decision
```

Quyết định này không chứng minh:

```text
JPEG quality 95 có detector performance tốt hơn quality 100.
JPG tương đương lâm sàng với DICOM gốc.
Mọi đặc trưng chẩn đoán đều được bảo toàn tuyệt đối.
Pipeline đạt full DICOM-standard conformance.
Training đã được phép.
```

## Chú ý quan trọng

- DICOM tiếp tục là **immutable raw medical source**.
- Biểu diễn JPG được xây dựng bằng **DICOM metadata-aware, standard-aligned reference representation pipeline**.
- Đây là pipeline biểu diễn tham chiếu, không được mô tả là phương pháp mới hoặc thuật toán mới.
- JPG quality 95 là processed training representation đã được khóa và kiểm chứng cho toàn bộ controlled scope.
- Toàn bộ 4,894 ảnh đã được chuyển đổi thống nhất với `quality=95`.
- Không sử dụng mixed JPEG quality giữa các ảnh hoặc subset.
- Không resize, crop, rotate, flip hoặc transpose trong DICOM-to-JPG conversion.
- Geometry ảnh được bảo toàn; không thực hiện bbox scaling.
- Không tự động clamp, xóa, hợp nhất hoặc sửa canonical bbox.
- `coco_master.json` tiếp tục là annotation master chính thức.
- `coco_master_jpg.json` là path-only training derivative; không thay thế annotation master.
- `coco_master_jpg.json` đã được tạo, validate và promotion thành công.
- `data/processed/images_jpg/train/` đã có đủ 4,894 ảnh JPG.
- 36,096 bbox đều có width, height và area dương, nằm trong biên ảnh.
- 500 ảnh No Finding đều có `annotation_count=0`.
- Không có ảnh No Finding nào giao với tập ảnh có bbox.
- MMDetection loading và chính sách giữ empty-GT image đã được validate bằng full pipeline audit.
- `filter_empty_gt=False` là cấu hình bắt buộc để giữ đủ 500 ảnh No Finding.
- `filter_empty_gt=True` loại đúng 500 ảnh zero-GT và không được dùng cho protocol chính thức.
- `jpg_training_representation_ready=true` chỉ xác nhận output biểu diễn JPG.
- `coco_jpg_training_annotation_ready=true` chỉ xác nhận COCO JPG derivative đã được tạo và validate.
- `dataset_training_ready=true` còn bao gồm bằng chứng MMDetection loading và empty-image retention từ Phase 2D.1C.
- `dataset_training_ready=true` không đồng nghĩa `training_authorized=true`.
- Training tiếp tục bị khóa cho đến khi hoàn tất gate được quy định ở Phase 2D.1D.
- Không chạy lại `--execute-full` nếu không có lý do kỹ thuật được ghi nhận và phê duyệt.
- Không commit 4,894 JPG files vào ordinary Git.
- Không tạo train/validation/test split, labeled/unlabeled split hoặc bắt đầu training trước đúng phase.

## Vai trò trong dự án

| Vai trò | Trách nhiệm |
|---|---|
| Người nghiên cứu | Quyết định hướng nghiên cứu, protocol và phạm vi thí nghiệm |
| GPT | Thiết kế quy trình, phản biện logic và review evidence |
| Claude | Viết code trong repo theo prompt đã được kiểm soát |
| Python | Chạy script, kiểm tra dữ liệu và tạo evidence |

Quy trình bắt buộc:

```text
script → output → DoD → GPT review → người nghiên cứu tick checklist
```

## Cấu trúc repo

Xem:

- [`STRUCTURE.md`](STRUCTURE.md)
- [`repository_structure.md`](repository_structure.md)
- [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md)
- [`PHASE_HANDOFF.md`](PHASE_HANDOFF.md)
- [`research_log.md`](research_log.md)

## Dữ liệu và annotation master

Controlled working scope:

```text
Images: 4,894
Abnormal images: 4,394
No Finding images: 500
Abnormal bbox annotations: 36,096
Abnormal detection classes: 14
```

COCO annotation master chính thức:

```text
data/processed/coco/coco_master.json
```

COCO master đã được validate:

```text
Images: 4,894
Annotations: 36,096
Categories: 14
No Finding annotations: 0
Invalid annotations: 0
pycocotools load: PASS
```

COCO JPG training derivative đã được tạo và validate:

```text
data/processed/coco/coco_master_jpg.json
```

Final processed image representation:

```text
data/processed/images_jpg/train/<image_id>.jpg
```

Final output integrity:

```text
JPG files: 4,894
COCO images: 4,894
COCO annotations: 36,096
COCO categories: 14
Missing JPG files referenced by COCO: 0
```

Artifact roles:

| Artifact | Vai trò |
|---|---|
| DICOM | Immutable raw medical source |
| JPG quality 95 | Processed training representation |
| `coco_master.json` | Official annotation master |
| `coco_master_jpg.json` | Path-only JPG training derivative |

## DICOM-to-JPG protocol

Protocol chính thức:

```text
configs/protocol/phase2D1_jpg_representation.yaml
```

Protocol version:

```text
1.0.0
```

Transformation order:

```text
DICOM decode
→ pixel-padding mask
→ modality transformation
→ VOI LUT/windowing
→ presentation-polarity normalization
→ deterministic uint8 conversion
→ JPEG encoding
```

Final encoding policy:

```text
JPEG quality: 95
Storage mode: grayscale L
Resize: false
Crop: false
Rotation: false
Flip: false
Transpose: false
BBox scaling: false
```

Full conversion status:

```text
Phase 2D.1B-Full: CLOSED / PASS
Full conversion completed: true
Validation passed: true
Promotion passed: true
Cleanup passed: true
Final output integrity passed: true
```

Primary implementation and evidence:

```text
scripts/02D1B_full_dicom_to_jpg.py
tests/test_phase2D1B_full_guardrails.py
reports/phase2D1B_full_preflight.json
reports/phase2D1B_full_validation.json
reports/phase2D1B_full_promotion.json
reports/phase2D1B_full_cleanup_audit.json
reports/phase2D1B_full_metadata_audit.csv
reports/phase2D1B_full_bbox_audit.csv
reports/phase2D1B_full_no_finding_audit.csv
reports/phase2D1B_full_errors.csv
```

## MMDetection dataset-loading validation

Validation config:

```text
configs/validation/phase2D1C_mmdet_dataset_loading.py
```

Implementation và regression tests:

```text
scripts/02D1C_validate_mmdet_dataset_loading.py
tests/test_phase2D1C_mmdet_dataset_loading_guardrails.py
```

Evidence:

```text
reports/phase2D1C_mmdet_dataset_errors.csv
reports/phase2D1C_mmdet_dataset_image_audit.csv
reports/phase2D1C_mmdet_dataset_loading_report.json
reports/phase2D1C_mmdet_dataset_loading_report.md
```

SHA-256 của evidence đã khóa:

```text
0780595f5ff69c36329f05d69f7bb353fd095f32a0df3f76b16f039143a5f2cf  reports/phase2D1C_mmdet_dataset_errors.csv
00df8ed311e6de0ba863fa8e5a90551d34ef080b12cdd0063b6397fdfd76e474  reports/phase2D1C_mmdet_dataset_image_audit.csv
dabb3dbf27373c5271cdb3137406b583a9d3b7ee607ca2faabe18033ab772ca8  reports/phase2D1C_mmdet_dataset_loading_report.json
fb0170cadee8b7b66d81be4681af0b8955ba3c4e6b584fb5faf35d8e054b9ce9  reports/phase2D1C_mmdet_dataset_loading_report.md
```

Locked loading policy:

```text
filter_empty_gt: false
dataset_training_ready: true
training_authorized: false
```

## Giao thức đánh giá

Xem:

```text
configs/protocol/checkpoint_policy.yaml
```

Tóm tắt:

```text
Primary metric: mAP@0.5:0.95
Checkpoint selection split: validation
Test usage: final evaluation only
```

Test set không được dùng để:

- tune threshold;
- chọn checkpoint;
- chọn model hoặc backbone;
- quyết định augmentation;
- lựa chọn pseudo-label filtering policy.

## Hướng dẫn cho Claude

Xem [`CLAUDE.md`](CLAUDE.md).

Claude không được tự thay đổi:

- protocol nghiên cứu;
- final JPEG quality;
- controlled dataset scope;
- No Finding policy;
- bbox semantics;
- category mapping;
- split policy;
- metric policy;
- training authorization.

Mọi thay đổi phải tuân theo:

```text
script → output → DoD → GPT review → researcher decision
```
