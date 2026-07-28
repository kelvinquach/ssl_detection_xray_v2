# ssl_detection_xray_v2

Semi-supervised object detection for anomaly detection on chest X-rays.

**Đề tài:** Nghiên cứu học bán giám sát cho dò tìm bất thường trên X-quang phổi.  
**Trọng tâm:** Semi-supervised object detection trên **VinBigData Chest X-ray**.  
**Framework chính:** [MMDetection](https://github.com/open-mmlab/mmdetection) (OpenMMLab). Detectron2 là *optional fallback*.

> **Trạng thái hiện tại: Phase 2D.1B-Full — Full Controlled-Scope DICOM-to-JPG Conversion & Validation**
>
> - Phase 2D.1B-Pilot — Representative DICOM-to-JPG Pilot: **CLOSED / PASS**
> - Final JPEG quality: **95 / LOCKED**
> - Full controlled-scope conversion: **AUTHORIZED**
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

## Trạng thái Phase 2D.1

| Hạng mục | Trạng thái |
|---|---|
| Phase 2D.1A — Image Representation Protocol Decision | CLOSED / PASS |
| Phase 2D.1B-Pilot — Representative DICOM-to-JPG Pilot | CLOSED / PASS |
| Final JPEG quality | 95 / LOCKED |
| Phase 2D.1B-Full — Full Controlled-Scope Conversion | OPEN / CURRENT |
| Phase 2D.1C — MMDetection Dataset / Empty-Image Loading Validation | LOCKED |
| Phase 2D.1D — Evidence Consolidation, GPT Review & Closure | LOCKED |
| JPG training representation ready | FALSE |
| COCO-JPG training annotation ready | FALSE |
| MMDetection dataset loading ready | FALSE |
| Empty-image retention ready | FALSE |
| Dataset training-ready | FALSE |
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
Dataset đã training-ready.
```

## Chú ý quan trọng

- DICOM tiếp tục là **immutable raw medical source**.
- JPG quality 95 là processed representation được khóa cho Phase 2D.1B-Full.
- Toàn bộ 4,894 ảnh trong full conversion phải sử dụng cùng `quality=95`.
- Không được dùng mixed JPEG quality giữa các ảnh hoặc subset.
- Không resize, crop, rotate, flip hoặc transpose trong DICOM-to-JPG conversion.
- Không tự động scale, clamp, xóa hoặc sửa bbox.
- `coco_master.json` tiếp tục là annotation master chính thức.
- `coco_master_jpg.json` chỉ được thay đổi `images[].file_name` từ `.dicom` sang `.jpg`.
- `coco_master_jpg.json` chưa được tạo ở pilot phase.
- MMDetection loading và `filter_empty_gt=False` chưa được validate.
- Việc giữ đủ 500 ảnh No Finding trong MMDetection chưa được chứng minh.
- Full conversion authorization không đồng nghĩa với training authorization.
- Không commit 4,894 JPG files vào ordinary Git.
- Không tạo train/val/test split, labeled/unlabeled split hoặc training trước khi đến đúng phase.

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

COCO master chính thức:

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

Planned JPG training derivative:

```text
data/processed/coco/coco_master_jpg.json
```

Planned processed image representation:

```text
data/processed/images_jpg/train/<image_id>.jpg
```

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