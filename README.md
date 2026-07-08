# ssl_detection_xray_v2

Semi-supervised object detection for anomaly detection on chest X-rays.

**Đề tài:** Nghiên cứu học bán giám sát cho dò tìm bất thường trên X-quang phổi.
**Trọng tâm:** Semi-supervised object detection trên **VinBigData Chest X-ray**.
**Framework chính:** [MMDetection](https://github.com/open-mmlab/mmdetection) (OpenMMLab). Detectron2 là *optional*.

> **Trạng thái hiện tại: Phase 2B — Canonical Schema**
> > Phase 2A — Data Standardization / Image-Boundary Validation
> Phase 1D — Label Reliability & Kappa Feasibility
> Phase 1C — Dataset Scope Decision
> PHASE 1B — Annotation Quality
> Phase 1A: dataset overview
> Phase 0: Cài đặt - Kiểm tra môi trường. *không* đọc dataset, *không* convert COCO, *không* tạo split, *không* train.


## Vai trò trong dự án

| Vai trò | Trách nhiệm |
|---|---|
| Người dùng | Quyết định nghiên cứu |
| GPT | Thiết kế / review logic |
| Claude | Viết code trong repo |
| Python | Chạy script, tạo bằng chứng (evidence) |

## Cấu trúc repo

Xem [`STRUCTURE.md`](STRUCTURE.md) và [`repository_structure.md`](repository_structure.md).

## Giao thức đánh giá

Xem [`configs/protocol/checkpoint_policy.yaml`](configs/protocol/checkpoint_policy.yaml).
Tóm tắt: metric chính `mAP@0.5:0.95`; chọn checkpoint trên **val**; **test chỉ dùng một lần** cho đánh giá cuối.

## Hướng dẫn cho Claude

Xem [`CLAUDE.md`](CLAUDE.md).
