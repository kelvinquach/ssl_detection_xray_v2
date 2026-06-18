# RESEARCH_CHECKLIST.md

Checklist theo phase cho `ssl_detection_xray_v2`. Chỉ tick khi có **evidence**.

## PHASE 0 — Setup repo & môi trường  *(hiện tại)*

- [ ] Cấu trúc thư mục đầy đủ (configs, data, src, scripts, experiments, reports, plots, models, logs, tests, draft).
- [ ] File tài liệu: README, CLAUDE, STRUCTURE, RESEARCH_CHECKLIST, repository_structure, research_log.
- [ ] File môi trường: `requirements.txt`, `environment.yml`, `scripts/00_setup_environment.sh`.
- [ ] Utility tái lập: `src/utils/seed.py`, `src/utils/env.py`.
- [ ] Script kiểm tra: `scripts/00_check_environment.py`.
- [ ] Protocol: `configs/protocol/checkpoint_policy.yaml`.
- [ ] `.gitignore` loại trừ dữ liệu nặng & trọng số.
- [ ] Chạy `00_check_environment.py` → có `reports/phase0_environment_check.json`.
- [ ] Có `data/manifests/seed_state_manifest.json`.
- [ ] Có `reports/phase0_pip_freeze.txt`.
- [ ] Xác nhận MMDetection import OK (hoặc ghi rõ `import_ok: false` nếu chưa cài).

**Ràng buộc Phase 0:** không đọc dataset, không convert COCO, không split, không train.

## PHASE 1 — Dữ liệu & split  *(chưa bắt đầu)*

- [ ] Khảo sát VinBigData (EDA) — *chỉ khi được phép.*
- [ ] Định nghĩa labeled/unlabeled cho semi-supervised.
- [ ] Tạo train/val/test split có manifest + seed.
- [ ] Convert annotation sang COCO format.

## PHASE 2 — Baseline supervised  *(chưa bắt đầu)*

- [ ] Train detector supervised baseline (MMDetection).
- [ ] Chọn checkpoint trên **val** theo `mAP@0.5:0.95`.

## PHASE 3 — Semi-supervised  *(chưa bắt đầu)*

- [ ] Pseudo-labeling / teacher-student pipeline.
- [ ] Tune ngưỡng pseudo-label trên **val** (không bao giờ trên test).

## PHASE 4 — Đánh giá cuối  *(chưa bắt đầu)*

- [ ] Đánh giá **test một lần duy nhất**.
- [ ] Báo cáo, biểu đồ, phân tích.

> Quy tắc đánh giá cố định: xem `configs/protocol/checkpoint_policy.yaml`.
