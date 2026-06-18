# CLAUDE.md — Hướng dẫn cho coding assistant

Tài liệu này định nghĩa cách Claude làm việc trong repo `ssl_detection_xray_v2`.

## Bối cảnh

- **Đề tài:** Học bán giám sát cho dò tìm bất thường trên X-quang phổi.
- **Trọng tâm:** Semi-supervised object detection trên VinBigData Chest X-ray.
- **Framework chính:** MMDetection. Detectron2 chỉ là *optional* (build khó trên Windows).

## Phân vai

- **Người dùng** quyết định nghiên cứu.
- **GPT** thiết kế / review logic.
- **Claude** viết code trong repo.
- **Python** chạy script và tạo bằng chứng (evidence).

## Quy tắc theo phase

### PHASE 0 (hiện tại) — Setup repo & môi trường
**ĐƯỢC PHÉP:**
- Tạo cấu trúc thư mục, file tài liệu, file môi trường.
- Viết utility tái lập (seed, env) và script kiểm tra môi trường.
- Tạo file protocol (checkpoint policy) và `.gitignore`.

**TUYỆT ĐỐI KHÔNG:**
- Không đọc dataset (kể cả `train.csv`).
- Không convert sang COCO.
- Không tạo train/val/test split.
- Không train, không inference trên dữ liệu.
- Không đụng dữ liệu VinBigData dưới mọi hình thức.

## Nguyên tắc kỹ thuật

1. **Tái lập (reproducibility) là bắt buộc.** Mọi randomness đi qua `src/utils/seed.py`.
2. **Defensive imports.** Code không được crash khi thiếu package; ghi `import_ok: false`.
3. **Evidence-first.** Mọi bước Phase 0 phải tạo artifact trong `reports/` hoặc `data/manifests/`.
4. **Hygiene đánh giá.** Tuân thủ `configs/protocol/checkpoint_policy.yaml`. Không bao giờ dùng test set để tuning.
5. **Seed mặc định:** `2026`.

## Khi không chắc

Hỏi lại người dùng / GPT trước khi mở rộng phạm vi sang phase tiếp theo.
