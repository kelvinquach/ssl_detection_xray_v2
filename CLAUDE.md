# CLAUDE.md — Hướng dẫn cho coding assistant

Tài liệu này định nghĩa cách Claude làm việc trong repo `ssl_detection_xray_v2`.

## Bối cảnh

- **Đề tài:** Học bán giám sát cho dò tìm bất thường trên X-quang phổi.
- **Trọng tâm:** Semi-supervised object detection trên VinBigData Chest X-ray.
- **Framework chính:** MMDetection. Detectron2 chỉ là *optional* (build khó trên Windows).
- **Trạng thái hiện tại:** Phase 2 (data prep), Phase 3A (dataset diagnostics) và Gate I
  (frozen protocol/data layer) đã `CLOSED / PASS`. Repo đang ở **implementation phase**
  (post-Gate-I, pre-Gate-II), triển khai theo từng gate — xem mục "Quy tắc theo phase"
  bên dưới cho gate hiện tại được authorize.

## Phân vai

- **Người dùng** quyết định nghiên cứu.
- **GPT** thiết kế / review logic.
- **Claude** viết code trong repo.
- **Python** chạy script và tạo bằng chứng (evidence).

## Quy tắc theo phase

### GATE I — Frozen protocol/data layer (`CLOSED / PASS`)
Gate I đã đóng. Các input sau đây đã khóa và **KHÔNG được sửa, ghi đè, hay rebuild**:
- `data/manifests/*`
- `data/processed/coco/*`
- `data/processed/images_jpg/*`
- Mọi file `configs/protocol/*.yaml` đánh dấu `RESEARCHER_APPROVED_LOCKED`, gồm tối
  thiểu: `phase2F_labeled_unlabeled.yaml`, `phase2F1_seed_protocol.yaml`,
  `d4_training_protocol.yaml`, `d4_ssl_protocol.yaml`, `checkpoint_policy.yaml`,
  `experimental_design.yaml`, `gateI_freeze_manifest.yaml`.

### GATE-BY-GATE IMPLEMENTATION DISCIPLINE (bắt buộc)
Implementation KHÔNG được triển khai toàn bộ cùng lúc. Mỗi gate phải theo đúng vòng:

```text
IMPLEMENT → TEST → EVIDENCE → REVIEW / PASS → NEXT GATE
```

Claude **không được tự ý chuyển sang gate kế tiếp** nếu gate hiện tại chưa có evidence
và chưa được người dùng/researcher review/authorize.

**CURRENT AUTHORIZED IMPLEMENTATION SCOPE: GATE 2 ONLY**
= Shared SUP infrastructure (dataset/dataloader đọc từ COCO JSON đã Gate-I-freeze, seed
propagation tới worker/sampler/augmentation, optimizer-update counter, AMP, checkpoint
save/resume, validation loop, configurable path roots cho Vast.ai).

Các gate sau đây là **future scope**, được nhắc để định hướng nhưng **chưa được
implement** cho tới khi Gate 2 PASS và được authorize riêng:
- Gate 3 — SUP R50/Swin-T preflight
- Gate 4 — SSL Teacher/Student
- Gate 5 — SSL hard preflight
- Gate 6 — Evaluator
- Gate 7 — Statistical fixtures

### GIAI ĐOẠN HIỆN TẠI — Implementation phase (sau Gate I, trước Gate II)
**ĐƯỢC PHÉP (trong phạm vi gate đang authorize):**
- Viết implementation code cho đúng gate hiện tại (xem CURRENT AUTHORIZED
  IMPLEMENTATION SCOPE ở trên), theo đúng `IMPLEMENTATION_HANDOFF.md`.
- Viết unit test, integration test, pilot/smoke run, preflight verification code cho
  gate đó.
- Đọc (không sửa) dữ liệu/manifest đã Gate I freeze để dataset loader hoạt động.
- Sửa/refactor code trong `src/`, `scripts/`, `tests/` thuộc phạm vi gate đang mở.

**TUYỆT ĐỐI KHÔNG:**
- Không triển khai gate chưa được authorize (xem GATE-BY-GATE IMPLEMENTATION
  DISCIPLINE).
- Không sửa giá trị khoa học trong bất kỳ file `configs/protocol/*.yaml` nào đã khóa
  `RESEARCHER_APPROVED_LOCKED`.
- Không sửa/ghi đè `data/manifests/*`, `data/processed/coco/*`, `data/processed/images_jpg/*`.
- Không rebuild train/val/test split hoặc L/U (labeled/unlabeled) membership.
- Không seed search; không đổi danh sách training seed đã khóa.
- Không để hidden ground-truth của `U_b` (unlabeled) đi vào training, pseudo-label
  filtering, threshold tuning, EMA tuning, augmentation selection, checkpoint
  selection, `Q_pseudo`, hoặc model selection.
- Không đọc/dùng test-set performance cho bất kỳ mục đích tuning hay lựa chọn nào.
- Không chạy OFFICIAL TRAINING. Official training chỉ được authorize khi TẤT CẢ hard
  preflight check ở Section 22 của `IMPLEMENTATION_HANDOFF.md` = PASS và có researcher
  authorization (Section 26).
- Pilot/debug/smoke run không được tính vào 10 training seed chính thức.

### NGUỒN CHÂN LÝ KHOA HỌC (source of truth)
- `sn-article.tex` (methodology đã giảng viên duyệt) là scientific source of truth cao
  nhất. Nếu `.tex`, `IMPLEMENTATION_HANDOFF.md`, code, config, README, hoặc script mâu
  thuẫn nhau → **`.tex` THẮNG**.
- `IMPLEMENTATION_HANDOFF.md` là implementation contract, không phải nguồn khoa học gốc.
- Nếu framework hiện tại không hỗ trợ trực tiếp một yêu cầu đã khóa: STOP, ghi nhận
  IMPLEMENTATION INCOMPATIBILITY, báo lại researcher — không tự sửa protocol.

### BÍ DANH TÀI LIỆU LOCAL (document aliases)
Repo dùng tên file rút gọn cho hai tài liệu tham chiếu chính:
- `IMPLEMENTATION_HANDOFF.md` = local working copy của
  `FINAL_IMPLEMENTATION_HANDOFF_WITH_SUPERVISOR_NOTES_VI_HARD_PREFLIGHT.md`.
- `sn-article.tex` = local working copy của bản đã duyệt
  `sn-article(20260902-092314).tex`.

Hai file này giữ nguyên nội dung; không sửa nội dung khoa học bên trong chúng, và
không commit chúng chỉ vì lý do cập nhật `CLAUDE.md`.

## Nguyên tắc kỹ thuật

1. **Tái lập (reproducibility) là bắt buộc.** Mọi randomness đi qua `src/utils/seed.py`.
2. **Defensive imports.** Code không được crash khi thiếu package; ghi `import_ok: false`.
3. **Evidence-first.**
   - Mọi implementation gate / preflight step phải tạo evidence artifact tương ứng.
   - Không được kết luận PASS nếu chưa có evidence.
   - Evidence phải lưu ở `reports/`, `data/manifests/`, hoặc output location được quy
     định bởi implementation contract (`IMPLEMENTATION_HANDOFF.md`).
4. **Hygiene đánh giá.** Tuân thủ `configs/protocol/checkpoint_policy.yaml`. Không bao giờ dùng test set để tuning.
5. **Seed policy.**
   - `partition_seed = 42` chỉ thuộc frozen data partition / labeled-subset
     construction (`configs/protocol/phase2F1_seed_protocol.yaml`,
     `phase2F_labeled_unlabeled.yaml`). Không dùng cho training.
   - Official training dùng ĐÚNG danh sách 10 training seed đã khóa, theo thứ tự, tại
     `configs/protocol/phase2F1_seed_protocol.yaml` (`ordered_training_seeds`).
   - SUP và SSL dùng CHUNG một ordered seed list, ghép cặp theo `seed_index`.
   - Không dùng generic default seed (ví dụ giá trị `2026` từng xuất hiện trong
     legacy/setup utility) cho official training. Nếu `2026` còn tồn tại ở bất kỳ
     script/tiện ích cũ nào, phải ghi rõ đó KHÔNG PHẢI official training seed.
   - Không seed search. Không replacement seed.
   - Không tự thay đổi 10 giá trị seed đã khóa trong protocol.
6. **Frozen protocol immutability.** File `configs/protocol/*.yaml` đã khóa
   `RESEARCHER_APPROVED_LOCKED` không được Claude tự sửa giá trị số/policy dưới bất kỳ
   lý do implementation nào.
7. **Official training gate.** Official training chỉ authorize khi Section 22
   (Pilot/Preflight) của `IMPLEMENTATION_HANDOFF.md` PASS toàn bộ và có researcher
   authorization rõ ràng.
8. **Gate-by-gate discipline.** Không tự chuyển gate (xem "GATE-BY-GATE IMPLEMENTATION
   DISCIPLINE" ở trên). Implementation chỉ giới hạn trong CURRENT AUTHORIZED
   IMPLEMENTATION SCOPE tại thời điểm đó.

## Khi không chắc

Hỏi lại người dùng / GPT trước khi mở rộng phạm vi sang phase tiếp theo.
