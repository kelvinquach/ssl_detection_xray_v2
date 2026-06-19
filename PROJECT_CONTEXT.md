# PROJECT_CONTEXT — SSL Object Detection trên X-quang phổi

> File này là **bộ nhớ trung tâm của dự án**.  
> Khi mở chat mới với GPT/Claude, hãy upload file này trước để giữ đúng ngữ cảnh, quy trình và các quyết định đã khóa.

---

## 1. Đề tài đã khóa

**Tên đề tài:**  
“Nghiên cứu học bán giám sát cho dò tìm bất thường trên X-quang phổi”

**Trọng tâm:**  
Semi-supervised object detection trên ảnh X-quang phổi.

**Dataset chính:**  
VinBigData Chest X-ray.

**Task:**  
Object detection bằng bounding box, **không phải classification**, **không phải segmentation**.

**Bản chất kỹ thuật:**  
Đây là bài toán **Semi-Supervised Object Detection (SSOD)**. Vì vậy pseudo-label không chỉ là nhãn lớp, mà gồm:

- `class_id`
- `confidence score`
- `bbox` theo format `[x, y, width, height]`
- thông tin lọc bbox như NMS, box size, aspect ratio, boundary validity

---

## 2. Quy ước nghiên cứu đã khóa

### 2.1 Dữ liệu và No Finding

- No Finding / normal **không phải detection class**.
- No Finding là **ảnh âm tính không có bbox**.
- Ảnh No Finding phải nằm trong `images` của COCO JSON.
- Ảnh No Finding **không có dòng annotation** trong `annotations`.
- `No Finding` **không được nằm trong `categories`**.
- Framework detection không được tự lọc ảnh empty / ảnh không có bbox.
- Cần có kiểm tra riêng: `filter_empty_gt=False` hoặc cấu hình tương đương.

### 2.2 Metric và đánh giá

- Metric chính để kết luận: **mAP@0.5:0.95**.
- Metric phụ: `mAP@0.5`, class-wise AP, recall/sensitivity, FP/image, FP per negative image.
- FP per negative image chỉ có ý nghĩa khi **test set có đủ ảnh No Finding**.
- Test set chỉ dùng để đánh giá cuối, **không dùng để chọn checkpoint, threshold, model tốt nhất, backbone hoặc hyperparameter**.
- Checkpoint được chọn bằng **validation mAP@0.5:0.95**.

### 2.3 Split, seed và reproducibility

- Labeled split: 1%, 5%, 10%, 20%.
- Nested sampling: **1% ⊂ 5% ⊂ 10% ⊂ 20%**.
- Supervised low-label và SSL phải dùng:
  - cùng labeled split;
  - cùng `split_seed`;
  - cùng fixed train/val/test;
  - cùng fixed test set.
- Stability dùng nhiều `training_seed`, thường 3–5 seeds.
- Phân biệt rõ:
  - `split_seed`: tạo train/val/test và labeled/unlabeled split;
  - `training_seed`: khởi tạo mô hình, dataloader shuffle, augmentation và training.
- Không khóa cùng một `training_seed` cho mọi run vì sẽ tạo variance giả thấp.
- Lưu đầy đủ:
  - seed number;
  - Python RNG state;
  - NumPy RNG state;
  - PyTorch CPU RNG state;
  - PyTorch CUDA RNG state;
  - deterministic flags;
  - config snapshot;
  - checkpoint;
  - log;
  - report.

### 2.4 Class imbalance

- Class imbalance được phân tích và báo cáo như một đặc trưng tự nhiên của VinBigData Chest X-ray.
- Không áp dụng kỹ thuật xử lý mất cân bằng độc lập như:
  - oversampling;
  - undersampling;
  - class weighting;
  - reweighting loss.
- Ảnh hưởng của class imbalance được phân tích thông qua:
  - class-wise AP;
  - pseudo-label count theo class;
  - retained pseudo-label ratio;
  - rare-class pseudo-label survival rate.
- Nếu có xử lý liên quan rare classes, nó chỉ được thực hiện trong phạm vi SSL / pseudo-label filtering, không được trình bày như một đóng góp xử lý imbalance độc lập.

---

## 3. Quy ước SSOD đã khóa

### 3.1 Teacher–Student SSL Detection

Pipeline chính:

```text
Supervised detector
→ Teacher model
→ pseudo-label unlabeled X-ray
→ confidence filtering
→ class-wise / dynamic thresholding
→ NMS
→ box quality filtering
→ transform pseudo-bbox từ weak view sang strong view
→ train Student trên labeled + pseudo-labeled data
→ EMA update Teacher
→ evaluate SSL gain/loss
```

### 3.2 Pseudo-BBox Generation

Trong SSOD, pseudo-label phải gồm bbox. Vì vậy Phase SSL phải kiểm tra:

- Teacher sinh prediction trên weakly augmented image.
- Pseudo-label gồm `class_id`, confidence và bbox `[x, y, w, h]`.
- Áp dụng confidence threshold trước hoặc cùng lúc với NMS.
- Áp dụng **NMS** để loại bbox trùng lặp.
- Áp dụng box quality filters:
  - box size;
  - aspect ratio;
  - boundary validity;
  - loại bbox quá nhỏ/quá lớn bất thường;
  - loại pseudo-box đáng ngờ trên ảnh No Finding nếu cần phân tích.
- Sau khi lọc, bbox phải được transform đúng sang strong view cho Student.
- Cần visualize một số mẫu để kiểm tra pseudo-box không bị lệch.

### 3.3 Anti-confirmation-bias safeguards

Confirmation bias có dạng:

```text
Teacher đoán sai
→ sinh pseudo-bbox sai
→ Student học sai
→ EMA cập nhật lại Teacher
→ lỗi được củng cố
```

Các cơ chế bắt buộc phải có hoặc phải được ghi rõ nếu không dùng:

- supervised burn-in / warm-up trước khi bật pseudo-label;
- EMA teacher;
- confidence threshold sweep;
- class-wise hoặc dynamic thresholding;
- NMS và box-quality filtering;
- theo dõi pseudo-box trên ảnh No Finding;
- theo dõi rare-class pseudo-label survival rate;
- không bật SSL quá sớm khi teacher chưa ổn định;
- log pseudo-label count theo class và theo threshold.

### 3.4 Class-wise / Dynamic Threshold

- Không mặc định dùng một threshold cố định cho cả 14 class mà không phân tích.
- Cần sweep threshold chung và/hoặc thử class-wise threshold.
- Threshold cho rare class chỉ được điều chỉnh trong phạm vi phân tích SSL/pseudo-label filtering.
- Không khóa cứng các giá trị như 0.9 hoặc 0.6 nếu chưa có validation evidence.
- Quyết định threshold cuối phải dựa trên validation / analysis split được định nghĩa trước, không dùng test set.

### 3.5 Positive/Negative mini-batch monitoring

- Không giả định ảnh normal chiếm đa số; phải kiểm tra tỷ lệ thật trong subset/split/batch.
- Cần log tỷ lệ positive/negative trong labeled batch.
- Cần log tỷ lệ positive/negative trong unlabeled batch nếu có metadata để phân tích.
- Cần theo dõi pseudo-box sinh ra trên unlabeled negative images.
- Nếu batch quá lệch, có thể dùng sampler kiểm soát nhẹ như một biện pháp ổn định training, nhưng không trình bày như đóng góp xử lý imbalance độc lập.

### 3.6 Compute / OOM fallback

Teacher–Student SSOD tốn GPU vì có labeled branch, unlabeled weak branch, unlabeled strong branch, teacher forward và student forward.

Fallback khi OOM:

- giảm batch size;
- dùng gradient accumulation;
- dùng mixed precision AMP;
- giảm image size có kiểm soát;
- giảm số unlabeled images per batch;
- chạy smoke test trước full training;
- chạy threshold sweep trên subset nhỏ trước;
- bật checkpoint/resume;
- chuyển sang Vast.ai nếu Colab/Kaggle không đủ.

---

## 4. Vai trò làm việc

- **Tôi** = người quyết định nghiên cứu.
- **GPT** = người thiết kế, phản biện và viết học thuật.
- **Claude** = người viết code trong repo.
- **Python** = công cụ chạy dữ liệu, train, evaluate và tạo bằng chứng.

Quy trình bắt buộc:

```text
script → output → DoD → GPT review → tôi tick checklist
```

Nguyên tắc:

- Không nhảy phase.
- Không train khi data/split/COCO/No Finding/seed/checkpoint criterion chưa pass DoD.
- Không kết luận khi chưa có output thật.
- Không để Claude tự đổi protocol nghiên cứu.

---

## 5. Các file điều phối chính

- `PROJECT_CONTEXT.md`: bộ nhớ trung tâm khi chuyển chat.
- `PHASE_HANDOFF.md`: trạng thái bàn giao phase hiện tại, dùng khi chuyển chat hoặc giao việc cho Claude.
- `STRUCTURE.md`: khung thesis/paper.
- `repository_structure.md`: cấu trúc repo.
- `RESEARCH_CHECKLIST.md`: checklist nghiên cứu tổng thể.
- `CHECKLIST_TRIEN_KHAI_FULL.md`: checklist triển khai theo phase.
- `CHECKLIST_TRIEN_KHAI_FULL.xlsx`: checklist trực quan có Dashboard, Checklist, Phase Summary, Lists.

Lưu ý thống nhất tên file:

- Dùng `STRUCTURE.md`, không dùng lẫn `RESEARCH_STRUCTURE.md` nếu repo đã chốt tên `STRUCTURE.md`.

---

## 6. Phase triển khai đã khóa

### PHASE 0 — Setup repo và môi trường

- Tạo repo.
- Tạo `.gitignore`.
- Tạo requirements/environment.
- Khóa seed, RNG state và deterministic flags.
- Tạo `README.md`, `CLAUDE.md`, `research_log.md`.

### PHASE 1 — Data & Medical Feasibility

- **1A:** Dataset overview.
- **1B:** Annotation quality.
- **1C:** Dataset scope decision.
- **1D:** Kappa feasibility / limitation-aware analysis.

### PHASE 2 — Data Standardization & Master Format

- **2A:** DICOM & bbox validation.
- **2B:** Canonical schema.
- **2C:** Framework & format decision.
- **2D:** COCO master conversion and validation.
- **2D.1:** Empty image loading check.
- **2E:** Fixed train/val/test split.
- **2F:** Labeled/unlabeled split for SSL.

### PHASE 3 — Pre-training Diagnostics

- **3A:** Dataset diagnostics before training.

### PHASE 4 — Supervised Baselines

- **4A.0:** Full-label supervised upper bound.
- **4A.1:** Low-label supervised baseline.
- **4B:** Attention / ViT-oriented supervised extension, optional.

### PHASE 5 — SSL Detection

- **5.1A:** Teacher–Student SSL pipeline.
- **5.1B:** Pseudo-BBox Generation.
- **5.1C:** Pseudo-label filtering with NMS and box quality filters.
- **5.1D:** Anti-confirmation-bias safeguards.
- **5.1E:** SSL main experiments.
- **5.1F:** Positive/negative mini-batch monitoring.
- **5.2 / 5.3:** Optional SSL extension, only if compute allows.

### PHASE 6 — Analysis

- Threshold sweep.
- Main evaluation.
- Error analysis.
- Negative image false positive analysis.
- Pseudo-label bias analysis.
- Seed stability analysis.
- Failure/fallback analysis.

### PHASE 7 — Contribution & Paper

- Final reports.
- Thesis/paper writing.
- Paper framing.
- Figures/tables.
- Limitations and claim guardrails.

---

## 7. Trạng thái hiện tại

**Current phase:** Phase 1A — Dataset Overview  
**Previous phase:** Phase 0 — Setup repo và môi trường  
**Git status:** Phase 0 core đã commit và push lên GitHub.

### 7.1 Phase 0 — Core setup & reproducibility

Status: **DONE / CORE PASS**

Date: 2026-06-18

Commit:

```text
b5127fd phase0: setup reproducible core environment
```

Remote:

```text
origin/main — https://github.com/kelvinquach/ssl_detection_xray_v2.git
```

Scripts run:

```cmd
python scripts/00_check_environment.py --seed 2026 --output reports/phase0_environment_check.json --seed-manifest data/manifests/seed_state_manifest.json --freeze-output reports/phase0_pip_freeze.txt
python -m pytest tests	est_phase0.py -q
pip check
```

Outputs generated:

- `reports/phase0_environment_check.json`
- `reports/phase0_pip_freeze.txt`
- `reports/reproducibility_settings.md`
- `data/manifests/seed_state_manifest.json`

DoD result:

- Repo structure: **PASS**
- Core Python environment: **PASS**
- Dependency check: **PASS**
- Seed/RNG/deterministic settings: **PASS**
- Checkpoint/evaluation policy: **PASS**
- Phase 0 tests: **PASS** (`5 passed`)
- Local MMDetection/GPU training environment: **DEFERRED**

Key findings:

- Python: `3.10.20`
- Conda env: `sslxray`
- Platform: `Windows-10-10.0.26200-SP0`
- PyTorch: `2.3.1`
- torchvision: `0.18.1`
- numpy: `1.24.3`
- pandas: `2.3.3`
- OpenCV/cv2: `4.11.0`
- pydicom: `3.0.2`
- pycocotools: `2.0.11`
- `pip check`: `No broken requirements found.`
- CUDA available: `False`
- MMDetection stack: not installed locally

Research decisions:

- Local Windows CPU-only environment is accepted for repo/data/script/report validation.
- Local environment is **not** accepted as training-ready.
- MMDetection/GPU setup will be done separately in a remote/GPU environment.
- Phase 1A is opened.
- Split, COCO conversion, training, pseudo-labeling, threshold tuning and test-set usage remain locked.

Issues / risks:

- Local machine has no CUDA.
- `mmengine/mmcv/mmdet` are not installed locally.
- Do not accidentally interpret Phase 0 core pass as training environment pass.

### 7.2 Current gate

```text
Phase 0 core: PASS
Phase 0 committed: PASS
Phase 0 pushed to GitHub: PASS
Phase 0 training framework: DEFERRED
Phase 1A: OPEN
Split: LOCKED
COCO conversion: LOCKED
Training: LOCKED
Pseudo-labeling: LOCKED
Threshold tuning: LOCKED
Test-set usage: LOCKED
```

### 7.3 Phase hiện tại cần làm

Phase 1A — Dataset Overview.

Mục tiêu:

- đọc annotation CSV;
- thống kê số dòng annotation;
- thống kê số unique image_id;
- thống kê class distribution;
- kiểm tra sơ bộ bbox validity;
- kiểm tra No Finding policy;
- tạo evidence report.

Không được làm trong Phase 1A:

- không split;
- không convert COCO;
- không copy ảnh;
- không đọc DICOM/PNG;
- không train;
- không pseudo-label;
- không tune threshold;
- không dùng test set.

### Phase 1A — Dataset Overview

Status: PASS

Phase 1A was performed on the full VinBigData `train.csv` source metadata.

Key findings:

```text
Total source images: 15,000
Abnormal images: 4,394
No Finding images: 10,606
Annotation rows: 67,914
Abnormal bbox rows: 36,096
Abnormal detection classes excluding No Finding: 14
Invalid bbox count: 0
No Finding rows with bbox: 0
Images with both No Finding and abnormal labels: 0
```

Research decision:

```text
Downstream controlled working scope is locked to 4,894 images:
4,394 abnormal images + 500 No Finding images.
```

Important note:

```text
The full 15,000-image CSV is source metadata only.
It must not be confused with the downstream controlled working subset.
The 4,894-image subset has not been constructed in Phase 1A.
```

Next phase:

```text
Phase 1B — Annotation Quality
```


**Phase tiếp theo dự kiến sau khi Phase 1A pass DoD:**  
Phase 1B — Annotation Quality.

---


## 8. Nguyên tắc review bắt buộc

Khi tôi đưa code/log/output, GPT phải kiểm tra:

1. Output đã đủ chưa?
2. Có đạt Definition of Done chưa?
3. Có lỗi logic nghiên cứu không?
4. Có rủi ro leakage không?
5. Có sai split/seed/metric không?
6. Có xử lý đúng No Finding không?
7. Có pseudo-bbox generation đúng bản chất SSOD không?
8. Có NMS và box quality filtering không?
9. Có rủi ro confirmation bias không?
10. Có rủi ro threshold làm rare class biến mất không?
11. Có test set bị dùng để tune không?
12. Có được tick checklist chưa?
13. Nếu chưa đạt, viết prompt sửa lỗi cho Claude.

---

## 9. Những lỗi nguy hiểm cần luôn nhắc

### 9.1 Data / COCO / No Finding

- No Finding bị đưa nhầm thành detection class.
- Ảnh No Finding bị framework lọc khỏi dataloader.
- Ảnh No Finding không nằm trong test set nên không đo được FP/negative.
- Bbox nhầm `xyxy` / `xywh`.
- Bbox bị lệch sau resize, flip, crop, scale.
- Split leakage.
- Unlabeled vô tình dùng ground truth.

### 9.2 Training / SSL

- Supervised và SSL không cùng labeled split.
- So sánh SSL vs supervised nhưng không cùng `split_seed`.
- Khóa nhầm `training_seed` giống nhau cho mọi run làm variance giả thấp.
- Chỉ chạy 1 seed rồi kết luận.
- Checkpoint chọn bằng test set.
- Threshold tune bằng test set.
- SSL gain nhỏ hơn std nhưng vẫn over-claim.
- Teacher bật pseudo-label quá sớm, gây confirmation bias.
- Không có burn-in.
- Không log λ của unlabeled loss.

### 9.3 SSOD-specific

- Pseudo-label chỉ ghi class/confidence mà quên bbox.
- Không dùng NMS trước khi lấy pseudo-bbox.
- Quá nhiều bbox trùng lặp làm student học nhiễu.
- Không transform bbox từ weak view sang strong view.
- Không kiểm tra box quality.
- Threshold cao làm rare class biến mất.
- Threshold thấp làm ảnh negative sinh nhiều pseudo-box sai.
- Không theo dõi pseudo-box trên unlabeled negative images.
- Không có fallback khi SSL không cải thiện.

### 9.4 Compute

- Teacher–Student OOM do batch quá lớn.
- Không có AMP / gradient accumulation / checkpoint resume.
- Chạy full training trước khi smoke test.
- Không ghi compute budget, GPU, VRAM, thời gian train.

---

## 10. Quy tắc kết luận kết quả

- SSL gain/loss phải tính bằng metric chính `mAP@0.5:0.95`.
- SSL gain phải so với supervised baseline cùng labeled split và cùng `split_seed`.
- SSL gain phải báo cáo kèm mean ± std theo nhiều `training_seed`.
- Nếu SSL gain nhỏ hơn hoặc tương đương std giữa các seed, chỉ nói: “chưa đủ bằng chứng ổn định”.
- Nếu SSL tăng mAP nhưng FP per negative image tăng mạnh, phải thảo luận trade-off y khoa.
- Nếu threshold làm rare-class survival rate giảm mạnh, không được chỉ báo cáo mAP tổng.
- Nếu SSL không cải thiện, phải phân tích failure mode trước khi thay đổi method.

---

## 11. Quy tắc cập nhật PROJECT_CONTEXT.md sau mỗi phase

Sau mỗi phase pass DoD, cập nhật ngắn gọn:

```md
### Phase X — Tên phase
Status: DONE / IN PROGRESS / BLOCKED
Date:
Scripts run:
Outputs generated:
DoD result:
Key findings:
Research decisions:
Issues / risks:
Next phase:
```

Không nhét toàn bộ log vào `PROJECT_CONTEXT.md`.  
Log chi tiết để trong `reports/`, `logs/`, `research_log.md` hoặc checklist Excel.

---

## 12. Prompt mở chat mới

Dán đoạn này khi mở chat mới:

```text
Tôi đang tiếp tục đề tài đã khóa:
“Nghiên cứu học bán giám sát cho dò tìm bất thường trên X-quang phổi”.

Hãy đọc PROJECT_CONTEXT.md trước.
Làm việc theo quy trình:
script → output → DoD → GPT review → tôi tick checklist.

Không nhảy phase.
Không train khi data/split/COCO/No Finding/seed/checkpoint criterion chưa pass DoD.
Nếu cần code, hãy viết prompt rõ ràng để tôi giao cho Claude.
Nếu tôi đưa output/log, hãy review theo DoD và chỉ ra lỗi logic nếu có.
```

---

## 13. Phase Progress Log

### Phase 0 — Setup repo và môi trường

Status: **DONE / CORE PASS**

Date: 2026-06-18

Scripts run:

```cmd
python scripts/00_check_environment.py --seed 2026 --output reports/phase0_environment_check.json --seed-manifest data/manifests/seed_state_manifest.json --freeze-output reports/phase0_pip_freeze.txt
python -m pytest tests	est_phase0.py -q
pip check
```

Outputs generated:

- `reports/phase0_environment_check.json`
- `reports/phase0_pip_freeze.txt`
- `reports/reproducibility_settings.md`
- `data/manifests/seed_state_manifest.json`
- `configs/protocol/checkpoint_policy.yaml`
- `research_log.md`
- `PHASE_HANDOFF.md`

DoD result:

- Repo structure: **PASS**
- Core environment: **PASS**
- Reproducibility evidence: **PASS**
- Checkpoint/evaluation policy: **PASS**
- Tests: **PASS**
- Local MMDetection/GPU training-ready: **DEFERRED**

Key findings:

- `core_import_ok = true`
- `framework_import_ok = false`
- `cuda_available = false`
- `pip check = No broken requirements found`
- `pytest = 5 passed`

Research decisions:

- Local environment is for validation/reporting only.
- Training-related setup will be done on remote/GPU.
- Test set policy is locked: final evaluation only.
- Primary metric is locked: `mAP@0.5:0.95`.

Issues / risks:

- Local environment cannot be used for detector training.
- MMDetection stack is not installed locally.
- Must not start split/COCO/training before their phase DoD.

Git:

```text
commit: b5127fd phase0: setup reproducible core environment
branch: main
remote: origin/main
```

---

### Phase 1A — Dataset Overview

Status: **OPEN / NEXT**

Current focus:

- tạo `scripts/01A_dataset_overview.py`;
- đọc annotation CSV;
- thống kê dataset overview;
- kiểm tra No Finding policy;
- kiểm tra bbox validity sơ bộ;
- tạo evidence report.

Expected script:

```text
scripts/01A_dataset_overview.py
```

Expected outputs:

- `reports/phase1A_dataset_overview.json`
- `reports/phase1A_dataset_overview.md`
- `reports/phase1A_class_distribution.csv`
- `reports/phase1A_image_level_summary.csv`
- `reports/phase1A_bbox_quality_summary.csv`

DoD:

- Có total rows.
- Có unique images.
- Có No Finding images.
- Có abnormal images.
- Có abnormal classes excluding No Finding.
- Có bbox missing/invalid summary.
- Có warning nếu No Finding có bbox.
- Có warning nếu abnormal class thiếu bbox.
- Không split.
- Không COCO.
- Không train.
- GPT review pass.

Next phase after pass:

- Phase 1B — Annotation Quality.

---
