# Nhật ký nghiên cứu `ssl_detection_xray_v2`

Ghi theo thứ tự thời gian. Mỗi entry cần có: mục tiêu, việc đã làm, evidence, kết quả review, quyết định tiếp theo.

---

## 2026-06-18 — PHASE 0: Khởi tạo repo & môi trường

### Mục tiêu

Dựng cấu trúc repo, tài liệu, môi trường Python cơ bản và utility tái lập cho đề tài:

**“Nghiên cứu học bán giám sát cho dò tìm bất thường trên X-quang phổi.”**

Trọng tâm: semi-supervised object detection trên VinBigData Chest X-ray.

Phase 0 chỉ phục vụ setup và reproducibility.

Không đọc dataset, không kiểm tra annotation, không convert COCO, không tạo split, không train.

---

### Đã làm

#### 1. Tạo cấu trúc repo

Đã tạo các thư mục chính:

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

#### 2. Tạo tài liệu dự án

Đã tạo:

* `README.md`
* `CLAUDE.md`
* `STRUCTURE.md`
* `RESEARCH_CHECKLIST.md`
* `repository_structure.md`
* `research_log.md`

#### 3. Tạo file môi trường

Đã tạo:

* `requirements.txt`
* `requirements_phase0_base.txt`
* `environment.yml`
* `scripts/00_setup_environment.sh`

Framework chính được định hướng là **MMDetection**.

Detectron2 chỉ được ghi nhận là optional, không dùng làm framework mặc định.

#### 4. Tạo utility tái lập

Đã tạo:

* `src/utils/seed.py`
* `src/utils/env.py`

`src/utils/seed.py` hỗ trợ:

* `set_global_seed()`
* `get_rng_state_summary()`
* `save_seed_manifest()`

`src/utils/env.py` hỗ trợ:

* ghi nhận Python/platform
* kiểm tra import package
* kiểm tra CUDA
* ghi nhận trạng thái framework detection

#### 5. Tạo script kiểm tra môi trường

Đã tạo và chạy:

* `scripts/00_check_environment.py`

Script có nhiệm vụ:

* set seed mặc định `2026`
* ghi environment report
* ghi seed manifest
* ghi pip freeze
* không crash nếu `mmengine/mmcv/mmdet` chưa cài
* ghi rõ `framework_import_ok: false` nếu MMDetection stack chưa import được

#### 6. Khóa protocol checkpoint/evaluation

Đã tạo:

* `configs/protocol/checkpoint_policy.yaml`

Nội dung protocol đã khóa:

* primary metric: `mAP@0.5:0.95`
* checkpoint selection split: `val`
* test set chỉ dùng cho final evaluation
* không dùng test set để tune threshold
* không dùng test set để chọn checkpoint
* không dùng test set để chọn model/backbone
* không dùng test set để quyết định augmentation

#### 7. Tạo `.gitignore`

`.gitignore` đã loại trừ dữ liệu nặng, ảnh, checkpoint, logs, virtual environment và cache.

---

### Evidence đã tạo

Đã chạy lệnh:

```cmd
python scripts/00_check_environment.py --seed 2026 --output reports/phase0_environment_check.json --seed-manifest data/manifests/seed_state_manifest.json --freeze-output reports/phase0_pip_freeze.txt
```

Các file evidence đã sinh:

* `reports/phase0_environment_check.json`
* `reports/phase0_pip_freeze.txt`
* `data/manifests/seed_state_manifest.json`

Đã kiểm tra dependency bằng:

```cmd
pip check
```

Kết quả:

```text
No broken requirements found.
```

---

### Kết quả environment check

Kết quả chính từ `reports/phase0_environment_check.json`:

* Python: `3.10.20`
* Platform: `Windows-10-10.0.26200-SP0`
* Conda environment: `sslxray`
* Python executable: `C:\Users\USER\anaconda3\envs\sslxray\python.exe`
* Seed: `2026`

Core imports:

* `torch`: OK, version `2.3.1`
* `torchvision`: OK, version `0.18.1`
* `numpy`: OK, version `1.24.3`
* `pandas`: OK, version `2.3.3`
* `cv2`: OK, version `4.11.0`
* `pydicom`: OK, version `3.0.2`
* `pycocotools`: OK, version `2.0.11`

Detection framework imports:

* `mmengine`: FAIL / not installed
* `mmcv`: FAIL / not installed
* `mmdet`: FAIL / not installed

CUDA:

* `torch.cuda.is_available()`: `False`
* `torch.version.cuda`: `null`
* GPU device count: `0`

Summary:

* `core_import_ok`: `true`
* `framework_import_ok`: `false`
* primary framework: `mmdetection`
* detectron2: `optional`

---

### Seed và deterministic settings

Seed manifest đã được tạo tại:

* `data/manifests/seed_state_manifest.json`

Seed settings:

* Global seed: `2026`
* `PYTHONHASHSEED`: `2026`
* Python random seed: enabled
* NumPy seed: enabled
* PyTorch CPU seed: enabled
* PyTorch CUDA seed: not applied because CUDA is unavailable

Deterministic flags:

* `torch.use_deterministic_algorithms`: `true`
* `torch.backends.cudnn.deterministic`: `true`
* `torch.backends.cudnn.benchmark`: `false`
* `CUBLAS_WORKSPACE_CONFIG`: `:4096:8`

---

### Vấn đề đã gặp

Trong quá trình thử cài OpenMIM/MMDetection local, `openmim` kéo thêm nhiều dependency phụ và làm môi trường bị lệch nhẹ. Sau đó đã gỡ `openmim` và repair môi trường.

Sau khi repair:

```cmd
pip check
```

cho kết quả:

```text
No broken requirements found.
```

Quyết định: không ép cài MMDetection trên Windows CPU-only local ở Phase 0.

---

### Review GPT

Phase 0A — Repository structure: **PASS**

Phase 0B — Core Python environment: **PASS**

Phase 0B — Local training framework: **DEFERRED**

Lý do:

* Core packages import được.
* Seed và deterministic settings đã được ghi nhận.
* Environment report, pip freeze và seed manifest đã được sinh.
* Local CUDA không khả dụng.
* MMDetection stack chưa import được.
* Local environment chưa được xem là training-ready.

---

### Quyết định

Local environment được chấp nhận cho:

* kiểm tra repo
* kiểm tra script
* kiểm tra metadata
* kiểm tra annotation
* tạo report
* tạo split sau khi Phase 1/2 pass DoD
* kiểm tra COCO format sau khi đến đúng phase

Local environment không được dùng cho:

* detector training
* checkpoint selection
* SSL pseudo-label training
* final evaluation
* threshold tuning
* model/backbone selection

MMDetection/GPU training environment sẽ được setup riêng ở môi trường remote/GPU sau.

---

### Ràng buộc tuân thủ

Trong Phase 0 đã tuân thủ:

* Không đọc dataset.
* Không đọc `train.csv`.
* Không convert COCO.
* Không tạo split.
* Không train.
* Không pseudo-label.
* Không tune threshold.
* Không dùng test set.

---

### Trạng thái checklist

Được tick:

* Phase 0A repo structure
* Phase 0B core environment
* pip dependency check
* PyTorch/torchvision import
* numpy/pandas/cv2/pydicom/pycocotools import
* seed manifest
* deterministic flags
* environment report
* pip freeze
* checkpoint policy

Chưa tick:

* MMDetection import OK
* `mmengine` import OK
* `mmcv` import OK
* `mmdet` import OK
* CUDA/GPU ready
* Local training-ready environment
* Full detection framework setup

---

### Việc cần làm tiếp trước khi mở Phase 1

Cần bổ sung:

* `reports/reproducibility_settings.md`

Cần chạy:

```cmd
python -m pytest tests\test_phase0.py -q
```

Sau khi `reproducibility_settings.md` tồn tại và test Phase 0 pass, có thể đóng **Phase 0 core** và xin review để mở **Phase 1A — Data Overview**.

---
## 2026-06-19 — PHASE 1A: Khởi tạo repo & môi trường

### Dataset scope clarification after Phase 1A

Phase 1A dataset overview was performed on the full VinBigData `train.csv` source metadata.

Observed source metadata:

```text
Total images: 15,000
Abnormal images: 4,394
No Finding images: 10,606
Annotation rows: 67,914
Abnormal bbox rows: 36,096
Abnormal detection classes: 14
```

Research decision:

```text
Downstream controlled working scope is locked to 4,894 images:
- 4,394 abnormal images
- 500 No Finding images
```

Important distinction:

```text
The full 15,000-image CSV is source metadata only.
It is not the downstream working dataset.
The 4,894-image subset has not been constructed in Phase 1A.
Subset manifest creation belongs to a later dataset scope / manifest phase.
```

Scope lock:

```text
No split was created in Phase 1A.
No COCO conversion was performed in Phase 1A.
No image files were copied or read in Phase 1A.
No training, pseudo-labeling, or threshold tuning was performed in Phase 1A.
```

### Phase 1B — Annotation Quality (kiểm tra chất lượng Annotation).

Status: PASS

Date: 2026-06-19

Scripts run:

```cmd
python scripts\01B_annotation_quality.py --train-csv data\raw\vinbigdata\annotations\train.csv

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

Next phase:

Phase 1C — Dataset Scope Decision


Lệnh commit:

```cmd
git status
git add scripts/01B_annotation_quality.py reports/phase1B_annotation_quality.json reports/phase1B_annotation_quality.md reports/annotation_sanity_report.md reports/invalid_bbox_rows.csv reports/duplicate_bbox_candidates.csv reports/phase1B_class_mapping.csv reports/phase1B_bbox_quality_by_class.csv reports/phase1B_image_label_consistency.csv PROJECT_CONTEXT.md research_log.md CHECKLIST_TRIEN_KHAI_FULL.xlsx
git commit -m "phase1B: validate annotation quality"
git push