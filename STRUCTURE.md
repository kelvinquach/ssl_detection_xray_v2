# STRUCTURE.md — Cấu trúc thực tế của repository

Tài liệu này mô tả cấu trúc đã được triển khai và xác nhận đến khi đóng
`Phase 2D.1B-Full` ngày 2026-07-29. Cấu trúc của các phase sau không được liệt
kê như artifact hiện hữu khi chúng chưa bắt đầu.

```text
ssl_detection_xray_v2/
├── configs/
│   ├── dataset/
│   │   └── coco_paths.yaml
│   ├── framework/
│   │   └── main_framework.yaml
│   └── protocol/
│       ├── checkpoint_policy.yaml
│       ├── coco_conversion_policy.yaml
│       ├── phase2D_coco_master_validation.yaml
│       └── phase2D1_jpg_representation.yaml
│
├── data/
│   ├── raw/
│   │   └── vinbigdata/
│   │       └── dicom_subset/
│   │           └── train/
│   │               └── <image_id>.dicom
│   ├── interim/
│   │   └── vinbigdata_phase1C_scope_annotations.csv
│   ├── manifests/
│   │   ├── seed_state_manifest.json
│   │   ├── phase1C_selected_images_manifest.csv
│   │   ├── phase1C_downloaded_image_inventory.csv
│   │   └── phase1C_combined_package_manifest.csv
│   └── processed/
│       ├── canonical/
│       │   ├── canonical_image_table.csv
│       │   ├── canonical_bbox_table.csv
│       │   └── canonical_class_mapping.csv
│       ├── coco/
│       │   ├── coco_master.json
│       │   └── coco_master_jpg.json
│       ├── images_jpg/
│       │   └── train/
│       │       └── <image_id>.jpg
│       └── images_jpg_pilot/
│           ├── reference_uint8/
│           ├── q95/
│           └── q100/
│
├── src/
│   └── utils/
│       ├── seed.py
│       └── env.py
│
├── scripts/
│   ├── 00_setup_environment.sh
│   ├── 00_check_environment.py
│   ├── 01A_dataset_overview.py
│   ├── 01B_annotation_quality.py
│   ├── 01C_dataset_scope_decision.py
│   ├── 01D_kappa_feasibility.py
│   ├── 02A_dicom_bbox_boundary_validation.py
│   ├── 02B_build_canonical_schema.py
│   ├── 02C_framework_format_decision.py
│   ├── 02D_build_coco_master.py
│   ├── 02D1A_image_representation_protocol.py
│   ├── 02D1B_pilot_dicom_to_jpg.py
│   └── 02D1B_full_dicom_to_jpg.py
│
├── tests/
│   ├── test_phase0.py
│   ├── test_phase2D_guardrails.py
│   ├── test_phase2D1A_protocol_guardrails.py
│   ├── test_phase2D1B_pilot_guardrails.py
│   └── test_phase2D1B_full_guardrails.py
│
├── reports/
│   ├── phase0_*
│   ├── phase1A_*
│   ├── phase1B_*
│   ├── phase1C_*
│   ├── phase1D_*
│   ├── phase2A_*
│   ├── phase2B_*
│   ├── phase2C_*
│   ├── phase2D_coco_*
│   ├── phase2D1B_pilot_*
│   └── phase2D1B_full_*
│
├── docs/
├── experiments/
├── plots/
├── models/
├── logs/
├── draft/
│
├── README.md
├── PROJECT_CONTEXT.md
├── PHASE_HANDOFF.md
├── STRUCTURE.md
├── repository_structure.md
├── RESEARCH_CHECKLIST.md
├── CHECKLIST_TRIEN_KHAI_FULL.xlsx
├── research_log.md
├── requirements.txt
├── environment.yml
├── CLAUDE.md
└── .gitignore
```

## Vai trò của các nhóm artifact

- `data/raw/`: nguồn DICOM bất biến; không chỉnh sửa và không commit dữ liệu ảnh.
- `data/interim/`: dữ liệu trung gian của controlled scope.
- `data/manifests/`: inventory, provenance và bằng chứng tái lập.
- `data/processed/canonical/`: ba bảng canonical được khóa ở Phase 2B.
- `data/processed/coco/coco_master.json`: annotation master chính thức.
- `data/processed/coco/coco_master_jpg.json`: derivative chỉ đổi đường dẫn sang
  JPG; không thay thế `coco_master.json`.
- `data/processed/images_jpg/train/`: 4.894 JPG quality 95 của controlled scope;
  không commit vào ordinary Git.
- `data/processed/images_jpg_pilot/`: đầu ra pilot phục vụ so sánh
  `reference_uint8`, quality 95 và quality 100.
- `configs/`: cấu hình dataset, framework và protocol đã được khóa.
- `scripts/`: entry point thực thi theo từng phase.
- `tests/`: guardrail tests; không phải detector-training tests.
- `reports/`: evidence, audit và validation; các hậu tố `*` trong cây trên biểu
  diễn nhóm nhiều file `.md`, `.json`, `.csv`, `.jsonl` hoặc `.txt`.
- `src/`: mã tiện ích có thể tái sử dụng; không chứa artifact dữ liệu.
- `experiments/`, `models/`, `logs/`: đầu ra huấn luyện trong tương lai; artifact
  nặng phải tuân theo `.gitignore`.
- `docs/` và `draft/`: tài liệu luận văn và bản nháp, không phải nguồn dữ liệu
  huấn luyện.

## Artifact chính của Phase 2D.1B-Full

```text
scripts/02D1B_full_dicom_to_jpg.py
tests/test_phase2D1B_full_guardrails.py

reports/phase2D1B_full_preflight.json
reports/phase2D1B_full_preflight.md
reports/phase2D1B_full_validation.json
reports/phase2D1B_full_validation.md
reports/phase2D1B_full_promotion.json
reports/phase2D1B_full_cleanup_audit.json
reports/phase2D1B_full_metadata_audit.csv
reports/phase2D1B_full_bbox_audit.csv
reports/phase2D1B_full_no_finding_audit.csv
reports/phase2D1B_full_errors.csv
reports/phase2D1B_full_mapping.csv
reports/phase2D1B_full_mapping.jsonl

data/processed/coco/coco_master_jpg.json
data/processed/images_jpg/train/<image_id>.jpg
```

## Quy tắc quản lý

1. Không commit DICOM, 4.894 JPG, pilot images, model weights, checkpoints hoặc
   training logs nặng vào ordinary Git.
2. Không sửa trực tiếp `coco_master.json`; mọi representation-specific file phải
   là derivative có traceability rõ ràng.
3. Không đặt report/audit vào `data/processed/`; evidence thuộc `reports/`.
4. Không đặt split hoặc seed manifest vào `reports/`; provenance thuộc
   `data/manifests/`.
5. Không thêm artifact Phase 2D.1C vào cây “đã triển khai” trước khi MMDetection
   loading và empty-image retention thực sự được chạy và review.
6. Tại trạng thái hiện hành, Phase 2D.1C là `NOT STARTED / NEXT`;
   `dataset_training_ready=false` và `training_authorized=false`.
