# STRUCTURE.md — Giải thích cấu trúc repo

```
ssl_detection_xray_v2/
├── configs/                  # File cấu hình thí nghiệm
│   └── protocol/             #   Giao thức cố định (đánh giá, checkpoint)
│       └── checkpoint_policy.yaml
├── data/                     # Dữ liệu (nội dung nặng bị .gitignore)
│   ├── raw/                  #   Dữ liệu gốc (KHÔNG commit)
│   ├── interim/              #   Trung gian (KHÔNG commit)
│   ├── processed/            #   Đã xử lý
│   │   └── images/           #     Ảnh đã xử lý (KHÔNG commit)
│   └── manifests/            #   Manifest/provenance (seed, split sau này)
├── src/                      # Mã nguồn
│   └── utils/                #   Tiện ích tái lập & môi trường
│       ├── seed.py           #     set_global_seed, manifest RNG
│       └── env.py            #     thu thập env, probe import, CUDA
├── scripts/                  # Script chạy được
│   ├── 00_setup_environment.sh
│   └── 00_check_environment.py
├── experiments/              # Output thí nghiệm (checkpoints bị .gitignore)
├── reports/                  # Báo cáo & evidence (JSON, pip freeze)
├── plots/                    # Hình vẽ / biểu đồ
├── models/                   # Trọng số (KHÔNG commit)
├── logs/                     # Log chạy (KHÔNG commit)
├── tests/                    # Unit tests
├── draft/                    # Bản nháp tạm
├── README.md
├── CLAUDE.md                 # Hướng dẫn cho coding assistant
├── STRUCTURE.md              # (file này)
├── RESEARCH_CHECKLIST.md     # Checklist theo phase
├── repository_structure.md   # Cây thư mục dạng tham chiếu nhanh
├── research_log.md           # Nhật ký nghiên cứu
├── requirements.txt
├── environment.yml
└── .gitignore
```

## Nguyên tắc đặt artifact

- **Evidence Phase 0** → `reports/` (vd `phase0_environment_check.json`, `phase0_pip_freeze.txt`).
- **Manifest tái lập** → `data/manifests/` (vd `seed_state_manifest.json`).
- **Cấu hình cố định** → `configs/protocol/`.
- **Mã tái sử dụng** → `src/`, script chạy tay → `scripts/`.
