# repository_structure.md

Tham chiếu nhanh cây thư mục (mức 3). Cập nhật khi cấu trúc thay đổi.

```
ssl_detection_xray_v2/
├── configs/
│   └── protocol/
│       └── checkpoint_policy.yaml
├── data/
│   ├── raw/                 (.gitkeep)
│   ├── interim/             (.gitkeep)
│   ├── processed/
│   │   └── images/
│   └── manifests/           (.gitkeep)
├── src/
│   ├── __init__.py
│   └── utils/
│       ├── __init__.py
│       ├── seed.py
│       └── env.py
├── scripts/
│   ├── 00_setup_environment.sh
│   └── 00_check_environment.py
├── experiments/             (.gitkeep)
├── reports/                 (.gitkeep)
├── plots/                   (.gitkeep)
├── models/                  (.gitkeep)
├── logs/                    (.gitkeep)
├── tests/
├── draft/                   (.gitkeep)
├── README.md
├── CLAUDE.md
├── STRUCTURE.md
├── RESEARCH_CHECKLIST.md
├── repository_structure.md
├── research_log.md
├── requirements.txt
├── environment.yml
└── .gitignore
```

## Artifact tạo ra khi chạy Phase 0

```
reports/phase0_environment_check.json
reports/phase0_pip_freeze.txt
data/manifests/seed_state_manifest.json
```
