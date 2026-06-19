# ssl_detection_xray_v2

Semi-supervised object detection for anomaly detection on chest X-rays.

**Đề tài:** Nghiên cứu học bán giám sát cho dò tìm bất thường trên X-quang phổi.
**Trọng tâm:** Semi-supervised object detection trên **VinBigData Chest X-ray**.
**Framework chính:** [MMDetection](https://github.com/open-mmlab/mmdetection) (OpenMMLab). Detectron2 là *optional*.

> **Trạng thái hiện tại: PHASE 1B — Annotation Quality**
> Phase 0: Cài đặt - Kiểm tra môi trường. *không* đọc dataset, *không* convert COCO, *không* tạo split, *không* train.
> Phase 1A: dataset overview

## Vai trò trong dự án

| Vai trò | Trách nhiệm |
|---|---|
| Người dùng | Quyết định nghiên cứu |
| GPT | Thiết kế / review logic |
| Claude | Viết code trong repo |
| Python | Chạy script, tạo bằng chứng (evidence) |

## Cài đặt (Phase 0)

```bash
# Cách A: conda
conda env create -f environment.yml
conda activate ssl_xray
bash scripts/00_setup_environment.sh

# Cách B: pip thuần (trong venv)
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
bash scripts/00_setup_environment.sh --cuda cu121   # hoặc cpu
```

## Kiểm tra môi trường (tạo evidence)

```bash
python scripts/00_check_environment.py \
    --seed 2026 \
    --output reports/phase0_environment_check.json \
    --seed-manifest data/manifests/seed_state_manifest.json \
    --freeze-output reports/phase0_pip_freeze.txt
```

Script này:
- set global seed (Python / NumPy / PyTorch CPU+CUDA) + ghi determinism flags;
- thu thập báo cáo môi trường (versions, CUDA, GPU);
- ghi `seed_state_manifest.json`;
- ghi `pip freeze`;
- **không crash** nếu mmdet/mmcv/mmengine chưa cài — chỉ ghi `import_ok: false`.

## Cấu trúc repo

Xem [`STRUCTURE.md`](STRUCTURE.md) và [`repository_structure.md`](repository_structure.md).

## Giao thức đánh giá

Xem [`configs/protocol/checkpoint_policy.yaml`](configs/protocol/checkpoint_policy.yaml).
Tóm tắt: metric chính `mAP@0.5:0.95`; chọn checkpoint trên **val**; **test chỉ dùng một lần** cho đánh giá cuối.

## Hướng dẫn cho Claude

Xem [`CLAUDE.md`](CLAUDE.md).
