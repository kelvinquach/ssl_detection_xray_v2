#!/usr/bin/env bash
# scripts/00_setup_environment.sh
# Phase 0 — set up the Python environment for ssl_detection_xray_v2.
# Primary detection framework: MMDetection. Detectron2 is OPTIONAL.
#
# This script does NOT touch any dataset.
#
# Usage:
#   bash scripts/00_setup_environment.sh [--cuda cu121|cpu]
#
set -euo pipefail

CUDA_TAG="cu121"
for arg in "$@"; do
  case "$arg" in
    --cuda) shift ;;
    cu118|cu121|cu124|cpu) CUDA_TAG="$arg" ;;
  esac
done

echo "=========================================================="
echo " ssl_detection_xray_v2 — Phase 0 environment setup"
echo " Primary framework: MMDetection (Detectron2 optional)"
echo " CUDA tag: ${CUDA_TAG}"
echo "=========================================================="

# 1) Upgrade pip tooling.
python -m pip install --upgrade pip setuptools wheel

# 2) Install PyTorch + torchvision matched to CUDA tag.
if [ "${CUDA_TAG}" = "cpu" ]; then
  python -m pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu
else
  python -m pip install torch torchvision \
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
fi

# 3) Install base requirements (excluding mmcv/mmdet which go via mim).
python -m pip install \
  "numpy>=1.24,<2.0" "pandas>=2.0" "scipy>=1.10" \
  "opencv-python>=4.8" "pydicom>=2.4" "Pillow>=10.0" \
  "pycocotools>=2.0.7" "matplotlib>=3.7" "seaborn>=0.13" \
  "tqdm>=4.66" "PyYAML>=6.0" "rich>=13.0" "pytest>=7.4"

# 4) Install the MMDetection stack via mim (handles torch/CUDA matching).
python -m pip install -U openmim
python -m mim install mmengine
python -m mim install "mmcv>=2.1"
python -m mim install "mmdet>=3.3"

# 5) OPTIONAL: Detectron2 is NOT installed here.
#    It is error-prone to build on Windows. To use it, install manually:
#      python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# 6) Run the environment check to produce Phase 0 evidence.
echo ""
echo "Running environment check..."
python scripts/00_check_environment.py

echo ""
echo "Done. See reports/phase0_environment_check.json for results."
