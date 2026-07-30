#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-/content/ssl_detection_xray_v2}"
PY="/content/miniconda/envs/mmdet330/bin/python"

test -x "${PY}" || {
    echo "ERROR: Python executable not found: ${PY}" >&2
    exit 1
}

export MPLBACKEND=Agg

"${PY}" - <<'PY'
import os
import sys

assert os.environ["MPLBACKEND"] == "Agg"

import cv2
import matplotlib
import mmcv
import mmengine
import mmdet
import numpy
import PIL
import torch
import torchvision

print("Python:", sys.version)
print("Python executable:", sys.executable)
print("PyTorch:", torch.__version__)
print("TorchVision:", torchvision.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print(
    "GPU:",
    torch.cuda.get_device_name(0)
    if torch.cuda.is_available()
    else None,
)
print("MMCV:", mmcv.__version__)
print("MMEngine:", mmengine.__version__)
print("MMDetection:", mmdet.__version__)
print("NumPy:", numpy.__version__)
print("OpenCV:", cv2.__version__)
print("Pillow:", PIL.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Matplotlib backend:", matplotlib.get_backend())

assert sys.version_info[:2] == (3, 10)
assert sys.executable == "/content/miniconda/envs/mmdet330/bin/python"
assert torch.__version__.startswith("2.1.0")
assert torchvision.__version__.startswith("0.16.0")
assert torch.version.cuda == "11.8"
assert torch.cuda.is_available()
assert mmcv.__version__ == "2.1.0"
assert mmengine.__version__ == "0.10.7"
assert mmdet.__version__ == "3.3.0"
assert numpy.__version__ == "1.26.4"
assert cv2.__version__ == "4.10.0"
assert matplotlib.get_backend().lower() == "agg"

from mmcv.ops import nms, roi_align
from mmdet.apis import init_detector
from mmdet.datasets import CocoDataset

boxes = torch.tensor(
    [
        [0.0, 0.0, 10.0, 10.0],
        [1.0, 1.0, 9.0, 9.0],
    ],
    device="cuda",
)
scores = torch.tensor([0.9, 0.8], device="cuda")

kept_boxes, kept_indices = nms(
    boxes,
    scores,
    iou_threshold=0.5,
)

assert kept_boxes.is_cuda
assert kept_indices.is_cuda
assert kept_indices.numel() == 1

print("MMCV CUDA NMS device:", kept_boxes.device)
print("MMCV CUDA ops: PASS")
print("MMDetection imports: PASS")
print("TRAINING ENVIRONMENT: PASS")
PY

"${PY}" -m pip check

echo "ENVIRONMENT VALIDATION: PASS"
