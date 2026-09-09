"""S2.01 — Fixed COCO dataset loading."""

from __future__ import annotations

import os
from pathlib import Path


SSOD_ROOT = Path(os.environ.get("SSOD_ROOT", "/workspace/ssod"))
COCO_ROOT = SSOD_ROOT / "data" / "coco"

# S1 stores the locked JPEGs flat under data/images.
# COCO file_name remains train/<image_id>.jpg, therefore this lightweight
# runtime view is used only so native MMDetection path resolution works.
MMDET_IMAGE_ROOT = SSOD_ROOT / "data" / "mmdet_images"

CLASSES = (
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
)

PIPELINE = [
    dict(type="LoadImageFromFile", color_type="color"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(type="Pad", size_divisor=32),
    dict(type="PackDetInputs"),
]


def coco_dataset(ann_file: str, *, test_mode: bool = False) -> dict:
    return dict(
        type="CocoDataset",
        ann_file=str(COCO_ROOT / ann_file),
        data_prefix=dict(img=str(MMDET_IMAGE_ROOT)),
        metainfo=dict(classes=CLASSES),
        filter_cfg=dict(filter_empty_gt=False),
        test_mode=test_mode,
        pipeline=PIPELINE,
    )


DATASETS = {
    "train": coco_dataset("instances_train.json"),
    "val": coco_dataset("instances_val.json", test_mode=True),
    "test": coco_dataset("instances_test.json", test_mode=True),

    "L_1pct": coco_dataset("instances_labeled_1pct.json"),
    "L_5pct": coco_dataset("instances_labeled_5pct.json"),
    "L_10pct": coco_dataset("instances_labeled_10pct.json"),
    "L_20pct": coco_dataset("instances_labeled_20pct.json"),

    "U_1pct": coco_dataset("instances_unlabeled_1pct.json"),
    "U_5pct": coco_dataset("instances_unlabeled_5pct.json"),
    "U_10pct": coco_dataset("instances_unlabeled_10pct.json"),
    "U_20pct": coco_dataset("instances_unlabeled_20pct.json"),
}

training_authorized = False