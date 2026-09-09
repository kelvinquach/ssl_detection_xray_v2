#!/usr/bin/env python3
"""S2.05 ? Validate retention of zero-GT / No Finding training samples."""

from __future__ import annotations

import argparse
import copy
import json
import os
import runpy
from datetime import datetime, timezone
from pathlib import Path

from mmdet.registry import DATASETS
from mmdet.utils import register_all_modules


EXPECTED_TRAIN_IMAGES = 3426
EXPECTED_TRAIN_ZERO_GT = 350


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default="/workspace/ssod/project",
    )
    parser.add_argument(
        "--ssod-root",
        default="/workspace/ssod",
    )
    parser.add_argument(
        "--output",
        default="/workspace/ssod/artifacts/preflight/ssl/zero_gt_test.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    repo_root = Path(args.repo_root).resolve()
    ssod_root = Path(args.ssod_root).resolve()
    output_path = Path(args.output).resolve()

    os.environ["SSOD_ROOT"] = str(ssod_root)
    register_all_modules()

    cfg = runpy.run_path(
        str(repo_root / "configs/dataset/s2_coco_dataset.py")
    )
    train_cfg = copy.deepcopy(cfg["DATASETS"]["train"])

    filter_empty_gt = train_cfg.get(
        "filter_cfg", {}
    ).get("filter_empty_gt")

    ann_file = Path(train_cfg["ann_file"])
    with ann_file.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    ann_count = {int(img["id"]): 0 for img in coco["images"]}
    for ann in coco["annotations"]:
        ann_count[int(ann["image_id"])] += 1

    zero_gt_ids = sorted(
        image_id
        for image_id, count in ann_count.items()
        if count == 0
    )

    dataset = DATASETS.build(train_cfg)
    dataset.full_init()

    retained_ids = {
        int(dataset.get_data_info(i)["img_id"])
        for i in range(len(dataset))
    }

    zero_gt_retained = set(zero_gt_ids).issubset(retained_ids)

    zero_gt_id = zero_gt_ids[0]
    zero_gt_index = next(
        i
        for i in range(len(dataset))
        if int(dataset.get_data_info(i)["img_id"]) == zero_gt_id
    )

    sample = dataset[zero_gt_index]
    gt = sample["data_samples"].gt_instances

    bbox_shape = list(gt.bboxes.tensor.shape)
    label_shape = list(gt.labels.shape)

    checks = {
        "filter_empty_gt_is_false": filter_empty_gt is False,
        "train_image_count_match": len(dataset) == EXPECTED_TRAIN_IMAGES,
        "zero_gt_count_match": len(zero_gt_ids) == EXPECTED_TRAIN_ZERO_GT,
        "all_zero_gt_ids_retained": zero_gt_retained,
        "zero_gt_bbox_shape_empty": bbox_shape == [0, 4],
        "zero_gt_label_shape_empty": label_shape == [0],
    }

    status = "PASS" if all(checks.values()) else "FAIL"

    report = {
        "schema_version": "1.1",
        "artifact_type": "ZERO_GT_TEST",
        "stage": "S2.05",
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_authorized": False,
        "config": {
            "dataset_type": train_cfg.get("type"),
            "ann_file": str(ann_file),
            "filter_empty_gt": filter_empty_gt,
        },
        "expected": {
            "train_images": EXPECTED_TRAIN_IMAGES,
            "train_zero_gt": EXPECTED_TRAIN_ZERO_GT,
        },
        "observed": {
            "dataset_length": len(dataset),
            "zero_gt_count": len(zero_gt_ids),
            "zero_gt_retained_count": len(
                set(zero_gt_ids) & retained_ids
            ),
        },
        "sample_evidence": {
            "image_id": zero_gt_id,
            "bbox_shape": bbox_shape,
            "label_shape": label_shape,
        },
        "checks": checks,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"OUTPUT = {output_path}")
    print(f"filter_empty_gt = {filter_empty_gt}")
    print(f"dataset_length = {len(dataset)}")
    print(f"zero_gt_count = {len(zero_gt_ids)}")
    print(
        "zero_gt_retained_count =",
        len(set(zero_gt_ids) & retained_ids),
    )
    print(f"sample_zero_gt_image_id = {zero_gt_id}")
    print(f"bbox_shape = {bbox_shape}")
    print(f"label_shape = {label_shape}")
    print(f"S2_05_ZERO_GT_TEST={status}")

    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
