#!/usr/bin/env python3
"""S3.01 — Faster R-CNN R50-FPN supervised execution-path preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import runpy
import sys
from pathlib import Path

import torch
from mmengine.config import Config
from mmdet.registry import DATASETS, MODELS
from mmdet.utils import register_all_modules


EXPECTED_CONFIG_SHA256 = (
    "e189ba24ea6ddd5dd445dab9c6a3c5db562c5ca251ee3634ffd7a4c6a65811f6"
)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/workspace/ssod/project")
    parser.add_argument("--ssod-root", default="/workspace/ssod")
    parser.add_argument(
        "--config",
        default=(
            "/workspace/ssod/project/configs/supervised/"
            "s3_01_faster_rcnn_r50_fpn_sup.py"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "/workspace/ssod/artifacts/preflight/sup/"
            "sup_r50_preflight.json"
        ),
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    ssod_root = Path(args.ssod_root).resolve()
    config_path = Path(args.config).resolve()
    output_path = Path(args.output).resolve()

    sys.path.insert(0, str(repo_root))
    register_all_modules()

    config_sha256 = file_sha256(config_path)
    cfg = Config.fromfile(str(config_path))

    dataset_cfg = runpy.run_path(
        str(repo_root / "configs/dataset/s2_coco_dataset.py")
    )
    dataset = DATASETS.build(dataset_cfg["DATASETS"]["L_1pct"])
    sample = dataset[0]

    inputs = sample["inputs"]
    data_sample = sample["data_samples"]

    model = MODELS.build(cfg.model)
    model.init_weights()
    model = model.cuda()
    model.train()

    frozen_backbone = [
        name
        for name, param in model.backbone.named_parameters()
        if not param.requires_grad
    ]

    data = dict(
        inputs=[inputs],
        data_samples=[data_sample],
    )
    data = model.data_preprocessor(data, training=True)

    losses = model(
        inputs=data["inputs"],
        data_samples=data["data_samples"],
        mode="loss",
    )

    loss_terms = {}
    total_loss = None

    for name, value in losses.items():
        if isinstance(value, (list, tuple)):
            value = sum(value)

        if torch.is_tensor(value):
            numeric = float(value.detach().cpu())
            loss_terms[name] = numeric

            if "loss" in name:
                total_loss = (
                    value
                    if total_loss is None
                    else total_loss + value
                )

    if total_loss is None:
        raise RuntimeError("NO_TRAINING_LOSS_FOUND")

    total_loss.backward()

    total_loss_value = float(total_loss.detach().cpu())

    grad_param_count = sum(
        1
        for param in model.parameters()
        if param.requires_grad and param.grad is not None
    )

    checks = {
        "config_sha256_matches_preflight_identity":
            config_sha256 == EXPECTED_CONFIG_SHA256,

        "model_type_is_faster_rcnn":
            cfg.model.type == "FasterRCNN",

        "backbone_is_resnet50":
            cfg.model.backbone.type == "ResNet"
            and cfg.model.backbone.depth == 50,

        "neck_is_fpn":
            cfg.model.neck.type == "FPN",

        "imagenet_backbone_initialization":
            cfg.model.backbone.init_cfg["type"] == "Pretrained"
            and cfg.model.backbone.init_cfg["checkpoint"]
            == "torchvision://resnet50",

        "no_detector_level_checkpoint":
            cfg.get("load_from", None) is None,

        "detection_class_count_is_14":
            cfg.model.roi_head.bbox_head.num_classes == 14,

        "all_backbone_parameters_trainable":
            len(frozen_backbone) == 0,

        "r50_norm_eval_locked_true":
            cfg.model.backbone.norm_eval is True
            and model.backbone.norm_eval is True,

        "labeled_dataset_is_1pct_34_images":
            len(dataset) == 34,

        "input_has_three_identical_channels":
            inputs.ndim == 3
            and inputs.shape[0] == 3
            and torch.equal(inputs[0], inputs[1])
            and torch.equal(inputs[1], inputs[2]),

        "loss_forward_succeeded":
            total_loss is not None
            and math.isfinite(total_loss_value),

        "backward_produced_gradients":
            grad_param_count > 0,

        "cuda_execution":
            next(model.parameters()).device.type == "cuda",
    }

    failed_checks = [
        name for name, passed in checks.items()
        if not passed
    ]

    report = {
        "schema_version": "1.0",
        "stage": "S3.01",
        "scope": "supervised_r50_execution_path_preflight",
        "official_training_authorized": False,
        "config": {
            "path": config_path.relative_to(repo_root).as_posix(),
            "sha256": config_sha256,
            "expected_sha256": EXPECTED_CONFIG_SHA256,
        },
        "dataset": {
            "key": "L_1pct",
            "length": len(dataset),
            "image_id": data_sample.img_id,
            "gt_instance_count": len(data_sample.gt_instances),
            "input_shape_before_preprocessor": list(inputs.shape),
            "input_batch_shape_after_preprocessor":
                list(data["inputs"].shape),
        },
        "model": {
            "type": cfg.model.type,
            "backbone_type": cfg.model.backbone.type,
            "backbone_depth": cfg.model.backbone.depth,
            "frozen_stages": cfg.model.backbone.frozen_stages,
            "norm_eval": cfg.model.backbone.norm_eval,
            "backbone_init": dict(cfg.model.backbone.init_cfg),
            "neck_type": cfg.model.neck.type,
            "num_classes": cfg.model.roi_head.bbox_head.num_classes,
            "load_from": cfg.get("load_from", None),
            "frozen_backbone_param_count": len(frozen_backbone),
        },
        "runtime": {
            "device": str(next(model.parameters()).device),
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "smoke_test": {
            "loss_terms": loss_terms,
            "total_loss": total_loss_value,
            "grad_param_count": grad_param_count,
        },
        "checks": checks,
        "check_count": len(checks),
        "failed_checks": failed_checks,
        "all_checks_pass": not failed_checks,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    for name, passed in checks.items():
        print(f"{name} = {passed}")

    print("failed_checks =", failed_checks)
    print("ALL_CHECKS_PASS =", not failed_checks)

    if failed_checks:
        print("S3_01_R50_SUP_PREFLIGHT=FAIL")
        return 1

    print("S3_01_R50_SUP_PREFLIGHT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
