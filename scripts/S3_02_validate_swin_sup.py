#!/usr/bin/env python3
"""S3.02 — Faster R-CNN Swin-T-FPN supervised execution-path preflight."""

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
    "cceed1f65333d80df5be2fdec25fbcec6d9c307638b1c690a3151c7c49f9cbe6"
)

EXPECTED_CHECKPOINT_SHA256 = (
    "9f71c168d837d1b99dd1dc29e14990a7a9e8bdc5f673d46b04fe36fe15590ad3"
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
            "s3_02_faster_rcnn_swin_t_fpn_sup.py"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "/workspace/ssod/artifacts/preflight/sup/"
            "sup_swin_preflight.json"
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

    checkpoint_path = Path(
        cfg.model.backbone.init_cfg["checkpoint"]
    ).resolve()

    checkpoint_exists = checkpoint_path.is_file()
    checkpoint_sha256 = (
        file_sha256(checkpoint_path)
        if checkpoint_exists
        else None
    )

    if not checkpoint_exists:
        raise RuntimeError(
            f"SWIN_T_CHECKPOINT_NOT_FOUND: {checkpoint_path}"
        )

    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "SWIN_T_CHECKPOINT_SHA256_MISMATCH: "
            f"expected={EXPECTED_CHECKPOINT_SHA256} "
            f"observed={checkpoint_sha256}"
        )

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

        "backbone_is_swin_t":
            cfg.model.backbone.type == "SwinTransformer",

        "swin_t_architecture_matches_canonical":
            cfg.model.backbone.embed_dims == 96
            and list(cfg.model.backbone.depths) == [2, 2, 6, 2]
            and list(cfg.model.backbone.num_heads) == [3, 6, 12, 24]
            and cfg.model.backbone.window_size == 7
            and cfg.model.backbone.mlp_ratio == 4
            and cfg.model.backbone.qkv_bias is True
            and cfg.model.backbone.patch_norm is True
            and tuple(cfg.model.backbone.out_indices) == (0, 1, 2, 3)
            and cfg.model.backbone.convert_weights is True,

        "neck_is_fpn":
            cfg.model.neck.type == "FPN",

        "fpn_in_channels_match_swin_t":
            list(cfg.model.neck.in_channels)
            == [96, 192, 384, 768],

        "imagenet_backbone_initialization":
            cfg.model.backbone.init_cfg["type"] == "Pretrained"
            and checkpoint_path
            == (
                ssod_root
                / "cache/pretrained/"
                "swin_tiny_patch4_window7_224.pth"
            ).resolve(),

        "checkpoint_sha256_matches_lock":
            checkpoint_exists
            and checkpoint_sha256 == EXPECTED_CHECKPOINT_SHA256,

        "no_detector_level_checkpoint":
            cfg.get("load_from", None) is None,

        "detection_class_count_is_14":
            cfg.model.roi_head.bbox_head.num_classes == 14,

        "all_backbone_parameters_trainable":
            len(frozen_backbone) == 0,

        "swin_frozen_stages_locked_minus1":
            cfg.model.backbone.frozen_stages == -1
            and model.backbone.frozen_stages == -1,

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
        "stage": "S3.02",
        "scope": "supervised_swin_t_execution_path_preflight",
        "official_training_authorized": False,
        "config": {
            "path": config_path.relative_to(repo_root).as_posix(),
            "sha256": config_sha256,
            "expected_sha256": EXPECTED_CONFIG_SHA256,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "exists": checkpoint_exists,
            "sha256": checkpoint_sha256,
            "expected_sha256": EXPECTED_CHECKPOINT_SHA256,
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
            "embed_dims": cfg.model.backbone.embed_dims,
            "depths": list(cfg.model.backbone.depths),
            "num_heads": list(cfg.model.backbone.num_heads),
            "window_size": cfg.model.backbone.window_size,
            "out_indices": list(cfg.model.backbone.out_indices),
            "convert_weights": cfg.model.backbone.convert_weights,
            "frozen_stages": cfg.model.backbone.frozen_stages,
            "backbone_init": dict(cfg.model.backbone.init_cfg),
            "neck_type": cfg.model.neck.type,
            "neck_in_channels": list(cfg.model.neck.in_channels),
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
        print("S3_02_SWIN_SUP_PREFLIGHT=FAIL")
        return 1

    print("S3_02_SWIN_SUP_PREFLIGHT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
