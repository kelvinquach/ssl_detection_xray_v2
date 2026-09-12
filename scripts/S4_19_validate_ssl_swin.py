#!/usr/bin/env python3
"""S4.19 — Faster R-CNN Swin-T-FPN SSL official execution-path preflight."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.structures import InstanceData
from mmengine.utils import import_modules_from_strings
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules


EXPECTED_CONFIG_SHA256 = {
    "1pct": "5003c14f0ef0622617b986b4a7ffd66738474d7e6c524ff9ad166b005c56d75a",
    "5pct": "d45ae41a58f36d1805a1ae9c4ceeb20947f08be60fc61e51565eec7e7c613259",
    "10pct": "6c64de2244fa0c40c9a4c224751fbd407b19f89b0d67dbabebaf8824867199c9",
    "20pct": "c07bcacf9000b1a66ab5a4181ae0af0ab7f22b9c5971a05f402e4a0d078b65c5",
}

CONFIG_PATHS = {
    "1pct": (
        "configs/ssl/"
        "s4_14_soft_teacher_swin_t_fpn_empty_pseudo_1pct.py"
    ),
    "5pct": (
        "configs/ssl/"
        "s4_14_soft_teacher_swin_t_fpn_empty_pseudo_5pct.py"
    ),
    "10pct": (
        "configs/ssl/"
        "s4_14_soft_teacher_swin_t_fpn_empty_pseudo_10pct.py"
    ),
    "20pct": (
        "configs/ssl/"
        "s4_14_soft_teacher_swin_t_fpn_empty_pseudo_20pct.py"
    ),
}

EXPECTED_SUP_CONFIG_SHA256 = {
    "1pct": "c067c331c8bec3261fad93da3f131d7bf9415b0f8dbb138e21daf9b79c12e1f4",
    "5pct": "de476d662ab207a5b4844b6fed100022f52b92ce7e51a05e4fb9a6428f67412a",
    "10pct": "ed588940c749c2eacfae5063321fb1710e2c99bc6e525d1d34fde82ab7fce35f",
    "20pct": "23343f0d95d9539e09fa7980307dce990198f224fd2cdd1ae6dbfdafcc1f79c1",
}

SUP_CONFIG_PATHS = {
    "1pct": (
        "configs/supervised/"
        "s3_13_faster_rcnn_swin_t_fpn_sup_1pct.py"
    ),
    "5pct": (
        "configs/supervised/"
        "s3_13_faster_rcnn_swin_t_fpn_sup_5pct.py"
    ),
    "10pct": (
        "configs/supervised/"
        "s3_13_faster_rcnn_swin_t_fpn_sup_10pct.py"
    ),
    "20pct": (
        "configs/supervised/"
        "s3_13_faster_rcnn_swin_t_fpn_sup_20pct.py"
    ),
}

EXPECTED_SWIN_CHECKPOINT_SHA256 = (
    "9f71c168d837d1b99dd1dc29e14990a7a9e8bdc5f673d46b04fe36fe15590ad3"
)

EXPECTED_SWIN_NO_DECAY = {
    "absolute_pos_embed": {"decay_mult": 0.0},
    "relative_position_bias_table": {"decay_mult": 0.0},
    "norm": {"decay_mult": 0.0},
}

EXPECTED_UPDATES = {
    "1pct": 1032,
    "5pct": 2064,
    "10pct": 2064,
    "20pct": 2064,
}

EXPECTED_EVIDENCE = {
    "teacher_initialization": (
        "artifacts/preflight/ssl/teacher_initialization_test.json",
        "c110dac233775e71e06e08868d2198abbcb358e1bbfc066b02bcb453c309b358",
        "all_checks_pass",
    ),
    "ema_timing": (
        "artifacts/preflight/ssl/ema_timing_test.json",
        "2fa821dbe18baea43486bfa163ba578ca15ae71eecf54a2352e208b9cfc0eda0",
        "all_checks_pass",
    ),
    "amp_ema_skip": (
        "artifacts/preflight/ssl/amp_ema_skip_test.json",
        "d8974bf9127407eb227a239e84fa5c59a4e5c6ef16744b58071ca17e0a13d49d",
        "all_checks_pass",
    ),
    "empty_pseudo_batch": (
        "artifacts/preflight/ssl/empty_pseudo_batch_test.json",
        "6ef438c3958444bbd1f4dd7dd69fbefea35af956f2015e38403290e9a4045b99",
        "overall_pass",
    ),
    "zero_gt": (
        "artifacts/preflight/ssl/zero_gt_test.json",
        "24524729a4a7ed7a88639c5d0ab847226a7e2750d20bc3b37085188fc789ecab",
        "overall_pass",
    ),
    "ssl_augmentation": (
        "artifacts/preflight/ssl/ssl_augmentation_manifest.json",
        "6e6fb0d680edd8f4a0da0a765cc9539c0b1a176257b910f67cea0907e8762ebb",
        "all_checks_pass",
    ),
    "pseudo_bbox_alignment": (
        "artifacts/preflight/ssl/pseudo_bbox_alignment_report.json",
        "27b69568818301ef21a1896fdca290ea4be2fca2ea348d3248e874e4ccc6142c",
        "all_checks_pass",
    ),
    "hidden_u_firewall": (
        "artifacts/preflight/firewall/hidden_u_gt_firewall_report.json",
        "f52fd896ea38b59fd353f191fae74a8303053eea184572973ffa18219432ac01",
        "all_checks_pass",
    ),
    "ssl_resume_equivalence": (
        "artifacts/preflight/resume/resume_equivalence_test.json",
        "f92db6919196d9f2f3098053017af57a115896c4544ef5ecc249c097a0ad5e80",
        "all_checks_pass",
    ),
    "s4_13_real_detector_one_update": (
        "artifacts/preflight/ssl/"
        "s4_13_real_detector_one_update/training_summary.json",
        "fa2a42d10ea2bbe5119da71124e2c305bf255db819158ce7580ca4b09803289f",
        None,
    ),
}

EXPECTED_HOOK_TYPES = {
    "TeacherInitializationHook",
    "ProtocolResumeCheckpointHook",
    "ActualUpdateMeanTeacherHook",
    "ActualUpdateBudgetSchedulerHook",
    "SSLLatestResumeCheckpointHook",
    "SSLTrainingSummaryHook",
}

REQUIRED_NORMAL_PSEUDO_LOSS_KEYS = {
    "unsup_loss_rpn_cls",
    "unsup_loss_rpn_bbox",
    "unsup_loss_cls",
    "unsup_loss_bbox",
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def flatten_loss_value(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)):
        if not value:
            raise RuntimeError("EMPTY_LOSS_SEQUENCE")
        return sum(flatten_loss_value(v) for v in value)
    raise TypeError(
        f"Unsupported loss value type: {type(value).__name__}"
    )


def loss_summary(losses: dict) -> tuple[dict, torch.Tensor]:
    numeric = {}
    total = None

    for name, value in losses.items():
        flat = flatten_loss_value(value)
        numeric[name] = float(flat.detach().cpu())

        if "loss" in name:
            total = flat if total is None else total + flat

    if total is None:
        raise RuntimeError("NO_TRAINING_LOSS_FOUND")

    return numeric, total


def gradient_summary(module) -> tuple[bool, float, int]:
    seen = False
    abs_sum = 0.0
    count = 0

    for parameter in module.parameters():
        if parameter.grad is not None:
            seen = True
            count += 1
            abs_sum += float(
                parameter.grad.detach().abs().sum().cpu()
            )

    return seen, abs_sum, count


def state_dict_equal(a, b) -> bool:
    a_state = a.state_dict()
    b_state = b.state_dict()

    if a_state.keys() != b_state.keys():
        return False

    return all(
        torch.equal(a_state[key], b_state[key])
        for key in a_state
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
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
        default=(
            "/workspace/ssod/artifacts/preflight/ssl/"
            "ssl_swin_preflight.json"
        ),
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    ssod_root = Path(args.ssod_root).resolve()
    output_path = Path(args.output).resolve()

    sys.path.insert(0, str(repo_root))
    register_all_modules()

    checks: dict[str, bool] = {}
    config_results = {}
    resolved_configs = {}

    swin_checkpoint_path = (
        ssod_root
        / "cache/pretrained/swin_tiny_patch4_window7_224.pth"
    ).resolve()
    swin_checkpoint_exists = swin_checkpoint_path.is_file()
    swin_checkpoint_sha256 = (
        file_sha256(swin_checkpoint_path)
        if swin_checkpoint_exists
        else None
    )

    print("===== S4.19 CONFIG MATRIX =====")

    for budget, relative_path in CONFIG_PATHS.items():
        config_path = repo_root / relative_path
        config_sha = file_sha256(config_path)
        cfg = Config.fromfile(str(config_path))

        if cfg.get("custom_imports", None):
            import_modules_from_strings(**cfg.custom_imports)

        resolved_configs[budget] = cfg

        sup_relative_path = SUP_CONFIG_PATHS[budget]
        sup_config_path = repo_root / sup_relative_path
        sup_config_sha = file_sha256(sup_config_path)
        sup_cfg = Config.fromfile(str(sup_config_path))

        detector = cfg.model.detector
        semi = cfg.model.semi_train_cfg
        optimizer = cfg.optim_wrapper.optimizer
        sup_optimizer = sup_cfg.optim_wrapper.optimizer

        ssl_custom_keys = {
            key: dict(value)
            for key, value
            in cfg.optim_wrapper.paramwise_cfg.custom_keys.items()
        }
        sup_custom_keys = {
            key: dict(value)
            for key, value
            in sup_cfg.optim_wrapper.paramwise_cfg.custom_keys.items()
        }

        hooks = list(cfg.custom_hooks)
        hook_types = [hook["type"] for hook in hooks]

        budget_hook = next(
            (
                hook
                for hook in hooks
                if hook["type"]
                == "ActualUpdateBudgetSchedulerHook"
            ),
            None,
        )

        local_checks = {
            "config_sha256_matches":
                config_sha == EXPECTED_CONFIG_SHA256[budget],

            "model_is_empty_pseudo_safe_softteacher":
                cfg.model.type == "EmptyPseudoSafeSoftTeacher",

            "detector_is_faster_rcnn":
                detector.type == "FasterRCNN",

            "backbone_is_swin_t":
                detector.backbone.type == "SwinTransformer",

            "swin_t_architecture_matches_canonical":
                int(detector.backbone.embed_dims) == 96
                and list(detector.backbone.depths)
                == [2, 2, 6, 2]
                and list(detector.backbone.num_heads)
                == [3, 6, 12, 24]
                and int(detector.backbone.window_size) == 7
                and math.isclose(
                    float(detector.backbone.mlp_ratio),
                    4.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and detector.backbone.qkv_bias is True
                and math.isclose(
                    float(detector.backbone.drop_rate),
                    0.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    float(detector.backbone.attn_drop_rate),
                    0.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    float(detector.backbone.drop_path_rate),
                    0.2,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and detector.backbone.patch_norm is True
                and tuple(detector.backbone.out_indices)
                == (0, 1, 2, 3)
                and detector.backbone.with_cp is False
                and detector.backbone.convert_weights is True,

            "neck_is_fpn":
                detector.neck.type == "FPN"
                and list(detector.neck.in_channels)
                == [96, 192, 384, 768],

            "imagenet_backbone_only":
                detector.backbone.init_cfg["type"]
                == "Pretrained"
                and Path(
                    detector.backbone.init_cfg["checkpoint"]
                ).resolve()
                == swin_checkpoint_path
                and detector.backbone.convert_weights is True
                and cfg.get("load_from", None) is None,

            "swin_checkpoint_exists":
                swin_checkpoint_exists,

            "swin_checkpoint_sha256_matches_lock":
                swin_checkpoint_exists
                and swin_checkpoint_sha256
                == EXPECTED_SWIN_CHECKPOINT_SHA256,

            "num_classes_is_14":
                detector.roi_head.bbox_head.num_classes == 14,

            "all_backbone_stages_trainable":
                detector.backbone.frozen_stages == -1,

            "amp_enabled":
                cfg.optim_wrapper.type == "AmpOptimWrapper",

            "amp_dynamic_loss_scale":
                cfg.optim_wrapper.get("loss_scale", None)
                == "dynamic",

            "gradient_clipping_off":
                cfg.optim_wrapper.get("clip_grad", None) is None,

            "optimizer_is_locked_adamw":
                optimizer.type == "AdamW"
                and math.isclose(
                    float(optimizer.lr),
                    0.000025,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and tuple(
                    float(value)
                    for value in optimizer.betas
                )
                == (0.9, 0.999)
                and math.isclose(
                    float(optimizer.weight_decay),
                    0.05,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),

            "native_swin_no_decay_rules_retained":
                ssl_custom_keys == EXPECTED_SWIN_NO_DECAY,

            "sup_reference_config_sha256_matches":
                sup_config_sha
                == EXPECTED_SUP_CONFIG_SHA256[budget],

            "within_architecture_sup_ssl_recipe_matches":
                cfg.optim_wrapper.type
                == sup_cfg.optim_wrapper.type
                and cfg.optim_wrapper.get(
                    "loss_scale",
                    None,
                )
                == sup_cfg.optim_wrapper.get(
                    "loss_scale",
                    None,
                )
                and optimizer.type
                == sup_optimizer.type
                and math.isclose(
                    float(optimizer.lr),
                    float(sup_optimizer.lr),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and tuple(
                    float(value)
                    for value in optimizer.betas
                )
                == tuple(
                    float(value)
                    for value in sup_optimizer.betas
                )
                and math.isclose(
                    float(optimizer.weight_decay),
                    float(sup_optimizer.weight_decay),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and cfg.optim_wrapper.get(
                    "clip_grad",
                    None,
                )
                == sup_cfg.optim_wrapper.get(
                    "clip_grad",
                    None,
                )
                and ssl_custom_keys
                == sup_custom_keys
                == EXPECTED_SWIN_NO_DECAY,

            "sup_weight_is_1":
                math.isclose(
                    float(semi.sup_weight),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),

            "unsup_weight_is_4":
                math.isclose(
                    float(semi.unsup_weight),
                    4.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),

            "initial_pseudo_threshold_is_0_5":
                math.isclose(
                    float(semi.pseudo_label_initial_score_thr),
                    0.5,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),

            "rpn_pseudo_threshold_is_0_9":
                math.isclose(
                    float(semi.rpn_pseudo_thr),
                    0.9,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),

            "cls_pseudo_threshold_is_0_9":
                math.isclose(
                    float(semi.cls_pseudo_thr),
                    0.9,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),

            "reg_pseudo_threshold_is_0_02":
                math.isclose(
                    float(semi.reg_pseudo_thr),
                    0.02,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),

            "batch_size_is_8":
                int(cfg.train_dataloader.batch_size) == 8,

            "sampler_is_multisource":
                cfg.train_dataloader.sampler.type
                == "MultiSourceSampler",

            "source_ratio_is_1_to_1":
                list(cfg.train_dataloader.sampler.source_ratio)
                == [1, 1],

            "actual_update_validation_loop":
                cfg.train_cfg.type
                == "ActualUpdateValidationIterBasedTrainLoop",

            "max_updates_match_budget":
                int(cfg.train_cfg.max_iters)
                == EXPECTED_UPDATES[budget],

            "validation_interval_is_172":
                int(cfg.train_cfg.val_interval) == 172,

            "required_hooks_present_once":
                set(hook_types) == EXPECTED_HOOK_TYPES
                and len(hook_types) == len(EXPECTED_HOOK_TYPES),

            "actual_update_budget_hook_matches":
                budget_hook is not None
                and int(
                    budget_hook.get(
                        "target_updates",
                        EXPECTED_UPDATES[budget],
                    )
                )
                == EXPECTED_UPDATES[budget],
        }

        for name, passed in local_checks.items():
            checks[f"{budget}_{name}"] = bool(passed)

        config_results[budget] = {
            "path": relative_path,
            "sha256": config_sha,
            "expected_sha256":
                EXPECTED_CONFIG_SHA256[budget],
            "sup_reference_path":
                sup_relative_path,
            "sup_reference_sha256":
                sup_config_sha,
            "expected_sup_reference_sha256":
                EXPECTED_SUP_CONFIG_SHA256[budget],
            "expected_optimizer_updates":
                EXPECTED_UPDATES[budget],
            "hook_types": hook_types,
            "checks": local_checks,
            "all_checks_pass":
                all(local_checks.values()),
        }

        print(
            f"{budget}: "
            f"{relative_path} "
            f"sha256={config_sha} "
            f"pass={all(local_checks.values())}"
        )

    checks["all_4_swin_budget_configs_resolve_pass"] = all(
        result["all_checks_pass"]
        for result in config_results.values()
    )

    print("\n===== PREREQUISITE EVIDENCE =====")

    prerequisite_evidence = {}

    for name, (
        relative_path,
        expected_sha,
        pass_field,
    ) in EXPECTED_EVIDENCE.items():
        path = repo_root / relative_path
        exists = path.is_file()
        actual_sha = file_sha256(path) if exists else None
        payload = load_json(path) if exists else {}

        if pass_field is None:
            if name == "s4_13_real_detector_one_update":
                content_pass = bool(
                    payload.get("observed_optimizer_updates")
                    == 1
                    and payload.get("ema_update_count") == 1
                    and payload.get(
                        "ema_updates_match_optimizer_updates"
                    )
                    is True
                    and payload.get("effective_labeled_batch")
                    == 4
                    and payload.get(
                        "effective_unlabeled_batch"
                    )
                    == 4
                    and payload.get("l_u_ratio") == "1:1"
                )
            else:
                content_pass = True
        else:
            content_pass = payload.get(pass_field) is True

        if name == "hidden_u_firewall":
            content_pass = bool(
                content_pass
                and payload.get("overall_pass") is True
                and len(payload.get("failed_checks", [])) == 0
            )

        if name in {
            "teacher_initialization",
            "ema_timing",
            "amp_ema_skip",
            "ssl_augmentation",
            "pseudo_bbox_alignment",
            "ssl_resume_equivalence",
        }:
            content_pass = bool(
                content_pass
                and len(payload.get("failed_checks", [])) == 0
            )

        passed = bool(
            exists
            and actual_sha == expected_sha
            and content_pass
        )

        checks[f"prerequisite_{name}_pass"] = passed

        prerequisite_evidence[name] = {
            "path": relative_path,
            "exists": exists,
            "sha256": actual_sha,
            "expected_sha256": expected_sha,
            "content_pass": content_pass,
            "pass": passed,
        }

        print(
            f"{name}: exists={exists} "
            f"sha_match={actual_sha == expected_sha} "
            f"content_pass={content_pass}"
        )

    print("\n===== OFFICIAL SWIN-T 1PCT RUNTIME PATH =====")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    cfg = resolved_configs["1pct"]

    loader = Runner.build_dataloader(
        cfg.train_dataloader,
        seed=42,
        diff_rank_seed=False,
    )

    batch = next(iter(loader))

    raw_inputs = batch["inputs"]
    raw_samples = batch["data_samples"]

    branch_counts = {}
    branch_indices = {}
    branch_image_ids = {}
    raw_channel_checks = {}

    for branch in (
        "sup",
        "unsup_teacher",
        "unsup_student",
    ):
        input_values = raw_inputs[branch]
        sample_values = raw_samples[branch]

        input_indices = [
            i
            for i, value in enumerate(input_values)
            if value is not None
        ]
        sample_indices = [
            i
            for i, value in enumerate(sample_values)
            if value is not None
        ]

        images = [
            value
            for value in input_values
            if value is not None
        ]
        samples = [
            value
            for value in sample_values
            if value is not None
        ]

        branch_counts[branch] = {
            "inputs": len(images),
            "data_samples": len(samples),
        }
        branch_indices[branch] = {
            "inputs": input_indices,
            "data_samples": sample_indices,
        }
        branch_image_ids[branch] = [
            getattr(sample, "img_id", None)
            for sample in samples
        ]

        raw_channel_checks[branch] = all(
            image.ndim == 3
            and image.shape[0] == 3
            and torch.equal(image[0], image[1])
            and torch.equal(image[1], image[2])
            for image in images
        )

    checks["runtime_effective_labeled_batch_is_4"] = (
        branch_counts["sup"]["data_samples"] == 4
    )
    checks["runtime_effective_unlabeled_teacher_batch_is_4"] = (
        branch_counts["unsup_teacher"]["data_samples"]
        == 4
    )
    checks["runtime_effective_unlabeled_student_batch_is_4"] = (
        branch_counts["unsup_student"]["data_samples"]
        == 4
    )
    checks["runtime_l_to_u_ratio_is_1_to_1"] = (
        branch_counts["sup"]["data_samples"]
        == branch_counts["unsup_teacher"]["data_samples"]
        == branch_counts["unsup_student"]["data_samples"]
        == 4
    )
    checks["teacher_student_unlabeled_image_ids_match"] = (
        branch_image_ids["unsup_teacher"]
        == branch_image_ids["unsup_student"]
    )
    checks["all_raw_branches_three_identical_channels"] = all(
        raw_channel_checks.values()
    )

    model = MODELS.build(cfg.model)
    model.init_weights()
    model = model.cuda()
    model.train()

    student_frozen_backbone = [
        name
        for name, parameter
        in model.student.backbone.named_parameters()
        if not parameter.requires_grad
    ]
    teacher_trainable = [
        name
        for name, parameter
        in model.teacher.named_parameters()
        if parameter.requires_grad
    ]

    def runtime_swin_signature(detector):
        backbone = detector.backbone
        neck = detector.neck

        stages = list(backbone.stages)

        return {
            "patch_projection_weight_shape":
                list(
                    backbone.patch_embed
                    .projection.weight.shape
                ),
            "patch_projection_kernel_size":
                list(
                    backbone.patch_embed
                    .projection.kernel_size
                ),
            "patch_projection_stride":
                list(
                    backbone.patch_embed
                    .projection.stride
                ),
            "stage_depths": [
                len(stage.blocks)
                for stage in stages
            ],
            "stage_heads": [
                int(
                    stage.blocks[0]
                    .attn.w_msa.num_heads
                )
                for stage in stages
            ],
            "stage_window_sizes": [
                list(
                    stage.blocks[0]
                    .attn.w_msa.window_size
                )
                for stage in stages
            ],
            "stage_embed_dims": [
                int(
                    stage.blocks[0]
                    .attn.w_msa.embed_dims
                )
                for stage in stages
            ],
            "num_features": [
                int(value)
                for value in backbone.num_features
            ],
            "out_indices": [
                int(value)
                for value in backbone.out_indices
            ],
            "frozen_stages":
                int(backbone.frozen_stages),
            "convert_weights":
                bool(backbone.convert_weights),
            "neck_type":
                type(neck).__name__,
            "fpn_lateral_in_channels": [
                int(conv_module.conv.in_channels)
                for conv_module
                in neck.lateral_convs
            ],
        }

    expected_runtime_swin_signature = {
        "patch_projection_weight_shape":
            [96, 3, 4, 4],
        "patch_projection_kernel_size":
            [4, 4],
        "patch_projection_stride":
            [4, 4],
        "stage_depths":
            [2, 2, 6, 2],
        "stage_heads":
            [3, 6, 12, 24],
        "stage_window_sizes":
            [[7, 7], [7, 7], [7, 7], [7, 7]],
        "stage_embed_dims":
            [96, 192, 384, 768],
        "num_features":
            [96, 192, 384, 768],
        "out_indices":
            [0, 1, 2, 3],
        "frozen_stages":
            -1,
        "convert_weights":
            True,
        "neck_type":
            "FPN",
        "fpn_lateral_in_channels":
            [96, 192, 384, 768],
    }

    student_swin_signature = runtime_swin_signature(
        model.student
    )
    teacher_swin_signature = runtime_swin_signature(
        model.teacher
    )

    checks["runtime_model_is_empty_pseudo_safe_softteacher"] = (
        type(model).__name__
        == "EmptyPseudoSafeSoftTeacher"
    )
    checks["runtime_student_is_faster_rcnn"] = (
        type(model.student).__name__ == "FasterRCNN"
    )
    checks["runtime_teacher_is_faster_rcnn"] = (
        type(model.teacher).__name__ == "FasterRCNN"
    )
    checks["runtime_student_backbone_is_swin_transformer"] = (
        type(model.student.backbone).__name__
        == "SwinTransformer"
    )
    checks["runtime_teacher_backbone_is_swin_transformer"] = (
        type(model.teacher.backbone).__name__
        == "SwinTransformer"
    )
    checks["runtime_student_swin_architecture_matches_canonical"] = (
        student_swin_signature
        == expected_runtime_swin_signature
    )
    checks["runtime_teacher_swin_architecture_matches_canonical"] = (
        teacher_swin_signature
        == expected_runtime_swin_signature
    )
    checks["runtime_student_teacher_swin_architecture_match"] = (
        student_swin_signature
        == teacher_swin_signature
    )
    checks["runtime_student_backbone_all_trainable"] = (
        len(student_frozen_backbone) == 0
    )
    checks["runtime_teacher_parameters_frozen"] = (
        len(teacher_trainable) == 0
    )
    checks["runtime_cuda_execution"] = (
        next(model.student.parameters()).device.type
        == "cuda"
    )

    model.teacher.load_state_dict(
        model.student.state_dict(),
        strict=True,
    )

    teacher_equals_student_t0 = state_dict_equal(
        model.teacher,
        model.student,
    )

    checks["runtime_teacher_t0_equals_student_t0"] = (
        teacher_equals_student_t0
    )

    processed = model.data_preprocessor(
        batch,
        training=True,
    )

    processed_shapes = {
        branch: list(processed["inputs"][branch].shape)
        for branch in (
            "sup",
            "unsup_teacher",
            "unsup_student",
        )
    }

    processed_counts = {
        branch: len(processed["data_samples"][branch])
        for branch in (
            "sup",
            "unsup_teacher",
            "unsup_student",
        )
    }

    checks["preprocessor_preserves_4_4_branch_counts"] = (
        processed_counts["sup"] == 4
        and processed_counts["unsup_teacher"] == 4
        and processed_counts["unsup_student"] == 4
    )

    print("\n===== OFFICIAL FULL SSL LOSS FORWARD =====")

    model.student.zero_grad(set_to_none=True)

    official_samples = copy.deepcopy(
        processed["data_samples"]
    )

    official_losses = model.loss(
        processed["inputs"],
        official_samples,
    )

    official_loss_terms, official_total = loss_summary(
        official_losses
    )

    official_total_value = float(
        official_total.detach().cpu()
    )

    official_loss_finite = bool(
        torch.isfinite(official_total).all().item()
    )

    official_total.backward()

    (
        official_student_grad_seen,
        official_student_grad_abs_sum,
        official_student_grad_param_count,
    ) = gradient_summary(model.student)

    (
        official_teacher_grad_seen,
        official_teacher_grad_abs_sum,
        official_teacher_grad_param_count,
    ) = gradient_summary(model.teacher)

    checks["official_ssl_loss_forward_finite"] = (
        official_loss_finite
        and math.isfinite(official_total_value)
    )
    checks["official_ssl_backward_student_gradient_positive"] = (
        official_student_grad_seen
        and math.isfinite(
            official_student_grad_abs_sum
        )
        and official_student_grad_abs_sum > 0.0
    )
    checks["official_ssl_teacher_has_no_gradient"] = (
        not official_teacher_grad_seen
        and official_teacher_grad_param_count == 0
        and official_teacher_grad_abs_sum == 0.0
    )

    del official_losses
    del official_total
    model.student.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    print("\n===== OFFICIAL-THRESHOLD PSEUDO PROBE =====")

    teacher_inputs = processed["inputs"]["unsup_teacher"]
    student_inputs = processed["inputs"]["unsup_student"]

    teacher_samples = copy.deepcopy(
        processed["data_samples"]["unsup_teacher"]
    )
    student_samples = copy.deepcopy(
        processed["data_samples"]["unsup_student"]
    )

    thresholds_before = {
        "pseudo_label_initial_score_thr":
            float(
                model.semi_train_cfg
                .pseudo_label_initial_score_thr
            ),
        "rpn_pseudo_thr":
            float(model.semi_train_cfg.rpn_pseudo_thr),
        "cls_pseudo_thr":
            float(model.semi_train_cfg.cls_pseudo_thr),
        "reg_pseudo_thr":
            float(model.semi_train_cfg.reg_pseudo_thr),
        "unsup_weight":
            float(model.semi_train_cfg.unsup_weight),
    }

    with torch.no_grad():
        origin_pseudo, batch_info = (
            model.get_pseudo_instances(
                teacher_inputs,
                teacher_samples,
            )
        )

    teacher_pseudo_counts = [
        len(sample.gt_instances)
        for sample in origin_pseudo
    ]

    projected = model.project_pseudo_instances(
        copy.deepcopy(origin_pseudo),
        copy.deepcopy(student_samples),
    )

    projected_pseudo_counts = [
        len(sample.gt_instances)
        for sample in projected
    ]

    checks["official_threshold_pseudo_path_executes"] = (
        len(teacher_pseudo_counts) == 4
        and len(projected_pseudo_counts) == 4
    )

    print(
        "teacher_pseudo_counts =",
        teacher_pseudo_counts,
    )
    print(
        "projected_pseudo_counts =",
        projected_pseudo_counts,
    )

    print("\n===== CONTROLLED NON-EMPTY PSEUDO BRANCH =====")

    controlled_samples = copy.deepcopy(
        student_samples
    )

    synthetic_counts = []

    for sample in controlled_samples:
        h, w = sample.img_shape[:2]

        x1 = max(1.0, 0.25 * float(w))
        y1 = max(1.0, 0.25 * float(h))
        x2 = min(float(w) - 1.0, 0.75 * float(w))
        y2 = min(float(h) - 1.0, 0.75 * float(h))

        if not (x2 > x1 and y2 > y1):
            raise RuntimeError(
                "INVALID_SYNTHETIC_PSEUDO_BBOX"
            )

        sample.gt_instances = InstanceData(
            bboxes=torch.tensor(
                [[x1, y1, x2, y2]],
                dtype=torch.float32,
                device=student_inputs.device,
            ),
            labels=torch.tensor(
                [0],
                dtype=torch.long,
                device=student_inputs.device,
            ),
            scores=torch.tensor(
                [1.0],
                dtype=torch.float32,
                device=student_inputs.device,
            ),
            reg_uncs=torch.tensor(
                [0.0],
                dtype=torch.float32,
                device=student_inputs.device,
            ),
        )

        synthetic_counts.append(
            len(sample.gt_instances)
        )

    model.student.zero_grad(set_to_none=True)

    controlled_losses = (
        model.loss_by_pseudo_instances(
            student_inputs,
            controlled_samples,
            batch_info,
        )
    )

    controlled_loss_terms, controlled_total = (
        loss_summary(controlled_losses)
    )

    controlled_total_value = float(
        controlled_total.detach().cpu()
    )

    controlled_total_finite = bool(
        torch.isfinite(controlled_total).all().item()
    )

    controlled_loss_keys = {
        key
        for key in controlled_losses
        if "loss" in key
    }

    controlled_total.backward()

    (
        controlled_student_grad_seen,
        controlled_student_grad_abs_sum,
        controlled_student_grad_param_count,
    ) = gradient_summary(model.student)

    (
        controlled_teacher_grad_seen,
        controlled_teacher_grad_abs_sum,
        controlled_teacher_grad_param_count,
    ) = gradient_summary(model.teacher)

    thresholds_after = {
        "pseudo_label_initial_score_thr":
            float(
                model.semi_train_cfg
                .pseudo_label_initial_score_thr
            ),
        "rpn_pseudo_thr":
            float(model.semi_train_cfg.rpn_pseudo_thr),
        "cls_pseudo_thr":
            float(model.semi_train_cfg.cls_pseudo_thr),
        "reg_pseudo_thr":
            float(model.semi_train_cfg.reg_pseudo_thr),
        "unsup_weight":
            float(model.semi_train_cfg.unsup_weight),
    }

    checks["controlled_nonempty_pseudo_count_is_4"] = (
        sum(synthetic_counts) == 4
        and synthetic_counts == [1, 1, 1, 1]
    )
    checks["controlled_normal_pseudo_required_loss_keys_present"] = (
        REQUIRED_NORMAL_PSEUDO_LOSS_KEYS
        .issubset(controlled_loss_keys)
    )
    checks["controlled_normal_pseudo_loss_finite"] = (
        controlled_total_finite
        and math.isfinite(controlled_total_value)
    )
    checks["controlled_normal_pseudo_loss_requires_grad"] = (
        controlled_total.requires_grad
    )
    checks["controlled_normal_pseudo_student_gradient_positive"] = (
        controlled_student_grad_seen
        and math.isfinite(
            controlled_student_grad_abs_sum
        )
        and controlled_student_grad_abs_sum > 0.0
    )
    checks["controlled_normal_pseudo_teacher_has_no_gradient"] = (
        not controlled_teacher_grad_seen
        and controlled_teacher_grad_param_count == 0
        and controlled_teacher_grad_abs_sum == 0.0
    )
    checks["canonical_thresholds_not_modified_by_runtime_fixture"] = (
        thresholds_before == thresholds_after
    )

    checks["empty_pseudo_branch_has_prior_canonical_pass"] = (
        prerequisite_evidence[
            "empty_pseudo_batch"
        ]["pass"]
    )

    checks["hidden_u_gt_firewall_has_prior_canonical_pass"] = (
        prerequisite_evidence[
            "hidden_u_firewall"
        ]["pass"]
    )

    checks["resume_equivalence_has_prior_canonical_pass"] = (
        prerequisite_evidence[
            "ssl_resume_equivalence"
        ]["pass"]
    )

    checks["amp_ema_skip_has_prior_canonical_pass"] = (
        prerequisite_evidence[
            "amp_ema_skip"
        ]["pass"]
    )

    failed_checks = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    report = {
        "schema_version": "1.0",
        "stage": "S4.19",
        "scope": "ssl_swin_t_execution_path_preflight",
        "official_training_authorized": False,
        "final_test_authorized": False,
        "swin_checkpoint": {
            "path": str(swin_checkpoint_path),
            "exists": swin_checkpoint_exists,
            "sha256": swin_checkpoint_sha256,
            "expected_sha256":
                EXPECTED_SWIN_CHECKPOINT_SHA256,
        },
        "config_matrix": config_results,
        "prerequisite_evidence": prerequisite_evidence,
        "runtime": {
            "device":
                str(next(model.student.parameters()).device),
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "raw_branch_counts": branch_counts,
            "raw_branch_indices": branch_indices,
            "raw_branch_image_ids": branch_image_ids,
            "raw_three_identical_channel_checks":
                raw_channel_checks,
            "processed_input_shapes":
                processed_shapes,
            "processed_sample_counts":
                processed_counts,
            "teacher_t0_equals_student_t0":
                teacher_equals_student_t0,
            "student_frozen_backbone_param_count":
                len(student_frozen_backbone),
            "teacher_trainable_param_count":
                len(teacher_trainable),
            "student_swin_signature":
                student_swin_signature,
            "teacher_swin_signature":
                teacher_swin_signature,
            "expected_swin_signature":
                expected_runtime_swin_signature,
        },
        "official_ssl_smoke": {
            "loss_terms": official_loss_terms,
            "total_loss": official_total_value,
            "total_loss_finite":
                official_loss_finite,
            "student_grad_seen":
                official_student_grad_seen,
            "student_grad_abs_sum":
                official_student_grad_abs_sum,
            "student_grad_param_count":
                official_student_grad_param_count,
            "teacher_grad_seen":
                official_teacher_grad_seen,
            "teacher_grad_abs_sum":
                official_teacher_grad_abs_sum,
            "teacher_grad_param_count":
                official_teacher_grad_param_count,
        },
        "official_threshold_pseudo_probe": {
            "canonical_thresholds":
                thresholds_before,
            "teacher_pseudo_counts":
                teacher_pseudo_counts,
            "total_teacher_pseudo":
                sum(teacher_pseudo_counts),
            "projected_pseudo_counts":
                projected_pseudo_counts,
            "total_projected_pseudo":
                sum(projected_pseudo_counts),
            "normal_pseudo_available_at_t0":
                sum(projected_pseudo_counts) > 0,
            "canonical_thresholds_modified":
                False,
        },
        "controlled_normal_pseudo_smoke": {
            "test_only_synthetic_pseudo":
                True,
            "synthetic_pseudo_counts":
                synthetic_counts,
            "total_synthetic_pseudo":
                sum(synthetic_counts),
            "loss_terms":
                controlled_loss_terms,
            "total_loss":
                controlled_total_value,
            "total_loss_finite":
                controlled_total_finite,
            "student_grad_seen":
                controlled_student_grad_seen,
            "student_grad_abs_sum":
                controlled_student_grad_abs_sum,
            "student_grad_param_count":
                controlled_student_grad_param_count,
            "teacher_grad_seen":
                controlled_teacher_grad_seen,
            "teacher_grad_abs_sum":
                controlled_teacher_grad_abs_sum,
            "teacher_grad_param_count":
                controlled_teacher_grad_param_count,
            "canonical_thresholds_before":
                thresholds_before,
            "canonical_thresholds_after":
                thresholds_after,
            "optimizer_step_executed":
                False,
            "ema_update_executed":
                False,
        },
        "deferred_preofficial_requirements": {
            "ssl_entrypoint_manifest": (
                "Required by artifact contract before "
                "official training; not owned by S4.19."
            ),
            "s4_stage_closure": (
                "Owned by S4.20."
            ),
            "global_preflight_report": (
                "Not closed by S4.19."
            ),
        },
        "checks": checks,
        "check_count": len(checks),
        "failed_checks": failed_checks,
        "all_checks_pass": not failed_checks,
        "overall_pass": not failed_checks,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n===== CHECKS =====")
    for name, passed in checks.items():
        print(f"{name} = {passed}")

    print("check_count =", len(checks))
    print("failed_checks =", failed_checks)
    print("ALL_CHECKS_PASS =", not failed_checks)
    print("report =", output_path)

    if failed_checks:
        print("S4_19_SSL_SWIN_PREFLIGHT=FAIL")
        return 1

    print("S4_19_SSL_SWIN_PREFLIGHT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())