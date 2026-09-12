#!/usr/bin/env python3
"""S4.07 — validate unlabeled strong view -> Student augmentation contract."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import runpy
import sys
from pathlib import Path

import numpy as np
from mmengine.config import Config


CONFIGS = {
    "r50": "configs/ssl/s4_07_soft_teacher_r50_fpn_strong.py",
    "swin": "configs/ssl/s4_07_soft_teacher_swin_t_fpn_strong.py",
}

EXPECTED_IMPORTS = {
    "src.hooks.teacher_initialization_hook",
    "src.hooks.actual_update_mean_teacher_hook",
    "src.utils.resume_checkpoint_hook",
    "src.transforms.grayscale_photometric",
}

PROHIBITED_STRONG = {
    "RandomFlip",
    "RandomCrop",
    "RandomResize",
    "RandomRotate",
    "Rotate",
    "Mosaic",
    "RandomErasing",
    "PhotoMetricDistortion",
    "RandAugment",
    "Color",
    "AutoContrast",
    "Equalize",
    "Sharpness",
    "Posterize",
    "Solarize",
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def channels_equal_chw(tensor) -> bool:
    x = tensor.detach().cpu().numpy()
    return bool(
        x.ndim == 3
        and x.shape[0] == 3
        and np.array_equal(x[0], x[1])
        and np.array_equal(x[1], x[2])
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default="/workspace/ssod/project",
    )
    parser.add_argument(
        "--output",
        default=(
            "/workspace/ssod/project/artifacts/preflight/ssl/"
            "ssl_augmentation_manifest.json"
        ),
    )
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    output = Path(args.output).resolve()
    sys.path.insert(0, str(repo))

    from mmdet.registry import DATASETS, TRANSFORMS
    from mmdet.utils import register_all_modules
    from src.utils.run_manifest import semantic_sha256

    register_all_modules()
    import src.transforms.grayscale_photometric  # noqa: F401

    if not output.exists():
        raise FileNotFoundError(
            "S4.06 ssl_augmentation_manifest.json must exist before S4.07."
        )

    existing = json.loads(output.read_text(encoding="utf-8"))
    prior_manifest_sha256 = file_sha256(output)

    checks: dict[str, bool] = {}

    # Preserve and verify the already-closed S4.06 weak evidence.
    checks["s4_06_prior_task"] = existing.get("task") == "S4.06"
    checks["s4_06_prior_scope"] = (
        existing.get("scope") == "unlabeled_weak_view_to_teacher"
    )
    checks["s4_06_prior_all_checks_pass"] = (
        existing.get("all_checks_pass") is True
    )
    checks["s4_06_prior_failed_checks_empty"] = (
        existing.get("failed_checks") == []
    )
    checks["s4_06_prior_teacher_branch"] = (
        existing.get("weak_branch_key") == "unsup_teacher"
    )
    checks["s4_06_prior_teacher_destination"] = (
        existing.get("weak_view_destination") == "EMA_TEACHER"
    )
    checks["s4_06_prior_hidden_u_gt_unused"] = (
        existing.get("runtime", {}).get("gt_bbox_count") == 0
    )

    prior_weak_id = existing.get("weak_augmentation_id")
    checks["s4_06_prior_weak_id_present"] = bool(prior_weak_id)

    config_reports = {}
    strong_ids = set()
    resolved_weak_ids = set()

    for arch, relpath in CONFIGS.items():
        cfg = Config.fromfile(str(repo / relpath))

        imports = set(cfg.custom_imports["imports"])
        weak = cfg.weak_pipeline
        strong = cfg.strong_pipeline
        outer = cfg.unlabeled_pipeline

        weak_types = [x["type"] for x in weak]
        strong_types = [x["type"] for x in strong]

        weak_resize = next(x for x in weak if x["type"] == "Resize")
        strong_resize = next(x for x in strong if x["type"] == "Resize")
        strong_aug = next(
            x for x in strong
            if x["type"] == "GrayscaleBrightnessContrast"
        )
        weak_pack = next(
            x for x in weak if x["type"] == "PackDetInputs"
        )
        strong_pack = next(
            x for x in strong if x["type"] == "PackDetInputs"
        )
        multibranch = next(
            x for x in outer if x["type"] == "MultiBranch"
        )

        weak_id = semantic_sha256(weak)
        strong_id = semantic_sha256(strong)

        resolved_weak_ids.add(weak_id)
        strong_ids.add(strong_id)

        ema_hooks = [
            h for h in cfg.custom_hooks
            if h.get("type") == "ActualUpdateMeanTeacherHook"
        ]

        arch_checks = {
            "custom_imports_exact":
                imports == EXPECTED_IMPORTS,
            "model_is_softteacher":
                cfg.model.type == "SoftTeacher",
            "multi_branch_preprocessor":
                cfg.model.data_preprocessor.type
                == "MultiBranchDataPreprocessor",
            "branch_field_exact":
                list(cfg.branch_field)
                == ["sup", "unsup_teacher", "unsup_student"],
            "weak_pipeline_preserved":
                weak_types == ["Resize", "Pad", "PackDetInputs"],
            "strong_transform_order_exact":
                strong_types == [
                    "Resize",
                    "GrayscaleBrightnessContrast",
                    "Pad",
                    "PackDetInputs",
                ],
            "weak_resize_exact":
                tuple(weak_resize["scale"]) == (1333, 800)
                and weak_resize["keep_ratio"] is True,
            "strong_resize_exact":
                tuple(strong_resize["scale"]) == (1333, 800)
                and strong_resize["keep_ratio"] is True,
            "weak_strong_geometry_config_match":
                tuple(weak_resize["scale"])
                == tuple(strong_resize["scale"])
                and weak_resize["keep_ratio"]
                == strong_resize["keep_ratio"],
            "brightness_range_exact":
                tuple(strong_aug["brightness_range"])
                == (0.90, 1.10),
            "contrast_range_exact":
                tuple(strong_aug["contrast_range"])
                == (0.90, 1.10),
            "no_prohibited_strong_transform":
                not any(x["type"] in PROHIBITED_STRONG for x in strong),
            "weak_homography_retained":
                "homography_matrix"
                in tuple(weak_pack["meta_keys"]),
            "strong_homography_retained":
                "homography_matrix"
                in tuple(strong_pack["meta_keys"]),
            "teacher_branch_present":
                "unsup_teacher" in multibranch,
            "student_branch_present":
                "unsup_student" in multibranch,
            "amp_preserved":
                cfg.optim_wrapper.type == "AmpOptimWrapper",
            "one_ema_hook":
                len(ema_hooks) == 1,
            "ema_momentum_preserved":
                len(ema_hooks) == 1
                and ema_hooks[0].get("momentum") == 0.001,
            "ema_skip_buffers_preserved":
                len(ema_hooks) == 1
                and ema_hooks[0].get("skip_buffers") is True,
        }

        for name, passed in arch_checks.items():
            checks[f"{arch}_{name}"] = bool(passed)

        config_reports[arch] = {
            "config": relpath,
            "weak_augmentation_id": weak_id,
            "strong_augmentation_id": strong_id,
            "branch_field": list(cfg.branch_field),
            "weak_pipeline": weak,
            "strong_pipeline": strong,
            "unlabeled_pipeline": outer,
            "checks": arch_checks,
        }

    checks["architectures_share_one_weak_id"] = (
        len(resolved_weak_ids) == 1
    )
    checks["architectures_share_one_strong_id"] = (
        len(strong_ids) == 1
    )

    resolved_weak_id = (
        next(iter(resolved_weak_ids))
        if len(resolved_weak_ids) == 1
        else None
    )
    strong_id = (
        next(iter(strong_ids))
        if len(strong_ids) == 1
        else None
    )

    checks["s4_06_weak_id_unchanged"] = (
        resolved_weak_id == prior_weak_id
    )

    # Explicit continuous-uniform sampling validation.
    transform = TRANSFORMS.build(
        dict(
            type="GrayscaleBrightnessContrast",
            brightness_range=(0.90, 1.10),
            contrast_range=(0.90, 1.10),
        )
    )

    np_state = np.random.get_state()
    np.random.seed(2026)
    try:
        brightness_samples = [
            transform._sample_brightness_factor()
            for _ in range(1000)
        ]
        contrast_samples = [
            transform._sample_contrast_factor()
            for _ in range(1000)
        ]
    finally:
        np.random.set_state(np_state)

    sampling = {
        "sample_size": 1000,
        "validation_seed": 2026,
        "brightness_min": min(brightness_samples),
        "brightness_max": max(brightness_samples),
        "brightness_unique_count": len(set(brightness_samples)),
        "contrast_min": min(contrast_samples),
        "contrast_max": max(contrast_samples),
        "contrast_unique_count": len(set(contrast_samples)),
    }

    checks["brightness_all_in_range"] = all(
        0.90 <= x <= 1.10 for x in brightness_samples
    )
    checks["contrast_all_in_range"] = all(
        0.90 <= x <= 1.10 for x in contrast_samples
    )
    checks["brightness_continuous_not_three_level"] = (
        len(set(brightness_samples)) > 3
    )
    checks["contrast_continuous_not_three_level"] = (
        len(set(contrast_samples)) > 3
    )

    # Real locked-U runtime fixture.
    dataset_cfg = runpy.run_path(
        str(repo / "configs/dataset/s2_coco_dataset.py")
    )
    u_cfg = copy.deepcopy(dataset_cfg["DATASETS"]["U_1pct"])

    runtime_cfg = Config.fromfile(
        str(repo / CONFIGS["r50"])
    )
    u_cfg["pipeline"] = copy.deepcopy(runtime_cfg.unlabeled_pipeline)

    np_state = np.random.get_state()
    np.random.seed(2026)
    try:
        dataset = DATASETS.build(u_cfg)
        item = dataset[0]
    finally:
        np.random.set_state(np_state)

    inputs = item["inputs"]
    samples = item["data_samples"]

    weak_input = inputs["unsup_teacher"]
    strong_input = inputs["unsup_student"]
    weak_sample = samples["unsup_teacher"]
    strong_sample = samples["unsup_student"]

    weak_h = np.asarray(weak_sample.homography_matrix)
    strong_h = np.asarray(strong_sample.homography_matrix)

    weak_gt_count = len(weak_sample.gt_instances.bboxes)
    strong_gt_count = len(strong_sample.gt_instances.bboxes)

    runtime = {
        "dataset_key": "U_1pct",
        "image_id": int(weak_sample.img_id),
        "teacher_branch": "unsup_teacher",
        "student_branch": "unsup_student",
        "same_image_id":
            weak_sample.img_id == strong_sample.img_id,
        "same_image_path":
            weak_sample.img_path == strong_sample.img_path,
        "weak_input_shape_chw": list(weak_input.shape),
        "strong_input_shape_chw": list(strong_input.shape),
        "weak_rgb_equal": channels_equal_chw(weak_input),
        "strong_rgb_equal": channels_equal_chw(strong_input),
        "strong_differs_from_weak": not np.array_equal(
            weak_input.detach().cpu().numpy(),
            strong_input.detach().cpu().numpy(),
        ),
        "weak_ori_shape": list(weak_sample.ori_shape),
        "strong_ori_shape": list(strong_sample.ori_shape),
        "weak_img_shape": list(weak_sample.img_shape),
        "strong_img_shape": list(strong_sample.img_shape),
        "weak_scale_factor": list(weak_sample.scale_factor),
        "strong_scale_factor": list(strong_sample.scale_factor),
        "weak_homography_matrix": weak_h.tolist(),
        "strong_homography_matrix": strong_h.tolist(),
        "homography_equal": bool(np.allclose(weak_h, strong_h)),
        "weak_gt_bbox_count": int(weak_gt_count),
        "strong_gt_bbox_count": int(strong_gt_count),
        "hidden_u_ground_truth_used": bool(
            weak_gt_count != 0 or strong_gt_count != 0
        ),
        "sup_branch_is_none": inputs["sup"] is None,
    }

    checks.update({
        "runtime_teacher_branch_exists":
            weak_input is not None,
        "runtime_student_branch_exists":
            strong_input is not None,
        "runtime_same_u_image":
            runtime["same_image_id"]
            and runtime["same_image_path"],
        "runtime_same_shape":
            tuple(weak_input.shape)
            == tuple(strong_input.shape),
        "runtime_weak_rgb_equal":
            runtime["weak_rgb_equal"],
        "runtime_strong_rgb_equal":
            runtime["strong_rgb_equal"],
        "runtime_strong_differs_photometrically":
            runtime["strong_differs_from_weak"],
        "runtime_img_shape_equal":
            tuple(weak_sample.img_shape)
            == tuple(strong_sample.img_shape),
        "runtime_scale_factor_equal":
            tuple(weak_sample.scale_factor)
            == tuple(strong_sample.scale_factor),
        "runtime_homography_equal":
            runtime["homography_equal"],
        "runtime_hidden_u_gt_not_used":
            runtime["hidden_u_ground_truth_used"] is False,
        "runtime_sup_branch_none":
            runtime["sup_branch_is_none"] is True,
    })

    failed = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    report = copy.deepcopy(existing)

    # Preserve explicit historical S4.06 identity before extending the same
    # canonical augmentation artifact for S4.07.
    report["schema_version"] = "1.1"
    report["stage"] = "S4"
    report["task"] = "S4.07"
    report["scope"] = (
        "unlabeled_weak_teacher_and_strong_student_views"
    )
    report["completed_tasks"] = ["S4.06", "S4.07"]
    report["official_training_authorized"] = False

    report["tracker_contract"] = {
        "category": "Unlabeled strong",
        "action": "Strong view -> Student",
        "constraint": (
            "brightness/contrast U(.90,1.10), grayscale R=G=B."
        ),
        "expected_result": "MATCH",
    }

    report["s4_06_evidence_snapshot"] = {
        "prior_manifest_sha256": prior_manifest_sha256,
        "task": existing.get("task"),
        "scope": existing.get("scope"),
        "weak_augmentation_id": prior_weak_id,
        "weak_view_destination":
            existing.get("weak_view_destination"),
        "weak_branch_key":
            existing.get("weak_branch_key"),
        "check_count": existing.get("check_count"),
        "failed_checks": existing.get("failed_checks"),
        "all_checks_pass": existing.get("all_checks_pass"),
    }

    report["weak_augmentation_id"] = prior_weak_id
    report["weak_view_destination"] = "EMA_TEACHER"
    report["weak_branch_key"] = "unsup_teacher"

    report["strong_augmentation_id"] = strong_id
    report["strong_view_destination"] = "STUDENT"
    report["strong_branch_key"] = "unsup_student"
    report["strong_view_status"] = "VALIDATED_S4_07"

    report["strong_scientific_contract"] = {
        "resize_scale": [1333, 800],
        "keep_ratio": True,
        "brightness_distribution": "Uniform(0.90,1.10)",
        "contrast_distribution": "Uniform(0.90,1.10)",
        "grayscale_constraint": "R=G=B",
        "pad_size_divisor": 32,
        "geometric_augmentation": False,
        "random_erasing": False,
        "mosaic": False,
        "hidden_u_ground_truth_used": False,
    }

    report["strong_configs"] = config_reports
    report["strong_sampling_validation"] = sampling
    report["strong_runtime"] = runtime

    report["weak_checks"] = existing.get("checks", {})
    report["weak_check_count"] = existing.get("check_count", 0)
    report["strong_checks"] = checks
    report["strong_check_count"] = len(checks)

    combined_checks = {}
    for name, value in report["weak_checks"].items():
        combined_checks[f"s4_06_{name}"] = bool(value)
    for name, value in checks.items():
        combined_checks[f"s4_07_{name}"] = bool(value)

    combined_failed = [
        name
        for name, passed in combined_checks.items()
        if not passed
    ]

    report["checks"] = combined_checks
    report["check_count"] = len(combined_checks)
    report["failed_checks"] = combined_failed
    report["all_checks_pass"] = not combined_failed

    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("EVIDENCE=", output)
    print("PRIOR_S4_06_MANIFEST_SHA256=", prior_manifest_sha256)
    print("WEAK_AUGMENTATION_ID=", prior_weak_id)
    print("RESOLVED_WEAK_AUGMENTATION_ID=", resolved_weak_id)
    print("STRONG_AUGMENTATION_ID=", strong_id)
    print("STRONG_VIEW_DESTINATION=STUDENT")
    print("STRONG_BRANCH_KEY=unsup_student")
    print("STRONG_CHECK_COUNT=", len(checks))
    print("TOTAL_CHECK_COUNT=", len(combined_checks))
    print("FAILED_CHECKS=", combined_failed)
    print(
        "RUNTIME_SAME_U_IMAGE=",
        runtime["same_image_id"] and runtime["same_image_path"],
    )
    print(
        "RUNTIME_STRONG_RGB_EQUAL=",
        runtime["strong_rgb_equal"],
    )
    print(
        "RUNTIME_HOMOGRAPHY_EQUAL=",
        runtime["homography_equal"],
    )
    print(
        "RUNTIME_HIDDEN_U_GT_USED=",
        runtime["hidden_u_ground_truth_used"],
    )

    if combined_failed:
        print("S4_07_SSL_STRONG_AUGMENTATION=FAIL")
        return 1

    print("S4_07_TRACKER_RESULT=MATCH")
    print("S4_07_SSL_STRONG_AUGMENTATION=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())