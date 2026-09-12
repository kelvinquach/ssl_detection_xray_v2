#!/usr/bin/env python3
"""S4.06 — validate unlabeled weak view -> EMA Teacher contract."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

from mmengine.config import Config


CONFIGS = {
    "r50": "configs/ssl/s4_06_soft_teacher_r50_fpn_weak.py",
    "swin": "configs/ssl/s4_06_soft_teacher_swin_t_fpn_weak.py",
}

PROHIBITED = {
    "RandomFlip",
    "RandomCrop",
    "RandomRotate",
    "Rotate",
    "PhotoMetricDistortion",
    "RandAugment",
    "ColorTransform",
    "AutoContrast",
    "Equalize",
    "Sharpness",
    "Posterize",
    "Solarize",
    "Color",
    "Contrast",
    "Brightness",
    "Mosaic",
    "RandomErasing",
}


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

    from mmdet.registry import DATASETS
    from mmdet.utils import register_all_modules
    from src.utils.run_manifest import semantic_sha256

    register_all_modules()

    checks: dict[str, bool] = {}
    config_reports = {}

    weak_ids = set()

    for arch, relpath in CONFIGS.items():
        cfg = Config.fromfile(str(repo / relpath))

        dp = cfg.model.data_preprocessor
        weak = cfg.weak_pipeline
        unsup = cfg.unlabeled_weak_pipeline

        weak_types = [x["type"] for x in weak]
        resize = next(x for x in weak if x["type"] == "Resize")
        pad = next(x for x in weak if x["type"] == "Pad")
        pack = next(x for x in weak if x["type"] == "PackDetInputs")
        multibranch = next(
            x for x in unsup if x["type"] == "MultiBranch"
        )

        weak_id = semantic_sha256(weak)
        weak_ids.add(weak_id)

        arch_checks = {
            "model_is_softteacher":
                cfg.model.type == "SoftTeacher",
            "multi_branch_preprocessor":
                dp.type == "MultiBranchDataPreprocessor",
            "inner_standard_preprocessor":
                dp.data_preprocessor.type == "DetDataPreprocessor",
            "branch_field_exact":
                list(cfg.branch_field)
                == ["sup", "unsup_teacher", "unsup_student"],
            "weak_transform_order_exact":
                weak_types == ["Resize", "Pad", "PackDetInputs"],
            "resize_scale_1333_800":
                tuple(resize["scale"]) == (1333, 800),
            "resize_keep_ratio":
                resize["keep_ratio"] is True,
            "pad_size_divisor_32":
                pad["size_divisor"] == 32,
            "homography_retained":
                "homography_matrix" in tuple(pack["meta_keys"]),
            "no_prohibited_weak_transform":
                not any(x["type"] in PROHIBITED for x in weak),
            "weak_routed_to_unsup_teacher":
                "unsup_teacher" in multibranch,
            "strong_not_implemented_in_s4_06":
                "unsup_student" not in multibranch,
        }

        for name, passed in arch_checks.items():
            checks[f"{arch}_{name}"] = bool(passed)

        config_reports[arch] = {
            "config": relpath,
            "weak_augmentation_id": weak_id,
            "branch_field": list(cfg.branch_field),
            "weak_pipeline": weak,
            "unlabeled_weak_pipeline": unsup,
            "standard_preprocessing": {
                "type": dp.data_preprocessor.type,
                "mean": list(dp.data_preprocessor.mean),
                "std": list(dp.data_preprocessor.std),
                "bgr_to_rgb": bool(
                    dp.data_preprocessor.bgr_to_rgb
                ),
                "pad_size_divisor": int(
                    dp.data_preprocessor.pad_size_divisor
                ),
            },
            "checks": arch_checks,
        }

    checks["architecture_weak_ids_match"] = len(weak_ids) == 1
    weak_id = next(iter(weak_ids)) if len(weak_ids) == 1 else None

    # Runtime fixture: build the locked U_1pct dataset with the S4.06
    # unlabeled weak pipeline. Hidden-U GT remains unused.
    import runpy
    dataset_cfg = runpy.run_path(
        str(repo / "configs/dataset/s2_coco_dataset.py")
    )
    u_cfg = copy.deepcopy(dataset_cfg["DATASETS"]["U_1pct"])

    reference_cfg = Config.fromfile(
        str(repo / CONFIGS["r50"])
    )
    u_cfg["pipeline"] = copy.deepcopy(
        reference_cfg.unlabeled_weak_pipeline
    )

    dataset = DATASETS.build(u_cfg)
    item = dataset[0]

    inputs = item["inputs"]
    samples = item["data_samples"]

    teacher_input = inputs["unsup_teacher"]
    teacher_sample = samples["unsup_teacher"]

    runtime = {
        "dataset_key": "U_1pct",
        "image_id": int(teacher_sample.img_id),
        "teacher_branch": "unsup_teacher",
        "teacher_input_shape_chw": list(teacher_input.shape),
        "ori_shape": list(teacher_sample.ori_shape),
        "img_shape": list(teacher_sample.img_shape),
        "scale_factor": list(teacher_sample.scale_factor),
        "homography_matrix": (
            teacher_sample.homography_matrix.tolist()
            if hasattr(teacher_sample, "homography_matrix")
            else None
        ),
        "gt_bbox_count": int(
            len(teacher_sample.gt_instances.bboxes)
        ),
        "sup_branch_is_none": inputs["sup"] is None,
        "unsup_student_branch_is_none":
            inputs["unsup_student"] is None,
    }

    checks.update({
        "runtime_unsup_teacher_tensor_exists":
            teacher_input is not None,
        "runtime_homography_present":
            runtime["homography_matrix"] is not None,
        "runtime_hidden_u_gt_not_loaded":
            runtime["gt_bbox_count"] == 0,
        "runtime_sup_branch_none":
            runtime["sup_branch_is_none"] is True,
        "runtime_unsup_student_branch_none_in_s4_06":
            runtime["unsup_student_branch_is_none"] is True,
    })

    failed = [
        name for name, passed in checks.items() if not passed
    ]

    report = {
        "schema_version": "1.0",
        "stage": "S4",
        "task": "S4.06",
        "scope": "unlabeled_weak_view_to_teacher",
        "official_training_authorized": False,
        "tracker_contract": {
            "category": "Unlabeled weak",
            "action": "Weak view -> Teacher",
            "constraint": (
                "Resize + standard preprocessing + padding; "
                "no extra augmentation."
            ),
            "expected_result": "MATCH",
        },
        "weak_augmentation_id": weak_id,
        "weak_view_destination": "EMA_TEACHER",
        "weak_branch_key": "unsup_teacher",
        "strong_augmentation_id": None,
        "strong_view_status": "NOT_IN_S4_06_SCOPE",
        "scientific_contract": {
            "resize_scale": [1333, 800],
            "keep_ratio": True,
            "pad_size_divisor": 32,
            "flip": False,
            "crop": False,
            "rotation": False,
            "photometric_augmentation": False,
            "hidden_u_ground_truth_used": False,
        },
        "configs": config_reports,
        "runtime": runtime,
        "checks": checks,
        "check_count": len(checks),
        "failed_checks": failed,
        "all_checks_pass": not failed,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("EVIDENCE=", output)
    print("WEAK_AUGMENTATION_ID=", weak_id)
    print("CHECK_COUNT=", len(checks))
    print("FAILED_CHECKS=", failed)
    print(
        "RUNTIME_TEACHER_BRANCH=",
        runtime["teacher_branch"],
    )
    print(
        "RUNTIME_HOMOGRAPHY_PRESENT=",
        runtime["homography_matrix"] is not None,
    )
    print(
        "RUNTIME_HIDDEN_U_GT_USED=",
        runtime["gt_bbox_count"] != 0,
    )

    if failed:
        print("S4_06_SSL_WEAK_AUGMENTATION=FAIL")
        return 1

    print("S4_06_SSL_WEAK_AUGMENTATION=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())