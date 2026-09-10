from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path


EXPECTED_PIPELINE_SHA256 = (
    "1a3a61907a17c60df390c1add85e42a1f024708b9941bfd80d22f94226db8de4"
)

EXPECTED_PIPELINE = [
    dict(type="LoadImageFromFile", color_type="color"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(type="Pad", size_divisor=32),
    dict(type="PackDetInputs"),
]

EXPECTED_RUNTIME_TRANSFORMS = [
    "LoadImageFromFile",
    "LoadAnnotations",
    "Resize",
    "Pad",
    "PackDetInputs",
]

PROHIBITED_TRANSFORMS = {
    "RandomFlip",
    "RandomCrop",
    "RandomResize",
    "PhotoMetricDistortion",
    "Mosaic",
    "RandomRotate",
    "Rotate",
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
            "/workspace/ssod/artifacts/preflight/sup/"
            "supervised_augmentation_manifest.json"
        ),
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_path = Path(args.output).resolve()

    sys.path.insert(0, str(repo_root))
    from src.utils.run_manifest import labeled_pipeline_identity

    from mmdet.registry import DATASETS
    from mmdet.utils import register_all_modules

    register_all_modules()

    pipeline, pipeline_sha256 = labeled_pipeline_identity(repo_root)

    dataset_cfg = runpy.run_path(
        str(repo_root / "configs/dataset/s2_coco_dataset.py")
    )
    dataset = DATASETS.build(dataset_cfg["DATASETS"]["L_1pct"])

    runtime_transforms = list(dataset.pipeline.transforms)
    runtime_names = [t.__class__.__name__ for t in runtime_transforms]
    runtime_repr = [str(t) for t in runtime_transforms]

    resize = runtime_transforms[2] if len(runtime_transforms) > 2 else None
    pad = runtime_transforms[3] if len(runtime_transforms) > 3 else None

    prohibited_present = [
        name for name in runtime_names if name in PROHIBITED_TRANSFORMS
    ]

    checks = {
        "labeled_pipeline_sha256_match": (
            pipeline_sha256 == EXPECTED_PIPELINE_SHA256
        ),
        "pipeline_exact_definition_match": pipeline == EXPECTED_PIPELINE,
        "runtime_transform_count_is_5": len(runtime_names) == 5,
        "runtime_transform_order_match": (
            runtime_names == EXPECTED_RUNTIME_TRANSFORMS
        ),
        "resize_scale_is_1333_800": (
            resize is not None
            and tuple(resize.scale) == (1333, 800)
        ),
        "resize_keep_ratio_true": (
            resize is not None
            and resize.keep_ratio is True
        ),
        "pad_size_divisor_is_32": (
            pad is not None
            and pad.size_divisor == 32
        ),
        "prohibited_augmentation_absent": not prohibited_present,
        "supervised_pipeline_has_no_stochastic_augmentation": (
            not prohibited_present
            and runtime_names == EXPECTED_RUNTIME_TRANSFORMS
        ),
        "official_training_remains_unauthorized": True,
    }

    failed_checks = [
        name for name, passed in checks.items() if not passed
    ]

    report = {
        "schema_version": "1.0",
        "stage": "S3.06",
        "scope": "supervised_augmentation_pipeline_manifest",
        "official_training_authorized": False,
        "scientific_contract": {
            "pipeline": "Load -> Resize -> Pack",
            "load_annotations_is_supervised_detection_plumbing": True,
            "resize": {
                "scale": [1333, 800],
                "keep_ratio": True,
            },
            "pad": {
                "size_divisor": 32,
            },
            "default_flip": False,
            "default_crop": False,
            "default_rotation": False,
            "default_mosaic": False,
            "default_aggressive_photometric_augmentation": False,
        },
        "pipeline_identity": {
            "source": "configs/dataset/s2_coco_dataset.py::PIPELINE",
            "labeled_pipeline_sha256": pipeline_sha256,
            "expected_labeled_pipeline_sha256": EXPECTED_PIPELINE_SHA256,
            "pipeline": pipeline,
        },
        "augmentation_rng_strategy": (
            "DETERMINISTIC_SUP_PIPELINE_NO_STOCHASTIC_AUGMENTATION"
        ),
        "runtime_pipeline": {
            "dataset_key": "L_1pct",
            "compose_class": dataset.pipeline.__class__.__name__,
            "transform_count": len(runtime_names),
            "transform_names": runtime_names,
            "transform_repr": runtime_repr,
            "prohibited_transforms_present": prohibited_present,
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
        print("S3_06_SUPERVISED_AUGMENTATION=FAIL")
        return 1

    print("S3_06_SUPERVISED_AUGMENTATION=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
