"""S2.15 hidden-U ground-truth isolation foundation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class HiddenUGroundTruthError(RuntimeError):
    """Raised when hidden unlabeled ground truth is exposed."""


FORBIDDEN_IMAGE_KEY_TOKENS = (
    "annotation",
    "bbox",
    "ground",
    "label",
    "class_presence",
    "finding",
    "negative",
)

ALLOWED_UNLABELED_IMAGE_KEYS = {
    "id",
    "file_name",
    "width",
    "height",
    "canonical_image_id",
    "original_image_id",
}


def load_unlabeled_coco(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()

    if not path.is_file():
        raise HiddenUGroundTruthError(
            f"Unlabeled COCO file does not exist: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise HiddenUGroundTruthError(
            "Unlabeled COCO root must be a JSON object."
        )

    return data


def inspect_unlabeled_coco(path: Path) -> dict[str, Any]:
    data = load_unlabeled_coco(path)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    if not isinstance(images, list):
        raise HiddenUGroundTruthError(
            "Unlabeled COCO images must be a list."
        )

    if not isinstance(annotations, list):
        raise HiddenUGroundTruthError(
            "Unlabeled COCO annotations must be a list."
        )

    if not isinstance(categories, list):
        raise HiddenUGroundTruthError(
            "Unlabeled COCO categories must be a list."
        )

    observed_image_keys = sorted({
        key
        for image in images
        if isinstance(image, dict)
        for key in image.keys()
    })

    forbidden_image_keys = sorted({
        key
        for key in observed_image_keys
        if (
            key not in ALLOWED_UNLABELED_IMAGE_KEYS
            and any(
                token in key.lower()
                for token in FORBIDDEN_IMAGE_KEY_TOKENS
            )
        )
    })

    unexpected_image_keys = sorted(
        set(observed_image_keys)
        - ALLOWED_UNLABELED_IMAGE_KEYS
    )

    image_ids = [
        image.get("id")
        for image in images
        if isinstance(image, dict)
    ]

    checks = {
        "annotations_empty": annotations == [],
        "no_forbidden_image_level_gt_keys":
            forbidden_image_keys == [],
        "no_unexpected_image_keys":
            unexpected_image_keys == [],
        "all_images_are_objects":
            all(isinstance(image, dict) for image in images),
        "all_images_have_id":
            all(
                isinstance(image, dict) and "id" in image
                for image in images
            ),
        "image_ids_unique":
            len(image_ids) == len(set(image_ids)),
    }

    return {
        "path": str(Path(path).resolve()),
        "image_count": len(images),
        "annotation_count": len(annotations),
        "category_count": len(categories),
        "observed_image_keys": observed_image_keys,
        "forbidden_image_keys": forbidden_image_keys,
        "unexpected_image_keys": unexpected_image_keys,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }


def assert_unlabeled_coco_isolated(path: Path) -> dict[str, Any]:
    report = inspect_unlabeled_coco(path)

    if not report["all_checks_pass"]:
        failed = [
            name
            for name, passed in report["checks"].items()
            if not passed
        ]
        raise HiddenUGroundTruthError(
            "Hidden-U firewall blocked unlabeled COCO: "
            + ", ".join(failed)
        )

    return report


def pipeline_exposes_annotations(
    pipeline: list[dict[str, Any]],
) -> bool:
    for transform in pipeline:
        if not isinstance(transform, dict):
            continue

        transform_type = str(
            transform.get("type", "")
        ).lower()

        if transform_type == "loadannotations":
            return True

    return False


def assert_unlabeled_pipeline_isolated(
    pipeline: list[dict[str, Any]],
) -> None:
    if pipeline_exposes_annotations(pipeline):
        raise HiddenUGroundTruthError(
            "Hidden-U firewall blocked pipeline containing "
            "LoadAnnotations."
        )