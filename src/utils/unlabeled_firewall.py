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
    pipeline: Any,
) -> bool:
    """Recursively detect LoadAnnotations, including nested MultiBranch."""
    if isinstance(pipeline, dict):
        transform_type = str(
            pipeline.get("type", "")
        ).lower()

        if transform_type == "loadannotations":
            return True

        return any(
            pipeline_exposes_annotations(value)
            for value in pipeline.values()
        )

    if isinstance(pipeline, (list, tuple)):
        return any(
            pipeline_exposes_annotations(value)
            for value in pipeline
        )

    return False

def assert_unlabeled_pipeline_isolated(
    pipeline: Any,
) -> None:
    if pipeline_exposes_annotations(pipeline):
        raise HiddenUGroundTruthError(
            "Hidden-U firewall blocked pipeline containing "
            "LoadAnnotations."
        )

CURRENT_S4_SSL_CONFIGS = {
    ("R50-FPN", "1pct"):
        "configs/ssl/"
        "s4_14_soft_teacher_r50_fpn_empty_pseudo_1pct.py",
    ("R50-FPN", "5pct"):
        "configs/ssl/"
        "s4_14_soft_teacher_r50_fpn_empty_pseudo_5pct.py",
    ("R50-FPN", "10pct"):
        "configs/ssl/"
        "s4_14_soft_teacher_r50_fpn_empty_pseudo_10pct.py",
    ("R50-FPN", "20pct"):
        "configs/ssl/"
        "s4_14_soft_teacher_r50_fpn_empty_pseudo_20pct.py",
    ("Swin-T-FPN", "1pct"):
        "configs/ssl/"
        "s4_14_soft_teacher_swin_t_fpn_empty_pseudo_1pct.py",
    ("Swin-T-FPN", "5pct"):
        "configs/ssl/"
        "s4_14_soft_teacher_swin_t_fpn_empty_pseudo_5pct.py",
    ("Swin-T-FPN", "10pct"):
        "configs/ssl/"
        "s4_14_soft_teacher_swin_t_fpn_empty_pseudo_10pct.py",
    ("Swin-T-FPN", "20pct"):
        "configs/ssl/"
        "s4_14_soft_teacher_swin_t_fpn_empty_pseudo_20pct.py",
}


def _walk_dataset_nodes(obj: Any) -> list[dict[str, Any]]:
    nodes = []

    if isinstance(obj, dict):
        if "ann_file" in obj:
            nodes.append(obj)

        for value in obj.values():
            nodes.extend(_walk_dataset_nodes(value))

    elif isinstance(obj, (list, tuple)):
        for value in obj:
            nodes.extend(_walk_dataset_nodes(value))

    return nodes


def _resolve_unlabeled_ann_path(
    repo_root: Path,
    configured_path: str,
) -> Path:
    configured = Path(configured_path)

    candidates = [
        configured,
        Path("/workspace/ssod/data/coco") / configured.name,
        (
            repo_root
            / "data"
            / "processed"
            / "coco"
            / "unlabeled_splits"
            / configured.name
        ),
        repo_root / "data" / "coco" / configured.name,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise HiddenUGroundTruthError(
        "Current SSL unlabeled COCO file cannot be resolved: "
        f"{configured_path}"
    )


def assert_current_ssl_execution_firewall(
    repo_root: Path,
    *,
    architecture: str,
    budget: str,
) -> dict[str, Any]:
    """Fail-fast hidden-U firewall for current S4 SSL execution path."""
    from mmengine.config import Config

    repo_root = Path(repo_root).resolve()
    key = (architecture, budget)

    if key not in CURRENT_S4_SSL_CONFIGS:
        raise HiddenUGroundTruthError(
            "Unsupported current S4 SSL firewall selector: "
            f"{key!r}"
        )

    config_rel = CURRENT_S4_SSL_CONFIGS[key]
    config_path = repo_root / config_rel

    if not config_path.is_file():
        raise HiddenUGroundTruthError(
            f"Current SSL config does not exist: {config_path}"
        )

    cfg = Config.fromfile(str(config_path))

    dataset_nodes = _walk_dataset_nodes(
        cfg.train_dataloader.dataset
    )

    unlabeled_nodes = [
        node
        for node in dataset_nodes
        if "instances_unlabeled_"
        in str(node.get("ann_file", ""))
    ]

    if len(unlabeled_nodes) != 1:
        raise HiddenUGroundTruthError(
            "Current SSL config must expose exactly one unlabeled "
            f"dataset node; observed={len(unlabeled_nodes)}"
        )

    node = unlabeled_nodes[0]
    expected_name = f"instances_unlabeled_{budget}.json"
    configured_ann = str(node["ann_file"])

    if Path(configured_ann).name != expected_name:
        raise HiddenUGroundTruthError(
            "Current SSL config points to wrong unlabeled split: "
            f"expected={expected_name}, "
            f"observed={Path(configured_ann).name}"
        )

    assert_unlabeled_pipeline_isolated(
        node.get("pipeline", [])
    )

    ann_path = _resolve_unlabeled_ann_path(
        repo_root,
        configured_ann,
    )

    source_report = assert_unlabeled_coco_isolated(
        ann_path
    )

    return {
        "architecture": architecture,
        "budget": budget,
        "config_path": str(config_path),
        "configured_ann_file": configured_ann,
        "resolved_ann_file": str(ann_path),
        "annotation_count":
            source_report["annotation_count"],
        "image_count":
            source_report["image_count"],
        "category_count":
            source_report["category_count"],
        "recursive_pipeline_isolated": True,
        "unlabeled_source_isolated":
            source_report["all_checks_pass"],
        "all_checks_pass": True,
    }
