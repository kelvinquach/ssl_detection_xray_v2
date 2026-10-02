"""Materialize S9 official data_provenance.json from runtime data/config."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.utils.run_manifest import (
    file_sha256,
    load_json,
    semantic_sha256,
)


class OfficialDataProvenanceError(RuntimeError):
    """Raised when official runtime data provenance is invalid."""


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        tmp = Path(tmp_name)
        if tmp.exists():
            tmp.unlink()


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _plain(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise OfficialDataProvenanceError(
        "Unsupported resolved dataloader value type: "
        f"{type(value).__name__}"
    )


def _load_coco(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise OfficialDataProvenanceError(
            f"Required runtime COCO file is missing: {path}"
        )
    payload = load_json(path)
    images = payload.get("images")
    annotations = payload.get("annotations")
    categories = payload.get("categories")
    if not isinstance(images, list):
        raise OfficialDataProvenanceError(
            f"COCO images must be a list: {path}"
        )
    if not isinstance(annotations, list):
        raise OfficialDataProvenanceError(
            f"COCO annotations must be a list: {path}"
        )
    if not isinstance(categories, list):
        raise OfficialDataProvenanceError(
            f"COCO categories must be a list: {path}"
        )
    return payload


def _image_ids(payload: dict[str, Any]) -> list[int]:
    values: list[int] = []
    for item in payload["images"]:
        try:
            values.append(int(item["id"]))
        except Exception as exc:
            raise OfficialDataProvenanceError(
                "COCO image id is not an integer identity."
            ) from exc
    if len(values) != len(set(values)):
        raise OfficialDataProvenanceError(
            "Duplicate image IDs detected in runtime COCO."
        )
    return values


def _phase2e_membership_sha256(values: list[int]) -> str:
    canonical = "\n".join(
        str(value)
        for value in sorted(values, key=lambda value: str(value))
    )
    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def _phase2f_membership_sha256(values: list[int]) -> str:
    canonical = "\n".join(
        str(value)
        for value in sorted(int(value) for value in values)
    )
    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def _collect_dataset_nodes(value: Any) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []

    if isinstance(value, Mapping):
        plain = dict(value)
        if "ann_file" in plain or "data_prefix" in plain:
            nodes.append(plain)
        for item in value.values():
            nodes.extend(_collect_dataset_nodes(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            nodes.extend(_collect_dataset_nodes(item))

    return nodes


def _contains_transform_type(value: Any, target: str) -> bool:
    if isinstance(value, Mapping):
        if value.get("type") == target:
            return True
        return any(
            _contains_transform_type(item, target)
            for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(
            _contains_transform_type(item, target)
            for item in value
        )
    return False


def _resolved_loader_identity(
    *,
    repo_root: Path,
    cfg: Any,
) -> tuple[str, str, list[dict[str, Any]]]:
    train_loader = getattr(cfg, "train_dataloader", None)
    if train_loader is None:
        raise OfficialDataProvenanceError(
            "Resolved config has no train_dataloader."
        )

    plain_loader = _plain(train_loader)
    if not isinstance(plain_loader, dict) or not plain_loader:
        raise OfficialDataProvenanceError(
            "Resolved train_dataloader must be a non-empty mapping."
        )

    nodes = _collect_dataset_nodes(plain_loader)
    roots: set[str] = set()
    for node in nodes:
        data_prefix = node.get("data_prefix")
        if isinstance(data_prefix, Mapping):
            image_root = data_prefix.get("img")
            if isinstance(image_root, str) and image_root:
                roots.add(image_root)

    if len(roots) != 1:
        raise OfficialDataProvenanceError(
            "Official train loader must resolve exactly one image root; "
            f"observed={sorted(roots)}"
        )

    image_root = next(iter(roots))
    expected_root = str(
        (repo_root / "data" / "mmdet_images").resolve()
    )
    if str(Path(image_root)) != expected_root:
        raise OfficialDataProvenanceError(
            "Resolved training image root differs from canonical runtime root."
        )

    return semantic_sha256(plain_loader), image_root, nodes


def materialize_official_data_provenance(
    *,
    repo_root: Path,
    attempt_dir: Path,
    expected: dict[str, Any],
    cfg: Any,
) -> Path:
    """Write and verify one official data_provenance.json."""
    repo_root = Path(repo_root).resolve()
    attempt_dir = Path(attempt_dir).resolve()

    frozen_path = (
        repo_root / "artifacts/frozen/frozen_data_manifest.json"
    )
    frozen = load_json(frozen_path)
    if frozen.get("status") != "PASS":
        raise OfficialDataProvenanceError(
            "Frozen data manifest is not PASS."
        )

    coco_root = repo_root / "data" / "coco"
    train_path = coco_root / "instances_train.json"
    val_path = coco_root / "instances_val.json"
    test_path = coco_root / "instances_test.json"

    train = _load_coco(train_path)
    val = _load_coco(val_path)
    test = _load_coco(test_path)

    train_ids = _image_ids(train)
    val_ids = _image_ids(val)
    test_ids = _image_ids(test)

    frozen_splits = {
        "train": (train, train_ids),
        "val": (val, val_ids),
        "test": (test, test_ids),
    }
    for name, (payload, ids) in frozen_splits.items():
        frozen_split = frozen[name]
        if len(ids) != int(frozen_split["image_count"]):
            raise OfficialDataProvenanceError(
                f"{name} image count differs from frozen identity."
            )
        if (
            _phase2e_membership_sha256(ids)
            != frozen_split["membership_sha256"]
        ):
            raise OfficialDataProvenanceError(
                f"{name} membership differs from frozen identity."
            )
        path = coco_root / f"instances_{name}.json"
        if file_sha256(path) != frozen_split["coco_sha256"]:
            raise OfficialDataProvenanceError(
                f"{name} COCO hash differs from frozen identity."
            )

    category_count = int(
        frozen["dataset_identity"]["category_count"]
    )
    if category_count != 14:
        raise OfficialDataProvenanceError(
            "Frozen category count must equal 14."
        )

    loader_sha256, image_root, dataset_nodes = (
        _resolved_loader_identity(
            repo_root=repo_root,
            cfg=cfg,
        )
    )

    method = str(expected["method"])
    budget = str(expected["budget"])
    condition_role = str(expected["condition_role"])

    if condition_role == "SUP_REFERENCE":
        labeled_path = train_path
        labeled = train
        labeled_ids = train_ids
        labeled_count = len(train_ids)
        unlabeled_count: int | str = "NOT_APPLICABLE"
        labeled_membership_sha256 = (
            _phase2e_membership_sha256(train_ids)
        )
        labeled_no_finding_count = int(
            frozen["train"]["no_finding_count"]
        )
    else:
        if budget not in frozen["budgets"]:
            raise OfficialDataProvenanceError(
                f"Unsupported frozen low-label budget: {budget}"
            )

        budget_info = frozen["budgets"][budget]
        labeled_path = (
            coco_root / f"instances_labeled_{budget}.json"
        )
        labeled = _load_coco(labeled_path)
        labeled_ids = _image_ids(labeled)
        labeled_count = len(labeled_ids)
        unlabeled_count = int(
            budget_info["unlabeled_image_count"]
        )
        labeled_membership_sha256 = (
            _phase2f_membership_sha256(labeled_ids)
        )
        labeled_no_finding_count = int(
            budget_info["labeled_no_finding_count"]
        )

        if labeled_count != int(
            budget_info["labeled_image_count"]
        ):
            raise OfficialDataProvenanceError(
                "Labeled image count differs from frozen budget."
            )
        if (
            labeled_membership_sha256
            != budget_info["labeled_membership_sha256"]
        ):
            raise OfficialDataProvenanceError(
                "Labeled membership differs from frozen budget."
            )

    if file_sha256(labeled_path) != expected[
        "labeled_manifest_sha256"
    ]:
        raise OfficialDataProvenanceError(
            "Runtime labeled COCO differs from expected manifest identity."
        )

    payload: dict[str, Any] = {
        "condition_role": condition_role,
        "train_image_count": len(train_ids),
        "val_image_count": len(val_ids),
        "test_image_count": len(test_ids),
        "budget": budget,
        "labeled_image_count": labeled_count,
        "unlabeled_image_count": unlabeled_count,
        "labeled_no_finding_count": labeled_no_finding_count,
        "category_count": category_count,
        "train_membership_sha256":
            frozen["train"]["membership_sha256"],
        "val_membership_sha256":
            frozen["val"]["membership_sha256"],
        "test_membership_sha256":
            frozen["test"]["membership_sha256"],
        "labeled_membership_sha256":
            labeled_membership_sha256,
        "dataset_loader_config_sha256":
            loader_sha256,
        "preprocessing_config_sha256":
            expected["preprocessing_config_sha256"],
        "image_root_identity": image_root,
        "hidden_u_annotations_present_in_training_object":
            False,
    }

    ann_files = {
        str(node.get("ann_file"))
        for node in dataset_nodes
        if isinstance(node.get("ann_file"), str)
    }

    if method == "SSL":
        unlabeled_path = (
            coco_root / f"instances_unlabeled_{budget}.json"
        )
        unlabeled = _load_coco(unlabeled_path)
        unlabeled_ids = _image_ids(unlabeled)

        if len(unlabeled_ids) != int(unlabeled_count):
            raise OfficialDataProvenanceError(
                "Unlabeled image count differs from frozen budget."
            )
        if file_sha256(unlabeled_path) != expected[
            "unlabeled_manifest_sha256"
        ]:
            raise OfficialDataProvenanceError(
                "Runtime unlabeled COCO differs from expected manifest identity."
            )
        if unlabeled.get("annotations"):
            raise OfficialDataProvenanceError(
                "Hidden-U annotations exist in runtime unlabeled COCO."
            )

        labeled_set = set(labeled_ids)
        unlabeled_set = set(unlabeled_ids)
        train_set = set(train_ids)

        disjoint = not bool(labeled_set & unlabeled_set)
        union_equals_train = (
            labeled_set | unlabeled_set
        ) == train_set
        exact_complement = (
            unlabeled_set == train_set - labeled_set
        )

        if not disjoint:
            raise OfficialDataProvenanceError(
                "Labeled and unlabeled memberships overlap."
            )
        if not union_equals_train or not exact_complement:
            raise OfficialDataProvenanceError(
                "Runtime U_b is not the exact complement of L_b."
            )

        expected_u_ann = str(unlabeled_path)
        matching_u_nodes = [
            node
            for node in dataset_nodes
            if str(node.get("ann_file")) == expected_u_ann
        ]
        if len(matching_u_nodes) != 1:
            raise OfficialDataProvenanceError(
                "Resolved SSL loader must expose exactly one U_b dataset."
            )
        if _contains_transform_type(
            matching_u_nodes[0].get("pipeline"),
            "LoadAnnotations",
        ):
            raise OfficialDataProvenanceError(
                "SSL unlabeled pipeline exposes LoadAnnotations."
            )

        payload.update(
            {
                "unlabeled_membership_sha256":
                    _phase2f_membership_sha256(
                        unlabeled_ids
                    ),
                "labeled_unlabeled_disjoint": True,
                "labeled_unlabeled_union_equals_train": True,
            }
        )
    elif method == "SUP":
        if any(
            "instances_unlabeled_" in ann_file
            for ann_file in ann_files
        ):
            raise OfficialDataProvenanceError(
                "SUP train loader unexpectedly references U_b."
            )
    else:
        raise OfficialDataProvenanceError(
            f"Unsupported official method: {method!r}"
        )

    output = attempt_dir / "data_provenance.json"
    _write_json_atomic(output, payload)
    return output


__all__ = [
    "OfficialDataProvenanceError",
    "materialize_official_data_provenance",
]
