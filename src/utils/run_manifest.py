"""S2.11 utilities for canonical run-manifest identity."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


def canonical_json(obj: Any) -> str:
    """Deterministic semantic serialization used by locked protocol code."""
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def semantic_sha256(obj: Any) -> str:
    """SHA-256 of canonical JSON semantic content."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """SHA-256 of exact file bytes."""
    digest = hashlib.sha256()

    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))

    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping: {path}")

    return data


def git_head(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
    ).strip()


def governance_identity(repo_root: Path) -> dict[str, str]:
    governance_root = repo_root / "artifacts" / "governance"

    scientific = load_json(
        governance_root / "scientific_source_manifest.json"
    )
    implementation = load_json(
        governance_root / "implementation_contract_manifest.json"
    )
    artifact = load_json(
        governance_root / "artifact_contract_manifest.json"
    )
    schema = load_json(
        governance_root / "schema_version.json"
    )

    return {
        "schema_version": schema["schema_version"],
        "scientific_source_name": scientific["scientific_source_name"],
        "scientific_source_sha256": scientific["scientific_source_sha256"],
        "implementation_contract_sha256": implementation[
            "implementation_contract_sha256"
        ],
        "artifact_contract_sha256": artifact[
            "artifact_contract_sha256"
        ],
    }


def fixed_data_identity(
    repo_root: Path,
    *,
    budget: str,
) -> dict[str, str]:
    split_path = (
        repo_root / "data" / "manifests" / "split_lock_manifest.json"
    )
    phase2f_path = (
        repo_root / "data" / "manifests" / "phase2F_lock_manifest.json"
    )

    split = load_json(split_path)
    phase2f = load_json(phase2f_path)

    if budget not in {"1pct", "5pct", "10pct", "20pct"}:
        raise ValueError(f"Unsupported low-label budget: {budget}")

    return {
        "dataset_manifest_sha256": file_sha256(split_path),
        "train_split_sha256": split["image_id_sha256"]["train"],
        "val_split_sha256": split["image_id_sha256"]["val"],
        "test_split_sha256": split["image_id_sha256"]["test"],
        "labeled_manifest_sha256": phase2f["coco_json_sha256"][
            "labeled"
        ][budget],
        "unlabeled_manifest_sha256": phase2f["coco_json_sha256"][
            "unlabeled"
        ][budget],
    }


def preprocessing_identity(repo_root: Path) -> tuple[dict[str, Any], str]:
    representation = load_yaml(
        repo_root
        / "configs"
        / "protocol"
        / "phase2D1_jpg_representation.yaml"
    )
    training = load_yaml(
        repo_root
        / "configs"
        / "protocol"
        / "d4_training_protocol.yaml"
    )

    fragment = {
        "jpg_storage": representation["output_channel_policy"][
            "jpg_storage"
        ],
        "model_input_channels": {
            "channels": representation["output_channel_policy"][
                "mmdetection_model_input"
            ]["channels"],
            "replicate_grayscale_in_loader": representation[
                "output_channel_policy"
            ]["mmdetection_model_input"][
                "replicate_grayscale_in_loader"
            ],
        },
        "training_input": training["input"],
    }

    return fragment, semantic_sha256(fragment)


def labeled_pipeline_identity(
    repo_root: Path,
) -> tuple[list[dict[str, Any]], str]:
    import importlib.util

    config_path = (
        repo_root / "configs" / "dataset" / "s2_coco_dataset.py"
    )

    spec = importlib.util.spec_from_file_location(
        "s2_coco_dataset_run_manifest",
        config_path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import dataset config: {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    pipeline = module.PIPELINE

    return pipeline, semantic_sha256(pipeline)


def optimizer_identity(
    repo_root: Path,
    *,
    architecture: str,
) -> tuple[dict[str, Any], str]:
    training = load_yaml(
        repo_root
        / "configs"
        / "protocol"
        / "d4_training_protocol.yaml"
    )

    key_map = {
        "R50-FPN": "r50",
        "Swin-T-FPN": "swint",
    }

    if architecture not in key_map:
        raise ValueError(f"Unsupported architecture: {architecture}")

    optimizer = training["architectures"][
        key_map[architecture]
    ]["optimizer"]

    return optimizer, semantic_sha256(optimizer)


def evaluator_identity(
    repo_root: Path,
) -> tuple[dict[str, Any], str]:
    evaluator = load_yaml(
        repo_root
        / "configs"
        / "protocol"
        / "evaluator_protocol.yaml"
    )

    return evaluator, semantic_sha256(evaluator)
