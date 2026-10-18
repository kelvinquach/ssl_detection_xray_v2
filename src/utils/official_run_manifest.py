"""Materialize canonical S9 official run_manifest.json."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from src.utils.run_manifest import git_head, load_json, load_yaml


SCHEMA_VERSION = "1.1"
ENTRYPOINT_ID = "scripts/S9_run_official.py"
DOCKER_EVIDENCE = Path(
    "artifacts/preflight/environment/docker_image_identity.json"
)
SSL_AUGMENTATION_EVIDENCE = Path(
    "artifacts/preflight/ssl/ssl_augmentation_manifest.json"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class OfficialRunManifestError(RuntimeError):
    """Raised when canonical official run-manifest evidence is invalid."""


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


def _runtime_docker_identity(repo_root: Path) -> tuple[str, str]:
    path = repo_root / DOCKER_EVIDENCE
    if not path.is_file():
        raise OfficialRunManifestError(
            f"Canonical Docker runtime evidence is missing: {path}"
        )

    docker = load_json(path)
    if docker.get("status") != "PASS":
        raise OfficialRunManifestError(
            "Docker runtime evidence status is not PASS."
        )

    checks = docker.get("checks")
    expected = docker.get("expected")
    observed = docker.get("observed")
    if not isinstance(checks, dict):
        raise OfficialRunManifestError(
            "Docker runtime evidence checks are missing."
        )
    if not isinstance(expected, dict):
        raise OfficialRunManifestError(
            "Docker runtime expected identity is missing."
        )
    if not isinstance(observed, dict):
        raise OfficialRunManifestError(
            "Docker runtime observed identity is missing."
        )

    if checks.get("vast_image_uuid_match") is not True:
        raise OfficialRunManifestError(
            "Vast image UUID runtime verification did not PASS."
        )
    if checks.get("registry_digest_match") is not True:
        raise OfficialRunManifestError(
            "Docker registry digest runtime verification did not PASS."
        )

    image_uuid = expected.get("image_uuid")
    digest = expected.get("digest")
    if not isinstance(image_uuid, str) or not image_uuid.strip():
        raise OfficialRunManifestError(
            "Expected Docker image UUID is missing."
        )
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise OfficialRunManifestError(
            "Expected Docker digest is invalid."
        )

    if observed.get("vast_image_uuid") != image_uuid:
        raise OfficialRunManifestError(
            "Observed Vast image UUID differs from expected identity."
        )
    if observed.get("registry_digest") != digest:
        raise OfficialRunManifestError(
            "Observed Docker registry digest differs from expected identity."
        )

    return image_uuid, digest


def _ssl_fields(repo_root: Path) -> dict[str, Any]:
    ssl = load_yaml(
        repo_root / "configs/protocol/d4_ssl_protocol.yaml"
    )
    augmentation_path = repo_root / SSL_AUGMENTATION_EVIDENCE
    if not augmentation_path.is_file():
        raise OfficialRunManifestError(
            f"SSL augmentation identity evidence is missing: {augmentation_path}"
        )
    augmentation = load_json(augmentation_path)

    weak_id = augmentation.get("weak_augmentation_id")
    strong_id = augmentation.get("strong_augmentation_id")
    if (
        not isinstance(weak_id, str)
        or SHA256_RE.fullmatch(weak_id) is None
    ):
        raise OfficialRunManifestError(
            "weak_augmentation_id is not a valid SHA-256 identity."
        )
    if (
        not isinstance(strong_id, str)
        or SHA256_RE.fullmatch(strong_id) is None
    ):
        raise OfficialRunManifestError(
            "strong_augmentation_id is not a valid SHA-256 identity."
        )

    ema = ssl["ema"]
    pseudo = ssl["pseudo_labels"]
    proposals = ssl["proposals"]
    regression = ssl["regression_filter"]
    loss = ssl["loss"]
    batch = ssl["effective_batch"]

    return {
        "ema_momentum": float(ema["mean_teacher_hook_momentum"]),
        "ema_skip_buffers": bool(ema["skip_buffers"]),
        "ema_update_basis": "ACTUAL_OPTIMIZER_UPDATE",
        "labeled_unlabeled_ratio": str(batch["ratio"]),
        "unsupervised_loss_weight": float(
            loss["unsupervised_weight"]
        ),
        "weak_augmentation_id": weak_id,
        "strong_augmentation_id": strong_id,
        "detector_score_floor": float(
            pseudo["detector_score_floor"]
        ),
        "initial_pseudo_score": float(
            pseudo["initial_score_threshold"]
        ),
        "rpn_pseudo_threshold": float(
            pseudo["rpn_threshold"]
        ),
        "rcnn_classification_threshold": float(
            pseudo["classification_threshold"]
        ),
        "rpn_nms_iou": float(proposals["rpn_nms_iou"]),
        "rpn_max": int(proposals["rpn_max"]),
        "rcnn_nms_iou": float(proposals["rcnn_nms_iou"]),
        "rcnn_max": int(proposals["rcnn_max"]),
        "jitter_times": int(regression["jitter_times"]),
        "jitter_scale": float(regression["jitter_scale"]),
        "reg_uncertainty_threshold": float(
            regression["uncertainty_threshold"]
        ),
        "min_pseudo_bbox_wh": [
            float(value)
            for value in regression["min_bbox_wh"]
        ],
    }


def materialize_official_run_manifest(
    *,
    repo_root: Path,
    attempt_dir: Path,
    run_id: str,
    attempt_id: str,
    expected: dict[str, Any],
) -> Path:
    """Write one canonical RUNNING-state official run manifest."""
    repo_root = Path(repo_root).resolve()
    attempt_dir = Path(attempt_dir).resolve()

    method = str(expected["method"])
    if method not in {"SUP", "SSL"}:
        raise OfficialRunManifestError(
            f"Unsupported official method: {method!r}"
        )

    training = load_yaml(
        repo_root / "configs/protocol/d4_training_protocol.yaml"
    )
    docker_image, docker_digest = _runtime_docker_identity(
        repo_root
    )

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "run_type": "OFFICIAL",
        "condition_role": expected["condition_role"],
        "method": method,
        "architecture": expected["architecture"],
        "budget": expected["budget"],
        "seed_index": int(expected["seed_index"]),
        "training_seed": int(expected["training_seed"]),
        "partition_seed": int(expected["partition_seed"]),

        "scientific_source_name":
            expected["scientific_source_name"],
        "scientific_source_sha256":
            expected["scientific_source_sha256"],
        "implementation_contract_sha256":
            expected["implementation_contract_sha256"],
        "artifact_contract_sha256":
            expected["artifact_contract_sha256"],

        "git_commit": git_head(repo_root),
        "docker_image": docker_image,
        "docker_digest": docker_digest,
        "entrypoint_id": ENTRYPOINT_ID,

        "dataset_manifest_sha256":
            expected["dataset_manifest_sha256"],
        "train_split_sha256":
            expected["train_split_sha256"],
        "val_split_sha256":
            expected["val_split_sha256"],
        "test_split_sha256":
            expected["test_split_sha256"],
        "labeled_manifest_sha256":
            expected["labeled_manifest_sha256"],
        "unlabeled_manifest_sha256":
            expected["unlabeled_manifest_sha256"],

        "preprocessing_config_sha256":
            expected["preprocessing_config_sha256"],
        "labeled_pipeline_sha256":
            expected["labeled_pipeline_sha256"],
        "optimizer_recipe_sha256":
            expected["optimizer_recipe_sha256"],
        "evaluator_config_sha256":
            expected["evaluator_config_sha256"],

        "optimizer": expected["optimizer"],
        "learning_rate": expected["learning_rate"],
        "weight_decay": expected["weight_decay"],
        "total_optimizer_updates":
            int(expected["total_optimizer_updates"]),
        "validation_interval_updates":
            int(expected["validation_interval_updates"]),
        "effective_labeled_batch":
            int(expected["effective_labeled_batch"]),
        "effective_unlabeled_batch":
            expected["effective_unlabeled_batch"],
        "amp": bool(expected["amp"]),
        "gradient_clipping": bool(
            training["numerical"]["gradient_clipping"]
        ),

        "checkpoint_selection_metric":
            expected["checkpoint_selection_metric"],
        "test_usage": expected["test_usage"],
        "training_status": "RUNNING",
        "final_evaluation_status":
            expected["final_evaluation_status"],
    }

    if method == "SUP":
        if manifest["effective_unlabeled_batch"] != "NOT_APPLICABLE":
            raise OfficialRunManifestError(
                "SUP effective_unlabeled_batch must be NOT_APPLICABLE."
            )
        if manifest["unlabeled_manifest_sha256"] != "NOT_APPLICABLE":
            raise OfficialRunManifestError(
                "SUP unlabeled_manifest_sha256 must be NOT_APPLICABLE."
            )
    else:
        manifest["effective_unlabeled_batch"] = int(
            manifest["effective_unlabeled_batch"]
        )
        manifest.update(_ssl_fields(repo_root))

    output = attempt_dir / "run_manifest.json"
    _write_json_atomic(output, manifest)
    return output


__all__ = [
    "OfficialRunManifestError",
    "materialize_official_run_manifest",
]
