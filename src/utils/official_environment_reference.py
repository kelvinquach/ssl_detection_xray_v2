"""Per-attempt official runtime-environment provenance.

Fresh attempts snapshot the canonical live Vast runtime evidence. Same-attempt
recovery preserves the original snapshot and appends the newly verified runtime
session without silently overwriting prior provenance.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class OfficialEnvironmentReferenceError(RuntimeError):
    """Raised when official environment evidence cannot be trusted."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise OfficialEnvironmentReferenceError(
            f"Cannot parse runtime evidence {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise OfficialEnvironmentReferenceError(
            f"Runtime evidence must be a JSON object: {path}"
        )
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise OfficialEnvironmentReferenceError(
            f"Runtime evidence field {key!r} must be an object."
        )
    return value


def _text(parent: dict[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise OfficialEnvironmentReferenceError(
            f"Runtime evidence field {key!r} must be non-empty text."
        )
    return value.strip()


def _positive_int(parent: dict[str, Any], key: str) -> int:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OfficialEnvironmentReferenceError(
            f"Runtime evidence field {key!r} must be a positive integer."
        )
    return int(value)


def _current_snapshot(repo_root: Path) -> dict[str, Any]:
    env_dir = repo_root / "artifacts" / "preflight" / "environment"
    vast_path = env_dir / "vast_environment.json"
    docker_path = env_dir / "docker_image_identity.json"
    if not vast_path.is_file() or not docker_path.is_file():
        raise OfficialEnvironmentReferenceError(
            "Canonical live Vast runtime evidence is missing. Run the frozen "
            "runtime verifier on the execution instance before official training."
        )

    vast = _load_json(vast_path)
    docker = _load_json(docker_path)

    if vast.get("artifact_type") != "VAST_RUNTIME_ENVIRONMENT":
        raise OfficialEnvironmentReferenceError(
            "Unexpected vast_environment artifact_type."
        )
    if vast.get("status") != "PASS":
        raise OfficialEnvironmentReferenceError(
            "Canonical vast_environment.json is not PASS."
        )
    if docker.get("artifact_type") != "DOCKER_IMAGE_IDENTITY":
        raise OfficialEnvironmentReferenceError(
            "Unexpected docker_image_identity artifact_type."
        )
    if docker.get("status") != "PASS":
        raise OfficialEnvironmentReferenceError(
            "Canonical docker_image_identity.json is not PASS."
        )

    checks = _mapping(docker, "checks")
    if checks.get("vast_image_uuid_match") is not True:
        raise OfficialEnvironmentReferenceError(
            "Docker Vast image UUID verification is not PASS."
        )
    if checks.get("registry_digest_match") is not True:
        raise OfficialEnvironmentReferenceError(
            "Docker registry digest verification is not PASS."
        )

    instance = _mapping(vast, "instance")
    os_info = _mapping(vast, "os")
    python_info = _mapping(vast, "python")
    pytorch = _mapping(vast, "pytorch")
    gpu = _mapping(vast, "gpu")
    openmmlab = _mapping(vast, "openmmlab")
    vast_docker = _mapping(vast, "docker")
    docker_expected = _mapping(docker, "expected")
    docker_observed = _mapping(docker, "observed")

    if pytorch.get("cuda_available") is not True:
        raise OfficialEnvironmentReferenceError(
            "Canonical runtime evidence reports CUDA unavailable."
        )
    cuda_device_count = _positive_int(pytorch, "cuda_device_count")
    gpu_memory_mib = _positive_int(gpu, "memory_total_mib")

    expected_digest = _text(docker_expected, "digest")
    observed_digest = _text(docker_observed, "registry_digest")
    if expected_digest != observed_digest:
        raise OfficialEnvironmentReferenceError(
            "Docker expected/observed registry digest mismatch."
        )
    if (
        vast_docker.get("digest_match") is not True
        or vast_docker.get("expected_digest") != expected_digest
        or vast_docker.get("observed_registry_digest") != observed_digest
    ):
        raise OfficialEnvironmentReferenceError(
            "vast_environment Docker digest evidence is inconsistent."
        )

    expected_uuid = _text(docker_expected, "image_uuid")
    observed_uuid = _text(docker_observed, "vast_image_uuid")
    if expected_uuid != observed_uuid:
        raise OfficialEnvironmentReferenceError(
            "Docker expected/observed Vast image UUID mismatch."
        )

    repository = _text(docker_expected, "repository")
    tag = _text(docker_expected, "tag")
    visible = gpu.get("cuda_visible_devices")
    if not isinstance(visible, str) or not visible.strip():
        visible = gpu.get("vast_device_idxs")
    if not isinstance(visible, str) or not visible.strip():
        raise OfficialEnvironmentReferenceError(
            "CUDA visible-device identity is unavailable."
        )

    cudnn_version = pytorch.get("cudnn_version")
    if cudnn_version in (None, ""):
        raise OfficialEnvironmentReferenceError(
            "cuDNN version is unavailable."
        )

    runtime = {
        "os": {
            "pretty_name": os_info.get("pretty_name"),
            "kernel": _text(os_info, "kernel"),
            "platform": _text(os_info, "platform"),
        },
        "python": {
            "version": _text(python_info, "version"),
        },
        "pytorch": {
            "version": _text(pytorch, "version"),
            "cuda_runtime": _text(pytorch, "cuda_runtime"),
            "cuda_available": True,
            "cuda_device_count": cuda_device_count,
            "cudnn_version": cudnn_version,
            "cudnn_enabled": pytorch.get("cudnn_enabled"),
        },
        "openmmlab": {
            "mmdet": _text(openmmlab, "mmdet"),
            "mmcv": _text(openmmlab, "mmcv"),
            "mmengine": _text(openmmlab, "mmengine"),
        },
        "gpu": {
            "name": _text(gpu, "name"),
            "memory_total_mib": gpu_memory_mib,
            "nvidia_driver": _text(gpu, "nvidia_driver"),
            "driver_supported_cuda": gpu.get("driver_supported_cuda"),
            "cuda_visible_device": visible.strip(),
        },
        "docker": {
            "image": f"{repository}:{tag}",
            "repository": repository,
            "tag": tag,
            "digest": observed_digest,
            "image_uuid": observed_uuid,
        },
    }

    identity_bytes = json.dumps(
        runtime,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    runtime_identity_sha256 = hashlib.sha256(identity_bytes).hexdigest()

    return {
        "runtime_identity_sha256": runtime_identity_sha256,
        "runtime": runtime,
        "instance": {
            "hostname": _text(instance, "hostname"),
            "container_id": instance.get("container_id"),
            "container_label": instance.get("container_label"),
            "vast_actual_status": instance.get("vast_actual_status"),
        },
        "source_evidence": {
            "vast_environment_path": (
                "artifacts/preflight/environment/vast_environment.json"
            ),
            "vast_environment_sha256": _sha256(vast_path),
            "vast_environment_captured_at_utc": vast.get("captured_at_utc"),
            "docker_image_identity_path": (
                "artifacts/preflight/environment/docker_image_identity.json"
            ),
            "docker_image_identity_sha256": _sha256(docker_path),
            "docker_identity_verified_at_utc": docker.get("verified_at_utc"),
            "runtime_verifier_sha256": vast.get("verifier", {}).get(
                "script_sha256"
            ),
        },
    }


def materialize_official_environment_reference(
    *,
    repo_root: Path,
    attempt_dir: Path,
    run_id: str,
    attempt_id: str,
    recovery: bool = False,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    attempt_dir = Path(attempt_dir).resolve()
    output = attempt_dir / "environment_reference.json"
    current = _current_snapshot(repo_root)

    if recovery:
        if not output.is_file():
            raise OfficialEnvironmentReferenceError(
                "Recovered attempt is missing environment_reference.json."
            )
        payload = _load_json(output)
        if payload.get("artifact_type") != "OFFICIAL_ENVIRONMENT_REFERENCE":
            raise OfficialEnvironmentReferenceError(
                "Recovered environment_reference artifact_type mismatch."
            )
        if payload.get("run_id") != str(run_id):
            raise OfficialEnvironmentReferenceError(
                "Recovered environment_reference run_id mismatch."
            )
        if payload.get("attempt_id") != str(attempt_id):
            raise OfficialEnvironmentReferenceError(
                "Recovered environment_reference attempt_id mismatch."
            )
        original = payload.get("original_environment")
        sessions = payload.get("recovery_sessions")
        if not isinstance(original, dict) or not isinstance(sessions, list):
            raise OfficialEnvironmentReferenceError(
                "Recovered environment_reference provenance is incomplete."
            )
        original_identity = original.get("runtime_identity_sha256")
        if not isinstance(original_identity, str) or len(original_identity) != 64:
            raise OfficialEnvironmentReferenceError(
                "Recovered original runtime identity is invalid."
            )
        sessions.append(
            {
                "recovery_index": len(sessions) + 1,
                "recorded_at_utc": _utc_now(),
                "same_runtime_identity_as_original": (
                    current["runtime_identity_sha256"] == original_identity
                ),
                "environment": current,
            }
        )
        payload["last_updated_at_utc"] = _utc_now()
        _write_json_atomic(output, payload)
        return payload

    if output.exists():
        raise OfficialEnvironmentReferenceError(
            "Fresh attempt unexpectedly already contains environment_reference.json."
        )

    payload = {
        "schema_version": "1.0",
        "artifact_type": "OFFICIAL_ENVIRONMENT_REFERENCE",
        "run_id": str(run_id),
        "attempt_id": str(attempt_id),
        "materialized_at_utc": _utc_now(),
        "original_environment": current,
        "recovery_sessions": [],
    }
    _write_json_atomic(output, payload)
    return payload


__all__ = [
    "OfficialEnvironmentReferenceError",
    "materialize_official_environment_reference",
]
