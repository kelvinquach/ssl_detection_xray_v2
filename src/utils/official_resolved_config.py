"""Materialize resolved MMEngine config and exact-byte SHA-256."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


class OfficialResolvedConfigError(RuntimeError):
    """Raised when resolved-config materialization is invalid."""


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        tmp = Path(tmp_name)
        if tmp.exists():
            tmp.unlink()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def _yaml_safe_value(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise OfficialResolvedConfigError(
                    "Resolved config contains non-string mapping key "
                    f"at {path}: "
                    f"{type(key).__module__}.{type(key).__qualname__}."
                )
            normalized[key] = _yaml_safe_value(
                item,
                path=f"{path}.{key}",
            )
        return normalized

    if isinstance(value, (list, tuple)):
        return [
            _yaml_safe_value(
                item,
                path=f"{path}[{index}]",
            )
            for index, item in enumerate(value)
        ]

    if isinstance(value, type):
        return {
            "__python_type__": (
                f"{value.__module__}.{value.__qualname__}"
            ),
        }

    value_type = type(value)
    if (
        value_type.__module__ == "__future__"
        and value_type.__name__ == "_Feature"
    ):
        optional_release = value.getOptionalRelease()
        mandatory_release = value.getMandatoryRelease()
        return {
            "__python_object__": "__future__._Feature",
            "optional_release": (
                list(optional_release)
                if optional_release is not None
                else None
            ),
            "mandatory_release": (
                list(mandatory_release)
                if mandatory_release is not None
                else None
            ),
            "compiler_flag": int(value.compiler_flag),
        }

    raise OfficialResolvedConfigError(
        "Resolved config contains unsupported YAML value "
        f"at {path}: "
        f"{value_type.__module__}.{value_type.__qualname__}."
    )


def materialize_official_resolved_config(
    *,
    attempt_dir: Path,
    cfg: Any,
) -> tuple[Path, Path]:
    """Write the exact resolved config passed to Runner.from_cfg()."""
    attempt_dir = Path(attempt_dir).resolve()

    to_dict = getattr(cfg, "to_dict", None)
    if not callable(to_dict):
        raise OfficialResolvedConfigError(
            "Resolved config does not expose callable to_dict()."
        )

    payload = to_dict()
    if not isinstance(payload, dict) or not payload:
        raise OfficialResolvedConfigError(
            "Resolved config must materialize as a non-empty mapping."
        )

    payload = _yaml_safe_value(
        payload,
        path="root",
    )

    try:
        rendered = yaml.safe_dump(
            payload,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
    except Exception as exc:
        raise OfficialResolvedConfigError(
            f"Cannot serialize resolved config to YAML: {exc}"
        ) from exc

    if not rendered.strip():
        raise OfficialResolvedConfigError(
            "Resolved config YAML serialization is empty."
        )

    config_bytes = rendered.encode("utf-8")
    config_path = attempt_dir / "resolved_config.yaml"
    hash_path = attempt_dir / "config.sha256"

    _write_bytes_atomic(config_path, config_bytes)

    digest = _sha256_bytes(config_bytes)
    _write_bytes_atomic(
        hash_path,
        (digest + "\n").encode("ascii"),
    )

    if _sha256_file(config_path) != digest:
        raise OfficialResolvedConfigError(
            "Resolved config SHA-256 verification failed."
        )

    recorded = hash_path.read_text(
        encoding="ascii"
    ).strip()
    if recorded != digest:
        raise OfficialResolvedConfigError(
            "config.sha256 round-trip verification failed."
        )

    return config_path, hash_path


__all__ = [
    "OfficialResolvedConfigError",
    "materialize_official_resolved_config",
]
