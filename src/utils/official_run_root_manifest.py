"""Immutable logical-run root manifest for official S9 execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class OfficialRunRootManifestError(RuntimeError):
    """Raised when logical-run root identity is invalid."""


def _expected_value(expected: dict[str, Any], key: str) -> Any:
    if key not in expected:
        raise OfficialRunRootManifestError(
            f"OFFICIAL_RUN_ROOT_EXPECTED_FIELD_MISSING:{key}"
        )
    return expected[key]


def _build_payload(
    *,
    run_id: str,
    expected: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1.1",
        "run_id": str(run_id),
        "run_type": "OFFICIAL",
        "condition_role": str(
            _expected_value(expected, "condition_role")
        ),
        "method": str(_expected_value(expected, "method")),
        "architecture": str(
            _expected_value(expected, "architecture")
        ),
        "budget": str(_expected_value(expected, "budget")),
        "seed_index": int(
            _expected_value(expected, "seed_index")
        ),
        "training_seed": int(
            _expected_value(expected, "training_seed")
        ),
        "partition_seed": int(
            _expected_value(expected, "partition_seed")
        ),
    }


def _load_existing(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8-sig")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise OfficialRunRootManifestError(
            "OFFICIAL_RUN_ROOT_MANIFEST_UNREADABLE"
        ) from exc

    if not isinstance(payload, dict):
        raise OfficialRunRootManifestError(
            "OFFICIAL_RUN_ROOT_MANIFEST_MUST_BE_OBJECT"
        )
    return payload


def validate_official_run_root_manifest(
    *,
    run_root: Path,
    run_id: str,
    expected: dict[str, Any],
) -> dict[str, Any]:
    """Require exact immutable logical-run root identity."""
    path = Path(run_root) / "run_root_manifest.json"
    if not path.is_file():
        raise OfficialRunRootManifestError(
            "OFFICIAL_RUN_ROOT_MANIFEST_MISSING"
        )

    observed = _load_existing(path)
    expected_payload = _build_payload(
        run_id=run_id,
        expected=expected,
    )
    if observed != expected_payload:
        raise OfficialRunRootManifestError(
            "OFFICIAL_RUN_ROOT_MANIFEST_IDENTITY_MISMATCH"
        )
    return observed


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def ensure_official_run_root_manifest(
    *,
    run_root: Path,
    run_id: str,
    attempt_id: str,
    expected: dict[str, Any],
    recovery: bool,
) -> dict[str, Any]:
    """Materialize or exact-validate immutable logical-run identity."""
    path = Path(run_root) / "run_root_manifest.json"
    expected_payload = _build_payload(
        run_id=run_id,
        expected=expected,
    )

    if path.is_file():
        return validate_official_run_root_manifest(
            run_root=run_root,
            run_id=run_id,
            expected=expected,
        )

    if path.exists():
        raise OfficialRunRootManifestError(
            "OFFICIAL_RUN_ROOT_MANIFEST_PATH_NOT_FILE"
        )

    if attempt_id != "attempt_001" and not recovery:
        raise OfficialRunRootManifestError(
            "OFFICIAL_RUN_ROOT_MANIFEST_MISSING_FOR_TECHNICAL_RETRY"
        )

    _write_new(path, expected_payload)

    return validate_official_run_root_manifest(
        run_root=run_root,
        run_id=run_id,
        expected=expected,
    )


__all__ = [
    "OfficialRunRootManifestError",
    "ensure_official_run_root_manifest",
    "validate_official_run_root_manifest",
]
