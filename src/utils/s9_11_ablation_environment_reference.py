"""S9.11 per-attempt runtime-environment provenance for ablation execution.

Scientific / execution role
---------------------------
S9.11 is a validation-only OFAT mechanistic ablation campaign anchored to
the frozen Main SSL Faster R-CNN R50-FPN 10% condition.

The runtime environment itself must remain identical in meaning to the
official Main campaign. Therefore this module deliberately reuses the
already-validated runtime snapshot implementation from
``official_environment_reference``.

Only the logical artifact identity is specialized:

    artifact_type  = ABLATION_ENVIRONMENT_REFERENCE
    run_type       = ABLATION
    condition_role = ABLATION

This prevents a non-Main ablation attempt from being represented as an
OFFICIAL run while preserving exactly the same Docker/CUDA/framework
runtime identity checks used by Main.

No final-test authority is represented or granted here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.official_environment_reference import (
    _current_snapshot,
    _load_json,
    _utc_now,
    _write_json_atomic,
)


ARTIFACT_TYPE = "ABLATION_ENVIRONMENT_REFERENCE"
RUN_TYPE = "ABLATION"
CONDITION_ROLE = "ABLATION"


class AblationEnvironmentReferenceError(RuntimeError):
    """Raised when S9.11 environment provenance cannot be trusted."""


def materialize_ablation_environment_reference(
    *,
    repo_root: Path,
    attempt_dir: Path,
    run_id: str,
    attempt_id: str,
    recovery: bool = False,
) -> dict[str, Any]:
    """Materialize or extend one S9.11 environment reference.

    Fresh attempt
    -------------
    Capture the same canonical runtime snapshot used for official Main
    execution, but label the artifact explicitly as ABLATION.

    Recovery
    --------
    Preserve the immutable original runtime identity and append one recovery
    session containing the newly observed runtime. The record explicitly
    reports whether the recovery runtime identity equals the original one.

    This function does not perform training and does not authorize test use.
    """

    repo_root = Path(repo_root).resolve()
    attempt_dir = Path(attempt_dir).resolve()

    run_id = str(run_id).strip()
    attempt_id = str(attempt_id).strip()

    if not run_id.startswith("ABL_R50_B10_"):
        raise AblationEnvironmentReferenceError(
            f"INVALID_ABLATION_RUN_ID:{run_id!r}"
        )

    if not attempt_id:
        raise AblationEnvironmentReferenceError(
            "ABLATION_ATTEMPT_ID_EMPTY"
        )

    output = attempt_dir / "environment_reference.json"

    # Reuse the canonical S9 runtime snapshot implementation.
    # This preserves the same frozen runtime evidence semantics as Main.
    current = _current_snapshot(repo_root)

    if recovery:
        if not output.is_file():
            raise AblationEnvironmentReferenceError(
                "RECOVERED_ABLATION_ATTEMPT_MISSING_ENVIRONMENT_REFERENCE"
            )

        payload = _load_json(output)

        if payload.get("artifact_type") != ARTIFACT_TYPE:
            raise AblationEnvironmentReferenceError(
                "ABLATION_ENVIRONMENT_ARTIFACT_TYPE_MISMATCH"
            )

        if payload.get("run_type") != RUN_TYPE:
            raise AblationEnvironmentReferenceError(
                "ABLATION_ENVIRONMENT_RUN_TYPE_MISMATCH"
            )

        if payload.get("condition_role") != CONDITION_ROLE:
            raise AblationEnvironmentReferenceError(
                "ABLATION_ENVIRONMENT_CONDITION_ROLE_MISMATCH"
            )

        if payload.get("run_id") != run_id:
            raise AblationEnvironmentReferenceError(
                "RECOVERED_ABLATION_ENVIRONMENT_RUN_ID_MISMATCH"
            )

        if payload.get("attempt_id") != attempt_id:
            raise AblationEnvironmentReferenceError(
                "RECOVERED_ABLATION_ENVIRONMENT_ATTEMPT_ID_MISMATCH"
            )

        original = payload.get("original_environment")
        sessions = payload.get("recovery_sessions")

        if not isinstance(original, dict):
            raise AblationEnvironmentReferenceError(
                "ABLATION_ORIGINAL_ENVIRONMENT_MISSING"
            )

        if not isinstance(sessions, list):
            raise AblationEnvironmentReferenceError(
                "ABLATION_RECOVERY_SESSIONS_INVALID"
            )

        original_identity = original.get(
            "runtime_identity_sha256"
        )

        if (
            not isinstance(original_identity, str)
            or len(original_identity) != 64
        ):
            raise AblationEnvironmentReferenceError(
                "ABLATION_ORIGINAL_RUNTIME_IDENTITY_INVALID"
            )

        sessions.append(
            {
                "recovery_index": len(sessions) + 1,
                "recorded_at_utc": _utc_now(),
                "same_runtime_identity_as_original": (
                    current["runtime_identity_sha256"]
                    == original_identity
                ),
                "environment": current,
            }
        )

        payload["last_updated_at_utc"] = _utc_now()

        _write_json_atomic(
            output,
            payload,
        )

        observed = _load_json(output)

        if observed != payload:
            raise AblationEnvironmentReferenceError(
                "ABLATION_ENVIRONMENT_RECOVERY_POSTWRITE_MISMATCH"
            )

        return observed

    if output.exists():
        raise AblationEnvironmentReferenceError(
            "FRESH_ABLATION_ATTEMPT_ALREADY_HAS_ENVIRONMENT_REFERENCE"
        )

    attempt_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "schema_version": "1.0",
        "artifact_type": ARTIFACT_TYPE,
        "run_type": RUN_TYPE,
        "condition_role": CONDITION_ROLE,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "materialized_at_utc": _utc_now(),
        "original_environment": current,
        "recovery_sessions": [],
    }

    _write_json_atomic(
        output,
        payload,
    )

    observed = _load_json(output)

    if observed != payload:
        raise AblationEnvironmentReferenceError(
            "ABLATION_ENVIRONMENT_POSTWRITE_MISMATCH"
        )

    return observed


__all__ = [
    "ARTIFACT_TYPE",
    "AblationEnvironmentReferenceError",
    "CONDITION_ROLE",
    "RUN_TYPE",
    "materialize_ablation_environment_reference",
]