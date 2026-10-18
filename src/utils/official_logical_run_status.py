"""Mutable logical-run status for official S9 execution."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.1"
REQUIRED_FIELDS = {
    "schema_version",
    "run_id",
    "training_seed",
    "training_status",
    "attempt_used",
    "training_completed",
    "training_valid",
    "eligible_for_final_test",
    "final_evaluation_status",
    "training_completion_report_path",
    "artifact_manifest_path",
}


class OfficialLogicalRunStatusError(RuntimeError):
    """Raised when logical-run mutable state is invalid."""


def _status_path(run_root: Path) -> Path:
    return Path(run_root) / "logical_run_status.json"


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8-sig")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise OfficialLogicalRunStatusError(
            "OFFICIAL_LOGICAL_RUN_STATUS_UNREADABLE"
        ) from exc
    if not isinstance(payload, dict):
        raise OfficialLogicalRunStatusError(
            "OFFICIAL_LOGICAL_RUN_STATUS_MUST_BE_OBJECT"
        )
    if set(payload) != REQUIRED_FIELDS:
        raise OfficialLogicalRunStatusError(
            "OFFICIAL_LOGICAL_RUN_STATUS_SCHEMA_MISMATCH"
        )
    return payload


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    tmp = Path(name)
    try:
        tmp.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _identity_check(
    payload: dict[str, Any],
    *,
    run_id: str,
    training_seed: int,
) -> None:
    if payload["schema_version"] != SCHEMA_VERSION:
        raise OfficialLogicalRunStatusError(
            "OFFICIAL_LOGICAL_RUN_STATUS_VERSION_MISMATCH"
        )
    if payload["run_id"] != run_id:
        raise OfficialLogicalRunStatusError(
            "OFFICIAL_LOGICAL_RUN_STATUS_RUN_ID_MISMATCH"
        )
    if payload["training_seed"] != training_seed:
        raise OfficialLogicalRunStatusError(
            "OFFICIAL_LOGICAL_RUN_STATUS_SEED_MISMATCH"
        )


def _base_payload(
    *,
    run_id: str,
    training_seed: int,
    attempt_id: str,
    training_status: str,
    training_completed: bool,
    training_valid: bool,
    eligible_for_final_test: bool,
    training_completion_report_path: str | None,
    artifact_manifest_path: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "training_seed": training_seed,
        "training_status": training_status,
        "attempt_used": attempt_id,
        "training_completed": training_completed,
        "training_valid": training_valid,
        "eligible_for_final_test": eligible_for_final_test,
        "final_evaluation_status": "NOT_AUTHORIZED",
        "training_completion_report_path": training_completion_report_path,
        "artifact_manifest_path": artifact_manifest_path,
    }


def set_official_logical_run_in_progress(
    *,
    run_root: Path,
    run_id: str,
    attempt_id: str,
    training_seed: int,
    recovery: bool,
) -> dict[str, Any]:
    path = _status_path(run_root)
    run_id = str(run_id)
    attempt_id = str(attempt_id)
    training_seed = int(training_seed)

    if path.is_file():
        previous = _load(path)
        _identity_check(
            previous,
            run_id=run_id,
            training_seed=training_seed,
        )
        if previous["training_status"] == "CLOSED_VERIFIED":
            raise OfficialLogicalRunStatusError(
                "OFFICIAL_LOGICAL_RUN_ALREADY_CLOSED"
            )
        if recovery:
            if (
                previous["attempt_used"] != attempt_id
                or previous["training_status"] != "IN_PROGRESS"
            ):
                raise OfficialLogicalRunStatusError(
                    "OFFICIAL_LOGICAL_RUN_RECOVERY_STATE_MISMATCH"
                )
        else:
            if previous["training_status"] == "IN_PROGRESS":
                raise OfficialLogicalRunStatusError(
                    "OFFICIAL_LOGICAL_RUN_PREVIOUS_ATTEMPT_STILL_IN_PROGRESS"
                )
            if previous["training_status"] != "TECHNICAL_FAILURE":
                raise OfficialLogicalRunStatusError(
                    "OFFICIAL_LOGICAL_RUN_NEW_ATTEMPT_STATE_INVALID"
                )
            if previous["attempt_used"] == attempt_id:
                raise OfficialLogicalRunStatusError(
                    "OFFICIAL_LOGICAL_RUN_TECHNICAL_RETRY_REQUIRES_NEW_ATTEMPT"
                )
    elif path.exists():
        raise OfficialLogicalRunStatusError(
            "OFFICIAL_LOGICAL_RUN_STATUS_PATH_NOT_FILE"
        )
    else:
        if recovery:
            raise OfficialLogicalRunStatusError(
                "OFFICIAL_LOGICAL_RUN_STATUS_MISSING_ON_RECOVERY"
            )
        if attempt_id != "attempt_001":
            raise OfficialLogicalRunStatusError(
                "OFFICIAL_LOGICAL_RUN_STATUS_MISSING_FOR_TECHNICAL_RETRY"
            )

    payload = _base_payload(
        run_id=run_id,
        training_seed=training_seed,
        attempt_id=attempt_id,
        training_status="IN_PROGRESS",
        training_completed=False,
        training_valid=False,
        eligible_for_final_test=False,
        training_completion_report_path=None,
        artifact_manifest_path=None,
    )
    _write_atomic(path, payload)
    observed = _load(path)
    if observed != payload:
        raise OfficialLogicalRunStatusError(
            "OFFICIAL_LOGICAL_RUN_STATUS_POSTWRITE_MISMATCH"
        )
    return observed


def set_official_logical_run_technical_failure(
    *,
    run_root: Path,
    run_id: str,
    attempt_id: str,
    training_seed: int,
) -> dict[str, Any]:
    path = _status_path(run_root)
    if not path.is_file():
        raise OfficialLogicalRunStatusError(
            "OFFICIAL_LOGICAL_RUN_STATUS_MISSING_ON_FAILURE"
        )
    previous = _load(path)
    _identity_check(
        previous,
        run_id=str(run_id),
        training_seed=int(training_seed),
    )
    if (
        previous["attempt_used"] != str(attempt_id)
        or previous["training_status"] != "IN_PROGRESS"
    ):
        raise OfficialLogicalRunStatusError(
            "OFFICIAL_LOGICAL_RUN_FAILURE_TRANSITION_INVALID"
        )

    payload = _base_payload(
        run_id=str(run_id),
        training_seed=int(training_seed),
        attempt_id=str(attempt_id),
        training_status="TECHNICAL_FAILURE",
        training_completed=False,
        training_valid=False,
        eligible_for_final_test=False,
        training_completion_report_path=None,
        artifact_manifest_path=None,
    )
    _write_atomic(path, payload)
    observed = _load(path)
    if observed != payload:
        raise OfficialLogicalRunStatusError(
            "OFFICIAL_LOGICAL_RUN_STATUS_POSTWRITE_MISMATCH"
        )
    return observed


def set_official_logical_run_closed_verified(
    *,
    run_root: Path,
    run_id: str,
    attempt_id: str,
    training_seed: int,
    eligible_for_final_test: bool,
) -> dict[str, Any]:
    path = _status_path(run_root)
    if not path.is_file():
        raise OfficialLogicalRunStatusError(
            "OFFICIAL_LOGICAL_RUN_STATUS_MISSING_ON_CLOSURE"
        )
    previous = _load(path)
    _identity_check(
        previous,
        run_id=str(run_id),
        training_seed=int(training_seed),
    )
    if (
        previous["attempt_used"] != str(attempt_id)
        or previous["training_status"] != "IN_PROGRESS"
    ):
        raise OfficialLogicalRunStatusError(
            "OFFICIAL_LOGICAL_RUN_CLOSURE_TRANSITION_INVALID"
        )

    payload = _base_payload(
        run_id=str(run_id),
        training_seed=int(training_seed),
        attempt_id=str(attempt_id),
        training_status="CLOSED_VERIFIED",
        training_completed=True,
        training_valid=True,
        eligible_for_final_test=bool(eligible_for_final_test),
        training_completion_report_path=(
            f"{attempt_id}/training_completion_report.json"
        ),
        artifact_manifest_path=(
            f"{attempt_id}/artifact_manifest.json"
        ),
    )
    _write_atomic(path, payload)
    observed = _load(path)
    if observed != payload:
        raise OfficialLogicalRunStatusError(
            "OFFICIAL_LOGICAL_RUN_STATUS_POSTWRITE_MISMATCH"
        )
    return observed


__all__ = [
    "OfficialLogicalRunStatusError",
    "set_official_logical_run_in_progress",
    "set_official_logical_run_technical_failure",
    "set_official_logical_run_closed_verified",
]
