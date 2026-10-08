"""S9.11 ablation logical-run identity and status foundation.

Purpose
-------
This module provides identity/state handling specifically for the locked
S9.11 validation-only OFAT ablation campaign.

It intentionally does NOT reuse the S9 official run-root/status helpers
because those helpers encode OFFICIAL run semantics.

Scientific anchor
-----------------
method         = SSL
architecture   = Faster R-CNN R50-FPN
labeled budget = 10%
seeds          = all 10 locked training seeds

Families:
    C0-C3 : confidence threshold
    A0-A2 : strong photometric augmentation
    F0-F1 : regression pseudo-label filtering

Main reference:
    C4 = A3 = F2 = frozen Main
    OFF_SSL_R50_B10_Sxx

Firewall
--------
Every non-Main S9.11 run is:

    run_type              = ABLATION
    condition_role        = ABLATION
    validation_only       = True
    final_test_eligible   = False
    test_access_authorized= False
    final_test_accessed   = False

Ablation results can therefore never become final-test eligible through
logical-run closure.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1

RUN_TYPE = "ABLATION"
CONDITION_ROLE = "ABLATION"
METHOD = "SSL"
ARCHITECTURE = "R50-FPN"
BUDGET = "10pct"

STATUS_IN_PROGRESS = "IN_PROGRESS"
STATUS_TECHNICAL_FAILURE = "TECHNICAL_FAILURE"
STATUS_CLOSED_VERIFIED = "CLOSED_VERIFIED"

RUN_ROOT_MANIFEST_NAME = "run_root_manifest.json"
LOGICAL_STATUS_NAME = "logical_run_status.json"

VALID_VARIANTS = {
    "C0",
    "C1",
    "C2",
    "C3",
    "A0",
    "A1",
    "A2",
    "F0",
    "F1",
}


class AblationIdentityError(RuntimeError):
    """Raised when S9.11 ablation identity/state is invalid."""


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text == "true":
        return True

    if text == "false":
        return False

    raise AblationIdentityError(
        f"INVALID_BOOLEAN_VALUE:{value!r}"
    )


def _write_json_atomic(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )

    tmp_path = Path(tmp_name)

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")

        os.replace(tmp_path, path)

    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _load_json(path: Path) -> dict[str, Any]:
    path = Path(path)

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise AblationIdentityError(
            f"ABLATON_JSON_UNREADABLE:{path}"
        ) from exc

    if not isinstance(payload, dict):
        raise AblationIdentityError(
            f"ABLATION_JSON_MUST_BE_OBJECT:{path}"
        )

    return payload


def build_ablation_identity(
    repo_root: Path,
    row: dict[str, Any],
) -> dict[str, Any]:
    """Build and validate the immutable scientific identity of one run."""

    repo_root = Path(repo_root).resolve()

    run_id = str(
        row.get("ablation_run_id", "")
    ).strip()

    config_id = str(
        row.get("ablation_config_id", "")
    ).strip().upper()

    family = str(
        row.get("family", "")
    ).strip()

    level = str(
        row.get("level", "")
    ).strip()

    executable_config_path = str(
        row.get("executable_config_path", "")
    ).strip()

    main_run_id = str(
        row.get("main_run_id", "")
    ).strip()

    if not run_id.startswith("ABL_R50_B10_"):
        raise AblationIdentityError(
            f"INVALID_ABLATION_RUN_ID:{run_id!r}"
        )

    if config_id not in VALID_VARIANTS:
        raise AblationIdentityError(
            f"INVALID_ABLATION_CONFIG_ID:{config_id!r}"
        )

    if str(row.get("condition_role", "")).strip() != CONDITION_ROLE:
        raise AblationIdentityError(
            "ABLATION_CONDITION_ROLE_MISMATCH"
        )

    if str(row.get("method", "")).strip() != METHOD:
        raise AblationIdentityError(
            "ABLATION_METHOD_MISMATCH"
        )

    if str(row.get("architecture", "")).strip() != ARCHITECTURE:
        raise AblationIdentityError(
            "ABLATION_ARCHITECTURE_MISMATCH"
        )

    if str(row.get("budget", "")).strip() != BUDGET:
        raise AblationIdentityError(
            "ABLATION_BUDGET_MISMATCH"
        )

    try:
        seed_index = int(
            str(row.get("seed_index", "")).strip()
        )
        training_seed = int(
            str(row.get("training_seed", "")).strip()
        )
    except ValueError as exc:
        raise AblationIdentityError(
            "ABLATION_SEED_IDENTITY_INVALID"
        ) from exc

    if seed_index < 1 or seed_index > 10:
        raise AblationIdentityError(
            f"ABLATION_SEED_INDEX_OUT_OF_RANGE:{seed_index}"
        )

    if training_seed < 0:
        raise AblationIdentityError(
            f"ABLATION_TRAINING_SEED_INVALID:{training_seed}"
        )

    expected_seed_suffix = f"S{seed_index:02d}"

    if not run_id.endswith(expected_seed_suffix):
        raise AblationIdentityError(
            "ABLATION_RUN_ID_SEED_SUFFIX_MISMATCH"
        )

    expected_main_run = (
        f"OFF_SSL_R50_B10_{expected_seed_suffix}"
    )

    if main_run_id != expected_main_run:
        raise AblationIdentityError(
            "ABLATION_MAIN_SAME_SEED_REFERENCE_MISMATCH"
        )

    config_path = (
        repo_root
        / executable_config_path
    ).resolve()

    try:
        config_path.relative_to(repo_root)
    except ValueError as exc:
        raise AblationIdentityError(
            "ABLATION_CONFIG_OUTSIDE_REPOSITORY"
        ) from exc

    if not config_path.is_file():
        raise AblationIdentityError(
            f"ABLATION_CONFIG_MISSING:{executable_config_path}"
        )

    if _parse_bool(row.get("is_main")):
        raise AblationIdentityError(
            "NON_MAIN_ABLATION_MARKED_AS_MAIN"
        )

    if not _parse_bool(row.get("expected")):
        raise AblationIdentityError(
            "ABLATION_RUN_NOT_EXPECTED"
        )

    if not _parse_bool(row.get("validation_only")):
        raise AblationIdentityError(
            "ABLATION_VALIDATION_ONLY_MUST_BE_TRUE"
        )

    if _parse_bool(row.get("final_test_eligible")):
        raise AblationIdentityError(
            "ABLATION_FINAL_TEST_ELIGIBILITY_MUST_BE_FALSE"
        )

    if _parse_bool(row.get("test_access_authorized")):
        raise AblationIdentityError(
            "ABLATION_TEST_ACCESS_AUTHORIZATION_MUST_BE_FALSE"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "run_type": RUN_TYPE,
        "condition_role": CONDITION_ROLE,
        "method": METHOD,
        "architecture": ARCHITECTURE,
        "budget": BUDGET,
        "ablation_config_id": config_id,
        "family": family,
        "level": level,
        "seed_index": seed_index,
        "training_seed": training_seed,
        "executable_config_path": executable_config_path,
        "main_run_id": main_run_id,
        "validation_only": True,
        "final_test_eligible": False,
        "test_access_authorized": False,
        "final_test_accessed": False,
    }


def ensure_ablation_run_root_manifest(
    run_root: Path,
    identity: dict[str, Any],
    *,
    attempt_id: str,
    recovery: bool,
) -> dict[str, Any]:
    """Create or validate immutable logical-run root identity.

    Initial execution may create the immutable root manifest for attempt_001.

    A later technical retry must inherit an already-existing root manifest
    rather than silently reconstructing the logical-run identity.

    Recovery follows the explicit recovery path while preserving the same
    immutable scientific identity.
    """

    run_root = Path(run_root)
    path = run_root / RUN_ROOT_MANIFEST_NAME

    attempt_id = str(attempt_id).strip()

    if not attempt_id:
        raise AblationIdentityError(
            "ABLATION_ATTEMPT_ID_EMPTY"
        )

    if not isinstance(recovery, bool):
        raise AblationIdentityError(
            "ABLATION_RECOVERY_FLAG_MUST_BE_BOOLEAN"
        )

    payload = dict(identity)

    if payload.get("run_type") != RUN_TYPE:
        raise AblationIdentityError(
            "ABLATION_RUN_ROOT_RUN_TYPE_INVALID"
        )

    if payload.get("condition_role") != CONDITION_ROLE:
        raise AblationIdentityError(
            "ABLATION_RUN_ROOT_CONDITION_ROLE_INVALID"
        )

    if path.exists():
        existing = _load_json(path)

        if existing != payload:
            raise AblationIdentityError(
                "ABLATION_RUN_ROOT_MANIFEST_IDENTITY_MISMATCH"
            )

        return existing

    if attempt_id != "attempt_001" and not recovery:
        raise AblationIdentityError(
            "ABLATION_RUN_ROOT_MANIFEST_MISSING_FOR_TECHNICAL_RETRY"
        )

    run_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    _write_json_atomic(
        path,
        payload,
    )

    observed = _load_json(path)

    if observed != payload:
        raise AblationIdentityError(
            "ABLATION_RUN_ROOT_MANIFEST_POSTWRITE_MISMATCH"
        )

    return observed


def _status_path(run_root: Path) -> Path:
    return (
        Path(run_root)
        / LOGICAL_STATUS_NAME
    )


def _base_status(
    identity: dict[str, Any],
    *,
    training_status: str,
    attempt_id: str,
    training_completed: bool,
    training_valid: bool,
    training_completion_report_path: str | None,
    artifact_manifest_path: str | None,
) -> dict[str, Any]:
    """Build firewall-safe mutable ablation logical-run status."""

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": identity["run_id"],
        "run_type": RUN_TYPE,
        "condition_role": CONDITION_ROLE,
        "training_seed": int(
            identity["training_seed"]
        ),
        "training_status": training_status,
        "attempt_used": attempt_id,
        "training_completed": bool(
            training_completed
        ),
        "training_valid": bool(
            training_valid
        ),
        "eligible_for_final_test": False,
        "validation_only": True,
        "test_access_authorized": False,
        "final_test_accessed": False,
        "training_completion_report_path":
            training_completion_report_path,
        "artifact_manifest_path":
            artifact_manifest_path,
    }


def set_ablation_logical_run_in_progress(
    run_root: Path,
    identity: dict[str, Any],
    attempt_id: str,
) -> dict[str, Any]:
    """Transition one ablation logical run to IN_PROGRESS."""

    path = _status_path(run_root)

    if path.exists():
        current = _load_json(path)

        if current.get("run_id") != identity["run_id"]:
            raise AblationIdentityError(
                "ABLATION_LOGICAL_STATUS_RUN_ID_MISMATCH"
            )

        if current.get("training_status") == STATUS_CLOSED_VERIFIED:
            raise AblationIdentityError(
                "ABLATION_LOGICAL_RUN_ALREADY_CLOSED"
            )

        if (
            current.get("training_status")
            == STATUS_IN_PROGRESS
            and current.get("attempt_used")
            != attempt_id
        ):
            raise AblationIdentityError(
                "ABLATION_PREVIOUS_ATTEMPT_STILL_IN_PROGRESS"
            )

    payload = _base_status(
        identity,
        training_status=STATUS_IN_PROGRESS,
        attempt_id=attempt_id,
        training_completed=False,
        training_valid=False,
        training_completion_report_path=None,
        artifact_manifest_path=None,
    )

    _write_json_atomic(
        path,
        payload,
    )

    return _load_json(path)


def set_ablation_logical_run_technical_failure(
    run_root: Path,
    identity: dict[str, Any],
    attempt_id: str,
) -> dict[str, Any]:
    """Record technical failure without changing scientific identity."""

    path = _status_path(run_root)

    if not path.is_file():
        raise AblationIdentityError(
            "ABLATION_LOGICAL_STATUS_MISSING_ON_FAILURE"
        )

    current = _load_json(path)

    if (
        current.get("run_id") != identity["run_id"]
        or current.get("attempt_used") != attempt_id
        or current.get("training_status") != STATUS_IN_PROGRESS
    ):
        raise AblationIdentityError(
            "ABLATION_TECHNICAL_FAILURE_TRANSITION_INVALID"
        )

    payload = _base_status(
        identity,
        training_status=STATUS_TECHNICAL_FAILURE,
        attempt_id=attempt_id,
        training_completed=False,
        training_valid=False,
        training_completion_report_path=None,
        artifact_manifest_path=None,
    )

    _write_json_atomic(
        path,
        payload,
    )

    return _load_json(path)


def set_ablation_logical_run_closed_verified(
    run_root: Path,
    identity: dict[str, Any],
    attempt_id: str,
) -> dict[str, Any]:
    """Close one verified ablation run; final-test eligibility stays false."""

    path = _status_path(run_root)

    if not path.is_file():
        raise AblationIdentityError(
            "ABLATION_LOGICAL_STATUS_MISSING_ON_CLOSURE"
        )

    current = _load_json(path)

    if (
        current.get("run_id") != identity["run_id"]
        or current.get("attempt_used") != attempt_id
        or current.get("training_status") != STATUS_IN_PROGRESS
    ):
        raise AblationIdentityError(
            "ABLATION_CLOSURE_TRANSITION_INVALID"
        )

    payload = _base_status(
        identity,
        training_status=STATUS_CLOSED_VERIFIED,
        attempt_id=attempt_id,
        training_completed=True,
        training_valid=True,
        training_completion_report_path=(
            f"{attempt_id}/training_completion_report.json"
        ),
        artifact_manifest_path=(
            f"{attempt_id}/artifact_manifest.json"
        ),
    )

    _write_json_atomic(
        path,
        payload,
    )

    observed = _load_json(path)

    if observed["eligible_for_final_test"] is not False:
        raise AblationIdentityError(
            "ABLATION_CLOSURE_BECAME_FINAL_TEST_ELIGIBLE"
        )

    if observed["final_test_accessed"] is not False:
        raise AblationIdentityError(
            "ABLATION_CLOSURE_FINAL_TEST_ACCESS_INVALID"
        )

    return observed


__all__ = [
    "AblationIdentityError",
    "ARCHITECTURE",
    "BUDGET",
    "CONDITION_ROLE",
    "METHOD",
    "RUN_TYPE",
    "STATUS_CLOSED_VERIFIED",
    "STATUS_IN_PROGRESS",
    "STATUS_TECHNICAL_FAILURE",
    "VALID_VARIANTS",
    "build_ablation_identity",
    "ensure_ablation_run_root_manifest",
    "set_ablation_logical_run_closed_verified",
    "set_ablation_logical_run_in_progress",
    "set_ablation_logical_run_technical_failure",
]