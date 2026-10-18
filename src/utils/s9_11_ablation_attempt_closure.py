"""S9.11 per-attempt closure for non-Main ablation training.

Scientific role
---------------
S9.11 is a prespecified validation-only OFAT campaign anchored to the
frozen SSL Faster R-CNN R50-FPN 10% Main condition.

This module intentionally reuses the already-verified generic evidence
validators from the official S9 training closure for:

- resolved configuration integrity;
- data provenance;
- deterministic seed/runtime evidence;
- optimizer-update accounting;
- training log completeness;
- validation/checkpoint evidence;
- retry/deviation evidence;
- Q_pseudo on fixed validation using LAST EMA Teacher.

It does NOT reuse Official identity/final-test semantics.

A non-Main S9.11 run can close only as:

    run_type = ABLATION
    condition_role = ABLATION
    validation_only = True
    eligible_for_final_test = False
    test_access_authorized = False
    final_test_accessed = False

No ablation closure may authorize or perform final-test evaluation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.utils.official_attempt_closure import (
    ROLE_BY_PATH,
    SCHEMA_VERSION as ARTIFACT_SCHEMA_VERSION,
    OfficialAttemptClosureError,
    _load_json,
    _mandatory_paths,
    _read_jsonl,
    _require_file,
    _require_keys,
    _validate_data_provenance,
    _validate_pre_run,
    _validate_qpseudo,
    _validate_resolved_config,
    _validate_retry,
    _validate_seed_runtime,
    _validate_train_log,
    _validate_training_summary,
    _validate_validation_artifacts,
    _write_json_atomic,
)

from src.utils.run_manifest import file_sha256

from src.utils.s9_11_ablation_identity import (
    AblationIdentityError,
    ensure_ablation_run_root_manifest,
    set_ablation_logical_run_closed_verified,
)


RUN_TYPE = "ABLATION"
CONDITION_ROLE = "ABLATION"
METHOD = "SSL"
ARCHITECTURE = "R50-FPN"
BUDGET = "10pct"

RUN_MANIFEST_SCHEMA_VERSION = "1.0"
ENTRYPOINT_ID = "scripts/S9_run_ablation.py"

ENVIRONMENT_ARTIFACT_TYPE = (
    "ABLATION_ENVIRONMENT_REFERENCE"
)

SHA256_RE = re.compile(
    r"^[0-9a-f]{64}$"
)


class AblationAttemptClosureError(RuntimeError):
    """Raised when an S9.11 ablation attempt cannot close safely."""


def _require_sha256(
    value: Any,
    *,
    label: str,
) -> str:
    value = str(value).strip()

    if SHA256_RE.fullmatch(value) is None:
        raise AblationAttemptClosureError(
            f"{label}_INVALID_SHA256"
        )

    return value


def _validate_ablation_run_root(
    *,
    run_root: Path,
    identity: dict[str, Any],
    attempt_id: str,
) -> None:
    """Require and validate the pre-existing immutable ablation root."""

    run_root = Path(run_root)

    root_manifest = (
        run_root
        / "run_root_manifest.json"
    )

    if not root_manifest.is_file():
        raise AblationAttemptClosureError(
            "ABLATION_RUN_ROOT_MANIFEST_MISSING_ON_CLOSURE"
        )

    try:
        observed = ensure_ablation_run_root_manifest(
            run_root,
            identity,
            attempt_id=attempt_id,
            recovery=False,
        )
    except AblationIdentityError as exc:
        raise AblationAttemptClosureError(
            f"ABLATION_RUN_ROOT_VALIDATION_FAILED:{exc}"
        ) from exc

    if observed != identity:
        raise AblationAttemptClosureError(
            "ABLATION_RUN_ROOT_IDENTITY_MISMATCH"
        )


def _validate_ablation_environment(
    *,
    attempt_dir: Path,
    run_id: str,
    attempt_id: str,
    runtime_identity_sha256: str,
) -> None:
    """Bind environment evidence to this exact ablation attempt."""

    path = _require_file(
        attempt_dir,
        "environment_reference.json",
    )

    payload = _load_json(path)

    _require_keys(
        payload,
        (
            "artifact_type",
            "run_type",
            "condition_role",
            "run_id",
            "attempt_id",
            "original_environment",
        ),
        "environment_reference.json",
    )

    expected_pairs = {
        "artifact_type":
            ENVIRONMENT_ARTIFACT_TYPE,
        "run_type":
            RUN_TYPE,
        "condition_role":
            CONDITION_ROLE,
        "run_id":
            run_id,
        "attempt_id":
            attempt_id,
    }

    for key, value in expected_pairs.items():
        if payload.get(key) != value:
            raise AblationAttemptClosureError(
                f"ABLATION_ENVIRONMENT_{key.upper()}_MISMATCH"
            )

    original = payload.get(
        "original_environment"
    )

    if not isinstance(original, dict):
        raise AblationAttemptClosureError(
            "ABLATION_ORIGINAL_ENVIRONMENT_INVALID"
        )

    observed_runtime_identity = _require_sha256(
        original.get(
            "runtime_identity_sha256"
        ),
        label="ABLATION_RUNTIME_IDENTITY",
    )

    if (
        observed_runtime_identity
        != runtime_identity_sha256
    ):
        raise AblationAttemptClosureError(
            "ABLATION_RUNTIME_IDENTITY_MANIFEST_MISMATCH"
        )


def _validate_ablation_run_manifest(
    *,
    attempt_dir: Path,
    run_id: str,
    attempt_id: str,
    expected: dict[str, Any],
) -> dict[str, Any]:
    """Validate immutable launch-time S9.11 run_manifest.json."""

    path = _require_file(
        attempt_dir,
        "run_manifest.json",
    )

    payload = _load_json(path)

    required = (
        "schema_version",
        "run_id",
        "attempt_id",
        "run_type",
        "condition_role",
        "method",
        "architecture",
        "budget",
        "ablation_config_id",
        "ablation_family",
        "ablation_level",
        "scientific_change",
        "main_run_id",
        "seed_index",
        "training_seed",
        "partition_seed",
        "scientific_source_name",
        "scientific_source_sha256",
        "implementation_contract_sha256",
        "artifact_contract_sha256",
        "git_commit",
        "docker_image",
        "docker_digest",
        "runtime_identity_sha256",
        "environment_reference_path",
        "entrypoint_id",
        "executable_config_path",
        "executable_config_sha256",
        "main_frozen_executable_config_path",
        "main_frozen_executable_config_sha256",
        "resolved_config_sha256",
        "dataset_manifest_sha256",
        "train_split_sha256",
        "val_split_sha256",
        "test_split_sha256",
        "labeled_manifest_sha256",
        "unlabeled_manifest_sha256",
        "preprocessing_config_sha256",
        "labeled_pipeline_sha256",
        "optimizer_recipe_sha256",
        "evaluator_config_sha256",
        "training_protocol_sha256",
        "checkpoint_policy_sha256",
        "ssl_protocol_sha256",
        "seed_protocol_sha256",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "total_optimizer_updates",
        "validation_interval_updates",
        "effective_labeled_batch",
        "effective_unlabeled_batch",
        "amp",
        "gradient_clipping",
        "checkpoint_selection_metric",
        "effective_scientific_fields",
        "validation_only",
        "final_test_eligible",
        "test_usage",
        "test_access_authorized",
        "final_test_accessed",
        "test_access_reason",
        "final_evaluation_status",
        "training_status",
    )

    _require_keys(
        payload,
        required,
        "run_manifest.json",
    )

    expected_pairs = {
        "schema_version":
            RUN_MANIFEST_SCHEMA_VERSION,
        "run_id":
            run_id,
        "attempt_id":
            attempt_id,
        "run_type":
            RUN_TYPE,
        "condition_role":
            CONDITION_ROLE,
        "method":
            METHOD,
        "architecture":
            ARCHITECTURE,
        "budget":
            BUDGET,

        "ablation_config_id":
            expected["ablation_config_id"],
        "ablation_family":
            str(
                expected["ablation_family"]
            ).strip().upper(),
        "ablation_level":
            str(
                expected["ablation_level"]
            ).strip().upper(),
        "scientific_change":
            expected["scientific_change"],
        "main_run_id":
            expected["main_run_id"],

        "seed_index":
            int(expected["seed_index"]),
        "training_seed":
            int(expected["training_seed"]),
        "partition_seed":
            int(expected["partition_seed"]),

        "scientific_source_name":
            expected["scientific_source_name"],
        "scientific_source_sha256":
            expected["scientific_source_sha256"],
        "implementation_contract_sha256":
            expected["implementation_contract_sha256"],
        "artifact_contract_sha256":
            expected["artifact_contract_sha256"],

        "entrypoint_id":
            ENTRYPOINT_ID,

        "executable_config_path":
            expected["executable_config_path"],
        "executable_config_sha256":
            expected["executable_config_sha256"],

        "main_frozen_executable_config_path":
            expected[
                "main_frozen_executable_config_path"
            ],
        "main_frozen_executable_config_sha256":
            expected[
                "main_frozen_executable_config_sha256"
            ],

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
        "training_protocol_sha256":
            expected["training_protocol_sha256"],
        "checkpoint_policy_sha256":
            expected["checkpoint_policy_sha256"],
        "ssl_protocol_sha256":
            expected["ssl_protocol_sha256"],
        "seed_protocol_sha256":
            expected["seed_protocol_sha256"],

        "optimizer":
            expected["optimizer"],
        "learning_rate":
            expected["learning_rate"],
        "weight_decay":
            expected["weight_decay"],
        "total_optimizer_updates":
            int(
                expected[
                    "total_optimizer_updates"
                ]
            ),
        "validation_interval_updates":
            int(
                expected[
                    "validation_interval_updates"
                ]
            ),
        "effective_labeled_batch":
            int(
                expected[
                    "effective_labeled_batch"
                ]
            ),
        "effective_unlabeled_batch":
            int(
                expected[
                    "effective_unlabeled_batch"
                ]
            ),
        "amp":
            bool(expected["amp"]),
        "checkpoint_selection_metric":
            expected[
                "checkpoint_selection_metric"
            ],

        "validation_only":
            True,
        "final_test_eligible":
            False,
        "test_usage":
            "PROHIBITED_NON_MAIN_ABLATION",
        "test_access_authorized":
            False,
        "final_test_accessed":
            False,
        "test_access_reason":
            "NON_MAIN_ABLATION_VALIDATION_ONLY",
        "final_evaluation_status":
            "NOT_ELIGIBLE",
        "training_status":
            "RUNNING",
    }

    for key, value in expected_pairs.items():
        if payload.get(key) != value:
            raise AblationAttemptClosureError(
                f"ABLATION_RUN_MANIFEST_MISMATCH:{key}"
            )

    resolved_hash = _require_sha256(
        payload[
            "resolved_config_sha256"
        ],
        label="ABLATION_RESOLVED_CONFIG",
    )

    config_hash_path = _require_file(
        attempt_dir,
        "config.sha256",
    )

    recorded_config_hash = (
        config_hash_path
        .read_text(
            encoding="utf-8-sig"
        )
        .strip()
        .lower()
    )

    if resolved_hash != recorded_config_hash:
        raise AblationAttemptClosureError(
            "ABLATION_RUN_MANIFEST_RESOLVED_CONFIG_HASH_MISMATCH"
        )

    runtime_identity = _require_sha256(
        payload[
            "runtime_identity_sha256"
        ],
        label="ABLATION_RUNTIME_IDENTITY",
    )

    if (
        payload[
            "environment_reference_path"
        ]
        != "environment_reference.json"
    ):
        raise AblationAttemptClosureError(
            "ABLATION_ENVIRONMENT_REFERENCE_PATH_INVALID"
        )

    if not str(
        payload["docker_image"]
    ).strip():
        raise AblationAttemptClosureError(
            "ABLATION_DOCKER_IMAGE_EMPTY"
        )

    if not str(
        payload["docker_digest"]
    ).strip():
        raise AblationAttemptClosureError(
            "ABLATION_DOCKER_DIGEST_EMPTY"
        )

    effective = payload[
        "effective_scientific_fields"
    ]

    if not isinstance(effective, dict):
        raise AblationAttemptClosureError(
            "ABLATION_EFFECTIVE_SCIENTIFIC_FIELDS_INVALID"
        )

    if (
        effective.get("ablation_family")
        != expected_pairs[
            "ablation_family"
        ]
    ):
        raise AblationAttemptClosureError(
            "ABLATION_EFFECTIVE_FAMILY_MISMATCH"
        )

    if (
        effective.get("ablation_level")
        != expected_pairs[
            "ablation_level"
        ]
    ):
        raise AblationAttemptClosureError(
            "ABLATION_EFFECTIVE_LEVEL_MISMATCH"
        )

    if (
        effective.get(
            "ofat_effective_semantics_valid"
        )
        is not True
    ):
        raise AblationAttemptClosureError(
            "ABLATION_OFAT_EFFECTIVE_SEMANTICS_INVALID"
        )

    if (
        effective.get("validation_only")
        is not True
    ):
        raise AblationAttemptClosureError(
            "ABLATION_EFFECTIVE_VALIDATION_ONLY_INVALID"
        )

    if (
        effective.get("final_test_eligible")
        is not False
    ):
        raise AblationAttemptClosureError(
            "ABLATION_EFFECTIVE_FINAL_TEST_ELIGIBILITY_INVALID"
        )

    _validate_ablation_environment(
        attempt_dir=attempt_dir,
        run_id=run_id,
        attempt_id=attempt_id,
        runtime_identity_sha256=runtime_identity,
    )

    return payload


def close_s9_11_ablation_attempt(
    *,
    attempt_dir: Path,
    run_id: str,
    attempt_id: str,
    identity: dict[str, Any],
    expected: dict[str, Any],
    validation_interval: int,
) -> dict[str, Any]:
    """Validate and close one S9.11 non-Main ablation attempt."""

    attempt_dir = Path(
        attempt_dir
    ).resolve()

    run_root = attempt_dir.parent

    if identity.get("run_id") != run_id:
        raise AblationAttemptClosureError(
            "ABLATION_CLOSURE_IDENTITY_RUN_ID_MISMATCH"
        )

    if identity.get("run_type") != RUN_TYPE:
        raise AblationAttemptClosureError(
            "ABLATION_CLOSURE_IDENTITY_RUN_TYPE_INVALID"
        )

    if (
        identity.get("condition_role")
        != CONDITION_ROLE
    ):
        raise AblationAttemptClosureError(
            "ABLATION_CLOSURE_IDENTITY_CONDITION_ROLE_INVALID"
        )

    if (
        identity.get("validation_only")
        is not True
    ):
        raise AblationAttemptClosureError(
            "ABLATION_CLOSURE_VALIDATION_ONLY_INVALID"
        )

    if (
        identity.get("final_test_eligible")
        is not False
    ):
        raise AblationAttemptClosureError(
            "ABLATION_CLOSURE_FINAL_TEST_ELIGIBILITY_INVALID"
        )

    if (
        identity.get("test_access_authorized")
        is not False
    ):
        raise AblationAttemptClosureError(
            "ABLATION_CLOSURE_TEST_ACCESS_AUTHORIZATION_INVALID"
        )

    if (
        identity.get("final_test_accessed")
        is not False
    ):
        raise AblationAttemptClosureError(
            "ABLATION_CLOSURE_FINAL_TEST_ACCESS_INVALID"
        )

    if (
        expected.get("final_test_accessed")
        is not False
    ):
        raise AblationAttemptClosureError(
            "ABLATION_EXPECTED_FINAL_TEST_ACCESS_INVALID"
        )

    _validate_ablation_run_root(
        run_root=run_root,
        identity=identity,
        attempt_id=attempt_id,
    )

    final_test_dir = (
        attempt_dir
        / "evaluation"
        / "final_test"
    )

    if final_test_dir.exists():
        raise AblationAttemptClosureError(
            "ABLATION_FINAL_TEST_ARTIFACTS_PROHIBITED"
        )

    mandatory_before_completion = (
        _mandatory_paths(METHOD)
    )

    try:
        for relative_path in (
            mandatory_before_completion
        ):
            _require_file(
                attempt_dir,
                relative_path,
            )

        _validate_pre_run(
            attempt_dir
            / "pre_run_assertions.json",
            run_id=run_id,
            attempt_id=attempt_id,
            expected=expected,
        )

        run_manifest = (
            _validate_ablation_run_manifest(
                attempt_dir=attempt_dir,
                run_id=run_id,
                attempt_id=attempt_id,
                expected=expected,
            )
        )

        _validate_resolved_config(
            attempt_dir
        )

        _validate_data_provenance(
            attempt_dir
            / "data_provenance.json",
            expected=expected,
        )

        _validate_seed_runtime(
            attempt_dir
            / "seed_runtime.json",
            expected=expected,
        )

        summary = _validate_training_summary(
            attempt_dir
            / "metrics"
            / "validation"
            / "training_summary.json",
            method=METHOD,
            expected=expected,
        )

        observed_updates = int(
            summary[
                "observed_optimizer_updates"
            ]
        )

        _validate_train_log(
            attempt_dir
            / "train.jsonl",
            method=METHOD,
            observed_updates=observed_updates,
        )

        _read_jsonl(
            attempt_dir
            / "runtime_events.jsonl"
        )

        _, last_entry = (
            _validate_validation_artifacts(
                attempt_dir,
                method=METHOD,
                observed_updates=observed_updates,
                validation_events_observed=int(
                    summary[
                        "validation_events_observed"
                    ]
                ),
            )
        )

        _validate_retry(
            attempt_dir
            / "retry_deviation.json",
            run_id=run_id,
            attempt_id=attempt_id,
            expected=expected,
        )

        _validate_qpseudo(
            attempt_dir,
            last_checkpoint_sha256=str(
                last_entry["sha256"]
            ),
        )

    except OfficialAttemptClosureError as exc:
        raise AblationAttemptClosureError(
            f"ABLATION_GENERIC_EVIDENCE_VALIDATION_FAILED:{exc}"
        ) from exc

    if (
        isinstance(validation_interval, bool)
        or not isinstance(
            validation_interval,
            int,
        )
        or validation_interval <= 0
    ):
        raise AblationAttemptClosureError(
            "ABLATION_VALIDATION_INTERVAL_INVALID"
        )

    if validation_interval != int(
        expected[
            "validation_interval_updates"
        ]
    ):
        raise AblationAttemptClosureError(
            "ABLATION_VALIDATION_INTERVAL_MISMATCH"
        )

    completion = {
        "run_id":
            run_id,
        "attempt_id":
            attempt_id,

        "run_type":
            RUN_TYPE,
        "condition_role":
            CONDITION_ROLE,

        "ablation_config_id":
            expected["ablation_config_id"],
        "ablation_family":
            expected["ablation_family"],
        "ablation_level":
            expected["ablation_level"],
        "main_run_id":
            expected["main_run_id"],

        "training_status":
            "CLOSED_VERIFIED",
        "training_completed":
            True,
        "training_valid":
            True,

        "expected_optimizer_updates":
            int(
                summary[
                    "expected_optimizer_updates"
                ]
            ),
        "observed_optimizer_updates":
            observed_updates,
        "validation_interval":
            validation_interval,

        "best_checkpoint_present":
            True,
        "last_checkpoint_present":
            True,

        "artifact_manifest_valid":
            True,
        "scientific_protocol_deviation":
            False,

        "validation_only":
            True,
        "eligible_for_final_test":
            False,
        "test_access_authorized":
            False,
        "final_test_accessed":
            False,
        "final_evaluation_status":
            "NOT_ELIGIBLE",

        "resolved_config_sha256":
            run_manifest[
                "resolved_config_sha256"
            ],
    }

    completion_path = (
        attempt_dir
        / "training_completion_report.json"
    )

    artifact_manifest_path = (
        attempt_dir
        / "artifact_manifest.json"
    )

    try:
        _write_json_atomic(
            completion_path,
            completion,
        )

        manifest_paths = (
            mandatory_before_completion
            + (
                "training_completion_report.json",
            )
        )

        artifacts: list[
            dict[str, Any]
        ] = []

        for relative_path in manifest_paths:
            path = _require_file(
                attempt_dir,
                relative_path,
            )

            role = ROLE_BY_PATH.get(
                relative_path
            )

            if role is None:
                raise AblationAttemptClosureError(
                    "ABLATION_ARTIFACT_ROLE_UNDEFINED:"
                    f"{relative_path}"
                )

            artifacts.append(
                {
                    "path":
                        relative_path,
                    "role":
                        role,
                    "size_bytes":
                        int(
                            path.stat().st_size
                        ),
                    "sha256":
                        file_sha256(path),
                    "required":
                        True,
                }
            )

        artifact_manifest = {
            "schema_version":
                ARTIFACT_SCHEMA_VERSION,
            "run_id":
                run_id,
            "attempt_id":
                attempt_id,
            "run_type":
                RUN_TYPE,
            "condition_role":
                CONDITION_ROLE,
            "validation_only":
                True,
            "eligible_for_final_test":
                False,
            "final_test_accessed":
                False,
            "artifacts":
                artifacts,
            "mandatory_artifacts_complete":
                True,
        }

        _write_json_atomic(
            artifact_manifest_path,
            artifact_manifest,
        )

        written_manifest = _load_json(
            artifact_manifest_path
        )

        if (
            written_manifest
            != artifact_manifest
        ):
            raise AblationAttemptClosureError(
                "ABLATION_ARTIFACT_MANIFEST_ROUNDTRIP_MISMATCH"
            )

        artifact_manifest_sha256 = (
            file_sha256(
                artifact_manifest_path
            )
        )

        _require_sha256(
            artifact_manifest_sha256,
            label="ABLATION_ARTIFACT_MANIFEST",
        )

        try:
            logical_status = (
                set_ablation_logical_run_closed_verified(
                    run_root=run_root,
                    identity=identity,
                    attempt_id=attempt_id,
                )
            )
        except AblationIdentityError as exc:
            raise AblationAttemptClosureError(
                f"ABLATION_LOGICAL_STATUS_CLOSURE_FAILED:{exc}"
            ) from exc

        if (
            logical_status[
                "eligible_for_final_test"
            ]
            is not False
        ):
            raise AblationAttemptClosureError(
                "ABLATION_LOGICAL_STATUS_FINAL_TEST_ELIGIBILITY_INVALID"
            )

        if (
            logical_status[
                "final_test_accessed"
            ]
            is not False
        ):
            raise AblationAttemptClosureError(
                "ABLATION_LOGICAL_STATUS_FINAL_TEST_ACCESS_INVALID"
            )

    except Exception:
        for closure_path in (
            artifact_manifest_path,
            completion_path,
        ):
            try:
                if closure_path.exists():
                    closure_path.unlink()
            except OSError:
                pass

        raise

    return {
        "training_completion_report":
            str(completion_path),
        "training_completion_report_sha256":
            file_sha256(
                completion_path
            ),

        "artifact_manifest":
            str(
                artifact_manifest_path
            ),
        "artifact_manifest_sha256":
            file_sha256(
                artifact_manifest_path
            ),

        "mandatory_artifact_count":
            len(artifacts),
        "mandatory_artifacts_complete":
            True,

        "training_status":
            "CLOSED_VERIFIED",
        "training_valid":
            True,

        "validation_only":
            True,
        "eligible_for_final_test":
            False,
        "test_access_authorized":
            False,
        "final_test_accessed":
            False,
        "final_evaluation_status":
            "NOT_ELIGIBLE",
    }


__all__ = [
    "AblationAttemptClosureError",
    "close_s9_11_ablation_attempt",
]