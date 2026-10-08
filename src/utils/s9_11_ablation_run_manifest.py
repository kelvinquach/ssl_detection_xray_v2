"""Materialize canonical S9.11 non-Main ablation run_manifest.json.

Scientific role
---------------
S9.11 is a validation-only secondary mechanistic OFAT campaign anchored to
the frozen official SSL Faster R-CNN R50-FPN 10% Main condition.

Each non-Main logical run must retain all frozen Main scientific/runtime
identities except exactly one prespecified ablation factor:

    C family:
        RCNN classification pseudo-label confidence threshold

    A family:
        Student strong-view brightness/contrast augmentation

    F family:
        RCNN bbox-regression pseudo-label reliability filtering

This manifest binds four evidence layers:

1. canonical Main-derived expected identity;
2. exact resolved MMEngine config SHA-256;
3. effective scientific semantics extracted from that resolved config
   and, where required, the runtime detector implementation;
4. runtime environment evidence captured with the same snapshot mechanism
   used by official execution.

The artifact is explicitly ABLATION, validation-only, and never eligible for
final-test access.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.utils.official_run_manifest import (
    _runtime_docker_identity,
    _write_json_atomic,
)
from src.utils.run_manifest import (
    git_head,
    load_json,
    load_yaml,
)


SCHEMA_VERSION = "1.0"
ENTRYPOINT_ID = "scripts/S9_run_ablation.py"

RUN_TYPE = "ABLATION"
CONDITION_ROLE = "ABLATION"
METHOD = "SSL"
ARCHITECTURE = "R50-FPN"
BUDGET = "10pct"

ENVIRONMENT_ARTIFACT_TYPE = (
    "ABLATION_ENVIRONMENT_REFERENCE"
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class AblationRunManifestError(RuntimeError):
    """Raised when one S9.11 ablation run manifest is invalid."""


def _require_sha256(
    value: Any,
    *,
    label: str,
) -> str:
    value = str(value).strip()

    if SHA256_RE.fullmatch(value) is None:
        raise AblationRunManifestError(
            f"{label}_INVALID_SHA256"
        )

    return value


def _resolved_config_sha256(
    attempt_dir: Path,
) -> str:
    """Read and verify the previously materialized resolved-config hash."""

    config_path = attempt_dir / "resolved_config.yaml"
    hash_path = attempt_dir / "config.sha256"

    if not config_path.is_file():
        raise AblationRunManifestError(
            "RESOLVED_CONFIG_MISSING"
        )

    if not hash_path.is_file():
        raise AblationRunManifestError(
            "RESOLVED_CONFIG_HASH_MISSING"
        )

    recorded = _require_sha256(
        hash_path.read_text(
            encoding="ascii"
        ).strip(),
        label="RESOLVED_CONFIG",
    )

    import hashlib

    digest = hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest()

    if digest != recorded:
        raise AblationRunManifestError(
            "RESOLVED_CONFIG_HASH_MISMATCH"
        )

    return recorded


def _environment_identity(
    *,
    attempt_dir: Path,
    run_id: str,
    attempt_id: str,
) -> tuple[str, str]:
    """Validate ablation environment evidence and return runtime identity."""

    path = (
        attempt_dir
        / "environment_reference.json"
    )

    if not path.is_file():
        raise AblationRunManifestError(
            "ABLATION_ENVIRONMENT_REFERENCE_MISSING"
        )

    payload = load_json(path)

    if (
        payload.get("artifact_type")
        != ENVIRONMENT_ARTIFACT_TYPE
    ):
        raise AblationRunManifestError(
            "ABLATION_ENVIRONMENT_ARTIFACT_TYPE_INVALID"
        )

    if payload.get("run_type") != RUN_TYPE:
        raise AblationRunManifestError(
            "ABLATION_ENVIRONMENT_RUN_TYPE_INVALID"
        )

    if (
        payload.get("condition_role")
        != CONDITION_ROLE
    ):
        raise AblationRunManifestError(
            "ABLATION_ENVIRONMENT_CONDITION_ROLE_INVALID"
        )

    if payload.get("run_id") != run_id:
        raise AblationRunManifestError(
            "ABLATION_ENVIRONMENT_RUN_ID_MISMATCH"
        )

    if payload.get("attempt_id") != attempt_id:
        raise AblationRunManifestError(
            "ABLATION_ENVIRONMENT_ATTEMPT_ID_MISMATCH"
        )

    original = payload.get(
        "original_environment"
    )

    if not isinstance(original, dict):
        raise AblationRunManifestError(
            "ABLATION_ORIGINAL_ENVIRONMENT_INVALID"
        )

    runtime_identity = _require_sha256(
        original.get(
            "runtime_identity_sha256"
        ),
        label="ABLATION_RUNTIME_IDENTITY",
    )

    return (
        runtime_identity,
        path.name,
    )


def materialize_s9_11_ablation_run_manifest(
    *,
    repo_root: Path,
    attempt_dir: Path,
    run_id: str,
    attempt_id: str,
    expected: dict[str, Any],
    effective_fields: dict[str, Any],
) -> Path:
    """Write one RUNNING-state S9.11 ablation run manifest."""

    repo_root = Path(
        repo_root
    ).resolve()

    attempt_dir = Path(
        attempt_dir
    ).resolve()

    run_id = str(
        run_id
    ).strip()

    attempt_id = str(
        attempt_id
    ).strip()

    if not run_id.startswith(
        "ABL_R50_B10_"
    ):
        raise AblationRunManifestError(
            f"INVALID_ABLATION_RUN_ID:{run_id!r}"
        )

    if not attempt_id:
        raise AblationRunManifestError(
            "ABLATION_ATTEMPT_ID_EMPTY"
        )

    # ------------------------------------------------------------------
    # Canonical logical identity.
    # ------------------------------------------------------------------

    if expected.get("run_id") != run_id:
        raise AblationRunManifestError(
            "EXPECTED_RUN_ID_MISMATCH"
        )

    if expected.get("run_type") != RUN_TYPE:
        raise AblationRunManifestError(
            "EXPECTED_RUN_TYPE_INVALID"
        )

    if (
        expected.get("condition_role")
        != CONDITION_ROLE
    ):
        raise AblationRunManifestError(
            "EXPECTED_CONDITION_ROLE_INVALID"
        )

    if expected.get("method") != METHOD:
        raise AblationRunManifestError(
            "EXPECTED_METHOD_INVALID"
        )

    if (
        expected.get("architecture")
        != ARCHITECTURE
    ):
        raise AblationRunManifestError(
            "EXPECTED_ARCHITECTURE_INVALID"
        )

    if expected.get("budget") != BUDGET:
        raise AblationRunManifestError(
            "EXPECTED_BUDGET_INVALID"
        )

    if expected.get("validation_only") is not True:
        raise AblationRunManifestError(
            "ABLATION_VALIDATION_ONLY_FIREWALL_INVALID"
        )

    if (
        expected.get("final_test_eligible")
        is not False
    ):
        raise AblationRunManifestError(
            "ABLATION_FINAL_TEST_ELIGIBILITY_INVALID"
        )

    if (
        expected.get("test_access_authorized")
        is not False
    ):
        raise AblationRunManifestError(
            "ABLATION_TEST_ACCESS_AUTHORIZATION_INVALID"
        )

    if expected.get("final_test_accessed") is not False:
        raise AblationRunManifestError(
            "ABLATION_TEST_ACCESS_STATE_INVALID"
        )

    # ------------------------------------------------------------------
    # Effective scientific semantics must agree with the inventory-bound
    # ablation family and level.
    # ------------------------------------------------------------------

    expected_family = str(
        expected["ablation_family"]
    ).strip().upper()

    expected_level = str(
        expected["ablation_level"]
    ).strip().upper()

    if (
        effective_fields.get(
            "ablation_family"
        )
        != expected_family
    ):
        raise AblationRunManifestError(
            "EFFECTIVE_ABLATION_FAMILY_MISMATCH"
        )

    if (
        effective_fields.get(
            "ablation_level"
        )
        != expected_level
    ):
        raise AblationRunManifestError(
            "EFFECTIVE_ABLATION_LEVEL_MISMATCH"
        )

    if (
        effective_fields.get(
            "ofat_effective_semantics_valid"
        )
        is not True
    ):
        raise AblationRunManifestError(
            "EFFECTIVE_OFAT_SEMANTICS_NOT_VALID"
        )

    if (
        effective_fields.get(
            "validation_only"
        )
        is not True
    ):
        raise AblationRunManifestError(
            "EFFECTIVE_VALIDATION_ONLY_INVALID"
        )

    if (
        effective_fields.get(
            "final_test_eligible"
        )
        is not False
    ):
        raise AblationRunManifestError(
            "EFFECTIVE_FINAL_TEST_ELIGIBILITY_INVALID"
        )

    # ------------------------------------------------------------------
    # Bind exact config and environment identities.
    # ------------------------------------------------------------------

    resolved_config_sha256 = (
        _resolved_config_sha256(
            attempt_dir
        )
    )

    (
        runtime_identity_sha256,
        environment_reference_path,
    ) = _environment_identity(
        attempt_dir=attempt_dir,
        run_id=run_id,
        attempt_id=attempt_id,
    )

    # Canonical frozen Docker identity is intentionally resolved through
    # the same helper used by official Main execution.
    docker_image, docker_digest = (
        _runtime_docker_identity(
            repo_root
        )
    )

    training = load_yaml(
        repo_root
        / "configs"
        / "protocol"
        / "d4_training_protocol.yaml"
    )

    # ------------------------------------------------------------------
    # Manifest.
    # ------------------------------------------------------------------

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,

        "run_id": run_id,
        "attempt_id": attempt_id,

        "run_type": RUN_TYPE,
        "condition_role": CONDITION_ROLE,
        "method": METHOD,
        "architecture": ARCHITECTURE,
        "budget": BUDGET,

        "ablation_config_id":
            expected["ablation_config_id"],

        "ablation_family":
            expected_family,

        "ablation_level":
            expected_level,

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

        "git_commit":
            git_head(repo_root),

        "docker_image":
            docker_image,

        "docker_digest":
            docker_digest,

        "runtime_identity_sha256":
            runtime_identity_sha256,

        "environment_reference_path":
            environment_reference_path,

        "entrypoint_id":
            ENTRYPOINT_ID,

        # Raw child config identity.
        "executable_config_path":
            expected["executable_config_path"],

        "executable_config_sha256":
            expected["executable_config_sha256"],

        # Frozen same-science Main config retained separately.
        "main_frozen_executable_config_path":
            expected[
                "main_frozen_executable_config_path"
            ],

        "main_frozen_executable_config_sha256":
            expected[
                "main_frozen_executable_config_sha256"
            ],

        # Exact resolved MMEngine configuration used by Runner.
        "resolved_config_sha256":
            resolved_config_sha256,

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

        "gradient_clipping":
            bool(
                training[
                    "numerical"
                ][
                    "gradient_clipping"
                ]
            ),

        "checkpoint_selection_metric":
            expected[
                "checkpoint_selection_metric"
            ],

        # Actual C/A/F semantics that will be executed.
        "effective_scientific_fields":
            effective_fields,

        # Hard non-Main firewall.
        "validation_only": True,
        "final_test_eligible": False,
        "test_usage":
            "PROHIBITED_NON_MAIN_ABLATION",
        "test_access_authorized": False,
        "final_test_accessed": False,
        "test_access_reason":
            "NON_MAIN_ABLATION_VALIDATION_ONLY",
        "final_evaluation_status":
            "NOT_ELIGIBLE",

        "training_status":
            "RUNNING",
    }

    output = (
        attempt_dir
        / "run_manifest.json"
    )

    if output.exists():
        raise AblationRunManifestError(
            "ABLATION_RUN_MANIFEST_ALREADY_EXISTS"
        )

    _write_json_atomic(
        output,
        manifest,
    )

    observed = load_json(
        output
    )

    if observed != manifest:
        raise AblationRunManifestError(
            "ABLATION_RUN_MANIFEST_POSTWRITE_MISMATCH"
        )

    return output


__all__ = [
    "AblationRunManifestError",
    "materialize_s9_11_ablation_run_manifest",
]