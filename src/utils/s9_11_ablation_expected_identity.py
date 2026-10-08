"""S9.11 canonical expected identity for validation-only OFAT ablation runs.

Design
------
Every S9.11 non-Main observation derives its scientific/runtime foundation
from the frozen official SSL Faster R-CNN R50-FPN 10% Main condition.

The canonical Main expected identity remains authoritative for:

- scientific / implementation / artifact contract hashes;
- frozen train/validation/test and L_10/U_10 memberships;
- preprocessing and labeled pipeline;
- optimizer and optimizer-update budget;
- validation evaluator;
- checkpoint and SSL protocol;
- partition seed and locked training-seed identity;
- hidden-U firewall.

S9.11 overlays only the ablation execution identity:

- run_type = ABLATION
- condition_role = ABLATION
- one prespecified non-Main config C0-C3/A0-A2/F0-F1
- executable ablation-config identity
- same-seed frozen Main reference
- validation-only / final-test-ineligible firewall

The frozen Main executable config identity is retained separately so later
OFAT verification can prove that every non-Main variant differs from the
same frozen Main only in its prespecified target scientific factor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.pre_run_guardrails import (
    build_expected_pre_run_identity,
)
from src.utils.run_manifest import file_sha256
from src.utils.s9_11_ablation_identity import (
    CONDITION_ROLE,
    RUN_TYPE,
    build_ablation_identity,
)


class AblationExpectedIdentityError(RuntimeError):
    """Raised when canonical Main -> S9.11 identity binding is invalid."""


def build_ablation_expected_identity(
    repo_root: Path,
    row: dict[str, Any],
) -> dict[str, Any]:
    """Build one canonical expected identity for a non-Main S9.11 run.

    The function first validates the ablation inventory row using the
    S9.11 identity foundation. It then asks the already-validated S9
    official guardrail implementation to materialize the canonical frozen
    SSL/R50/10% Main identity for the same seed.

    No data, optimizer, evaluator, seed, update-budget or protocol identity
    is independently reconstructed here.
    """

    repo_root = Path(repo_root).resolve()

    ablation = build_ablation_identity(
        repo_root,
        row,
    )

    # Official architecture identity uses the canonical short token R50.
    # S9.11 inventory uses the descriptive R50-FPN label, but both identify
    # the same locked Faster R-CNN R50-FPN scientific architecture.
    main = build_expected_pre_run_identity(
        repo_root,
        method="SSL",
        architecture="R50-FPN",
        budget="10pct",
        seed_index=int(ablation["seed_index"]),
    )

    if main.get("condition_role") != "MAIN_LOW_LABEL":
        raise AblationExpectedIdentityError(
            "MAIN_REFERENCE_CONDITION_ROLE_INVALID"
        )

    if main.get("method") != "SSL":
        raise AblationExpectedIdentityError(
            "MAIN_REFERENCE_METHOD_INVALID"
        )

    if main.get("architecture") != "R50-FPN":
        raise AblationExpectedIdentityError(
            "MAIN_REFERENCE_ARCHITECTURE_INVALID"
        )

    if main.get("budget") != "10pct":
        raise AblationExpectedIdentityError(
            "MAIN_REFERENCE_BUDGET_INVALID"
        )

    if int(main["training_seed"]) != int(
        ablation["training_seed"]
    ):
        raise AblationExpectedIdentityError(
            "MAIN_REFERENCE_TRAINING_SEED_MISMATCH"
        )

    if int(main["seed_index"]) != int(
        ablation["seed_index"]
    ):
        raise AblationExpectedIdentityError(
            "MAIN_REFERENCE_SEED_INDEX_MISMATCH"
        )

    expected_main_run_id = (
        f"OFF_SSL_R50_B10_S{int(ablation['seed_index']):02d}"
    )

    if ablation["main_run_id"] != expected_main_run_id:
        raise AblationExpectedIdentityError(
            "MAIN_REFERENCE_RUN_ID_MISMATCH"
        )

    ablation_config_rel = str(
        ablation["executable_config_path"]
    )

    ablation_config = (
        repo_root
        / ablation_config_rel
    ).resolve()

    try:
        ablation_config.relative_to(repo_root)
    except ValueError as exc:
        raise AblationExpectedIdentityError(
            "ABLATION_CONFIG_OUTSIDE_REPOSITORY"
        ) from exc

    if not ablation_config.is_file():
        raise AblationExpectedIdentityError(
            f"ABLATION_CONFIG_MISSING:{ablation_config_rel}"
        )

    expected_ablation_prefix = (
        Path("configs")
        / "ablation"
    )

    relative_path = ablation_config.relative_to(
        repo_root
    )

    if (
        len(relative_path.parts) < 3
        or Path(*relative_path.parts[:2])
        != expected_ablation_prefix
    ):
        raise AblationExpectedIdentityError(
            "ABLATION_CONFIG_NOT_IN_CONFIGS_ABLATION"
        )

    ablation_config_sha256 = file_sha256(
        ablation_config
    )

    main_config_path = str(
        main["executable_config_path"]
    )

    main_config_sha256 = str(
        main["executable_config_sha256"]
    )

    # Start with the canonical frozen Main identity. This preserves all
    # scientific/runtime identities that are NOT the OFAT target factor.
    expected = dict(main)

    # Preserve explicit same-seed Main provenance before replacing the
    # executable config fields with the non-Main ablation config.
    expected[
        "main_run_id"
    ] = ablation["main_run_id"]

    expected[
        "main_frozen_executable_config_path"
    ] = main_config_path

    expected[
        "main_frozen_executable_config_sha256"
    ] = main_config_sha256

    # S9.11 logical-run identity.
    expected["run_id"] = ablation["run_id"]
    expected["run_type"] = RUN_TYPE
    expected["condition_role"] = CONDITION_ROLE

    expected[
        "ablation_config_id"
    ] = ablation["ablation_config_id"]

    expected[
        "ablation_family"
    ] = ablation["family"]

    expected[
        "ablation_level"
    ] = ablation["level"]

    scientific_change = str(
        row.get("scientific_change", "")
    ).strip()

    if not scientific_change:
        raise AblationExpectedIdentityError(
            "ABLATION_SCIENTIFIC_CHANGE_MISSING"
        )

    expected[
        "scientific_change"
    ] = scientific_change

    # Execution uses the prespecified non-Main child config. Its raw-byte
    # SHA-256 is recorded independently from the frozen Main config hash.
    expected[
        "executable_config_path"
    ] = ablation_config_rel

    expected[
        "executable_config_sha256"
    ] = ablation_config_sha256

    # Human-readable architecture identity used by the ablation inventory.
    # The inherited canonical architecture token remains separately retained.
    expected[
        "ablation_architecture"
    ] = ablation["architecture"]

    # Explicit secondary mechanistic-analysis firewall.
    expected["validation_only"] = True
    expected["final_test_eligible"] = False
    expected["test_usage"] = "PROHIBITED_NON_MAIN_ABLATION"
    expected["test_access_authorized"] = False
    expected["final_test_accessed"] = False
    expected[
        "test_access_reason"
    ] = "NON_MAIN_ABLATION_VALIDATION_ONLY"
    expected[
        "final_evaluation_status"
    ] = "NOT_ELIGIBLE"

    # Guard against accidental mutation of Main scientific anchors.
    if expected["training_seed"] != main["training_seed"]:
        raise AblationExpectedIdentityError(
            "ABLATION_OVERLAY_CHANGED_TRAINING_SEED"
        )

    if expected["partition_seed"] != main["partition_seed"]:
        raise AblationExpectedIdentityError(
            "ABLATION_OVERLAY_CHANGED_PARTITION_SEED"
        )

    if (
        expected["total_optimizer_updates"]
        != main["total_optimizer_updates"]
    ):
        raise AblationExpectedIdentityError(
            "ABLATION_OVERLAY_CHANGED_UPDATE_BUDGET"
        )

    if (
        expected["evaluator_config_sha256"]
        != main["evaluator_config_sha256"]
    ):
        raise AblationExpectedIdentityError(
            "ABLATION_OVERLAY_CHANGED_EVALUATOR"
        )

    if expected["labeled_manifest_sha256"] != main[
        "labeled_manifest_sha256"
    ]:
        raise AblationExpectedIdentityError(
            "ABLATION_OVERLAY_CHANGED_LABELED_MEMBERSHIP"
        )

    if expected["unlabeled_manifest_sha256"] != main[
        "unlabeled_manifest_sha256"
    ]:
        raise AblationExpectedIdentityError(
            "ABLATION_OVERLAY_CHANGED_UNLABELED_MEMBERSHIP"
        )

    return expected


__all__ = [
    "AblationExpectedIdentityError",
    "build_ablation_expected_identity",
]