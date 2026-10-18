"""S2.14 fail-fast pre-run guardrails for source/config/data/seed identity."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from src.utils.run_manifest import (
    evaluator_identity,
    file_sha256,
    fixed_data_identity,
    labeled_pipeline_identity,
    load_json,
    load_yaml,
    optimizer_identity,
    preprocessing_identity,
    semantic_sha256,
)
from src.utils.unlabeled_firewall import (
    assert_current_ssl_execution_firewall,
)


SUPPORTED_METHODS = {"SUP", "SSL"}
SUPPORTED_ARCHITECTURES = {"R50-FPN", "Swin-T-FPN"}
SUPPORTED_LOW_LABEL_BUDGETS = {"1pct", "5pct", "10pct", "20pct"}
SUPPORTED_SUP_REFERENCE_BUDGETS = {"100pct"}
LOCKED_PARTITION_SEED = 42
CANONICAL_SCIENTIFIC_SOURCE = "sn-article.tex"


class PreRunGuardrailError(RuntimeError):
    """Raised when a pre-run identity assertion blocks execution."""


def _repo_relative_path(
    repo_root: Path,
    path: Path,
) -> str:
    """Return one repository-relative POSIX path."""
    return (
        Path(path)
        .resolve()
        .relative_to(Path(repo_root).resolve())
        .as_posix()
    )


def _git_blob_content_sha256(
    repo_root: Path,
    path: Path,
) -> str:
    """Hash committed file content using the canonical Git-blob basis."""
    repo_root = Path(repo_root).resolve()
    rel_path = _repo_relative_path(repo_root, path)

    process = subprocess.run(
        ["git", "show", f"HEAD:{rel_path}"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if process.returncode != 0:
        raise PreRunGuardrailError(
            "Cannot read canonical Git content for "
            f"{rel_path}: "
            + process.stderr.decode(
                "utf-8",
                errors="replace",
            ).strip()
        )

    return hashlib.sha256(process.stdout).hexdigest()


def _worktree_matches_head(
    repo_root: Path,
    path: Path,
) -> bool:
    """Require current worktree content to match HEAD under Git semantics."""
    repo_root = Path(repo_root).resolve()
    rel_path = _repo_relative_path(repo_root, path)

    process = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            "HEAD",
            "--",
            rel_path,
        ],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

    return process.returncode == 0


def verify_locked_source_integrity(
    repo_root: Path,
) -> dict[str, bool]:
    """Verify canonical source/contract files against locked manifests."""
    repo_root = Path(repo_root).resolve()
    governance_root = repo_root / "artifacts" / "governance"

    scientific = load_json(
        governance_root / "scientific_source_manifest.json"
    )
    implementation = load_json(
        governance_root / "implementation_contract_manifest.json"
    )
    artifact = load_json(
        governance_root / "artifact_contract_manifest.json"
    )

    scientific_path = repo_root / scientific["scientific_source_name"]
    implementation_path = (
        repo_root / implementation["implementation_contract_name"]
    )
    artifact_path = repo_root / artifact["artifact_contract_name"]

    checks = {
        "scientific_source_name_canonical":
            scientific["scientific_source_name"]
            == CANONICAL_SCIENTIFIC_SOURCE,

        "scientific_source_exists":
            scientific_path.is_file(),

        "implementation_contract_exists":
            implementation_path.is_file(),

        "artifact_contract_exists":
            artifact_path.is_file(),

        "scientific_source_worktree_matches_head":
            scientific_path.is_file()
            and _worktree_matches_head(
                repo_root,
                scientific_path,
            ),

        "implementation_contract_worktree_matches_head":
            implementation_path.is_file()
            and _worktree_matches_head(
                repo_root,
                implementation_path,
            ),

        "artifact_contract_worktree_matches_head":
            artifact_path.is_file()
            and _worktree_matches_head(
                repo_root,
                artifact_path,
            ),

        "scientific_source_hash_matches_lock":
            scientific_path.is_file()
            and _git_blob_content_sha256(
                repo_root,
                scientific_path,
            )
            == scientific["scientific_source_sha256"],

        "implementation_contract_hash_matches_lock":
            implementation_path.is_file()
            and _git_blob_content_sha256(
                repo_root,
                implementation_path,
            )
            == implementation["implementation_contract_sha256"],

        "artifact_contract_hash_matches_lock":
            artifact_path.is_file()
            and _git_blob_content_sha256(
                repo_root,
                artifact_path,
            )
            == artifact["artifact_contract_sha256"],

        "implementation_manifest_scientific_hash_consistent":
            implementation["scientific_source_sha256"]
            == scientific["scientific_source_sha256"],

        "artifact_manifest_scientific_hash_consistent":
            artifact["scientific_source_sha256"]
            == scientific["scientific_source_sha256"],

        "artifact_manifest_implementation_hash_consistent":
            artifact["implementation_contract_sha256"]
            == implementation["implementation_contract_sha256"],
    }

    return checks


def locked_seed_identity(
    repo_root: Path,
) -> dict[str, Any]:
    """Return and cross-check the locked partition/training-seed identity."""
    repo_root = Path(repo_root).resolve()

    seed_manifest_path = (
        repo_root / "data" / "manifests" / "seed_manifest.json"
    )
    seed_state_path = (
        repo_root / "data" / "manifests" / "seed_state_manifest.json"
    )
    seed_protocol_path = (
        repo_root
        / "configs"
        / "protocol"
        / "phase2F1_seed_protocol.yaml"
    )

    manifest = load_json(seed_manifest_path)
    state = load_json(seed_state_path)
    protocol = load_yaml(seed_protocol_path)

    manifest_values = manifest["training_seed"][
        "ordered_training_seed_values"
    ]
    state_values = state["ordered_training_seed_values"]
    protocol_values = protocol["training_seed"][
        "ordered_training_seed_values"
    ]

    protocol_seed_records = protocol["training_seed"][
        "ordered_training_seeds"
    ]
    protocol_record_values = [
        record["training_seed"]
        for record in protocol_seed_records
    ]

    manifest_seed_records = manifest["training_seed"]["seeds"]
    manifest_record_values = [
        record["training_seed"]
        for record in manifest_seed_records
    ]

    actual_seed_protocol_sha = file_sha256(seed_protocol_path)

    checks = {
        "partition_seed_manifest_locked":
            manifest["partition"]["partition_seed"]
            == LOCKED_PARTITION_SEED,

        "partition_seed_state_locked":
            state["partition_seed"]
            == LOCKED_PARTITION_SEED,

        "partition_seed_protocol_locked":
            protocol["partition"]["partition_seed"]
            == LOCKED_PARTITION_SEED,

        "training_seed_count_manifest":
            manifest["training_seed"]["training_seed_count"] == 10,

        "training_seed_count_state":
            state["training_seed_count"] == 10,

        "training_seed_count_protocol":
            protocol["training_seed"]["training_seed_count"] == 10,

        "ordered_seed_list_manifest_state_match":
            manifest_values == state_values,

        "ordered_seed_list_manifest_protocol_match":
            manifest_values == protocol_values,

        "ordered_seed_records_protocol_match":
            protocol_record_values == protocol_values,

        "ordered_seed_records_manifest_match":
            manifest_record_values == manifest_values,

        "ordered_seed_list_unique":
            len(set(manifest_values)) == 10,

        "seed_protocol_file_hash_matches_manifest":
            actual_seed_protocol_sha
            == manifest["source_config_sha256"],

        "seed_protocol_file_hash_matches_state":
            actual_seed_protocol_sha
            == state["source_config_sha256"],

        "seed_search_forbidden":
            manifest["training_seed"]["selection_constraints"][
                "seed_search_performed"
            ] is False,

        "replacement_seed_forbidden":
            manifest["training_seed"]["selection_constraints"][
                "additional_seeds_allowed"
            ] is False
            and manifest["retry_policy"][
                "retry_changes_training_seed"
            ] is False,

        "training_seed_may_not_alter_partition":
            manifest["partition"]["constraints"][
                "training_seed_may_alter_partition"
            ] is False,
    }

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise PreRunGuardrailError(
            "Locked seed identity failed: " + ", ".join(failed)
        )

    return {
        "partition_seed": LOCKED_PARTITION_SEED,
        "ordered_training_seed_values": list(manifest_values),
        "seed_index_to_training_seed": {
            index + 1: value
            for index, value in enumerate(manifest_values)
        },
        "seed_protocol_sha256": actual_seed_protocol_sha,
        "checks": checks,
    }


def _sup_reference_data_identity(
    repo_root: Path,
) -> dict[str, str]:
    """Build the frozen full-train identity for 100%-SUP reference runs."""
    repo_root = Path(repo_root).resolve()

    split_path = (
        repo_root
        / "data"
        / "manifests"
        / "split_lock_manifest.json"
    )
    frozen_data_path = (
        repo_root
        / "artifacts"
        / "frozen"
        / "frozen_data_manifest.json"
    )

    split = load_json(split_path)
    frozen_data = load_json(frozen_data_path)

    if frozen_data.get("status") != "PASS":
        raise PreRunGuardrailError(
            "Frozen data manifest is not PASS."
        )

    train = frozen_data["train"]

    if (
        train["membership_sha256"]
        != split["image_id_sha256"]["train"]
    ):
        raise PreRunGuardrailError(
            "100%-SUP train membership does not match "
            "the fixed train split."
        )

    return {
        "dataset_manifest_sha256": file_sha256(split_path),
        "train_split_sha256":
            split["image_id_sha256"]["train"],
        "val_split_sha256":
            split["image_id_sha256"]["val"],
        "test_split_sha256":
            split["image_id_sha256"]["test"],
        "labeled_manifest_sha256":
            train["coco_sha256"],
        "unlabeled_manifest_sha256":
            "NOT_APPLICABLE",
        "full_train_membership_sha256":
            train["membership_sha256"],
        "full_train_coco_sha256":
            train["coco_sha256"],
    }


def _frozen_executable_config_identity(
    repo_root: Path,
    *,
    method: str,
    architecture: str,
    budget: str,
) -> dict[str, str]:
    """Resolve and verify exactly one frozen executable config."""
    repo_root = Path(repo_root).resolve()

    manifest_path = (
        repo_root
        / "artifacts"
        / "frozen"
        / "frozen_config_manifest.json"
    )
    manifest = load_json(manifest_path)

    matches = [
        item
        for item in manifest["executable_configs"]
        if item["method"] == method
        and item["architecture"] == architecture
        and item["budget"] == budget
    ]

    if len(matches) != 1:
        raise PreRunGuardrailError(
            "Expected exactly one frozen executable config for "
            f"{method}/{architecture}/{budget}; "
            f"observed={len(matches)}"
        )

    item = matches[0]
    config_path = repo_root / item["path"]

    if not config_path.is_file():
        raise PreRunGuardrailError(
            "Frozen executable config is missing: "
            f"{item['path']}"
        )

    if not _worktree_matches_head(
        repo_root,
        config_path,
    ):
        raise PreRunGuardrailError(
            "Frozen executable config differs from HEAD: "
            f"{item['path']}"
        )

    observed_sha256 = _git_blob_content_sha256(
        repo_root,
        config_path,
    )

    if observed_sha256 != item["sha256"]:
        raise PreRunGuardrailError(
            "Frozen executable config hash mismatch: "
            f"{item['path']}"
        )

    return {
        "executable_config_path": item["path"],
        "executable_config_sha256": item["sha256"],
    }


def _protocol_config_identity(
    repo_root: Path,
    *,
    method: str,
) -> dict[str, str]:
    """Build semantic identities of current locked executable protocols."""
    training_path = (
        repo_root / "configs" / "protocol" / "d4_training_protocol.yaml"
    )
    ssl_path = (
        repo_root / "configs" / "protocol" / "d4_ssl_protocol.yaml"
    )
    checkpoint_path = (
        repo_root / "configs" / "protocol" / "checkpoint_policy.yaml"
    )

    training = load_yaml(training_path)
    checkpoint = load_yaml(checkpoint_path)

    result = {
        "training_protocol_sha256": semantic_sha256(training),
        "checkpoint_policy_sha256": semantic_sha256(checkpoint),
    }

    if method == "SSL":
        ssl = load_yaml(ssl_path)
        result["ssl_protocol_sha256"] = semantic_sha256(ssl)
    else:
        result["ssl_protocol_sha256"] = "NOT_APPLICABLE"

    return result


def build_expected_pre_run_identity(
    repo_root: Path,
    *,
    method: str,
    architecture: str,
    budget: str,
    seed_index: int,
) -> dict[str, Any]:
    """Materialize the canonical pre-run identity for one official run."""
    repo_root = Path(repo_root).resolve()

    if method not in SUPPORTED_METHODS:
        raise PreRunGuardrailError(
            f"Unsupported method: {method!r}"
        )

    if architecture not in SUPPORTED_ARCHITECTURES:
        raise PreRunGuardrailError(
            f"Unsupported architecture: {architecture!r}"
        )

    is_low_label = budget in SUPPORTED_LOW_LABEL_BUDGETS
    is_sup_reference = (
        method == "SUP"
        and budget in SUPPORTED_SUP_REFERENCE_BUDGETS
    )

    if not is_low_label and not is_sup_reference:
        raise PreRunGuardrailError(
            "Unsupported official method/budget combination: "
            f"{method}/{budget}"
        )

    source_checks = verify_locked_source_integrity(repo_root)
    failed_sources = [
        name for name, passed in source_checks.items() if not passed
    ]
    if failed_sources:
        raise PreRunGuardrailError(
            "Locked source integrity failed: "
            + ", ".join(failed_sources)
        )

    seed = locked_seed_identity(repo_root)

    if (
        not isinstance(seed_index, int)
        or isinstance(seed_index, bool)
        or seed_index not in seed["seed_index_to_training_seed"]
    ):
        raise PreRunGuardrailError(
            f"Invalid seed_index: {seed_index!r}"
        )

    training_seed = seed["seed_index_to_training_seed"][seed_index]

    governance_root = repo_root / "artifacts" / "governance"
    scientific = load_json(
        governance_root / "scientific_source_manifest.json"
    )
    implementation = load_json(
        governance_root / "implementation_contract_manifest.json"
    )
    artifact = load_json(
        governance_root / "artifact_contract_manifest.json"
    )

    if is_sup_reference:
        data = _sup_reference_data_identity(repo_root)
    else:
        data = fixed_data_identity(
            repo_root,
            budget=budget,
        )

    config_identity = _frozen_executable_config_identity(
        repo_root,
        method=method,
        architecture=architecture,
        budget=budget,
    )

    _, preprocessing_sha = preprocessing_identity(repo_root)
    _, labeled_pipeline_sha = labeled_pipeline_identity(repo_root)
    optimizer, optimizer_sha = optimizer_identity(
        repo_root,
        architecture=architecture,
    )
    _, evaluator_sha = evaluator_identity(repo_root)

    training = load_yaml(
        repo_root
        / "configs"
        / "protocol"
        / "d4_training_protocol.yaml"
    )

    update_budget_key = (
        "100pct_sup"
        if is_sup_reference
        else budget
    )
    update_budget = training[
        "optimizer_update_budget"
    ][update_budget_key]

    expected = {
        "method": method,
        "architecture": architecture,
        "budget": budget,
        "seed_index": seed_index,
        "training_seed": training_seed,
        "partition_seed": LOCKED_PARTITION_SEED,
        "condition_role": (
            "SUP_REFERENCE"
            if is_sup_reference
            else "MAIN_LOW_LABEL"
        ),

        "scientific_source_name":
            scientific["scientific_source_name"],
        "scientific_source_sha256":
            scientific["scientific_source_sha256"],
        "implementation_contract_sha256":
            implementation["implementation_contract_sha256"],
        "artifact_contract_sha256":
            artifact["artifact_contract_sha256"],

        **config_identity,
        **data,

        "preprocessing_config_sha256": preprocessing_sha,
        "labeled_pipeline_sha256": labeled_pipeline_sha,
        "optimizer_recipe_sha256": optimizer_sha,
        "evaluator_config_sha256": evaluator_sha,

        "optimizer": optimizer["type"],
        "learning_rate": optimizer["lr"],
        "weight_decay": optimizer["weight_decay"],
        "total_optimizer_updates": update_budget,
        "validation_interval_updates":
            training["validation_interval_optimizer_updates"],
        "effective_labeled_batch":
            training["effective_labeled_batch"],
        "amp": training["numerical"]["amp"],

        "checkpoint_selection_metric":
            "bbox_mAP_50_95_validation",
        "test_usage": "FINAL_ONLY",
        "test_access_authorized": False,
        "test_accessed": False,
        "test_access_reason": "FINAL_ONLY",
        "final_evaluation_status": "NOT_AUTHORIZED",

        "seed_protocol_sha256": seed["seed_protocol_sha256"],

        **_protocol_config_identity(
            repo_root,
            method=method,
        ),
    }

    if method == "SUP":
        expected["unlabeled_manifest_sha256"] = "NOT_APPLICABLE"
        expected["effective_unlabeled_batch"] = "NOT_APPLICABLE"
    else:
        ssl = load_yaml(
            repo_root
            / "configs"
            / "protocol"
            / "d4_ssl_protocol.yaml"
        )
        expected["effective_unlabeled_batch"] = (
            ssl["effective_batch"]["unlabeled"]
        )

        # S4.16 hidden-U firewall is part of the active SSL
        # fail-fast pre-run path. Only the stripped unlabeled COCO
        # is inspected; U_b is never reverse-mapped to master GT.
        assert_current_ssl_execution_firewall(
            repo_root,
            architecture=architecture,
            budget=budget,
        )

    return expected


def evaluate_pre_run_identity(
    repo_root: Path,
    observed: dict[str, Any],
) -> dict[str, Any]:
    """Compare one proposed launch identity against locked expectations."""
    repo_root = Path(repo_root).resolve()

    selector_fields = (
        "method",
        "architecture",
        "budget",
        "seed_index",
    )

    missing_selectors = [
        field for field in selector_fields
        if field not in observed
    ]

    if missing_selectors:
        return {
            "schema_version": "1.0",
            "stage": "S2.14",
            "guardrail_scope":
                "source_config_data_seed_pre_run_foundation",
            "launch_status": "BLOCKED",
            "training_authorized_by_s2_14": False,
            "all_checks_pass": False,
            "missing_fields": missing_selectors,
            "mismatches": [],
            "error": "Missing selector fields.",
        }

    try:
        expected = build_expected_pre_run_identity(
            repo_root,
            method=observed["method"],
            architecture=observed["architecture"],
            budget=observed["budget"],
            seed_index=observed["seed_index"],
        )
    except Exception as exc:
        return {
            "schema_version": "1.0",
            "stage": "S2.14",
            "guardrail_scope":
                "source_config_data_seed_pre_run_foundation",
            "launch_status": "BLOCKED",
            "training_authorized_by_s2_14": False,
            "all_checks_pass": False,
            "missing_fields": [],
            "mismatches": [],
            "error": f"{type(exc).__name__}: {exc}",
        }

    missing = [
        field for field in expected
        if field not in observed
    ]

    mismatches = []
    for field, expected_value in expected.items():
        if field not in observed:
            continue

        observed_value = observed[field]
        if observed_value != expected_value:
            mismatches.append(
                {
                    "field": field,
                    "expected": expected_value,
                    "observed": observed_value,
                }
            )

    all_pass = not missing and not mismatches

    return {
        "schema_version": "1.0",
        "stage": "S2.14",
        "guardrail_scope":
            "source_config_data_seed_pre_run_foundation",
        "launch_status": "PASS" if all_pass else "BLOCKED",
        "training_authorized_by_s2_14": False,
        "all_checks_pass": all_pass,
        "missing_fields": missing,
        "mismatches": mismatches,
        "error": None,
    }


def assert_pre_run_identity(
    repo_root: Path,
    observed: dict[str, Any],
) -> dict[str, Any]:
    """Fail-fast assertion suitable for a future launcher entrypoint."""
    report = evaluate_pre_run_identity(repo_root, observed)

    if not report["all_checks_pass"]:
        raise PreRunGuardrailError(
            json.dumps(
                report,
                sort_keys=True,
                ensure_ascii=False,
            )
        )

    return report