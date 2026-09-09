#!/usr/bin/env python3
"""S2.11 preflight for canonical run_manifest.json identity."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default="/workspace/ssod/project",
    )
    parser.add_argument(
        "--output",
        default=(
            "/workspace/ssod/artifacts/runs/pilot/"
            "PILOT_SUP_R50_RUNMANIFEST_001/"
            "attempt_001/run_manifest.json"
        ),
    )
    parser.add_argument(
        "--docker-evidence",
        default=(
            "/workspace/ssod/artifacts/preflight/environment/"
            "docker_image_identity.json"
        ),
    )
    return parser.parse_args()


def require_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ValueError(f"Invalid SHA-256 field {name}: {value!r}")


def main() -> int:
    args = parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_path = Path(args.output).resolve()
    docker_path = Path(args.docker_evidence).resolve()

    sys.path.insert(0, str(repo_root))

    from src.utils.run_manifest import (
        evaluator_identity,
        fixed_data_identity,
        git_head,
        governance_identity,
        labeled_pipeline_identity,
        optimizer_identity,
        preprocessing_identity,
    )

    governance = governance_identity(repo_root)
    data = fixed_data_identity(repo_root, budget="10pct")

    _, preprocessing_sha = preprocessing_identity(repo_root)
    _, labeled_pipeline_sha = labeled_pipeline_identity(repo_root)
    optimizer, optimizer_sha = optimizer_identity(
        repo_root,
        architecture="R50-FPN",
    )
    _, evaluator_sha = evaluator_identity(repo_root)

    docker = json.loads(
        docker_path.read_text(encoding="utf-8")
    )

    manifest = {
        "schema_version": governance["schema_version"],
        "run_id": "PILOT_SUP_R50_RUNMANIFEST_001",
        "attempt_id": "attempt_001",
        "run_type": "PILOT",
        "condition_role": "PILOT",
        "method": "SUP",
        "architecture": "R50-FPN",
        "budget": "10pct",
        "seed_index": 1,
        "training_seed": 204886845,
        "partition_seed": 42,

        "scientific_source_name": governance[
            "scientific_source_name"
        ],
        "scientific_source_sha256": governance[
            "scientific_source_sha256"
        ],
        "implementation_contract_sha256": governance[
            "implementation_contract_sha256"
        ],
        "artifact_contract_sha256": governance[
            "artifact_contract_sha256"
        ],

        "git_commit": git_head(repo_root),
        "docker_image": docker["expected"]["image_uuid"],
        "docker_digest": docker["expected"]["digest"],
        "entrypoint_id": "scripts/S2_11_validate_run_manifest.py",

        **data,
        "unlabeled_manifest_sha256": "NOT_APPLICABLE",

        "preprocessing_config_sha256": preprocessing_sha,
        "labeled_pipeline_sha256": labeled_pipeline_sha,
        "optimizer_recipe_sha256": optimizer_sha,
        "evaluator_config_sha256": evaluator_sha,

        "optimizer": optimizer["type"],
        "learning_rate": optimizer["lr"],
        "weight_decay": optimizer["weight_decay"],
        "total_optimizer_updates": 2064,
        "validation_interval_updates": 172,
        "effective_labeled_batch": 4,
        "effective_unlabeled_batch": "NOT_APPLICABLE",
        "amp": True,
        "gradient_clipping": False,

        "checkpoint_selection_metric":
            "bbox_mAP_50_95_validation",
        "test_usage": "FINAL_ONLY",
        "training_status": "RUNNING",
        "final_evaluation_status": "NOT_AUTHORIZED",
    }

    required = [
        "schema_version",
        "run_id",
        "attempt_id",
        "run_type",
        "condition_role",
        "method",
        "architecture",
        "budget",
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
        "entrypoint_id",
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
        "test_usage",
        "training_status",
        "final_evaluation_status",
    ]

    missing = [key for key in required if key not in manifest]

    hash_fields = [
        key for key in manifest
        if key.endswith("_sha256")
        and not (
            key == "unlabeled_manifest_sha256"
            and manifest["method"] == "SUP"
            and manifest[key] == "NOT_APPLICABLE"
        )
    ]

    checks = {
        "required_fields_present": not missing,
        "schema_version_1_1":
            manifest["schema_version"] == "1.1",
        "pilot_run_type":
            manifest["run_type"] == "PILOT",
        "pilot_condition_role":
            manifest["condition_role"] == "PILOT",
        "run_id_contract":
            manifest["run_id"]
            == "PILOT_SUP_R50_RUNMANIFEST_001",
        "attempt_id_contract":
            manifest["attempt_id"] == "attempt_001",
        "method_sup":
            manifest["method"] == "SUP",
        "sup_unlabeled_batch_na":
            manifest["effective_unlabeled_batch"]
            == "NOT_APPLICABLE",
        "sup_unlabeled_manifest_na":
            manifest["unlabeled_manifest_sha256"]
            == "NOT_APPLICABLE",
        "partition_seed_locked":
            manifest["partition_seed"] == 42,
        "training_seed_locked":
            manifest["training_seed"] == 204886845,
        "test_usage_final_only":
            manifest["test_usage"] == "FINAL_ONLY",
        "final_eval_not_authorized":
            manifest["final_evaluation_status"]
            == "NOT_AUTHORIZED",
        "docker_evidence_exists":
            docker_path.is_file(),
    }

    try:
        for key in hash_fields:
            require_sha256(key, manifest[key])
        checks["hash_fields_valid"] = True
    except ValueError:
        checks["hash_fields_valid"] = False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print("run_manifest_path =", output_path)
    print("missing_required_fields =", missing)

    for key, value in checks.items():
        print(f"{key} =", value)

    all_pass = all(checks.values())

    print("ALL_CHECKS_PASS =", all_pass)

    if all_pass:
        print("S2_11_RUN_MANIFEST_SCHEMA=PASS")
        return 0

    print("S2_11_RUN_MANIFEST_SCHEMA=FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
