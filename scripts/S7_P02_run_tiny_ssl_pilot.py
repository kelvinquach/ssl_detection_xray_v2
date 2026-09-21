#!/usr/bin/env python3
"""S7.P02 launcher for the locked NON-OFFICIAL tiny SSL end-to-end pilots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from mmengine.config import Config
from mmengine.runner import Runner


TRAINING_SEED = 204886845
PARTITION_SEED = 42
ATTEMPT_ID = "attempt_001"

MANIFEST_REL = Path(
    "artifacts/preflight/pilot/pilot_end_to_end_manifest.json"
)
DOCKER_EVIDENCE = Path(
    "/workspace/ssod/artifacts/preflight/environment/docker_image_identity.json"
)
S5_FIREWALL_REL = Path(
    "artifacts/preflight/firewall/test_firewall_preflight.json"
)

ARCH_MAP = {
    "R50": "R50-FPN",
    "SWIN": "Swin-T-FPN",
}

EXPECTED_SSL = {
    "ema_momentum": 0.001,
    "ema_skip_buffers": True,
    "ema_update_basis": "ACTUAL_OPTIMIZER_UPDATE",
    "labeled_unlabeled_ratio": "1:1",
    "unsupervised_loss_weight": 4.0,
    "weak_augmentation_id":
        "7f5343daba335b50d0168b9f52ba771b0eccf4eea3360f9b75705ba811ebabe0",
    "strong_augmentation_id":
        "15fa1478c634e352782f0f3f3212821b6f82a7e63962f509fdef7b366492a796",
    "detector_score_floor": 0.05,
    "initial_pseudo_score": 0.50,
    "rpn_pseudo_threshold": 0.90,
    "rcnn_classification_threshold": 0.90,
    "rpn_nms_iou": 0.70,
    "rpn_max": 1000,
    "rcnn_nms_iou": 0.50,
    "rcnn_max": 100,
    "jitter_times": 10,
    "jitter_scale": 0.06,
    "reg_uncertainty_threshold": 0.02,
    "min_pseudo_bbox_wh": [0.01, 0.01],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default="/workspace/ssod/project",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        choices=[
            "PILOT_SSL_R50_E2E_001",
            "PILOT_SSL_SWIN_E2E_001",
        ],
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return data


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def tree_state(root: Path) -> dict[str, Any]:
    from src.utils.run_manifest import file_sha256, semantic_sha256

    if not root.exists():
        return {
            "exists": False,
            "file_count": 0,
            "semantic_sha256": semantic_sha256([]),
            "files": [],
        }

    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": file_sha256(path),
            }
        )

    return {
        "exists": True,
        "file_count": len(rows),
        "semantic_sha256": semantic_sha256(rows),
        "files": rows,
    }


def select_condition(
    manifest: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    conditions = manifest.get("conditions")
    if not isinstance(conditions, list):
        raise RuntimeError("S7.P01 manifest conditions must be a list")

    matches = [
        item for item in conditions
        if isinstance(item, dict)
        and item.get("pilot_run_id") == run_id
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one locked condition for {run_id}"
        )
    return matches[0]


def verify_s7p01_lock(
    repo_root: Path,
    manifest: dict[str, Any],
    condition: dict[str, Any],
) -> None:
    from src.utils.run_manifest import file_sha256

    if manifest.get("schema_version") != "1.1":
        raise RuntimeError("Unexpected S7.P01 manifest schema")
    if manifest.get("task") != "S7.P01":
        raise RuntimeError("Unexpected S7.P01 task identity")
    if manifest.get("status") != "LOCKED_PRE_EXECUTION":
        raise RuntimeError("S7.P01 manifest is not locked pre-execution")
    if manifest.get("execution_status") != "NOT_STARTED":
        raise RuntimeError("S7.P01 manifest execution status is not NOT_STARTED")
    if manifest.get("scientific_protocol_change") is not False:
        raise RuntimeError("Scientific protocol change must remain false")

    if condition.get("method") != "SSL":
        raise RuntimeError("Pilot condition must be SSL")
    if condition.get("label_budget") != "1pct":
        raise RuntimeError("Pilot label budget must be 1pct")
    if condition.get("resolved_model_type") != "EmptyPseudoSafeSoftTeacher":
        raise RuntimeError("Unexpected resolved model type")
    if condition.get("expected_optimizer_updates") != 1:
        raise RuntimeError("Pilot must target exactly one optimizer update")
    if condition.get("execution_status") != "NOT_STARTED":
        raise RuntimeError("Pilot condition was already marked started")

    expected_runtime = (
        f"artifacts/runs/pilot/{condition['pilot_run_id']}/{ATTEMPT_ID}"
    )
    if condition.get("runtime_path") != expected_runtime:
        raise RuntimeError("Locked pilot runtime path mismatch")

    config_path = repo_root / str(condition["pilot_config"])
    parent_path = repo_root / str(
        condition["parent_official_family_config"]
    )
    if file_sha256(config_path) != condition["pilot_config_sha256"]:
        raise RuntimeError("Locked pilot config SHA-256 mismatch")
    if file_sha256(parent_path) != condition[
        "parent_official_family_config_sha256"
    ]:
        raise RuntimeError("Parent official-family config SHA-256 mismatch")


def build_run_manifest(
    *,
    repo_root: Path,
    condition: dict[str, Any],
    architecture: str,
) -> dict[str, Any]:
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
    data = fixed_data_identity(repo_root, budget="1pct")
    _, preprocessing_sha = preprocessing_identity(repo_root)
    _, labeled_pipeline_sha = labeled_pipeline_identity(repo_root)
    optimizer, optimizer_sha = optimizer_identity(
        repo_root,
        architecture=architecture,
    )
    _, evaluator_sha = evaluator_identity(repo_root)

    docker = load_json(DOCKER_EVIDENCE)

    payload: dict[str, Any] = {
        "schema_version": governance["schema_version"],
        "run_id": condition["pilot_run_id"],
        "attempt_id": ATTEMPT_ID,
        "run_type": "PILOT",
        "condition_role": "PILOT",
        "method": "SSL",
        "architecture": architecture,
        "budget": "1pct",
        "seed_index": 1,
        "training_seed": TRAINING_SEED,
        "partition_seed": PARTITION_SEED,
        "scientific_source_name": governance["scientific_source_name"],
        "scientific_source_sha256":
            governance["scientific_source_sha256"],
        "implementation_contract_sha256":
            governance["implementation_contract_sha256"],
        "artifact_contract_sha256":
            governance["artifact_contract_sha256"],
        "git_commit": git_head(repo_root),
        "docker_image": docker["expected"]["image_uuid"],
        "docker_digest": docker["expected"]["digest"],
        "entrypoint_id": "scripts/S7_P02_run_tiny_ssl_pilot.py",
        **data,
        "preprocessing_config_sha256": preprocessing_sha,
        "labeled_pipeline_sha256": labeled_pipeline_sha,
        "optimizer_recipe_sha256": optimizer_sha,
        "evaluator_config_sha256": evaluator_sha,
        "optimizer": optimizer["type"],
        "learning_rate": optimizer["lr"],
        "weight_decay": optimizer["weight_decay"],
        "total_optimizer_updates": 1,
        "validation_interval_updates": 1,
        "effective_labeled_batch": 4,
        "effective_unlabeled_batch": 4,
        "amp": True,
        "gradient_clipping": False,
        "checkpoint_selection_metric": "bbox_mAP_50_95_validation",
        "test_usage": "FINAL_ONLY",
        "training_status": "RUNNING",
        "final_evaluation_status": "NOT_AUTHORIZED",
        "test_access_authorized": False,
        "test_accessed": False,
        "test_access_reason": "FINAL_ONLY",
        "pilot_config": condition["pilot_config"],
        "pilot_config_sha256": condition["pilot_config_sha256"],
        "parent_official_family_config":
            condition["parent_official_family_config"],
        "parent_official_family_config_sha256":
            condition["parent_official_family_config_sha256"],
        "pilot_operational_shortening": True,
        "official_scientific_update_budget_changed": False,
    }
    payload.update(EXPECTED_SSL)
    return payload


def configure_runner(
    *,
    repo_root: Path,
    condition: dict[str, Any],
    work_dir: Path,
) -> Config:
    cfg = Config.fromfile(str(repo_root / condition["pilot_config"]))

    if cfg.get("test_dataloader", None) is not None:
        raise RuntimeError("Pilot config exposes a test_dataloader")
    if cfg.get("test_evaluator", None) is not None:
        raise RuntimeError("Pilot config exposes a test_evaluator")

    cfg.work_dir = str(work_dir)
    cfg.randomness = dict(
        seed=TRAINING_SEED,
        diff_rank_seed=False,
        deterministic=True,
    )
    cfg.resume = False
    cfg.load_from = None
    cfg.launcher = "none"

    custom_imports = cfg.get("custom_imports", {})
    imports = list(custom_imports.get("imports", []))
    observer_module = "src.hooks.s7_p02_pilot_observation_hook"
    if observer_module not in imports:
        imports.append(observer_module)
    cfg.custom_imports = dict(
        imports=imports,
        allow_failed_imports=False,
    )

    hooks = list(cfg.get("custom_hooks", []))
    if not any(
        isinstance(hook, dict)
        and hook.get("type") == "S7P02PilotObservationHook"
        for hook in hooks
    ):
        hooks.append(dict(type="S7P02PilotObservationHook"))
    cfg.custom_hooks = hooks

    return cfg


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo_root))

    from src.utils.run_manifest import file_sha256, git_head
    from src.utils.seed import set_global_seed
    from src.utils.unlabeled_firewall import (
        assert_current_ssl_execution_firewall,
    )

    manifest_path = repo_root / MANIFEST_REL
    manifest = load_json(manifest_path)
    condition = select_condition(manifest, args.run_id)
    verify_s7p01_lock(repo_root, manifest, condition)

    architecture = ARCH_MAP.get(str(condition["architecture"]))
    if architecture is None:
        raise RuntimeError("Unsupported locked pilot architecture")

    work_dir = (repo_root / str(condition["runtime_path"])).resolve()
    expected_parent = (
        repo_root / "artifacts" / "runs" / "pilot" / args.run_id
    ).resolve()
    if work_dir.parent != expected_parent:
        raise RuntimeError("Pilot work_dir escaped locked pilot namespace")
    if work_dir.exists():
        raise RuntimeError(
            f"Refusing to overwrite existing pilot attempt: {work_dir}"
        )

    official_root = (
        repo_root / str(manifest["isolation"]["official_runtime_root"])
    ).resolve()
    if official_root == work_dir or official_root in work_dir.parents:
        raise RuntimeError("Pilot work_dir overlaps official runtime root")

    firewall_evidence = load_json(repo_root / S5_FIREWALL_REL)
    firewall_state = firewall_evidence.get("firewall_state", {})
    if firewall_evidence.get("failed_count") != 0:
        raise RuntimeError("S5.09 firewall evidence has failed checks")
    if firewall_state.get("test_access_authorized") is not False:
        raise RuntimeError("Test access is not closed")
    if firewall_state.get("test_accessed") is not False:
        raise RuntimeError("Existing firewall state records test access")
    if firewall_state.get("final_evaluation_status") != "NOT_AUTHORIZED":
        raise RuntimeError("Final evaluation unexpectedly authorized")

    official_before = tree_state(official_root)

    hidden_u = assert_current_ssl_execution_firewall(
        repo_root,
        architecture=architecture,
        budget="1pct",
    )
    if hidden_u.get("all_checks") is not True:
        raise RuntimeError("Hidden-U execution firewall did not PASS")

    seed_observation = set_global_seed(
        TRAINING_SEED,
        deterministic=True,
    )

    cfg = configure_runner(
        repo_root=repo_root,
        condition=condition,
        work_dir=work_dir,
    )

    if cfg.train_cfg.get("max_iters") != 1:
        raise RuntimeError("Resolved pilot max_iters must equal 1")
    if cfg.train_cfg.get("val_interval") != 1:
        raise RuntimeError("Resolved pilot val_interval must equal 1")
    if cfg.model.get("type") != "EmptyPseudoSafeSoftTeacher":
        raise RuntimeError("Resolved pilot model type mismatch")

    work_dir.mkdir(parents=True, exist_ok=False)

    run_manifest = build_run_manifest(
        repo_root=repo_root,
        condition=condition,
        architecture=architecture,
    )
    write_json(work_dir / "run_manifest.json", run_manifest)

    launch_context = {
        "schema_version": "1.0",
        "task": "S7.P02",
        "run_id": args.run_id,
        "attempt_id": ATTEMPT_ID,
        "status": "PRE_RUN_PASS",
        "git_commit": git_head(repo_root),
        "s7_p01_manifest": MANIFEST_REL.as_posix(),
        "s7_p01_manifest_sha256": file_sha256(manifest_path),
        "pilot_config": condition["pilot_config"],
        "pilot_config_sha256": condition["pilot_config_sha256"],
        "technical_training_seed": TRAINING_SEED,
        "seed_observation": seed_observation,
        "hidden_u_firewall": hidden_u,
        "test_firewall": {
            "source": S5_FIREWALL_REL.as_posix(),
            "source_sha256":
                file_sha256(repo_root / S5_FIREWALL_REL),
            "test_access_authorized": False,
            "test_accessed": False,
            "test_access_count": 0,
            "test_access_reason": "FINAL_ONLY",
            "final_evaluation_status": "NOT_AUTHORIZED",
            "test_dataloader_present": False,
            "test_evaluator_present": False,
        },
        "official_runtime_before": official_before,
    }
    write_json(work_dir / "pilot_launch_context.json", launch_context)

    runner = Runner.from_cfg(cfg)
    runner.train()

    official_after = tree_state(official_root)
    isolation_pass = (
        official_before["semantic_sha256"]
        == official_after["semantic_sha256"]
        and official_before["file_count"]
        == official_after["file_count"]
    )

    observation_path = work_dir / "pilot_runtime_observation.json"
    summary_path = work_dir / "training_summary.json"
    checkpoint_index_path = (
        work_dir / "checkpoints" / "checkpoint_index.json"
    )
    validation_best_path = (
        work_dir
        / "metrics"
        / "validation"
        / "validation_best_teacher.json"
    )

    observation = load_json(observation_path)
    summary = load_json(summary_path)
    checkpoint_index = load_json(checkpoint_index_path)
    validation_best = load_json(validation_best_path)

    observation_assertions = observation.get("assertions", {})
    entries = checkpoint_index.get("entries", [])
    if checkpoint_index.get("checkpoint_policy") != "LOCKED":
        raise RuntimeError("Checkpoint index policy is not LOCKED")
    if not isinstance(entries, list):
        raise RuntimeError("Checkpoint index entries must be a list")

    best_entries = [
        item for item in entries
        if isinstance(item, dict) and item.get("role") == "BEST"
    ]
    last_entries = [
        item for item in entries
        if isinstance(item, dict) and item.get("role") == "LAST"
    ]
    best_entry = best_entries[0] if len(best_entries) == 1 else {}
    last_entry = last_entries[0] if len(last_entries) == 1 else {}

    best_path = (
        work_dir / "checkpoints" / str(best_entry.get("path", ""))
    )
    last_path = (
        work_dir / "checkpoints" / str(last_entry.get("path", ""))
    )

    best_checkpoint_ok = bool(
        len(best_entries) == 1
        and best_entry.get("optimizer_update") == 1
        and best_entry.get("path") == "best_ema_teacher.pth"
        and best_entry.get("retained") is True
        and best_path.is_file()
        and file_sha256(best_path) == best_entry.get("sha256")
    )
    last_checkpoint_ok = bool(
        len(last_entries) == 1
        and last_entry.get("optimizer_update") == 1
        and last_entry.get("path") == "last_ema_teacher.pth"
        and last_entry.get("retained") is True
        and last_path.is_file()
        and file_sha256(last_path) == last_entry.get("sha256")
    )

    summary_ok = bool(
        summary.get("expected_optimizer_updates") == 1
        and summary.get("observed_optimizer_updates") == 1
        and summary.get("effective_labeled_batch") == 4
        and summary.get("effective_unlabeled_batch") == 4
        and summary.get("validation_events_observed") == 1
        and summary.get("ema_update_count") == 1
        and summary.get("labeled_exposure_matches_update_count") is True
        and summary.get("unlabeled_exposure_matches_update_count") is True
        and summary.get("ema_updates_match_optimizer_updates") is True
        and summary.get("expected_update_budget_reached") is True
    )

    expected_val_ann = "/workspace/ssod/data/coco/instances_val.json"
    val_dataloader = cfg.get("val_dataloader", {})
    val_dataset = val_dataloader.get("dataset", {})
    val_evaluator = cfg.get("val_evaluator", {})

    validation_ok = bool(
        validation_best.get("selection_split") == "validation"
        and validation_best.get("selected_update") == 1
        and validation_best.get("checkpoint_path")
        == "checkpoints/best_ema_teacher.pth"
        and validation_best.get("checkpoint_sha256")
        == best_entry.get("sha256")
        and val_dataset.get("ann_file") == expected_val_ann
        and val_evaluator.get("ann_file") == expected_val_ann
    )


    runtime_assertions = {
        "teacher_initialized_from_student_at_t0": (
            observation_assertions.get(
                "teacher_initialized_from_student_at_t0"
            ) is True
        ),
        "ssl_labeled_and_unlabeled_paths_exercised": (
            observation_assertions.get(
                "ssl_labeled_and_unlabeled_paths_exercised"
            ) is True
        ),
        "pseudo_label_generation_path_exercised": (
            observation_assertions.get(
                "pseudo_label_generation_path_exercised"
            ) is True
        ),
        "empty_pseudo_safe_path_supported": (
            observation_assertions.get(
                "empty_pseudo_safe_path_supported"
            ) is True
        ),
        "optimizer_actual_update_count_equals_1": bool(
            summary_ok
            and summary.get("observed_optimizer_updates") == 1
        ),
        "ema_update_count_matches_optimizer_updates": bool(
            summary_ok
            and summary.get("ema_update_count") == 1
            and summary.get("ema_updates_match_optimizer_updates") is True
        ),
        "validation_uses_fixed_validation_only": bool(
            summary.get("validation_events_observed") == 1
            and validation_ok
            and cfg.get("test_dataloader", None) is None
            and cfg.get("test_evaluator", None) is None
        ),
        "best_last_checkpoint_path_exercised": bool(
            best_checkpoint_ok and last_checkpoint_ok
        ),
        "test_access_count_equals_0": bool(
            firewall_state.get("test_access_authorized") is False
            and firewall_state.get("test_accessed") is False
            and cfg.get("test_dataloader", None) is None
            and cfg.get("test_evaluator", None) is None
        ),
        "pilot_runtime_isolated_from_official_runtime": isolation_pass,
    }

    all_runtime_assertions_pass = all(runtime_assertions.values())

    if all_runtime_assertions_pass:
        run_manifest["training_status"] = "CLOSED_VERIFIED"
        write_json(work_dir / "run_manifest.json", run_manifest)

    completion = {
        "schema_version": "1.0",
        "task": "S7.P02",
        "run_id": args.run_id,
        "attempt_id": ATTEMPT_ID,
        "status": "PASS" if all_runtime_assertions_pass else "FAIL",
        "runner_train_returned": True,
        "pilot_runtime_path": str(work_dir),
        "official_runtime_root": str(official_root),
        "official_runtime_before": official_before,
        "official_runtime_after": official_after,
        "runtime_assertions": runtime_assertions,
        "training_summary": summary,
        "pilot_runtime_observation":
            observation_path.relative_to(work_dir).as_posix(),
        "checkpoint_index":
            checkpoint_index_path.relative_to(work_dir).as_posix(),
        "validation_best_teacher":
            validation_best_path.relative_to(work_dir).as_posix(),
        "best_checkpoint_sha256":
            best_entry.get("sha256"),
        "last_checkpoint_sha256":
            last_entry.get("sha256"),
        "pilot_runtime_isolated_from_official_runtime": isolation_pass,
        "test_access_authorized": False,
        "test_accessed": False,
        "test_access_count": 0,
        "final_evaluation_status": "NOT_AUTHORIZED",
        "run_manifest_training_status":
            run_manifest.get("training_status"),
    }
    write_json(work_dir / "pilot_execution_completion.json", completion)

    if not all_runtime_assertions_pass:
        failed = [
            key for key, passed in runtime_assertions.items()
            if not passed
        ]
        raise RuntimeError(
            "S7.P02 runtime assertions failed: "
            + ", ".join(failed)
        )

    print(f"S7_P02_RUN_ID={args.run_id}")
    print(f"S7_P02_WORK_DIR={work_dir}")
    print("S7_P02_RUNNER_TRAIN_RETURNED=TRUE")
    print("S7_P02_RUNTIME_ASSERTIONS=PASS")
    print("S7_P02_PILOT_OFFICIAL_ISOLATION=PASS")
    print("S7_P02_TEST_ACCESS_COUNT=0")
    print("S7_P02_TRAINING_STATUS=CLOSED_VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
