#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0"
TASK = "S7.P02"
ARTIFACT_TYPE = "PILOT_END_TO_END_REPORT"
TRAINING_SEED = 204886845
SUCCESS_GIT_COMMIT = "4ed9c5202b19415a8f0c22403e4e53f5bc390d73"
SWIN_CHECKPOINT_SHA256 = (
    "9f71c168d837d1b99dd1dc29e14990a7a9e8bdc5f673d46b04fe36fe15590ad3"
)

REQUIRED_RUNTIME_ASSERTIONS = (
    "teacher_initialized_from_student_at_t0",
    "ssl_labeled_and_unlabeled_paths_exercised",
    "pseudo_label_generation_path_exercised",
    "empty_pseudo_safe_path_supported",
    "optimizer_actual_update_count_equals_1",
    "ema_update_count_matches_optimizer_updates",
    "validation_uses_fixed_validation_only",
    "best_last_checkpoint_path_exercised",
    "test_access_count_equals_0",
    "pilot_runtime_isolated_from_official_runtime",
)

CONDITIONS = (
    {
        "run_id": "PILOT_SSL_R50_E2E_001",
        "architecture": "R50-FPN",
        "successful_attempt": "attempt_003",
        "retry_of": "attempt_002",
        "failed_attempts": ("attempt_001", "attempt_002"),
    },
    {
        "run_id": "PILOT_SSL_SWIN_E2E_001",
        "architecture": "Swin-T-FPN",
        "successful_attempt": "attempt_002",
        "retry_of": "attempt_001",
        "failed_attempts": ("attempt_001",),
    },
)


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return data


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/workspace/ssod/project")
    parser.add_argument(
        "--output",
        default=(
            "/workspace/ssod/project/artifacts/preflight/pilot/"
            "pilot_end_to_end_report.json"
        ),
    )
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    output = Path(args.output).resolve()
    pilot_root = repo / "artifacts/runs/pilot"

    checks: dict[str, bool] = {}
    condition_reports: list[dict[str, Any]] = []

    manifest_path = repo / "artifacts/preflight/pilot/pilot_end_to_end_manifest.json"
    manifest = load_json(manifest_path)
    checks["s7_p01_manifest_locked"] = (
        manifest.get("task") == "S7.P01"
        and manifest.get("status") == "LOCKED_PRE_EXECUTION"
        and manifest.get("scientific_protocol_change") is False
    )

    expected_run_ids = {
        c.get("pilot_run_id") for c in manifest.get("conditions", [])
    }
    checks["locked_condition_set_exact"] = expected_run_ids == {
        "PILOT_SSL_R50_E2E_001",
        "PILOT_SSL_SWIN_E2E_001",
    }

    for spec in CONDITIONS:
        run_id = spec["run_id"]
        prefix = "r50" if spec["architecture"] == "R50-FPN" else "swin"
        run_root = pilot_root / run_id
        success = run_root / spec["successful_attempt"]

        run_manifest = load_json(success / "run_manifest.json")
        retry = load_json(success / "retry_deviation.json")
        observation = load_json(success / "pilot_runtime_observation.json")
        summary = load_json(success / "training_summary.json")
        completion = load_json(success / "pilot_execution_completion.json")
        best = load_json(
            success / "metrics/validation/validation_best_teacher.json"
        )
        checkpoint_index = load_json(
            success / "checkpoints/checkpoint_index.json"
        )

        runtime_assertions = completion.get("runtime_assertions", {})
        observation_assertions = observation.get("assertions", {})

        checks[f"{prefix}_run_identity"] = (
            run_manifest.get("run_id") == run_id
            and run_manifest.get("architecture") == spec["architecture"]
            and run_manifest.get("attempt_id") == spec["successful_attempt"]
            and run_manifest.get("retry_of") == spec["retry_of"]
        )
        checks[f"{prefix}_success_commit"] = (
            run_manifest.get("git_commit") == SUCCESS_GIT_COMMIT
        )
        checks[f"{prefix}_same_seed"] = (
            run_manifest.get("training_seed") == TRAINING_SEED
            and retry.get("training_seed") == TRAINING_SEED
        )
        checks[f"{prefix}_closed_verified"] = (
            run_manifest.get("training_status") == "CLOSED_VERIFIED"
            and completion.get("status") == "PASS"
            and completion.get("run_manifest_training_status")
            == "CLOSED_VERIFIED"
            and completion.get("runner_train_returned") is True
        )
        checks[f"{prefix}_successful_attempt_not_retry_failure"] = (
            retry.get("technical_failure") is False
            and retry.get("retry_required") is False
            and retry.get("scientific_protocol_changed") is False
        )
        checks[f"{prefix}_optimizer_update_exactly_one"] = (
            run_manifest.get("total_optimizer_updates") == 1
            and summary.get("expected_optimizer_updates") == 1
            and summary.get("observed_optimizer_updates") == 1
            and summary.get("expected_update_budget_reached") is True
        )
        checks[f"{prefix}_ema_matches_update"] = (
            summary.get("ema_update_count") == 1
            and summary.get("ema_updates_match_optimizer_updates") is True
        )
        checks[f"{prefix}_labeled_unlabeled_exposure"] = (
            summary.get("labeled_exposure_matches_update_count") is True
            and summary.get("unlabeled_exposure_matches_update_count") is True
            and observation_assertions.get(
                "ssl_labeled_and_unlabeled_paths_exercised"
            )
            is True
        )
        checks[f"{prefix}_teacher_initialized_t0"] = (
            observation.get("teacher_initialized_from_student_at_t0") is True
            and observation_assertions.get(
                "teacher_initialized_from_student_at_t0"
            )
            is True
        )
        checks[f"{prefix}_pseudo_path_exercised"] = (
            observation_assertions.get(
                "pseudo_label_generation_path_exercised"
            )
            is True
        )
        checks[f"{prefix}_empty_pseudo_safe"] = (
            observation.get("empty_pseudo_safe_model_active") is True
            and observation.get("empty_pseudo_zero_loss_verified") is True
            and observation_assertions.get(
                "empty_pseudo_safe_path_supported"
            )
            is True
        )
        checks[f"{prefix}_validation_once"] = (
            run_manifest.get("validation_interval_updates") == 1
            and summary.get("validation_events_observed") == 1
            and runtime_assertions.get("validation_uses_fixed_validation_only")
            is True
        )
        checks[f"{prefix}_best_last"] = (
            runtime_assertions.get("best_last_checkpoint_path_exercised")
            is True
            and (success / "checkpoints/best_ema_teacher.pth").is_file()
            and (success / "checkpoints/last_ema_teacher.pth").is_file()
            and best.get("selected_update") == 1
            and best.get("selection_split") == "validation"
            and best.get("selection_model") == "EMA_TEACHER"
            and len(checkpoint_index.get("entries", [])) >= 2
        )
        checks[f"{prefix}_test_firewall"] = (
            run_manifest.get("test_access_authorized") is False
            and run_manifest.get("test_accessed") is False
            and run_manifest.get("final_evaluation_status")
            == "NOT_AUTHORIZED"
            and completion.get("test_access_authorized") is False
            and completion.get("test_accessed") is False
            and completion.get("test_access_count") == 0
            and completion.get("final_evaluation_status")
            == "NOT_AUTHORIZED"
        )
        checks[f"{prefix}_official_isolation"] = (
            completion.get("pilot_runtime_isolated_from_official_runtime")
            is True
            and completion.get("official_runtime_before")
            == completion.get("official_runtime_after")
        )
        checks[f"{prefix}_all_runtime_assertions"] = all(
            runtime_assertions.get(name) is True
            for name in REQUIRED_RUNTIME_ASSERTIONS
        )

        failed_attempt_reports = []
        for failed_id in spec["failed_attempts"]:
            failed_dir = run_root / failed_id
            failed_manifest = load_json(failed_dir / "run_manifest.json")
            failed_retry = load_json(failed_dir / "retry_deviation.json")
            failed_attempt_reports.append(
                {
                    "attempt_id": failed_id,
                    "training_seed": failed_manifest.get("training_seed"),
                    "training_status": failed_manifest.get("training_status"),
                    "technical_failure": failed_retry.get("technical_failure"),
                    "retry_required": failed_retry.get("retry_required"),
                    "retry_reason": failed_retry.get("retry_reason"),
                    "failure_reason": failed_retry.get("failure_reason"),
                    "same_seed_confirmed": failed_retry.get(
                        "same_seed_confirmed"
                    ),
                    "scientific_protocol_changed": failed_retry.get(
                        "scientific_protocol_changed"
                    ),
                }
            )
            checks[f"{prefix}_{failed_id}_technical_failure"] = (
                failed_manifest.get("run_id") == run_id
                and failed_manifest.get("attempt_id") == failed_id
                and failed_manifest.get("training_seed") == TRAINING_SEED
                and failed_manifest.get("training_status")
                != "CLOSED_VERIFIED"
                and failed_retry.get("technical_failure") is True
                and failed_retry.get("retry_required") is True
                and failed_retry.get("retry_reason")
                == "TECHNICAL_FAILURE_ONLY"
                and failed_retry.get("same_seed_confirmed") is True
                and failed_retry.get("scientific_protocol_changed") is False
            )

        condition_reports.append(
            {
                "run_id": run_id,
                "architecture": spec["architecture"],
                "successful_attempt": spec["successful_attempt"],
                "retry_of": spec["retry_of"],
                "training_seed": TRAINING_SEED,
                "training_status": run_manifest.get("training_status"),
                "git_commit": run_manifest.get("git_commit"),
                "runtime_path": str(success.relative_to(repo)),
                "run_manifest_sha256": sha256(success / "run_manifest.json"),
                "training_summary_sha256": sha256(
                    success / "training_summary.json"
                ),
                "pilot_runtime_observation_sha256": sha256(
                    success / "pilot_runtime_observation.json"
                ),
                "pilot_execution_completion_sha256": sha256(
                    success / "pilot_execution_completion.json"
                ),
                "best_checkpoint_sha256": sha256(
                    success / "checkpoints/best_ema_teacher.pth"
                ),
                "last_checkpoint_sha256": sha256(
                    success / "checkpoints/last_ema_teacher.pth"
                ),
                "runtime_assertions": {
                    name: runtime_assertions.get(name)
                    for name in REQUIRED_RUNTIME_ASSERTIONS
                },
                "failed_attempts": failed_attempt_reports,
            }
        )

    swin_checkpoint = (
        Path("/workspace/ssod/cache/pretrained")
        / "swin_tiny_patch4_window7_224.pth"
    )
    checks["swin_pretrained_identity"] = (
        swin_checkpoint.is_file()
        and sha256(swin_checkpoint) == SWIN_CHECKPOINT_SHA256
    )

    failed = [name for name, passed in checks.items() if not passed]
    all_pass = not failed

    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "task": TASK,
        "status": "PASS" if all_pass else "FAIL",
        "scientific_protocol_change": False,
        "official_training_authorized": False,
        "final_test_authorized": False,
        "test_accessed": False,
        "performance_driven_tuning_used": False,
        "technical_training_seed": TRAINING_SEED,
        "successful_git_commit": SUCCESS_GIT_COMMIT,
        "s7_p01_manifest": {
            "path": str(manifest_path.relative_to(repo)),
            "sha256": sha256(manifest_path),
        },
        "conditions": condition_reports,
        "checks": checks,
        "check_count": len(checks),
        "failed_checks": failed,
        "all_checks_pass": all_pass,
        "overall_pass": all_pass,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(output)

    print("=== S7.P02 TINY SSL PILOT VALIDATOR ===")
    print("REPORT=" + str(output))
    print("CONDITION_COUNT=" + str(len(condition_reports)))
    print("TOTAL_CHECKS=" + str(len(checks)))
    print("FAILED_CHECKS=" + str(len(failed)))
    if failed:
        print("FAILED_NAMES=" + ",".join(failed))
        print("S7_P02_PILOT_END_TO_END=FAIL")
        return 1
    print("S7_P02_PILOT_END_TO_END=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
