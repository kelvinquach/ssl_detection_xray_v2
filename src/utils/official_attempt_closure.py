"""S9 official per-attempt training closure and artifact manifest."""

from __future__ import annotations

import csv
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import yaml

from src.utils.run_manifest import file_sha256


SCHEMA_VERSION = "1.1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

COMMON_REQUIRED_PATHS = (
    "pre_run_assertions.json",
    "run_manifest.json",
    "resolved_config.yaml",
    "config.sha256",
    "data_provenance.json",
    "seed_runtime.json",
    "environment_reference.json",
    "train.jsonl",
    "runtime_events.jsonl",
    "metrics/validation/validation_history.csv",
    "metrics/validation/classwise_validation_ap.csv",
    "metrics/validation/training_summary.json",
    "checkpoints/checkpoint_index.json",
    "checkpoints/latest_resume.pth",
    "retry_deviation.json",
)

SUP_REQUIRED_PATHS = (
    "metrics/validation/validation_best.json",
    "checkpoints/best_model.pth",
    "checkpoints/last_model.pth",
)

SSL_REQUIRED_PATHS = (
    "metrics/validation/validation_best_teacher.json",
    "checkpoints/best_ema_teacher.pth",
    "checkpoints/last_ema_teacher.pth",
    "pseudo/qpseudo_validation.json",
    "pseudo/val_last_teacher_predictions.json",
    "pseudo/qpseudo_classwise.csv",
)

ROLE_BY_PATH = {
    "pre_run_assertions.json": "PRE_RUN_ASSERTIONS",
    "run_manifest.json": "RUN_MANIFEST",
    "resolved_config.yaml": "RESOLVED_CONFIG",
    "config.sha256": "RESOLVED_CONFIG_SHA256",
    "data_provenance.json": "DATA_PROVENANCE",
    "seed_runtime.json": "SEED_RUNTIME",
    "environment_reference.json": "ENVIRONMENT_REFERENCE",
    "train.jsonl": "TRAINING_UPDATE_LOG",
    "runtime_events.jsonl": "RUNTIME_EVENTS",
    "metrics/validation/validation_history.csv": "VALIDATION_HISTORY",
    "metrics/validation/validation_best.json": "VALIDATION_BEST",
    "metrics/validation/validation_best_teacher.json": "VALIDATION_BEST_TEACHER",
    "metrics/validation/classwise_validation_ap.csv": "CLASSWISE_VALIDATION_AP",
    "metrics/validation/training_summary.json": "TRAINING_SUMMARY",
    "checkpoints/checkpoint_index.json": "CHECKPOINT_INDEX",
    "checkpoints/best_model.pth": "BEST_CHECKPOINT",
    "checkpoints/last_model.pth": "LAST_CHECKPOINT",
    "checkpoints/best_ema_teacher.pth": "BEST_EMA_TEACHER_CHECKPOINT",
    "checkpoints/last_ema_teacher.pth": "LAST_EMA_TEACHER_CHECKPOINT",
    "checkpoints/latest_resume.pth": "LATEST_RESUME_CHECKPOINT",
    "pseudo/qpseudo_validation.json": "QPSEUDO_VALIDATION",
    "pseudo/val_last_teacher_predictions.json": "QPSEUDO_PREDICTIONS",
    "pseudo/qpseudo_classwise.csv": "QPSEUDO_CLASSWISE",
    "retry_deviation.json": "RETRY_DEVIATION",
    "training_completion_report.json": "TRAINING_COMPLETION_REPORT",
}


class OfficialAttemptClosureError(RuntimeError):
    """Raised when an official attempt cannot be closed safely."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise OfficialAttemptClosureError(
            f"Cannot parse JSON artifact {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise OfficialAttemptClosureError(
            f"Expected JSON object: {path}"
        )
    return payload


def _write_json_atomic(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        tmp = Path(tmp_name)
        if tmp.exists():
            tmp.unlink()


def _require_file(
    attempt_dir: Path,
    relative_path: str,
) -> Path:
    path = attempt_dir / relative_path
    if not path.is_file():
        raise OfficialAttemptClosureError(
            f"Mandatory artifact missing: {relative_path}"
        )
    if path.stat().st_size <= 0:
        raise OfficialAttemptClosureError(
            f"Mandatory artifact is empty: {relative_path}"
        )
    return path


def _require_keys(
    payload: dict[str, Any],
    required: tuple[str, ...],
    artifact: str,
) -> None:
    missing = [key for key in required if key not in payload]
    if missing:
        raise OfficialAttemptClosureError(
            f"{artifact} missing required fields: {missing}"
        )


def _read_csv(
    path: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames:
        raise OfficialAttemptClosureError(
            f"CSV has no header: {path}"
        )
    return fieldnames, rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except Exception as exc:
                raise OfficialAttemptClosureError(
                    f"Invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise OfficialAttemptClosureError(
                    f"Expected JSON object at {path}:{line_number}"
                )
            records.append(value)
    if not records:
        raise OfficialAttemptClosureError(
            f"JSONL contains no records: {path}"
        )
    return records


def _mandatory_paths(method: str) -> tuple[str, ...]:
    if method == "SUP":
        return COMMON_REQUIRED_PATHS + SUP_REQUIRED_PATHS
    if method == "SSL":
        return COMMON_REQUIRED_PATHS + SSL_REQUIRED_PATHS
    raise OfficialAttemptClosureError(
        f"Unsupported official method: {method!r}"
    )


def _validate_pre_run(
    path: Path,
    *,
    run_id: str,
    attempt_id: str,
    expected: dict[str, Any],
) -> None:
    payload = _load_json(path)
    _require_keys(
        payload,
        (
            "schema_version",
            "run_id",
            "attempt_id",
            "launch_status",
            "all_checks_pass",
            "expected_identity",
            "observed_identity",
            "final_test_accessed",
        ),
        "pre_run_assertions.json",
    )
    if payload["run_id"] != run_id:
        raise OfficialAttemptClosureError(
            "pre_run_assertions run_id mismatch."
        )
    if payload["attempt_id"] != attempt_id:
        raise OfficialAttemptClosureError(
            "pre_run_assertions attempt_id mismatch."
        )
    if payload["launch_status"] != "PASS":
        raise OfficialAttemptClosureError(
            "pre_run_assertions launch_status is not PASS."
        )
    if payload["all_checks_pass"] is not True:
        raise OfficialAttemptClosureError(
            "pre_run_assertions checks did not all PASS."
        )
    if payload["final_test_accessed"] is not False:
        raise OfficialAttemptClosureError(
            "Final-test firewall was violated."
        )
    if payload["expected_identity"] != expected:
        raise OfficialAttemptClosureError(
            "pre_run_assertions expected identity mismatch."
        )


def _validate_run_manifest(
    path: Path,
    *,
    run_id: str,
    attempt_id: str,
    expected: dict[str, Any],
) -> None:
    payload = _load_json(path)
    common = (
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
    )
    _require_keys(payload, common, "run_manifest.json")

    expected_pairs = {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "run_type": "OFFICIAL",
        "condition_role": expected["condition_role"],
        "method": expected["method"],
        "architecture": expected["architecture"],
        "budget": expected["budget"],
        "seed_index": expected["seed_index"],
        "training_seed": expected["training_seed"],
        "partition_seed": expected["partition_seed"],
        "total_optimizer_updates": expected["total_optimizer_updates"],
        "validation_interval_updates":
            expected["validation_interval_updates"],
        "effective_labeled_batch":
            expected["effective_labeled_batch"],
        "checkpoint_selection_metric":
            "bbox_mAP_50_95_validation",
        "test_usage": "FINAL_ONLY",
        "training_status": "RUNNING",
        "final_evaluation_status": "NOT_AUTHORIZED",
    }
    for key, value in expected_pairs.items():
        if payload.get(key) != value:
            raise OfficialAttemptClosureError(
                f"run_manifest identity mismatch for {key}."
            )

    if expected["method"] == "SUP":
        if payload["effective_unlabeled_batch"] != "NOT_APPLICABLE":
            raise OfficialAttemptClosureError(
                "SUP effective_unlabeled_batch must be NOT_APPLICABLE."
            )
        if payload["unlabeled_manifest_sha256"] != "NOT_APPLICABLE":
            raise OfficialAttemptClosureError(
                "SUP unlabeled_manifest_sha256 must be NOT_APPLICABLE."
            )
    else:
        _require_keys(
            payload,
            (
                "ema_momentum",
                "ema_skip_buffers",
                "ema_update_basis",
                "labeled_unlabeled_ratio",
                "unsupervised_loss_weight",
                "weak_augmentation_id",
                "strong_augmentation_id",
                "detector_score_floor",
                "initial_pseudo_score",
                "rpn_pseudo_threshold",
                "rcnn_classification_threshold",
                "rpn_nms_iou",
                "rpn_max",
                "rcnn_nms_iou",
                "rcnn_max",
                "jitter_times",
                "jitter_scale",
                "reg_uncertainty_threshold",
                "min_pseudo_bbox_wh",
            ),
            "run_manifest.json",
        )


def validate_official_run_manifest(
    *,
    attempt_dir: Path,
    run_id: str,
    attempt_id: str,
    expected: dict[str, Any],
) -> None:
    """Validate a restored official run manifest without rewriting it."""
    attempt_dir = Path(attempt_dir)
    path = _require_file(attempt_dir, "run_manifest.json")
    _validate_run_manifest(
        path,
        run_id=run_id,
        attempt_id=attempt_id,
        expected=expected,
    )


def _validate_resolved_config(
    attempt_dir: Path,
) -> None:
    config_path = _require_file(
        attempt_dir,
        "resolved_config.yaml",
    )
    try:
        parsed = yaml.safe_load(
            config_path.read_text(encoding="utf-8-sig")
        )
    except Exception as exc:
        raise OfficialAttemptClosureError(
            f"Cannot parse resolved_config.yaml: {exc}"
        ) from exc
    if not isinstance(parsed, dict) or not parsed:
        raise OfficialAttemptClosureError(
            "resolved_config.yaml must be a non-empty mapping."
        )

    hash_path = _require_file(
        attempt_dir,
        "config.sha256",
    )
    recorded = hash_path.read_text(
        encoding="utf-8-sig"
    ).strip().lower()
    if not SHA256_RE.fullmatch(recorded):
        raise OfficialAttemptClosureError(
            "config.sha256 must contain one lowercase SHA-256 digest."
        )
    if recorded != file_sha256(config_path):
        raise OfficialAttemptClosureError(
            "config.sha256 does not match resolved_config.yaml bytes."
        )


def _validate_data_provenance(
    path: Path,
    *,
    expected: dict[str, Any],
) -> None:
    payload = _load_json(path)
    _require_keys(
        payload,
        (
            "condition_role",
            "train_image_count",
            "val_image_count",
            "test_image_count",
            "budget",
            "labeled_image_count",
            "unlabeled_image_count",
            "category_count",
            "train_membership_sha256",
            "val_membership_sha256",
            "test_membership_sha256",
            "labeled_membership_sha256",
            "dataset_loader_config_sha256",
            "preprocessing_config_sha256",
            "image_root_identity",
            "hidden_u_annotations_present_in_training_object",
        ),
        "data_provenance.json",
    )
    if payload["condition_role"] != expected["condition_role"]:
        raise OfficialAttemptClosureError(
            "data_provenance condition_role mismatch."
        )
    if payload["budget"] != expected["budget"]:
        raise OfficialAttemptClosureError(
            "data_provenance budget mismatch."
        )
    if payload["hidden_u_annotations_present_in_training_object"] is not False:
        raise OfficialAttemptClosureError(
            "Hidden-U annotations were present in training object."
        )
    if expected["method"] == "SSL":
        _require_keys(
            payload,
            (
                "unlabeled_membership_sha256",
                "labeled_unlabeled_disjoint",
                "labeled_unlabeled_union_equals_train",
            ),
            "data_provenance.json",
        )
        if payload["labeled_unlabeled_disjoint"] is not True:
            raise OfficialAttemptClosureError(
                "SSL labeled/unlabeled memberships are not disjoint."
            )
        if payload["labeled_unlabeled_union_equals_train"] is not True:
            raise OfficialAttemptClosureError(
                "SSL labeled/unlabeled union does not equal train."
            )


def _validate_seed_runtime(
    path: Path,
    *,
    expected: dict[str, Any],
) -> None:
    payload = _load_json(path)
    _require_keys(
        payload,
        (
            "base_training_seed",
            "python_seed_observed",
            "numpy_seed_observed",
            "torch_initial_seed_observed",
            "cuda_seed_policy",
            "sampler_seed_observed",
            "worker_seed_strategy",
            "worker_seed_examples",
            "augmentation_rng_strategy",
            "torch_deterministic_algorithms_observed",
            "cudnn_deterministic_observed",
            "cudnn_benchmark_observed",
            "membership_hash_before_seed_application",
            "membership_hash_after_seed_application",
            "membership_unchanged",
        ),
        "seed_runtime.json",
    )
    if int(payload["base_training_seed"]) != int(
        expected["training_seed"]
    ):
        raise OfficialAttemptClosureError(
            "seed_runtime base training seed mismatch."
        )
    if payload["membership_unchanged"] is not True:
        raise OfficialAttemptClosureError(
            "Training seed changed data membership."
        )
    if (
        payload["membership_hash_before_seed_application"]
        != payload["membership_hash_after_seed_application"]
    ):
        raise OfficialAttemptClosureError(
            "Membership hash changed after seed application."
        )


def _validate_training_summary(
    path: Path,
    *,
    method: str,
    expected: dict[str, Any],
) -> dict[str, Any]:
    summary = _load_json(path)

    common = (
        "expected_optimizer_updates",
        "observed_optimizer_updates",
        "effective_labeled_batch",
        "labeled_images_consumed_total",
        "validation_events_observed",
        "amp_skipped_step_count",
        "labeled_exposure_matches_update_count",
        "expected_update_budget_reached",
    )
    _require_keys(summary, common, "training_summary.json")

    expected_updates = int(expected["total_optimizer_updates"])
    observed_updates = int(summary["observed_optimizer_updates"])

    if int(summary["expected_optimizer_updates"]) != expected_updates:
        raise OfficialAttemptClosureError(
            "training_summary expected update budget mismatch."
        )
    if observed_updates != expected_updates:
        raise OfficialAttemptClosureError(
            "Official training did not reach expected optimizer updates."
        )
    if summary["expected_update_budget_reached"] is not True:
        raise OfficialAttemptClosureError(
            "training_summary reports incomplete update budget."
        )
    if summary["labeled_exposure_matches_update_count"] is not True:
        raise OfficialAttemptClosureError(
            "Labeled exposure does not match optimizer updates."
        )
    if (
        int(summary["effective_labeled_batch"])
        != int(expected["effective_labeled_batch"])
    ):
        raise OfficialAttemptClosureError(
            "Effective labeled batch mismatch."
        )

    if method == "SSL":
        _require_keys(
            summary,
            (
                "effective_unlabeled_batch",
                "unlabeled_images_consumed_total",
                "ema_update_count",
                "l_u_ratio",
                "unlabeled_exposure_matches_update_count",
                "ema_updates_match_optimizer_updates",
            ),
            "training_summary.json",
        )
        if summary["unlabeled_exposure_matches_update_count"] is not True:
            raise OfficialAttemptClosureError(
                "Unlabeled exposure does not match optimizer updates."
            )
        if summary["ema_updates_match_optimizer_updates"] is not True:
            raise OfficialAttemptClosureError(
                "EMA update count does not match optimizer updates."
            )
        if int(summary["ema_update_count"]) != observed_updates:
            raise OfficialAttemptClosureError(
                "EMA update count mismatch."
            )

    return summary


def _validate_train_log(
    path: Path,
    *,
    method: str,
    observed_updates: int,
) -> None:
    records = _read_jsonl(path)
    if len(records) != observed_updates:
        raise OfficialAttemptClosureError(
            "train.jsonl record count does not equal observed updates."
        )

    common = (
        "optimizer_update",
        "raw_iteration",
        "learning_rate",
        "loss_total",
        "loss_supervised",
        "amp_scale",
        "optimizer_step_executed",
        "elapsed_seconds",
        "gpu_memory_allocated_mb",
        "labeled_microbatches_consumed_cumulative",
        "labeled_images_consumed_cumulative",
    )
    ssl_extra = (
        "loss_unsupervised",
        "pseudo_count",
        "classification_pseudo_count",
        "regression_pseudo_count",
        "unlabeled_microbatches_consumed_cumulative",
        "unlabeled_images_consumed_cumulative",
        "ema_step_executed",
    )

    for index, record in enumerate(records, start=1):
        _require_keys(record, common, "train.jsonl record")
        if int(record["optimizer_update"]) != index:
            raise OfficialAttemptClosureError(
                "train.jsonl optimizer_update sequence is not contiguous."
            )
        if record["optimizer_step_executed"] is not True:
            raise OfficialAttemptClosureError(
                "train.jsonl contains non-executed optimizer step."
            )
        if method == "SSL":
            _require_keys(record, ssl_extra, "SSL train.jsonl record")
            if record["ema_step_executed"] is not True:
                raise OfficialAttemptClosureError(
                    "SSL optimizer update lacks corresponding EMA step."
                )


def _validate_validation_artifacts(
    attempt_dir: Path,
    *,
    method: str,
    observed_updates: int,
    validation_events_observed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    history_path = _require_file(
        attempt_dir,
        "metrics/validation/validation_history.csv",
    )
    fields, rows = _read_csv(history_path)
    required_fields = {
        "optimizer_update",
        "bbox_mAP_50_95",
        "AP50",
        "AP75",
        "AR_50_95_max100",
        "checkpoint_candidate",
        "best_so_far",
        "checkpoint_path",
    }
    if not required_fields.issubset(set(fields)):
        raise OfficialAttemptClosureError(
            "validation_history.csv schema mismatch."
        )
    if len(rows) != validation_events_observed:
        raise OfficialAttemptClosureError(
            "validation_history row count differs from training_summary."
        )

    classwise_path = _require_file(
        attempt_dir,
        "metrics/validation/classwise_validation_ap.csv",
    )
    class_fields, class_rows = _read_csv(classwise_path)
    required_class_fields = {
        "category_id",
        "class_name",
        "AP_50_95",
        "AP50",
        "AP75",
    }
    if not required_class_fields.issubset(set(class_fields)):
        raise OfficialAttemptClosureError(
            "classwise_validation_ap.csv schema mismatch."
        )
    if len(class_rows) != 14:
        raise OfficialAttemptClosureError(
            "classwise_validation_ap.csv must contain 14 category rows."
        )

    best_rel = (
        "metrics/validation/validation_best.json"
        if method == "SUP"
        else "metrics/validation/validation_best_teacher.json"
    )
    best = _load_json(_require_file(attempt_dir, best_rel))

    best_required = (
        "selection_split",
        "selection_metric",
        "selected_update",
        "selected_metric_value",
        "checkpoint_path",
        "checkpoint_sha256",
        "rule",
    )
    _require_keys(best, best_required, best_rel)

    if best["selection_split"] != "validation":
        raise OfficialAttemptClosureError(
            "BEST checkpoint was not selected on validation."
        )
    if best["selection_metric"] != "bbox_mAP_50_95":
        raise OfficialAttemptClosureError(
            "BEST selection metric mismatch."
        )
    if not 0 < int(best["selected_update"]) <= observed_updates:
        raise OfficialAttemptClosureError(
            "BEST selected_update is outside observed update range."
        )

    if method == "SSL":
        if best.get("selection_model") != "EMA_TEACHER":
            raise OfficialAttemptClosureError(
                "SSL BEST selection model is not EMA_TEACHER."
            )

    checkpoint_index = _load_json(
        _require_file(
            attempt_dir,
            "checkpoints/checkpoint_index.json",
        )
    )
    _require_keys(
        checkpoint_index,
        ("checkpoint_policy", "entries"),
        "checkpoint_index.json",
    )
    if checkpoint_index["checkpoint_policy"] != "LOCKED":
        raise OfficialAttemptClosureError(
            "checkpoint_index policy is not LOCKED."
        )
    entries = checkpoint_index["entries"]
    if not isinstance(entries, list):
        raise OfficialAttemptClosureError(
            "checkpoint_index entries must be a list."
        )

    best_entries = [
        item for item in entries
        if isinstance(item, dict) and item.get("role") == "BEST"
    ]
    last_entries = [
        item for item in entries
        if isinstance(item, dict) and item.get("role") == "LAST"
    ]
    if len(best_entries) != 1 or len(last_entries) != 1:
        raise OfficialAttemptClosureError(
            "checkpoint_index requires exactly one BEST and one LAST entry."
        )

    best_entry = best_entries[0]
    last_entry = last_entries[0]

    expected_best_name = (
        "best_model.pth"
        if method == "SUP"
        else "best_ema_teacher.pth"
    )
    expected_last_name = (
        "last_model.pth"
        if method == "SUP"
        else "last_ema_teacher.pth"
    )

    if best_entry.get("path") != expected_best_name:
        raise OfficialAttemptClosureError(
            "checkpoint_index BEST path mismatch."
        )
    if last_entry.get("path") != expected_last_name:
        raise OfficialAttemptClosureError(
            "checkpoint_index LAST path mismatch."
        )
    if best_entry.get("retained") is not True:
        raise OfficialAttemptClosureError(
            "BEST checkpoint is not retained."
        )
    if last_entry.get("retained") is not True:
        raise OfficialAttemptClosureError(
            "LAST checkpoint is not retained."
        )
    if int(last_entry.get("optimizer_update")) != observed_updates:
        raise OfficialAttemptClosureError(
            "LAST checkpoint update does not equal final optimizer update."
        )

    if method == "SSL":
        if best_entry.get("model_role") != "EMA_TEACHER":
            raise OfficialAttemptClosureError(
                "SSL BEST checkpoint model_role mismatch."
            )
        if last_entry.get("model_role") != "EMA_TEACHER":
            raise OfficialAttemptClosureError(
                "SSL LAST checkpoint model_role mismatch."
            )

    best_checkpoint_path = _require_file(
        attempt_dir,
        f"checkpoints/{expected_best_name}",
    )
    last_checkpoint_path = _require_file(
        attempt_dir,
        f"checkpoints/{expected_last_name}",
    )

    if best_entry.get("sha256") != file_sha256(best_checkpoint_path):
        raise OfficialAttemptClosureError(
            "BEST checkpoint SHA-256 mismatch."
        )
    if last_entry.get("sha256") != file_sha256(last_checkpoint_path):
        raise OfficialAttemptClosureError(
            "LAST checkpoint SHA-256 mismatch."
        )
    if best.get("checkpoint_sha256") != best_entry.get("sha256"):
        raise OfficialAttemptClosureError(
            "validation_best checkpoint SHA does not match checkpoint_index."
        )
    if best.get("checkpoint_path") != f"checkpoints/{expected_best_name}":
        raise OfficialAttemptClosureError(
            "validation_best checkpoint path mismatch."
        )

    _require_file(
        attempt_dir,
        "checkpoints/latest_resume.pth",
    )

    return best_entry, last_entry


def _validate_retry(
    path: Path,
    *,
    run_id: str,
    attempt_id: str,
    expected: dict[str, Any],
) -> None:
    payload = _load_json(path)
    _require_keys(
        payload,
        (
            "run_id",
            "attempt_id",
            "training_seed",
            "technical_failure",
            "failure_reason",
            "retry_required",
            "retry_reason",
            "same_seed_confirmed",
            "scientific_protocol_changed",
            "controlled_revision_id",
            "researcher_approval",
        ),
        "retry_deviation.json",
    )
    if payload["run_id"] != run_id:
        raise OfficialAttemptClosureError(
            "retry_deviation run_id mismatch."
        )
    if payload["attempt_id"] != attempt_id:
        raise OfficialAttemptClosureError(
            "retry_deviation attempt_id mismatch."
        )
    if int(payload["training_seed"]) != int(expected["training_seed"]):
        raise OfficialAttemptClosureError(
            "retry_deviation training_seed mismatch."
        )
    if payload["technical_failure"] is not False:
        raise OfficialAttemptClosureError(
            "Cannot close an attempt marked technical failure."
        )
    if payload["retry_required"] is not False:
        raise OfficialAttemptClosureError(
            "Cannot close an attempt requiring retry."
        )
    if payload["same_seed_confirmed"] is not True:
        raise OfficialAttemptClosureError(
            "Same-seed confirmation is absent."
        )
    if payload["scientific_protocol_changed"] is not False:
        raise OfficialAttemptClosureError(
            "Scientific protocol deviation prevents closure."
        )


def _validate_qpseudo(
    attempt_dir: Path,
    *,
    last_checkpoint_sha256: str,
) -> None:
    qpseudo = _load_json(
        _require_file(
            attempt_dir,
            "pseudo/qpseudo_validation.json",
        )
    )
    _require_keys(
        qpseudo,
        (
            "checkpoint_role",
            "checkpoint_sha256",
            "dataset",
            "validation_split_sha256",
            "pseudo_set",
            "pseudo_acceptance_config_sha256",
            "evaluator_config_sha256",
            "hidden_u_gt_used",
            "test_used",
            "PL_mAP_50_95",
            "PL_AP50",
            "PL_AP75",
        ),
        "qpseudo_validation.json",
    )
    if qpseudo["checkpoint_role"] != "LAST_EMA_TEACHER":
        raise OfficialAttemptClosureError(
            "Q_pseudo checkpoint role mismatch."
        )
    if qpseudo["checkpoint_sha256"] != last_checkpoint_sha256:
        raise OfficialAttemptClosureError(
            "Q_pseudo checkpoint SHA does not match LAST EMA Teacher."
        )
    if qpseudo["dataset"] != "fixed_validation":
        raise OfficialAttemptClosureError(
            "Q_pseudo dataset is not fixed_validation."
        )
    if qpseudo["pseudo_set"] != "accepted_rcnn_classification":
        raise OfficialAttemptClosureError(
            "Q_pseudo pseudo-set semantics mismatch."
        )
    if qpseudo["hidden_u_gt_used"] is not False:
        raise OfficialAttemptClosureError(
            "Q_pseudo used hidden-U GT."
        )
    if qpseudo["test_used"] is not False:
        raise OfficialAttemptClosureError(
            "Q_pseudo used test data."
        )

    predictions = _load_json(
        _require_file(
            attempt_dir,
            "pseudo/val_last_teacher_predictions.json",
        )
    )
    if not predictions:
        raise OfficialAttemptClosureError(
            "val_last_teacher_predictions.json is empty."
        )

    fields, rows = _read_csv(
        _require_file(
            attempt_dir,
            "pseudo/qpseudo_classwise.csv",
        )
    )
    required = {
        "category_id",
        "class_name",
        "PL_AP_50_95",
        "PL_AP50",
        "PL_AP75",
    }
    if not required.issubset(set(fields)):
        raise OfficialAttemptClosureError(
            "qpseudo_classwise.csv schema mismatch."
        )
    if len(rows) != 14:
        raise OfficialAttemptClosureError(
            "qpseudo_classwise.csv must contain 14 category rows."
        )


def close_official_attempt(
    *,
    attempt_dir: Path,
    run_id: str,
    attempt_id: str,
    expected: dict[str, Any],
    validation_interval: int,
) -> dict[str, Any]:
    """Validate official training evidence and materialize artifacts 21/22."""
    attempt_dir = Path(attempt_dir).resolve()
    method = str(expected["method"])

    from src.utils.official_run_root_manifest import (
        OfficialRunRootManifestError,
        validate_official_run_root_manifest,
    )

    try:
        validate_official_run_root_manifest(
            run_root=attempt_dir.parent,
            run_id=run_id,
            expected=expected,
        )
    except OfficialRunRootManifestError as exc:
        raise OfficialAttemptClosureError(
            f"Logical-run root manifest validation failed: {exc}"
        ) from exc

    if (attempt_dir / "evaluation" / "final_test").exists():
        raise OfficialAttemptClosureError(
            "Final-test artifacts are prohibited during training closure."
        )

    mandatory_before_completion = _mandatory_paths(method)
    for relative_path in mandatory_before_completion:
        _require_file(attempt_dir, relative_path)

    _validate_pre_run(
        attempt_dir / "pre_run_assertions.json",
        run_id=run_id,
        attempt_id=attempt_id,
        expected=expected,
    )
    _validate_run_manifest(
        attempt_dir / "run_manifest.json",
        run_id=run_id,
        attempt_id=attempt_id,
        expected=expected,
    )
    _validate_resolved_config(attempt_dir)

    environment = _load_json(
        attempt_dir / "environment_reference.json"
    )
    if not environment:
        raise OfficialAttemptClosureError(
            "environment_reference.json must be non-empty."
        )

    _validate_data_provenance(
        attempt_dir / "data_provenance.json",
        expected=expected,
    )
    _validate_seed_runtime(
        attempt_dir / "seed_runtime.json",
        expected=expected,
    )

    summary = _validate_training_summary(
        attempt_dir
        / "metrics"
        / "validation"
        / "training_summary.json",
        method=method,
        expected=expected,
    )
    observed_updates = int(summary["observed_optimizer_updates"])

    _validate_train_log(
        attempt_dir / "train.jsonl",
        method=method,
        observed_updates=observed_updates,
    )
    _read_jsonl(
        attempt_dir / "runtime_events.jsonl"
    )

    _, last_entry = _validate_validation_artifacts(
        attempt_dir,
        method=method,
        observed_updates=observed_updates,
        validation_events_observed=int(
            summary["validation_events_observed"]
        ),
    )

    _validate_retry(
        attempt_dir / "retry_deviation.json",
        run_id=run_id,
        attempt_id=attempt_id,
        expected=expected,
    )

    if method == "SSL":
        _validate_qpseudo(
            attempt_dir,
            last_checkpoint_sha256=str(last_entry["sha256"]),
        )

    if (
        isinstance(validation_interval, bool)
        or not isinstance(validation_interval, int)
        or validation_interval <= 0
    ):
        raise OfficialAttemptClosureError(
            "Resolved validation_interval must be a positive integer."
        )
    if validation_interval != int(
        expected["validation_interval_updates"]
    ):
        raise OfficialAttemptClosureError(
            "Resolved validation_interval differs from locked expectation."
        )

    eligible = expected["condition_role"] in {
        "MAIN_LOW_LABEL",
        "SUP_REFERENCE",
    }
    if not eligible:
        raise OfficialAttemptClosureError(
            "Only Main/reference official runs may become final-test eligible."
        )

    completion = {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "training_status": "CLOSED_VERIFIED",
        "expected_optimizer_updates":
            int(summary["expected_optimizer_updates"]),
        "observed_optimizer_updates":
            observed_updates,
        "validation_interval": validation_interval,
        "best_checkpoint_present": True,
        "last_checkpoint_present": True,
        "artifact_manifest_valid": True,
        "scientific_protocol_deviation": False,
        "eligible_for_final_test": True,
        "final_evaluation_status": "NOT_AUTHORIZED",
    }

    completion_path = (
        attempt_dir / "training_completion_report.json"
    )
    manifest_path = attempt_dir / "artifact_manifest.json"
    try:
        _write_json_atomic(completion_path, completion)

        manifest_paths = (
            mandatory_before_completion
            + ("training_completion_report.json",)
        )

        artifacts: list[dict[str, Any]] = []
        for relative_path in manifest_paths:
            path = _require_file(attempt_dir, relative_path)
            role = ROLE_BY_PATH.get(relative_path)
            if role is None:
                raise OfficialAttemptClosureError(
                    f"No deterministic manifest role for {relative_path}"
                )
            artifacts.append(
                {
                    "path": relative_path,
                    "role": role,
                    "size_bytes": int(path.stat().st_size),
                    "sha256": file_sha256(path),
                    "required": True,
                }
            )

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "artifacts": artifacts,
            "mandatory_artifacts_complete": True,
        }

        _write_json_atomic(manifest_path, manifest)

        written_manifest = _load_json(manifest_path)
        if written_manifest != manifest:
            raise OfficialAttemptClosureError(
                "artifact_manifest round-trip validation failed."
            )
        manifest_sha256 = file_sha256(manifest_path)
        if not SHA256_RE.fullmatch(manifest_sha256):
            raise OfficialAttemptClosureError(
                "artifact_manifest hash validation failed."
            )

        from src.utils.official_logical_run_status import (
            OfficialLogicalRunStatusError,
            set_official_logical_run_closed_verified,
        )

        try:
            set_official_logical_run_closed_verified(
                run_root=attempt_dir.parent,
                run_id=run_id,
                attempt_id=attempt_id,
                training_seed=int(expected["training_seed"]),
                eligible_for_final_test=eligible,
            )
        except OfficialLogicalRunStatusError as exc:
            raise OfficialAttemptClosureError(
                f"Logical-run status closure failed: {exc}"
            ) from exc
    except Exception:
        for closure_path in (manifest_path, completion_path):
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
            file_sha256(completion_path),
        "artifact_manifest":
            str(manifest_path),
        "artifact_manifest_sha256":
            file_sha256(manifest_path),
        "mandatory_artifact_count":
            len(artifacts),
        "mandatory_artifacts_complete": True,
        "training_status": "CLOSED_VERIFIED",
        "eligible_for_final_test": True,
        "final_evaluation_status": "NOT_AUTHORIZED",
    }


__all__ = [
    "OfficialAttemptClosureError",
    "validate_official_run_manifest",
    "close_official_attempt",
]
