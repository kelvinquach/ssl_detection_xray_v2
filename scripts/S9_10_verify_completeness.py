#!/usr/bin/env python3
"""S9.10 official-training completeness verification.

DETERMINISTIC
NO_TRAINING
NO_FINAL_TEST_ACCESS
NO_SCIENTIFIC_PROTOCOL_CHANGE

Canonical output:
    artifacts/results/completeness_report.json
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "artifacts/results/official_run_inventory.csv"
RUN_ROOT = ROOT / "artifacts/runs/official"
OUTPUT = ROOT / "artifacts/results/completeness_report.json"

RCLONE = ROOT / "tools/rclone-v1.75.1/rclone"
RCLONE_CONF = ROOT / "config/rclone/rclone.conf"
REMOTE_BASE = (
    "gdrive:SSOD_Backup/vast_54286944/"
    "artifacts/runs/official"
)

EXPECTED_GROUPS = {
    "SUP_LOW_LABEL": 80,
    "SUP_REFERENCE": 20,
    "SSL_LOW_LABEL": 80,
}

LOCKED_SEEDS = {
    1: 204886845,
    2: 1480646854,
    3: 1798418854,
    4: 2045683682,
    5: 1814859839,
    6: 1603952859,
    7: 1878351743,
    8: 875651179,
    9: 477581743,
    10: 869675675,
}

QPSEUDO_REQUIRED = (
    "pseudo/qpseudo_validation.json",
    "pseudo/qpseudo_classwise.csv",
    "pseudo/val_last_teacher_predictions.json",
)

LATEST_RESUME = "checkpoints/latest_resume.pth"
LAST_EMA = "checkpoints/last_ema_teacher.pth"

ATTEMPT_RE = re.compile(r"^attempt_[0-9]{3}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not object: {path}")
    return value


def git_head() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return "UNAVAILABLE"
    return proc.stdout.strip()


def classify(row: dict[str, str]) -> str | None:
    method = row["method"].strip()
    role = row["condition_role"].strip()

    if method == "SUP" and role == "MAIN_LOW_LABEL":
        return "SUP_LOW_LABEL"

    if method == "SUP" and role == "SUP_REFERENCE":
        return "SUP_REFERENCE"

    if method == "SSL" and role == "MAIN_LOW_LABEL":
        return "SSL_LOW_LABEL"

    return None


def main() -> int:
    failures: list[dict[str, str]] = []

    def fail(run_id: str, check: str, detail: Any) -> None:
        failures.append(
            {
                "run_id": run_id,
                "check": check,
                "detail": str(detail),
            }
        )

    with INVENTORY.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    run_ids = [row["run_id"].strip() for row in rows]

    group_counts: Counter[str] = Counter()
    state_counts: Counter[str] = Counter()
    final_status_counts: Counter[str] = Counter()

    closed_verified = 0
    artifact_manifest_complete = 0
    backup_pass_healthy = 0
    qpseudo_complete = 0

    required_artifact_entries = 0
    local_required_present = 0
    cold_tiered_latest_resume = 0
    missing_non_resume_required = 0

    attempt_count = 0
    attempt_seed_integrity_pass = 0
    technical_failure_attempts = 0

    #
    # Direct durable-storage audit.
    # One recursive metadata listing only; no checkpoint download.
    #
    proc = subprocess.run(
        [
            str(RCLONE),
            "--config",
            str(RCLONE_CONF),
            "lsjson",
            REMOTE_BASE,
            "--recursive",
            "--files-only",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    remote_map: dict[str, int] = {}

    if proc.returncode != 0:
        fail(
            "GLOBAL",
            "REMOTE_LIST",
            proc.stderr.strip(),
        )
    else:
        try:
            listing = json.loads(proc.stdout)
            remote_map = {
                item["Path"]: int(item["Size"])
                for item in listing
                if isinstance(item, dict)
                and "Path" in item
                and "Size" in item
            }
        except Exception as exc:
            fail(
                "GLOBAL",
                "REMOTE_LIST_PARSE",
                exc,
            )

    remote_expected = 0
    remote_present = 0
    remote_size_match = 0

    latest_remote_expected = 0
    latest_remote_present = 0
    latest_remote_size_match = 0

    #
    # Global inventory invariants.
    #
    if len(rows) != 180:
        fail(
            "GLOBAL",
            "INVENTORY_ROW_COUNT",
            f"expected=180 observed={len(rows)}",
        )

    if len(set(run_ids)) != 180:
        fail(
            "GLOBAL",
            "UNIQUE_RUN_ID_COUNT",
            f"expected=180 observed={len(set(run_ids))}",
        )

    #
    # Per logical run.
    #
    for row in rows:
        run_id = row["run_id"].strip()

        group = classify(row)

        if group is None:
            fail(
                run_id,
                "GROUP_CLASSIFICATION",
                (
                    f"method={row.get('method')} "
                    f"condition_role={row.get('condition_role')}"
                ),
            )
        else:
            group_counts[group] += 1

        try:
            seed_index = int(row["seed_index"])
            training_seed = int(row["training_seed"])
        except Exception as exc:
            fail(
                run_id,
                "SEED_PARSE",
                exc,
            )
            continue

        if LOCKED_SEEDS.get(seed_index) != training_seed:
            fail(
                run_id,
                "LOCKED_SEED_MAPPING",
                (
                    f"seed_index={seed_index} "
                    f"expected={LOCKED_SEEDS.get(seed_index)} "
                    f"observed={training_seed}"
                ),
            )

        run_dir = RUN_ROOT / run_id
        status_path = run_dir / "logical_run_status.json"

        try:
            status = load_json(status_path)
        except Exception as exc:
            fail(
                run_id,
                "LOGICAL_RUN_STATUS",
                exc,
            )
            continue

        state_counts[str(status.get("training_status"))] += 1
        final_status_counts[
            str(status.get("final_evaluation_status"))
        ] += 1

        if status.get("run_id") != run_id:
            fail(
                run_id,
                "STATUS_RUN_ID",
                status.get("run_id"),
            )

        if status.get("training_seed") != training_seed:
            fail(
                run_id,
                "STATUS_TRAINING_SEED",
                status.get("training_seed"),
            )

        if status.get("training_status") != "CLOSED_VERIFIED":
            fail(
                run_id,
                "TRAINING_STATUS",
                status.get("training_status"),
            )
            continue

        if status.get("training_completed") is not True:
            fail(
                run_id,
                "TRAINING_COMPLETED",
                status.get("training_completed"),
            )

        if status.get("training_valid") is not True:
            fail(
                run_id,
                "TRAINING_VALID",
                status.get("training_valid"),
            )

        if status.get("eligible_for_final_test") is not True:
            fail(
                run_id,
                "ELIGIBLE_FOR_FINAL_TEST",
                status.get("eligible_for_final_test"),
            )

        if (
            status.get("final_evaluation_status")
            != "NOT_AUTHORIZED"
        ):
            fail(
                run_id,
                "FINAL_TEST_FIREWALL",
                status.get("final_evaluation_status"),
            )

        attempt_id = status.get("attempt_used")

        if (
            not isinstance(attempt_id, str)
            or ATTEMPT_RE.fullmatch(attempt_id) is None
        ):
            fail(
                run_id,
                "ATTEMPT_USED",
                attempt_id,
            )
            continue

        attempt_dir = run_dir / attempt_id

        if not attempt_dir.is_dir():
            fail(
                run_id,
                "SUCCESSFUL_ATTEMPT_DIRECTORY",
                attempt_dir,
            )
            continue

        expected_completion_path = (
            f"{attempt_id}/training_completion_report.json"
        )
        expected_manifest_path = (
            f"{attempt_id}/artifact_manifest.json"
        )

        if (
            status.get("training_completion_report_path")
            != expected_completion_path
        ):
            fail(
                run_id,
                "TRAINING_COMPLETION_REPORT_PATH",
                status.get("training_completion_report_path"),
            )

        if (
            status.get("artifact_manifest_path")
            != expected_manifest_path
        ):
            fail(
                run_id,
                "ARTIFACT_MANIFEST_PATH",
                status.get("artifact_manifest_path"),
            )

        #
        # Successful completion report.
        #
        try:
            completion = load_json(
                attempt_dir / "training_completion_report.json"
            )
        except Exception as exc:
            fail(
                run_id,
                "TRAINING_COMPLETION_REPORT",
                exc,
            )
            continue

        completion_checks = {
            "run_id":
                completion.get("run_id") == run_id,
            "attempt_id":
                completion.get("attempt_id") == attempt_id,
            "training_status":
                completion.get("training_status")
                == "CLOSED_VERIFIED",
            "optimizer_updates":
                completion.get("expected_optimizer_updates")
                == completion.get("observed_optimizer_updates"),
            "best_checkpoint_present":
                completion.get("best_checkpoint_present") is True,
            "last_checkpoint_present":
                completion.get("last_checkpoint_present") is True,
            "artifact_manifest_valid":
                completion.get("artifact_manifest_valid") is True,
            "scientific_protocol_deviation":
                completion.get(
                    "scientific_protocol_deviation"
                ) is False,
            "eligible_for_final_test":
                completion.get(
                    "eligible_for_final_test"
                ) is True,
            "final_evaluation_status":
                completion.get(
                    "final_evaluation_status"
                ) == "NOT_AUTHORIZED",
        }

        for name, passed in completion_checks.items():
            if not passed:
                fail(
                    run_id,
                    "COMPLETION_" + name.upper(),
                    completion,
                )

        #
        # Successful attempt retry/deviation record.
        #
        try:
            successful_retry = load_json(
                attempt_dir / "retry_deviation.json"
            )
        except Exception as exc:
            fail(
                run_id,
                "SUCCESSFUL_RETRY_DEVIATION",
                exc,
            )
            continue

        if not (
            successful_retry.get("run_id") == run_id
            and successful_retry.get("attempt_id") == attempt_id
            and successful_retry.get("training_seed")
                == training_seed
            and successful_retry.get("technical_failure")
                is False
            and successful_retry.get("retry_required")
                is False
            and successful_retry.get("same_seed_confirmed")
                is True
            and successful_retry.get(
                "scientific_protocol_changed"
            ) is False
        ):
            fail(
                run_id,
                "SUCCESSFUL_RETRY_INTEGRITY",
                successful_retry,
            )

        #
        # Artifact manifest.
        #
        try:
            manifest = load_json(
                attempt_dir / "artifact_manifest.json"
            )
        except Exception as exc:
            fail(
                run_id,
                "ARTIFACT_MANIFEST",
                exc,
            )
            continue

        if (
            manifest.get("run_id") != run_id
            or manifest.get("attempt_id") != attempt_id
        ):
            fail(
                run_id,
                "ARTIFACT_MANIFEST_IDENTITY",
                (
                    f"{manifest.get('run_id')}|"
                    f"{manifest.get('attempt_id')}"
                ),
            )

        if manifest.get("mandatory_artifacts_complete") is True:
            artifact_manifest_complete += 1
        else:
            fail(
                run_id,
                "MANDATORY_ARTIFACTS_COMPLETE",
                manifest.get("mandatory_artifacts_complete"),
            )

        #
        # Backup manifest.
        #
        try:
            backup = load_json(
                attempt_dir / "backup_sync_manifest.json"
            )
        except Exception as exc:
            fail(
                run_id,
                "BACKUP_SYNC_MANIFEST",
                exc,
            )
            continue

        backup_ok = (
            backup.get("verification_status") == "PASS"
            and backup.get("BACKUP_WORKER_STATUS") == "HEALTHY"
            and backup.get("last_error") in (None, "")
        )

        if backup_ok:
            backup_pass_healthy += 1
        else:
            fail(
                run_id,
                "BACKUP_PASS_HEALTHY",
                (
                    f"{backup.get('verification_status')}|"
                    f"{backup.get('BACKUP_WORKER_STATUS')}|"
                    f"{backup.get('last_error')}"
                ),
            )

        backup_files = {
            entry.get("path"): entry
            for entry in backup.get("files", [])
            if isinstance(entry, dict)
            and isinstance(entry.get("path"), str)
        }

        durable = (
            backup.get(
                "LATEST_VERIFIED_DURABLE_RECOVERY_POINT"
            )
            or {}
        )

        required_paths: set[str] = set()
        artifact_by_path: dict[str, dict[str, Any]] = {}

        artifacts = manifest.get("artifacts")

        if not isinstance(artifacts, list):
            fail(
                run_id,
                "ARTIFACT_LIST",
                type(artifacts).__name__,
            )
            artifacts = []

        for entry in artifacts:
            if (
                not isinstance(entry, dict)
                or entry.get("required") is not True
            ):
                continue

            required_artifact_entries += 1

            relative_path = entry.get("path")
            expected_size = entry.get("size_bytes")
            expected_sha = entry.get("sha256")

            if not isinstance(relative_path, str):
                fail(
                    run_id,
                    "ARTIFACT_PATH",
                    entry,
                )
                continue

            required_paths.add(relative_path)
            artifact_by_path[relative_path] = entry

            if (
                not isinstance(expected_sha, str)
                or SHA256_RE.fullmatch(expected_sha) is None
            ):
                fail(
                    run_id,
                    "ARTIFACT_SHA256",
                    f"{relative_path}|{expected_sha}",
                )

            local_path = attempt_dir / relative_path

            if local_path.is_file():
                local_required_present += 1

                try:
                    if (
                        local_path.stat().st_size
                        != int(expected_size)
                    ):
                        fail(
                            run_id,
                            "LOCAL_ARTIFACT_SIZE",
                            relative_path,
                        )
                except Exception as exc:
                    fail(
                        run_id,
                        "LOCAL_ARTIFACT_SIZE_PARSE",
                        f"{relative_path}|{exc}",
                    )

                continue

            #
            # Only operational LATEST_RESUME may have been cold-tiered.
            #
            if relative_path != LATEST_RESUME:
                missing_non_resume_required += 1
                fail(
                    run_id,
                    "MISSING_REQUIRED_NON_RESUME",
                    relative_path,
                )
                continue

            backup_entry = backup_files.get(
                LATEST_RESUME
            ) or {}

            try:
                manifest_size = int(expected_size)
                backup_size = int(
                    backup_entry.get("size_bytes", -1)
                )
                durable_size = int(
                    durable.get("size_bytes", -2)
                )
            except Exception:
                manifest_size = -3
                backup_size = -4
                durable_size = -5

            cold_tier_ok = (
                backup_ok
                and backup_entry.get(
                    "verification_status"
                ) == "PASS"
                and backup_entry.get("sha256")
                    == expected_sha
                and manifest_size
                    == backup_size
                    == durable_size
                and durable.get("path")
                    == LATEST_RESUME
                and durable.get("sha256")
                    == expected_sha
            )

            if cold_tier_ok:
                cold_tiered_latest_resume += 1
            else:
                fail(
                    run_id,
                    "LATEST_RESUME_DURABLE_INTEGRITY",
                    relative_path,
                )

        #
        # Q_pseudo completeness + relational binding.
        #
        if row["method"].strip() == "SSL":
            qpseudo_ok = True

            for relative_path in QPSEUDO_REQUIRED:
                if relative_path not in required_paths:
                    qpseudo_ok = False
                    fail(
                        run_id,
                        "QPSEUDO_NOT_REQUIRED_IN_MANIFEST",
                        relative_path,
                    )

                if not (attempt_dir / relative_path).is_file():
                    qpseudo_ok = False
                    fail(
                        run_id,
                        "QPSEUDO_FILE_MISSING",
                        relative_path,
                    )

            try:
                qpseudo = load_json(
                    attempt_dir
                    / "pseudo/qpseudo_validation.json"
                )
            except Exception as exc:
                qpseudo_ok = False
                fail(
                    run_id,
                    "QPSEUDO_VALIDATION_PARSE",
                    exc,
                )
                qpseudo = {}

            last_entry = artifact_by_path.get(LAST_EMA) or {}

            if qpseudo.get("checkpoint_role") != "LAST_EMA_TEACHER":
                qpseudo_ok = False
                fail(
                    run_id,
                    "QPSEUDO_CHECKPOINT_ROLE",
                    qpseudo.get("checkpoint_role"),
                )

            if qpseudo.get("dataset") != "fixed_validation":
                qpseudo_ok = False
                fail(
                    run_id,
                    "QPSEUDO_DATASET",
                    qpseudo.get("dataset"),
                )

            if (
                qpseudo.get("pseudo_set")
                != "accepted_rcnn_classification"
            ):
                qpseudo_ok = False
                fail(
                    run_id,
                    "QPSEUDO_SET",
                    qpseudo.get("pseudo_set"),
                )

            if qpseudo.get("hidden_u_gt_used") is not False:
                qpseudo_ok = False
                fail(
                    run_id,
                    "QPSEUDO_HIDDEN_U_FIREWALL",
                    qpseudo.get("hidden_u_gt_used"),
                )

            if qpseudo.get("test_used") is not False:
                qpseudo_ok = False
                fail(
                    run_id,
                    "QPSEUDO_TEST_FIREWALL",
                    qpseudo.get("test_used"),
                )

            if (
                not last_entry
                or qpseudo.get("checkpoint_sha256")
                    != last_entry.get("sha256")
            ):
                qpseudo_ok = False
                fail(
                    run_id,
                    "QPSEUDO_LAST_EMA_HASH_BINDING",
                    (
                        f"qpseudo="
                        f"{qpseudo.get('checkpoint_sha256')} "
                        f"last="
                        f"{last_entry.get('sha256')}"
                    ),
                )

            if qpseudo_ok:
                qpseudo_complete += 1

        #
        # Direct remote check for every file in the successful
        # attempt backup manifest.
        #
        for relative_path, backup_entry in backup_files.items():
            remote_key = (
                f"{run_id}/{attempt_id}/{relative_path}"
            )

            remote_expected += 1

            if relative_path == LATEST_RESUME:
                latest_remote_expected += 1

            if remote_key not in remote_map:
                fail(
                    run_id,
                    "REMOTE_FILE_MISSING",
                    relative_path,
                )
                continue

            remote_present += 1

            if relative_path == LATEST_RESUME:
                latest_remote_present += 1

            try:
                expected_remote_size = int(
                    backup_entry["size_bytes"]
                )
            except Exception as exc:
                fail(
                    run_id,
                    "REMOTE_EXPECTED_SIZE_PARSE",
                    f"{relative_path}|{exc}",
                )
                continue

            if (
                remote_map[remote_key]
                != expected_remote_size
            ):
                fail(
                    run_id,
                    "REMOTE_SIZE_MISMATCH",
                    (
                        f"{relative_path}|"
                        f"expected={expected_remote_size}|"
                        f"observed={remote_map[remote_key]}"
                    ),
                )
            else:
                remote_size_match += 1

                if relative_path == LATEST_RESUME:
                    latest_remote_size_match += 1

        #
        # All technical attempts must preserve exact seed/protocol.
        #
        for attempt_dir_any in sorted(
            run_dir.glob("attempt_*")
        ):
            if (
                not attempt_dir_any.is_dir()
                or ATTEMPT_RE.fullmatch(
                    attempt_dir_any.name
                ) is None
            ):
                continue

            retry_path = (
                attempt_dir_any / "retry_deviation.json"
            )

            if not retry_path.is_file():
                fail(
                    run_id,
                    "ATTEMPT_RETRY_DEVIATION_MISSING",
                    attempt_dir_any.name,
                )
                continue

            attempt_count += 1

            try:
                retry = load_json(retry_path)
            except Exception as exc:
                fail(
                    run_id,
                    "ATTEMPT_RETRY_DEVIATION_PARSE",
                    f"{attempt_dir_any.name}|{exc}",
                )
                continue

            exact_attempt_integrity = (
                retry.get("run_id") == run_id
                and retry.get("attempt_id")
                    == attempt_dir_any.name
                and retry.get("training_seed")
                    == training_seed
                and retry.get("same_seed_confirmed")
                    is True
                and retry.get(
                    "scientific_protocol_changed"
                ) is False
            )

            if exact_attempt_integrity:
                attempt_seed_integrity_pass += 1
            else:
                fail(
                    run_id,
                    "ATTEMPT_SEED_PROTOCOL_INTEGRITY",
                    attempt_dir_any.name,
                )

            if retry.get("technical_failure") is True:
                technical_failure_attempts += 1

        closed_verified += 1

    #
    # Global completeness.
    #
    for group_name, expected_count in EXPECTED_GROUPS.items():
        if group_counts[group_name] != expected_count:
            fail(
                "GLOBAL",
                group_name + "_COUNT",
                (
                    f"expected={expected_count} "
                    f"observed={group_counts[group_name]}"
                ),
            )

    if closed_verified != 180:
        fail(
            "GLOBAL",
            "CLOSED_VERIFIED_COUNT",
            closed_verified,
        )

    if artifact_manifest_complete != 180:
        fail(
            "GLOBAL",
            "ARTIFACT_MANIFEST_COMPLETE_COUNT",
            artifact_manifest_complete,
        )

    if backup_pass_healthy != 180:
        fail(
            "GLOBAL",
            "BACKUP_PASS_HEALTHY_COUNT",
            backup_pass_healthy,
        )

    if qpseudo_complete != 80:
        fail(
            "GLOBAL",
            "QPSEUDO_COMPLETE_COUNT",
            qpseudo_complete,
        )

    if attempt_seed_integrity_pass != attempt_count:
        fail(
            "GLOBAL",
            "NO_REPLACEMENT_SEED",
            (
                f"attempts={attempt_count} "
                f"integrity_pass="
                f"{attempt_seed_integrity_pass}"
            ),
        )

    if (
        remote_present != remote_expected
        or remote_size_match != remote_expected
    ):
        fail(
            "GLOBAL",
            "DIRECT_REMOTE_BACKUP_COMPLETENESS",
            (
                f"expected={remote_expected} "
                f"present={remote_present} "
                f"size_match={remote_size_match}"
            ),
        )

    if not (
        latest_remote_expected == 180
        and latest_remote_present == 180
        and latest_remote_size_match == 180
    ):
        fail(
            "GLOBAL",
            "DIRECT_REMOTE_LATEST_RESUME",
            (
                f"expected={latest_remote_expected} "
                f"present={latest_remote_present} "
                f"size_match={latest_remote_size_match}"
            ),
        )

    #
    # Final-test firewall remains closed.
    #
    final_results = (
        ROOT
        / "artifacts/final_test/"
        "official_final_test_results.csv"
    )

    final_reports = list(
        RUN_ROOT.glob(
            "*/attempt_*/final_evaluation_report.json"
        )
    )

    final_test_accessed = (
        final_results.exists()
        or bool(final_reports)
        or final_status_counts.get(
            "NOT_AUTHORIZED",
            0,
        ) != 180
    )

    if final_test_accessed:
        fail(
            "GLOBAL",
            "FINAL_TEST_FIREWALL",
            (
                f"official_results="
                f"{final_results.exists()} "
                f"final_reports="
                f"{len(final_reports)} "
                f"not_authorized="
                f"{final_status_counts.get('NOT_AUTHORIZED', 0)}"
            ),
        )

    checks = {
        "inventory_exact_180":
            len(rows) == 180,
        "unique_run_ids_180":
            len(set(run_ids)) == 180,
        "sup_low_label_80":
            group_counts["SUP_LOW_LABEL"] == 80,
        "sup_reference_20":
            group_counts["SUP_REFERENCE"] == 20,
        "ssl_low_label_80":
            group_counts["SSL_LOW_LABEL"] == 80,
        "closed_verified_180":
            closed_verified == 180,
        "artifact_manifests_complete_180":
            artifact_manifest_complete == 180,
        "backup_pass_healthy_180":
            backup_pass_healthy == 180,
        "qpseudo_complete_and_bound_80":
            qpseudo_complete == 80,
        "no_replacement_seed":
            attempt_seed_integrity_pass == attempt_count,
        "missing_non_resume_required_zero":
            missing_non_resume_required == 0,
        "direct_remote_backup_complete":
            (
                remote_expected
                == remote_present
                == remote_size_match
            ),
        "direct_remote_latest_resume_180":
            (
                latest_remote_expected
                == latest_remote_present
                == latest_remote_size_match
                == 180
            ),
        "final_test_not_accessed":
            not final_test_accessed,
    }

    status = (
        "PASS"
        if not failures and all(checks.values())
        else "FAIL"
    )

    report = {
        "schema_version": "1.1",
        "stage": "S9.10",
        "task": "OFFICIAL_TRAINING_COMPLETENESS",
        "status": status,
        "git_commit": git_head(),
        "inventory_path":
            "artifacts/results/official_run_inventory.csv",
        "inventory_rows": len(rows),
        "unique_run_id_count": len(set(run_ids)),
        "expected_total_runs": 180,
        "closed_verified_count": closed_verified,
        "group_counts": {
            "SUP_LOW_LABEL":
                group_counts["SUP_LOW_LABEL"],
            "SUP_REFERENCE":
                group_counts["SUP_REFERENCE"],
            "SSL_LOW_LABEL":
                group_counts["SSL_LOW_LABEL"],
        },
        "logical_state_counts":
            dict(sorted(state_counts.items())),
        "final_evaluation_status_counts":
            dict(sorted(final_status_counts.items())),
        "artifact_manifest_complete_count":
            artifact_manifest_complete,
        "required_artifact_entries":
            required_artifact_entries,
        "local_required_artifact_entries":
            local_required_present,
        "cold_tiered_durable_latest_resume_count":
            cold_tiered_latest_resume,
        "missing_non_resume_required_count":
            missing_non_resume_required,
        "backup_pass_healthy_count":
            backup_pass_healthy,
        "qpseudo_complete_count":
            qpseudo_complete,
        "attempt_count":
            attempt_count,
        "technical_failure_attempt_count":
            technical_failure_attempts,
        "retry_seed_integrity_pass_count":
            attempt_seed_integrity_pass,
        "no_replacement_seed":
            attempt_seed_integrity_pass == attempt_count,
        "direct_remote_audit": {
            "remote_files_listed":
                len(remote_map),
            "expected_successful_attempt_files":
                remote_expected,
            "remote_present":
                remote_present,
            "remote_size_match":
                remote_size_match,
            "latest_resume_expected":
                latest_remote_expected,
            "latest_resume_remote_present":
                latest_remote_present,
            "latest_resume_remote_size_match":
                latest_remote_size_match,
        },
        "final_test_accessed":
            final_test_accessed,
        "checks":
            checks,
        "failure_count":
            len(failures),
        "failures":
            failures,
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("S9_10_STATUS=" + status)
    print(
        "REPORT="
        + str(OUTPUT.relative_to(ROOT))
    )
    print(
        "CLOSED_VERIFIED="
        + str(closed_verified)
    )
    print(
        "SUP_LOW_LABEL="
        + str(group_counts["SUP_LOW_LABEL"])
    )
    print(
        "SUP_REFERENCE="
        + str(group_counts["SUP_REFERENCE"])
    )
    print(
        "SSL_LOW_LABEL="
        + str(group_counts["SSL_LOW_LABEL"])
    )
    print(
        "ARTIFACT_MANIFEST_COMPLETE="
        + str(artifact_manifest_complete)
    )
    print(
        "BACKUP_PASS_HEALTHY="
        + str(backup_pass_healthy)
    )
    print(
        "QPSEUDO_COMPLETE="
        + str(qpseudo_complete)
    )
    print(
        "COLD_TIERED_DURABLE_LATEST_RESUME="
        + str(cold_tiered_latest_resume)
    )
    print(
        "MISSING_NON_RESUME_REQUIRED="
        + str(missing_non_resume_required)
    )
    print(
        "ATTEMPT_SEED_INTEGRITY="
        + f"{attempt_seed_integrity_pass}/{attempt_count}"
    )
    print(
        "REMOTE_BACKUP_FILES="
        + f"{remote_size_match}/{remote_expected}"
    )
    print(
        "REMOTE_LATEST_RESUME="
        + f"{latest_remote_size_match}/180"
    )
    print(
        "FINAL_TEST_ACCESSED="
        + str(final_test_accessed)
    )
    print(
        "FAILURE_COUNT="
        + str(len(failures))
    )

    for item in failures:
        print(
            "FAIL | "
            + item["run_id"]
            + " | "
            + item["check"]
            + " | "
            + item["detail"]
        )

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
