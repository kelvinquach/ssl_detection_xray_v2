"""S9.11 ablation-specific active backup and same-attempt recovery.

The verified S9 backup transport, checksum, round-trip verification,
15-minute scheduling, checkpoint durability, and registry semantics are
reused from ``src.utils.s9_active_backup``.

Only the logical namespace and ablation provenance are specialized:

    artifacts/runs/ablation/<run_id>/<attempt_id>

This prevents S9.11 non-Main validation-only runs from being mixed with
the 180 Official training runs.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from src.utils.s9_active_backup import (
    ActiveBackupManager,
    BackupRuntimeError,
    _download_verified_file,
    _load_json,
    _run,
    _safe_relative_path,
    _sha256,
    _write_json_atomic,
    register_active_backup,
    signal_active_backup,
    unregister_active_backup,
    validate_backup_runtime,
)


RUN_TYPE = "ABLATION"
CONDITION_ROLE = "ABLATION"

ABLATION_PREFIX = Path(
    "artifacts/runs/ablation"
)


class AblationActiveBackupManager(
    ActiveBackupManager
):
    """S9.11 backup manager using the isolated ablation namespace."""

    def __init__(
        self,
        repo_root: Path,
        attempt_dir: Path,
    ) -> None:
        self.repo_root = Path(
            repo_root
        ).resolve()

        self.attempt_dir = Path(
            attempt_dir
        ).resolve()

        try:
            relative = (
                self.attempt_dir
                .relative_to(
                    self.repo_root
                )
            )
        except ValueError as exc:
            raise BackupRuntimeError(
                "attempt_dir must be inside repo_root."
            ) from exc

        prefix = ABLATION_PREFIX

        if (
            relative.parts[
                : len(prefix.parts)
            ]
            != prefix.parts
        ):
            raise BackupRuntimeError(
                "Ablation active backup requires "
                "artifacts/runs/ablation/..."
            )

        if (
            len(relative.parts)
            != len(prefix.parts) + 2
        ):
            raise BackupRuntimeError(
                "Ablation attempt path must end "
                "in run_id/attempt_id."
            )

        self.run_id = (
            relative.parts[-2]
        )
        self.attempt_id = (
            relative.parts[-1]
        )

        if not self.run_id.startswith(
            "ABL_R50_B10_"
        ):
            raise BackupRuntimeError(
                "Invalid S9.11 ablation run_id "
                f"for backup: {self.run_id!r}"
            )

        self.runtime = (
            validate_backup_runtime(
                self.repo_root
            )
        )

        self.remote_root = (
            self.runtime["backup_root"]
            + "/"
            + relative.as_posix()
        )

        self.run_root = (
            self.attempt_dir.parent
        )

        self.remote_run_root = (
            self.runtime["backup_root"]
            + "/"
            + relative.parent.as_posix()
        )

        self.run_root_manifest_path = (
            self.run_root
            / "run_root_manifest.json"
        )

        self.manifest_path = (
            self.attempt_dir
            / "backup_sync_manifest.json"
        )

        # State layout intentionally mirrors the verified
        # ActiveBackupManager implementation exactly.
        self._verified: dict[
            str,
            dict[str, Any],
        ] = {}

        self._last_success: (
            str | None
        ) = None

        self._recovery_point: (
            dict[str, Any] | None
        ) = None

        self._last_error: (
            str | None
        ) = None

        self._status = "HEALTHY"

        self._stop = threading.Event()
        self._wake = threading.Event()

        self._sync_lock = (
            threading.Lock()
        )

        self._reason_lock = (
            threading.Lock()
        )

        self._pending_reason: (
            str | None
        ) = None

        self._thread: (
            threading.Thread | None
        ) = None

        self._last_periodic = (
            time.monotonic()
        )

        self._critical_signatures: dict[
            str,
            tuple[int, int],
        ] = {}

        self._run_root_manifest_verified: (
            dict[str, Any] | None
        ) = None

        self._logical_run_status_verified: (
            dict[str, Any] | None
        ) = None

    def _manifest(
        self,
        reason: str,
        verification_status: str,
    ) -> dict[str, Any]:
        """Extend the verified backup manifest with ablation identity."""

        payload = super()._manifest(
            reason,
            verification_status,
        )

        payload.update(
            {
                "run_type":
                    RUN_TYPE,
                "condition_role":
                    CONDITION_ROLE,
                "validation_only":
                    True,
                "eligible_for_final_test":
                    False,
                "test_access_authorized":
                    False,
                "final_test_accessed":
                    False,
            }
        )

        return payload

    def start(self) -> None:
        """Start verified active backup for one S9.11 attempt."""

        if self._thread is not None:
            raise BackupRuntimeError(
                "Ablation active backup worker "
                "already started."
            )

        self.sync(
            "ablation_attempt_start",
            raise_on_error=True,
        )

        self._last_periodic = (
            time.monotonic()
        )

        self._thread = threading.Thread(
            target=self._loop,
            name=(
                f"s9-ablation-backup-"
                f"{self.run_id}-"
                f"{self.attempt_id}"
            ),
            daemon=True,
        )

        self._thread.start()


    def finalize(self) -> None:
        """Finalize durable backup, then cold-tier local recovery-only files."""
        super().finalize()

        manifest = _load_json(
            self.manifest_path
        )

        if (
            manifest.get("verification_status")
            != "PASS"
            or manifest.get("BACKUP_WORKER_STATUS")
            != "HEALTHY"
            or manifest.get("last_error")
            is not None
            or manifest.get("last_sync_reason")
            != "run_completion_final_forced_sync"
        ):
            raise BackupRuntimeError(
                "Ablation cold-tier requires a verified "
                "final forced backup."
            )

        logical_status = _load_json(
            self.run_root
            / "logical_run_status.json"
        )

        if (
            logical_status.get("training_status")
            != "CLOSED_VERIFIED"
            or logical_status.get("training_completed")
            is not True
            or logical_status.get("training_valid")
            is not True
            or logical_status.get("eligible_for_final_test")
            is not False
            or logical_status.get("validation_only")
            is not True
            or logical_status.get("test_access_authorized")
            is not False
            or logical_status.get("final_test_accessed")
            is not False
        ):
            raise BackupRuntimeError(
                "Ablation cold-tier requires CLOSED_VERIFIED "
                "validation-only firewall-safe logical-run state."
            )

        recovery = manifest.get(
            "LATEST_VERIFIED_DURABLE_RECOVERY_POINT"
        )

        if not isinstance(recovery, dict):
            raise BackupRuntimeError(
                "Ablation cold-tier requires a durable "
                "latest-resume recovery point."
            )

        latest_rel = recovery.get("path")

        if latest_rel != "checkpoints/latest_resume.pth":
            raise BackupRuntimeError(
                "Ablation durable recovery path is invalid."
            )

        latest_path = (
            self.attempt_dir
            / latest_rel
        )

        if not latest_path.is_file():
            raise BackupRuntimeError(
                "Local ablation latest_resume.pth is missing "
                "before cold-tier."
            )

        latest_size = latest_path.stat().st_size
        latest_sha256 = _sha256(
            latest_path
        )

        if (
            int(recovery.get("size_bytes", -1))
            != latest_size
            or recovery.get("sha256")
            != latest_sha256
        ):
            raise BackupRuntimeError(
                "Local ablation latest_resume.pth does not "
                "match durable recovery identity."
            )

        latest_manifest_entry = next(
            (
                item
                for item in manifest.get("files", [])
                if (
                    isinstance(item, dict)
                    and item.get("path")
                    == latest_rel
                    and item.get("verification_status")
                    == "PASS"
                    and item.get("sha256")
                    == latest_sha256
                    and int(item.get("size_bytes", -1))
                    == latest_size
                )
            ),
            None,
        )

        if latest_manifest_entry is None:
            raise BackupRuntimeError(
                "Verified backup manifest does not bind "
                "ablation latest_resume.pth."
            )

        epoch_paths = sorted(
            path
            for path in self.attempt_dir.glob(
                "epoch_*.pth"
            )
            if path.is_file()
        )

        epoch_bytes = sum(
            path.stat().st_size
            for path in epoch_paths
        )

        for path in epoch_paths:
            path.unlink()

        latest_path.unlink()

        print(
            "S9_11_POST_BACKUP_COLD_TIER=PASS "
            f"RUN_ID={self.run_id} "
            f"ATTEMPT_ID={self.attempt_id} "
            f"EPOCH_FILES_DELETED={len(epoch_paths)} "
            "LATEST_RESUME_DELETED=1 "
            f"BYTES_FREED={epoch_bytes + latest_size}"
        )



def recover_ablation_attempt(
    repo_root: Path,
    run_id: str,
    attempt_id: str,
) -> dict[str, Any]:
    """Rehydrate one same-attempt S9.11 run from verified durable backup."""

    repo_root = Path(
        repo_root
    ).resolve()

    run_id = str(
        run_id
    ).strip()

    attempt_id = str(
        attempt_id
    ).strip()

    if not run_id or not attempt_id:
        raise BackupRuntimeError(
            "run_id and attempt_id are required."
        )

    if not run_id.startswith(
        "ABL_R50_B10_"
    ):
        raise BackupRuntimeError(
            "Invalid S9.11 ablation run_id "
            f"for recovery: {run_id!r}"
        )

    attempt_dir = (
        repo_root
        / "artifacts"
        / "runs"
        / "ablation"
        / run_id
        / attempt_id
    ).resolve()

    try:
        attempt_dir.relative_to(
            repo_root
        )
    except ValueError as exc:
        raise BackupRuntimeError(
            "Ablation recovery attempt path "
            "escapes repo_root."
        ) from exc

    if attempt_dir.exists():
        raise BackupRuntimeError(
            "Recovery requires a clean local "
            "ablation attempt path: "
            + str(attempt_dir)
        )

    runtime = (
        validate_backup_runtime(
            repo_root
        )
    )

    relative_attempt = (
        ABLATION_PREFIX
        / run_id
        / attempt_id
    )

    remote_root = (
        runtime["backup_root"]
        + "/"
        + relative_attempt.as_posix()
    )

    manifest_remote = (
        remote_root
        + "/backup_sync_manifest.json"
    )

    fd, manifest_name = (
        tempfile.mkstemp(
            prefix=(
                "s9.11.ablation."
                "recovery.manifest."
            ),
            suffix=".json",
        )
    )

    os.close(fd)

    manifest_tmp = Path(
        manifest_name
    )

    try:
        if manifest_tmp.exists():
            manifest_tmp.unlink()

        _run(
            [
                runtime["rclone"],
                "--config",
                runtime["rclone_conf"],
                "copyto",
                manifest_remote,
                str(manifest_tmp),
            ],
            "ablation recovery manifest download",
        )

        manifest = _load_json(
            manifest_tmp
        )

        if (
            manifest.get(
                "verification_status"
            )
            != "PASS"
        ):
            raise BackupRuntimeError(
                "Remote ablation backup "
                "manifest is not PASS."
            )

        if (
            manifest.get("run_id")
            != run_id
            or manifest.get(
                "attempt_id"
            )
            != attempt_id
        ):
            raise BackupRuntimeError(
                "Remote ablation backup "
                "manifest identity mismatch."
            )

        if (
            manifest.get(
                "destination_identity"
            )
            != remote_root
        ):
            raise BackupRuntimeError(
                "Remote ablation backup "
                "destination identity mismatch."
            )

        if (
            manifest.get("run_type")
            != RUN_TYPE
        ):
            raise BackupRuntimeError(
                "Remote backup run_type "
                "is not ABLATION."
            )

        if (
            manifest.get(
                "condition_role"
            )
            != CONDITION_ROLE
        ):
            raise BackupRuntimeError(
                "Remote backup condition_role "
                "is not ABLATION."
            )

        if (
            manifest.get(
                "validation_only"
            )
            is not True
        ):
            raise BackupRuntimeError(
                "Remote ablation backup "
                "is not validation-only."
            )

        if (
            manifest.get(
                "eligible_for_final_test"
            )
            is not False
        ):
            raise BackupRuntimeError(
                "Remote ablation backup became "
                "final-test eligible."
            )

        if (
            manifest.get(
                "test_access_authorized"
            )
            is not False
        ):
            raise BackupRuntimeError(
                "Remote ablation backup has "
                "test access authorization."
            )

        if (
            manifest.get(
                "final_test_accessed"
            )
            is not False
        ):
            raise BackupRuntimeError(
                "Remote ablation backup records "
                "final-test access."
            )

        logical_status = (
            manifest.get(
                "logical_run_status"
            )
        )

        if (
            not isinstance(
                logical_status,
                dict,
            )
            or logical_status.get(
                "verification_status"
            )
            != "PASS"
            or logical_status.get(
                "path"
            )
            != "logical_run_status.json"
            or logical_status.get(
                "source_scope"
            )
            != "LOGICAL_RUN_ROOT"
        ):
            raise BackupRuntimeError(
                "Remote ablation backup lacks "
                "verified logical_run_status.json."
            )

        logical_size = (
            logical_status.get(
                "size_bytes"
            )
        )

        logical_sha = (
            logical_status.get(
                "sha256"
            )
        )

        if (
            isinstance(
                logical_size,
                bool,
            )
            or not isinstance(
                logical_size,
                int,
            )
            or logical_size <= 0
            or not isinstance(
                logical_sha,
                str,
            )
            or len(logical_sha) != 64
            or any(
                ch
                not in
                "0123456789abcdef"
                for ch
                in logical_sha.lower()
            )
        ):
            raise BackupRuntimeError(
                "Remote ablation logical-run "
                "status evidence is invalid."
            )

        remote_run_root = (
            runtime["backup_root"]
            + "/"
            + relative_attempt
            .parent
            .as_posix()
        )

        expected_logical_remote = (
            remote_run_root
            + "/logical_run_status.json"
        )

        if (
            logical_status.get(
                "destination"
            )
            != expected_logical_remote
        ):
            raise BackupRuntimeError(
                "Remote ablation logical status "
                "destination mismatch."
            )

        logical_status_path = (
            repo_root
            / "artifacts"
            / "runs"
            / "ablation"
            / run_id
            / "logical_run_status.json"
        )

        if (
            logical_status_path.exists()
        ):
            raise BackupRuntimeError(
                "Recovery requires local "
                "logical_run_status.json "
                "to be absent."
            )

        recovery_point = (
            manifest.get(
                "LATEST_VERIFIED_DURABLE_RECOVERY_POINT"
            )
        )

        if not isinstance(
            recovery_point,
            dict,
        ):
            raise BackupRuntimeError(
                "No verified durable recovery "
                "point in ablation backup."
            )

        if (
            recovery_point.get(
                "path"
            )
            != "checkpoints/latest_resume.pth"
        ):
            raise BackupRuntimeError(
                "Unexpected ablation durable "
                "recovery path."
            )

        update = (
            recovery_point.get(
                "optimizer_update"
            )
        )

        if (
            isinstance(
                update,
                bool,
            )
            or not isinstance(
                update,
                int,
            )
            or update < 0
        ):
            raise BackupRuntimeError(
                "Invalid ablation durable "
                "recovery optimizer_update."
            )

        files = manifest.get(
            "files"
        )

        if (
            not isinstance(
                files,
                list,
            )
            or not files
        ):
            raise BackupRuntimeError(
                "Remote ablation backup "
                "manifest has no verified files."
            )

        by_path: dict[
            str,
            dict[str, Any],
        ] = {}

        for item in files:
            if (
                not isinstance(
                    item,
                    dict,
                )
                or item.get(
                    "verification_status"
                )
                != "PASS"
            ):
                raise BackupRuntimeError(
                    "Remote ablation backup "
                    "contains unverified file."
                )

            rel = _safe_relative_path(
                str(
                    item.get(
                        "path",
                        "",
                    )
                )
            )

            if rel in by_path:
                raise BackupRuntimeError(
                    "Duplicate path in remote "
                    "ablation backup manifest: "
                    + rel
                )

            size = item.get(
                "size_bytes"
            )

            sha = item.get(
                "sha256"
            )

            if (
                isinstance(
                    size,
                    bool,
                )
                or not isinstance(
                    size,
                    int,
                )
                or size < 0
            ):
                raise BackupRuntimeError(
                    "Invalid ablation recovery "
                    "file size in manifest."
                )

            if (
                not isinstance(
                    sha,
                    str,
                )
                or len(sha) != 64
                or any(
                    ch
                    not in
                    "0123456789abcdef"
                    for ch
                    in sha.lower()
                )
            ):
                raise BackupRuntimeError(
                    "Invalid ablation recovery "
                    "SHA-256 in manifest."
                )

            by_path[rel] = item

        latest_rel = (
            "checkpoints/latest_resume.pth"
        )

        index_rel = (
            "checkpoints/checkpoint_index.json"
        )

        for required in (
            latest_rel,
            index_rel,
        ):
            if required not in by_path:
                raise BackupRuntimeError(
                    "Ablation recovery manifest "
                    "missing required file: "
                    + required
                )

        latest_entry = (
            by_path[
                latest_rel
            ]
        )

        if (
            recovery_point.get(
                "sha256"
            )
            != latest_entry.get(
                "sha256"
            )
            or recovery_point.get(
                "size_bytes"
            )
            != latest_entry.get(
                "size_bytes"
            )
        ):
            raise BackupRuntimeError(
                "Ablation durable recovery "
                "point does not match "
                "latest_resume manifest entry."
            )

        attempt_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        logical_status_restored = (
            False
        )

        try:
            _download_verified_file(
                runtime=runtime,
                remote=(
                    expected_logical_remote
                ),
                destination=(
                    logical_status_path
                ),
                expected_size=int(
                    logical_size
                ),
                expected_sha256=str(
                    logical_sha
                ),
            )

            logical_status_restored = (
                True
            )

            restored_status = (
                _load_json(
                    logical_status_path
                )
            )

            if (
                restored_status.get(
                    "run_id"
                )
                != run_id
                or restored_status.get(
                    "run_type"
                )
                != RUN_TYPE
                or restored_status.get(
                    "condition_role"
                )
                != CONDITION_ROLE
                or restored_status.get(
                    "eligible_for_final_test"
                )
                is not False
                or restored_status.get(
                    "validation_only"
                )
                is not True
                or restored_status.get(
                    "test_access_authorized"
                )
                is not False
                or restored_status.get(
                    "final_test_accessed"
                )
                is not False
            ):
                raise BackupRuntimeError(
                    "Restored ablation logical "
                    "status violates identity "
                    "or final-test firewall."
                )

            for rel, item in sorted(
                by_path.items()
            ):
                _download_verified_file(
                    runtime=runtime,
                    remote=(
                        remote_root
                        + "/"
                        + rel
                    ),
                    destination=(
                        attempt_dir
                        / rel
                    ),
                    expected_size=int(
                        item[
                            "size_bytes"
                        ]
                    ),
                    expected_sha256=str(
                        item[
                            "sha256"
                        ]
                    ),
                )

            index_path = (
                attempt_dir
                / index_rel
            )

            latest_path = (
                attempt_dir
                / latest_rel
            )

            index = _load_json(
                index_path
            )

            match = None

            for entry in index.get(
                "entries",
                [],
            ):
                if (
                    isinstance(
                        entry,
                        dict,
                    )
                    and entry.get(
                        "role"
                    )
                    == "LATEST_RESUME"
                    and entry.get(
                        "path"
                    )
                    == "latest_resume.pth"
                    and entry.get(
                        "sha256"
                    )
                    == _sha256(
                        latest_path
                    )
                ):
                    match = entry
                    break

            if match is None:
                raise BackupRuntimeError(
                    "Restored ablation "
                    "checkpoint_index does not "
                    "bind latest_resume.pth."
                )

            restored_update = (
                match.get(
                    "optimizer_update"
                )
            )

            if (
                isinstance(
                    restored_update,
                    bool,
                )
                or not isinstance(
                    restored_update,
                    int,
                )
                or restored_update
                != update
            ):
                raise BackupRuntimeError(
                    "Restored ablation recovery "
                    "update does not match "
                    "durable manifest."
                )

            if (
                _sha256(
                    index_path
                )
                != str(
                    recovery_point.get(
                        "checkpoint_index_sha256",
                        "",
                    )
                )
            ):
                raise BackupRuntimeError(
                    "Restored ablation "
                    "checkpoint_index SHA-256 "
                    "does not match durable "
                    "recovery point."
                )

            if (
                _sha256(
                    latest_path
                )
                != str(
                    recovery_point.get(
                        "sha256",
                        "",
                    )
                )
            ):
                raise BackupRuntimeError(
                    "Restored ablation "
                    "latest_resume SHA-256 "
                    "does not match durable "
                    "recovery point."
                )

            _write_json_atomic(
                attempt_dir
                / "backup_sync_manifest.json",
                manifest,
            )

        except Exception:
            if attempt_dir.exists():
                shutil.rmtree(
                    attempt_dir
                )

            if (
                logical_status_restored
                and
                logical_status_path.exists()
            ):
                logical_status_path.unlink()

            raise

        return {
            "run_id":
                run_id,
            "attempt_id":
                attempt_id,
            "attempt_dir":
                str(attempt_dir),
            "latest_resume_path":
                str(
                    attempt_dir
                    / latest_rel
                ),
            "optimizer_update":
                update,
            "remote_root":
                remote_root,
        }

    finally:
        if manifest_tmp.exists():
            manifest_tmp.unlink()


__all__ = [
    "ABLATION_PREFIX",
    "AblationActiveBackupManager",
    "BackupRuntimeError",
    "recover_ablation_attempt",
    "register_active_backup",
    "signal_active_backup",
    "unregister_active_backup",
    "validate_backup_runtime",
]