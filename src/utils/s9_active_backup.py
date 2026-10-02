"""S9 active per-attempt Vast -> Google Drive durability support."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BACKUP_SYNC_INTERVAL_MINUTES = 15
BACKUP_SYNC_INTERVAL_SECONDS = BACKUP_SYNC_INTERVAL_MINUTES * 60
BACKUP_EVENT_POLL_SECONDS = 1.0
RCLONE_EXPECTED_SHA256 = "f66d8c1d552ad90296a11bc8b46d56a7fa5da1a7fa05e7ca522d95df92c4a4c0"

ENV_DIR = Path("artifacts/preflight/environment")
STORAGE_ENV = ENV_DIR / "storage_environment.json"
CHECKSUM_ENV = ENV_DIR / "backup_checksum_verification_test.json"
RESTORE_ENV = ENV_DIR / "storage_restore_test.json"

BASE_PATHS = (
    "pre_run_assertions.json",
    "run_manifest.json",
    "resolved_config.yaml",
    "config.sha256",
    "data_provenance.json",
    "seed_runtime.json",
    "environment_reference.json",
    "train.jsonl",
    "runtime_events.jsonl",
    "retry_deviation.json",
    "training_completion_report.json",
    "artifact_manifest.json",
    "metrics/validation/validation_history.csv",
    "metrics/validation/validation_best.json",
    "metrics/validation/validation_best_teacher.json",
    "metrics/validation/classwise_validation_ap.csv",
    "metrics/validation/training_summary.json",
    "checkpoints/checkpoint_index.json",
    "checkpoints/best_model.pth",
    "checkpoints/last_model.pth",
    "checkpoints/best_ema_teacher.pth",
    "checkpoints/last_ema_teacher.pth",
    "checkpoints/latest_resume.pth",
    "pseudo/qpseudo_validation.json",
    "pseudo/qpseudo_classwise.csv",
    "pseudo/val_last_teacher_predictions.json",
    "pseudo/pseudo_diagnostics.json",
)

CRITICAL_PATHS = frozenset(
    path for path in BASE_PATHS
    if path.startswith(("checkpoints/", "metrics/validation/", "pseudo/"))
    or path in {"training_completion_report.json", "artifact_manifest.json"}
)


class BackupRuntimeError(RuntimeError):
    """Raised when S9 active backup cannot satisfy its durability contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise BackupRuntimeError(f"Cannot parse {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BackupRuntimeError(f"Expected JSON object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _all_checks_pass(payload: dict[str, Any]) -> bool:
    checks = payload.get("checks")
    return (
        isinstance(checks, dict)
        and bool(checks)
        and all(value is True for value in checks.values())
    )


def _run(args: list[str], description: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args, check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise BackupRuntimeError(
            f"{description} failed: {detail or result.returncode}"
        )
    return result


def validate_backup_runtime(repo_root: Path) -> dict[str, str]:
    """Require post-S1 backup evidence plus live Drive accessibility."""
    repo_root = Path(repo_root).resolve()
    paths = [repo_root / STORAGE_ENV, repo_root / CHECKSUM_ENV, repo_root / RESTORE_ENV]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise BackupRuntimeError(
            "Backup runtime evidence missing; run locked Vast backup bootstrap: "
            + ", ".join(missing)
        )

    storage, checksum, restore = (_load_json(path) for path in paths)

    if storage.get("status") != "PASS":
        raise BackupRuntimeError("storage_environment status is not PASS.")
    if storage.get("s1_13_backup_path_writable") != "PASS":
        raise BackupRuntimeError("S1.13 backup path writable gate is not PASS.")
    if not _all_checks_pass(storage):
        raise BackupRuntimeError("storage_environment contains failed checks.")

    backup = storage.get("backup")
    if not isinstance(backup, dict):
        raise BackupRuntimeError(
            "storage_environment is not post-backup-preparation evidence."
        )

    destination = backup.get("destination")
    tool = backup.get("tool")
    auth = backup.get("authentication")
    io_policy = backup.get("training_io_policy")
    if not all(isinstance(x, dict) for x in (destination, tool, auth, io_policy)):
        raise BackupRuntimeError("Backup runtime evidence is incomplete.")

    if destination.get("remote_name") != "gdrive:":
        raise BackupRuntimeError("Unexpected rclone remote identity.")
    if destination.get("write_test_verified") is not True:
        raise BackupRuntimeError("Backup write test is not verified.")
    backup_root = str(destination.get("backup_root", "")).strip().rstrip("/")
    if not backup_root:
        raise BackupRuntimeError("Backup root is missing.")

    if tool.get("name") != "rclone" or tool.get("version") != "rclone v1.75.1":
        raise BackupRuntimeError("Pinned rclone identity mismatch.")
    if tool.get("binary_sha256") != RCLONE_EXPECTED_SHA256:
        raise BackupRuntimeError("Recorded rclone SHA-256 mismatch.")
    rclone = Path(str(tool.get("binary_path", "")))
    if not rclone.is_file() or _sha256(rclone) != RCLONE_EXPECTED_SHA256:
        raise BackupRuntimeError("Observed rclone binary identity mismatch.")

    conf = Path(str(auth.get("configuration_path", "")))
    if not conf.is_file():
        raise BackupRuntimeError(f"rclone config missing: {conf}")
    if stat.S_IMODE(conf.stat().st_mode) != 0o600:
        raise BackupRuntimeError("rclone config permissions must be 600.")
    if str(auth.get("configuration_permissions")) != "600":
        raise BackupRuntimeError("Recorded rclone config permissions are not 600.")
    if auth.get("credentials_embedded_in_evidence") is not False:
        raise BackupRuntimeError("Credentials must not be embedded in evidence.")

    if io_policy.get("training_uses_google_drive_directly") is not False:
        raise BackupRuntimeError("Google Drive must not be a training source.")

    if (
        checksum.get("status") != "PASS"
        or checksum.get("verification_status") != "PASS"
        or not _all_checks_pass(checksum)
    ):
        raise BackupRuntimeError("Backup checksum foundation is not PASS.")
    if restore.get("status") != "PASS" or not _all_checks_pass(restore):
        raise BackupRuntimeError("Storage restore foundation is not PASS.")

    _run(
        [str(rclone), "--config", str(conf), "lsf", backup_root],
        "live backup destination accessibility check",
    )

    return {
        "rclone": str(rclone),
        "rclone_conf": str(conf),
        "backup_root": backup_root,
    }


class ActiveBackupManager:
    """Periodic 15-minute and event-triggered per-attempt backup manager."""

    def __init__(self, repo_root: Path, attempt_dir: Path) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.attempt_dir = Path(attempt_dir).resolve()
        try:
            relative = self.attempt_dir.relative_to(self.repo_root)
        except ValueError as exc:
            raise BackupRuntimeError("attempt_dir must be inside repo_root.") from exc

        prefix = Path("artifacts/runs/official")
        if relative.parts[: len(prefix.parts)] != prefix.parts:
            raise BackupRuntimeError("Active backup requires an official attempt path.")
        if len(relative.parts) != len(prefix.parts) + 2:
            raise BackupRuntimeError("Official attempt path must end in run_id/attempt_id.")

        self.run_id = relative.parts[-2]
        self.attempt_id = relative.parts[-1]
        self.runtime = validate_backup_runtime(self.repo_root)
        self.remote_root = self.runtime["backup_root"] + "/" + relative.as_posix()
        self.run_root = self.attempt_dir.parent
        self.remote_run_root = (
            self.runtime["backup_root"]
            + "/"
            + relative.parent.as_posix()
        )
        self.run_root_manifest_path = (
            self.run_root / "run_root_manifest.json"
        )
        self.manifest_path = self.attempt_dir / "backup_sync_manifest.json"

        self._verified: dict[str, dict[str, Any]] = {}
        self._last_success: str | None = None
        self._recovery_point: dict[str, Any] | None = None
        self._last_error: str | None = None
        self._status = "HEALTHY"

        self._stop = threading.Event()
        self._wake = threading.Event()
        self._sync_lock = threading.Lock()
        self._reason_lock = threading.Lock()
        self._pending_reason: str | None = None
        self._thread: threading.Thread | None = None
        self._last_periodic = time.monotonic()
        self._critical_signatures: dict[str, tuple[int, int]] = {}
        self._run_root_manifest_verified: dict[str, Any] | None = None
        self._logical_run_status_verified: dict[str, Any] | None = None

    def _copy_verify_run_root_manifest(self) -> dict[str, Any]:
        source = self.run_root_manifest_path
        if not source.is_file():
            raise BackupRuntimeError(
                "run_root_manifest.json is missing before backup."
            )

        snapshot, size_bytes, sha256 = self._snapshot(source)
        try:
            previous = self._run_root_manifest_verified
            if previous is not None:
                if (
                    previous.get("sha256") != sha256
                    or previous.get("size_bytes") != size_bytes
                ):
                    raise BackupRuntimeError(
                        "Immutable run_root_manifest.json changed after verified backup."
                    )
                if (
                    previous.get("verification_status") == "PASS"
                    and previous.get("destination")
                    == self.remote_run_root + "/run_root_manifest.json"
                ):
                    return previous

            remote = self.remote_run_root + "/run_root_manifest.json"
            _run(
                [
                    self.runtime["rclone"],
                    "--config",
                    self.runtime["rclone_conf"],
                    "copyto",
                    str(snapshot),
                    remote,
                ],
                "run_root_manifest backup copy",
            )

            listing = _run(
                [
                    self.runtime["rclone"],
                    "--config",
                    self.runtime["rclone_conf"],
                    "lsf",
                    remote,
                ],
                "run_root_manifest remote presence check",
            )
            if "run_root_manifest.json" not in {
                line.strip()
                for line in listing.stdout.splitlines()
                if line.strip()
            }:
                raise BackupRuntimeError(
                    "Remote run_root_manifest.json is missing."
                )

            fd, verify_name = tempfile.mkstemp(
                dir=str(self.attempt_dir / ".backup_tmp"),
                prefix="run-root-manifest.roundtrip.",
                suffix=".json",
            )
            os.close(fd)
            verify = Path(verify_name)
            try:
                verify.unlink()
                _run(
                    [
                        self.runtime["rclone"],
                        "--config",
                        self.runtime["rclone_conf"],
                        "copyto",
                        remote,
                        str(verify),
                    ],
                    "run_root_manifest round-trip restore",
                )
                if (
                    verify.stat().st_size != size_bytes
                    or _sha256(verify) != sha256
                ):
                    raise BackupRuntimeError(
                        "run_root_manifest remote checksum/byte verification failed."
                    )
            finally:
                if verify.exists():
                    verify.unlink()

            entry = {
                "path": "run_root_manifest.json",
                "source_scope": "LOGICAL_RUN_ROOT",
                "source_path": str(source),
                "destination": remote,
                "size_bytes": size_bytes,
                "sha256": sha256,
                "copy_timestamp_utc": _utc_now(),
                "verification_status": "PASS",
            }
            self._run_root_manifest_verified = entry
            return entry
        finally:
            if snapshot.exists():
                snapshot.unlink()

    def _copy_verify_logical_run_status(self) -> dict[str, Any] | None:
        source = self.run_root / "logical_run_status.json"
        if not source.exists():
            return None
        if not source.is_file():
            raise BackupRuntimeError(
                "logical_run_status.json path is not a file."
            )

        snapshot, size_bytes, sha256 = self._snapshot(source)
        try:
            previous = self._logical_run_status_verified
            if (
                previous is not None
                and previous.get("sha256") == sha256
                and previous.get("size_bytes") == size_bytes
                and previous.get("verification_status") == "PASS"
            ):
                return previous

            remote = self.remote_run_root + "/logical_run_status.json"
            _run(
                [
                    self.runtime["rclone"],
                    "--config",
                    self.runtime["rclone_conf"],
                    "copyto",
                    str(snapshot),
                    remote,
                ],
                "logical_run_status backup copy",
            )

            fd, verify_name = tempfile.mkstemp(
                dir=str(self.attempt_dir / ".backup_tmp"),
                prefix="logical-run-status.roundtrip.",
                suffix=".json",
            )
            os.close(fd)
            verify = Path(verify_name)
            try:
                verify.unlink()
                _run(
                    [
                        self.runtime["rclone"],
                        "--config",
                        self.runtime["rclone_conf"],
                        "copyto",
                        remote,
                        str(verify),
                    ],
                    "logical_run_status round-trip restore",
                )
                if (
                    verify.stat().st_size != size_bytes
                    or _sha256(verify) != sha256
                ):
                    raise BackupRuntimeError(
                        "logical_run_status remote checksum/byte verification failed."
                    )
            finally:
                if verify.exists():
                    verify.unlink()

            entry = {
                "path": "logical_run_status.json",
                "source_scope": "LOGICAL_RUN_ROOT",
                "source_path": str(source),
                "destination": remote,
                "size_bytes": size_bytes,
                "sha256": sha256,
                "copy_timestamp_utc": _utc_now(),
                "verification_status": "PASS",
            }
            self._logical_run_status_verified = entry
            return entry
        finally:
            if snapshot.exists():
                snapshot.unlink()

    def _required_from_artifact_manifest(self) -> set[str]:
        path = self.attempt_dir / "artifact_manifest.json"
        if not path.is_file():
            return set()
        try:
            artifacts = _load_json(path).get("artifacts", [])
        except BackupRuntimeError:
            return set()

        result: set[str] = set()
        if isinstance(artifacts, list):
            for item in artifacts:
                if not isinstance(item, dict) or item.get("required") is not True:
                    continue
                raw = item.get("path")
                if not isinstance(raw, str) or not raw:
                    continue
                rel = Path(raw)
                if rel.is_absolute() or ".." in rel.parts:
                    continue
                if rel.as_posix() != "backup_sync_manifest.json":
                    result.add(rel.as_posix())
        return result

    def _existing_files(self) -> list[tuple[str, Path]]:
        candidates = set(BASE_PATHS) | self._required_from_artifact_manifest()
        candidates.discard("backup_sync_manifest.json")
        return [
            (rel, self.attempt_dir / rel)
            for rel in sorted(candidates)
            if (self.attempt_dir / rel).is_file()
        ]

    def _snapshot(self, source: Path) -> tuple[Path, int, str]:
        temp_dir = self.attempt_dir / ".backup_tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=str(temp_dir), prefix="snapshot.", suffix=".bin")
        os.close(fd)
        path = Path(name)
        try:
            shutil.copyfile(source, path)
            return path, path.stat().st_size, _sha256(path)
        except Exception:
            if path.exists():
                path.unlink()
            raise

    def _remote(self, relative: str) -> str:
        return self.remote_root + "/" + relative.replace("\\", "/")

    def _copy_verify(self, relative: str, source: Path) -> dict[str, Any]:
        snapshot, size_bytes, sha256 = self._snapshot(source)
        try:
            previous = self._verified.get(relative)
            if (
                previous
                and previous.get("sha256") == sha256
                and previous.get("size_bytes") == size_bytes
                and previous.get("verification_status") == "PASS"
            ):
                return previous

            remote = self._remote(relative)
            _run(
                [
                    self.runtime["rclone"], "--config", self.runtime["rclone_conf"],
                    "copyto", str(snapshot), remote,
                ],
                f"backup copy {relative}",
            )
            listing = _run(
                [
                    self.runtime["rclone"], "--config", self.runtime["rclone_conf"],
                    "lsf", remote,
                ],
                f"remote presence check {relative}",
            )
            if Path(relative).name not in {
                line.strip() for line in listing.stdout.splitlines() if line.strip()
            }:
                raise BackupRuntimeError(f"Remote object missing: {remote}")

            fd, verify_name = tempfile.mkstemp(
                dir=str(self.attempt_dir / ".backup_tmp"),
                prefix="roundtrip.",
                suffix=".bin",
            )
            os.close(fd)
            verify = Path(verify_name)
            try:
                verify.unlink()
                _run(
                    [
                        self.runtime["rclone"], "--config", self.runtime["rclone_conf"],
                        "copyto", remote, str(verify),
                    ],
                    f"round-trip restore {relative}",
                )
                if verify.stat().st_size != size_bytes or _sha256(verify) != sha256:
                    raise BackupRuntimeError(
                        f"Remote checksum/byte verification failed: {relative}"
                    )
            finally:
                if verify.exists():
                    verify.unlink()

            entry = {
                "path": relative,
                "destination": remote,
                "size_bytes": size_bytes,
                "sha256": sha256,
                "copy_timestamp_utc": _utc_now(),
                "verification_status": "PASS",
            }
            self._verified[relative] = entry
            return entry
        finally:
            if snapshot.exists():
                snapshot.unlink()

    def _update_recovery_point(self) -> None:
        latest_rel = "checkpoints/latest_resume.pth"
        index_rel = "checkpoints/checkpoint_index.json"
        latest = self._verified.get(latest_rel)
        index = self._verified.get(index_rel)
        if not latest or not index:
            return

        latest_path = self.attempt_dir / latest_rel
        index_path = self.attempt_dir / index_rel
        if not latest_path.is_file() or not index_path.is_file():
            return
        if _sha256(latest_path) != latest.get("sha256"):
            return
        if _sha256(index_path) != index.get("sha256"):
            return

        update = None
        try:
            for entry in _load_json(index_path).get("entries", []):
                if (
                    isinstance(entry, dict)
                    and entry.get("role") == "LATEST_RESUME"
                    and entry.get("path") == "latest_resume.pth"
                    and entry.get("sha256") == latest.get("sha256")
                ):
                    candidate_update = entry.get("optimizer_update")
                    if (
                        isinstance(candidate_update, bool)
                        or not isinstance(candidate_update, int)
                        or candidate_update < 0
                    ):
                        return
                    update = candidate_update
                    break
        except BackupRuntimeError:
            return

        if update is None:
            return

        self._recovery_point = {
            "path": latest_rel,
            "sha256": latest["sha256"],
            "size_bytes": latest["size_bytes"],
            "optimizer_update": update,
            "checkpoint_index_sha256": index["sha256"],
            "verified_at_utc": _utc_now(),
        }

    def _manifest(self, reason: str, verification_status: str) -> dict[str, Any]:
        files = [self._verified[key] for key in sorted(self._verified)]
        return {
            "schema_version": "1.1",
            "artifact_type": "PER_ATTEMPT_BACKUP_SYNC_MANIFEST",
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "source_path": str(self.attempt_dir),
            "destination_identity": self.remote_root,
            "run_root_manifest": self._run_root_manifest_verified,
            "logical_run_status": self._logical_run_status_verified,
            "file_count": len(files),
            "bytes": sum(int(item["size_bytes"]) for item in files),
            "files": files,
            "copy_timestamp": self._last_success,
            "verification_status": verification_status,
            "BACKUP_SYNC_INTERVAL_MINUTES": BACKUP_SYNC_INTERVAL_MINUTES,
            "LAST_SUCCESSFUL_BACKUP_SYNC": self._last_success,
            "LATEST_VERIFIED_DURABLE_RECOVERY_POINT": self._recovery_point,
            "BACKUP_WORKER_STATUS": self._status,
            "last_sync_reason": reason,
            "last_error": self._last_error,
            "credentials_embedded_in_evidence": False,
        }

    def _write_verify_manifest(self, reason: str) -> None:
        _write_json_atomic(self.manifest_path, self._manifest(reason, "PASS"))
        snapshot, size_bytes, sha256 = self._snapshot(self.manifest_path)
        try:
            remote = self._remote("backup_sync_manifest.json")
            _run(
                [
                    self.runtime["rclone"], "--config", self.runtime["rclone_conf"],
                    "copyto", str(snapshot), remote,
                ],
                "backup_sync_manifest copy",
            )
            fd, verify_name = tempfile.mkstemp(
                dir=str(self.attempt_dir / ".backup_tmp"),
                prefix="manifest.roundtrip.",
                suffix=".json",
            )
            os.close(fd)
            verify = Path(verify_name)
            try:
                verify.unlink()
                _run(
                    [
                        self.runtime["rclone"], "--config", self.runtime["rclone_conf"],
                        "copyto", remote, str(verify),
                    ],
                    "backup_sync_manifest round-trip restore",
                )
                if verify.stat().st_size != size_bytes or _sha256(verify) != sha256:
                    raise BackupRuntimeError(
                        "backup_sync_manifest remote verification failed."
                    )
            finally:
                if verify.exists():
                    verify.unlink()
        finally:
            if snapshot.exists():
                snapshot.unlink()

    def sync(self, reason: str, *, raise_on_error: bool) -> bool:
        with self._sync_lock:
            try:
                self._status = "HEALTHY"
                self._last_error = None
                self._copy_verify_run_root_manifest()
                self._copy_verify_logical_run_status()
                for relative, source in self._existing_files():
                    self._copy_verify(relative, source)
                self._update_recovery_point()
                self._last_success = _utc_now()
                self._write_verify_manifest(reason)
                self._refresh_signatures()
                return True
            except Exception as exc:
                self._status = "DEGRADED"
                self._last_error = f"{type(exc).__name__}: {exc}"
                try:
                    _write_json_atomic(
                        self.manifest_path, self._manifest(reason, "FAIL")
                    )
                except Exception:
                    pass
                if raise_on_error:
                    if isinstance(exc, BackupRuntimeError):
                        raise
                    raise BackupRuntimeError(str(exc)) from exc
                return False

    def _signature(self, relative: str) -> tuple[int, int] | None:
        path = self.attempt_dir / relative
        if not path.is_file():
            return None
        meta = path.stat()
        return (meta.st_size, meta.st_mtime_ns)

    def _refresh_signatures(self) -> None:
        self._critical_signatures = {
            rel: signature
            for rel in CRITICAL_PATHS
            if (signature := self._signature(rel)) is not None
        }

    def _critical_changed(self) -> bool:
        current = {
            rel: signature
            for rel in CRITICAL_PATHS
            if (signature := self._signature(rel)) is not None
        }
        return current != self._critical_signatures

    def signal(self, reason: str) -> None:
        with self._reason_lock:
            self._pending_reason = str(reason)
        self._wake.set()

    def _consume_reason(self) -> str:
        with self._reason_lock:
            reason = self._pending_reason or "critical_artifact_change"
            self._pending_reason = None
        return reason

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=BACKUP_EVENT_POLL_SECONDS)
            self._wake.clear()
            if self._stop.is_set():
                return

            periodic = (
                time.monotonic() - self._last_periodic
                >= BACKUP_SYNC_INTERVAL_SECONDS
            )
            event = self._pending_reason is not None
            critical = self._critical_changed()
            if not (periodic or event or critical):
                continue

            reason = (
                self._consume_reason()
                if event
                else "critical_artifact_change"
                if critical
                else "scheduled_15_minute"
            )
            if self.sync(reason, raise_on_error=False):
                self._last_periodic = time.monotonic()

    def start(self) -> None:
        if self._thread is not None:
            raise BackupRuntimeError("Active backup worker already started.")
        self.sync("official_attempt_start", raise_on_error=True)
        self._last_periodic = time.monotonic()
        self._thread = threading.Thread(
            target=self._loop,
            name=f"s9-backup-{self.run_id}-{self.attempt_id}",
            daemon=True,
        )
        self._thread.start()

    def finalize(self) -> None:
        self.stop()
        self.sync("run_completion_final_forced_sync", raise_on_error=True)

    def record_failure_state(self) -> None:
        self.sync("technical_failure", raise_on_error=False)

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
            self._thread = None
        temp_dir = self.attempt_dir / ".backup_tmp"
        if temp_dir.is_dir():
            try:
                temp_dir.rmdir()
            except OSError:
                pass


_MANAGERS: dict[str, ActiveBackupManager] = {}
_MANAGERS_LOCK = threading.Lock()


def _key(attempt_dir: str | Path) -> str:
    return str(Path(attempt_dir).resolve())


def register_active_backup(manager: ActiveBackupManager) -> None:
    key = _key(manager.attempt_dir)
    with _MANAGERS_LOCK:
        if key in _MANAGERS:
            raise BackupRuntimeError(
                f"Active backup manager already registered: {key}"
            )
        _MANAGERS[key] = manager


def unregister_active_backup(attempt_dir: str | Path) -> None:
    with _MANAGERS_LOCK:
        _MANAGERS.pop(_key(attempt_dir), None)


def signal_active_backup(attempt_dir: str | Path, reason: str) -> None:
    key = _key(attempt_dir)
    with _MANAGERS_LOCK:
        manager = _MANAGERS.get(key)
    if manager is None:
        raise BackupRuntimeError(
            f"No active backup manager registered for {key}"
        )
    manager.signal(reason)


def _safe_relative_path(raw: str) -> str:
    rel = Path(raw)
    if not raw or rel.is_absolute() or ".." in rel.parts:
        raise BackupRuntimeError(f"Unsafe backup path: {raw!r}")
    return rel.as_posix()


def _download_verified_file(
    *,
    runtime: dict[str, str],
    remote: str,
    destination: Path,
    expected_size: int,
    expected_sha256: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(destination.parent),
        prefix=f".{destination.name}.recovery.",
        suffix=".tmp",
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        if tmp.exists():
            tmp.unlink()
        _run(
            [
                runtime["rclone"],
                "--config",
                runtime["rclone_conf"],
                "copyto",
                remote,
                str(tmp),
            ],
            f"recovery download {remote}",
        )
        if not tmp.is_file():
            raise BackupRuntimeError(f"Recovery download missing: {remote}")
        if tmp.stat().st_size != expected_size:
            raise BackupRuntimeError(
                f"Recovery byte-count mismatch: {remote}"
            )
        if _sha256(tmp) != expected_sha256:
            raise BackupRuntimeError(
                f"Recovery SHA-256 mismatch: {remote}"
            )
        os.replace(tmp, destination)
    finally:
        if tmp.exists():
            tmp.unlink()


def recover_official_attempt(
    repo_root: Path,
    run_id: str,
    attempt_id: str,
) -> dict[str, Any]:
    """Rehydrate one same-attempt official run from verified durable backup."""
    repo_root = Path(repo_root).resolve()
    run_id = str(run_id).strip()
    attempt_id = str(attempt_id).strip()
    if not run_id or not attempt_id:
        raise BackupRuntimeError("run_id and attempt_id are required.")

    attempt_dir = (
        repo_root
        / "artifacts"
        / "runs"
        / "official"
        / run_id
        / attempt_id
    ).resolve()
    try:
        attempt_dir.relative_to(repo_root)
    except ValueError as exc:
        raise BackupRuntimeError(
            "Recovery attempt path escapes repo_root."
        ) from exc

    if attempt_dir.exists():
        raise BackupRuntimeError(
            "Recovery requires a clean local attempt path: "
            + str(attempt_dir)
        )

    runtime = validate_backup_runtime(repo_root)
    relative_attempt = (
        Path("artifacts/runs/official") / run_id / attempt_id
    )
    remote_root = (
        runtime["backup_root"]
        + "/"
        + relative_attempt.as_posix()
    )

    manifest_remote = remote_root + "/backup_sync_manifest.json"
    fd, manifest_name = tempfile.mkstemp(
        prefix="s9.recovery.manifest.",
        suffix=".json",
    )
    os.close(fd)
    manifest_tmp = Path(manifest_name)

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
            "recovery manifest download",
        )
        manifest = _load_json(manifest_tmp)

        if manifest.get("verification_status") != "PASS":
            raise BackupRuntimeError(
                "Remote backup manifest is not PASS."
            )
        if (
            manifest.get("run_id") != run_id
            or manifest.get("attempt_id") != attempt_id
        ):
            raise BackupRuntimeError(
                "Remote backup manifest identity mismatch."
            )
        if manifest.get("destination_identity") != remote_root:
            raise BackupRuntimeError(
                "Remote backup destination identity mismatch."
            )

        logical_status = manifest.get("logical_run_status")
        if (
            not isinstance(logical_status, dict)
            or logical_status.get("verification_status") != "PASS"
            or logical_status.get("path") != "logical_run_status.json"
            or logical_status.get("source_scope") != "LOGICAL_RUN_ROOT"
        ):
            raise BackupRuntimeError(
                "Remote backup lacks verified logical_run_status.json."
            )
        logical_size = logical_status.get("size_bytes")
        logical_sha = logical_status.get("sha256")
        if (
            isinstance(logical_size, bool)
            or not isinstance(logical_size, int)
            or logical_size <= 0
            or not isinstance(logical_sha, str)
            or len(logical_sha) != 64
            or any(
                ch not in "0123456789abcdef"
                for ch in logical_sha.lower()
            )
        ):
            raise BackupRuntimeError(
                "Remote logical_run_status evidence is invalid."
            )
        remote_run_root = (
            runtime["backup_root"]
            + "/"
            + relative_attempt.parent.as_posix()
        )
        expected_logical_remote = (
            remote_run_root + "/logical_run_status.json"
        )
        if logical_status.get("destination") != expected_logical_remote:
            raise BackupRuntimeError(
                "Remote logical_run_status destination mismatch."
            )
        logical_status_path = (
            repo_root
            / "artifacts"
            / "runs"
            / "official"
            / run_id
            / "logical_run_status.json"
        )
        if logical_status_path.exists():
            raise BackupRuntimeError(
                "Recovery requires logical_run_status.json to be absent locally."
            )

        recovery_point = manifest.get(
            "LATEST_VERIFIED_DURABLE_RECOVERY_POINT"
        )
        if not isinstance(recovery_point, dict):
            raise BackupRuntimeError(
                "No verified durable recovery point in remote manifest."
            )
        if (
            recovery_point.get("path")
            != "checkpoints/latest_resume.pth"
        ):
            raise BackupRuntimeError(
                "Unexpected durable recovery path."
            )

        update = recovery_point.get("optimizer_update")
        if (
            isinstance(update, bool)
            or not isinstance(update, int)
            or update < 0
        ):
            raise BackupRuntimeError(
                "Invalid durable recovery optimizer_update."
            )

        files = manifest.get("files")
        if not isinstance(files, list) or not files:
            raise BackupRuntimeError(
                "Remote backup manifest has no verified files."
            )

        by_path: dict[str, dict[str, Any]] = {}
        for item in files:
            if (
                not isinstance(item, dict)
                or item.get("verification_status") != "PASS"
            ):
                raise BackupRuntimeError(
                    "Remote backup manifest contains unverified file."
                )
            rel = _safe_relative_path(
                str(item.get("path", ""))
            )
            if rel in by_path:
                raise BackupRuntimeError(
                    "Duplicate path in remote backup manifest: " + rel
                )
            size = item.get("size_bytes")
            sha = item.get("sha256")
            if (
                isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
            ):
                raise BackupRuntimeError(
                    "Invalid recovery file size in manifest."
                )
            if (
                not isinstance(sha, str)
                or len(sha) != 64
                or any(ch not in "0123456789abcdef" for ch in sha.lower())
            ):
                raise BackupRuntimeError(
                    "Invalid recovery SHA-256 in manifest."
                )
            by_path[rel] = item

        latest_rel = "checkpoints/latest_resume.pth"
        index_rel = "checkpoints/checkpoint_index.json"
        for required in (latest_rel, index_rel):
            if required not in by_path:
                raise BackupRuntimeError(
                    "Recovery manifest missing required file: "
                    + required
                )

        latest_entry = by_path[latest_rel]
        if (
            recovery_point.get("sha256")
            != latest_entry.get("sha256")
            or recovery_point.get("size_bytes")
            != latest_entry.get("size_bytes")
        ):
            raise BackupRuntimeError(
                "Durable recovery point does not match latest_resume manifest entry."
            )

        attempt_dir.mkdir(parents=True, exist_ok=False)
        logical_status_restored = False
        try:
            _download_verified_file(
                runtime=runtime,
                remote=expected_logical_remote,
                destination=logical_status_path,
                expected_size=int(logical_size),
                expected_sha256=str(logical_sha),
            )
            logical_status_restored = True

            for rel, item in sorted(by_path.items()):
                _download_verified_file(
                    runtime=runtime,
                    remote=remote_root + "/" + rel,
                    destination=attempt_dir / rel,
                    expected_size=int(item["size_bytes"]),
                    expected_sha256=str(item["sha256"]),
                )

            index_path = attempt_dir / index_rel
            latest_path = attempt_dir / latest_rel
            index = _load_json(index_path)

            match = None
            for entry in index.get("entries", []):
                if (
                    isinstance(entry, dict)
                    and entry.get("role") == "LATEST_RESUME"
                    and entry.get("path") == "latest_resume.pth"
                    and entry.get("sha256") == _sha256(latest_path)
                ):
                    match = entry
                    break

            if match is None:
                raise BackupRuntimeError(
                    "Restored checkpoint_index does not bind latest_resume.pth."
                )

            restored_update = match.get("optimizer_update")
            if (
                isinstance(restored_update, bool)
                or not isinstance(restored_update, int)
                or restored_update != update
            ):
                raise BackupRuntimeError(
                    "Restored recovery update does not match durable manifest."
                )

            if (
                _sha256(index_path)
                != str(
                    recovery_point.get(
                        "checkpoint_index_sha256",
                        "",
                    )
                )
            ):
                raise BackupRuntimeError(
                    "Restored checkpoint_index SHA-256 does not match durable recovery point."
                )

            if (
                _sha256(latest_path)
                != str(recovery_point.get("sha256", ""))
            ):
                raise BackupRuntimeError(
                    "Restored latest_resume SHA-256 does not match durable recovery point."
                )

            _write_json_atomic(
                attempt_dir / "backup_sync_manifest.json",
                manifest,
            )
        except Exception:
            if attempt_dir.exists():
                shutil.rmtree(attempt_dir)
            if logical_status_restored and logical_status_path.exists():
                logical_status_path.unlink()
            raise

        return {
            "run_id": run_id,
            "attempt_id": attempt_id,
            "attempt_dir": str(attempt_dir),
            "latest_resume_path": str(
                attempt_dir / latest_rel
            ),
            "optimizer_update": update,
            "remote_root": remote_root,
        }
    finally:
        if manifest_tmp.exists():
            manifest_tmp.unlink()

__all__ = [
    "ActiveBackupManager",
    "BACKUP_SYNC_INTERVAL_MINUTES",
    "BackupRuntimeError",
    "register_active_backup",
    "signal_active_backup",
    "unregister_active_backup",
    "recover_official_attempt",
    "validate_backup_runtime",
]
