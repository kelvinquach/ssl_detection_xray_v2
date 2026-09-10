"""Operational LATEST_RESUME checkpoint handling for supervised training.

S3.13 scope:
- LATEST_RESUME is operational only and independent from scientific BEST/LAST.
- A fresh SUP attempt materializes checkpoints/latest_resume.pth at update 0.
- The rolling checkpoint is refreshed every 172 actual optimizer updates,
  after validation has fully completed.
- Resume continues the same attempt_id.
- Technical retry is a separate governance path using a new attempt_id.
- MMEngine owns model, optimizer/AMP-wrapper, scheduler, message-hub and
  raw-iteration state; ProtocolResumeCheckpointHook injects seed/counter/RNG.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

from mmengine.hooks import Hook
from mmengine.registry import HOOKS

from src.utils.optimizer_update_counter import ActualOptimizerUpdateCounter
from src.utils.resume_checkpoint_hook import (
    RUNNER_COUNTER_ATTR,
    RUNNER_PENDING_RNG_RESTORE_ATTR,
)
from src.utils.run_manifest import file_sha256
from src.utils.training_event_logger import TrainingEventLogger


LATEST_RESUME_ROLE = "LATEST_RESUME"
LATEST_RESUME_FILENAME = "latest_resume.pth"
DEFAULT_REFRESH_INTERVAL = 172


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def _write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


@HOOKS.register_module()
class SupervisedLatestResumeCheckpointHook(Hook):
    """Materialize and provenance-track operational SUP LATEST_RESUME."""

    priority = "NORMAL"

    def __init__(self, refresh_interval: int = DEFAULT_REFRESH_INTERVAL) -> None:
        if (
            isinstance(refresh_interval, bool)
            or not isinstance(refresh_interval, int)
            or refresh_interval <= 0
        ):
            raise ValueError("refresh_interval must be a positive integer.")
        self.refresh_interval = refresh_interval

    def _attempt_root(self, runner: Any) -> Path:
        work_dir = getattr(runner, "work_dir", None)
        if not work_dir:
            raise RuntimeError(
                "runner.work_dir is required as the attempt root."
            )
        return Path(work_dir)

    def _counter(self, runner: Any) -> ActualOptimizerUpdateCounter:
        counter = getattr(runner, RUNNER_COUNTER_ATTR, None)
        if not isinstance(counter, ActualOptimizerUpdateCounter):
            raise RuntimeError(
                "Authoritative ActualOptimizerUpdateCounter is not attached."
            )
        return counter

    def _paths(self, runner: Any) -> Dict[str, Path]:
        root = self._attempt_root(runner)
        checkpoint_dir = root / "checkpoints"
        return {
            "root": root,
            "checkpoint_dir": checkpoint_dir,
            "latest_resume": checkpoint_dir / LATEST_RESUME_FILENAME,
            "checkpoint_index": checkpoint_dir / "checkpoint_index.json",
            "runtime_events": root / "runtime_events.jsonl",
            "train_log": root / "train.jsonl",
        }

    def _load_checkpoint_index(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {
                "checkpoint_policy": "LOCKED",
                "entries": [],
            }

        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("checkpoint_policy") != "LOCKED":
            raise RuntimeError(
                "Existing checkpoint_index.json has invalid checkpoint_policy."
            )
        if not isinstance(payload.get("entries"), list):
            raise RuntimeError(
                "Existing checkpoint_index.json entries must be a list."
            )
        return payload

    def _upsert_checkpoint_index(
        self,
        path: Path,
        *,
        optimizer_update: int,
        sha256: str,
    ) -> None:
        payload = self._load_checkpoint_index(path)

        entries = [
            item
            for item in payload["entries"]
            if item.get("role") != LATEST_RESUME_ROLE
        ]
        entries.append(
            {
                "role": LATEST_RESUME_ROLE,
                "optimizer_update": optimizer_update,
                "path": LATEST_RESUME_FILENAME,
                "sha256": sha256,
                "retained": True,
            }
        )

        role_order = {
            "BEST": 0,
            "LAST": 1,
            LATEST_RESUME_ROLE: 2,
        }
        entries.sort(
            key=lambda item: role_order.get(item.get("role"), 99)
        )
        payload["entries"] = entries
        _write_json_atomic(path, payload)

    def _save_latest_resume(
        self,
        runner: Any,
        *,
        reason: str,
    ) -> str:
        paths = self._paths(runner)
        destination = paths["latest_resume"]
        destination.parent.mkdir(parents=True, exist_ok=True)

        update = self._counter(runner).count
        raw_iteration = int(runner.iter)

        fd, tmp_name = tempfile.mkstemp(
            dir=str(destination.parent),
            prefix=f".{destination.stem}.",
            suffix=".pth",
        )
        os.close(fd)
        os.remove(tmp_name)

        tmp_path = Path(tmp_name)
        try:
            runner.save_checkpoint(
                out_dir=str(destination.parent),
                filename=tmp_path.name,
                save_optimizer=True,
                save_param_scheduler=True,
                meta={
                    # Runner.save_checkpoint(by_epoch=False) otherwise
                    # defaults to runner.iter + 1. At S3.13 save points,
                    # runner.iter already equals consumed raw iterations.
                    "iter": raw_iteration,
                },
                by_epoch=False,
            )
            if not tmp_path.is_file():
                raise RuntimeError(
                    f"Checkpoint save did not materialize {tmp_path}."
                )
            os.replace(tmp_path, destination)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        checkpoint_sha = file_sha256(destination)

        self._upsert_checkpoint_index(
            paths["checkpoint_index"],
            optimizer_update=update,
            sha256=checkpoint_sha,
        )

        TrainingEventLogger(
            train_log_path=paths["train_log"],
            runtime_events_path=paths["runtime_events"],
        ).log_runtime_event(
            "RESUME_SAVE",
            optimizer_update=update,
            raw_iteration=raw_iteration,
            checkpoint_path="checkpoints/latest_resume.pth",
            checkpoint_sha256=checkpoint_sha,
            reason=reason,
        )

        return checkpoint_sha

    def before_train(self, runner: Any) -> None:
        """Create the initial update-0 checkpoint only for a fresh attempt."""
        counter = self._counter(runner)

        pending_restore = bool(
            getattr(
                runner,
                RUNNER_PENDING_RNG_RESTORE_ATTR,
                False,
            )
        )

        if pending_restore:
            # A resumed attempt must first complete delayed RNG restoration.
            # Never overwrite its source checkpoint before that happens.
            return

        if counter.count != 0 or int(runner.iter) != 0:
            return

        self._save_latest_resume(
            runner,
            reason="initial_update_0",
        )

    def after_val(self, runner: Any) -> None:
        """Refresh only after a completed locked validation boundary."""
        update = self._counter(runner).count

        if update <= 0 or update % self.refresh_interval != 0:
            return

        self._save_latest_resume(
            runner,
            reason="post_validation_boundary",
        )


__all__ = [
    "DEFAULT_REFRESH_INTERVAL",
    "LATEST_RESUME_FILENAME",
    "LATEST_RESUME_ROLE",
    "SupervisedLatestResumeCheckpointHook",
]