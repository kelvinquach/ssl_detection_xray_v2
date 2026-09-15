"""Scientific BEST/LAST EMA Teacher checkpoint handling for SSL.

Controlled repair discovered during S5.06. Scientific SSL checkpoints are
Teacher-only artifacts:
- BEST EMA Teacher: selected by fixed-validation Teacher bbox mAP@[0.50:0.95].
- LAST EMA Teacher: terminal EMA Teacher retained for Q_pseudo.
- LATEST_RESUME remains operational only and is handled separately.
"""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from mmengine.hooks import Hook
from mmdet.registry import HOOKS

from src.utils.optimizer_update_counter import ActualOptimizerUpdateCounter
from src.utils.resume_checkpoint_hook import RUNNER_COUNTER_ATTR
from src.utils.run_manifest import file_sha256
from src.utils.training_event_logger import TrainingEventLogger


RUNTIME_PRIMARY_METRIC_KEY = "coco/bbox_mAP"
SCIENTIFIC_PRIMARY_METRIC = "bbox_mAP_50_95"
MODEL_ROLE = "EMA_TEACHER"
SERIALIZATION_SCOPE = "EMA_TEACHER_ONLY"
BEST_RULE = "ARGMAX_VALIDATION_TEACHER_BBOX_MAP_50_95"
BEST_TIE_BREAK = "STRICT_GREATER_KEEP_EARLIEST_MAX"

VALIDATION_HISTORY_FIELDS = (
    "optimizer_update",
    "bbox_mAP_50_95",
    "AP50",
    "AP75",
    "AR_50_95_max100",
    "model_role",
    "checkpoint_candidate",
    "best_so_far",
    "checkpoint_path",
)


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
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _metric_value(metrics: Dict[str, Any], key: str) -> float:
    if key not in metrics:
        raise KeyError(
            f"Validation metrics are missing required checkpoint metric {key!r}."
        )
    value = metrics[key]
    if isinstance(value, bool):
        raise TypeError(f"Metric {key!r} must be numeric, not bool.")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Metric {key!r} must be numeric.") from exc
    if not math.isfinite(value):
        raise ValueError(f"Metric {key!r} must be finite.")
    return value


@HOOKS.register_module()
class SSLBestLastCheckpointHook(Hook):
    """Materialize scientific BEST/LAST EMA Teacher checkpoints."""

    priority = "NORMAL"

    def __init__(
        self,
        metric_key: str = RUNTIME_PRIMARY_METRIC_KEY,
    ) -> None:
        if not metric_key:
            raise ValueError("metric_key must be non-empty.")
        self.metric_key = metric_key
        self._best_score: Optional[float] = None
        self._best_update: Optional[int] = None

    def _attempt_root(self, runner: Any) -> Path:
        work_dir = getattr(runner, "work_dir", None)
        if not work_dir:
            raise RuntimeError(
                "runner.work_dir is required as the official SSL attempt root."
            )
        return Path(work_dir)

    def _counter(self, runner: Any) -> ActualOptimizerUpdateCounter:
        counter = getattr(runner, RUNNER_COUNTER_ATTR, None)
        if not isinstance(counter, ActualOptimizerUpdateCounter):
            raise RuntimeError(
                "Authoritative ActualOptimizerUpdateCounter is not attached."
            )
        return counter

    def _model(self, runner: Any) -> Any:
        return getattr(runner.model, "module", runner.model)

    def _teacher(self, runner: Any) -> Any:
        model = self._model(runner)
        if not hasattr(model, "teacher") or not hasattr(model, "student"):
            raise RuntimeError(
                "SSL scientific checkpointing requires Student and Teacher."
            )
        semi_test_cfg = getattr(model, "semi_test_cfg", None)
        if semi_test_cfg is None:
            raise RuntimeError(
                "SSL scientific checkpointing requires semi_test_cfg."
            )
        predict_on = semi_test_cfg.get("predict_on", "teacher")
        if predict_on != "teacher":
            raise RuntimeError(
                "SSL BEST selection must evaluate EMA Teacher; "
                f"predict_on={predict_on!r}."
            )
        return model.teacher

    def _paths(self, runner: Any) -> Dict[str, Path]:
        root = self._attempt_root(runner)
        validation_dir = root / "metrics" / "validation"
        checkpoint_dir = root / "checkpoints"
        return {
            "validation_history": validation_dir / "validation_history.csv",
            "validation_best": validation_dir / "validation_best_teacher.json",
            "checkpoint_index": checkpoint_dir / "checkpoint_index.json",
            "best": checkpoint_dir / "best_ema_teacher.pth",
            "last": checkpoint_dir / "last_ema_teacher.pth",
            "runtime_events": root / "runtime_events.jsonl",
            "train_log": root / "train.jsonl",
        }

    def before_run(self, runner: Any) -> None:
        """Restore BEST-selection provenance when the same attempt continues."""
        paths = self._paths(runner)
        best_json = paths["validation_best"]
        best_ckpt = paths["best"]

        if not best_json.exists() and not best_ckpt.exists():
            self._best_score = None
            self._best_update = None
            return

        if best_json.exists() != best_ckpt.exists():
            raise RuntimeError(
                "SSL BEST provenance is incomplete: "
                "validation_best_teacher.json and best_ema_teacher.pth "
                "must either both exist or both be absent."
            )

        payload = json.loads(best_json.read_text(encoding="utf-8"))
        required = {
            "selection_split": "validation",
            "selection_model": MODEL_ROLE,
            "selection_metric": SCIENTIFIC_PRIMARY_METRIC,
            "rule": BEST_RULE,
            "tie_break": BEST_TIE_BREAK,
        }
        for key, expected in required.items():
            if payload.get(key) != expected:
                raise RuntimeError(
                    f"Existing SSL BEST has invalid {key}: "
                    f"{payload.get(key)!r}."
                )

        selected_update = payload.get("selected_update")
        selected_metric = payload.get("selected_metric_value")
        if (
            isinstance(selected_update, bool)
            or not isinstance(selected_update, int)
            or selected_update < 0
        ):
            raise RuntimeError(
                "Existing SSL BEST selected_update is invalid."
            )
        if isinstance(selected_metric, bool):
            raise RuntimeError(
                "Existing SSL BEST selected_metric_value is invalid."
            )
        try:
            selected_metric = float(selected_metric)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Existing SSL BEST selected_metric_value is invalid."
            ) from exc
        if not math.isfinite(selected_metric):
            raise RuntimeError(
                "Existing SSL BEST selected_metric_value is non-finite."
            )

        expected_sha = payload.get("checkpoint_sha256")
        if expected_sha != file_sha256(best_ckpt):
            raise RuntimeError(
                "Existing SSL BEST checkpoint SHA-256 does not match "
                "validation_best_teacher.json."
            )

        checkpoint = torch.load(best_ckpt, map_location="cpu")
        meta = checkpoint.get("meta", {})
        if meta.get("model_role") != MODEL_ROLE:
            raise RuntimeError(
                "Existing SSL BEST checkpoint has invalid model_role."
            )
        if meta.get("serialization_scope") != SERIALIZATION_SCOPE:
            raise RuntimeError(
                "Existing SSL BEST checkpoint has invalid serialization_scope."
            )

        self._best_score = selected_metric
        self._best_update = selected_update

    def _save_teacher_checkpoint(
        self,
        runner: Any,
        *,
        destination: Path,
        role: str,
        optimizer_update: int,
    ) -> str:
        teacher = self._teacher(runner)
        teacher_state = {
            key: value.detach().cpu()
            for key, value in teacher.state_dict().items()
        }
        if not teacher_state:
            raise RuntimeError("EMA Teacher state_dict is empty.")

        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(destination.parent),
            prefix=f".{destination.stem}.",
            suffix=".pth",
        )
        os.close(fd)

        tmp_path = Path(tmp_name)
        payload = {
            "meta": {
                "scientific_checkpoint_role": role,
                "model_role": MODEL_ROLE,
                "serialization_scope": SERIALIZATION_SCOPE,
                "optimizer_update": optimizer_update,
            },
            "state_dict": teacher_state,
        }
        try:
            with tmp_path.open("wb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, destination)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        return file_sha256(destination)

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
        entry: Dict[str, Any],
    ) -> None:
        payload = self._load_checkpoint_index(path)
        entries = [
            item
            for item in payload["entries"]
            if item.get("role") != entry["role"]
        ]
        entries.append(entry)
        role_order = {
            "BEST": 0,
            "LAST": 1,
            "LATEST_RESUME": 2,
        }
        entries.sort(
            key=lambda item: role_order.get(item.get("role"), 99)
        )
        payload["entries"] = entries
        _write_json_atomic(path, payload)

    def _append_validation_history(
        self,
        path: Path,
        row: Dict[str, Any],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists()
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=VALIDATION_HISTORY_FIELDS,
                extrasaction="raise",
            )
            if not exists:
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())

    def after_val_epoch(
        self,
        runner: Any,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._teacher(runner)
        if not metrics:
            raise RuntimeError(
                "SSL BEST selection requires non-empty validation metrics."
            )

        update = self._counter(runner).count
        score = _metric_value(metrics, self.metric_key)
        is_new_best = self._best_score is None or score > self._best_score
        paths = self._paths(runner)

        if is_new_best:
            best_sha = self._save_teacher_checkpoint(
                runner,
                destination=paths["best"],
                role="BEST",
                optimizer_update=update,
            )
            self._best_score = score
            self._best_update = update

            validation_best = {
                "selection_split": "validation",
                "selection_model": MODEL_ROLE,
                "selection_metric": SCIENTIFIC_PRIMARY_METRIC,
                "selected_update": update,
                "selected_metric_value": score,
                "checkpoint_path": "checkpoints/best_ema_teacher.pth",
                "checkpoint_sha256": best_sha,
                "rule": BEST_RULE,
                "tie_break": BEST_TIE_BREAK,
            }
            _write_json_atomic(paths["validation_best"], validation_best)

            self._upsert_checkpoint_index(
                paths["checkpoint_index"],
                {
                    "role": "BEST",
                    "model_role": MODEL_ROLE,
                    "optimizer_update": update,
                    "validation_bbox_mAP_50_95": score,
                    "path": "best_ema_teacher.pth",
                    "sha256": best_sha,
                    "retained": True,
                },
            )

            TrainingEventLogger(
                train_log_path=paths["train_log"],
                runtime_events_path=paths["runtime_events"],
            ).log_runtime_event(
                "SSL_BEST_EMA_TEACHER_UPDATED",
                optimizer_update=update,
                selection_split="validation",
                selection_model=MODEL_ROLE,
                selection_metric=SCIENTIFIC_PRIMARY_METRIC,
                validation_bbox_mAP_50_95=score,
                checkpoint_path="checkpoints/best_ema_teacher.pth",
                checkpoint_sha256=best_sha,
            )

        self._append_validation_history(
            paths["validation_history"],
            {
                "optimizer_update": update,
                "bbox_mAP_50_95": score,
                "AP50": metrics.get("coco/bbox_mAP_50", ""),
                "AP75": metrics.get("coco/bbox_mAP_75", ""),
                "AR_50_95_max100": metrics.get(
                    "coco/AR_50_95_max100",
                    "",
                ),
                "model_role": MODEL_ROLE,
                "checkpoint_candidate": is_new_best,
                "best_so_far": self._best_score,
                "checkpoint_path": "checkpoints/best_ema_teacher.pth",
            },
        )

    def after_train(self, runner: Any) -> None:
        self._teacher(runner)
        update = self._counter(runner).count
        paths = self._paths(runner)

        last_sha = self._save_teacher_checkpoint(
            runner,
            destination=paths["last"],
            role="LAST",
            optimizer_update=update,
        )
        self._upsert_checkpoint_index(
            paths["checkpoint_index"],
            {
                "role": "LAST",
                "model_role": MODEL_ROLE,
                "optimizer_update": update,
                "path": "last_ema_teacher.pth",
                "sha256": last_sha,
                "retained": True,
            },
        )

        TrainingEventLogger(
            train_log_path=paths["train_log"],
            runtime_events_path=paths["runtime_events"],
        ).log_runtime_event(
            "SSL_LAST_EMA_TEACHER_SAVED",
            optimizer_update=update,
            model_role=MODEL_ROLE,
            checkpoint_path="checkpoints/last_ema_teacher.pth",
            checkpoint_sha256=last_sha,
        )


__all__ = [
    "BEST_RULE",
    "BEST_TIE_BREAK",
    "MODEL_ROLE",
    "RUNTIME_PRIMARY_METRIC_KEY",
    "SCIENTIFIC_PRIMARY_METRIC",
    "SERIALIZATION_SCOPE",
    "SSLBestLastCheckpointHook",
    "VALIDATION_HISTORY_FIELDS",
]
