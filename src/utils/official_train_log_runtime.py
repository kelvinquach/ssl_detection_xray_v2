"""Shared runtime helpers for official per-update train.jsonl production."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from src.utils.training_event_logger import TrainingEventLogger


class OfficialTrainLogRuntime:
    """Resume-safe observed runtime state for official train.jsonl logging."""

    def __init__(self, runner: Any, restored_optimizer_updates: int) -> None:
        self.work_dir = Path(runner.work_dir)
        self.train_log_path = self.work_dir / "train.jsonl"
        self.runtime_events_path = self.work_dir / "runtime_events.jsonl"
        self.logger = TrainingEventLogger(
            train_log_path=self.train_log_path,
            runtime_events_path=self.runtime_events_path,
        )

        if (
            isinstance(restored_optimizer_updates, bool)
            or not isinstance(restored_optimizer_updates, int)
            or restored_optimizer_updates < 0
        ):
            raise RuntimeError(
                "OFFICIAL_TRAIN_LOG_INVALID_RESTORED_UPDATE_COUNT"
            )

        records = self._read_records()

        if restored_optimizer_updates == 0:
            if records:
                raise RuntimeError(
                    "OFFICIAL_TRAIN_LOG_FRESH_RUN_ALREADY_HAS_RECORDS"
                )
            self._elapsed_offset = 0.0
        else:
            if len(records) < restored_optimizer_updates:
                raise RuntimeError(
                    "OFFICIAL_TRAIN_LOG_BEHIND_RESTORED_CHECKPOINT"
                )

            for index, record in enumerate(records, start=1):
                if int(record.get("optimizer_update", -1)) != index:
                    raise RuntimeError(
                        "OFFICIAL_TRAIN_LOG_NONCONTIGUOUS_EXISTING_SEQUENCE"
                    )

            elapsed_values = []
            for record in records:
                value = record.get("elapsed_seconds")
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or float(value) < 0.0
                ):
                    raise RuntimeError(
                        "OFFICIAL_TRAIN_LOG_INVALID_EXISTING_ELAPSED_SECONDS"
                    )
                elapsed_values.append(float(value))

            if len(records) > restored_optimizer_updates:
                records = records[:restored_optimizer_updates]
                self._rewrite_records(records)

            self._elapsed_offset = float(
                records[-1]["elapsed_seconds"]
            )

        self._segment_started = time.monotonic()

    def _read_records(self) -> list[dict[str, Any]]:
        if not self.train_log_path.is_file():
            return []

        records = []
        with self.train_log_path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                text = raw.strip()
                if not text:
                    continue
                try:
                    record = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        "OFFICIAL_TRAIN_LOG_INVALID_JSONL_AT_LINE_"
                        + str(line_number)
                    ) from exc
                if not isinstance(record, dict):
                    raise RuntimeError(
                        "OFFICIAL_TRAIN_LOG_RECORD_MUST_BE_OBJECT"
                    )
                records.append(record)
        return records

    def _rewrite_records(self, records: list[dict[str, Any]]) -> None:
        temp = self.train_log_path.with_suffix(".jsonl.tmp")
        temp.parent.mkdir(parents=True, exist_ok=True)
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")
        temp.replace(self.train_log_path)

    def elapsed_seconds(self) -> float:
        return float(
            self._elapsed_offset
            + (time.monotonic() - self._segment_started)
        )

    @staticmethod
    def learning_rate(runner: Any) -> float:
        optimizer = getattr(runner.optim_wrapper, "optimizer", None)
        groups = getattr(optimizer, "param_groups", None)
        if not isinstance(groups, list) or not groups:
            raise RuntimeError(
                "OFFICIAL_TRAIN_LOG_OPTIMIZER_PARAM_GROUPS_UNAVAILABLE"
            )

        values = []
        for group in groups:
            if not isinstance(group, dict) or "lr" not in group:
                raise RuntimeError(
                    "OFFICIAL_TRAIN_LOG_PARAM_GROUP_LR_MISSING"
                )
            value = float(group["lr"])
            if not math.isfinite(value) or value < 0.0:
                raise RuntimeError(
                    "OFFICIAL_TRAIN_LOG_INVALID_LEARNING_RATE"
                )
            values.append(value)

        reference = values[0]
        if any(
            not math.isclose(
                value,
                reference,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            for value in values[1:]
        ):
            raise RuntimeError(
                "OFFICIAL_TRAIN_LOG_MULTIPLE_DISTINCT_LEARNING_RATES"
            )

        return float(reference)

    @staticmethod
    def gpu_memory_allocated_mb() -> float:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "OFFICIAL_TRAIN_LOG_CUDA_UNAVAILABLE"
            )
        return float(torch.cuda.memory_allocated() / (1024.0 * 1024.0))

    @staticmethod
    def scalar(value: Any, name: str) -> float:
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise RuntimeError(
                    f"OFFICIAL_TRAIN_LOG_NONSCALAR_OUTPUT:{name}"
                )
            value = value.detach().item()

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(
                f"OFFICIAL_TRAIN_LOG_NONNUMERIC_OUTPUT:{name}"
            )

        result = float(value)
        if not math.isfinite(result):
            raise RuntimeError(
                f"OFFICIAL_TRAIN_LOG_NONFINITE_OUTPUT:{name}"
            )
        return result

    @classmethod
    def output_value(
        cls,
        outputs: Any,
        key: str,
    ) -> float:
        if not isinstance(outputs, dict):
            raise RuntimeError(
                "OFFICIAL_TRAIN_LOG_OUTPUTS_MUST_BE_DICT"
            )
        if key not in outputs:
            raise RuntimeError(
                f"OFFICIAL_TRAIN_LOG_OUTPUT_KEY_MISSING:{key}"
            )
        return cls.scalar(outputs[key], key)

    @classmethod
    def loss_sum_by_prefix(
        cls,
        outputs: Any,
        prefix: str,
    ) -> float:
        if not isinstance(outputs, dict):
            raise RuntimeError(
                "OFFICIAL_TRAIN_LOG_OUTPUTS_MUST_BE_DICT"
            )

        keys = [
            key
            for key in outputs
            if key.startswith(prefix) and "loss" in key
        ]
        if not keys:
            raise RuntimeError(
                f"OFFICIAL_TRAIN_LOG_LOSS_PREFIX_MISSING:{prefix}"
            )

        return float(
            sum(cls.scalar(outputs[key], key) for key in keys)
        )


__all__ = ["OfficialTrainLogRuntime"]
