"""Training and runtime event JSONL logging.

S2.12 scope:
- ``train.jsonl`` contains exactly one record per actual optimizer update.
- ``runtime_events.jsonl`` contains discrete runtime events.
- AMP-skipped optimizer attempts must not appear as optimizer-update records.
- SSL-specific train fields are supported without implementing SSL/EMA itself.

Actual optimizer-update detection remains the responsibility of
``ActualOptimizerUpdateCounter`` from S2.10. This logger records observed
facts; it does not maintain a second optimizer-update counter.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


SUP_TRAIN_REQUIRED_FIELDS = (
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

SSL_TRAIN_ADDITIONAL_FIELDS = (
    "loss_unsupervised",
    "pseudo_count",
    "classification_pseudo_count",
    "regression_pseudo_count",
    "unlabeled_microbatches_consumed_cumulative",
    "unlabeled_images_consumed_cumulative",
    "ema_step_executed",
)


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp using the ``Z`` suffix."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _missing_fields(
    record: Dict[str, Any],
    required: Iterable[str],
) -> list[str]:
    return [key for key in required if key not in record]


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Append one JSON object as one UTF-8 JSONL record."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                record,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        handle.write("\n")
        handle.flush()


class TrainingEventLogger:
    """Write contract-aligned ``train.jsonl`` and runtime events."""

    def __init__(
        self,
        train_log_path: Path | str,
        runtime_events_path: Path | str,
    ) -> None:
        self.train_log_path = Path(train_log_path)
        self.runtime_events_path = Path(runtime_events_path)

    def log_optimizer_update(
        self,
        record: Dict[str, Any],
        *,
        ssl: bool = False,
    ) -> None:
        """Append one actual optimizer-update record.

        ``train.jsonl`` is not an attempted-step log. Therefore
        ``optimizer_step_executed`` must be exactly ``True``.

        Args:
            record: Training record containing the contract fields.
            ssl: Require the additional SSL fields when ``True``.

        Raises:
            ValueError: If required fields are missing or the record does not
                represent an actual optimizer update.
        """
        required = list(SUP_TRAIN_REQUIRED_FIELDS)

        if ssl:
            required.extend(SSL_TRAIN_ADDITIONAL_FIELDS)

        missing = _missing_fields(record, required)
        if missing:
            raise ValueError(
                "Missing required train.jsonl fields: "
                + ", ".join(missing)
            )

        if record["optimizer_step_executed"] is not True:
            raise ValueError(
                "train.jsonl accepts only actual optimizer-update events; "
                "AMP-skipped attempts belong in runtime_events.jsonl."
            )

        if ssl and record["ema_step_executed"] is not True:
            raise ValueError(
                "An SSL train.jsonl actual-update record must report "
                "ema_step_executed=True under the locked Main protocol."
            )

        _append_jsonl(self.train_log_path, record)

    def log_runtime_event(
        self,
        event: str,
        *,
        timestamp_utc: Optional[str] = None,
        **fields: Any,
    ) -> None:
        """Append one runtime event record."""
        if not event:
            raise ValueError("Runtime event name must be non-empty.")

        record: Dict[str, Any] = {
            "timestamp_utc": timestamp_utc or utc_timestamp(),
            "event": event,
        }
        record.update(fields)

        _append_jsonl(self.runtime_events_path, record)

    def log_amp_skip(
        self,
        *,
        optimizer_update_before: int,
        reason: str,
        amp_scale_before: Optional[float] = None,
        amp_scale_after: Optional[float] = None,
        ssl: bool = False,
    ) -> None:
        """Record an AMP-skipped optimizer attempt.

        For SSL, synchronized EMA-skip evidence is emitted as a separate
        ``EMA_STEP_SKIPPED`` runtime event. This method records evidence only;
        it does not control EMA execution.
        """
        timestamp = utc_timestamp()

        common: Dict[str, Any] = {
            "optimizer_update_before": optimizer_update_before,
            "reason": reason,
            "optimizer_step_executed": False,
        }

        if amp_scale_before is not None:
            common["amp_scale_before"] = amp_scale_before

        if amp_scale_after is not None:
            common["amp_scale_after"] = amp_scale_after

        if ssl:
            common["ema_step_executed"] = False

        self.log_runtime_event(
            "OPTIMIZER_STEP_SKIPPED_BY_AMP",
            timestamp_utc=timestamp,
            **common,
        )

        if ssl:
            self.log_runtime_event(
                "EMA_STEP_SKIPPED",
                timestamp_utc=timestamp,
                optimizer_update_before=optimizer_update_before,
                reason="optimizer_step_skipped_by_amp",
                optimizer_step_executed=False,
                ema_step_executed=False,
            )


__all__ = [
    "SUP_TRAIN_REQUIRED_FIELDS",
    "SSL_TRAIN_ADDITIONAL_FIELDS",
    "TrainingEventLogger",
    "utc_timestamp",
]
