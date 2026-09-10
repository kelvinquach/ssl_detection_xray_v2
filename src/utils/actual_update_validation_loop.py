"""Validation cadence driven by actual optimizer updates for S3.11."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from mmengine.logging import print_log
from mmengine.registry import LOOPS

from src.utils.optimizer_update_counter import ActualOptimizerUpdateCounter
from src.utils.resume_aware_loop import ResumeAwareIterBasedTrainLoop
from src.utils.resume_checkpoint_hook import (
    RUNNER_COUNTER_ATTR,
    RUNNER_PENDING_RESUME_STATE_ATTR,
    RUNNER_PENDING_RNG_RESTORE_ATTR,
)
from src.utils.seed import restore_full_rng_state
from src.utils.training_event_logger import TrainingEventLogger


RUNNER_VALIDATION_HISTORY_ATTR = "_s3_actual_update_validation_history"


@LOOPS.register_module()
class ActualUpdateValidationIterBasedTrainLoop(ResumeAwareIterBasedTrainLoop):
    """Resume-aware train loop validating on actual optimizer-update cadence.

    ``val_interval`` and ``val_begin`` are interpreted in actual optimizer
    updates, not raw dataloader iterations.

    Dynamic validation intervals are intentionally unsupported because
    MMEngine's native dynamic-interval mechanism is raw-iteration based.
    """

    def __init__(
        self,
        runner,
        dataloader,
        max_iters: int,
        val_begin: int = 1,
        val_interval: int = 172,
        dynamic_intervals: Optional[list] = None,
    ) -> None:
        if dynamic_intervals is not None:
            raise ValueError(
                "S3.11 actual-update validation does not support "
                "dynamic_intervals."
            )
        if (
            isinstance(val_interval, bool)
            or not isinstance(val_interval, int)
            or val_interval <= 0
        ):
            raise ValueError("val_interval must be a positive integer.")
        if (
            isinstance(val_begin, bool)
            or not isinstance(val_begin, int)
            or val_begin <= 0
        ):
            raise ValueError("val_begin must be a positive integer.")

        super().__init__(
            runner=runner,
            dataloader=dataloader,
            max_iters=max_iters,
            val_begin=val_begin,
            val_interval=val_interval,
            dynamic_intervals=None,
        )

    def _counter(self) -> ActualOptimizerUpdateCounter:
        counter = getattr(self.runner, RUNNER_COUNTER_ATTR, None)
        if not isinstance(counter, ActualOptimizerUpdateCounter):
            raise RuntimeError(
                "Actual optimizer-update counter is unavailable for "
                "S3.11 validation."
            )
        return counter

    def _validation_history(self) -> list:
        history = getattr(
            self.runner,
            RUNNER_VALIDATION_HISTORY_ATTR,
            None,
        )
        if history is None:
            history = []
            setattr(
                self.runner,
                RUNNER_VALIDATION_HISTORY_ATTR,
                history,
            )
        if not isinstance(history, list):
            raise RuntimeError("S3.11 validation history must be a list.")
        return history

    def run(self):
        """Launch training with validation tied to actual optimizer updates."""
        self.runner.call_hook("before_train")
        self.runner.call_hook("before_train_epoch")

        if self._iter > 0:
            print_log(
                f"Advance dataloader {self._iter} steps to skip data "
                "that has already been trained",
                logger="current",
                level=logging.WARNING,
            )
            for _ in range(self._iter):
                next(self.dataloader_iterator)

        pending_restore = bool(
            getattr(
                self.runner,
                RUNNER_PENDING_RNG_RESTORE_ATTR,
                False,
            )
        )

        if pending_restore:
            state = getattr(
                self.runner,
                RUNNER_PENDING_RESUME_STATE_ATTR,
                None,
            )
            if not isinstance(state, dict):
                raise RuntimeError(
                    "Resume RNG restoration is pending but protocol resume "
                    "state is unavailable."
                )

            rng_state = state.get("rng_state")
            if not isinstance(rng_state, dict):
                raise RuntimeError(
                    "Protocol resume state does not contain a valid rng_state."
                )

            restore_full_rng_state(rng_state)

            TrainingEventLogger(
                train_log_path=Path(self.runner.work_dir) / "train.jsonl",
                runtime_events_path=(
                    Path(self.runner.work_dir) / "runtime_events.jsonl"
                ),
            ).log_runtime_event(
                "RESUME_RESTORE",
                optimizer_update=self._counter().count,
                raw_iteration=int(self.runner.iter),
                training_seed=int(self.runner.seed),
                checkpoint_path="checkpoints/latest_resume.pth",
            )

            setattr(
                self.runner,
                RUNNER_PENDING_RNG_RESTORE_ATTR,
                False,
            )
            setattr(
                self.runner,
                RUNNER_PENDING_RESUME_STATE_ATTR,
                None,
            )

        history = self._validation_history()

        while self._iter < self._max_iters and not self.stop_training:
            self.runner.model.train()

            counter = self._counter()
            updates_before = counter.count

            data_batch = next(self.dataloader_iterator)
            self.run_iter(data_batch)

            updates_after = counter.count
            delta = updates_after - updates_before

            if delta not in (0, 1):
                raise RuntimeError(
                    f"Invalid actual-update delta during validation: {delta}"
                )

            should_validate = (
                self.runner.val_loop is not None
                and delta == 1
                and updates_after >= self.val_begin
                and updates_after % self.val_interval == 0
            )

            if should_validate:
                self.runner.val_loop.run()
                history.append(
                    {
                        "actual_optimizer_updates": updates_after,
                        "raw_iterations": self._iter,
                    }
                )

        self.runner.call_hook("after_train_epoch")
        self.runner.call_hook("after_train")

        return self.runner.model


__all__ = [
    "ActualUpdateValidationIterBasedTrainLoop",
    "RUNNER_VALIDATION_HISTORY_ATTR",
]