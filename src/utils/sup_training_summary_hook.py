"""Official supervised observed exposure accounting and training summary."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

from mmengine.hooks import Hook
from mmengine.registry import HOOKS

from src.utils.optimizer_update_counter import ActualOptimizerUpdateCounter
from src.utils.official_train_log_runtime import OfficialTrainLogRuntime
from src.utils.resume_checkpoint_hook import (
    RUNNER_COUNTER_ATTR,
    RUNNER_PENDING_RESUME_STATE_ATTR,
    RUNNER_TRAINING_SUMMARY_STATE_ATTR,
)


@HOOKS.register_module()
class SupervisedTrainingSummaryHook(Hook):
    """Record observed SUP exposure using the authoritative update counter."""

    priority = "LOW"

    def __init__(
        self,
        expected_optimizer_updates: int,
        effective_labeled_batch: int,
    ) -> None:
        if (
            isinstance(expected_optimizer_updates, bool)
            or not isinstance(expected_optimizer_updates, int)
            or expected_optimizer_updates <= 0
        ):
            raise ValueError(
                "expected_optimizer_updates must be a positive integer."
            )
        if (
            isinstance(effective_labeled_batch, bool)
            or not isinstance(effective_labeled_batch, int)
            or effective_labeled_batch <= 0
        ):
            raise ValueError(
                "effective_labeled_batch must be a positive integer."
            )

        self.expected_optimizer_updates = int(
            expected_optimizer_updates
        )
        self.effective_labeled_batch = int(
            effective_labeled_batch
        )

        self._last_updates: Optional[int] = None
        self._scale_before: Optional[float] = None
        self._labeled_images = 0
        self._amp_skipped_steps = 0
        self._train_log_runtime: Optional[OfficialTrainLogRuntime] = None

    @staticmethod
    def _counter(runner: Any) -> ActualOptimizerUpdateCounter:
        counter = getattr(runner, RUNNER_COUNTER_ATTR, None)
        if not isinstance(counter, ActualOptimizerUpdateCounter):
            raise RuntimeError(
                "SupervisedTrainingSummaryHook requires the authoritative "
                "ActualOptimizerUpdateCounter."
            )
        return counter

    @staticmethod
    def _loss_scale(runner: Any) -> Optional[float]:
        optim_wrapper = runner.optim_wrapper
        scaler = getattr(optim_wrapper, "loss_scaler", None)

        if scaler is None:
            return None

        get_scale = getattr(scaler, "get_scale", None)
        if get_scale is None:
            return None

        return float(get_scale())

    @staticmethod
    def _batch_count(data_batch: Any) -> int:
        if not isinstance(data_batch, dict):
            raise RuntimeError(
                "SupervisedTrainingSummaryHook expects dict data_batch."
            )

        inputs = data_batch.get("inputs")
        if inputs is None:
            raise RuntimeError(
                "SupervisedTrainingSummaryHook requires data_batch['inputs']."
            )

        try:
            count = len(inputs)
        except TypeError as exc:
            raise RuntimeError(
                "SupervisedTrainingSummaryHook cannot determine batch size."
            ) from exc

        if count <= 0:
            raise RuntimeError(
                "Observed supervised batch size must be positive."
            )

        return int(count)

    @staticmethod
    def _state_payload(
        labeled_images: int,
        amp_skipped_steps: int,
    ) -> dict[str, int]:
        return {
            "labeled_images_consumed_total": int(labeled_images),
            "amp_skipped_step_count": int(amp_skipped_steps),
        }

    def before_train(self, runner: Any) -> None:
        counter = self._counter(runner)
        self._last_updates = int(counter.count)

        if self._last_updates == 0:
            self._labeled_images = 0
            self._amp_skipped_steps = 0
        else:
            pending_state = getattr(
                runner,
                RUNNER_PENDING_RESUME_STATE_ATTR,
                None,
            )
            if not isinstance(pending_state, dict):
                raise RuntimeError(
                    "Resumed SUP summary requires pending protocol resume state."
                )

            summary_state = pending_state.get("training_summary_state")
            if not isinstance(summary_state, dict):
                raise RuntimeError(
                    "Resumed SUP summary is missing training_summary_state."
                )

            required = (
                "labeled_images_consumed_total",
                "amp_skipped_step_count",
            )
            for key in required:
                value = summary_state.get(key)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                ):
                    raise RuntimeError(
                        f"Invalid resumed SUP training-summary field {key!r}."
                    )

            self._labeled_images = int(
                summary_state["labeled_images_consumed_total"]
            )
            self._amp_skipped_steps = int(
                summary_state["amp_skipped_step_count"]
            )

            if (
                self._labeled_images
                != self._last_updates * self.effective_labeled_batch
            ):
                raise RuntimeError(
                    "Resumed SUP labeled exposure does not match "
                    "restored optimizer updates."
                )

        accumulative_counts = getattr(
            runner.optim_wrapper,
            "_accumulative_counts",
            None,
        )
        if accumulative_counts != 1:
            raise RuntimeError(
                "Official SUP training summary requires accumulative_counts=1."
            )

        self._train_log_runtime = OfficialTrainLogRuntime(
            runner,
            restored_optimizer_updates=self._last_updates,
        )

        setattr(
            runner,
            RUNNER_TRAINING_SUMMARY_STATE_ATTR,
            self._state_payload(
                self._labeled_images,
                self._amp_skipped_steps,
            ),
        )

    def before_train_iter(
        self,
        runner: Any,
        batch_idx: int,
        data_batch: Any = None,
    ) -> None:
        self._scale_before = self._loss_scale(runner)

    def after_train_iter(
        self,
        runner: Any,
        batch_idx: int,
        data_batch: Any = None,
        outputs: Any = None,
    ) -> None:
        if self._last_updates is None:
            raise RuntimeError(
                "SupervisedTrainingSummaryHook was not initialized."
            )

        current = int(self._counter(runner).count)
        delta = current - self._last_updates

        if delta not in (0, 1):
            raise RuntimeError(
                f"Invalid SUP actual-update delta: {delta}"
            )

        scale_after = self._loss_scale(runner)

        if delta == 1:
            labeled = self._batch_count(data_batch)

            if labeled != self.effective_labeled_batch:
                raise RuntimeError(
                    "Observed labeled batch does not match official SUP "
                    f"effective batch: {labeled} != "
                    f"{self.effective_labeled_batch}"
                )

            self._labeled_images += labeled

            runtime = self._train_log_runtime
            if not isinstance(runtime, OfficialTrainLogRuntime):
                raise RuntimeError(
                    "OFFICIAL_SUP_TRAIN_LOG_RUNTIME_NOT_INITIALIZED"
                )

            if scale_after is None:
                raise RuntimeError(
                    "OFFICIAL_SUP_AMP_SCALE_UNAVAILABLE"
                )

            loss_total = runtime.output_value(outputs, "loss")

            runtime.logger.log_optimizer_update(
                {
                    "optimizer_update": int(current),
                    "raw_iteration": int(runner.iter) + 1,
                    "learning_rate": runtime.learning_rate(runner),
                    "loss_total": loss_total,
                    "loss_supervised": loss_total,
                    "amp_scale": float(scale_after),
                    "optimizer_step_executed": True,
                    "elapsed_seconds": runtime.elapsed_seconds(),
                    "gpu_memory_allocated_mb": (
                        runtime.gpu_memory_allocated_mb()
                    ),
                    "labeled_microbatches_consumed_cumulative": (
                        int(current)
                    ),
                    "labeled_images_consumed_cumulative": int(
                        self._labeled_images
                    ),
                },
                ssl=False,
            )
        else:
            amp_skip = (
                self._scale_before is not None
                and scale_after is not None
                and scale_after < self._scale_before
            )

            if amp_skip:
                self._amp_skipped_steps += 1
            else:
                raise RuntimeError(
                    "No actual SUP optimizer update occurred, but no AMP "
                    "loss-scale decrease was observed. With "
                    "accumulative_counts=1 this is unexpected."
                )

        self._last_updates = current

        setattr(
            runner,
            RUNNER_TRAINING_SUMMARY_STATE_ATTR,
            self._state_payload(
                self._labeled_images,
                self._amp_skipped_steps,
            ),
        )

    @staticmethod
    def _validation_event_count(runner: Any) -> int:
        path = (
            Path(runner.work_dir)
            / "metrics"
            / "validation"
            / "validation_history.csv"
        )

        if not path.is_file():
            return 0

        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)

            if reader.fieldnames is None:
                raise RuntimeError(
                    "validation_history.csv is missing its header."
                )

            if "optimizer_update" not in reader.fieldnames:
                raise RuntimeError(
                    "validation_history.csv is missing optimizer_update."
                )

            return sum(1 for _ in reader)

    def after_train(self, runner: Any) -> None:
        observed_updates = int(self._counter(runner).count)
        validation_events_observed = self._validation_event_count(runner)

        summary = {
            "expected_optimizer_updates":
                self.expected_optimizer_updates,
            "observed_optimizer_updates":
                observed_updates,
            "effective_labeled_batch":
                self.effective_labeled_batch,
            "labeled_images_consumed_total":
                int(self._labeled_images),
            "validation_events_observed":
                validation_events_observed,
            "amp_skipped_step_count":
                int(self._amp_skipped_steps),
            "labeled_exposure_matches_update_count": (
                self._labeled_images
                == observed_updates
                * self.effective_labeled_batch
            ),
            "expected_update_budget_reached": (
                observed_updates
                == self.expected_optimizer_updates
            ),
        }

        path = (
            Path(runner.work_dir)
            / "metrics"
            / "validation"
            / "training_summary.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        temp_path = path.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temp_path.replace(path)


__all__ = ["SupervisedTrainingSummaryHook"]