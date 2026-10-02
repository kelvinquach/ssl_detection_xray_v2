"""S4.13 observed SSL batch/exposure accounting and training summary."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmdet.registry import HOOKS

from src.utils.optimizer_update_counter import (
    ActualOptimizerUpdateCounter,
)
from src.utils.official_train_log_runtime import OfficialTrainLogRuntime
from src.utils.resume_checkpoint_hook import (
    RUNNER_COUNTER_ATTR,
    RUNNER_PENDING_RESUME_STATE_ATTR,
    RUNNER_TRAINING_SUMMARY_STATE_ATTR,
)


@HOOKS.register_module()
class SSLTrainingSummaryHook(Hook):
    """Record observed SSL exposure using the authoritative update counter.

    Locked S4.13 assumptions:
    - L:U = 1:1.
    - effective labeled batch = 4.
    - effective unlabeled batch = 4.
    - accumulative_counts = 1.
    - ActualUpdateMeanTeacherHook runs before this hook.
    """

    priority = "LOW"

    def __init__(
        self,
        expected_optimizer_updates: int,
        effective_labeled_batch: int = 4,
        effective_unlabeled_batch: int = 4,
    ) -> None:
        if expected_optimizer_updates <= 0:
            raise ValueError(
                "expected_optimizer_updates must be positive."
            )
        if effective_labeled_batch <= 0:
            raise ValueError(
                "effective_labeled_batch must be positive."
            )
        if effective_unlabeled_batch <= 0:
            raise ValueError(
                "effective_unlabeled_batch must be positive."
            )

        self.expected_optimizer_updates = int(
            expected_optimizer_updates
        )
        self.effective_labeled_batch = int(
            effective_labeled_batch
        )
        self.effective_unlabeled_batch = int(
            effective_unlabeled_batch
        )

        self._last_updates: Optional[int] = None
        self._scale_before: Optional[float] = None

        self._labeled_images = 0
        self._unlabeled_images = 0
        self._amp_skipped_steps = 0
        self._ema_updates = 0
        self._train_log_runtime: Optional[OfficialTrainLogRuntime] = None

    @staticmethod
    def _counter(runner: Any) -> ActualOptimizerUpdateCounter:
        counter = getattr(runner, RUNNER_COUNTER_ATTR, None)
        if not isinstance(counter, ActualOptimizerUpdateCounter):
            raise RuntimeError(
                "SSLTrainingSummaryHook requires the authoritative "
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
    def _branch_count(data_batch: Any, branch: str) -> int:
        if not isinstance(data_batch, dict):
            raise RuntimeError(
                "SSLTrainingSummaryHook expects dict data_batch."
            )

        inputs = data_batch.get("inputs")
        if not isinstance(inputs, dict):
            raise RuntimeError(
                "SSLTrainingSummaryHook expects branch-wise inputs."
            )

        values = inputs.get(branch)
        if values is None:
            raise RuntimeError(
                f"Missing SSL branch in data_batch: {branch}"
            )

        return sum(value is not None for value in values)

    @staticmethod
    def _validate_ema_hook_order(runner: Any) -> None:
        hooks = list(getattr(runner, "_hooks", []))

        summary_index = None
        ema_indices = []

        for index, hook in enumerate(hooks):
            name = type(hook).__name__

            if name == "SSLTrainingSummaryHook":
                summary_index = index
            elif name == "ActualUpdateMeanTeacherHook":
                ema_indices.append(index)

        if summary_index is None:
            raise RuntimeError(
                "SSLTrainingSummaryHook is absent from runner hooks."
            )

        if len(ema_indices) != 1:
            raise RuntimeError(
                "Exactly one ActualUpdateMeanTeacherHook is required."
            )

        if ema_indices[0] >= summary_index:
            raise RuntimeError(
                "SSLTrainingSummaryHook must run after "
                "ActualUpdateMeanTeacherHook."
            )

    def before_train(self, runner: Any) -> None:
        self._validate_ema_hook_order(runner)

        counter = self._counter(runner)
        self._last_updates = counter.count

        if self._last_updates == 0:
            self._labeled_images = 0
            self._unlabeled_images = 0
            self._amp_skipped_steps = 0
            self._ema_updates = 0
        else:
            pending_state = getattr(
                runner,
                RUNNER_PENDING_RESUME_STATE_ATTR,
                None,
            )
            if not isinstance(pending_state, dict):
                raise RuntimeError(
                    "Resumed SSL summary requires pending protocol resume state."
                )
            summary_state = pending_state.get("training_summary_state")
            if not isinstance(summary_state, dict):
                raise RuntimeError(
                    "Resumed SSL summary is missing training_summary_state."
                )

            required = (
                "labeled_images_consumed_total",
                "unlabeled_images_consumed_total",
                "amp_skipped_step_count",
                "ema_update_count",
            )
            for key in required:
                value = summary_state.get(key)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                ):
                    raise RuntimeError(
                        f"Invalid resumed SSL training-summary field {key!r}."
                    )

            self._labeled_images = int(
                summary_state["labeled_images_consumed_total"]
            )
            self._unlabeled_images = int(
                summary_state["unlabeled_images_consumed_total"]
            )
            self._amp_skipped_steps = int(
                summary_state["amp_skipped_step_count"]
            )
            self._ema_updates = int(
                summary_state["ema_update_count"]
            )

            if (
                self._labeled_images
                != self._last_updates * self.effective_labeled_batch
            ):
                raise RuntimeError(
                    "Resumed labeled exposure does not match restored updates."
                )
            if (
                self._unlabeled_images
                != self._last_updates * self.effective_unlabeled_batch
            ):
                raise RuntimeError(
                    "Resumed unlabeled exposure does not match restored updates."
                )
            if self._ema_updates != self._last_updates:
                raise RuntimeError(
                    "Resumed EMA count does not match restored updates."
                )

        setattr(
            runner,
            RUNNER_TRAINING_SUMMARY_STATE_ATTR,
            {
                "labeled_images_consumed_total": int(self._labeled_images),
                "unlabeled_images_consumed_total": int(self._unlabeled_images),
                "amp_skipped_step_count": int(self._amp_skipped_steps),
                "ema_update_count": int(self._ema_updates),
            },
        )

        accumulative_counts = getattr(
            runner.optim_wrapper,
            "_accumulative_counts",
            None,
        )
        if accumulative_counts != 1:
            raise RuntimeError(
                "S4.13 requires accumulative_counts=1."
            )

        self._train_log_runtime = OfficialTrainLogRuntime(
            runner,
            restored_optimizer_updates=int(self._last_updates),
        )

    def before_train_iter(
        self,
        runner: Any,
        batch_idx: int,
        data_batch: Any = None,
    ) -> None:
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        setattr(model, "_official_last_pseudo_counts", None)

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
                "SSLTrainingSummaryHook was not initialized."
            )

        current = self._counter(runner).count
        delta = current - self._last_updates

        if delta not in (0, 1):
            raise RuntimeError(
                f"Invalid actual-update delta in S4.13: {delta}"
            )

        scale_after = self._loss_scale(runner)

        if delta == 1:
            labeled = self._branch_count(
                data_batch,
                "sup",
            )
            unsup_teacher = self._branch_count(
                data_batch,
                "unsup_teacher",
            )
            unsup_student = self._branch_count(
                data_batch,
                "unsup_student",
            )

            if labeled != self.effective_labeled_batch:
                raise RuntimeError(
                    "Observed labeled batch does not match S4.13: "
                    f"{labeled} != {self.effective_labeled_batch}"
                )

            if (
                unsup_teacher
                != self.effective_unlabeled_batch
                or unsup_student
                != self.effective_unlabeled_batch
            ):
                raise RuntimeError(
                    "Observed unlabeled batch does not match S4.13: "
                    f"teacher={unsup_teacher}, "
                    f"student={unsup_student}, "
                    f"expected={self.effective_unlabeled_batch}"
                )

            self._labeled_images += labeled
            self._unlabeled_images += unsup_teacher

            # The locked EMA hook runs earlier in after_train_iter and
            # updates exactly once for this same authoritative delta.
            self._ema_updates += 1

            runtime = self._train_log_runtime
            if not isinstance(runtime, OfficialTrainLogRuntime):
                raise RuntimeError(
                    "OFFICIAL_SSL_TRAIN_LOG_RUNTIME_NOT_INITIALIZED"
                )

            if scale_after is None:
                raise RuntimeError(
                    "OFFICIAL_SSL_AMP_SCALE_UNAVAILABLE"
                )

            model = runner.model
            if is_model_wrapper(model):
                model = model.module

            pseudo_counts = getattr(
                model,
                "_official_last_pseudo_counts",
                None,
            )
            if not isinstance(pseudo_counts, dict):
                raise RuntimeError(
                    "OFFICIAL_SSL_PSEUDO_COUNTS_UNAVAILABLE"
                )

            required_pseudo_fields = (
                "pseudo_count",
                "classification_pseudo_count",
                "regression_pseudo_count",
            )
            for field in required_pseudo_fields:
                value = pseudo_counts.get(field)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                ):
                    raise RuntimeError(
                        f"OFFICIAL_SSL_INVALID_PSEUDO_COUNT:{field}"
                    )

            if (
                pseudo_counts["classification_pseudo_count"]
                > pseudo_counts["pseudo_count"]
                or pseudo_counts["regression_pseudo_count"]
                > pseudo_counts["pseudo_count"]
            ):
                raise RuntimeError(
                    "OFFICIAL_SSL_PSEUDO_SUBSET_COUNT_EXCEEDS_BASE"
                )

            loss_total = runtime.output_value(outputs, "loss")
            loss_supervised = runtime.loss_sum_by_prefix(
                outputs,
                "sup_",
            )
            loss_unsupervised = runtime.loss_sum_by_prefix(
                outputs,
                "unsup_",
            )

            runtime.logger.log_optimizer_update(
                {
                    "optimizer_update": int(current),
                    "raw_iteration": int(runner.iter) + 1,
                    "learning_rate": runtime.learning_rate(runner),
                    "loss_total": loss_total,
                    "loss_supervised": loss_supervised,
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
                    "loss_unsupervised": loss_unsupervised,
                    "pseudo_count": int(
                        pseudo_counts["pseudo_count"]
                    ),
                    "classification_pseudo_count": int(
                        pseudo_counts["classification_pseudo_count"]
                    ),
                    "regression_pseudo_count": int(
                        pseudo_counts["regression_pseudo_count"]
                    ),
                    "unlabeled_microbatches_consumed_cumulative": (
                        int(current)
                    ),
                    "unlabeled_images_consumed_cumulative": int(
                        self._unlabeled_images
                    ),
                    "ema_step_executed": True,
                },
                ssl=True,
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
                    "No actual optimizer update occurred, but no AMP "
                    "loss-scale decrease was observed. With locked "
                    "accumulative_counts=1 this is unexpected."
                )

        self._last_updates = current

        setattr(
            runner,
            RUNNER_TRAINING_SUMMARY_STATE_ATTR,
            {
                "labeled_images_consumed_total": int(self._labeled_images),
                "unlabeled_images_consumed_total": int(self._unlabeled_images),
                "amp_skipped_step_count": int(self._amp_skipped_steps),
                "ema_update_count": int(self._ema_updates),
            },
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
        counter = self._counter(runner)

        validation_events_observed = self._validation_event_count(runner)

        observed_updates = int(counter.count)

        summary = {
            "expected_optimizer_updates":
                self.expected_optimizer_updates,
            "observed_optimizer_updates":
                observed_updates,
            "effective_labeled_batch":
                self.effective_labeled_batch,
            "effective_unlabeled_batch":
                self.effective_unlabeled_batch,
            "labeled_images_consumed_total":
                int(self._labeled_images),
            "unlabeled_images_consumed_total":
                int(self._unlabeled_images),
            "validation_events_observed":
                validation_events_observed,
            "amp_skipped_step_count":
                int(self._amp_skipped_steps),
            "ema_update_count":
                int(self._ema_updates),
            "l_u_ratio": "1:1",
            "labeled_exposure_matches_update_count": (
                self._labeled_images
                == observed_updates
                * self.effective_labeled_batch
            ),
            "unlabeled_exposure_matches_update_count": (
                self._unlabeled_images
                == observed_updates
                * self.effective_unlabeled_batch
            ),
            "ema_updates_match_optimizer_updates": (
                self._ema_updates == observed_updates
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


__all__ = ["SSLTrainingSummaryHook"]