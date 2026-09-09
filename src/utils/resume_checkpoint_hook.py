"""MMEngine checkpoint hook foundation for S2.13."""

from __future__ import annotations

from typing import Any, Dict, Optional

from mmengine.hooks import Hook
from mmengine.registry import HOOKS

from src.utils.optimizer_update_counter import ActualOptimizerUpdateCounter
from src.utils.resume_state import (
    PROTOCOL_RESUME_STATE_KEY,
    build_protocol_resume_state,
    restore_protocol_resume_state,
    validate_protocol_resume_state,
)


RUNNER_COUNTER_ATTR = "_s2_actual_optimizer_update_counter"
RUNNER_PENDING_RESUME_STATE_ATTR = "_s2_pending_protocol_resume_state"
RUNNER_PENDING_RNG_RESTORE_ATTR = "_s2_pending_rng_restore"


@HOOKS.register_module()
class ProtocolResumeCheckpointHook(Hook):
    """Inject and restore protocol resume state around MMEngine checkpoints."""

    def before_run(self, runner) -> None:
        """Attach the actual optimizer-update counter once."""
        if hasattr(runner, RUNNER_COUNTER_ATTR):
            raise RuntimeError(
                "ProtocolResumeCheckpointHook found an existing S2 counter."
            )

        optim_wrapper = runner.optim_wrapper
        optimizer = getattr(optim_wrapper, "optimizer", None)

        if optimizer is None:
            raise RuntimeError(
                "ProtocolResumeCheckpointHook requires a single underlying "
                "optimizer exposed as runner.optim_wrapper.optimizer."
            )

        counter = ActualOptimizerUpdateCounter()
        counter.attach(optimizer)

        setattr(runner, RUNNER_COUNTER_ATTR, counter)
        setattr(runner, RUNNER_PENDING_RESUME_STATE_ATTR, None)
        setattr(runner, RUNNER_PENDING_RNG_RESTORE_ATTR, False)

    def before_save_checkpoint(
        self,
        runner,
        checkpoint: Dict[str, Any],
    ) -> None:
        """Inject protocol state into an MMEngine checkpoint."""
        counter = getattr(runner, RUNNER_COUNTER_ATTR, None)

        if not isinstance(counter, ActualOptimizerUpdateCounter):
            raise RuntimeError(
                "Actual optimizer-update counter is unavailable at checkpoint save."
            )

        checkpoint[PROTOCOL_RESUME_STATE_KEY] = build_protocol_resume_state(
            training_seed=int(runner.seed),
            update_counter=counter,
        )

    def after_load_checkpoint(
        self,
        runner,
        checkpoint: Dict[str, Any],
    ) -> None:
        """Validate loaded protocol state and restore non-RNG custom state.

        Full RNG restoration is deliberately delayed until after MMEngine
        replays consumed raw iterations in the iteration-based train loop.
        """
        if PROTOCOL_RESUME_STATE_KEY not in checkpoint:
            raise KeyError(
                f"Checkpoint is missing {PROTOCOL_RESUME_STATE_KEY!r}."
            )

        state = checkpoint[PROTOCOL_RESUME_STATE_KEY]

        validate_protocol_resume_state(
            state,
            expected_training_seed=int(runner.seed),
        )

        counter = getattr(runner, RUNNER_COUNTER_ATTR, None)

        if not isinstance(counter, ActualOptimizerUpdateCounter):
            raise RuntimeError(
                "Actual optimizer-update counter is unavailable at resume."
            )

        restore_protocol_resume_state(
            state,
            expected_training_seed=int(runner.seed),
            update_counter=counter,
            restore_rng=False,
        )

        setattr(runner, RUNNER_PENDING_RESUME_STATE_ATTR, state)
        setattr(runner, RUNNER_PENDING_RNG_RESTORE_ATTR, True)


__all__ = [
    "ProtocolResumeCheckpointHook",
    "RUNNER_COUNTER_ATTR",
    "RUNNER_PENDING_RESUME_STATE_ATTR",
    "RUNNER_PENDING_RNG_RESTORE_ATTR",
]
