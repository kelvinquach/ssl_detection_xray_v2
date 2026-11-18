"""MMEngine checkpoint hook foundation for S2.13."""

from __future__ import annotations

from typing import Any, Dict, Optional

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
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
RUNNER_TRAINING_SUMMARY_STATE_ATTR = "_s9_training_summary_state"

SSL_NON_VLM_REFINEMENT_HISTORY_STATE_KEY = (
    "ssl_non_vlm_refinement_committed_history_state"
)


def _unwrap_runner_model(runner: Any) -> Any:
    """Return the underlying model when MMEngine wraps the model."""
    model = runner.model
    if is_model_wrapper(model):
        model = model.module
    return model


def _get_non_vlm_refinement_history_interface(model: Any):
    """Return optional SSL + REF history save/restore methods."""
    history_getter = getattr(
        model,
        "get_non_vlm_refinement_history_state",
        None,
    )
    history_restorer = getattr(
        model,
        "restore_non_vlm_refinement_history_state",
        None,
    )

    getter_available = callable(history_getter)
    restorer_available = callable(history_restorer)

    if getter_available != restorer_available:
        raise RuntimeError(
            "SSL + REF model must implement both "
            "get_non_vlm_refinement_history_state() and "
            "restore_non_vlm_refinement_history_state()."
        )

    if not getter_available:
        return None, None

    return history_getter, history_restorer


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
        setattr(runner, RUNNER_TRAINING_SUMMARY_STATE_ATTR, None)

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

        state = build_protocol_resume_state(
            training_seed=int(runner.seed),
            update_counter=counter,
        )

        model = _unwrap_runner_model(runner)
        history_getter, _ = _get_non_vlm_refinement_history_interface(
            model
        )

        if history_getter is not None:
            history_state = history_getter()
            if not isinstance(history_state, dict):
                raise RuntimeError(
                    "SSL + REF committed history state must be a dictionary."
                )
            state[
                SSL_NON_VLM_REFINEMENT_HISTORY_STATE_KEY
            ] = history_state

        summary_state = getattr(
            runner,
            RUNNER_TRAINING_SUMMARY_STATE_ATTR,
            None,
        )
        if summary_state is not None:
            if not isinstance(summary_state, dict):
                raise RuntimeError(
                    "Training-summary resume state must be a dict."
                )
            state["training_summary_state"] = dict(summary_state)

        checkpoint[PROTOCOL_RESUME_STATE_KEY] = state

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

        model = _unwrap_runner_model(runner)
        _, history_restorer = _get_non_vlm_refinement_history_interface(
            model
        )

        history_state_present = (
            SSL_NON_VLM_REFINEMENT_HISTORY_STATE_KEY in state
        )

        if history_restorer is not None:
            if not history_state_present:
                raise KeyError(
                    "SSL + REF checkpoint is missing committed "
                    "confidence history state."
                )

            history_state = state[
                SSL_NON_VLM_REFINEMENT_HISTORY_STATE_KEY
            ]
            if not isinstance(history_state, dict):
                raise TypeError(
                    "SSL + REF committed confidence history state "
                    "must be a dictionary."
                )

            history_restorer(history_state)

        elif history_state_present:
            raise RuntimeError(
                "Checkpoint contains SSL + REF committed confidence "
                "history, but the current model does not support "
                "SSL + REF history restore."
            )

        setattr(runner, RUNNER_PENDING_RESUME_STATE_ATTR, state)
        setattr(runner, RUNNER_PENDING_RNG_RESTORE_ATTR, True)


__all__ = [
    "ProtocolResumeCheckpointHook",
    "RUNNER_COUNTER_ATTR",
    "RUNNER_PENDING_RESUME_STATE_ATTR",
    "RUNNER_PENDING_RNG_RESTORE_ATTR",
    "RUNNER_TRAINING_SUMMARY_STATE_ATTR",
    "SSL_NON_VLM_REFINEMENT_HISTORY_STATE_KEY",
]
