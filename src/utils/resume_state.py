"""Protocol resume-state helpers for S2.13.

MMEngine already owns model, optimizer, AMP-scaler, parameter-scheduler,
message-hub, and raw-iteration checkpoint state. This module adds only the
protocol state that MMEngine 0.10.7 does not preserve exactly:

- locked training-seed identity;
- actual optimizer-update counter;
- full Python / NumPy / PyTorch CPU / CUDA RNG states.

Sampler/DataLoader state is not serialized here because the locked
MMEngine DefaultSampler exposes no state_dict/load_state_dict. IterBasedTrainLoop
reconstructs its position by deterministic raw-iteration replay.
"""

from __future__ import annotations

from typing import Any, Dict

from src.utils.optimizer_update_counter import ActualOptimizerUpdateCounter
from src.utils.seed import get_full_rng_state, restore_full_rng_state


PROTOCOL_RESUME_STATE_KEY = "protocol_resume_state"
PROTOCOL_RESUME_SCHEMA_VERSION = "1.0"


def _validate_training_seed(seed: Any, *, field_name: str) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return seed


def build_protocol_resume_state(
    *,
    training_seed: int,
    update_counter: ActualOptimizerUpdateCounter,
) -> Dict[str, Any]:
    """Capture protocol state to inject into an MMEngine checkpoint."""
    training_seed = _validate_training_seed(
        training_seed,
        field_name="training_seed",
    )

    return {
        "schema_version": PROTOCOL_RESUME_SCHEMA_VERSION,
        "training_seed": training_seed,
        "actual_optimizer_update_counter": update_counter.state_dict(),
        "rng_state": get_full_rng_state(),
        "sampler_dataloader_resume_mode": (
            "MMENGINE_ITERBASED_RAW_ITER_REPLAY"
        ),
    }


def validate_protocol_resume_state(
    state: Dict[str, Any],
    *,
    expected_training_seed: int,
) -> None:
    """Validate checkpoint protocol state before a technical resume."""
    if not isinstance(state, dict):
        raise TypeError("Protocol resume state must be a dict.")

    required = (
        "schema_version",
        "training_seed",
        "actual_optimizer_update_counter",
        "rng_state",
        "sampler_dataloader_resume_mode",
    )
    missing = [key for key in required if key not in state]
    if missing:
        raise KeyError(
            "Protocol resume state missing required field(s): "
            + ", ".join(missing)
        )

    if state["schema_version"] != PROTOCOL_RESUME_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported protocol resume schema_version: "
            f"{state['schema_version']!r}."
        )

    expected_training_seed = _validate_training_seed(
        expected_training_seed,
        field_name="expected_training_seed",
    )
    checkpoint_seed = _validate_training_seed(
        state["training_seed"],
        field_name="checkpoint training_seed",
    )

    if checkpoint_seed != expected_training_seed:
        raise RuntimeError(
            "Training-seed mismatch on resume: "
            f"checkpoint={checkpoint_seed}, "
            f"current={expected_training_seed}."
        )

    if state["sampler_dataloader_resume_mode"] != (
        "MMENGINE_ITERBASED_RAW_ITER_REPLAY"
    ):
        raise ValueError(
            "Unsupported sampler/DataLoader resume mode: "
            f"{state['sampler_dataloader_resume_mode']!r}."
        )

    counter_state = state["actual_optimizer_update_counter"]
    probe = ActualOptimizerUpdateCounter()
    probe.load_state_dict(counter_state)

    if not isinstance(state["rng_state"], dict):
        raise TypeError("rng_state must be a dict.")


def restore_protocol_resume_state(
    state: Dict[str, Any],
    *,
    expected_training_seed: int,
    update_counter: ActualOptimizerUpdateCounter,
    restore_rng: bool = True,
) -> None:
    """Restore protocol state after validation.

    RNG restoration can be delayed until after MMEngine has replayed consumed
    raw iterations. This is required so the next fetched batch continues from
    the checkpoint RNG trajectory rather than from the replay trajectory.
    """
    validate_protocol_resume_state(
        state,
        expected_training_seed=expected_training_seed,
    )

    update_counter.load_state_dict(
        state["actual_optimizer_update_counter"]
    )

    if restore_rng:
        restore_full_rng_state(state["rng_state"])


__all__ = [
    "PROTOCOL_RESUME_STATE_KEY",
    "PROTOCOL_RESUME_SCHEMA_VERSION",
    "build_protocol_resume_state",
    "validate_protocol_resume_state",
    "restore_protocol_resume_state",
]
