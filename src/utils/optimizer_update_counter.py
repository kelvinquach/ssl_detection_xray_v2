"""Actual optimizer-update accounting.

The counter is driven by the underlying PyTorch optimizer's step post-hook.
Therefore it advances only when ``optimizer.step()`` is actually executed.

This intentionally differs from raw training iteration counts and MMEngine
OptimWrapper ``_inner_count``. It also naturally does not advance when
``GradScaler`` skips the underlying optimizer step because of non-finite
gradients.
"""

from __future__ import annotations

from typing import Any, Optional


class ActualOptimizerUpdateCounter:
    """Count successful underlying optimizer updates.

    The counter attaches to a PyTorch optimizer through
    ``register_step_post_hook``. A count increment therefore represents an
    actual call to the optimizer's ``step`` implementation, not a dataloader
    iteration, microbatch, gradient-accumulation iteration, or attempted AMP
    step.

    S2.10 scope is accounting only. Logging, checkpoint/resume integration,
    EMA timing, and AMP skipped-step event logging are implemented by later
    stages.
    """

    def __init__(self) -> None:
        self._count = 0
        self._handle: Optional[Any] = None
        self._optimizer: Optional[Any] = None

    @property
    def count(self) -> int:
        """Return the number of actual optimizer updates observed."""
        return self._count

    @property
    def attached(self) -> bool:
        """Return whether the counter is currently attached."""
        return self._handle is not None

    def attach(self, optimizer: Any) -> None:
        """Attach the counter to one underlying PyTorch optimizer.

        Args:
            optimizer: Optimizer exposing ``register_step_post_hook``.

        Raises:
            RuntimeError: If this counter is already attached.
            TypeError: If the optimizer lacks step-post-hook support.
        """
        if self.attached:
            raise RuntimeError(
                "ActualOptimizerUpdateCounter is already attached."
            )

        register_hook = getattr(
            optimizer,
            "register_step_post_hook",
            None,
        )
        if register_hook is None:
            raise TypeError(
                "Optimizer does not expose register_step_post_hook()."
            )

        def _after_optimizer_step(
            _optimizer: Any,
            _args: Any,
            _kwargs: Any,
        ) -> None:
            self._count += 1

        self._optimizer = optimizer
        self._handle = register_hook(_after_optimizer_step)

    def detach(self) -> None:
        """Remove the optimizer hook if attached."""
        if self._handle is not None:
            self._handle.remove()

        self._handle = None
        self._optimizer = None


__all__ = ["ActualOptimizerUpdateCounter"]
