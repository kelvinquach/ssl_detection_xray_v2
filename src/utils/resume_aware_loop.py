"""Resume-aware MMEngine iteration loop for S2.13.

This loop preserves the standard MMEngine IterBasedTrainLoop behavior and
adds exactly one operation for technical resume:

    replay consumed raw iterations
    -> restore checkpoint RNG state
    -> fetch the next training batch

The insertion point is required because restoring RNG before replay would let
the replay itself consume the restored RNG trajectory, while restoring in
before_train_iter would be too late because the next batch has already been
fetched.
"""

from __future__ import annotations

import logging

from mmengine.logging import print_log
from mmengine.registry import LOOPS
from mmengine.runner.loops import IterBasedTrainLoop

from src.utils.resume_checkpoint_hook import (
    RUNNER_PENDING_RESUME_STATE_ATTR,
    RUNNER_PENDING_RNG_RESTORE_ATTR,
)
from src.utils.seed import restore_full_rng_state


@LOOPS.register_module()
class ResumeAwareIterBasedTrainLoop(IterBasedTrainLoop):
    """MMEngine iteration loop with delayed RNG restoration after replay."""

    def run(self):
        """Launch training while preserving resume RNG trajectory."""
        self.runner.call_hook("before_train")

        # MMEngine treats iteration-based training as one large epoch.
        self.runner.call_hook("before_train_epoch")

        # Preserve standard MMEngine resume replay behavior.
        if self._iter > 0:
            print_log(
                f"Advance dataloader {self._iter} steps to skip data "
                "that has already been trained",
                logger="current",
                level=logging.WARNING,
            )
            for _ in range(self._iter):
                next(self.dataloader_iterator)

        # S2.13 insertion point:
        # replay is complete, but the next training batch has not yet been
        # fetched. Restore checkpoint RNG exactly here.
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

        while self._iter < self._max_iters and not self.stop_training:
            self.runner.model.train()

            data_batch = next(self.dataloader_iterator)
            self.run_iter(data_batch)

            self._decide_current_val_interval()

            if (
                self.runner.val_loop is not None
                and self._iter >= self.val_begin
                and (
                    self._iter % self.val_interval == 0
                    or self._iter == self._max_iters
                )
            ):
                self.runner.val_loop.run()

        self.runner.call_hook("after_train_epoch")
        self.runner.call_hook("after_train")

        return self.runner.model


__all__ = ["ResumeAwareIterBasedTrainLoop"]
