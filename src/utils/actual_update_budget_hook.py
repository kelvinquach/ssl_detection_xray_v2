"""Actual optimizer-update budget and scheduler control."""

from mmengine.hooks import Hook
from mmengine.registry import HOOKS

from src.utils.optimizer_update_counter import ActualOptimizerUpdateCounter
from src.utils.resume_checkpoint_hook import RUNNER_COUNTER_ATTR


@HOOKS.register_module()
class ActualUpdateBudgetSchedulerHook(Hook):
    """Step schedulers and stop training by actual optimizer updates."""

    priority = "ABOVE_NORMAL"

    def __init__(self, target_updates: int):
        if not isinstance(target_updates, int) or target_updates <= 0:
            raise ValueError("target_updates must be a positive integer.")
        self.target_updates = target_updates
        self._last_updates = None

    def _counter(self, runner):
        counter = getattr(runner, RUNNER_COUNTER_ATTR, None)
        if not isinstance(counter, ActualOptimizerUpdateCounter):
            raise RuntimeError("Actual optimizer-update counter is unavailable.")
        return counter

    def _schedulers(self, runner):
        schedulers = runner.param_schedulers
        if schedulers is None:
            raise RuntimeError("param_scheduler is required.")
        if isinstance(schedulers, dict):
            result = [s for group in schedulers.values() for s in group]
        else:
            result = list(schedulers)
        if any(s.by_epoch for s in result):
            raise RuntimeError("All S3 schedulers must use by_epoch=False.")
        return result

    def before_train(self, runner):
        current = self._counter(runner).count
        if current > self.target_updates:
            raise RuntimeError("Actual optimizer updates exceed target.")
        self._schedulers(runner)
        self._last_updates = current

        if current < self.target_updates and runner.iter >= runner.max_iters:
            extension = max(
                1,
                int(getattr(runner.optim_wrapper, "_accumulative_counts", 1)),
            )
            new_max_iters = int(runner.iter) + extension
            runner.train_loop._max_iters = new_max_iters
            runner.message_hub.update_info("max_iters", new_max_iters)
            runner.optim_wrapper._max_counts = new_max_iters
            runner.optim_wrapper._remainder_counts = (
                new_max_iters % runner.optim_wrapper._accumulative_counts
            )

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        current = self._counter(runner).count
        delta = current - self._last_updates

        if delta not in (0, 1):
            raise RuntimeError(f"Invalid actual-update delta: {delta}")

        if delta == 1:
            for scheduler in self._schedulers(runner):
                scheduler.step()

        self._last_updates = current

        if current > self.target_updates:
            raise RuntimeError("Actual optimizer updates exceeded target.")

        if current == self.target_updates:
            final_raw_iter = runner.iter + 1
            runner.train_loop._max_iters = final_raw_iter
            runner.message_hub.update_info("max_iters", final_raw_iter)
            return

        if runner.iter + 1 >= runner.max_iters:
            optim_wrapper = runner.optim_wrapper
            extension = int(
                getattr(optim_wrapper, "_accumulative_counts", 1)
            )
            new_max_iters = runner.max_iters + max(1, extension)

            runner.train_loop._max_iters = new_max_iters
            runner.message_hub.update_info("max_iters", new_max_iters)

            if hasattr(optim_wrapper, "_max_counts"):
                optim_wrapper._max_counts = new_max_iters
                optim_wrapper._remainder_counts = (
                    new_max_iters % optim_wrapper._accumulative_counts
                )


__all__ = ["ActualUpdateBudgetSchedulerHook"]
