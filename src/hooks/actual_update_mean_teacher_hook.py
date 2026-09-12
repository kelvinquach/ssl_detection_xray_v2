"""S4.02 Mean Teacher EMA driven by actual optimizer updates.

Canonical contract:
- momentum = 0.001
- Teacher_new = 0.999 * Teacher_old + 0.001 * Student_new
- skip_buffers = True
- EMA executes exactly once per actual Student optimizer update
- raw iterations without optimizer.step() do not update Teacher
- AMP-skipped optimizer steps therefore do not update Teacher

Teacher initialization is owned exclusively by S4.01
TeacherInitializationHook and is not repeated here.
"""

from mmengine.model import is_model_wrapper
from mmdet.engine.hooks.mean_teacher_hook import MeanTeacherHook
from mmdet.registry import HOOKS

from src.utils.optimizer_update_counter import ActualOptimizerUpdateCounter
from src.utils.resume_checkpoint_hook import RUNNER_COUNTER_ATTR


@HOOKS.register_module()
class ActualUpdateMeanTeacherHook(MeanTeacherHook):
    """Update Teacher EMA only after actual optimizer updates."""

    priority = "HIGH"

    def __init__(
        self,
        momentum: float = 0.001,
        skip_buffers: bool = True,
    ) -> None:
        super().__init__(
            momentum=momentum,
            interval=1,
            skip_buffer=skip_buffers,
        )
        self._last_updates = None

    def _counter(self, runner) -> ActualOptimizerUpdateCounter:
        counter = getattr(runner, RUNNER_COUNTER_ATTR, None)
        if not isinstance(counter, ActualOptimizerUpdateCounter):
            raise RuntimeError(
                "ActualUpdateMeanTeacherHook requires the authoritative "
                "ActualOptimizerUpdateCounter."
            )
        return counter

    @staticmethod
    def _model(runner):
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        if not hasattr(model, "student") or not hasattr(model, "teacher"):
            raise RuntimeError(
                "ACTUAL_UPDATE_MEAN_TEACHER_REQUIRES_STUDENT_AND_TEACHER"
            )
        return model

    def before_train(self, runner) -> None:
        # S4.01 owns Teacher(t0)=Student(t0). Do not synchronize here.
        self._model(runner)
        self._last_updates = self._counter(runner).count

    def after_train_iter(
        self,
        runner,
        batch_idx: int,
        data_batch=None,
        outputs=None,
    ) -> None:
        if self._last_updates is None:
            raise RuntimeError(
                "ActualUpdateMeanTeacherHook was not initialized by before_train."
            )

        current = self._counter(runner).count
        delta = current - self._last_updates

        if delta not in (0, 1):
            raise RuntimeError(f"Invalid actual-update delta for EMA: {delta}")

        if delta == 1:
            self.momentum_update(
                self._model(runner),
                self.momentum,
            )

        self._last_updates = current


__all__ = ["ActualUpdateMeanTeacherHook"]
