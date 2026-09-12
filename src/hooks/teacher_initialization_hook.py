"""S4.01 teacher initialization hook.

Fresh run only:
    Teacher(t0) = Student(t0)

Resume runs are intentionally not re-synchronized. EMA update cadence and
AMP-skip synchronization belong to later S4 tasks.
"""

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmdet.registry import HOOKS


@HOOKS.register_module()
class TeacherInitializationHook(Hook):
    """Synchronize Teacher from Student once at fresh-run initialization."""

    priority = "VERY_HIGH"

    def before_train(self, runner) -> None:
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        if not hasattr(model, "student") or not hasattr(model, "teacher"):
            raise RuntimeError(
                "TEACHER_INITIALIZATION_REQUIRES_STUDENT_AND_TEACHER"
            )

        # Resume must restore the independently saved Teacher state.
        # Never overwrite it with Student after a resumed iteration > 0.
        if runner.iter != 0:
            return

        model.teacher.load_state_dict(
            model.student.state_dict(),
            strict=True,
        )

        model.teacher.eval()
        for param in model.teacher.parameters():
            param.requires_grad = False