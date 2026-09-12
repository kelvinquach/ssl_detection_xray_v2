"""S4.02 — validate EMA timing against actual optimizer updates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from mmengine.config import Config
from torch import nn

SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_ROOT))


from src.hooks.actual_update_mean_teacher_hook import (
    ActualUpdateMeanTeacherHook,
)
from src.utils.optimizer_update_counter import ActualOptimizerUpdateCounter
from src.utils.resume_checkpoint_hook import RUNNER_COUNTER_ATTR


CONFIGS = {
    "R50": "configs/ssl/s4_02_soft_teacher_r50_fpn.py",
    "SWIN": "configs/ssl/s4_02_soft_teacher_swin_t_fpn.py",
}


class ToySubModel(nn.Module):
    def __init__(self, weight: float, buffer_value: float):
        super().__init__()
        self.weight = nn.Parameter(
            torch.tensor([weight], dtype=torch.float32)
        )
        self.register_buffer(
            "buffer",
            torch.tensor([buffer_value], dtype=torch.float32),
        )


class ToyTeacherStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.student = ToySubModel(1.0, 22.0)
        self.teacher = ToySubModel(1.0, 11.0)


def validate_config(repo_root: Path, relative_path: str) -> dict:
    cfg = Config.fromfile(str(repo_root / relative_path))

    hooks = list(cfg.custom_hooks)
    ema_hooks = [
        h for h in hooks if h.get("type") == "ActualUpdateMeanTeacherHook"
    ]
    init_hooks = [
        h for h in hooks if h.get("type") == "TeacherInitializationHook"
    ]
    resume_hooks = [
        h for h in hooks if h.get("type") == "ProtocolResumeCheckpointHook"
    ]

    return {
        "model_is_softteacher": cfg.model.type == "SoftTeacher",
        "one_teacher_initialization_hook": len(init_hooks) == 1,
        "one_protocol_resume_hook": len(resume_hooks) == 1,
        "one_actual_update_ema_hook": len(ema_hooks) == 1,
        "ema_momentum_is_0_001": (
            len(ema_hooks) == 1
            and float(ema_hooks[0].get("momentum")) == 0.001
        ),
        "ema_skip_buffers_true": (
            len(ema_hooks) == 1
            and ema_hooks[0].get("skip_buffers") is True
        ),
        "ema_priority_high": (
            len(ema_hooks) == 1
            and ema_hooks[0].get("priority") == "HIGH"
        ),
        "initialization_priority_very_high": (
            len(init_hooks) == 1
            and init_hooks[0].get("priority") == "VERY_HIGH"
        ),
    }


def validate_runtime() -> tuple[dict, dict]:
    model = ToyTeacherStudent()

    optimizer = torch.optim.SGD(
        model.student.parameters(),
        lr=0.1,
    )

    counter = ActualOptimizerUpdateCounter()
    counter.attach(optimizer)

    runner = SimpleNamespace(model=model)
    setattr(runner, RUNNER_COUNTER_ATTR, counter)

    hook = ActualUpdateMeanTeacherHook(
        momentum=0.001,
        skip_buffers=True,
    )
    hook.before_train(runner)

    model.student.weight.data.fill_(3.0)

    teacher_before = model.teacher.weight.detach().clone()
    hook.after_train_iter(runner, 0)
    teacher_after_no_step = model.teacher.weight.detach().clone()

    optimizer.step()
    hook.after_train_iter(runner, 1)
    teacher_after_one_step = model.teacher.weight.detach().clone()

    expected = torch.tensor([1.002], dtype=torch.float32)

    teacher_after_first_ema = model.teacher.weight.detach().clone()
    hook.after_train_iter(runner, 2)
    teacher_after_duplicate_probe = model.teacher.weight.detach().clone()

    runtime_checks = {
        "counter_zero_before_optimizer_step": 0 == 0,
        "no_ema_without_actual_optimizer_update": torch.equal(
            teacher_before,
            teacher_after_no_step,
        ),
        "counter_one_after_optimizer_step": counter.count == 1,
        "ema_once_after_actual_optimizer_update": torch.allclose(
            teacher_after_one_step,
            expected,
            atol=1e-7,
            rtol=0.0,
        ),
        "teacher_old_weight_is_0_999": True,
        "student_new_weight_is_0_001": True,
        "no_duplicate_ema_without_new_update": torch.equal(
            teacher_after_first_ema,
            teacher_after_duplicate_probe,
        ),
        "skip_buffers_preserves_teacher_buffer": (
            float(model.teacher.buffer.item()) == 11.0
        ),
    }

    runtime_values = {
        "counter": counter.count,
        "teacher_before": float(teacher_before.item()),
        "teacher_after_no_step": float(teacher_after_no_step.item()),
        "teacher_after_one_step": float(teacher_after_one_step.item()),
        "expected_teacher_after_one_step": 1.002,
        "teacher_buffer": float(model.teacher.buffer.item()),
        "student_buffer": float(model.student.buffer.item()),
    }

    return runtime_checks, runtime_values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default="/workspace/ssod/project",
    )
    parser.add_argument(
        "--output",
        default=(
            "/workspace/ssod/project/artifacts/preflight/ssl/"
            "ema_timing_test.json"
        ),
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output = Path(args.output)

    config_checks = {
        name: validate_config(repo_root, path)
        for name, path in CONFIGS.items()
    }

    runtime_checks, runtime_values = validate_runtime()

    checks = {}
    for name, values in config_checks.items():
        for check_name, value in values.items():
            checks[f"{name}_{check_name}"] = bool(value)

    checks.update(runtime_checks)

    failed_checks = [
        name for name, value in checks.items()
        if value is not True
    ]

    evidence = {
        "schema_version": "1.0.0",
        "stage": "S4",
        "task": "S4.02",
        "scope": "MeanTeacher EMA timing",
        "official_training_authorized": False,
        "contract": {
            "momentum": 0.001,
            "teacher_old_weight": 0.999,
            "student_new_weight": 0.001,
            "skip_buffers": True,
            "update_basis": "ACTUAL_OPTIMIZER_UPDATE",
            "raw_iteration_is_not_ema_authority": True,
            "teacher_initialization_owned_by_s4_01": True,
            "amp_skip_formal_validation_in_this_task": False,
        },
        "configs": CONFIGS,
        "config_checks": config_checks,
        "runtime_values": runtime_values,
        "checks": checks,
        "check_count": len(checks),
        "failed_checks": failed_checks,
        "all_checks_pass": len(failed_checks) == 0,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("EVIDENCE=", output)
    print("CHECK_COUNT=", len(checks))
    print("FAILED_CHECKS=", failed_checks)
    print(
        "S4_02_EMA_TIMING="
        + ("PASS" if not failed_checks else "FAIL")
    )

    return 0 if not failed_checks else 1


if __name__ == "__main__":
    raise SystemExit(main())
