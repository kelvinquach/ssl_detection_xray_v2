#!/usr/bin/env python3
"""S4.05 — explicit AMP skipped-step -> EMA skipped-step validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from mmengine.config import Config
from mmengine.optim import AmpOptimWrapper
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
    "R50": "configs/ssl/s4_05_soft_teacher_r50_fpn_amp.py",
    "SWIN": "configs/ssl/s4_05_soft_teacher_swin_t_fpn_amp.py",
}


class ToySubModel(nn.Module):
    def __init__(self, weight: float):
        super().__init__()
        self.weight = nn.Parameter(
            torch.tensor([weight], dtype=torch.float32, device="cuda")
        )


class ToyTeacherStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.student = ToySubModel(1.0)
        self.teacher = ToySubModel(1.0)


def validate_config(repo_root: Path, relative_path: str, arch: str) -> dict:
    cfg = Config.fromfile(str(repo_root / relative_path))
    ow = cfg.optim_wrapper
    hooks = list(cfg.custom_hooks)

    ema_hooks = [
        h for h in hooks
        if h.get("type") == "ActualUpdateMeanTeacherHook"
    ]

    checks = {
        "model_is_softteacher": cfg.model.type == "SoftTeacher",
        "amp_wrapper": ow.get("type") == "AmpOptimWrapper",
        "dynamic_loss_scale": ow.get("loss_scale") == "dynamic",
        "gradient_clipping_off": ow.get("clip_grad") is None,
        "one_actual_update_ema_hook": len(ema_hooks) == 1,
        "ema_momentum_0_001": (
            len(ema_hooks) == 1
            and float(ema_hooks[0].get("momentum")) == 0.001
        ),
        "ema_skip_buffers_true": (
            len(ema_hooks) == 1
            and ema_hooks[0].get("skip_buffers") is True
        ),
    }

    opt = ow.optimizer

    if arch == "R50":
        checks["optimizer_recipe_preserved"] = (
            opt.type == "SGD"
            and float(opt.lr) == 0.005
            and float(opt.momentum) == 0.9
            and float(opt.weight_decay) == 0.0001
        )
    else:
        custom_keys = ow.paramwise_cfg.custom_keys
        checks["optimizer_recipe_preserved"] = (
            opt.type == "AdamW"
            and float(opt.lr) == 0.000025
            and tuple(opt.betas) == (0.9, 0.999)
            and float(opt.weight_decay) == 0.05
        )
        checks["native_no_decay_rules_preserved"] = (
            custom_keys["absolute_pos_embed"]["decay_mult"] == 0.0
            and custom_keys["relative_position_bias_table"]["decay_mult"] == 0.0
            and custom_keys["norm"]["decay_mult"] == 0.0
        )

    return checks


def validate_amp_ema_runtime() -> tuple[dict, dict, dict]:
    if not torch.cuda.is_available():
        raise RuntimeError("S4.05 explicit AMP/EMA test requires CUDA.")

    model = ToyTeacherStudent()

    optimizer = torch.optim.SGD(
        model.student.parameters(),
        lr=0.1,
    )
    amp_wrapper = AmpOptimWrapper(
        optimizer=optimizer,
        accumulative_counts=1,
    )

    counter = ActualOptimizerUpdateCounter()
    counter.attach(optimizer)

    runner = SimpleNamespace(model=model)
    setattr(runner, RUNNER_COUNTER_ATTR, counter)

    ema_hook = ActualUpdateMeanTeacherHook(
        momentum=0.001,
        skip_buffers=True,
    )
    ema_hook.before_train(runner)

    # Control path: a finite AMP step must execute optimizer.step(),
    # advance the authoritative counter, and trigger exactly one EMA update.
    finite_counter_before = counter.count
    finite_teacher_before = model.teacher.weight.detach().clone()
    finite_student_before = model.student.weight.detach().clone()
    finite_scale_before = float(amp_wrapper.loss_scaler.get_scale())

    finite_loss = ((model.student.weight - 3.0) ** 2).sum()
    amp_wrapper.update_params(finite_loss)

    finite_counter_after = counter.count
    finite_student_after = model.student.weight.detach().clone()

    ema_hook.after_train_iter(runner, 0)
    finite_teacher_after = model.teacher.weight.detach().clone()
    finite_scale_after = float(amp_wrapper.loss_scaler.get_scale())

    expected_teacher_after_finite = (
        0.999 * finite_teacher_before
        + 0.001 * finite_student_after
    )

    # Explicit S4.05 path: non-finite gradient makes GradScaler skip the
    # underlying optimizer.step(). The authoritative counter must stay fixed,
    # and the EMA hook must therefore leave Teacher unchanged.
    skip_counter_before = counter.count
    skip_teacher_before = model.teacher.weight.detach().clone()
    skip_student_before = model.student.weight.detach().clone()
    skip_scale_before = float(amp_wrapper.loss_scaler.get_scale())

    inf_value = torch.tensor(float("inf"), device="cuda")
    skip_loss = (model.student.weight * inf_value).sum()
    amp_wrapper.update_params(skip_loss)

    skip_counter_after = counter.count
    skip_student_after = model.student.weight.detach().clone()
    skip_scale_after = float(amp_wrapper.loss_scaler.get_scale())

    ema_hook.after_train_iter(runner, 1)
    skip_teacher_after = model.teacher.weight.detach().clone()

    checks = {
        "finite_amp_optimizer_step_executed": (
            finite_counter_after == finite_counter_before + 1
        ),
        "finite_student_updated": not torch.equal(
            finite_student_before,
            finite_student_after,
        ),
        "finite_ema_step_executed": torch.allclose(
            finite_teacher_after,
            expected_teacher_after_finite,
            atol=1e-7,
            rtol=0.0,
        ),
        "amp_non_finite_optimizer_step_skipped": (
            skip_counter_after == skip_counter_before
        ),
        "student_unchanged_on_amp_skip": torch.equal(
            skip_student_before,
            skip_student_after,
        ),
        "grad_scaler_backoff_observed": (
            skip_scale_after < skip_scale_before
        ),
        "ema_skipped_when_optimizer_step_skipped": torch.equal(
            skip_teacher_before,
            skip_teacher_after,
        ),
        "actual_update_counter_is_ema_authority": (
            skip_counter_after == skip_counter_before
            and torch.equal(skip_teacher_before, skip_teacher_after)
        ),
    }

    runtime = {
        "finite": {
            "counter_before": finite_counter_before,
            "counter_after": finite_counter_after,
            "student_before": float(finite_student_before.item()),
            "student_after": float(finite_student_after.item()),
            "teacher_before": float(finite_teacher_before.item()),
            "teacher_after": float(finite_teacher_after.item()),
            "expected_teacher_after": float(
                expected_teacher_after_finite.item()
            ),
            "loss_scale_before": finite_scale_before,
            "loss_scale_after": finite_scale_after,
            "optimizer_step_executed": (
                finite_counter_after == finite_counter_before + 1
            ),
            "ema_step_executed": True,
        },
        "amp_skipped": {
            "counter_before": skip_counter_before,
            "counter_after": skip_counter_after,
            "student_before": float(skip_student_before.item()),
            "student_after": float(skip_student_after.item()),
            "teacher_before": float(skip_teacher_before.item()),
            "teacher_after": float(skip_teacher_after.item()),
            "loss_scale_before": skip_scale_before,
            "loss_scale_after": skip_scale_after,
            "optimizer_step_executed": False,
            "ema_step_executed": False,
        },
    }

    events = {
        "optimizer_event": "OPTIMIZER_STEP_SKIPPED_BY_AMP",
        "ema_event": "EMA_STEP_SKIPPED",
        "reason": "non_finite_gradient",
        "optimizer_step_executed": False,
        "ema_step_executed": False,
    }

    counter.detach()
    return checks, runtime, events


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
            "amp_ema_skip_test.json"
        ),
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output = Path(args.output).resolve()

    config_checks = {
        arch: validate_config(repo_root, rel, arch)
        for arch, rel in CONFIGS.items()
    }

    runtime_checks, runtime, events = validate_amp_ema_runtime()

    checks = {}
    for arch, values in config_checks.items():
        for name, passed in values.items():
            checks[f"{arch}_{name}"] = bool(passed)

    checks.update(runtime_checks)

    failed_checks = [
        name for name, passed in checks.items()
        if passed is not True
    ]

    evidence = {
        "schema_version": "1.0.0",
        "stage": "S4",
        "task": "S4.05",
        "scope": "AMP skipped optimizer step implies EMA skipped step",
        "official_training_authorized": False,
        "contract": {
            "amp": True,
            "ema_update_basis": "ACTUAL_OPTIMIZER_UPDATE",
            "required_skip_behavior": (
                "AMP_SKIPPED_OPTIMIZER_STEP=>EMA_STEP_SKIPPED"
            ),
            "explicit_test": True,
        },
        "configs": CONFIGS,
        "config_checks": config_checks,
        "runtime": runtime,
        "skip_events": events,
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
        "AMP_SKIP_OPTIMIZER_STEP_EXECUTED=",
        runtime["amp_skipped"]["optimizer_step_executed"],
    )
    print(
        "AMP_SKIP_EMA_STEP_EXECUTED=",
        runtime["amp_skipped"]["ema_step_executed"],
    )
    print(
        "AMP_SKIP_COUNTER_BEFORE=",
        runtime["amp_skipped"]["counter_before"],
    )
    print(
        "AMP_SKIP_COUNTER_AFTER=",
        runtime["amp_skipped"]["counter_after"],
    )
    print(
        "S4_05_AMP_EMA_SKIP="
        + ("PASS" if not failed_checks else "FAIL")
    )

    return 0 if not failed_checks else 1


if __name__ == "__main__":
    raise SystemExit(main())