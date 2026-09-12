"""Validate S4.01 Student + Teacher initialization contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.utils import import_modules_from_strings
from mmdet.registry import HOOKS, MODELS
from mmdet.utils import register_all_modules


EXPECTED = {
    "r50_config": {
        "path": "configs/ssl/s4_01_soft_teacher_r50_fpn.py",
        "sha256": "264a7d8f0d3bd1b975de41ee01d78a6b4b2143a3fdeb36747c3f5ca895ee967c",
    },
    "swin_config": {
        "path": "configs/ssl/s4_01_soft_teacher_swin_t_fpn.py",
        "sha256": "b89aa4f8127e38abd2243f0c9a89fcbacd0b00651c7bfc3490a3368460f1811e",
    },
    "init_hook": {
        "path": "src/hooks/teacher_initialization_hook.py",
        "sha256": "8a56444c94045ac7f0cfd9e7812d79a599655389074fbee8731c0b9179f439bc",
    },
}

SWIN_CHECKPOINT = Path(
    "/workspace/ssod/cache/pretrained/swin_tiny_patch4_window7_224.pth"
)
SWIN_CHECKPOINT_SHA256 = (
    "9f71c168d837d1b99dd1dc29e14990a7a9e8bdc5f673d46b04fe36fe15590ad3"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def full_state_equal(student, teacher) -> bool:
    s = student.state_dict()
    t = teacher.state_dict()
    return (
        s.keys() == t.keys()
        and all(torch.equal(s[k], t[k]) for k in s)
    )


def validate_model(name: str, config_path: Path) -> dict:
    cfg = Config.fromfile(str(config_path))
    import_modules_from_strings(**cfg.custom_imports)

    hook_cfg = dict(cfg.custom_hooks[0])
    priority = hook_cfg.pop("priority", "NORMAL")
    hook = HOOKS.build(hook_cfg)

    model = MODELS.build(cfg.model)
    model.init_weights()

    runner = type("RunnerStub", (), {})()
    runner.model = model
    runner.iter = 0

    pre_equal = full_state_equal(model.student, model.teacher)
    hook.before_train(runner)

    fresh_equal = full_state_equal(model.student, model.teacher)
    teacher_frozen = all(
        not p.requires_grad for p in model.teacher.parameters()
    )
    student_trainable = any(
        p.requires_grad for p in model.student.parameters()
    )

    teacher_param = next(iter(model.teacher.parameters()))
    before_resume_probe = teacher_param.detach().clone()
    teacher_param.data.add_(1.0)
    changed = not torch.equal(
        teacher_param.detach(), before_resume_probe
    )

    runner.iter = 7
    hook.before_train(runner)
    resume_not_resynced = (
        changed
        and not torch.equal(
            teacher_param.detach(), before_resume_probe
        )
    )

    checks = {
        "model_is_softteacher": cfg.model.type == "SoftTeacher",
        "detector_is_faster_rcnn": cfg.model.detector.type == "FasterRCNN",
        "teacher_frozen_by_config":
            cfg.model.semi_train_cfg.freeze_teacher is True,
        "semi_train_cfg_scope_is_s4_01_only":
            set(cfg.model.semi_train_cfg.keys()) == {"freeze_teacher"},
        "teacher_initialization_hook_registered":
            type(hook).__name__ == "TeacherInitializationHook",
        "teacher_initialization_hook_priority_very_high":
            priority == "VERY_HIGH",
        "hook_has_no_custom_after_train_iter":
            "after_train_iter" not in type(hook).__dict__,
        "fresh_teacher_equals_student_full_state": fresh_equal,
        "teacher_all_frozen": teacher_frozen,
        "student_has_trainable_parameters": student_trainable,
        "resume_teacher_not_resynchronized": resume_not_resynced,
    }

    return {
        "name": name,
        "config": str(config_path),
        "pre_hook_full_state_equal": pre_equal,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default="/workspace/ssod/project",
    )
    parser.add_argument(
        "--output",
        default="/workspace/ssod/project/artifacts/preflight/ssl/"
                "teacher_initialization_test.json",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    output = Path(args.output).resolve()

    register_all_modules()

    checks = {}
    identities = {}

    for key, spec in EXPECTED.items():
        path = repo_root / spec["path"]
        observed = sha256(path) if path.is_file() else None
        identities[key] = {
            "path": str(path),
            "expected_sha256": spec["sha256"],
            "observed_sha256": observed,
        }
        checks[f"{key}_exists"] = path.is_file()
        checks[f"{key}_sha256_match"] = observed == spec["sha256"]

    swin_observed = (
        sha256(SWIN_CHECKPOINT)
        if SWIN_CHECKPOINT.is_file()
        else None
    )
    checks["swin_checkpoint_exists"] = SWIN_CHECKPOINT.is_file()
    checks["swin_checkpoint_sha256_match"] = (
        swin_observed == SWIN_CHECKPOINT_SHA256
    )

    model_results = []
    for name, key in (("R50", "r50_config"), ("SWIN", "swin_config")):
        result = validate_model(
            name,
            repo_root / EXPECTED[key]["path"],
        )
        model_results.append(result)
        checks[f"{name.lower()}_initialization_contract"] = (
            result["all_checks_pass"]
        )

    failed_checks = [
        name for name, passed in checks.items() if not passed
    ]
    all_checks_pass = not failed_checks

    report = {
        "schema_version": "1.0.0",
        "stage": "S4",
        "task": "S4.01",
        "scope": "Student + EMA Teacher initialization",
        "status": "PASS" if all_checks_pass else "FAIL",
        "official_training_authorized": False,
        "contract": {
            "teacher_t0_equals_student_t0": True,
            "supervised_burn_in": False,
            "resume_resynchronization_forbidden": True,
            "ema_update_timing_validated_here": False,
            "amp_ema_skip_validated_here": False,
        },
        "implementation_identity": identities,
        "swin_checkpoint": {
            "path": str(SWIN_CHECKPOINT),
            "expected_sha256": SWIN_CHECKPOINT_SHA256,
            "observed_sha256": swin_observed,
        },
        "models": model_results,
        "checks": checks,
        "check_count": len(checks),
        "failed_checks": failed_checks,
        "all_checks_pass": all_checks_pass,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"EVIDENCE={output}")
    print(f"CHECK_COUNT={len(checks)}")
    print(f"FAILED_CHECKS={failed_checks}")
    print(f"S4_01_TEACHER_INITIALIZATION={'PASS' if all_checks_pass else 'FAIL'}")
    return 0 if all_checks_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())