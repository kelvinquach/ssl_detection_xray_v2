from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.model import BaseModel
from mmengine.runner import Runner
from torch.utils.data import DataLoader

EXPECTED_UPDATES = 1032
EXPECTED_MILESTONES = [688, 946]
EXPECTED_END = 1033

CONFIGS = {
    "R50": "configs/supervised/s3_08_faster_rcnn_r50_fpn_sup_1pct.py",
    "SWIN_T": "configs/supervised/s3_08_faster_rcnn_swin_t_fpn_sup_1pct.py",
}


class ToyModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(0.0))
        self.calls = 0

    def forward(self, *args, **kwargs):
        return self.w

    def train_step(self, data, optim_wrapper):
        self.calls += 1
        loss = (self.w - 1.0) ** 2
        if self.calls != 2:
            optim_wrapper.update_params(loss)
        return {"loss": loss.detach()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/workspace/ssod/project")
    parser.add_argument(
        "--output",
        default="/workspace/ssod/artifacts/preflight/sup/schedule_1pct_preflight.json",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo_root))

    checks = {}
    architectures = {}

    for arch, rel_path in CONFIGS.items():
        cfg = Config.fromfile(str(repo_root / rel_path))
        scheduler = cfg.param_scheduler[0]
        hook_types = [h["type"] for h in cfg.custom_hooks]

        arch_checks = {
            "max_iters_is_1032": cfg.train_cfg.max_iters == EXPECTED_UPDATES,
            "target_updates_is_1032": (
                cfg.custom_hooks[1].target_updates == EXPECTED_UPDATES
            ),
            "milestones_are_688_946": (
                list(scheduler.milestones) == EXPECTED_MILESTONES
            ),
            "scheduler_end_is_1033": scheduler.end == EXPECTED_END,
            "scheduler_is_update_based": scheduler.by_epoch is False,
            "default_scheduler_disabled": (
                cfg.default_hooks.param_scheduler is None
            ),
            "default_checkpoint_disabled": (
                cfg.default_hooks.checkpoint is None
            ),
            "required_custom_hooks_present": hook_types == [
                "ProtocolResumeCheckpointHook",
                "ActualUpdateBudgetSchedulerHook",
            ],
        }

        checks.update({
            f"{arch.lower()}_{name}": passed
            for name, passed in arch_checks.items()
        })
        architectures[arch] = {
            "config": rel_path,
            "checks": arch_checks,
        }

    import src.utils.resume_aware_loop  # noqa: F401
    import src.utils.resume_checkpoint_hook  # noqa: F401
    import src.utils.actual_update_budget_hook  # noqa: F401
    from src.utils.resume_checkpoint_hook import RUNNER_COUNTER_ATTR

    runner = Runner(
        model=ToyModel(),
        work_dir="/tmp/s3_08_validator",
        train_dataloader=DataLoader(list(range(EXPECTED_UPDATES + 1)), batch_size=1),
        train_cfg=dict(
            type="ResumeAwareIterBasedTrainLoop",
            max_iters=EXPECTED_UPDATES,
            val_interval=999,
        ),
        optim_wrapper=dict(
            type="OptimWrapper",
            optimizer=dict(type="SGD", lr=0.1),
        ),
        param_scheduler=[
            dict(
                type="MultiStepLR",
                by_epoch=False,
                end=EXPECTED_END,
                milestones=EXPECTED_MILESTONES,
                gamma=0.1,
            )
        ],
        default_hooks=dict(
            runtime_info=None,
            timer=None,
            sampler_seed=None,
            logger=None,
            param_scheduler=None,
            checkpoint=None,
        ),
        custom_hooks=[
            dict(type="ProtocolResumeCheckpointHook"),
            dict(type="ActualUpdateBudgetSchedulerHook", target_updates=EXPECTED_UPDATES),
        ],
        randomness=dict(seed=42),
        log_level="ERROR",
    )

    runner.train()

    actual_updates = getattr(runner, RUNNER_COUNTER_ATTR).count
    runtime_checks = {
        "skip_does_not_count_as_update": runner.iter == EXPECTED_UPDATES + 1,
        "actual_update_target_reached": actual_updates == EXPECTED_UPDATES,
        "scheduler_tracks_actual_updates": (
            runner.param_schedulers[0].last_step == EXPECTED_UPDATES
        ),
        "raw_ceiling_extended_after_skip": runner.max_iters == EXPECTED_UPDATES + 1,
        "second_milestone_applied": abs(
            runner.optim_wrapper.optimizer.param_groups[0]["lr"] - 0.001
        ) < 1e-12,
    }
    checks.update(runtime_checks)

    checks["official_training_remains_unauthorized"] = True
    failed_checks = [name for name, passed in checks.items() if not passed]

    report = {
        "schema_version": "1.0",
        "stage": "S3.08",
        "scope": "supervised_1pct_actual_update_scheduler_preflight",
        "official_training_authorized": False,
        "scientific_contract": {
            "optimizer_updates": EXPECTED_UPDATES,
            "milestones": EXPECTED_MILESTONES,
        },
        "architectures": architectures,
        "runtime_fixture": {
            "raw_iterations": runner.iter,
            "actual_optimizer_updates": actual_updates,
            "scheduler_last_step": runner.param_schedulers[0].last_step,
            "final_max_iters": runner.max_iters,
        },
        "checks": checks,
        "check_count": len(checks),
        "failed_checks": failed_checks,
        "all_checks_pass": not failed_checks,
    }

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("check_count =", len(checks))
    print("failed_checks =", failed_checks)
    print("ALL_CHECKS_PASS =", not failed_checks)

    if failed_checks:
        print("S3_08_1PCT_SCHEDULE=FAIL")
        return 1

    print("S3_08_1PCT_SCHEDULE=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())