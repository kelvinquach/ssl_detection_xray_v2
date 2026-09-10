from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.model import BaseModel
from mmengine.runner import Runner
from mmengine.runner.base_loop import BaseLoop
from torch.utils.data import DataLoader


VALIDATION_INTERVAL = 172
RUNTIME_TARGET_UPDATES = 344
RUNTIME_SKIP_CALLS = {2, 174}
EXPECTED_BATCH_SIZE = 4
EXPECTED_ACCUMULATION = 1

BUDGET_SPECS = {
    "1pct": {
        "updates": 1032,
        "ann_file": "instances_labeled_1pct.json",
        "milestones": [688, 946],
        "scheduler_end": 1033,
    },
    "5pct": {
        "updates": 2064,
        "ann_file": "instances_labeled_5pct.json",
        "milestones": [1376, 1892],
        "scheduler_end": 2065,
    },
    "10pct": {
        "updates": 2064,
        "ann_file": "instances_labeled_10pct.json",
        "milestones": [1376, 1892],
        "scheduler_end": 2065,
    },
    "20pct": {
        "updates": 2064,
        "ann_file": "instances_labeled_20pct.json",
        "milestones": [1376, 1892],
        "scheduler_end": 2065,
    },
    "100pct": {
        "updates": 10284,
        "ann_file": "instances_train.json",
        "milestones": [6856, 9427],
        "scheduler_end": 10285,
    },
}

CONFIGS = {
    "R50": {
        budget: f"configs/supervised/s3_11_faster_rcnn_r50_fpn_sup_{budget}.py"
        for budget in BUDGET_SPECS
    },
    "SWIN_T": {
        budget: f"configs/supervised/s3_11_faster_rcnn_swin_t_fpn_sup_{budget}.py"
        for budget in BUDGET_SPECS
    },
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
        if self.calls not in RUNTIME_SKIP_CALLS:
            optim_wrapper.update_params(loss)
        return {"loss": loss.detach()}


class ToyValLoop(BaseLoop):
    def __init__(self, runner):
        super().__init__(runner=runner, dataloader=[])
        self.calls = []

    def run(self):
        from src.utils.resume_checkpoint_hook import RUNNER_COUNTER_ATTR

        counter = getattr(self._runner, RUNNER_COUNTER_ATTR)
        self.calls.append(
            {
                "actual_optimizer_updates": counter.count,
                "raw_iterations_at_val_call": self._runner.iter,
            }
        )
        return {"toy_metric": float(counter.count)}


def expected_validation_updates(total_updates: int) -> list[int]:
    return list(
        range(
            VALIDATION_INTERVAL,
            total_updates + 1,
            VALIDATION_INTERVAL,
        )
    )


def structural_signature(cfg: Config) -> dict:
    scheduler = cfg.param_scheduler[0]
    hook_types = [h["type"] for h in cfg.custom_hooks]
    target_hook = [
        h
        for h in cfg.custom_hooks
        if h["type"] == "ActualUpdateBudgetSchedulerHook"
    ][0]

    return {
        "train_loop_type": cfg.train_cfg.type,
        "max_iters": cfg.train_cfg.max_iters,
        "val_interval": cfg.train_cfg.val_interval,
        "batch_size": cfg.train_dataloader.batch_size,
        "accumulative_counts": cfg.optim_wrapper.accumulative_counts,
        "scheduler_type": scheduler.type,
        "scheduler_by_epoch": scheduler.by_epoch,
        "scheduler_end": scheduler.end,
        "scheduler_milestones": list(scheduler.milestones),
        "scheduler_gamma": scheduler.gamma,
        "default_param_scheduler": cfg.default_hooks.param_scheduler,
        "default_checkpoint": cfg.default_hooks.checkpoint,
        "custom_hook_types": hook_types,
        "target_updates": target_hook.target_updates,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/workspace/ssod/project")
    parser.add_argument(
        "--output",
        default=(
            "/workspace/ssod/artifacts/preflight/sup/"
            "validation_172_actual_update_preflight.json"
        ),
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo_root))

    checks = {}
    architectures = {}
    signatures = {}

    for arch, budget_configs in CONFIGS.items():
        architectures[arch] = {}
        signatures[arch] = {}

        for budget, rel_path in budget_configs.items():
            spec = BUDGET_SPECS[budget]
            cfg = Config.fromfile(str(repo_root / rel_path))
            scheduler = cfg.param_scheduler[0]
            hook_types = [h["type"] for h in cfg.custom_hooks]
            target_hook = [
                h
                for h in cfg.custom_hooks
                if h["type"] == "ActualUpdateBudgetSchedulerHook"
            ][0]

            ann_file = str(cfg.train_dataloader.dataset.ann_file)
            expected_history = expected_validation_updates(spec["updates"])

            config_checks = {
                "dataset_matches_budget": ann_file.endswith(spec["ann_file"]),
                "actual_update_validation_loop_enabled": (
                    cfg.train_cfg.type
                    == "ActualUpdateValidationIterBasedTrainLoop"
                ),
                "validation_interval_is_172": (
                    cfg.train_cfg.val_interval == VALIDATION_INTERVAL
                ),
                "max_iters_preserved": (
                    cfg.train_cfg.max_iters == spec["updates"]
                ),
                "target_updates_preserved": (
                    target_hook.target_updates == spec["updates"]
                ),
                "scheduler_milestones_preserved": (
                    list(scheduler.milestones) == spec["milestones"]
                ),
                "scheduler_end_preserved": (
                    scheduler.end == spec["scheduler_end"]
                ),
                "scheduler_is_update_based": scheduler.by_epoch is False,
                "effective_batch_components_preserved": (
                    cfg.train_dataloader.batch_size == EXPECTED_BATCH_SIZE
                    and cfg.optim_wrapper.accumulative_counts
                    == EXPECTED_ACCUMULATION
                ),
                "default_scheduler_disabled": (
                    cfg.default_hooks.param_scheduler is None
                ),
                "default_checkpoint_disabled": (
                    cfg.default_hooks.checkpoint is None
                ),
                "required_custom_hooks_preserved": hook_types == [
                    "ProtocolResumeCheckpointHook",
                    "ActualUpdateBudgetSchedulerHook",
                ],
                "expected_validation_history_nonempty": bool(expected_history),
            }

            prefix = f"{arch.lower()}_{budget}"
            checks.update(
                {
                    f"{prefix}_{name}": passed
                    for name, passed in config_checks.items()
                }
            )

            signature = structural_signature(cfg)
            signatures[arch][budget] = signature

            architectures[arch][budget] = {
                "config": rel_path,
                "ann_file": ann_file,
                "checks": config_checks,
                "structural_signature": signature,
                "expected_validation_history": expected_history,
                "expected_validation_count": len(expected_history),
                "last_periodic_validation_update": expected_history[-1],
            }

    for budget in BUDGET_SPECS:
        r50 = signatures["R50"][budget]
        swin = signatures["SWIN_T"][budget]
        checks[f"{budget}_r50_swint_validation_structure_match"] = (
            r50 == swin
        )

    checks["all_configs_use_same_validation_interval"] = all(
        signatures[arch][budget]["val_interval"] == VALIDATION_INTERVAL
        for arch in signatures
        for budget in signatures[arch]
    )

    checks["100pct_last_periodic_validation_is_10148"] = (
        expected_validation_updates(10284)[-1] == 10148
    )
    checks["100pct_does_not_force_terminal_validation"] = (
        10284 not in expected_validation_updates(10284)
    )

    import src.utils.resume_aware_loop  # noqa: F401
    import src.utils.resume_checkpoint_hook  # noqa: F401
    import src.utils.actual_update_budget_hook  # noqa: F401
    import src.utils.actual_update_validation_loop  # noqa: F401
    from src.utils.actual_update_validation_loop import (
        RUNNER_VALIDATION_HISTORY_ATTR,
    )
    from src.utils.resume_checkpoint_hook import RUNNER_COUNTER_ATTR

    runner = Runner(
        model=ToyModel(),
        work_dir="/tmp/s3_11_validator",
        train_dataloader=DataLoader(
            list(
                range(
                    RUNTIME_TARGET_UPDATES
                    + len(RUNTIME_SKIP_CALLS)
                    + 2
                )
            ),
            batch_size=1,
        ),
        train_cfg=dict(
            type="ActualUpdateValidationIterBasedTrainLoop",
            max_iters=RUNTIME_TARGET_UPDATES,
            val_interval=VALIDATION_INTERVAL,
        ),
        optim_wrapper=dict(
            type="OptimWrapper",
            optimizer=dict(type="SGD", lr=0.1),
        ),
        param_scheduler=[
            dict(
                type="MultiStepLR",
                by_epoch=False,
                end=RUNTIME_TARGET_UPDATES + 1,
                milestones=[172, 300],
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
            dict(
                type="ActualUpdateBudgetSchedulerHook",
                target_updates=RUNTIME_TARGET_UPDATES,
            ),
        ],
        randomness=dict(seed=42),
        log_level="ERROR",
    )

    toy_val_loop = ToyValLoop(runner)
    runner._val_loop = toy_val_loop

    runner.train()

    actual_updates = getattr(runner, RUNNER_COUNTER_ATTR).count
    history = getattr(runner, RUNNER_VALIDATION_HISTORY_ATTR)
    history_updates = [
        item["actual_optimizer_updates"]
        for item in history
    ]
    val_call_updates = [
        item["actual_optimizer_updates"]
        for item in toy_val_loop.calls
    ]

    expected_runtime_history = [172, 344]

    runtime_checks = {
        "two_skips_do_not_count_as_updates": (
            runner.iter
            == RUNTIME_TARGET_UPDATES + len(RUNTIME_SKIP_CALLS)
        ),
        "actual_update_target_reached": (
            actual_updates == RUNTIME_TARGET_UPDATES
        ),
        "scheduler_tracks_actual_updates": (
            runner.param_schedulers[0].last_step
            == RUNTIME_TARGET_UPDATES
        ),
        "validation_history_exact_actual_updates": (
            history_updates == expected_runtime_history
        ),
        "val_loop_calls_exact_actual_updates": (
            val_call_updates == expected_runtime_history
        ),
        "validation_count_is_two": len(history) == 2,
        "skip_after_172_does_not_duplicate_validation": (
            history_updates.count(172) == 1
        ),
        "first_validation_shifted_by_earlier_skip": (
            history[0]["raw_iterations"] == 173
        ),
        "second_validation_shifted_by_two_skips": (
            history[1]["raw_iterations"] == 346
        ),
        "raw_iteration_not_used_as_validation_authority": (
            history[0]["raw_iterations"]
            != history[0]["actual_optimizer_updates"]
            and history[1]["raw_iterations"]
            != history[1]["actual_optimizer_updates"]
        ),
    }
    checks.update(runtime_checks)

    checks["official_training_remains_unauthorized"] = True

    failed_checks = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    report = {
        "schema_version": "1.0",
        "stage": "S3.11",
        "scope": "supervised_validation_every_172_actual_optimizer_updates",
        "official_training_authorized": False,
        "scientific_contract": {
            "validation_interval_actual_optimizer_updates": (
                VALIDATION_INTERVAL
            ),
            "basis": "actual_optimizer_update",
            "raw_iteration_is_not_validation_authority": True,
        },
        "architectures": architectures,
        "runtime_fixture": {
            "target_actual_optimizer_updates": RUNTIME_TARGET_UPDATES,
            "skipped_raw_calls": sorted(RUNTIME_SKIP_CALLS),
            "raw_iterations": runner.iter,
            "actual_optimizer_updates": actual_updates,
            "scheduler_last_step": runner.param_schedulers[0].last_step,
            "validation_history": history,
            "val_loop_observed_history": toy_val_loop.calls,
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
    print("validation_history =", history)
    print("ALL_CHECKS_PASS =", not failed_checks)

    if failed_checks:
        print("S3_11_ACTUAL_UPDATE_VALIDATION=FAIL")
        return 1

    print("S3_11_ACTUAL_UPDATE_VALIDATION=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())