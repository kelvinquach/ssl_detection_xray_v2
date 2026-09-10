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

EXPECTED_UPDATES = 2064
EXPECTED_MILESTONES = [1376, 1892]
EXPECTED_END = 2065
EXPECTED_BATCH_SIZE = 4
EXPECTED_ACCUMULATION = 1

CONFIGS = {
    "R50": {
        "5pct": "configs/supervised/s3_09_faster_rcnn_r50_fpn_sup_5pct.py",
        "10pct": "configs/supervised/s3_09_faster_rcnn_r50_fpn_sup_10pct.py",
        "20pct": "configs/supervised/s3_09_faster_rcnn_r50_fpn_sup_20pct.py",
    },
    "SWIN_T": {
        "5pct": "configs/supervised/s3_09_faster_rcnn_swin_t_fpn_sup_5pct.py",
        "10pct": "configs/supervised/s3_09_faster_rcnn_swin_t_fpn_sup_10pct.py",
        "20pct": "configs/supervised/s3_09_faster_rcnn_swin_t_fpn_sup_20pct.py",
    },
}

EXPECTED_ANN_FILES = {
    "5pct": "instances_labeled_5pct.json",
    "10pct": "instances_labeled_10pct.json",
    "20pct": "instances_labeled_20pct.json",
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


def structural_signature(cfg: Config) -> dict:
    scheduler = cfg.param_scheduler[0]
    hook_types = [h["type"] for h in cfg.custom_hooks]
    target_hook = [
        h for h in cfg.custom_hooks
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
        default="/workspace/ssod/artifacts/preflight/sup/schedule_2064_preflight.json",
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
            cfg = Config.fromfile(str(repo_root / rel_path))
            scheduler = cfg.param_scheduler[0]
            hook_types = [h["type"] for h in cfg.custom_hooks]
            target_hook = [
                h for h in cfg.custom_hooks
                if h["type"] == "ActualUpdateBudgetSchedulerHook"
            ][0]

            ann_file = str(cfg.train_dataloader.dataset.ann_file)
            expected_ann = EXPECTED_ANN_FILES[budget]

            config_checks = {
                "dataset_matches_budget": ann_file.endswith(expected_ann),
                "max_iters_is_2064": cfg.train_cfg.max_iters == EXPECTED_UPDATES,
                "target_updates_is_2064": (
                    target_hook.target_updates == EXPECTED_UPDATES
                ),
                "milestones_are_1376_1892": (
                    list(scheduler.milestones) == EXPECTED_MILESTONES
                ),
                "scheduler_end_is_2065": scheduler.end == EXPECTED_END,
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
                "required_custom_hooks_present": hook_types == [
                    "ProtocolResumeCheckpointHook",
                    "ActualUpdateBudgetSchedulerHook",
                ],
            }

            prefix = f"{arch.lower()}_{budget}"
            checks.update({
                f"{prefix}_{name}": passed
                for name, passed in config_checks.items()
            })

            signature = structural_signature(cfg)
            signatures[arch][budget] = signature

            architectures[arch][budget] = {
                "config": rel_path,
                "ann_file": ann_file,
                "checks": config_checks,
                "structural_signature": signature,
            }

        reference = signatures[arch]["5pct"]
        equivalence = {
            budget: signatures[arch][budget] == reference
            for budget in ("10pct", "20pct")
        }
        checks[f"{arch.lower()}_5_10_20_structural_equivalence"] = all(
            equivalence.values()
        )
        architectures[arch]["structural_equivalence"] = {
            "reference_budget": "5pct",
            "allowed_difference": "dataset identity only",
            "equivalent_to_reference": equivalence,
            "all_equivalent": all(equivalence.values()),
        }

    schedule_signatures = []
    for arch in CONFIGS:
        for budget in CONFIGS[arch]:
            sig = signatures[arch][budget]
            schedule_signatures.append({
                "train_loop_type": sig["train_loop_type"],
                "max_iters": sig["max_iters"],
                "scheduler_type": sig["scheduler_type"],
                "scheduler_by_epoch": sig["scheduler_by_epoch"],
                "scheduler_end": sig["scheduler_end"],
                "scheduler_milestones": sig["scheduler_milestones"],
                "scheduler_gamma": sig["scheduler_gamma"],
                "target_updates": sig["target_updates"],
            })

    checks["all_six_share_same_2064_schedule_structure"] = all(
        sig == schedule_signatures[0]
        for sig in schedule_signatures[1:]
    )

    import src.utils.resume_aware_loop  # noqa: F401
    import src.utils.resume_checkpoint_hook  # noqa: F401
    import src.utils.actual_update_budget_hook  # noqa: F401
    from src.utils.resume_checkpoint_hook import RUNNER_COUNTER_ATTR

    runner = Runner(
        model=ToyModel(),
        work_dir="/tmp/s3_09_validator",
        train_dataloader=DataLoader(
            list(range(EXPECTED_UPDATES + 1)),
            batch_size=1,
        ),
        train_cfg=dict(
            type="ResumeAwareIterBasedTrainLoop",
            max_iters=EXPECTED_UPDATES,
            val_interval=9999,
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
            dict(
                type="ActualUpdateBudgetSchedulerHook",
                target_updates=EXPECTED_UPDATES,
            ),
        ],
        randomness=dict(seed=42),
        log_level="ERROR",
    )

    runner.train()

    actual_updates = getattr(runner, RUNNER_COUNTER_ATTR).count
    runtime_checks = {
        "skip_does_not_count_as_update": (
            runner.iter == EXPECTED_UPDATES + 1
        ),
        "actual_update_target_reached": (
            actual_updates == EXPECTED_UPDATES
        ),
        "scheduler_tracks_actual_updates": (
            runner.param_schedulers[0].last_step == EXPECTED_UPDATES
        ),
        "raw_ceiling_extended_after_skip": (
            runner.max_iters == EXPECTED_UPDATES + 1
        ),
        "second_milestone_applied": abs(
            runner.optim_wrapper.optimizer.param_groups[0]["lr"] - 0.001
        ) < 1e-12,
    }
    checks.update(runtime_checks)

    checks["official_training_remains_unauthorized"] = True

    failed_checks = [
        name for name, passed in checks.items()
        if not passed
    ]

    report = {
        "schema_version": "1.0",
        "stage": "S3.09",
        "scope": "supervised_5_10_20pct_2064_actual_update_scheduler_preflight",
        "official_training_authorized": False,
        "scientific_contract": {
            "budgets": ["5pct", "10pct", "20pct"],
            "optimizer_updates": EXPECTED_UPDATES,
            "milestones": EXPECTED_MILESTONES,
            "effective_labeled_batch": EXPECTED_BATCH_SIZE,
        },
        "architectures": architectures,
        "runtime_fixture": {
            "raw_iterations": runner.iter,
            "actual_optimizer_updates": actual_updates,
            "scheduler_last_step": runner.param_schedulers[0].last_step,
            "final_max_iters": runner.max_iters,
            "final_lr": (
                runner.optim_wrapper.optimizer.param_groups[0]["lr"]
            ),
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
        print("S3_09_2064_SCHEDULE=FAIL")
        return 1

    print("S3_09_2064_SCHEDULE=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())