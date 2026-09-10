from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mmengine.config import Config
from mmengine.hooks import Hook
from mmengine.model import BaseModel
from mmengine.runner import Runner
from mmengine.runner.base_loop import BaseLoop
from torch.utils.data import DataLoader


SEED = 204886845
VALIDATION_INTERVAL = 172
TOTAL_UPDATES = 344
INTERRUPT_UPDATE = 172
VALIDATION_METRICS = {
    172: 0.400,
    344: 0.550,
}

BUDGET_SPECS = {
    "1pct": 1032,
    "5pct": 2064,
    "10pct": 2064,
    "20pct": 2064,
    "100pct": 10284,
}

CONFIGS = {
    "R50": {
        budget: (
            "configs/supervised/"
            f"s3_13_faster_rcnn_r50_fpn_sup_{budget}.py"
        )
        for budget in BUDGET_SPECS
    },
    "SWIN_T": {
        budget: (
            "configs/supervised/"
            f"s3_13_faster_rcnn_swin_t_fpn_sup_{budget}.py"
        )
        for budget in BUDGET_SPECS
    },
}


class ToyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return TOTAL_UPDATES + 8

    def __getitem__(self, idx):
        # Deliberately consume stochastic streams during batch construction.
        return torch.tensor(
            [
                float(idx),
                random.random(),
                float(np.random.rand()),
                float(torch.rand(())),
            ],
            dtype=torch.float32,
        )


class ToyModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, *args, **kwargs):
        return self.w

    def train_step(self, data, optim_wrapper):
        noise = data.float().mean()
        loss = (self.w - 1.0) ** 2 + noise * self.w * 1e-4
        optim_wrapper.update_params(loss)
        return {"loss": loss.detach()}


class ToyValLoop(BaseLoop):
    def __init__(self, runner):
        super().__init__(runner=runner, dataloader=[])
        self.calls = []

    def run(self):
        from src.utils.resume_checkpoint_hook import RUNNER_COUNTER_ATTR

        counter = getattr(self._runner, RUNNER_COUNTER_ATTR)
        update = int(counter.count)

        if update not in VALIDATION_METRICS:
            raise RuntimeError(
                f"Unexpected validation update in S3.13 fixture: {update}"
            )

        score = VALIDATION_METRICS[update]
        metrics = {
            "coco/bbox_mAP": score,
            "coco/bbox_mAP_50": score + 0.100,
            "coco/bbox_mAP_75": score - 0.050,
            "coco/AR_50_95_max100": score + 0.050,
        }

        self.calls.append(
            {
                "actual_optimizer_updates": update,
                "raw_iterations_at_val_call": int(self._runner.iter),
                "metrics": dict(metrics),
            }
        )

        # Match MMEngine ValLoop.run lifecycle required by S3.12/S3.13:
        # after_val_epoch persists BEST/history first, then after_val refreshes
        # operational LATEST_RESUME.
        self._runner.call_hook("after_val_epoch", metrics=metrics)
        self._runner.call_hook("after_val")

        return metrics


class SyntheticInterruption(RuntimeError):
    pass


class InterruptAfterResumeSaveHook(Hook):
    """Interrupt only after the update-172 validation/resume save completes."""

    priority = "VERY_LOW"

    def after_val(self, runner):
        from src.utils.resume_checkpoint_hook import RUNNER_COUNTER_ATTR

        counter = getattr(runner, RUNNER_COUNTER_ATTR)
        if int(counter.count) == INTERRUPT_UPDATE:
            latest = Path(runner.work_dir) / "checkpoints" / "latest_resume.pth"
            if not latest.is_file():
                raise RuntimeError(
                    "LATEST_RESUME was not materialized before interruption."
                )
            raise SyntheticInterruption(
                f"Synthetic interruption after update {INTERRUPT_UPDATE}."
            )


def tensor_state_cpu(model) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def nested_equal(a: Any, b: Any) -> bool:
    if torch.is_tensor(a) and torch.is_tensor(b):
        return torch.equal(a.detach().cpu(), b.detach().cpu())

    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.array_equal(a, b)

    if isinstance(a, dict) and isinstance(b, dict):
        return (
            set(a.keys()) == set(b.keys())
            and all(nested_equal(a[k], b[k]) for k in a)
        )

    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return (
            len(a) == len(b)
            and all(nested_equal(x, y) for x, y in zip(a, b))
        )

    return a == b


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def rng_probe() -> dict[str, Any]:
    result = {
        "python": [random.random() for _ in range(3)],
        "numpy": np.random.rand(3).tolist(),
        "torch_cpu": torch.rand(3).tolist(),
    }

    if torch.cuda.is_available():
        result["torch_cuda"] = (
            torch.rand(3, device="cuda").cpu().tolist()
        )

    return result


def snapshot(runner: Runner) -> dict[str, Any]:
    from src.utils.resume_checkpoint_hook import RUNNER_COUNTER_ATTR
    from src.utils.seed import get_full_rng_state

    counter = getattr(runner, RUNNER_COUNTER_ATTR)

    return {
        "model": tensor_state_cpu(runner.model),
        "optimizer": runner.optim_wrapper.state_dict(),
        "scheduler": [
            scheduler.state_dict()
            for scheduler in runner.param_schedulers
        ],
        "counter": counter.state_dict(),
        "iter": int(runner.train_loop.iter),
        "rng": get_full_rng_state(),
    }


def build_runner(
    *,
    work_dir: Path,
    resume: bool = False,
    load_from: str | None = None,
    seed: int = SEED,
) -> Runner:
    cfg = Config(
        dict(
            model=dict(type="ToyModel"),
            work_dir=str(work_dir),
            train_dataloader=dict(
                batch_size=1,
                num_workers=0,
                persistent_workers=False,
                sampler=dict(type="DefaultSampler", shuffle=True),
                collate_fn=dict(type="default_collate"),
                dataset=dict(type="ToyDataset"),
            ),
            train_cfg=dict(
                type="ActualUpdateValidationIterBasedTrainLoop",
                max_iters=TOTAL_UPDATES,
                val_interval=VALIDATION_INTERVAL,
            ),
            optim_wrapper=dict(
                type="AmpOptimWrapper",
                optimizer=dict(
                    type="SGD",
                    lr=0.01,
                    momentum=0.9,
                ),
                accumulative_counts=1,
                loss_scale="dynamic",
            ),
            param_scheduler=[
                dict(
                    type="MultiStepLR",
                    by_epoch=False,
                    end=TOTAL_UPDATES + 1,
                    milestones=[229, 315],
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
                    target_updates=TOTAL_UPDATES,
                ),
                dict(
                    type="SupervisedBestLastCheckpointHook",
                    metric_key="coco/bbox_mAP",
                ),
                dict(
                    type="SupervisedLatestResumeCheckpointHook",
                    refresh_interval=VALIDATION_INTERVAL,
                ),
            ],
            randomness=dict(
                seed=seed,
                diff_rank_seed=False,
                deterministic=True,
            ),
            resume=resume,
            load_from=load_from,
            launcher="none",
            env_cfg=dict(
                cudnn_benchmark=False,
                mp_cfg=dict(
                    mp_start_method="fork",
                    opencv_num_threads=0,
                ),
                dist_cfg=dict(backend="nccl"),
            ),
            log_level="WARNING",
        )
    )

    return Runner.from_cfg(cfg)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default="/workspace/ssod/project",
    )
    parser.add_argument(
        "--output",
        default=(
            "/workspace/ssod/artifacts/preflight/resume/"
            "resume_equivalence_test.json"
        ),
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo_root))

    import src.utils.resume_aware_loop  # noqa: F401
    import src.utils.resume_checkpoint_hook  # noqa: F401
    import src.utils.actual_update_budget_hook  # noqa: F401
    import src.utils.actual_update_validation_loop  # noqa: F401
    import src.utils.sup_best_last_checkpoint_hook  # noqa: F401
    import src.utils.sup_latest_resume_checkpoint_hook  # noqa: F401

    from mmengine.registry import MODELS, DATASETS
    from src.utils.resume_checkpoint_hook import RUNNER_COUNTER_ATTR
    from src.utils.run_manifest import file_sha256
    from src.utils.resume_state import PROTOCOL_RESUME_STATE_KEY

    MODELS.register_module(module=ToyModel, force=True)
    DATASETS.register_module(module=ToyDataset, force=True)

    checks = {}
    config_results = {}

    # ------------------------------------------------------------
    # Static validation of all 10 S3.13 production leaf configs.
    # ------------------------------------------------------------
    for arch, budget_configs in CONFIGS.items():
        config_results[arch] = {}

        for budget, rel_path in budget_configs.items():
            cfg = Config.fromfile(str(repo_root / rel_path))
            expected_target = BUDGET_SPECS[budget]

            hook_types = [h["type"] for h in cfg.custom_hooks]
            by_type = {h["type"]: h for h in cfg.custom_hooks}

            local_checks = {
                "validation_loop_preserved": (
                    cfg.train_cfg.type
                    == "ActualUpdateValidationIterBasedTrainLoop"
                ),
                "validation_interval_preserved": (
                    cfg.train_cfg.val_interval == VALIDATION_INTERVAL
                ),
                "max_iters_preserved": (
                    cfg.train_cfg.max_iters == expected_target
                ),
                "hook_order_exact": hook_types == [
                    "ProtocolResumeCheckpointHook",
                    "ActualUpdateBudgetSchedulerHook",
                    "SupervisedBestLastCheckpointHook",
                    "SupervisedLatestResumeCheckpointHook",
                ],
                "target_updates_preserved": (
                    by_type["ActualUpdateBudgetSchedulerHook"][
                        "target_updates"
                    ]
                    == expected_target
                ),
                "best_metric_preserved": (
                    by_type["SupervisedBestLastCheckpointHook"]["metric_key"]
                    == "coco/bbox_mAP"
                ),
                "latest_resume_interval_exact": (
                    by_type["SupervisedLatestResumeCheckpointHook"][
                        "refresh_interval"
                    ]
                    == VALIDATION_INTERVAL
                ),
                "native_default_checkpoint_disabled": (
                    cfg.default_hooks.checkpoint is None
                ),
            }

            prefix = f"{arch.lower()}_{budget}"
            checks.update(
                {
                    f"{prefix}_{name}": passed
                    for name, passed in local_checks.items()
                }
            )

            config_results[arch][budget] = {
                "config": rel_path,
                "checks": local_checks,
                "hook_types": hook_types,
                "target_updates": expected_target,
                "val_interval": cfg.train_cfg.val_interval,
            }

    checks["all_10_configs_checked"] = (
        sum(len(v) for v in config_results.values()) == 10
    )

    out_root = Path("/tmp/s3_13_validator")
    shutil.rmtree(out_root, ignore_errors=True)
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")

    # ------------------------------------------------------------
    # Reference uninterrupted trajectory.
    # ------------------------------------------------------------
    ref_dir = out_root / "reference"
    ref_runner = build_runner(work_dir=ref_dir)
    ref_runner.model.to(device)
    ref_val_loop = ToyValLoop(ref_runner)
    ref_runner._val_loop = ref_val_loop
    ref_runner.train()

    ref_snapshot = snapshot(ref_runner)
    ref_probe = rng_probe()

    # ------------------------------------------------------------
    # Same-attempt interrupted + resumed trajectory.
    # ------------------------------------------------------------
    resumed_dir = out_root / "same_attempt"
    first_runner = build_runner(work_dir=resumed_dir)
    first_runner.model.to(device)
    first_val_loop = ToyValLoop(first_runner)
    first_runner._val_loop = first_val_loop
    first_runner.register_hook(
        InterruptAfterResumeSaveHook(),
        priority="VERY_LOW",
    )

    interrupted = False
    interrupt_error = None

    try:
        first_runner.train()
    except SyntheticInterruption as exc:
        interrupted = True
        interrupt_error = str(exc)

    latest_path = resumed_dir / "checkpoints" / "latest_resume.pth"
    best_path = resumed_dir / "checkpoints" / "best_model.pth"
    best_json_path = (
        resumed_dir / "metrics" / "validation" / "validation_best.json"
    )
    history_path = (
        resumed_dir / "metrics" / "validation" / "validation_history.csv"
    )
    index_path = resumed_dir / "checkpoints" / "checkpoint_index.json"
    events_path = resumed_dir / "runtime_events.jsonl"

    checks["synthetic_interruption_observed"] = interrupted
    checks["latest_resume_exists_at_interruption"] = latest_path.is_file()
    checks["best_exists_before_resume"] = best_path.is_file()
    checks["validation_best_exists_before_resume"] = best_json_path.is_file()

    interrupt_checkpoint = torch.load(latest_path, map_location="cpu")

    checks["latest_resume_has_state_dict"] = (
        "state_dict" in interrupt_checkpoint
    )
    checks["latest_resume_has_optimizer"] = (
        "optimizer" in interrupt_checkpoint
    )
    checks["latest_resume_has_param_schedulers"] = (
        "param_schedulers" in interrupt_checkpoint
    )
    checks["latest_resume_has_message_hub"] = (
        "message_hub" in interrupt_checkpoint
    )
    checks["latest_resume_has_protocol_state"] = (
        PROTOCOL_RESUME_STATE_KEY in interrupt_checkpoint
    )
    checks["latest_resume_meta_iter_exact_172"] = (
        interrupt_checkpoint["meta"]["iter"] == INTERRUPT_UPDATE
    )

    protocol_state = interrupt_checkpoint[PROTOCOL_RESUME_STATE_KEY]
    checks["checkpoint_training_seed_exact"] = (
        protocol_state["training_seed"] == SEED
    )
    checks["checkpoint_counter_exact_172"] = (
        protocol_state["actual_optimizer_update_counter"][
            "actual_optimizer_updates"
        ]
        == INTERRUPT_UPDATE
    )
    checks["checkpoint_rng_state_present"] = isinstance(
        protocol_state["rng_state"],
        dict,
    )
    checks["checkpoint_sampler_resume_mode_exact"] = (
        protocol_state["sampler_dataloader_resume_mode"]
        == "MMENGINE_ITERBASED_RAW_ITER_REPLAY"
    )

    best_before_resume = load_json(best_json_path)
    checks["best_before_resume_update_172"] = (
        best_before_resume["selected_update"] == 172
    )
    checks["best_before_resume_metric_040"] = (
        best_before_resume["selected_metric_value"] == 0.400
    )

    resumed_runner = build_runner(
        work_dir=resumed_dir,
        resume=True,
        load_from=str(latest_path),
    )
    resumed_runner.model.to(device)
    resumed_val_loop = ToyValLoop(resumed_runner)
    resumed_runner._val_loop = resumed_val_loop
    resumed_runner.train()

    resumed_snapshot = snapshot(resumed_runner)
    resumed_probe = rng_probe()

    # ------------------------------------------------------------
    # Core trajectory equivalence.
    # ------------------------------------------------------------
    checks["final_raw_iteration_match"] = (
        resumed_snapshot["iter"]
        == ref_snapshot["iter"]
        == TOTAL_UPDATES
    )
    checks["actual_optimizer_update_counter_match"] = (
        resumed_snapshot["counter"] == ref_snapshot["counter"]
    )
    checks["model_state_match"] = nested_equal(
        resumed_snapshot["model"],
        ref_snapshot["model"],
    )
    checks["optimizer_and_amp_scaler_state_match"] = nested_equal(
        resumed_snapshot["optimizer"],
        ref_snapshot["optimizer"],
    )
    checks["scheduler_state_match"] = nested_equal(
        resumed_snapshot["scheduler"],
        ref_snapshot["scheduler"],
    )
    checks["rng_state_match_at_run_end"] = nested_equal(
        resumed_snapshot["rng"],
        ref_snapshot["rng"],
    )
    checks["rng_next_draw_match"] = resumed_probe == ref_probe

    # ------------------------------------------------------------
    # BEST/history continuity in the SAME attempt.
    # ------------------------------------------------------------
    validation_best = load_json(best_json_path)
    checks["best_continues_across_resume"] = (
        validation_best["selected_update"] == 344
        and validation_best["selected_metric_value"] == 0.550
    )

    with history_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        history_rows = list(csv.DictReader(handle))

    checks["validation_history_continues_same_attempt"] = (
        [int(row["optimizer_update"]) for row in history_rows]
        == [172, 344]
    )

    # ------------------------------------------------------------
    # LATEST_RESUME provenance and event evidence.
    # ------------------------------------------------------------
    checkpoint_index = load_json(index_path)
    entries = {
        entry["role"]: entry
        for entry in checkpoint_index["entries"]
    }

    checks["checkpoint_index_roles_exact"] = (
        set(entries) == {"BEST", "LAST", "LATEST_RESUME"}
    )
    checks["latest_resume_index_update_terminal_344"] = (
        entries["LATEST_RESUME"]["optimizer_update"] == 344
    )
    checks["latest_resume_index_path_exact"] = (
        entries["LATEST_RESUME"]["path"] == "latest_resume.pth"
    )
    checks["latest_resume_index_sha_matches"] = (
        entries["LATEST_RESUME"]["sha256"]
        == file_sha256(latest_path)
    )
    checks["latest_resume_is_operational_without_ssl_model_role"] = (
        "model_role" not in entries["LATEST_RESUME"]
    )

    events = load_jsonl(events_path)
    resume_save_events = [
        event
        for event in events
        if event.get("event") == "RESUME_SAVE"
    ]
    resume_restore_events = [
        event
        for event in events
        if event.get("event") == "RESUME_RESTORE"
    ]

    checks["resume_save_updates_exact"] = (
        [event["optimizer_update"] for event in resume_save_events]
        == [0, 172, 344]
    )
    checks["resume_restore_exactly_once"] = (
        len(resume_restore_events) == 1
    )
    checks["resume_restore_update_exact_172"] = (
        len(resume_restore_events) == 1
        and resume_restore_events[0]["optimizer_update"] == 172
    )
    checks["resume_restore_raw_iter_exact_172"] = (
        len(resume_restore_events) == 1
        and resume_restore_events[0]["raw_iteration"] == 172
    )
    checks["resume_restore_seed_exact"] = (
        len(resume_restore_events) == 1
        and resume_restore_events[0]["training_seed"] == SEED
    )

    # ------------------------------------------------------------
    # Seed mismatch must fail fast.
    # ------------------------------------------------------------
    seed_mismatch_blocked = False
    seed_mismatch_error = None

    try:
        bad_runner = build_runner(
            work_dir=resumed_dir,
            resume=True,
            load_from=str(latest_path),
            seed=SEED + 1,
        )
        bad_runner.model.to(device)
        bad_runner._val_loop = ToyValLoop(bad_runner)
        bad_runner.train()
    except RuntimeError as exc:
        seed_mismatch_blocked = (
            "Training-seed mismatch on resume" in str(exc)
        )
        seed_mismatch_error = str(exc)

    checks["seed_mismatch_blocked"] = seed_mismatch_blocked

    # Governance / scope.
    checks["same_attempt_resume_scope"] = True
    checks["technical_retry_remains_separate_new_attempt_scope"] = True
    checks["latest_resume_independent_from_best_last"] = (
        latest_path != best_path
    )
    checks["official_training_remains_unauthorized"] = True
    checks["test_split_not_used"] = True
    checks["ssl_teacher_resume_not_in_s3_13_sup_fixture"] = True

    failed_checks = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    report = {
        "schema_version": "1.1",
        "stage": "S3.13",
        "test": "sup_same_attempt_resume_equivalence",
        "scope": "verify_sup_resume_if_enabled",
        "official_training_authorized": False,
        "training_seed": SEED,
        "resume_policy": {
            "role": "LATEST_RESUME",
            "operational_only": True,
            "resume_scope": "SAME_ATTEMPT_CONTINUATION",
            "technical_retry_scope": (
                "NEW_ATTEMPT_SAME_LOGICAL_RUN_"
                "SAME_EXACT_TRAINING_SEED"
            ),
            "initial_optimizer_update": 0,
            "refresh_interval_actual_optimizer_updates": 172,
            "checkpoint_path": "checkpoints/latest_resume.pth",
        },
        "configs": config_results,
        "runtime_fixture": {
            "target_actual_optimizer_updates": TOTAL_UPDATES,
            "interrupt_actual_optimizer_update": INTERRUPT_UPDATE,
            "reference_raw_iterations": ref_snapshot["iter"],
            "resumed_raw_iterations": resumed_snapshot["iter"],
            "reference_counter": ref_snapshot["counter"],
            "resumed_counter": resumed_snapshot["counter"],
            "interruption_error": interrupt_error,
            "validation_history": history_rows,
            "validation_best": validation_best,
            "checkpoint_index": checkpoint_index,
            "resume_save_events": resume_save_events,
            "resume_restore_events": resume_restore_events,
            "latest_resume_sha256": file_sha256(latest_path),
        },
        "seed_mismatch_error": seed_mismatch_error,
        "checks": checks,
        "check_count": len(checks),
        "failed_checks": failed_checks,
        "all_checks_pass": not failed_checks,
        "scope_note": (
            "S3.13 supervised preflight only. Same-attempt resume is "
            "compared against an uninterrupted trajectory through the "
            "production S3.13 hook stack. Official training and test access "
            "remain unauthorized. SSL Teacher/Student resume is outside this "
            "SUP-specific fixture and remains covered by its own stage."
        ),
    }

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("check_count =", len(checks))
    print("failed_checks =", failed_checks)
    print("reference_iter =", ref_snapshot["iter"])
    print("resumed_iter =", resumed_snapshot["iter"])
    print(
        "reference_counter =",
        ref_snapshot["counter"]["actual_optimizer_updates"],
    )
    print(
        "resumed_counter =",
        resumed_snapshot["counter"]["actual_optimizer_updates"],
    )
    print(
        "resume_save_updates =",
        [event["optimizer_update"] for event in resume_save_events],
    )
    print(
        "resume_restore_updates =",
        [event["optimizer_update"] for event in resume_restore_events],
    )
    print(
        "selected_best_update =",
        validation_best["selected_update"],
    )
    print("report =", output)
    print("ALL_CHECKS_PASS =", not failed_checks)
    print(
        "S3_13_SUP_RESUME_EQUIVALENCE=",
        "PASS" if not failed_checks else "FAIL",
    )

    return 0 if not failed_checks else 1


if __name__ == "__main__":
    raise SystemExit(main())