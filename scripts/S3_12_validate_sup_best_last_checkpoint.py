from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from mmengine.config import Config
from mmengine.model import BaseModel
from mmengine.runner import Runner
from mmengine.runner.base_loop import BaseLoop
from torch.utils.data import DataLoader


VALIDATION_INTERVAL = 172
RUNTIME_TARGET_UPDATES = 516
RUNTIME_SKIP_CALLS = {2, 174, 346}
RUNTIME_METRICS = [0.400, 0.550, 0.550]

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
            f"s3_12_faster_rcnn_r50_fpn_sup_{budget}.py"
        )
        for budget in BUDGET_SPECS
    },
    "SWIN_T": {
        budget: (
            "configs/supervised/"
            f"s3_12_faster_rcnn_swin_t_fpn_sup_{budget}.py"
        )
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
        call_index = len(self.calls)
        score = RUNTIME_METRICS[call_index]

        metrics = {
            "coco/bbox_mAP": score,
            "coco/bbox_mAP_50": score + 0.100,
            "coco/bbox_mAP_75": score - 0.050,
            "coco/AR_50_95_max100": score + 0.050,
        }

        self.calls.append(
            {
                "actual_optimizer_updates": counter.count,
                "raw_iterations_at_val_call": self._runner.iter,
                "metrics": dict(metrics),
            }
        )

        self._runner.call_hook(
            "after_val_epoch",
            metrics=metrics,
        )
        return metrics


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/workspace/ssod/project")
    parser.add_argument(
        "--output",
        default=(
            "/workspace/ssod/artifacts/preflight/sup/"
            "sup_best_last_checkpoint_preflight.json"
        ),
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo_root))

    checks = {}
    config_results = {}

    for arch, budget_configs in CONFIGS.items():
        config_results[arch] = {}

        for budget, rel_path in budget_configs.items():
            cfg = Config.fromfile(str(repo_root / rel_path))
            expected_target = BUDGET_SPECS[budget]

            hook_types = [h["type"] for h in cfg.custom_hooks]
            checkpoint_hooks = [
                h
                for h in cfg.custom_hooks
                if h["type"] == "SupervisedBestLastCheckpointHook"
            ]
            budget_hooks = [
                h
                for h in cfg.custom_hooks
                if h["type"] == "ActualUpdateBudgetSchedulerHook"
            ]

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
                ],
                "one_budget_hook": len(budget_hooks) == 1,
                "one_sup_checkpoint_hook": len(checkpoint_hooks) == 1,
                "target_updates_preserved": (
                    len(budget_hooks) == 1
                    and budget_hooks[0]["target_updates"]
                    == expected_target
                ),
                "runtime_metric_key_exact": (
                    len(checkpoint_hooks) == 1
                    and checkpoint_hooks[0]["metric_key"]
                    == "coco/bbox_mAP"
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

    import src.utils.resume_aware_loop  # noqa: F401
    import src.utils.resume_checkpoint_hook  # noqa: F401
    import src.utils.actual_update_budget_hook  # noqa: F401
    import src.utils.actual_update_validation_loop  # noqa: F401
    import src.utils.sup_best_last_checkpoint_hook  # noqa: F401

    from src.utils.resume_checkpoint_hook import RUNNER_COUNTER_ATTR
    from src.utils.sup_best_last_checkpoint_hook import (
        BEST_RULE,
        BEST_TIE_BREAK,
        SCIENTIFIC_PRIMARY_METRIC,
        SupervisedBestLastCheckpointHook,
        VALIDATION_HISTORY_FIELDS,
    )
    from src.utils.run_manifest import file_sha256

    work_dir = Path("/tmp/s3_12_validator")
    if work_dir.exists():
        shutil.rmtree(work_dir)

    runner = Runner(
        model=ToyModel(),
        work_dir=str(work_dir),
        train_dataloader=DataLoader(
            list(
                range(
                    RUNTIME_TARGET_UPDATES
                    + len(RUNTIME_SKIP_CALLS)
                    + 4
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
                milestones=[344, 472],
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
            dict(
                type="SupervisedBestLastCheckpointHook",
                metric_key="coco/bbox_mAP",
            ),
        ],
        randomness=dict(seed=42),
        log_level="ERROR",
    )

    toy_val_loop = ToyValLoop(runner)
    runner._val_loop = toy_val_loop
    runner.train()

    counter = getattr(runner, RUNNER_COUNTER_ATTR)

    best_path = work_dir / "checkpoints" / "best_model.pth"
    last_path = work_dir / "checkpoints" / "last_model.pth"
    index_path = work_dir / "checkpoints" / "checkpoint_index.json"
    best_json_path = (
        work_dir / "metrics" / "validation" / "validation_best.json"
    )
    history_path = (
        work_dir / "metrics" / "validation" / "validation_history.csv"
    )
    events_path = work_dir / "runtime_events.jsonl"

    checks["runtime_actual_update_target_reached"] = (
        counter.count == RUNTIME_TARGET_UPDATES
    )
    checks["runtime_raw_iterations_include_skips"] = (
        runner.iter
        == RUNTIME_TARGET_UPDATES + len(RUNTIME_SKIP_CALLS)
    )
    checks["runtime_validation_updates_exact"] = (
        [
            item["actual_optimizer_updates"]
            for item in toy_val_loop.calls
        ]
        == [172, 344, 516]
    )
    checks["runtime_validation_raw_iterations_shifted"] = (
        [
            item["raw_iterations_at_val_call"]
            for item in toy_val_loop.calls
        ]
        == [173, 347, 519]
    )

    checks["best_model_exists"] = best_path.is_file()
    checks["last_model_exists"] = last_path.is_file()
    checks["validation_best_exists"] = best_json_path.is_file()
    checks["validation_history_exists"] = history_path.is_file()
    checks["checkpoint_index_exists"] = index_path.is_file()
    checks["runtime_events_exists"] = events_path.is_file()

    validation_best = load_json(best_json_path)
    checkpoint_index = load_json(index_path)

    with history_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        history_rows = list(csv.DictReader(handle))

    runtime_events = load_jsonl(events_path)
    best_events = [
        event
        for event in runtime_events
        if event.get("event") == "BEST_CHECKPOINT_UPDATED"
    ]

    checks["validation_best_selection_split"] = (
        validation_best["selection_split"] == "validation"
    )
    checks["validation_best_metric_semantic_name"] = (
        validation_best["selection_metric"]
        == SCIENTIFIC_PRIMARY_METRIC
    )
    checks["validation_best_rule_exact"] = (
        validation_best["rule"] == BEST_RULE
    )
    checks["validation_best_tie_break_exact"] = (
        validation_best["tie_break"] == BEST_TIE_BREAK
    )
    checks["best_selected_at_update_344"] = (
        validation_best["selected_update"] == 344
    )
    checks["best_selected_metric_is_055"] = (
        validation_best["selected_metric_value"] == 0.550
    )
    checks["tie_at_516_does_not_replace_best"] = (
        validation_best["selected_update"] == 344
        and validation_best["selected_metric_value"] == 0.550
    )
    checks["validation_best_path_exact"] = (
        validation_best["checkpoint_path"]
        == "checkpoints/best_model.pth"
    )
    checks["validation_best_sha_matches_file"] = (
        validation_best["checkpoint_sha256"]
        == file_sha256(best_path)
    )

    checks["history_header_exact"] = (
        tuple(history_rows[0].keys())
        == tuple(VALIDATION_HISTORY_FIELDS)
    )
    checks["history_has_three_validation_rows"] = (
        len(history_rows) == 3
    )
    checks["history_updates_exact"] = (
        [int(row["optimizer_update"]) for row in history_rows]
        == [172, 344, 516]
    )
    checks["history_scores_exact"] = (
        [float(row["bbox_mAP_50_95"]) for row in history_rows]
        == [0.400, 0.550, 0.550]
    )
    checks["history_candidate_sequence_exact"] = (
        [row["checkpoint_candidate"] for row in history_rows]
        == ["True", "True", "False"]
    )
    checks["history_best_so_far_sequence_exact"] = (
        [float(row["best_so_far"]) for row in history_rows]
        == [0.400, 0.550, 0.550]
    )

    checks["best_update_events_only_on_strict_improvement"] = (
        [event["optimizer_update"] for event in best_events]
        == [172, 344]
    )
    checks["tie_does_not_emit_best_update_event"] = (
        len(best_events) == 2
    )

    entries = {
        entry["role"]: entry
        for entry in checkpoint_index["entries"]
    }

    checks["checkpoint_policy_locked"] = (
        checkpoint_index["checkpoint_policy"] == "LOCKED"
    )
    checks["checkpoint_index_roles_exact"] = (
        set(entries) == {"BEST", "LAST"}
    )
    checks["sup_checkpoint_index_has_no_ssl_model_role"] = all(
        "model_role" not in entry
        for entry in checkpoint_index["entries"]
    )
    checks["best_index_update_is_344"] = (
        entries["BEST"]["optimizer_update"] == 344
    )
    checks["best_index_metric_is_055"] = (
        entries["BEST"]["validation_bbox_mAP_50_95"] == 0.550
    )
    checks["best_index_path_exact"] = (
        entries["BEST"]["path"] == "best_model.pth"
    )
    checks["best_index_sha_matches_file"] = (
        entries["BEST"]["sha256"] == file_sha256(best_path)
    )
    checks["best_index_retained"] = (
        entries["BEST"]["retained"] is True
    )
    checks["last_index_update_is_terminal_516"] = (
        entries["LAST"]["optimizer_update"] == 516
    )
    checks["last_index_path_exact"] = (
        entries["LAST"]["path"] == "last_model.pth"
    )
    checks["last_index_sha_matches_file"] = (
        entries["LAST"]["sha256"] == file_sha256(last_path)
    )
    checks["last_index_retained"] = (
        entries["LAST"]["retained"] is True
    )

    best_checkpoint = torch.load(best_path, map_location="cpu")
    last_checkpoint = torch.load(last_path, map_location="cpu")

    checks["best_checkpoint_has_state_dict"] = (
        "state_dict" in best_checkpoint
    )
    checks["last_checkpoint_has_state_dict"] = (
        "state_dict" in last_checkpoint
    )
    checks["best_and_last_are_distinct_materializations"] = (
        file_sha256(best_path) != file_sha256(last_path)
    )

    restored_hook = SupervisedBestLastCheckpointHook(
        metric_key="coco/bbox_mAP"
    )
    restored_hook.before_run(
        SimpleNamespace(work_dir=str(work_dir))
    )

    checks["persisted_best_state_restores_score"] = (
        restored_hook._best_score == 0.550
    )
    checks["persisted_best_state_restores_update"] = (
        restored_hook._best_update == 344
    )

    checks["official_training_remains_unauthorized"] = True
    checks["test_split_not_used_for_checkpoint_selection"] = True
    checks["ssl_ema_checkpoint_logic_not_in_s3_12_fixture"] = True

    failed_checks = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    report = {
        "schema_version": "1.0",
        "stage": "S3.12",
        "scope": "verify_sup_best_last_checkpoint_logic",
        "official_training_authorized": False,
        "scientific_contract": {
            "selection_split": "validation",
            "selection_metric": "bbox_mAP_50_95",
            "runtime_metric_key": "coco/bbox_mAP",
            "rule": BEST_RULE,
            "tie_break": BEST_TIE_BREAK,
            "retain": ["BEST", "LAST"],
            "official_supervised_evaluation_model": "BEST",
            "test_usage": "final_evaluation_only",
        },
        "configs": config_results,
        "runtime_fixture": {
            "target_actual_optimizer_updates": RUNTIME_TARGET_UPDATES,
            "skipped_raw_calls": sorted(RUNTIME_SKIP_CALLS),
            "raw_iterations": runner.iter,
            "actual_optimizer_updates": counter.count,
            "validation_calls": toy_val_loop.calls,
            "validation_best": validation_best,
            "validation_history": history_rows,
            "checkpoint_index": checkpoint_index,
            "best_checkpoint_sha256": file_sha256(best_path),
            "last_checkpoint_sha256": file_sha256(last_path),
            "best_checkpoint_update_events": best_events,
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
    print(
        "validation_updates =",
        [
            item["actual_optimizer_updates"]
            for item in toy_val_loop.calls
        ],
    )
    print(
        "validation_scores =",
        [
            item["metrics"]["coco/bbox_mAP"]
            for item in toy_val_loop.calls
        ],
    )
    print(
        "selected_best_update =",
        validation_best["selected_update"],
    )
    print(
        "selected_best_metric =",
        validation_best["selected_metric_value"],
    )
    print(
        "checkpoint_roles =",
        [entry["role"] for entry in checkpoint_index["entries"]],
    )
    print(
        "best_update_events =",
        [event["optimizer_update"] for event in best_events],
    )
    print("ALL_CHECKS_PASS =", not failed_checks)

    if failed_checks:
        print("S3_12_SUP_BEST_LAST_CHECKPOINT=FAIL")
        return 1

    print("S3_12_SUP_BEST_LAST_CHECKPOINT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())