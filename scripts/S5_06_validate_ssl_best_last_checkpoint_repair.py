"""S5.06 controlled-repair validator for SSL scientific BEST/LAST EMA Teacher.

This validator is additive evidence for the controlled upstream repair discovered
while implementing S5.06. It does not alter the locked scientific protocol.
"""

from __future__ import annotations

import copy
import csv
import glob
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from mmengine.config import Config
from mmdet.utils import register_all_modules

register_all_modules(init_default_scope=True)

from mmdet.registry import DATASETS, HOOKS, METRICS  # noqa: E402

import src.hooks.actual_update_mean_teacher_hook  # noqa: F401,E402
import src.hooks.ssl_best_last_checkpoint_hook  # noqa: F401,E402
import src.hooks.ssl_latest_resume_checkpoint_hook  # noqa: F401,E402
import src.hooks.ssl_training_summary_hook  # noqa: F401,E402
import src.hooks.teacher_initialization_hook  # noqa: F401,E402
import src.metrics.protocol_coco_metric  # noqa: F401,E402
import src.utils.actual_update_budget_hook  # noqa: F401,E402
import src.utils.resume_checkpoint_hook  # noqa: F401,E402
from src.hooks.ssl_best_last_checkpoint_hook import SSLBestLastCheckpointHook  # noqa: E402
from src.utils.optimizer_update_counter import ActualOptimizerUpdateCounter  # noqa: E402
from src.utils.resume_checkpoint_hook import RUNNER_COUNTER_ATTR  # noqa: E402


EVIDENCE_PATH = Path(
    "artifacts/preflight/ssl/ssl_best_last_checkpoint_repair_preflight.json"
)

CANONICAL_SHA256 = {
    "sn-article.tex": "8d97840cc546a58aef7354af0c856119b1aea16f8be1cb7bbc1ab1a324d40fcc",
    "IMPLEMENTATION_HANDOFF.md": "540c94746571612ff819fb62a1e9fac6123c4035bc2642518eaa7740770820d0",
    "ARTIFACT_AND_EVIDENCE_CONTRACT.md": "885d4eeb66857e99744f6e8a27083e5bb695bbcc0d172cf4ebff722c53f655a4",
}

REPAIR_SHA256 = {
    "src/hooks/ssl_best_last_checkpoint_hook.py":
        "11e366cf0d26ab7ba81f2207807f425b5e1299b2ba87454ccb328115b3b6b09e",
    "configs/ssl/s5_06_ssl_scientific_validation_common.py":
        "14232e84c4e36b1627ee7bd74c8346075c63ef3d8a9a347ccb57adbf63469f10",
    "configs/ssl/s5_06_soft_teacher_r50_fpn_scientific_eval_1pct.py":
        "d76c4b66025369e6f2ee9b90ba0671a9c5ae31d62a1001033571497221f0539a",
    "configs/ssl/s5_06_soft_teacher_r50_fpn_scientific_eval_5pct.py":
        "5eb9450eb123da4c6bc7eabb4b9cc4d37d81a99d8f8089f2601e438eec3d21e1",
    "configs/ssl/s5_06_soft_teacher_r50_fpn_scientific_eval_10pct.py":
        "11cab476a73411512c86837382c881b37cdd9cbae1bfb3b15cf1e0ba4b02d870",
    "configs/ssl/s5_06_soft_teacher_r50_fpn_scientific_eval_20pct.py":
        "0303beb8c63d67480af81de06517eb25a95c8fdcf8073af0da47e3bfef99ec6d",
    "configs/ssl/s5_06_soft_teacher_swin_t_fpn_scientific_eval_1pct.py":
        "3fce5bf62d2864088abfb143acc1287205cdc259495df45d2ea5ab71f476ef56",
    "configs/ssl/s5_06_soft_teacher_swin_t_fpn_scientific_eval_5pct.py":
        "b67c7449aa93901944abb5d0a434b59728de776f9ff33176e29a9e865f54d527",
    "configs/ssl/s5_06_soft_teacher_swin_t_fpn_scientific_eval_10pct.py":
        "b2bdc1ded47b94803f73563299c8a0838fd1a6015bd4a6c250225206472ec9fb",
    "configs/ssl/s5_06_soft_teacher_swin_t_fpn_scientific_eval_20pct.py":
        "15acb3b0e8be95e8040f17e1a47ad7d95e77b1767b7b002779b36f6994a03269",
}

CELLS: List[Tuple[str, str, str, int]] = [
    ("R50", "1", "configs/ssl/s5_06_soft_teacher_r50_fpn_scientific_eval_1pct.py", 1032),
    ("R50", "5", "configs/ssl/s5_06_soft_teacher_r50_fpn_scientific_eval_5pct.py", 2064),
    ("R50", "10", "configs/ssl/s5_06_soft_teacher_r50_fpn_scientific_eval_10pct.py", 2064),
    ("R50", "20", "configs/ssl/s5_06_soft_teacher_r50_fpn_scientific_eval_20pct.py", 2064),
    ("SWIN", "1", "configs/ssl/s5_06_soft_teacher_swin_t_fpn_scientific_eval_1pct.py", 1032),
    ("SWIN", "5", "configs/ssl/s5_06_soft_teacher_swin_t_fpn_scientific_eval_5pct.py", 2064),
    ("SWIN", "10", "configs/ssl/s5_06_soft_teacher_swin_t_fpn_scientific_eval_10pct.py", 2064),
    ("SWIN", "20", "configs/ssl/s5_06_soft_teacher_swin_t_fpn_scientific_eval_20pct.py", 2064),
]

ORIGINAL_S4_CONFIGS = [
    "configs/ssl/s4_14_soft_teacher_r50_fpn_empty_pseudo_1pct.py",
    "configs/ssl/s4_14_soft_teacher_r50_fpn_empty_pseudo_5pct.py",
    "configs/ssl/s4_14_soft_teacher_r50_fpn_empty_pseudo_10pct.py",
    "configs/ssl/s4_14_soft_teacher_r50_fpn_empty_pseudo_20pct.py",
    "configs/ssl/s4_14_soft_teacher_swin_t_fpn_empty_pseudo_1pct.py",
    "configs/ssl/s4_14_soft_teacher_swin_t_fpn_empty_pseudo_5pct.py",
    "configs/ssl/s4_14_soft_teacher_swin_t_fpn_empty_pseudo_10pct.py",
    "configs/ssl/s4_14_soft_teacher_swin_t_fpn_empty_pseudo_20pct.py",
]

EXPECTED_HOOKS = [
    "TeacherInitializationHook",
    "ProtocolResumeCheckpointHook",
    "ActualUpdateMeanTeacherHook",
    "ActualUpdateBudgetSchedulerHook",
    "SSLBestLastCheckpointHook",
    "SSLLatestResumeCheckpointHook",
    "SSLTrainingSummaryHook",
]


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


checks: Dict[str, bool] = {}
observed: Dict[str, Any] = {}
matrix: Dict[str, Any] = {}


def check(name: str, value: bool) -> None:
    checks[name] = bool(value)


for path, expected in CANONICAL_SHA256.items():
    check(f"canonical_sha::{path}", Path(path).is_file() and sha256(path) == expected)

for path, expected in REPAIR_SHA256.items():
    check(f"repair_sha::{path}", Path(path).is_file() and sha256(path) == expected)

# Historical gap diagnosis: the S4 final configs had an actual-update validation
# loop type but no attached validation dataloader/evaluator and no scientific
# SSL BEST/LAST hook. This is the implementation gap repaired here.
original_gap_rows = []
for path in ORIGINAL_S4_CONFIGS:
    cfg = Config.fromfile(path)
    hook_types = [h.type for h in cfg.custom_hooks]
    row = {
        "path": path,
        "val_dataloader_present": cfg.get("val_dataloader", None) is not None,
        "val_evaluator_present": cfg.get("val_evaluator", None) is not None,
        "ssl_best_last_hook_present": "SSLBestLastCheckpointHook" in hook_types,
    }
    original_gap_rows.append(row)

check(
    "historical_s4_final_configs_missing_validation_path",
    all(
        not row["val_dataloader_present"] and not row["val_evaluator_present"]
        for row in original_gap_rows
    ),
)
check(
    "historical_s4_final_configs_missing_ssl_scientific_best_last_hook",
    all(not row["ssl_best_last_hook_present"] for row in original_gap_rows),
)

closure_matches = glob.glob(
    "artifacts/**/S4_CLOSURE_RECORD.json",
    recursive=True,
)
observed["s4_closure_record_matches"] = closure_matches
if len(closure_matches) == 1:
    closure_payload = json.loads(Path(closure_matches[0]).read_text(encoding="utf-8"))
    closure_text = json.dumps(closure_payload, sort_keys=True)
    check(
        "historical_s4_closure_claimed_best_last_pass",
        "17_BEST_LAST_CHECKPOINT_LOGIC" in closure_text and "PASS" in closure_text,
    )
else:
    check("historical_s4_closure_claimed_best_last_pass", False)

for arch, budget, path, updates in CELLS:
    cfg = Config.fromfile(path)
    hooks = [dict(x) for x in cfg.custom_hooks]
    hook_types = [h["type"] for h in hooks]
    by_type = {h["type"]: h for h in hooks}

    cell_checks: Dict[str, bool] = {
        "train_loop_type":
            cfg.train_cfg.type == "ActualUpdateValidationIterBasedTrainLoop",
        "max_iters_exact":
            int(cfg.train_cfg.max_iters) == updates,
        "val_interval_exact":
            int(cfg.train_cfg.val_interval) == 172,
        "budget_target_exact":
            int(by_type["ActualUpdateBudgetSchedulerHook"]["target_updates"]) == updates,
        "summary_updates_exact":
            int(by_type["SSLTrainingSummaryHook"]["expected_optimizer_updates"]) == updates,
        "effective_labeled_batch_exact":
            int(by_type["SSLTrainingSummaryHook"]["effective_labeled_batch"]) == 4,
        "effective_unlabeled_batch_exact":
            int(by_type["SSLTrainingSummaryHook"]["effective_unlabeled_batch"]) == 4,
        "ema_momentum_exact":
            abs(float(by_type["ActualUpdateMeanTeacherHook"]["momentum"]) - 0.001)
            < 1e-12,
        "ema_skip_buffers_true":
            bool(by_type["ActualUpdateMeanTeacherHook"]["skip_buffers"]) is True,
        "latest_refresh_exact":
            int(by_type["SSLLatestResumeCheckpointHook"]["refresh_interval"]) == 172,
        "hook_order_exact":
            hook_types == EXPECTED_HOOKS,
        "best_last_once":
            hook_types.count("SSLBestLastCheckpointHook") == 1,
        "latest_resume_once":
            hook_types.count("SSLLatestResumeCheckpointHook") == 1,
        "val_cfg_valloop":
            cfg.val_cfg.type == "ValLoop",
        "val_ann_fixed":
            str(cfg.val_dataloader.dataset.ann_file)
            == "/workspace/ssod/data/coco/instances_val.json",
        "val_test_mode_true":
            bool(cfg.val_dataloader.dataset.test_mode) is True,
        "val_shuffle_false":
            bool(cfg.val_dataloader.sampler.shuffle) is False,
        "val_batch_one":
            int(cfg.val_dataloader.batch_size) == 1,
        "metric_protocol_coco":
            cfg.val_evaluator.type == "ProtocolCocoMetric",
        "metric_ann_fixed":
            str(cfg.val_evaluator.ann_file)
            == "/workspace/ssod/data/coco/instances_val.json",
        "metric_bbox":
            cfg.val_evaluator.metric == "bbox",
        "metric_proposal_nums":
            tuple(cfg.val_evaluator.proposal_nums) == (1, 10, 100),
        "metric_prefix_coco":
            cfg.val_evaluator.prefix == "coco",
        "predict_on_teacher":
            cfg.model.semi_test_cfg.predict_on == "teacher",
        "forward_on_teacher":
            cfg.model.semi_test_cfg.forward_on == "teacher",
        "extract_feat_on_teacher":
            cfg.model.semi_test_cfg.extract_feat_on == "teacher",
        "test_dataloader_absent":
            cfg.get("test_dataloader", None) is None,
        "test_evaluator_absent":
            cfg.get("test_evaluator", None) is None,
    }

    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    cell_checks["dataset_len_734"] = len(dataset) == 734
    cell_checks["dataset_runtime_test_mode"] = bool(dataset.test_mode) is True

    metric = METRICS.build(cfg.val_evaluator)
    cell_checks["metric_runtime_class"] = (
        type(metric).__name__ == "ProtocolCocoMetric"
    )
    cell_checks["metric_runtime_maxdets"] = (
        tuple(metric.proposal_nums) == (1, 10, 100)
    )
    cell_checks["metric_runtime_iou_grid"] = (
        [round(float(x), 2) for x in metric.iou_thrs]
        == [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    )

    built = []
    priorities = []
    for raw in cfg.custom_hooks:
        hook_cfg = copy.deepcopy(dict(raw))
        priority = hook_cfg.pop("priority", "NORMAL")
        obj = HOOKS.build(hook_cfg)
        built.append(type(obj).__name__)
        priorities.append(priority)
    cell_checks["all_hooks_build"] = built == EXPECTED_HOOKS
    cell_checks["priority_sequence_exact"] = priorities == [
        "VERY_HIGH", "NORMAL", "HIGH", "NORMAL", "NORMAL", "NORMAL", "LOW"
    ]

    for key, value in cell_checks.items():
        check(f"matrix::{arch}::{budget}::{key}", value)

    matrix[f"{arch}_{budget}pct"] = {
        "config_path": path,
        "expected_optimizer_updates": updates,
        "check_count": len(cell_checks),
        "failed_checks": [k for k, v in cell_checks.items() if not v],
        "status": "PASS" if all(cell_checks.values()) else "FAIL",
    }


class FixedCounter(ActualOptimizerUpdateCounter):
    def __init__(self, value: int) -> None:
        self.value = value

    @property
    def count(self) -> int:
        return self.value


class ToySSL(nn.Module):
    def __init__(self, predict_on: str = "teacher") -> None:
        super().__init__()
        self.student = nn.Linear(2, 2)
        self.teacher = nn.Linear(2, 2)
        self.semi_test_cfg = {"predict_on": predict_on}


def fill(module: nn.Module, value: float) -> None:
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.fill_(value)


def state_equal(saved: Dict[str, torch.Tensor], module: nn.Module) -> bool:
    ref = module.state_dict()
    return set(saved) == set(ref) and all(
        torch.equal(saved[key], ref[key].detach().cpu()) for key in ref
    )


with tempfile.TemporaryDirectory(prefix="s5_06_ssl_repair_") as td:
    root = Path(td)
    checkpoint_dir = root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "checkpoint_index.json").write_text(
        json.dumps(
            {
                "checkpoint_policy": "LOCKED",
                "entries": [
                    {
                        "role": "LATEST_RESUME",
                        "optimizer_update": 0,
                        "path": "latest_resume.pth",
                        "sha256": "synthetic-latest",
                        "retained": True,
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    model = ToySSL("teacher")
    fill(model.student, -3.0)
    fill(model.teacher, 2.0)
    counter = FixedCounter(172)
    runner = SimpleNamespace(model=model, work_dir=str(root))
    setattr(runner, RUNNER_COUNTER_ATTR, counter)

    hook = SSLBestLastCheckpointHook(metric_key="coco/bbox_mAP")
    hook.before_run(runner)
    hook.after_val_epoch(
        runner,
        {
            "coco/bbox_mAP": 0.40,
            "coco/bbox_mAP_50": 0.60,
            "coco/bbox_mAP_75": 0.30,
            "coco/AR_50_95_max100": 0.50,
        },
    )

    best = checkpoint_dir / "best_ema_teacher.pth"
    best_json = root / "metrics" / "validation" / "validation_best_teacher.json"
    best_payload = torch.load(best, map_location="cpu")
    first_sha = sha256(best)

    check("synthetic::best_created", best.is_file())
    check("synthetic::best_json_created", best_json.is_file())
    check(
        "synthetic::best_teacher_state_exact",
        state_equal(best_payload["state_dict"], model.teacher),
    )
    check(
        "synthetic::best_not_student_state",
        not state_equal(best_payload["state_dict"], model.student),
    )
    check(
        "synthetic::best_meta_role",
        best_payload["meta"].get("scientific_checkpoint_role") == "BEST",
    )
    check(
        "synthetic::best_model_role",
        best_payload["meta"].get("model_role") == "EMA_TEACHER",
    )
    check(
        "synthetic::best_scope",
        best_payload["meta"].get("serialization_scope") == "EMA_TEACHER_ONLY",
    )
    check(
        "synthetic::best_has_no_optimizer",
        "optimizer" not in best_payload and "optimizer_wrapper" not in best_payload,
    )

    counter.value = 344
    fill(model.teacher, 4.0)
    hook.after_val_epoch(
        runner,
        {
            "coco/bbox_mAP": 0.40,
            "coco/bbox_mAP_50": 0.61,
            "coco/bbox_mAP_75": 0.31,
            "coco/AR_50_95_max100": 0.51,
        },
    )
    check("synthetic::tie_keeps_earliest_best", sha256(best) == first_sha)

    counter.value = 516
    hook.after_val_epoch(
        runner,
        {
            "coco/bbox_mAP": 0.50,
            "coco/bbox_mAP_50": 0.70,
            "coco/bbox_mAP_75": 0.40,
            "coco/AR_50_95_max100": 0.60,
        },
    )
    second_sha = sha256(best)
    check("synthetic::higher_metric_replaces_best", second_sha != first_sha)
    second_payload = torch.load(best, map_location="cpu")
    check(
        "synthetic::replacement_is_current_teacher",
        state_equal(second_payload["state_dict"], model.teacher),
    )

    validation_best = json.loads(best_json.read_text(encoding="utf-8"))
    check(
        "synthetic::best_selected_update_exact",
        validation_best.get("selected_update") == 516,
    )
    check(
        "synthetic::best_selected_metric_exact",
        abs(float(validation_best.get("selected_metric_value")) - 0.50) < 1e-12,
    )
    check(
        "synthetic::best_selection_model_exact",
        validation_best.get("selection_model") == "EMA_TEACHER",
    )
    check(
        "synthetic::best_rule_exact",
        validation_best.get("rule")
        == "ARGMAX_VALIDATION_TEACHER_BBOX_MAP_50_95",
    )

    restored = SSLBestLastCheckpointHook(metric_key="coco/bbox_mAP")
    restored.before_run(runner)
    check(
        "synthetic::best_provenance_restore_score",
        abs(float(restored._best_score) - 0.50) < 1e-12,
    )
    check(
        "synthetic::best_provenance_restore_update",
        restored._best_update == 516,
    )

    counter.value = 688
    fill(model.teacher, 6.0)
    hook.after_train(runner)
    last = checkpoint_dir / "last_ema_teacher.pth"
    last_payload = torch.load(last, map_location="cpu")
    check("synthetic::last_created", last.is_file())
    check(
        "synthetic::last_terminal_teacher_exact",
        state_equal(last_payload["state_dict"], model.teacher),
    )
    check(
        "synthetic::last_not_student_state",
        not state_equal(last_payload["state_dict"], model.student),
    )
    check(
        "synthetic::last_meta_role",
        last_payload["meta"].get("scientific_checkpoint_role") == "LAST",
    )
    check(
        "synthetic::last_update_exact",
        last_payload["meta"].get("optimizer_update") == 688,
    )

    index = json.loads(
        (checkpoint_dir / "checkpoint_index.json").read_text(encoding="utf-8")
    )
    roles = [entry.get("role") for entry in index["entries"]]
    by_role = {entry["role"]: entry for entry in index["entries"]}
    check(
        "synthetic::checkpoint_index_roles_exact",
        roles == ["BEST", "LAST", "LATEST_RESUME"],
    )
    check(
        "synthetic::latest_resume_preserved",
        by_role["LATEST_RESUME"].get("sha256") == "synthetic-latest",
    )
    check(
        "synthetic::best_index_model_role",
        by_role["BEST"].get("model_role") == "EMA_TEACHER",
    )
    check(
        "synthetic::last_index_model_role",
        by_role["LAST"].get("model_role") == "EMA_TEACHER",
    )
    check(
        "synthetic::best_index_sha_exact",
        by_role["BEST"].get("sha256") == sha256(best),
    )
    check(
        "synthetic::last_index_sha_exact",
        by_role["LAST"].get("sha256") == sha256(last),
    )

    history = root / "metrics" / "validation" / "validation_history.csv"
    rows = list(csv.DictReader(history.open("r", encoding="utf-8", newline="")))
    check("synthetic::validation_history_three_rows", len(rows) == 3)
    check(
        "synthetic::validation_history_model_role",
        all(row["model_role"] == "EMA_TEACHER" for row in rows),
    )
    check(
        "synthetic::validation_history_tie_not_candidate",
        rows[1]["checkpoint_candidate"] == "False",
    )

    bad_model = ToySSL("student")
    bad_runner = SimpleNamespace(model=bad_model, work_dir=str(root / "bad"))
    setattr(bad_runner, RUNNER_COUNTER_ATTR, FixedCounter(172))
    failed_closed = False
    try:
        SSLBestLastCheckpointHook().after_val_epoch(
            bad_runner,
            {"coco/bbox_mAP": 0.1},
        )
    except RuntimeError as exc:
        failed_closed = "must evaluate EMA Teacher" in str(exc)
    check("synthetic::predict_on_student_fails_closed", failed_closed)

failed = [name for name, value in checks.items() if not value]
status = "PASS" if not failed else "FAIL"

evidence = {
    "schema_version": "1.0.0",
    "stage": "S5.06",
    "evidence_type": "SSL_BEST_LAST_CHECKPOINT_CONTROLLED_REPAIR_PREFLIGHT",
    "status": status,
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "scientific_protocol_changed": False,
    "controlled_repair_required": True,
    "historical_s4_gate17_overclaim_detected": True,
    "historical_gap": {
        "description": (
            "S4 closure claimed BEST/LAST checkpoint logic PASS, while the "
            "final executable S4 SSL configs lacked an attached validation "
            "dataloader/evaluator and lacked an SSL scientific BEST/LAST hook."
        ),
        "original_final_config_audit": original_gap_rows,
        "closure_record_matches": closure_matches,
    },
    "repair_scope": {
        "fixed_validation_only": True,
        "official_ssl_model_role": "BEST_EMA_TEACHER",
        "qpseudo_checkpoint_role": "LAST_EMA_TEACHER",
        "scientific_checkpoint_serialization_scope": "EMA_TEACHER_ONLY",
        "latest_resume_role": "OPERATIONAL_ONLY",
        "test_used": False,
        "hidden_u_gt_used": False,
    },
    "canonical_sha256": CANONICAL_SHA256,
    "repair_sha256": REPAIR_SHA256,
    "matrix": matrix,
    "check_count": len(checks),
    "failed_count": len(failed),
    "failed_checks": failed,
    "checks": checks,
}

EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
EVIDENCE_PATH.write_text(
    json.dumps(evidence, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print("===== S5.06 SSL BEST/LAST CONTROLLED REPAIR PREFLIGHT =====")
print(f"EVIDENCE={EVIDENCE_PATH}")
print(f"EVIDENCE_SHA256={sha256(EVIDENCE_PATH)}")
print(f"CHECK_COUNT={len(checks)}")
print(f"FAILED_COUNT={len(failed)}")
print("FAILED_CHECKS=" + (",".join(failed) if failed else "NONE"))
print(f"STATUS={status}")
print("SCIENTIFIC_PROTOCOL_CHANGED=False")
print("HISTORICAL_S4_GATE17_OVERCLAIM_DETECTED=True")
print("TEST_USED=False")
print("HIDDEN_U_GT_USED=False")

if failed:
    raise SystemExit(1)
