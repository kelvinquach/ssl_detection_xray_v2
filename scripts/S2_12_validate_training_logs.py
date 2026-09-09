#!/usr/bin/env python3
"""S2.12 preflight for train.jsonl and runtime_events.jsonl."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import torch
from mmengine.optim import AmpOptimWrapper

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.utils.optimizer_update_counter import (
    ActualOptimizerUpdateCounter,
)
from src.utils.training_event_logger import (
    SSL_TRAIN_ADDITIONAL_FIELDS,
    SUP_TRAIN_REQUIRED_FIELDS,
    TrainingEventLogger,
)


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> int:
    out_dir = Path(
        "/workspace/ssod/artifacts/preflight/training/"
        "s2_12_sample_logs"
    )
    train_path = out_dir / "train.jsonl"
    events_path = out_dir / "runtime_events.jsonl"

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = TrainingEventLogger(
        train_log_path=train_path,
        runtime_events_path=events_path,
    )

    checks: dict[str, bool] = {}

    print("torch =", torch.__version__)
    print("cuda_available =", torch.cuda.is_available())

    if not torch.cuda.is_available():
        print("S2.12 AMP runtime validation requires CUDA.")
        return 1

    # --------------------------------------------------------------
    # TEST 1: finite AMP step -> one train.jsonl actual-update record
    # --------------------------------------------------------------
    print("\n=== TEST 1: FINITE AMP ACTUAL UPDATE ===")

    p1 = torch.nn.Parameter(
        torch.tensor([1.0], device="cuda")
    )
    optimizer = torch.optim.SGD([p1], lr=0.005)

    wrapper = AmpOptimWrapper(
        optimizer=optimizer,
        accumulative_counts=1,
    )

    counter = ActualOptimizerUpdateCounter()
    counter.attach(optimizer)

    raw_iteration = 1
    counter_before = counter.count
    scale_before = float(wrapper.loss_scaler.get_scale())

    loss = (p1 ** 2).sum()
    loss_value = float(loss.detach().cpu())
    wrapper.update_params(loss)

    counter_after = counter.count
    scale_after = float(wrapper.loss_scaler.get_scale())

    step_executed = counter_after == counter_before + 1

    checks["finite_amp_actual_update_detected"] = step_executed

    if step_executed:
        logger.log_optimizer_update(
            {
                "optimizer_update": counter_after,
                "raw_iteration": raw_iteration,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "loss_total": loss_value,
                "loss_supervised": loss_value,
                "amp_scale": scale_before,
                "optimizer_step_executed": True,
                "elapsed_seconds": 0.0,
                "gpu_memory_allocated_mb": (
                    torch.cuda.memory_allocated() / (1024 ** 2)
                ),
                "labeled_microbatches_consumed_cumulative": 1,
                "labeled_images_consumed_cumulative": 4,
            },
            ssl=False,
        )

    print("counter_before =", counter_before)
    print("counter_after =", counter_after)
    print("scale_before =", scale_before)
    print("scale_after =", scale_after)
    print("optimizer_step_executed =", step_executed)

    # --------------------------------------------------------------
    # TEST 2: non-finite AMP attempt -> runtime event, no train row
    # --------------------------------------------------------------
    print("\n=== TEST 2: AMP SKIPPED STEP ===")

    raw_iteration += 1
    counter_before_skip = counter.count
    scale_before_skip = float(wrapper.loss_scaler.get_scale())

    inf_value = torch.tensor(float("inf"), device="cuda")
    wrapper.update_params((p1 * inf_value).sum())

    counter_after_skip = counter.count
    scale_after_skip = float(wrapper.loss_scaler.get_scale())

    skip_detected = counter_after_skip == counter_before_skip

    checks["amp_skip_detected_by_actual_update_counter"] = skip_detected

    if skip_detected:
        logger.log_amp_skip(
            optimizer_update_before=counter_before_skip,
            reason="non_finite_gradient",
            amp_scale_before=scale_before_skip,
            amp_scale_after=scale_after_skip,
            ssl=False,
        )

    print("counter_before_skip =", counter_before_skip)
    print("counter_after_skip =", counter_after_skip)
    print("scale_before_skip =", scale_before_skip)
    print("scale_after_skip =", scale_after_skip)
    print("amp_skip_detected =", skip_detected)

    counter.detach()

    # --------------------------------------------------------------
    # TEST 3: contract validation of produced sample logs
    # --------------------------------------------------------------
    print("\n=== TEST 3: SAMPLE LOG CONTRACT ===")

    train_records = read_jsonl(train_path)
    event_records = read_jsonl(events_path)

    checks["train_jsonl_exists"] = train_path.is_file()
    checks["runtime_events_jsonl_exists"] = events_path.is_file()

    checks["one_train_record_for_one_actual_update"] = (
        len(train_records) == 1
        and train_records[0]["optimizer_update"] == 1
        and train_records[0]["raw_iteration"] == 1
        and train_records[0]["optimizer_step_executed"] is True
    )

    checks["sup_train_required_fields_present"] = (
        len(train_records) == 1
        and all(
            key in train_records[0]
            for key in SUP_TRAIN_REQUIRED_FIELDS
        )
    )

    checks["amp_skip_not_written_to_train_jsonl"] = (
        len(train_records) == counter_after_skip
        and counter_after_skip == 1
    )

    amp_skip_events = [
        record
        for record in event_records
        if record.get("event")
        == "OPTIMIZER_STEP_SKIPPED_BY_AMP"
    ]

    checks["amp_skip_runtime_event_present"] = (
        len(amp_skip_events) == 1
    )

    checks["amp_skip_runtime_event_contract"] = (
        len(amp_skip_events) == 1
        and "timestamp_utc" in amp_skip_events[0]
        and amp_skip_events[0]["optimizer_update_before"] == 1
        and amp_skip_events[0]["reason"]
        == "non_finite_gradient"
        and amp_skip_events[0]["optimizer_step_executed"]
        is False
    )

    # --------------------------------------------------------------
    # TEST 4: logger-level SSL schema / synchronized skip evidence
    # This does NOT validate EMA implementation or timing.
    # --------------------------------------------------------------
    print("\n=== TEST 4: SSL LOGGING SCHEMA ONLY ===")

    ssl_dir = out_dir / "ssl_schema_only"
    ssl_train = ssl_dir / "train.jsonl"
    ssl_events = ssl_dir / "runtime_events.jsonl"

    ssl_logger = TrainingEventLogger(
        train_log_path=ssl_train,
        runtime_events_path=ssl_events,
    )

    ssl_record = {
        "optimizer_update": 1,
        "raw_iteration": 1,
        "learning_rate": 0.005,
        "loss_total": 1.5,
        "loss_supervised": 1.0,
        "amp_scale": 65536.0,
        "optimizer_step_executed": True,
        "elapsed_seconds": 0.0,
        "gpu_memory_allocated_mb": 0.0,
        "labeled_microbatches_consumed_cumulative": 1,
        "labeled_images_consumed_cumulative": 4,
        "loss_unsupervised": 0.5,
        "pseudo_count": 3,
        "classification_pseudo_count": 3,
        "regression_pseudo_count": 2,
        "unlabeled_microbatches_consumed_cumulative": 1,
        "unlabeled_images_consumed_cumulative": 4,
        "ema_step_executed": True,
    }

    ssl_logger.log_optimizer_update(
        ssl_record,
        ssl=True,
    )

    ssl_logger.log_amp_skip(
        optimizer_update_before=1,
        reason="non_finite_gradient",
        amp_scale_before=65536.0,
        amp_scale_after=32768.0,
        ssl=True,
    )

    ssl_train_records = read_jsonl(ssl_train)
    ssl_event_records = read_jsonl(ssl_events)

    checks["ssl_additional_fields_present"] = (
        len(ssl_train_records) == 1
        and all(
            key in ssl_train_records[0]
            for key in SSL_TRAIN_ADDITIONAL_FIELDS
        )
    )

    ssl_amp_events = [
        r for r in ssl_event_records
        if r.get("event")
        == "OPTIMIZER_STEP_SKIPPED_BY_AMP"
    ]
    ssl_ema_skip_events = [
        r for r in ssl_event_records
        if r.get("event") == "EMA_STEP_SKIPPED"
    ]

    checks["ssl_amp_skip_records_ema_false"] = (
        len(ssl_amp_events) == 1
        and ssl_amp_events[0]["optimizer_step_executed"]
        is False
        and ssl_amp_events[0]["ema_step_executed"] is False
    )

    checks["ssl_ema_skip_event_present"] = (
        len(ssl_ema_skip_events) == 1
        and ssl_ema_skip_events[0]["ema_step_executed"] is False
    )

    print("train_records =", len(train_records))
    print("runtime_event_records =", len(event_records))
    print("ssl_train_records =", len(ssl_train_records))
    print("ssl_runtime_event_records =", len(ssl_event_records))

    print("\n=== CHECKS ===")
    for name, value in checks.items():
        print(f"{name} = {value}")

    status = "PASS" if all(checks.values()) else "FAIL"

    print(f"\nS2_12_TRAINING_LOG_CONTRACT={status}")
    print(f"sample_log_dir = {out_dir}")

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
