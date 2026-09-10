from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import os
import runpy
import sys
from pathlib import Path

import torch
from mmengine.config import Config
from mmdet.registry import DATASETS, MODELS
from mmdet.utils import register_all_modules


EXPECTED_EFFECTIVE_BATCH = 4

CONFIGS = {
    "R50": (
        "configs/supervised/s3_07_faster_rcnn_r50_fpn_sup_batch4.py",
        "SGD",
    ),
    "SWIN_T": (
        "configs/supervised/s3_07_faster_rcnn_swin_t_fpn_sup_batch4.py",
        "AdamW",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/workspace/ssod/project")
    parser.add_argument(
        "--output",
        default=(
            "/workspace/ssod/artifacts/preflight/sup/"
            "supervised_effective_batch_manifest.json"
        ),
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_path = Path(args.output).resolve()
    sys.path.insert(0, str(repo_root))

    register_all_modules()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    dataset_cfg = runpy.run_path(
        str(repo_root / "configs/dataset/s2_coco_dataset.py")
    )
    dataset = DATASETS.build(dataset_cfg["DATASETS"]["L_1pct"])
    samples = [dataset[i] for i in range(4)]

    checks = {}
    results = {}

    for arch, (rel_config, expected_optimizer) in CONFIGS.items():
        cfg = Config.fromfile(str(repo_root / rel_config))

        batch_size = int(cfg.train_dataloader.batch_size)
        accumulation = int(cfg.optim_wrapper.accumulative_counts)
        effective_batch = batch_size * world_size * accumulation

        arch_checks = {
            "batch_size_is_4": batch_size == 4,
            "accumulative_counts_is_1": accumulation == 1,
            "effective_labeled_batch_is_4": (
                effective_batch == EXPECTED_EFFECTIVE_BATCH
            ),
            "amp_wrapper_preserved": (
                cfg.optim_wrapper.type == "AmpOptimWrapper"
            ),
            "optimizer_preserved": (
                cfg.optim_wrapper.optimizer.type == expected_optimizer
            ),
            "filter_empty_gt_false": (
                cfg.train_dataloader.dataset.filter_cfg.filter_empty_gt
                is False
            ),
            "labeled_1pct_dataset": str(
                cfg.train_dataloader.dataset.ann_file
            ).endswith("instances_labeled_1pct.json"),
        }

        data = {
            "inputs": [s["inputs"] for s in samples],
            "data_samples": [s["data_samples"] for s in samples],
        }

        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                model = MODELS.build(cfg.model)
                model.init_weights()

            model = model.cuda().train()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            data = model.data_preprocessor(data, training=True)

            with torch.cuda.amp.autocast():
                losses = model(
                    inputs=data["inputs"],
                    data_samples=data["data_samples"],
                    mode="loss",
                )
                total_loss = sum(
                    sum(v) if isinstance(v, (list, tuple)) else v
                    for k, v in losses.items()
                    if "loss" in k
                )

            total_loss.backward()

            actual_batch = int(data["inputs"].shape[0])
            loss_finite = bool(torch.isfinite(total_loss).item())
            peak_memory_mb = round(
                torch.cuda.max_memory_allocated() / (1024 ** 2), 2
            )

            arch_checks["actual_runtime_batch_is_4"] = actual_batch == 4
            arch_checks["forward_backward_pass"] = True
            arch_checks["loss_finite"] = loss_finite

        except torch.cuda.OutOfMemoryError:
            actual_batch = None
            loss_finite = False
            peak_memory_mb = None
            arch_checks["actual_runtime_batch_is_4"] = False
            arch_checks["forward_backward_pass"] = False
            arch_checks["loss_finite"] = False

        checks.update({
            f"{arch.lower()}_{k}": v
            for k, v in arch_checks.items()
        })

        results[arch] = {
            "config": rel_config,
            "batch_size_per_process": batch_size,
            "world_size": world_size,
            "accumulative_counts": accumulation,
            "effective_labeled_batch": effective_batch,
            "actual_runtime_batch": actual_batch,
            "peak_memory_mb": peak_memory_mb,
            "loss_finite": loss_finite,
        }

        if "model" in locals():
            del model
        gc.collect()
        torch.cuda.empty_cache()

    checks["official_training_remains_unauthorized"] = True

    failed_checks = [
        name for name, passed in checks.items() if not passed
    ]

    report = {
        "schema_version": "1.0",
        "stage": "S3.07",
        "scope": "supervised_effective_labeled_batch",
        "official_training_authorized": False,
        "scientific_contract": {
            "effective_labeled_batch": EXPECTED_EFFECTIVE_BATCH,
            "execution_decomposition_is_implementation_detail": True,
        },
        "runtime": {
            "cuda_device_count": torch.cuda.device_count(),
            "world_size_env": os.environ.get("WORLD_SIZE"),
            "resolved_world_size": world_size,
        },
        "architectures": results,
        "checks": checks,
        "check_count": len(checks),
        "failed_checks": failed_checks,
        "all_checks_pass": not failed_checks,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    for name, passed in checks.items():
        print(f"{name} = {passed}")

    print("failed_checks =", failed_checks)
    print("ALL_CHECKS_PASS =", not failed_checks)

    if failed_checks:
        print("S3_07_EFFECTIVE_LABELED_BATCH=FAIL")
        return 1

    print("S3_07_EFFECTIVE_LABELED_BATCH=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())