#!/usr/bin/env python3
"""S3.05 — Supervised AMP / gradient-clipping numerics preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.optim import AmpOptimWrapper, build_optim_wrapper
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules


EXPECTED_CONFIG_SHA256 = {
    "R50": "d3d68045630e8a2eaa8fec6118619312b33b7d2f7df683d5af0a486d9e002e5a",
    "Swin-T": "49efb3d3f18dd9eccb47e20e171d9d9e55d9fa7c54e8276355eef88d3008521c",
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/workspace/ssod/project")
    parser.add_argument(
        "--output",
        default=(
            "/workspace/ssod/artifacts/preflight/sup/"
            "sup_amp_numerics_preflight.json"
        ),
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_path = Path(args.output).resolve()
    sys.path.insert(0, str(repo_root))

    from src.utils.optimizer_update_counter import ActualOptimizerUpdateCounter

    register_all_modules()

    if not torch.cuda.is_available():
        raise RuntimeError("S3.05 AMP runtime preflight requires CUDA.")

    config_paths = {
        "R50": repo_root / "configs/supervised/s3_05_faster_rcnn_r50_fpn_sup_amp.py",
        "Swin-T": repo_root / "configs/supervised/s3_05_faster_rcnn_swin_t_fpn_sup_amp.py",
    }

    observed = {}
    checks = {}

    for arch, path in config_paths.items():
        config_sha = file_sha256(path)
        cfg = Config.fromfile(str(path))

        model = MODELS.build(cfg.model)
        wrapper = build_optim_wrapper(model, cfg.optim_wrapper)

        checks[f"{arch}_config_sha256_match"] = (
            config_sha == EXPECTED_CONFIG_SHA256[arch]
        )
        checks[f"{arch}_amp_wrapper"] = (
            type(wrapper).__name__ == "AmpOptimWrapper"
        )
        checks[f"{arch}_grad_scaler"] = (
            type(wrapper.loss_scaler).__name__ == "GradScaler"
        )
        checks[f"{arch}_dynamic_loss_scale"] = (
            cfg.optim_wrapper.get("loss_scale") == "dynamic"
        )
        checks[f"{arch}_gradient_clipping_off"] = (
            wrapper.clip_grad_kwargs is None
        )

        if arch == "R50":
            opt = wrapper.optimizer
            checks["R50_optimizer_recipe_preserved"] = (
                type(opt).__name__ == "SGD"
                and opt.defaults["lr"] == 0.005
                and opt.defaults["momentum"] == 0.9
                and opt.defaults["weight_decay"] == 0.0001
            )
        else:
            opt = wrapper.optimizer
            checks["Swin_T_optimizer_recipe_preserved"] = (
                type(opt).__name__ == "AdamW"
                and opt.defaults["lr"] == 0.000025
                and tuple(opt.defaults["betas"]) == (0.9, 0.999)
                and opt.defaults["weight_decay"] == 0.05
            )

            custom_keys = cfg.optim_wrapper.paramwise_cfg.custom_keys
            checks["Swin_T_native_no_decay_rules_preserved"] = (
                custom_keys["absolute_pos_embed"]["decay_mult"] == 0.0
                and custom_keys["relative_position_bias_table"]["decay_mult"] == 0.0
                and custom_keys["norm"]["decay_mult"] == 0.0
            )

        observed[arch] = {
            "config_path": path.relative_to(repo_root).as_posix(),
            "config_sha256": config_sha,
            "wrapper_class": type(wrapper).__name__,
            "scaler_class": type(wrapper.loss_scaler).__name__,
            "loss_scale": cfg.optim_wrapper.get("loss_scale"),
            "clip_grad_kwargs": wrapper.clip_grad_kwargs,
            "optimizer_class": type(wrapper.optimizer).__name__,
            "optimizer_defaults": {
                key: (
                    list(value)
                    if isinstance(value, tuple)
                    else value
                )
                for key, value in wrapper.optimizer.defaults.items()
                if isinstance(value, (str, int, float, bool, tuple))
                or value is None
            },
        }

        del wrapper
        del model
        torch.cuda.empty_cache()

    # Reuse the locked S2.10/S2.12 AMP accounting pattern.
    p = torch.nn.Parameter(torch.tensor([1.0], device="cuda"))
    optimizer = torch.optim.SGD([p], lr=0.005)
    amp_wrapper = AmpOptimWrapper(
        optimizer=optimizer,
        accumulative_counts=1,
    )

    counter = ActualOptimizerUpdateCounter()
    counter.attach(optimizer)

    finite_before = counter.count
    finite_scale_before = float(amp_wrapper.loss_scaler.get_scale())
    amp_wrapper.update_params((p ** 2).sum())
    finite_after = counter.count
    finite_scale_after = float(amp_wrapper.loss_scaler.get_scale())

    checks["finite_amp_actual_optimizer_update"] = (
        finite_after == finite_before + 1
    )

    skip_before = counter.count
    skip_scale_before = float(amp_wrapper.loss_scaler.get_scale())
    inf_value = torch.tensor(float("inf"), device="cuda")
    amp_wrapper.update_params((p * inf_value).sum())
    skip_after = counter.count
    skip_scale_after = float(amp_wrapper.loss_scaler.get_scale())

    checks["non_finite_amp_step_skipped"] = (
        skip_after == skip_before
    )
    checks["grad_scaler_backoff_observed_on_skip"] = (
        skip_scale_after < skip_scale_before
    )

    counter.detach()

    failed_checks = [
        name for name, passed in checks.items()
        if not passed
    ]

    report = {
        "schema_version": "1.0",
        "stage": "S3.05",
        "scope": "supervised_amp_gradient_clipping_numerics_preflight",
        "official_training_authorized": False,
        "scientific_contract": {
            "amp": True,
            "gradient_clipping": False,
        },
        "implementation_choice": {
            "loss_scale": "dynamic",
            "basis": "MMEngine 0.10.7 canonical/default AMP behavior",
        },
        "architectures": observed,
        "amp_runtime": {
            "finite_counter_before": finite_before,
            "finite_counter_after": finite_after,
            "finite_scale_before": finite_scale_before,
            "finite_scale_after": finite_scale_after,
            "skip_counter_before": skip_before,
            "skip_counter_after": skip_after,
            "skip_scale_before": skip_scale_before,
            "skip_scale_after": skip_scale_after,
        },
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
        print("S3_05_AMP_NUMERICS_PREFLIGHT=FAIL")
        return 1

    print("S3_05_AMP_NUMERICS_PREFLIGHT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())