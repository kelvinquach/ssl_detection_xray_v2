#!/usr/bin/env python3
"""S2.10 preflight for actual optimizer-update accounting."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import mmengine

from mmengine.optim import AmpOptimWrapper, OptimWrapper


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from src.utils.optimizer_update_counter import (
        ActualOptimizerUpdateCounter,
    )

    checks = {}

    print("torch =", torch.__version__)
    print("mmengine =", mmengine.__version__)
    print("cuda_available =", torch.cuda.is_available())

    # --------------------------------------------------------------
    # TEST 1: direct optimizer.step()
    # --------------------------------------------------------------
    print("\n=== TEST 1: DIRECT OPTIMIZER STEP ===")

    p1 = torch.nn.Parameter(torch.tensor([1.0]))
    opt1 = torch.optim.SGD([p1], lr=0.1)

    counter1 = ActualOptimizerUpdateCounter()
    counter1.attach(opt1)

    count_before = counter1.count

    opt1.zero_grad()
    (p1 ** 2).sum().backward()
    opt1.step()

    count_after = counter1.count

    checks["direct_optimizer_step_counted_once"] = (
        count_before == 0
        and count_after == 1
    )

    counter1.detach()

    print("count_before =", count_before)
    print("count_after =", count_after)
    print(
        "direct_optimizer_step_counted_once =",
        checks["direct_optimizer_step_counted_once"],
    )

    # --------------------------------------------------------------
    # TEST 2: MMEngine gradient accumulation
    # --------------------------------------------------------------
    print("\n=== TEST 2: GRADIENT ACCUMULATION ===")

    p2 = torch.nn.Parameter(torch.tensor([1.0]))
    opt2 = torch.optim.SGD([p2], lr=0.1)

    wrapper2 = OptimWrapper(
        optimizer=opt2,
        accumulative_counts=2,
    )

    counter2 = ActualOptimizerUpdateCounter()
    counter2.attach(opt2)

    before = p2.detach().clone()

    wrapper2.update_params((p2 ** 2).sum())

    after_raw_iter_1 = p2.detach().clone()
    count_after_raw_iter_1 = counter2.count

    wrapper2.update_params((p2 ** 2).sum())

    after_raw_iter_2 = p2.detach().clone()
    count_after_raw_iter_2 = counter2.count

    checks["raw_iteration_not_counted_as_optimizer_update"] = (
        count_after_raw_iter_1 == 0
        and torch.equal(before, after_raw_iter_1)
    )

    checks["accumulation_boundary_counted_once"] = (
        count_after_raw_iter_2 == 1
        and not torch.equal(
            after_raw_iter_1,
            after_raw_iter_2,
        )
    )

    counter2.detach()

    print(
        "count_after_raw_iter_1 =",
        count_after_raw_iter_1,
    )
    print(
        "count_after_raw_iter_2 =",
        count_after_raw_iter_2,
    )
    print(
        "raw_iteration_not_counted_as_optimizer_update =",
        checks[
            "raw_iteration_not_counted_as_optimizer_update"
        ],
    )
    print(
        "accumulation_boundary_counted_once =",
        checks["accumulation_boundary_counted_once"],
    )

    # --------------------------------------------------------------
    # TEST 3: MMEngine AMP finite vs skipped step
    # --------------------------------------------------------------
    print("\n=== TEST 3: AMP FINITE + SKIPPED STEP ===")

    if not torch.cuda.is_available():
        checks["amp_runtime_available"] = False
        checks["finite_amp_step_counted_once"] = False
        checks["amp_skipped_step_not_counted"] = False

        print("AMP test requires CUDA runtime.")
    else:
        checks["amp_runtime_available"] = True

        device = "cuda"

        p3 = torch.nn.Parameter(
            torch.tensor([1.0], device=device)
        )
        opt3 = torch.optim.SGD([p3], lr=0.1)

        wrapper3 = AmpOptimWrapper(
            optimizer=opt3,
            accumulative_counts=1,
        )

        counter3 = ActualOptimizerUpdateCounter()
        counter3.attach(opt3)

        before_finite = p3.detach().clone()

        wrapper3.update_params((p3 ** 2).sum())

        after_finite = p3.detach().clone()
        count_after_finite = counter3.count

        checks["finite_amp_step_counted_once"] = (
            count_after_finite == 1
            and not torch.equal(
                before_finite,
                after_finite,
            )
        )

        before_skip = p3.detach().clone()

        inf_value = torch.tensor(
            float("inf"),
            device=device,
        )

        wrapper3.update_params(
            (p3 * inf_value).sum()
        )

        after_skip = p3.detach().clone()
        count_after_skip = counter3.count

        checks["amp_skipped_step_not_counted"] = (
            count_after_skip == count_after_finite
            and torch.equal(
                before_skip,
                after_skip,
            )
        )

        counter3.detach()

        print(
            "count_after_finite =",
            count_after_finite,
        )
        print(
            "finite_amp_step_counted_once =",
            checks["finite_amp_step_counted_once"],
        )
        print(
            "count_after_amp_skip =",
            count_after_skip,
        )
        print(
            "amp_skipped_step_not_counted =",
            checks["amp_skipped_step_not_counted"],
        )


    # --------------------------------------------------------------
    # TEST 4: AdamW AMP finite vs skipped step
    # Covers the Swin-T optimizer family used by the project.
    # --------------------------------------------------------------
    print("\n=== TEST 4: ADAMW AMP FINITE + SKIPPED STEP ===")

    if not torch.cuda.is_available():
        checks["adamw_amp_runtime_available"] = False
        checks["adamw_finite_step_counted_once"] = False
        checks["adamw_amp_skipped_step_not_counted"] = False

        print("AdamW AMP test requires CUDA runtime.")
    else:
        checks["adamw_amp_runtime_available"] = True

        device = "cuda"

        p4 = torch.nn.Parameter(
            torch.tensor([1.0], device=device)
        )

        opt4 = torch.optim.AdamW(
            [p4],
            lr=1e-3,
            weight_decay=0.05,
        )

        wrapper4 = AmpOptimWrapper(
            optimizer=opt4,
            accumulative_counts=1,
        )

        counter4 = ActualOptimizerUpdateCounter()
        counter4.attach(opt4)

        before_finite = p4.detach().clone()

        wrapper4.update_params((p4 ** 2).sum())

        after_finite = p4.detach().clone()
        count_after_finite = counter4.count

        checks["adamw_finite_step_counted_once"] = (
            count_after_finite == 1
            and not torch.equal(
                before_finite,
                after_finite,
            )
        )

        before_skip = p4.detach().clone()

        inf_value = torch.tensor(
            float("inf"),
            device=device,
        )

        wrapper4.update_params(
            (p4 * inf_value).sum()
        )

        after_skip = p4.detach().clone()
        count_after_skip = counter4.count

        checks["adamw_amp_skipped_step_not_counted"] = (
            count_after_skip == count_after_finite
            and torch.equal(
                before_skip,
                after_skip,
            )
        )

        counter4.detach()

        print(
            "adamw_count_after_finite =",
            count_after_finite,
        )
        print(
            "adamw_finite_step_counted_once =",
            checks["adamw_finite_step_counted_once"],
        )
        print(
            "adamw_count_after_amp_skip =",
            count_after_skip,
        )
        print(
            "adamw_amp_skipped_step_not_counted =",
            checks["adamw_amp_skipped_step_not_counted"],
        )

    status = "PASS" if all(checks.values()) else "FAIL"

    print("\n=== CHECKS ===")
    for name, value in checks.items():
        print(f"{name} = {value}")

    print(
        "S2_10_ACTUAL_OPTIMIZER_UPDATE_COUNTER="
        f"{status}"
    )

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
