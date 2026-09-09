"""S2.13 checkpoint/resume equivalence preflight.

Synthetic MMEngine fixture only. No project dataset and no official training.

The test verifies that an interrupted run resumed from ``latest_resume.pth``
matches an uninterrupted reference trajectory for:
- model parameters;
- optimizer state;
- AMP GradScaler state;
- parameter-scheduler state;
- actual optimizer-update counter;
- raw iteration;
- Python / NumPy / Torch CPU / Torch CUDA RNG trajectory.

It also verifies that a mismatched training seed is blocked.
"""

from __future__ import annotations

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
from mmengine.registry import DATASETS, MODELS
from mmengine.runner import Runner

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Register S2.13 custom components.
import src.utils.resume_aware_loop  # noqa: F401,E402
import src.utils.resume_checkpoint_hook  # noqa: F401,E402

from src.utils.resume_checkpoint_hook import (  # noqa: E402
    RUNNER_COUNTER_ATTR,
)
from src.utils.seed import get_full_rng_state  # noqa: E402


SEED = 204886845
TOTAL_ITERS = 6
INTERRUPT_ITERS = 3
OUT_ROOT = Path(
    "/workspace/ssod/artifacts/preflight/resume/s2_13_fixture"
)
CHECKPOINT_PATH = OUT_ROOT / "interrupted" / "latest_resume.pth"
SSL_CHECKPOINT_PATH = (
    OUT_ROOT / "ssl_teacher_student" / "latest_resume.pth"
)
REPORT_PATH = Path(
    "/workspace/ssod/artifacts/preflight/resume/"
    "resume_equivalence_test.json"
)


@MODELS.register_module(force=True)
class S213ToyModel(BaseModel):
    """Tiny CUDA-safe MMEngine model for resume equivalence testing."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, inputs, data_samples=None, mode="tensor"):
        return self.linear(inputs)

    def train_step(self, data, optim_wrapper):
        device = self.linear.weight.device
        x = data["inputs"].float().to(device)
        target = data["target"].float().to(device)

        with optim_wrapper.optim_context(self):
            pred = self.linear(x)
            loss = ((pred - target) ** 2).mean()

        optim_wrapper.update_params(loss)
        return {"loss": loss.detach()}


@MODELS.register_module(force=True)
class S213ToySSLStateModel(BaseModel):
    """Synthetic Student/Teacher model for resume-state preservation only."""

    def __init__(self) -> None:
        super().__init__()

        self.student = torch.nn.Linear(1, 1)
        self.teacher = torch.nn.Linear(1, 1)

        # Start Teacher distinct from Student and keep Teacher frozen.
        with torch.no_grad():
            self.teacher.load_state_dict(self.student.state_dict())
            for parameter in self.teacher.parameters():
                parameter.add_(1.0)

        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)

    def forward(self, inputs, data_samples=None, mode="tensor"):
        return self.student(inputs)

    def train_step(self, data, optim_wrapper):
        device = self.student.weight.device
        x = data["inputs"].float().to(device)
        target = data["target"].float().to(device)

        with optim_wrapper.optim_context(self):
            pred = self.student(x)
            loss = ((pred - target) ** 2).mean()

        optim_wrapper.update_params(loss)
        return {"loss": loss.detach()}


@DATASETS.register_module(force=True)
class S213ToyDataset(torch.utils.data.Dataset):
    """Tiny registry-built dataset for the S2.13 resume fixture."""

    metainfo = {"fixture": "S2.13"}

    def __len__(self):
        return 8

    def __getitem__(self, idx):
        # Deliberately consume all three main-process RNG families.
        py = random.random()
        nr = float(np.random.rand())
        tr = float(torch.rand(1).item())

        x = torch.tensor(
            [[float(idx) + py + nr + tr]],
            dtype=torch.float32,
        ).squeeze(0)
        target = torch.tensor(
            [[float(idx) * 0.1 + 0.25]],
            dtype=torch.float32,
        ).squeeze(0)

        return {"inputs": x, "target": target}


def build_runner(
    *,
    work_dir: Path,
    max_iters: int,
    resume: bool = False,
    load_from: str | None = None,
    seed: int = SEED,
    model_type: str = "S213ToyModel",
) -> Runner:
    cfg = Config(
        dict(
            model=dict(type=model_type),
            work_dir=str(work_dir),
            train_dataloader=dict(
                batch_size=1,
                num_workers=0,
                persistent_workers=False,
                sampler=dict(
                    type="DefaultSampler",
                    shuffle=True,
                ),
                collate_fn=dict(
                    type="default_collate",
                ),
                dataset=dict(
                    type="S213ToyDataset",
                ),
            ),
            train_cfg=dict(
                type="ResumeAwareIterBasedTrainLoop",
                max_iters=max_iters,
                val_interval=999999,
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
                    milestones=[2, 4],
                    gamma=0.1,
                )
            ],
            custom_hooks=[
                dict(type="ProtocolResumeCheckpointHook")
            ],
            default_hooks=dict(
                checkpoint=dict(
                    type="CheckpointHook",
                    interval=-1,
                )
            ),
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


def snapshot(runner: Runner) -> dict[str, Any]:
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


class SyntheticInterruption(RuntimeError):
    """Intentional stop used only by the S2.13 synthetic fixture."""


class SaveAndInterruptHook(Hook):
    """Save a resume checkpoint at raw iteration 3, then stop."""

    def after_train_iter(
        self,
        runner,
        batch_idx,
        data_batch=None,
        outputs=None,
    ):
        # after_train_iter observes zero-based runner.iter.
        if int(runner.iter) != INTERRUPT_ITERS - 1:
            return

        runner.save_checkpoint(
            out_dir=str(CHECKPOINT_PATH.parent),
            filename=CHECKPOINT_PATH.name,
            save_optimizer=True,
            save_param_scheduler=True,
            by_epoch=False,
        )

        raise SyntheticInterruption(
            f"Synthetic interruption after {INTERRUPT_ITERS} raw iterations."
        )


def _prefixed_state(
    state: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    return {
        key[len(prefix):]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }


def run_ssl_teacher_student_resume_fixture(
    device: torch.device,
) -> dict[str, bool]:
    """Verify Student + Teacher preservation without Teacher resync.

    This is a resume-state schema fixture only. It does not implement or
    validate the EMA update algorithm, which belongs to the SSL stage.
    """

    print("\n=== SSL TEACHER/STUDENT RESUME STATE ===")

    source_runner = build_runner(
        work_dir=OUT_ROOT / "ssl_teacher_student" / "source",
        max_iters=1,
        model_type="S213ToySSLStateModel",
    )
    source_runner.model.to(device)
    source_runner.train()

    # Materialize deliberately different states that cannot be confused
    # with fresh initialization or Student->Teacher resynchronization.
    with torch.no_grad():
        for parameter in source_runner.model.student.parameters():
            parameter.fill_(0.125)

        for parameter in source_runner.model.teacher.parameters():
            parameter.fill_(1.875)

    SSL_CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_runner.save_checkpoint(
        out_dir=str(SSL_CHECKPOINT_PATH.parent),
        filename=SSL_CHECKPOINT_PATH.name,
        save_optimizer=True,
        save_param_scheduler=True,
        meta={
            "epoch": int(source_runner.epoch),
            "iter": int(source_runner.iter),
        },
        by_epoch=False,
    )

    checkpoint = torch.load(
        SSL_CHECKPOINT_PATH,
        map_location="cpu",
    )
    checkpoint_state = checkpoint["state_dict"]

    checkpoint_student = _prefixed_state(
        checkpoint_state,
        "student.",
    )
    checkpoint_teacher = _prefixed_state(
        checkpoint_state,
        "teacher.",
    )

    restored_runner = build_runner(
        work_dir=OUT_ROOT / "ssl_teacher_student" / "restored",
        max_iters=1,
        resume=True,
        load_from=str(SSL_CHECKPOINT_PATH),
        model_type="S213ToySSLStateModel",
    )
    restored_runner.model.to(device)

    fresh_state = tensor_state_cpu(restored_runner.model)
    fresh_student = _prefixed_state(
        fresh_state,
        "student.",
    )
    fresh_teacher = _prefixed_state(
        fresh_state,
        "teacher.",
    )

    restored_runner.train()

    restored_state = tensor_state_cpu(restored_runner.model)
    restored_student = _prefixed_state(
        restored_state,
        "student.",
    )
    restored_teacher = _prefixed_state(
        restored_state,
        "teacher.",
    )

    checks = {
        "ssl_checkpoint_exists": SSL_CHECKPOINT_PATH.exists(),
        "ssl_checkpoint_contains_student_state": (
            bool(checkpoint_student)
        ),
        "ssl_checkpoint_contains_teacher_state": (
            bool(checkpoint_teacher)
        ),
        "ssl_student_checkpoint_differs_from_fresh_init": (
            not nested_equal(
                checkpoint_student,
                fresh_student,
            )
        ),
        "ssl_teacher_checkpoint_differs_from_fresh_init": (
            not nested_equal(
                checkpoint_teacher,
                fresh_teacher,
            )
        ),
        "ssl_student_state_restored": nested_equal(
            restored_student,
            checkpoint_student,
        ),
        "ssl_teacher_state_restored": nested_equal(
            restored_teacher,
            checkpoint_teacher,
        ),
        "ssl_teacher_not_resynced_from_student": (
            not nested_equal(
                restored_teacher,
                restored_student,
            )
        ),
    }

    print("ssl_checkpoint =", SSL_CHECKPOINT_PATH)
    for key, value in checks.items():
        print(f"{key} = {value}")

    return checks


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA_REQUIRED = FAIL")
        return 1

    shutil.rmtree(OUT_ROOT, ignore_errors=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")

    print("torch =", torch.__version__)
    print("cuda_available =", torch.cuda.is_available())
    print("seed =", SEED)

    # ------------------------------------------------------------
    # Reference uninterrupted trajectory.
    # ------------------------------------------------------------
    print("\n=== REFERENCE UNINTERRUPTED RUN ===")

    ref_runner = build_runner(
        work_dir=OUT_ROOT / "reference",
        max_iters=TOTAL_ITERS,
    )
    ref_runner.model.to(device)
    ref_runner.train()

    ref_snapshot = snapshot(ref_runner)
    ref_rng_probe = rng_probe()

    print("reference_iter =", ref_snapshot["iter"])
    print(
        "reference_counter =",
        ref_snapshot["counter"]["actual_optimizer_updates"],
    )

    # ------------------------------------------------------------
    # Interrupted first segment.
    # ------------------------------------------------------------
    print("\n=== INTERRUPTED SEGMENT ===")

    first_runner = build_runner(
        work_dir=OUT_ROOT / "interrupted",
        max_iters=TOTAL_ITERS,
    )
    first_runner.model.to(device)
    first_runner.register_hook(
        SaveAndInterruptHook(),
        priority="VERY_LOW",
    )

    synthetic_interrupt_observed = False
    try:
        first_runner.train()
    except SyntheticInterruption as exc:
        synthetic_interrupt_observed = True
        print("synthetic_interrupt =", str(exc))

    if not synthetic_interrupt_observed:
        raise RuntimeError(
            "Synthetic interruption hook did not stop the fixture."
        )

    print("checkpoint_exists =", CHECKPOINT_PATH.exists())
    print(
        "interrupt_counter =",
        getattr(
            first_runner,
            RUNNER_COUNTER_ATTR,
        ).count,
    )

    # ------------------------------------------------------------
    # Resume to the same TOTAL_ITERS target.
    # ------------------------------------------------------------
    print("\n=== RESUMED SEGMENT ===")

    resumed_runner = build_runner(
        work_dir=OUT_ROOT / "resumed",
        max_iters=TOTAL_ITERS,
        resume=True,
        load_from=str(CHECKPOINT_PATH),
    )
    resumed_runner.model.to(device)
    resumed_runner.train()

    resumed_snapshot = snapshot(resumed_runner)
    resumed_rng_probe = rng_probe()

    print("resumed_iter =", resumed_snapshot["iter"])
    print(
        "resumed_counter =",
        resumed_snapshot["counter"]["actual_optimizer_updates"],
    )

    # ------------------------------------------------------------
    # Seed mismatch guardrail.
    # ------------------------------------------------------------
    print("\n=== SEED MISMATCH GUARDRAIL ===")

    seed_mismatch_blocked = False
    seed_mismatch_error = None

    try:
        bad_runner = build_runner(
            work_dir=OUT_ROOT / "bad_seed",
            max_iters=TOTAL_ITERS,
            resume=True,
            load_from=str(CHECKPOINT_PATH),
            seed=SEED + 1,
        )
        bad_runner.model.to(device)
        bad_runner.train()
    except RuntimeError as exc:
        seed_mismatch_blocked = (
            "Training-seed mismatch on resume" in str(exc)
        )
        seed_mismatch_error = str(exc)

    print("seed_mismatch_blocked =", seed_mismatch_blocked)
    print("seed_mismatch_error =", seed_mismatch_error)

    ssl_checks = run_ssl_teacher_student_resume_fixture(
        device,
    )

    checks = {
        "checkpoint_exists": CHECKPOINT_PATH.exists(),
        "final_raw_iteration_match": (
            resumed_snapshot["iter"]
            == ref_snapshot["iter"]
            == TOTAL_ITERS
        ),
        "actual_optimizer_update_counter_match": (
            resumed_snapshot["counter"]
            == ref_snapshot["counter"]
        ),
        "model_state_match": nested_equal(
            resumed_snapshot["model"],
            ref_snapshot["model"],
        ),
        "optimizer_and_amp_scaler_state_match": nested_equal(
            resumed_snapshot["optimizer"],
            ref_snapshot["optimizer"],
        ),
        "scheduler_state_match": nested_equal(
            resumed_snapshot["scheduler"],
            ref_snapshot["scheduler"],
        ),
        "rng_state_match_at_run_end": nested_equal(
            resumed_snapshot["rng"],
            ref_snapshot["rng"],
        ),
        "rng_next_draw_match": (
            resumed_rng_probe == ref_rng_probe
        ),
        "seed_mismatch_blocked": seed_mismatch_blocked,
    }
    checks.update(ssl_checks)

    all_pass = all(checks.values())

    report = {
        "schema_version": "1.0",
        "stage": "S2.13",
        "test": "checkpoint_resume_equivalence",
        "training_seed": SEED,
        "reference_total_raw_iterations": TOTAL_ITERS,
        "interrupt_raw_iterations": INTERRUPT_ITERS,
        "resume_checkpoint": str(CHECKPOINT_PATH),
        "ssl_resume_checkpoint": str(SSL_CHECKPOINT_PATH),
        "checks": checks,
        "all_checks_pass": all_pass,
        "scope_note": (
            "Synthetic MMEngine checkpoint/resume fixtures only; "
            "not official training. Core trajectory equivalence is "
            "tested together with separate SSL Student/Teacher state "
            "preservation and no-resync checks. EMA update behavior is "
            "outside S2.13 and is not tested here."
        ),
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=False),
        encoding="utf-8",
    )

    print("\n=== CHECKS ===")
    for key, value in checks.items():
        print(f"{key} = {value}")

    print("\nreport =", REPORT_PATH)
    print(
        "S2_13_RESUME_EQUIVALENCE =",
        "PASS" if all_pass else "FAIL",
    )

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
