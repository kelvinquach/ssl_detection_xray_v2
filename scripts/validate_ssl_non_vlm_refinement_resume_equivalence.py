"""Validate exact resume of committed confidence history for SSL + REF.

SSL + REF means:
Semi-Supervised Learning with non-VLM pseudo-label refinement.

Scope:
- synthetic MMEngine fixture only;
- validates checkpoint save/restore of committed confidence history;
- compares uninterrupted and interrupted+resumed trajectories;
- does NOT validate AMP-skip commit/discard behavior;
- does NOT authorize project training.
"""

from __future__ import annotations

import argparse
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

import src.utils.resume_aware_loop  # noqa: F401,E402
import src.utils.resume_checkpoint_hook  # noqa: F401,E402

from src.utils.resume_checkpoint_hook import (  # noqa: E402
    RUNNER_COUNTER_ATTR,
    SSL_NON_VLM_REFINEMENT_HISTORY_STATE_KEY,
)
from src.utils.resume_state import (  # noqa: E402
    PROTOCOL_RESUME_STATE_KEY,
)
from src.utils.seed import get_full_rng_state  # noqa: E402
from src.utils.ssl_non_vlm_refinement_history import (  # noqa: E402
    ClassConfidenceHistory,
)
from src.utils.ssl_non_vlm_refinement_policy import (  # noqa: E402
    adaptive_class_threshold,
)


SEED = 204886845
NUMBER_OF_CLASSES = 14
TOTAL_ITERS = 300
INTERRUPT_ITERS = 260


@DATASETS.register_module(force=True)
class SSLNonVLMRefinementResumeDataset(torch.utils.data.Dataset):
    """Synthetic dataset consuming the main RNG families."""

    def __len__(self):
        return TOTAL_ITERS + 8

    def __getitem__(self, idx):
        py = random.random()
        nr = float(np.random.rand())
        tr = float(torch.rand(()))

        return {
            "inputs": torch.tensor(
                [float(idx) + py + nr + tr],
                dtype=torch.float32,
            ),
            "target": torch.tensor(
                [float(idx) * 0.05 + 0.25],
                dtype=torch.float32,
            ),
            "history_base": torch.tensor(
                [(py + nr + tr) / 3.0],
                dtype=torch.float32,
            ),
        }


@MODELS.register_module(force=True)
class SSLNonVLMRefinementResumeModel(BaseModel):
    """Toy model whose confidence history is not part of model.state_dict()."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.confidence_history = ClassConfidenceHistory(
            number_of_classes=NUMBER_OF_CLASSES,
            maximum_scores_per_class=256,
        )

    def forward(self, inputs, data_samples=None, mode="tensor"):
        return self.linear(inputs)

    def get_non_vlm_refinement_history_state(self) -> dict:
        return self.confidence_history.committed_state()

    def restore_non_vlm_refinement_history_state(
        self,
        state: dict,
    ) -> None:
        self.confidence_history.restore_committed_state(state)

    def train_step(self, data, optim_wrapper):
        device = self.linear.weight.device
        inputs = data["inputs"].float().to(device)
        target = data["target"].float().to(device)

        history_base = float(
            data["history_base"].reshape(-1)[0].item()
        )

        scores_by_class = {}
        for class_id in range(NUMBER_OF_CLASSES):
            shifted = (
                history_base
                + class_id * 0.037
            ) % 1.0

            score = 0.51 + 0.47 * shifted
            scores_by_class[class_id] = [score]

        self.confidence_history.stage_current_batch(
            scores_by_class
        )

        with optim_wrapper.optim_context(self):
            prediction = self.linear(inputs)
            loss = ((prediction - target) ** 2).mean()

        try:
            optim_wrapper.update_params(loss)
        except Exception:
            self.confidence_history.discard_staged_scores()
            raise

        self.confidence_history.commit_staged_scores()

        return {"loss": loss.detach()}


class SyntheticInterruption(RuntimeError):
    pass


class SaveAndInterruptHook(Hook):
    """Save checkpoint after a committed iteration, then interrupt."""

    priority = "VERY_LOW"

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path

    def after_train_iter(
        self,
        runner,
        batch_idx,
        data_batch=None,
        outputs=None,
    ):
        if int(runner.iter) != INTERRUPT_ITERS - 1:
            return

        self.checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        runner.save_checkpoint(
            out_dir=str(self.checkpoint_path.parent),
            filename=self.checkpoint_path.name,
            save_optimizer=True,
            save_param_scheduler=True,
            by_epoch=False,
        )

        raise SyntheticInterruption(
            f"Intentional interruption after "
            f"{INTERRUPT_ITERS} iterations."
        )


class RestoredHistoryProbeHook(Hook):
    """Observe history immediately after checkpoint restore."""

    priority = "NORMAL"

    def __init__(self, expected_state: dict):
        self.expected_state = expected_state
        self.observed = False
        self.matches_checkpoint = False

    def before_train(self, runner):
        if int(runner.iter) == 0:
            return

        self.observed = True

        current = (
            runner.model
            .get_non_vlm_refinement_history_state()
        )

        self.matches_checkpoint = nested_equal(
            current,
            self.expected_state,
        )


def nested_equal(a: Any, b: Any) -> bool:
    if torch.is_tensor(a) and torch.is_tensor(b):
        return torch.equal(
            a.detach().cpu(),
            b.detach().cpu(),
        )

    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.array_equal(a, b)

    if isinstance(a, dict) and isinstance(b, dict):
        return (
            set(a.keys()) == set(b.keys())
            and all(
                nested_equal(a[key], b[key])
                for key in a
            )
        )

    if isinstance(a, (list, tuple)) and isinstance(
        b,
        (list, tuple),
    ):
        return (
            len(a) == len(b)
            and all(
                nested_equal(x, y)
                for x, y in zip(a, b)
            )
        )

    return a == b


def tensor_state_cpu(model) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def threshold_snapshot(model) -> dict[str, float]:
    result = {}

    for class_id in range(NUMBER_OF_CLASSES):
        scores = (
            model.confidence_history
            .get_committed_scores(class_id)
        )

        result[str(class_id)] = adaptive_class_threshold(
            scores
        )

    return result


def snapshot(runner: Runner) -> dict[str, Any]:
    counter = getattr(
        runner,
        RUNNER_COUNTER_ATTR,
    )

    return {
        "model": tensor_state_cpu(runner.model),
        "optimizer": runner.optim_wrapper.state_dict(),
        "counter": counter.state_dict(),
        "iter": int(runner.train_loop.iter),
        "rng": get_full_rng_state(),
        "history": (
            runner.model
            .get_non_vlm_refinement_history_state()
        ),
        "thresholds": threshold_snapshot(
            runner.model
        ),
    }


def rng_probe() -> dict[str, Any]:
    result = {
        "python": [random.random() for _ in range(3)],
        "numpy": np.random.rand(3).tolist(),
        "torch_cpu": torch.rand(3).tolist(),
    }

    if torch.cuda.is_available():
        result["torch_cuda"] = (
            torch.rand(
                3,
                device="cuda",
            )
            .cpu()
            .tolist()
        )

    return result


def build_runner(
    *,
    work_dir: Path,
    resume: bool = False,
    load_from: str | None = None,
) -> Runner:
    cfg = Config(
        dict(
            model=dict(
                type="SSLNonVLMRefinementResumeModel"
            ),
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
                    type="default_collate"
                ),
                dataset=dict(
                    type="SSLNonVLMRefinementResumeDataset"
                ),
            ),
            train_cfg=dict(
                type="ResumeAwareIterBasedTrainLoop",
                max_iters=TOTAL_ITERS,
                val_interval=999999,
            ),
            optim_wrapper=dict(
                type="OptimWrapper",
                optimizer=dict(
                    type="SGD",
                    lr=0.01,
                    momentum=0.9,
                ),
                accumulative_counts=1,
            ),
            custom_hooks=[
                dict(
                    type="ProtocolResumeCheckpointHook"
                )
            ],
            default_hooks=dict(
                checkpoint=dict(
                    type="CheckpointHook",
                    interval=-1,
                )
            ),
            randomness=dict(
                seed=SEED,
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
                dist_cfg=dict(
                    backend="nccl"
                ),
            ),
            log_level="WARNING",
        )
    )

    return Runner.from_cfg(cfg)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work-root",
        default=(
            "/tmp/"
            "ssl_non_vlm_refinement_resume_validator"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "/tmp/"
            "ssl_non_vlm_refinement_resume_validator/"
            "report.json"
        ),
    )
    args = parser.parse_args()

    work_root = Path(args.work_root).resolve()
    output = Path(args.output).resolve()

    shutil.rmtree(
        work_root,
        ignore_errors=True,
    )
    work_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        work_root
        / "interrupted"
        / "latest_resume.pth"
    )

    # Uninterrupted reference.
    reference_runner = build_runner(
        work_dir=work_root / "reference"
    )
    reference_runner.train()

    reference_snapshot = snapshot(
        reference_runner
    )
    reference_probe = rng_probe()

    # Interrupted first segment.
    interrupted_runner = build_runner(
        work_dir=work_root / "interrupted"
    )

    interrupted_runner.register_hook(
        SaveAndInterruptHook(
            checkpoint_path
        ),
        priority="VERY_LOW",
    )

    interrupted = False

    try:
        interrupted_runner.train()
    except SyntheticInterruption:
        interrupted = True

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    protocol_state = checkpoint[
        PROTOCOL_RESUME_STATE_KEY
    ]

    history_state_present = (
        SSL_NON_VLM_REFINEMENT_HISTORY_STATE_KEY
        in protocol_state
    )

    checkpoint_history = protocol_state[
        SSL_NON_VLM_REFINEMENT_HISTORY_STATE_KEY
    ]

    checkpoint_scores_by_class = checkpoint_history.get(
        "scores_by_class",
        {},
    )

    # Resume same trajectory.
    resumed_runner = build_runner(
        work_dir=work_root / "interrupted",
        resume=True,
        load_from=str(checkpoint_path),
    )

    history_probe = RestoredHistoryProbeHook(
        checkpoint_history
    )

    resumed_runner.register_hook(
        history_probe,
        priority="NORMAL",
    )

    resumed_runner.train()

    resumed_snapshot = snapshot(
        resumed_runner
    )
    resumed_probe = rng_probe()

    checks = {
        "synthetic_interruption_observed": interrupted,
        "checkpoint_exists": checkpoint_path.is_file(),
        "checkpoint_has_protocol_state": (
            PROTOCOL_RESUME_STATE_KEY
            in checkpoint
        ),
        "checkpoint_has_committed_history": (
            history_state_present
        ),
        "checkpoint_history_has_no_staged_state": (
            "_staged_scores"
            not in checkpoint_history
            and "staged_scores"
            not in checkpoint_history
        ),
        "checkpoint_history_has_all_14_classes": (
            set(checkpoint_scores_by_class.keys())
            == {
                str(class_id)
                for class_id in range(NUMBER_OF_CLASSES)
            }
        ),
        "checkpoint_history_each_class_at_capacity": (
            all(
                len(
                    checkpoint_scores_by_class.get(
                        str(class_id),
                        [],
                    )
                )
                == 256
                for class_id in range(NUMBER_OF_CLASSES)
            )
        ),
        "restore_probe_observed": (
            history_probe.observed
        ),
        "history_restored_exact_before_resume_training": (
            history_probe.matches_checkpoint
        ),
        "final_iteration_match": (
            resumed_snapshot["iter"]
            == reference_snapshot["iter"]
            == TOTAL_ITERS
        ),
        "actual_optimizer_update_counter_match": (
            resumed_snapshot["counter"]
            == reference_snapshot["counter"]
        ),
        "model_state_match": nested_equal(
            resumed_snapshot["model"],
            reference_snapshot["model"],
        ),
        "optimizer_state_match": nested_equal(
            resumed_snapshot["optimizer"],
            reference_snapshot["optimizer"],
        ),
        "committed_history_match": nested_equal(
            resumed_snapshot["history"],
            reference_snapshot["history"],
        ),
        "reference_history_each_class_at_capacity": (
            all(
                len(
                    reference_snapshot["history"][
                        "scores_by_class"
                    ][str(class_id)]
                )
                == 256
                for class_id in range(NUMBER_OF_CLASSES)
            )
        ),
        "resumed_history_each_class_at_capacity": (
            all(
                len(
                    resumed_snapshot["history"][
                        "scores_by_class"
                    ][str(class_id)]
                )
                == 256
                for class_id in range(NUMBER_OF_CLASSES)
            )
        ),
        "adaptive_thresholds_match": nested_equal(
            resumed_snapshot["thresholds"],
            reference_snapshot["thresholds"],
        ),
        "rng_state_match_at_run_end": nested_equal(
            resumed_snapshot["rng"],
            reference_snapshot["rng"],
        ),
        "rng_next_draw_match": (
            resumed_probe
            == reference_probe
        ),
    }

    failed_checks = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    report = {
        "schema_version": "1.0",
        "test": (
            "ssl_non_vlm_refinement_"
            "committed_history_resume_equivalence"
        ),
        "training_seed": SEED,
        "number_of_classes": NUMBER_OF_CLASSES,
        "total_iterations": TOTAL_ITERS,
        "interrupt_iterations": INTERRUPT_ITERS,
        "checks": checks,
        "failed_checks": failed_checks,
        "all_checks_pass": not failed_checks,
        "scope_note": (
            "Synthetic MMEngine fixture only. "
            "Validates exact checkpoint save/restore and "
            "trajectory equivalence of committed confidence "
            "history and adaptive thresholds. "
            "AMP-skip commit/discard integration, detector "
            "integration and project training are not tested."
        ),
    }

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "check_count =",
        len(checks),
    )
    print(
        "failed_checks =",
        failed_checks,
    )
    print(
        "reference_counter =",
        reference_snapshot["counter"][
            "actual_optimizer_updates"
        ],
    )
    print(
        "resumed_counter =",
        resumed_snapshot["counter"][
            "actual_optimizer_updates"
        ],
    )
    print(
        "history_restore_probe =",
        history_probe.matches_checkpoint,
    )
    print(
        "report =",
        output,
    )
    print(
        "SSL_NON_VLM_REFINEMENT_HISTORY_RESUME =",
        "PASS" if not failed_checks else "FAIL",
    )

    return 0 if not failed_checks else 1


if __name__ == "__main__":
    raise SystemExit(main())
