#!/usr/bin/env python3
"""S2.07 - Audit MMEngine DataLoader worker seeding."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, get_worker_info

from mmengine.runner import Runner


AUDIT_NUM_WORKERS = 2


class WorkerSeedProbeDataset(Dataset):
    """Small synthetic dataset used only to observe worker RNG state."""

    def __len__(self) -> int:
        return AUDIT_NUM_WORKERS

    def __getitem__(self, index: int) -> dict:
        info = get_worker_info()

        return {
            "index": int(index),
            "worker_id": None if info is None else int(info.id),
            "torch_initial_seed": int(torch.initial_seed()),
            "python_draw": float(random.random()),
            "numpy_draw": float(np.random.random()),
            "torch_draw": float(torch.rand(1).item()),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default="/workspace/ssod/project",
    )
    parser.add_argument(
        "--seed-index",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--output",
        default=(
            "/workspace/ssod/artifacts/preflight/seed/"
            "dataloader_worker_seed_audit.json"
        ),
    )
    return parser.parse_args()


def expected_draws(worker_seed: int) -> dict:
    py_rng = random.Random(worker_seed)
    np_rng = np.random.RandomState(worker_seed)

    torch_gen = torch.Generator(device="cpu")
    torch_gen.manual_seed(worker_seed)

    return {
        "python_draw": float(py_rng.random()),
        "numpy_draw": float(np_rng.random_sample()),
        "torch_draw": float(
            torch.rand(1, generator=torch_gen).item()
        ),
    }


def scalar(value):
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return scalar(value[0])
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    return value


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()

    protocol_path = (
        repo_root
        / "configs"
        / "protocol"
        / "phase2F1_seed_protocol.yaml"
    )

    with protocol_path.open("r", encoding="utf-8") as fh:
        protocol = yaml.safe_load(fh)

    seed_cfg = protocol["training_seed"]
    seed_mapping = {
        int(record["index"]): int(record["training_seed"])
        for record in seed_cfg["ordered_training_seeds"]
    }

    if args.seed_index not in seed_mapping:
        raise SystemExit(
            f"seed_index {args.seed_index} not present in locked seed list"
        )

    training_seed = seed_mapping[args.seed_index]

    dataset = WorkerSeedProbeDataset()

    # sampler=None deliberately keeps sampler-seeding outside S2.07.
    dataloader_cfg = {
        "dataset": dataset,
        "sampler": None,
        "batch_size": 1,
        "num_workers": AUDIT_NUM_WORKERS,
        "persistent_workers": False,
    }

    loader = Runner.build_dataloader(
        dataloader_cfg,
        seed=training_seed,
        diff_rank_seed=False,
    )

    observations = []

    for batch in loader:
        obs = {
            "index": int(scalar(batch["index"])),
            "worker_id": int(scalar(batch["worker_id"])),
            "torch_initial_seed": int(
                scalar(batch["torch_initial_seed"])
            ),
            "python_draw": float(
                scalar(batch["python_draw"])
            ),
            "numpy_draw": float(
                scalar(batch["numpy_draw"])
            ),
            "torch_draw": float(
                scalar(batch["torch_draw"])
            ),
        }
        observations.append(obs)

    observations.sort(key=lambda item: item["worker_id"])

    rank = 0

    examples = []
    all_worker_checks = []

    for obs in observations:
        worker_id = obs["worker_id"]

        expected_worker_seed = (
            AUDIT_NUM_WORKERS * rank
            + worker_id
            + training_seed
        )

        expected = expected_draws(expected_worker_seed)

        checks = {
            "torch_initial_seed_matches": (
                obs["torch_initial_seed"]
                == expected_worker_seed
            ),
            "python_rng_matches": (
                obs["python_draw"]
                == expected["python_draw"]
            ),
            "numpy_rng_matches": (
                obs["numpy_draw"]
                == expected["numpy_draw"]
            ),
            "torch_rng_matches": (
                obs["torch_draw"]
                == expected["torch_draw"]
            ),
        }

        all_worker_checks.append(all(checks.values()))

        examples.append(
            {
                "worker_id": worker_id,
                "rank": rank,
                "expected_worker_seed": expected_worker_seed,
                "observed_torch_initial_seed": (
                    obs["torch_initial_seed"]
                ),
                "observed_draws": {
                    "python": obs["python_draw"],
                    "numpy": obs["numpy_draw"],
                    "torch": obs["torch_draw"],
                },
                "expected_draws": {
                    "python": expected["python_draw"],
                    "numpy": expected["numpy_draw"],
                    "torch": expected["torch_draw"],
                },
                "checks": checks,
            }
        )

    observed_worker_ids = [
        item["worker_id"]
        for item in observations
    ]

    checks = {
        "locked_training_seed_count_is_10": (
            len(seed_mapping) == 10
        ),
        "selected_seed_in_locked_mapping": (
            seed_mapping.get(args.seed_index)
            == training_seed
        ),
        "audit_num_workers_is_multiworker": (
            AUDIT_NUM_WORKERS >= 2
        ),
        "observed_expected_worker_count": (
            len(observations) == AUDIT_NUM_WORKERS
        ),
        "observed_worker_ids_match": (
            observed_worker_ids
            == list(range(AUDIT_NUM_WORKERS))
        ),
        "all_worker_seed_checks_pass": (
            all(all_worker_checks)
            and len(all_worker_checks)
            == AUDIT_NUM_WORKERS
        ),
    }

    status = "PASS" if all(checks.values()) else "FAIL"

    report = {
        "schema_version": "1.1",
        "artifact_type": "DATALOADER_WORKER_SEED_AUDIT",
        "stage": "S2.07",
        "status": status,
        "generated_at_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "training_authorized": False,
        "seed_source": str(protocol_path),
        "seed_index": args.seed_index,
        "base_training_seed": training_seed,
        "worker_seed_strategy": {
            "framework": "MMEngine",
            "framework_version_observed": "0.10.7",
            "implementation": (
                "mmengine.dataset.utils.worker_init_fn"
            ),
            "derivation": (
                "worker_seed = num_workers * rank "
                "+ worker_id + base_training_seed"
            ),
            "rngs_seeded_per_worker": [
                "Python random",
                "NumPy RNG",
                "PyTorch CPU RNG",
            ],
        },
        "audit_configuration": {
            "purpose": "controlled_preflight_only",
            "official_training_num_workers": (
                "NOT_SPECIFIED_BY_CANONICAL_SOURCES"
            ),
            "audit_num_workers": AUDIT_NUM_WORKERS,
            "batch_size": 1,
            "rank": rank,
            "diff_rank_seed": False,
            "sampler": (
                "None; sampler seeding intentionally "
                "outside S2.07 scope"
            ),
        },
        "worker_seed_examples": examples,
        "checks": checks,
    }

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"OUTPUT = {output_path}")
    print(f"seed_index = {args.seed_index}")
    print(f"base_training_seed = {training_seed}")
    print(f"audit_num_workers = {AUDIT_NUM_WORKERS}")
    print(f"observed_worker_ids = {observed_worker_ids}")

    for item in examples:
        print(
            "worker",
            item["worker_id"],
            "expected_seed =",
            item["expected_worker_seed"],
            "observed_seed =",
            item["observed_torch_initial_seed"],
            "checks =",
            item["checks"],
        )

    print(f"S2_07_DATALOADER_WORKER_SEED={status}")

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
