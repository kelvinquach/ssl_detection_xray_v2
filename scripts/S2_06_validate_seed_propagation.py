#!/usr/bin/env python3
"""S2.06 - Validate training-seed propagation to core RNG streams."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.seed import set_global_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/workspace/ssod/project")
    parser.add_argument("--seed-index", type=int, default=1)
    parser.add_argument(
        "--output",
        default="/workspace/ssod/artifacts/preflight/seed/seed_propagation_audit.json",
    )
    return parser.parse_args()


def capture_draws() -> dict:
    result = {
        "python": __import__("random").random(),
        "numpy": np.random.random(4).tolist(),
        "torch_cpu": torch.rand(4).tolist(),
        "torch_cuda": [],
    }

    if torch.cuda.is_available():
        for device_index in range(torch.cuda.device_count()):
            values = torch.rand(
                4,
                device=f"cuda:{device_index}",
            ).cpu().tolist()
            result["torch_cuda"].append(
                {
                    "device_index": device_index,
                    "values": values,
                }
            )

    return result


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
    ordered = seed_cfg["ordered_training_seeds"]

    seed_mapping = {
        int(record["index"]): int(record["training_seed"])
        for record in ordered
    }

    if args.seed_index not in seed_mapping:
        raise SystemExit(
            f"seed_index {args.seed_index} not present in locked seed list"
        )

    training_seed = seed_mapping[args.seed_index]

    # First application of the locked training seed.
    applied_first = set_global_seed(
        training_seed,
        deterministic=True,
    )

    python_seed_observed = (
        training_seed
        if applied_first["applied"].get("python_random") is True
        else None
    )

    numpy_state = np.random.get_state()
    numpy_seed_observed = int(numpy_state[1][0])

    torch_initial_seed_observed = int(torch.initial_seed())

    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count())

    cuda_initial_seeds_observed = []
    if cuda_available:
        for device_index in range(cuda_device_count):
            with torch.cuda.device(device_index):
                cuda_initial_seeds_observed.append(
                    {
                        "device_index": device_index,
                        "initial_seed": int(torch.cuda.initial_seed()),
                    }
                )

    first_draws = capture_draws()

    # Re-apply the exact same seed and verify stream replay.
    applied_second = set_global_seed(
        training_seed,
        deterministic=True,
    )
    second_draws = capture_draws()

    python_replay_match = (
        first_draws["python"] == second_draws["python"]
    )
    numpy_replay_match = (
        first_draws["numpy"] == second_draws["numpy"]
    )
    torch_cpu_replay_match = (
        first_draws["torch_cpu"] == second_draws["torch_cpu"]
    )
    torch_cuda_replay_match = (
        first_draws["torch_cuda"] == second_draws["torch_cuda"]
    )

    cuda_initial_seed_match = (
        cuda_available
        and cuda_device_count > 0
        and all(
            item["initial_seed"] == training_seed
            for item in cuda_initial_seeds_observed
        )
    )

    checks = {
        "locked_training_seed_count_is_10": len(seed_mapping) == 10,
        "selected_seed_in_locked_mapping": (
            seed_mapping.get(args.seed_index) == training_seed
        ),
        "python_rng_applied": (
            applied_first["applied"].get("python_random") is True
        ),
        "numpy_rng_applied": (
            applied_first["applied"].get("numpy") is True
        ),
        "torch_cpu_rng_applied": (
            applied_first["applied"].get("torch_cpu") is True
        ),
        "torch_cuda_rng_applied": (
            applied_first["applied"].get("torch_cuda") is True
        ),
        "numpy_seed_observed_matches": (
            numpy_seed_observed == training_seed
        ),
        "torch_initial_seed_observed_matches": (
            torch_initial_seed_observed == training_seed
        ),
        "cuda_available": cuda_available,
        "cuda_initial_seeds_match": cuda_initial_seed_match,
        "python_same_seed_replay_match": python_replay_match,
        "numpy_same_seed_replay_match": numpy_replay_match,
        "torch_cpu_same_seed_replay_match": torch_cpu_replay_match,
        "torch_cuda_same_seed_replay_match": torch_cuda_replay_match,
    }

    status = "PASS" if all(checks.values()) else "FAIL"

    report = {
        "schema_version": "1.1",
        "artifact_type": "SEED_PROPAGATION_AUDIT",
        "stage": "S2.06",
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_authorized": False,
        "seed_source": str(protocol_path),
        "seed_index": args.seed_index,
        "base_training_seed": training_seed,
        "deterministic_policy": seed_cfg.get("deterministic_policy"),
        "python_seed_observed": python_seed_observed,
        "python_seed_observation_method": (
            "set_global_seed application record plus same-seed replay"
        ),
        "numpy_seed_observed": numpy_seed_observed,
        "torch_initial_seed_observed": torch_initial_seed_observed,
        "cuda_seed_policy": (
            "set_global_seed applies torch.manual_seed(seed), "
            "torch.cuda.manual_seed(seed), and "
            "torch.cuda.manual_seed_all(seed) when CUDA is available"
        ),
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_initial_seeds_observed": cuda_initial_seeds_observed,
        "same_seed_replay": {
            "python": python_replay_match,
            "numpy": numpy_replay_match,
            "torch_cpu": torch_cpu_replay_match,
            "torch_cuda": torch_cuda_replay_match,
        },
        "applied_first": applied_first,
        "applied_second": applied_second,
        "checks": checks,
    }

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # S2.06 and S2.08 intentionally share seed_propagation_audit.json.
    # If S2.08 sampler evidence already exists, rerunning S2.06 must
    # preserve it rather than silently deleting downstream evidence.
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))

        if existing.get("artifact_type") != "SEED_PROPAGATION_AUDIT":
            raise SystemExit(
                "Refusing to overwrite unexpected artifact_type at output path"
            )

        for key in (
            "sampler_seed_observed",
            "sampler_seed_strategy",
            "sampler_seed_examples",
            "sampler_seed_audit",
            "evidence_stages",
            "last_updated_at_utc",
        ):
            if key in existing:
                report[key] = existing[key]

        sampler_audit = report.get("sampler_seed_audit")
        if sampler_audit is not None:
            # S2.08 extends this shared artifact to schema v1.2.
            # Rerunning S2.06 must not downgrade the cumulative schema.
            report["schema_version"] = "1.2"

            sampler_pass = sampler_audit.get("status") == "PASS"
            report["checks"]["sampler_seed_controlled_by_training_seed"] = (
                sampler_pass
            )
            report["status"] = (
                "PASS"
                if all(report["checks"].values())
                else "FAIL"
            )

    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"OUTPUT = {output_path}")
    print(f"seed_index = {args.seed_index}")
    print(f"base_training_seed = {training_seed}")
    print(f"python_seed_observed = {python_seed_observed}")
    print(f"numpy_seed_observed = {numpy_seed_observed}")
    print(
        "torch_initial_seed_observed =",
        torch_initial_seed_observed,
    )
    print(f"cuda_available = {cuda_available}")
    print(f"cuda_device_count = {cuda_device_count}")
    print(
        "cuda_initial_seeds_observed =",
        cuda_initial_seeds_observed,
    )
    print(f"python_replay_match = {python_replay_match}")
    print(f"numpy_replay_match = {numpy_replay_match}")
    print(f"torch_cpu_replay_match = {torch_cpu_replay_match}")
    print(f"torch_cuda_replay_match = {torch_cuda_replay_match}")
    print(f"S2_06_SEED_PROPAGATION={status}")

    if status != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
