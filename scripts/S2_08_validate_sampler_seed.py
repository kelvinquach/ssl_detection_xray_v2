#!/usr/bin/env python3
"""S2.08 - Validate sampler seeding controlled by training_seed."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import mmengine
import torch
import yaml
from mmengine.runner import Runner
from torch.utils.data import Dataset


class SamplerProbeDataset(Dataset):
    """Synthetic deterministic dataset for sampler-only preflight."""

    def __init__(self, size: int = 12) -> None:
        self.size = int(size)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        return int(index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/workspace/ssod/project")
    parser.add_argument("--seed-index-a", type=int, default=1)
    parser.add_argument("--seed-index-b", type=int, default=2)
    parser.add_argument(
        "--output",
        default="/workspace/ssod/artifacts/preflight/seed/seed_propagation_audit.json",
    )
    return parser.parse_args()


def build_loader(seed: int):
    cfg = {
        "dataset": SamplerProbeDataset(size=12),
        "sampler": {
            "type": "DefaultSampler",
            "shuffle": True,
            "round_up": False,
        },
        "batch_size": 1,
        "num_workers": 0,
    }

    return Runner.build_dataloader(
        cfg,
        seed=seed,
        diff_rank_seed=False,
    )


def observed_order(loader) -> list[int]:
    return [int(x) for x in iter(loader.sampler)]


def expected_order(size: int, effective_seed: int) -> list[int]:
    generator = torch.Generator()
    generator.manual_seed(int(effective_seed))
    return torch.randperm(size, generator=generator).tolist()


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

    ordered = protocol["training_seed"]["ordered_training_seeds"]
    seed_mapping = {
        int(record["index"]): int(record["training_seed"])
        for record in ordered
    }

    if args.seed_index_a not in seed_mapping:
        raise SystemExit("seed_index_a not present in locked seed mapping")
    if args.seed_index_b not in seed_mapping:
        raise SystemExit("seed_index_b not present in locked seed mapping")

    seed_a = seed_mapping[args.seed_index_a]
    seed_b = seed_mapping[args.seed_index_b]

    if seed_a == seed_b:
        raise SystemExit("S2.08 requires two distinct training seeds")

    loader_a1 = build_loader(seed_a)
    loader_a2 = build_loader(seed_a)
    loader_b1 = build_loader(seed_b)

    sampler_a1 = loader_a1.sampler
    sampler_a2 = loader_a2.sampler
    sampler_b1 = loader_b1.sampler

    epoch0_a1 = observed_order(loader_a1)
    epoch0_a2 = observed_order(loader_a2)
    epoch0_b1 = observed_order(loader_b1)

    expected_a_e0 = expected_order(12, seed_a)
    expected_b_e0 = expected_order(12, seed_b)

    sampler_a1.set_epoch(1)
    sampler_a2.set_epoch(1)
    sampler_b1.set_epoch(1)

    epoch1_a1 = observed_order(loader_a1)
    epoch1_a2 = observed_order(loader_a2)
    epoch1_b1 = observed_order(loader_b1)

    expected_a_e1 = expected_order(12, seed_a + 1)
    expected_b_e1 = expected_order(12, seed_b + 1)

    checks = {
        "locked_training_seed_count_is_10": len(seed_mapping) == 10,
        "selected_seed_a_in_locked_mapping": (
            seed_mapping.get(args.seed_index_a) == seed_a
        ),
        "selected_seed_b_in_locked_mapping": (
            seed_mapping.get(args.seed_index_b) == seed_b
        ),
        "sampler_type_is_DefaultSampler": (
            type(sampler_a1).__name__ == "DefaultSampler"
            and type(sampler_a2).__name__ == "DefaultSampler"
            and type(sampler_b1).__name__ == "DefaultSampler"
        ),
        "shuffle_enabled": (
            sampler_a1.shuffle is True
            and sampler_a2.shuffle is True
            and sampler_b1.shuffle is True
        ),
        "sampler_seed_a_matches_training_seed": (
            int(sampler_a1.seed) == seed_a
            and int(sampler_a2.seed) == seed_a
        ),
        "sampler_seed_b_matches_training_seed": (
            int(sampler_b1.seed) == seed_b
        ),
        "same_seed_same_order_epoch0": epoch0_a1 == epoch0_a2,
        "different_seed_different_order_epoch0": epoch0_a1 != epoch0_b1,
        "same_seed_same_order_epoch1": epoch1_a1 == epoch1_a2,
        "different_seed_different_order_epoch1": epoch1_a1 != epoch1_b1,
        "same_seed_epoch_changes_order": epoch0_a1 != epoch1_a1,
        "epoch0_seed_a_matches_expected_derivation": (
            epoch0_a1 == expected_a_e0
        ),
        "epoch0_seed_b_matches_expected_derivation": (
            epoch0_b1 == expected_b_e0
        ),
        "epoch1_seed_a_matches_expected_derivation": (
            epoch1_a1 == expected_a_e1
        ),
        "epoch1_seed_b_matches_expected_derivation": (
            epoch1_b1 == expected_b_e1
        ),
    }

    sampler_status = "PASS" if all(checks.values()) else "FAIL"

    output_path = Path(args.output).resolve()

    if not output_path.exists():
        raise SystemExit(
            "Existing S2.06 seed_propagation_audit.json is required"
        )

    report = json.loads(output_path.read_text(encoding="utf-8"))

    if report.get("artifact_type") != "SEED_PROPAGATION_AUDIT":
        raise SystemExit("Unexpected artifact_type at output path")

    if report.get("status") != "PASS":
        raise SystemExit("Existing S2.06 seed evidence is not PASS")

    if int(report.get("base_training_seed")) != seed_a:
        raise SystemExit(
            "Existing S2.06 base_training_seed does not match seed A"
        )

    report["schema_version"] = "1.2"
    report["evidence_stages"] = ["S2.06", "S2.08"]
    report["last_updated_at_utc"] = datetime.now(timezone.utc).isoformat()

    report["sampler_seed_observed"] = int(sampler_a1.seed)

    report["sampler_seed_strategy"] = {
        "framework": "MMEngine",
        "framework_version_observed": mmengine.__version__,
        "implementation": "mmengine.dataset.DefaultSampler",
        "construction_path": (
            "Runner.build_dataloader -> "
            "DATA_SAMPLERS.build(DefaultSampler)"
        ),
        "runner_seed_argument": "training_seed",
        "diff_rank_seed": False,
        "shuffle": True,
        "derivation": (
            "DefaultSampler.seed = training_seed; "
            "shuffle generator seed = sampler.seed + epoch"
        ),
    }

    report["sampler_seed_examples"] = [
        {
            "seed_index": args.seed_index_a,
            "training_seed": seed_a,
            "observed_sampler_seed": int(sampler_a1.seed),
            "epoch0_effective_seed": seed_a,
            "epoch1_effective_seed": seed_a + 1,
        },
        {
            "seed_index": args.seed_index_b,
            "training_seed": seed_b,
            "observed_sampler_seed": int(sampler_b1.seed),
            "epoch0_effective_seed": seed_b,
            "epoch1_effective_seed": seed_b + 1,
        },
    ]

    report["sampler_seed_audit"] = {
        "stage": "S2.08",
        "status": sampler_status,
        "training_authorized": False,
        "purpose": "controlled_preflight_only",
        "probe_configuration": {
            "dataset_size": 12,
            "batch_size": 1,
            "num_workers": 0,
            "sampler": "DefaultSampler",
            "shuffle": True,
            "round_up": False,
            "diff_rank_seed": False,
            "note": (
                "num_workers=0 isolates sampler behavior; "
                "worker seeding is separately covered by S2.07"
            ),
        },
        "epoch0": {
            "seed_a_run1_order": epoch0_a1,
            "seed_a_run2_order": epoch0_a2,
            "seed_b_order": epoch0_b1,
            "seed_a_expected_order": expected_a_e0,
            "seed_b_expected_order": expected_b_e0,
        },
        "epoch1": {
            "seed_a_run1_order": epoch1_a1,
            "seed_a_run2_order": epoch1_a2,
            "seed_b_order": epoch1_b1,
            "seed_a_expected_order": expected_a_e1,
            "seed_b_expected_order": expected_b_e1,
        },
        "checks": checks,
    }

    report["checks"]["sampler_seed_controlled_by_training_seed"] = (
        sampler_status == "PASS"
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
    print(f"framework_version = {mmengine.__version__}")
    print(f"seed_a = {seed_a}")
    print(f"seed_b = {seed_b}")
    print(f"sampler_a_seed = {sampler_a1.seed}")
    print(f"sampler_b_seed = {sampler_b1.seed}")
    print(f"same_seed_same_order_epoch0 = {epoch0_a1 == epoch0_a2}")
    print(f"different_seed_different_order_epoch0 = {epoch0_a1 != epoch0_b1}")
    print(f"same_seed_same_order_epoch1 = {epoch1_a1 == epoch1_a2}")
    print(f"different_seed_different_order_epoch1 = {epoch1_a1 != epoch1_b1}")
    print(f"same_seed_epoch_changes_order = {epoch0_a1 != epoch1_a1}")
    print(f"epoch0_seed_a_matches_expected = {epoch0_a1 == expected_a_e0}")
    print(f"epoch0_seed_b_matches_expected = {epoch0_b1 == expected_b_e0}")
    print(f"epoch1_seed_a_matches_expected = {epoch1_a1 == expected_a_e1}")
    print(f"epoch1_seed_b_matches_expected = {epoch1_b1 == expected_b_e1}")
    print(f"S2_08_SAMPLER_SEED={sampler_status}")

    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
