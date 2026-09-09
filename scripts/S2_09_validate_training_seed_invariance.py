#!/usr/bin/env python3
"""S2.09 - Prove training_seed does not change fixed data membership."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


DATASETS = {
    "train": "instances_train.json",
    "val": "instances_val.json",
    "test": "instances_test.json",
    "L_1pct": "instances_labeled_1pct.json",
    "L_5pct": "instances_labeled_5pct.json",
    "L_10pct": "instances_labeled_10pct.json",
    "L_20pct": "instances_labeled_20pct.json",
    "U_1pct": "instances_unlabeled_1pct.json",
    "U_5pct": "instances_unlabeled_5pct.json",
    "U_10pct": "instances_unlabeled_10pct.json",
    "U_20pct": "instances_unlabeled_20pct.json",
}

EXPECTED_COUNTS = {
    "train": 3426,
    "val": 734,
    "test": 734,
    "L_1pct": 34,
    "L_5pct": 171,
    "L_10pct": 343,
    "L_20pct": 685,
    "U_1pct": 3392,
    "U_5pct": 3255,
    "U_10pct": 3083,
    "U_20pct": 2741,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default="/workspace/ssod/project",
    )
    parser.add_argument(
        "--data-root",
        default="/workspace/ssod/data/coco",
    )
    parser.add_argument(
        "--output",
        default=(
            "/workspace/ssod/artifacts/preflight/seed/"
            "training_seed_invariance_audit.json"
        ),
    )
    return parser.parse_args()


def membership_sha256(image_ids: list[int]) -> str:
    payload = "\n".join(
        str(value)
        for value in sorted(image_ids, key=lambda value: int(value))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_membership(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))

    # S2.09 intentionally reads image membership only.
    # It does not inspect hidden annotations for unlabeled subsets.
    image_ids = [image["id"] for image in data["images"]]

    return {
        "count": len(image_ids),
        "unique_count": len(set(image_ids)),
        "membership_sha256": membership_sha256(image_ids),
    }


def snapshot(data_root: Path) -> dict:
    result = {}

    for logical_name, filename in DATASETS.items():
        path = data_root / filename

        if not path.is_file():
            raise SystemExit(f"Missing canonical membership file: {path}")

        result[logical_name] = read_membership(path)

    return result


def combined_membership_sha256(snapshot_data: dict) -> str:
    payload = "\n".join(
        f"{name}:{snapshot_data[name]['membership_sha256']}"
        for name in sorted(snapshot_data)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> int:
    args = parse_args()

    repo_root = Path(args.repo_root).resolve()
    data_root = Path(args.data_root).resolve()
    output_path = Path(args.output).resolve()

    sys.path.insert(0, str(repo_root))

    from src.utils.seed import set_global_seed

    protocol_path = (
        repo_root
        / "configs"
        / "protocol"
        / "phase2F1_seed_protocol.yaml"
    )

    with protocol_path.open("r", encoding="utf-8") as fh:
        protocol = yaml.safe_load(fh)

    ordered_seed_records = protocol["training_seed"]["ordered_training_seeds"]

    training_seeds = [
        {
            "seed_index": int(record["index"]),
            "training_seed": int(record["training_seed"]),
        }
        for record in ordered_seed_records
    ]

    baseline = snapshot(data_root)
    baseline_combined_hash = combined_membership_sha256(baseline)

    baseline_checks = {
        "locked_training_seed_count_is_10": len(training_seeds) == 10,
        "all_membership_files_present": len(baseline) == 11,
        "all_counts_match_locked_values": all(
            baseline[name]["count"] == EXPECTED_COUNTS[name]
            for name in EXPECTED_COUNTS
        ),
        "all_memberships_have_unique_image_ids": all(
            baseline[name]["count"] == baseline[name]["unique_count"]
            for name in baseline
        ),
    }

    per_seed_results = []

    for record in training_seeds:
        seed_index = record["seed_index"]
        training_seed = record["training_seed"]

        set_global_seed(training_seed, deterministic=True)

        observed = snapshot(data_root)
        observed_combined_hash = combined_membership_sha256(observed)

        per_dataset_match = {
            name: (
                observed[name]["count"] == baseline[name]["count"]
                and observed[name]["unique_count"]
                == baseline[name]["unique_count"]
                and observed[name]["membership_sha256"]
                == baseline[name]["membership_sha256"]
            )
            for name in baseline
        }

        seed_pass = (
            all(per_dataset_match.values())
            and observed_combined_hash == baseline_combined_hash
        )

        per_seed_results.append(
            {
                "seed_index": seed_index,
                "training_seed": training_seed,
                "combined_membership_sha256": observed_combined_hash,
                "combined_hash_matches_baseline": (
                    observed_combined_hash == baseline_combined_hash
                ),
                "per_dataset_hash_match": per_dataset_match,
                "status": "PASS" if seed_pass else "FAIL",
            }
        )

    all_seed_runs_pass = all(
        record["status"] == "PASS"
        for record in per_seed_results
    )

    checks = {
        **baseline_checks,
        "all_10_training_seeds_preserve_train_val_test_membership": all(
            all(
                record["per_dataset_hash_match"][name]
                for name in ("train", "val", "test")
            )
            for record in per_seed_results
        ),
        "all_10_training_seeds_preserve_labeled_membership": all(
            all(
                record["per_dataset_hash_match"][name]
                for name in (
                    "L_1pct",
                    "L_5pct",
                    "L_10pct",
                    "L_20pct",
                )
            )
            for record in per_seed_results
        ),
        "all_10_training_seeds_preserve_unlabeled_membership": all(
            all(
                record["per_dataset_hash_match"][name]
                for name in (
                    "U_1pct",
                    "U_5pct",
                    "U_10pct",
                    "U_20pct",
                )
            )
            for record in per_seed_results
        ),
        "all_combined_membership_hashes_match_baseline": all(
            record["combined_hash_matches_baseline"]
            for record in per_seed_results
        ),
        "all_seed_runs_pass": all_seed_runs_pass,
    }

    status = "PASS" if all(checks.values()) else "FAIL"

    report = {
        "schema_version": "1.0",
        "artifact_type": "TRAINING_SEED_INVARIANCE_AUDIT",
        "stage": "S2.09",
        "status": status,
        "training_authorized": False,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "partition_seed": int(
            protocol["partition"]["partition_seed"]
        ),
        "partition_seed_policy": (
            protocol["partition"]["partition_seed_policy"]
        ),
        "training_seed_deterministic_policy": (
            protocol["training_seed"]["deterministic_policy"]
        ),
        "training_seed_ordering": (
            protocol["training_seed"]["ordering"]
        ),
        "training_seed_count": len(training_seeds),
        "training_seeds": training_seeds,
        "data_root": str(data_root),
        "membership_definition": (
            "SHA-256 of sorted images[].id values only; "
            "annotations are not read for membership hashing"
        ),
        "baseline": {
            "combined_membership_sha256": baseline_combined_hash,
            "datasets": baseline,
        },
        "per_training_seed": per_seed_results,
        "checks": checks,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"OUTPUT = {output_path}")
    print(f"partition_seed = {report['partition_seed']}")
    print(f"training_seed_count = {len(training_seeds)}")
    print(f"baseline_combined_hash = {baseline_combined_hash}")

    for record in per_seed_results:
        print(
            f"seed_index={record['seed_index']} "
            f"training_seed={record['training_seed']} "
            f"combined_hash_match="
            f"{record['combined_hash_matches_baseline']} "
            f"status={record['status']}"
        )

    print(
        "train_val_test_invariance =",
        checks[
            "all_10_training_seeds_preserve_train_val_test_membership"
        ],
    )
    print(
        "labeled_invariance =",
        checks[
            "all_10_training_seeds_preserve_labeled_membership"
        ],
    )
    print(
        "unlabeled_invariance =",
        checks[
            "all_10_training_seeds_preserve_unlabeled_membership"
        ],
    )
    print(f"S2_09_TRAINING_SEED_INVARIANCE={status}")

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
