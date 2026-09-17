"""S6.01 validator for locked seed-level mean/sample-SD semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


DEFAULT_ROOT = Path("/workspace/ssod/project")
DEFAULT_OUTPUT = (
    DEFAULT_ROOT
    / "artifacts"
    / "preflight"
    / "statistics"
    / "statistical_fixture_report.json"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.repo_root).resolve()
    output = Path(args.output).resolve()
    sys.path.insert(0, str(root))

    from src.statistics import seed_summary as S

    print("===== S6.01 STATISTICAL FIXTURE VALIDATOR =====")
    checks = []

    def check(name, passed, detail=None):
        passed = bool(passed)
        checks.append({"name": name, "pass": passed, "detail": detail})
        print(f"{name}={passed}")

    seed_manifest_path = root / "data" / "manifests" / "seed_manifest.json"
    seed_manifest = json.loads(
        seed_manifest_path.read_text(encoding="utf-8-sig")
    )
    canonical_seeds = tuple(
        seed_manifest["training_seed"]["ordered_training_seed_values"]
    )

    check(
        "contract::replication_unit",
        S.REPLICATION_UNIT == "TRAINING_SEED_TRAINED_RUN",
        S.REPLICATION_UNIT,
    )
    check(
        "contract::official_seed_count",
        S.OFFICIAL_SEED_COUNT == 10,
        S.OFFICIAL_SEED_COUNT,
    )
    check(
        "contract::ordered_seed_identity",
        S.OFFICIAL_TRAINING_SEEDS == canonical_seeds,
        list(S.OFFICIAL_TRAINING_SEEDS),
    )
    check(
        "contract::sample_sd_ddof",
        S.SAMPLE_SD_DDOF == 1
        and seed_manifest["aggregation_policy"]["standard_deviation_ddof"] == 1,
        S.SAMPLE_SD_DDOF,
    )
    check(
        "contract::cross_seed_prediction_pooling_prohibited",
        S.CROSS_SEED_PREDICTION_POOLING_ALLOWED is False,
        S.CROSS_SEED_PREDICTION_POOLING_ALLOWED,
    )

    fixture = {seed: float(i) for i, seed in enumerate(canonical_seeds, 1)}
    summary = S.summarize_seed_runs(fixture)
    expected_sd = math.sqrt(82.5 / 9.0)

    check("positive::n_10", summary["n"] == 10, summary["n"])
    check(
        "positive::mean",
        abs(summary["mean"] - 5.5) < 1e-12,
        summary["mean"],
    )
    check(
        "positive::sample_sd_ddof1",
        abs(summary["sample_sd"] - expected_sd) < 1e-12,
        summary["sample_sd"],
    )
    check(
        "positive::sample_sd_denominator_n_minus_1",
        summary["sample_sd_denominator"] == 9,
        summary["sample_sd_denominator"],
    )
    check(
        "positive::seed_order_preserved",
        tuple(summary["training_seeds"]) == canonical_seeds,
        summary["training_seeds"],
    )

    negative_cases = {}

    def expect_block(name, payload):
        blocked = False
        error = None
        try:
            S.summarize_seed_runs(payload)
        except S.SeedSummaryError as exc:
            blocked = True
            error = str(exc)
        negative_cases[name] = {"blocked": blocked, "error": error}
        check(f"fail_closed::{name}", blocked, error)

    expect_block(
        "missing_seed",
        dict(list(fixture.items())[:-1]),
    )

    reversed_fixture = {
        seed: fixture[seed] for seed in reversed(canonical_seeds)
    }
    expect_block("reordered_seeds", reversed_fixture)

    nonfinite_fixture = dict(fixture)
    nonfinite_fixture[canonical_seeds[0]] = float("nan")
    expect_block("nonfinite_metric", nonfinite_fixture)

    pooled_fixture = {seed: [fixture[seed], fixture[seed]] for seed in canonical_seeds}
    expect_block("non_scalar_per_seed", pooled_fixture)

    failed = [item for item in checks if not item["pass"]]
    print(f"CHECKS_PASSED={len(checks) - len(failed)}")
    print(f"CHECKS_TOTAL={len(checks)}")
    print(f"FAILED_COUNT={len(failed)}")
    print(
        "FAILED_CHECKS="
        + ("NONE" if not failed else ",".join(item["name"] for item in failed))
    )

    if failed:
        raise SystemExit(1)

    implementation_path = root / "src" / "statistics" / "seed_summary.py"
    validator_path = root / "scripts" / "S6_01_validate_seed_summary.py"

    report = {
        "schema_version": "1.0",
        "evidence_type": "statistical_fixture_report",
        "task": "S6.01",
        "status": "PASS",
        "tracker_action": "Implement mean/SD ddof=1",
        "inferential_replication_unit": S.REPLICATION_UNIT,
        "official_seed_count": S.OFFICIAL_SEED_COUNT,
        "ordered_training_seeds": list(canonical_seeds),
        "aggregation": {
            "mean_across_training_seeds": True,
            "sample_standard_deviation": True,
            "ddof": S.SAMPLE_SD_DDOF,
            "denominator": "n-1",
            "cross_seed_prediction_pooling_allowed": False,
        },
        "positive_fixture": {
            "input_values": list(fixture.values()),
            "mean": summary["mean"],
            "sample_sd": summary["sample_sd"],
            "sample_sd_denominator": summary["sample_sd_denominator"],
        },
        "negative_fixtures": negative_cases,
        "source_sha256": {
            "src/statistics/seed_summary.py": sha256(implementation_path),
            "scripts/S6_01_validate_seed_summary.py": sha256(validator_path),
            "data/manifests/seed_manifest.json": sha256(seed_manifest_path),
        },
        "scientific_boundaries": {
            "image_is_replication": False,
            "bbox_is_replication": False,
            "detection_is_replication": False,
            "training_seed_trained_run_is_replication": True,
            "test_used": False,
            "hidden_unlabeled_gt_used": False,
            "official_training_authorized": False,
            "final_test_authorized": False,
        },
        "checks": checks,
        "checks_passed": len(checks),
        "checks_total": len(checks),
        "failed_count": 0,
        "failed_checks": [],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"EVIDENCE={output}")
    print(f"EVIDENCE_SHA256={sha256(output)}")
    print("STATUS=PASS")
    print("S6_01_STATISTICAL_FIXTURE=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
