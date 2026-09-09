#!/usr/bin/env python3
"""S2.14 preflight validator for fail-fast source/config/data/seed guardrails."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default="/workspace/ssod/project",
    )
    parser.add_argument(
        "--output",
        default=(
            "/workspace/ssod/artifacts/preflight/guardrails/"
            "s2_14_fail_fast_guardrail_test.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_path = Path(args.output).resolve()

    sys.path.insert(0, str(repo_root))

    from src.utils.pre_run_guardrails import (
        PreRunGuardrailError,
        assert_pre_run_identity,
        build_expected_pre_run_identity,
        evaluate_pre_run_identity,
        locked_seed_identity,
        verify_locked_source_integrity,
    )

    source_checks = verify_locked_source_integrity(repo_root)
    seed_identity = locked_seed_identity(repo_root)

    expected = build_expected_pre_run_identity(
        repo_root,
        method="SUP",
        architecture="R50-FPN",
        budget="10pct",
        seed_index=1,
    )

    positive_report = evaluate_pre_run_identity(
        repo_root,
        deepcopy(expected),
    )

    cases = {}

    def run_block_case(name, field, bad_value):
        observed = deepcopy(expected)
        observed[field] = bad_value

        report = evaluate_pre_run_identity(
            repo_root,
            observed,
        )

        blocked_by_assert = False
        try:
            assert_pre_run_identity(
                repo_root,
                observed,
            )
        except PreRunGuardrailError:
            blocked_by_assert = True

        mismatch_fields = [
            item["field"]
            for item in report["mismatches"]
        ]

        cases[name] = {
            "launch_status": report["launch_status"],
            "all_checks_pass": report["all_checks_pass"],
            "mismatch_fields": mismatch_fields,
            "blocked_by_assert": blocked_by_assert,
            "expected_field_detected": field in mismatch_fields,
        }

    run_block_case(
        "scientific_source_mismatch",
        "scientific_source_sha256",
        "0" * 64,
    )

    run_block_case(
        "config_mismatch",
        "optimizer_recipe_sha256",
        "1" * 64,
    )

    run_block_case(
        "data_mismatch",
        "train_split_sha256",
        "2" * 64,
    )

    run_block_case(
        "seed_mismatch",
        "training_seed",
        expected["training_seed"] + 1,
    )

    missing_observed = deepcopy(expected)
    del missing_observed["dataset_manifest_sha256"]

    missing_report = evaluate_pre_run_identity(
        repo_root,
        missing_observed,
    )

    missing_blocked = False
    try:
        assert_pre_run_identity(
            repo_root,
            missing_observed,
        )
    except PreRunGuardrailError:
        missing_blocked = True

    cases["missing_required_identity_field"] = {
        "launch_status": missing_report["launch_status"],
        "all_checks_pass": missing_report["all_checks_pass"],
        "missing_fields": missing_report["missing_fields"],
        "blocked_by_assert": missing_blocked,
        "expected_field_detected":
            "dataset_manifest_sha256"
            in missing_report["missing_fields"],
    }

    checks = {
        "locked_source_integrity_all_pass":
            all(source_checks.values()),

        "locked_partition_seed_is_42":
            seed_identity["partition_seed"] == 42,

        "locked_training_seed_count_is_10":
            len(
                seed_identity["ordered_training_seed_values"]
            ) == 10,

        "seed_index_1_mapping_correct":
            seed_identity["seed_index_to_training_seed"][1]
            == 204886845,

        "positive_identity_passes":
            positive_report["all_checks_pass"]
            and positive_report["launch_status"] == "PASS",

        "s2_14_does_not_authorize_training":
            positive_report["training_authorized_by_s2_14"]
            is False,

        "scientific_source_mismatch_blocked":
            cases["scientific_source_mismatch"][
                "blocked_by_assert"
            ]
            and cases["scientific_source_mismatch"][
                "expected_field_detected"
            ],

        "config_mismatch_blocked":
            cases["config_mismatch"]["blocked_by_assert"]
            and cases["config_mismatch"][
                "expected_field_detected"
            ],

        "data_mismatch_blocked":
            cases["data_mismatch"]["blocked_by_assert"]
            and cases["data_mismatch"][
                "expected_field_detected"
            ],

        "seed_mismatch_blocked":
            cases["seed_mismatch"]["blocked_by_assert"]
            and cases["seed_mismatch"][
                "expected_field_detected"
            ],

        "missing_identity_field_blocked":
            cases["missing_required_identity_field"][
                "blocked_by_assert"
            ]
            and cases["missing_required_identity_field"][
                "expected_field_detected"
            ],
    }

    failed = [
        name for name, passed in checks.items()
        if not passed
    ]

    report = {
        "schema_version": "1.0",
        "stage": "S2.14",
        "scope":
            "fail_fast_source_config_data_seed_checks",
        "official_training_authorized": False,
        "source_integrity_checks": source_checks,
        "seed_identity": {
            "partition_seed":
                seed_identity["partition_seed"],
            "training_seed_count":
                len(
                    seed_identity[
                        "ordered_training_seed_values"
                    ]
                ),
            "seed_index_1_training_seed":
                seed_identity[
                    "seed_index_to_training_seed"
                ][1],
        },
        "positive_case": positive_report,
        "failure_injection_cases": cases,
        "checks": checks,
        "check_count": len(checks),
        "failed_checks": failed,
        "all_checks_pass": not failed,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    for name, passed in checks.items():
        print(f"{name} = {passed}")

    print("failed_checks =", failed)
    print("ALL_CHECKS_PASS =", not failed)

    if failed:
        print("S2_14_FAIL_FAST_GUARDRAILS=FAIL")
        return 1

    print("S2_14_FAIL_FAST_GUARDRAILS=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())