#!/usr/bin/env python3
"""S4.16 hidden-U GT firewall verification. No training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CASES = [
    ("R50-FPN", "1pct", 3392),
    ("R50-FPN", "5pct", 3255),
    ("R50-FPN", "10pct", 3083),
    ("R50-FPN", "20pct", 2741),
    ("Swin-T-FPN", "1pct", 3392),
    ("Swin-T-FPN", "5pct", 3255),
    ("Swin-T-FPN", "10pct", 3083),
    ("Swin-T-FPN", "20pct", 2741),
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-root",
        default="/workspace/ssod/project",
    )

    parser.add_argument(
        "--output",
        default=(
            "/workspace/ssod/artifacts/preflight/firewall/"
            "hidden_u_gt_firewall_report.json"
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = Path(args.repo_root).resolve()
    output = Path(args.output).resolve()

    sys.path.insert(0, str(repo_root))

    import src.utils.pre_run_guardrails as pr
    import src.utils.unlabeled_firewall as fw

    checks = {}
    execution_reports = {}

    # --------------------------------------------------------------
    # Positive current-S4 execution-path verification.
    # --------------------------------------------------------------
    for architecture, budget, expected_u in CASES:
        key = f"{architecture}:{budget}"

        try:
            firewall = fw.assert_current_ssl_execution_firewall(
                repo_root,
                architecture=architecture,
                budget=budget,
            )

            identity = pr.build_expected_pre_run_identity(
                repo_root,
                method="SSL",
                architecture=architecture,
                budget=budget,
                seed_index=1,
            )

            current_path_ok = (
                firewall["all_checks_pass"]
                and firewall["annotation_count"] == 0
                and firewall["image_count"] == expected_u
                and firewall["category_count"] == 14
                and firewall["recursive_pipeline_isolated"]
                and firewall["unlabeled_source_isolated"]
                and identity["method"] == "SSL"
                and identity["architecture"] == architecture
                and identity["budget"] == budget
                and identity["effective_unlabeled_batch"] == 4
            )

            error = None

        except Exception as exc:
            firewall = None
            current_path_ok = False
            error = f"{type(exc).__name__}: {exc}"

        checks[
            f"{key}_current_ssl_firewall_pass"
        ] = current_path_ok

        execution_reports[key] = {
            "architecture": architecture,
            "budget": budget,
            "expected_unlabeled_image_count": expected_u,
            "firewall_report": firewall,
            "pass": current_path_ok,
            "error": error,
        }

    # --------------------------------------------------------------
    # SUP must remain independent of the hidden-U firewall.
    # --------------------------------------------------------------
    try:
        sup = pr.build_expected_pre_run_identity(
            repo_root,
            method="SUP",
            architecture="R50-FPN",
            budget="10pct",
            seed_index=1,
        )

        sup_ok = (
            sup["method"] == "SUP"
            and sup["unlabeled_manifest_sha256"]
            == "NOT_APPLICABLE"
            and sup["effective_unlabeled_batch"]
            == "NOT_APPLICABLE"
        )

        sup_error = None

    except Exception as exc:
        sup_ok = False
        sup_error = f"{type(exc).__name__}: {exc}"

    checks["sup_path_unaffected"] = sup_ok

    # --------------------------------------------------------------
    # Negative fixture 1:
    # nested LoadAnnotations inside MultiBranch must be blocked.
    # --------------------------------------------------------------
    contaminated_pipeline = [
        {
            "type": "LoadImageFromFile",
        },
        {
            "type": "MultiBranch",
            "branch_field": [
                "unsup_teacher",
                "unsup_student",
            ],
            "unsup_teacher": [
                {"type": "Resize"},
                {
                    "type": "LoadAnnotations",
                    "with_bbox": True,
                },
            ],
            "unsup_student": [
                {"type": "Resize"},
            ],
        },
    ]

    nested_detected = fw.pipeline_exposes_annotations(
        contaminated_pipeline
    )

    nested_blocked = False

    try:
        fw.assert_unlabeled_pipeline_isolated(
            contaminated_pipeline
        )
    except fw.HiddenUGroundTruthError:
        nested_blocked = True

    checks[
        "nested_LoadAnnotations_detected"
    ] = nested_detected

    checks[
        "nested_LoadAnnotations_blocked"
    ] = nested_blocked

    # --------------------------------------------------------------
    # Negative fixture 2:
    # contaminated unlabeled COCO must be blocked.
    # Uses only a synthetic temporary copy of stripped U input.
    # --------------------------------------------------------------
    import tempfile

    source_report = fw.assert_current_ssl_execution_firewall(
        repo_root,
        architecture="R50-FPN",
        budget="10pct",
    )

    source_path = Path(
        source_report["resolved_ann_file"]
    )

    payload = json.loads(
        source_path.read_text(encoding="utf-8")
    )

    contaminated = dict(payload)
    contaminated["images"] = list(payload["images"])
    contaminated["categories"] = list(payload["categories"])
    contaminated["annotations"] = [{
        "id": 999999999,
        "image_id":
            contaminated["images"][0]["id"],
        "category_id": 1,
        "bbox": [1.0, 1.0, 10.0, 10.0],
        "area": 100.0,
        "iscrowd": 0,
    }]

    contaminated_source_blocked = False
    contaminated_source_error = None

    with tempfile.TemporaryDirectory() as tmp:
        contaminated_path = (
            Path(tmp)
            / "instances_unlabeled_10pct.json"
        )

        contaminated_path.write_text(
            json.dumps(
                contaminated,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        try:
            fw.assert_unlabeled_coco_isolated(
                contaminated_path
            )
        except fw.HiddenUGroundTruthError as exc:
            contaminated_source_blocked = True
            contaminated_source_error = str(exc)

    checks[
        "contaminated_unlabeled_source_blocked"
    ] = contaminated_source_blocked

    # --------------------------------------------------------------
    # Negative fixture 3:
    # active SSL pre-run must propagate firewall failure.
    # --------------------------------------------------------------
    original_assert = (
        pr.assert_current_ssl_execution_firewall
    )

    active_pre_run_blocked = False
    active_pre_run_error = None

    def forced_hidden_u_failure(
        repo_root,
        *,
        architecture,
        budget,
    ):
        raise fw.HiddenUGroundTruthError(
            "synthetic hidden-U contamination"
        )

    pr.assert_current_ssl_execution_firewall = (
        forced_hidden_u_failure
    )

    try:
        pr.build_expected_pre_run_identity(
            repo_root,
            method="SSL",
            architecture="R50-FPN",
            budget="10pct",
            seed_index=1,
        )
    except fw.HiddenUGroundTruthError as exc:
        active_pre_run_blocked = True
        active_pre_run_error = str(exc)
    finally:
        pr.assert_current_ssl_execution_firewall = (
            original_assert
        )

    checks[
        "active_ssl_pre_run_blocks_firewall_failure"
    ] = active_pre_run_blocked

    # --------------------------------------------------------------
    # Explicit firewall scope declarations.
    # These are execution facts of this validator, not inferred GT.
    # --------------------------------------------------------------
    checks[
        "hidden_u_master_gt_not_opened"
    ] = True

    checks[
        "real_unlabeled_coco_not_modified"
    ] = True

    checks[
        "model_forward_not_executed"
    ] = True

    checks[
        "optimizer_step_not_executed"
    ] = True

    checks[
        "q_pseudo_not_executed"
    ] = True

    checks[
        "official_training_not_authorized"
    ] = True

    failed = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    report = {
        "schema_version": "1.0",
        "stage": "S4.16",
        "scope": "hidden_u_gt_firewall",
        "constraint":
            "No hidden U GT in development/Q_pseudo.",
        "official_training_authorized": False,
        "hidden_u_master_gt_opened": False,
        "real_unlabeled_coco_modified": False,
        "model_forward_executed": False,
        "optimizer_step_executed": False,
        "q_pseudo_executed": False,
        "current_ssl_execution_reports":
            execution_reports,
        "sup_non_regression": {
            "pass": sup_ok,
            "error": sup_error,
        },
        "negative_fixture_results": {
            "nested_LoadAnnotations_detected":
                nested_detected,
            "nested_LoadAnnotations_blocked":
                nested_blocked,
            "contaminated_unlabeled_source_blocked":
                contaminated_source_blocked,
            "contaminated_unlabeled_source_error":
                contaminated_source_error,
            "active_ssl_pre_run_blocks_firewall_failure":
                active_pre_run_blocked,
            "active_ssl_pre_run_error":
                active_pre_run_error,
        },
        "checks": checks,
        "check_count": len(checks),
        "failed_checks": failed,
        "all_checks_pass": not failed,
        "overall_pass": not failed,
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
        ) + "\n",
        encoding="utf-8",
    )

    for name, passed in checks.items():
        print(f"{name} = {passed}")

    print("failed_checks =", failed)
    print("ALL_CHECKS_PASS =", not failed)

    if failed:
        print("S4_16_HIDDEN_U_GT_FIREWALL=FAIL")
        return 1

    print("S4_16_HIDDEN_U_GT_FIREWALL=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
