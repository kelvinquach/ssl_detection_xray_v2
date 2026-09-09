#!/usr/bin/env python3
"""S2.15 hidden-U GT firewall runtime preflight. No training."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import tempfile
from pathlib import Path


EXPECTED_U = {
    "U_1pct": 3392,
    "U_5pct": 3255,
    "U_10pct": 3083,
    "U_20pct": 2741,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default="/workspace/ssod/project",
    )
    parser.add_argument(
        "--ssod-root",
        default="/workspace/ssod",
    )
    parser.add_argument(
        "--output",
        default=(
            "/workspace/ssod/artifacts/preflight/firewall/"
            "hidden_u_gt_firewall_report.json"
        ),
    )
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    ssod = Path(args.ssod_root).resolve()
    output = Path(args.output).resolve()

    sys.path.insert(0, str(repo))

    from src.utils.unlabeled_firewall import (
        HiddenUGroundTruthError,
        assert_unlabeled_coco_isolated,
        assert_unlabeled_pipeline_isolated,
    )

    os.environ["SSOD_ROOT"] = str(ssod)

    cfg = runpy.run_path(
        str(repo / "configs/dataset/s2_coco_dataset.py")
    )
    dataset_cfgs = cfg["DATASETS"]

    from mmengine.registry import init_default_scope
    from mmdet.registry import DATASETS

    init_default_scope("mmdet")

    checks = {}
    source_reports = {}
    runtime_reports = {}

    unlabeled_pipeline = cfg["UNLABELED_PIPELINE"]
    labeled_pipeline = cfg["PIPELINE"]

    try:
        assert_unlabeled_pipeline_isolated(
            unlabeled_pipeline
        )
        checks["unlabeled_pipeline_has_no_LoadAnnotations"] = True
    except HiddenUGroundTruthError:
        checks["unlabeled_pipeline_has_no_LoadAnnotations"] = False

    checks["labeled_pipeline_still_has_LoadAnnotations"] = any(
        isinstance(step, dict)
        and str(step.get("type", "")).lower() == "loadannotations"
        for step in labeled_pipeline
    )

    for name, expected_len in EXPECTED_U.items():
        dataset_cfg = dataset_cfgs[name]
        ann_file = Path(dataset_cfg["ann_file"])

        source_report = assert_unlabeled_coco_isolated(
            ann_file
        )
        source_reports[name] = source_report

        checks[f"{name}_source_annotations_empty"] = (
            source_report["annotation_count"] == 0
            and source_report["all_checks_pass"]
        )

        try:
            assert_unlabeled_pipeline_isolated(
                dataset_cfg["pipeline"]
            )
            pipeline_ok = True
        except HiddenUGroundTruthError:
            pipeline_ok = False

        checks[f"{name}_config_pipeline_isolated"] = (
            pipeline_ok
        )

        dataset = DATASETS.build(dataset_cfg)
        dataset.full_init()

        length_ok = len(dataset) == expected_len
        all_runtime_instances_empty = True

        for index in range(len(dataset)):
            info = dataset.get_data_info(index)
            instances = info.get("instances", [])
            if instances:
                all_runtime_instances_empty = False
                break

        sample = dataset[0]
        sample_ok = sample is not None

        hidden_gt_in_sample = False
        if sample_ok and isinstance(sample, dict):
            data_sample = sample.get("data_samples")

            if data_sample is not None:
                if hasattr(data_sample, "gt_instances"):
                    try:
                        hidden_gt_in_sample = (
                            len(data_sample.gt_instances) > 0
                        )
                    except Exception:
                        hidden_gt_in_sample = True

                if hasattr(data_sample, "gt_instances_ignore"):
                    try:
                        hidden_gt_in_sample = (
                            hidden_gt_in_sample
                            or len(
                                data_sample.gt_instances_ignore
                            ) > 0
                        )
                    except Exception:
                        hidden_gt_in_sample = True

        checks[f"{name}_runtime_length_correct"] = length_ok
        checks[f"{name}_runtime_instances_empty"] = (
            all_runtime_instances_empty
        )
        checks[f"{name}_sample_loads"] = sample_ok
        checks[f"{name}_sample_hidden_gt_absent"] = (
            not hidden_gt_in_sample
        )

        runtime_reports[name] = {
            "image_count": len(dataset),
            "expected_image_count": expected_len,
            "all_data_info_instances_empty":
                all_runtime_instances_empty,
            "sample_0_loaded": sample_ok,
            "sample_0_hidden_gt_present":
                hidden_gt_in_sample,
        }

    # Negative fixture 1:
    # a contaminated unlabeled COCO file must be blocked.
    source_u = Path(
        dataset_cfgs["U_10pct"]["ann_file"]
    )
    with source_u.open("r", encoding="utf-8") as f:
        contaminated = json.load(f)

    contaminated["annotations"] = [{
        "id": 999999999,
        "image_id": contaminated["images"][0]["id"],
        "category_id": 1,
        "bbox": [1.0, 1.0, 10.0, 10.0],
        "area": 100.0,
        "iscrowd": 0,
    }]

    contaminated_source_blocked = False

    with tempfile.TemporaryDirectory() as tmp:
        contaminated_path = (
            Path(tmp) / "instances_unlabeled_contaminated.json"
        )
        contaminated_path.write_text(
            json.dumps(
                contaminated,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        try:
            assert_unlabeled_coco_isolated(
                contaminated_path
            )
        except HiddenUGroundTruthError:
            contaminated_source_blocked = True

    checks["contaminated_unlabeled_source_blocked"] = (
        contaminated_source_blocked
    )

    # Negative fixture 2:
    # an unlabeled pipeline containing LoadAnnotations must block.
    contaminated_pipeline = [
        dict(
            type="LoadImageFromFile",
            color_type="color",
        ),
        dict(
            type="LoadAnnotations",
            with_bbox=True,
        ),
        dict(
            type="Resize",
            scale=(1333, 800),
            keep_ratio=True,
        ),
        dict(
            type="Pad",
            size_divisor=32,
        ),
        dict(
            type="PackDetInputs",
        ),
    ]

    contaminated_pipeline_blocked = False

    try:
        assert_unlabeled_pipeline_isolated(
            contaminated_pipeline
        )
    except HiddenUGroundTruthError:
        contaminated_pipeline_blocked = True

    checks["LoadAnnotations_in_unlabeled_pipeline_blocked"] = (
        contaminated_pipeline_blocked
    )

    failed = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    report = {
        "schema_version": "1.0",
        "stage": "S2.15",
        "scope": "hidden_u_gt_isolation_foundation",
        "official_training_authorized": False,
        "hidden_u_annotations_present_in_training_object": False
        if not failed
        else None,
        "unlabeled_source_reports": source_reports,
        "runtime_dataset_reports": runtime_reports,
        "negative_fixture_results": {
            "contaminated_unlabeled_source_blocked":
                contaminated_source_blocked,
            "LoadAnnotations_in_unlabeled_pipeline_blocked":
                contaminated_pipeline_blocked,
        },
        "checks": checks,
        "check_count": len(checks),
        "failed_checks": failed,
        "all_checks_pass": not failed,
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
        print("S2_15_HIDDEN_U_GT_FIREWALL=FAIL")
        return 1

    print("S2_15_HIDDEN_U_GT_FIREWALL=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())