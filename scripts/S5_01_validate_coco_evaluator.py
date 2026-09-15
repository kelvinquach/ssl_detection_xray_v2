#!/usr/bin/env python3
"""S5.01 — Golden test for the fixed COCO bbox evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import torch
from mmengine.config import Config
from mmdet import __version__ as mmdet_version
from mmdet.evaluation.metrics import CocoMetric


CONFIG_REL = "configs/evaluation/s5_01_coco_bbox_evaluator.py"
EXPECTED_IOU_THRS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
EXPECTED_PROPOSAL_NUMS = [1, 10, 100]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_coco(path: Path, *, categories: list[dict], annotations: list[dict]) -> None:
    image_ids = sorted({int(a["image_id"]) for a in annotations} | {1})
    images = [
        {"id": image_id, "width": 100, "height": 100, "file_name": f"{image_id}.jpg"}
        for image_id in image_ids
    ]
    payload = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "licenses": [],
        "info": {},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_metric(
    ann_file: Path,
    classes: tuple[str, ...],
    data_samples: list[dict],
) -> dict:
    metric = CocoMetric(
        ann_file=str(ann_file),
        metric="bbox",
        classwise=False,
        proposal_nums=(1, 10, 100),
        iou_thrs=EXPECTED_IOU_THRS,
        metric_items=["mAP"],
        format_only=False,
        prefix="coco",
    )
    metric.dataset_meta = {"classes": classes}
    metric.process({}, data_samples)
    return metric.evaluate(size=len(data_samples))


def sample(
    *,
    img_id: int,
    bboxes: list[list[float]],
    scores: list[float],
    labels: list[int],
) -> dict:
    return {
        "img_id": img_id,
        "ori_shape": (100, 100),
        "pred_instances": {
            "bboxes": torch.tensor(bboxes, dtype=torch.float32).reshape(-1, 4),
            "scores": torch.tensor(scores, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/workspace/ssod/project")
    parser.add_argument(
        "--output",
        default=(
            "/workspace/ssod/artifacts/preflight/evaluation/"
            "coco_evaluator_golden_test.json"
        ),
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo_root))

    config_path = repo_root / CONFIG_REL
    checks: dict[str, bool] = {}
    observed: dict[str, object] = {}

    try:
        cfg = Config.fromfile(str(config_path))
        evaluator = dict(cfg.val_evaluator)
        dataset = dict(cfg.val_dataloader["dataset"])

        checks["config_val_cfg_is_valloop"] = cfg.val_cfg["type"] == "ValLoop"
        checks["config_uses_fixed_validation"] = (
            str(dataset["ann_file"]).endswith("instances_val.json")
            and bool(dataset["test_mode"])
            and str(evaluator["ann_file"]).endswith("instances_val.json")
        )
        checks["config_metric_is_bbox"] = evaluator["metric"] == "bbox"
        checks["config_classwise_false"] = evaluator["classwise"] is False
        checks["config_proposal_nums_1_10_100"] = (
            list(evaluator["proposal_nums"]) == EXPECTED_PROPOSAL_NUMS
        )
        checks["config_iou_grid_exact"] = [
            round(float(x), 2) for x in evaluator["iou_thrs"]
        ] == EXPECTED_IOU_THRS
        checks["config_metric_items_map_only"] = list(evaluator["metric_items"]) == ["mAP"]
        checks["config_format_only_false"] = evaluator["format_only"] is False
        checks["config_prefix_coco"] = evaluator["prefix"] == "coco"
        checks["config_test_not_attached"] = (
            "test_dataloader" not in cfg
            and "test_evaluator" not in cfg
            and "test_cfg" not in cfg
        )

        with tempfile.TemporaryDirectory(prefix="s5_01_coco_") as tmp:
            tmp_dir = Path(tmp)

            low_score_ann = tmp_dir / "low_score.json"
            write_coco(
                low_score_ann,
                categories=[{"id": 1, "name": "A"}],
                annotations=[
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [10, 10, 20, 20],
                        "area": 400,
                        "iscrowd": 0,
                    }
                ],
            )
            low_score_metrics = run_metric(
                low_score_ann,
                ("A",),
                [
                    sample(
                        img_id=1,
                        bboxes=[[10, 10, 30, 30]],
                        scores=[0.001],
                        labels=[0],
                    )
                ],
            )
            low_score_map = float(low_score_metrics["coco/bbox_mAP"])
            observed["low_score_tp_bbox_mAP"] = low_score_map
            checks["no_extra_fixed_score_threshold_before_ap"] = abs(low_score_map - 1.0) < 1e-9

            wrong_cat_ann = tmp_dir / "wrong_category.json"
            write_coco(
                wrong_cat_ann,
                categories=[
                    {"id": 1, "name": "A"},
                    {"id": 2, "name": "B"},
                ],
                annotations=[
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [10, 10, 20, 20],
                        "area": 400,
                        "iscrowd": 0,
                    }
                ],
            )
            wrong_cat_metrics = run_metric(
                wrong_cat_ann,
                ("A", "B"),
                [
                    sample(
                        img_id=1,
                        bboxes=[[10, 10, 30, 30]],
                        scores=[0.99],
                        labels=[1],
                    )
                ],
            )
            wrong_cat_map = float(wrong_cat_metrics["coco/bbox_mAP"])
            observed["wrong_category_bbox_mAP"] = wrong_cat_map
            checks["category_aware_usecCats_1_semantics"] = abs(wrong_cat_map - 0.0) < 1e-9

            max_dets_ann = tmp_dir / "max_dets.json"
            write_coco(
                max_dets_ann,
                categories=[{"id": 1, "name": "A"}],
                annotations=[
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [10, 10, 20, 20],
                        "area": 400,
                        "iscrowd": 0,
                    }
                ],
            )
            false_boxes = [[60, 60, 70, 70] for _ in range(100)]
            false_scores = [1.0 - (i * 0.001) for i in range(100)]
            max_dets_metrics = run_metric(
                max_dets_ann,
                ("A",),
                [
                    sample(
                        img_id=1,
                        bboxes=false_boxes + [[10, 10, 30, 30]],
                        scores=false_scores + [0.001],
                        labels=[0] * 101,
                    )
                ],
            )
            max_dets_map = float(max_dets_metrics["coco/bbox_mAP"])
            observed["rank_101_tp_bbox_mAP"] = max_dets_map
            checks["ap_enforces_max_dets_100"] = abs(max_dets_map - 0.0) < 1e-9

        all_checks_pass = all(checks.values())
        evidence = {
            "schema_version": "1.0.0",
            "stage": "S5",
            "task": "S5.01",
            "status": "PASS" if all_checks_pass else "FAIL",
            "all_checks_pass": all_checks_pass,
            "scientific_contract": {
                "primary_metric": "bbox_mAP_50_95",
                "iou_thresholds": EXPECTED_IOU_THRS,
                "useCats": 1,
                "maxDets": 100,
                "extra_fixed_score_threshold_before_AP": False,
                "dataset": "fixed_validation",
                "test_attached": False,
            },
            "implementation": {
                "config": CONFIG_REL,
                "config_sha256": file_sha256(config_path),
                "runtime_metric_key": "coco/bbox_mAP",
            },
            "runtime": {
                "mmdet_version": mmdet_version,
                "metric_class": "mmdet.evaluation.metrics.CocoMetric",
            },
            "checks": checks,
            "observed": observed,
        }
        output_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(evidence, indent=2))
        print(f"EVIDENCE={output_path}")
        return 0 if all_checks_pass else 1

    except Exception as exc:
        evidence = {
            "schema_version": "1.0.0",
            "stage": "S5",
            "task": "S5.01",
            "status": "FAIL",
            "all_checks_pass": False,
            "checks": checks,
            "observed": observed,
            "error": f"{type(exc).__name__}: {exc}",
        }
        output_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(evidence, indent=2))
        print(f"EVIDENCE={output_path}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
