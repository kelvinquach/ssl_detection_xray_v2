"""S5.02 validator — COCO secondary metrics and class-wise outputs."""

import hashlib
import json
import tempfile
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.utils import import_modules_from_strings
from mmdet.evaluation.metrics import CocoMetric
from mmdet.registry import METRICS


ROOT = Path("/workspace/ssod/project")
CONFIG = ROOT / "configs/evaluation/s5_02_coco_secondary_evaluator.py"
METRIC_IMPL = ROOT / "src/metrics/protocol_coco_metric.py"
EVIDENCE = ROOT / "artifacts/preflight/evaluation/coco_evaluator_golden_test.json"
VAL_ANN = Path("/workspace/ssod/data/coco/instances_val.json")

EXPECTED_IOU = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_fixture(tmp: str):
    ann = Path(tmp) / "ann.json"
    payload = {
        "images": [
            {"id": 1, "width": 100, "height": 100, "file_name": "1.jpg"}
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 10, 10],
                "area": 100,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [40, 10, 10, 10],
                "area": 100,
                "iscrowd": 0,
            },
            {
                "id": 3,
                "image_id": 1,
                "category_id": 2,
                "bbox": [10, 40, 10, 10],
                "area": 100,
                "iscrowd": 0,
            },
        ],
        "categories": [
            {"id": 1, "name": "A"},
            {"id": 2, "name": "B"},
        ],
        "licenses": [],
        "info": {},
    }
    ann.write_text(json.dumps(payload), encoding="utf-8")

    sample = {
        "img_id": 1,
        "ori_shape": (100, 100),
        "pred_instances": {
            "bboxes": torch.tensor(
                [
                    [10, 10, 20, 20],
                    [40, 10, 50, 20],
                    [10, 40, 20, 50],
                ],
                dtype=torch.float32,
            ),
            "scores": torch.tensor([0.99, 0.98, 0.97], dtype=torch.float32),
            "labels": torch.tensor([0, 0, 1], dtype=torch.int64),
        },
    }
    return ann, sample


def main():
    cfg = Config.fromfile(str(CONFIG))
    import_modules_from_strings(**cfg.custom_imports)

    metric = METRICS.build(cfg.val_evaluator)

    with VAL_ANN.open("r", encoding="utf-8") as f:
        val_payload = json.load(f)

    val_categories = val_payload["categories"]
    val_category_ids = [int(x["id"]) for x in val_categories]

    checks = {
        "config_type_protocol_coco_metric":
            cfg.val_evaluator.type == "ProtocolCocoMetric",
        "config_fixed_validation":
            cfg.val_evaluator.ann_file == "/workspace/ssod/data/coco/instances_val.json",
        "config_metric_bbox":
            cfg.val_evaluator.metric == "bbox",
        "config_classwise_true":
            cfg.val_evaluator.classwise is True,
        "config_proposal_nums_1_10_100":
            tuple(cfg.val_evaluator.proposal_nums) == (1, 10, 100),
        "config_metric_items_exact":
            list(cfg.val_evaluator.metric_items)
            == ["mAP", "mAP_50", "mAP_75", "AR@1000"],
        "config_no_explicit_iou_thrs":
            "iou_thrs" not in cfg.val_evaluator,
        "config_test_not_attached":
            "test_evaluator" not in cfg and "test_dataloader" not in cfg,
        "runtime_iou_type_ndarray":
            type(metric.iou_thrs).__name__ == "ndarray",
        "runtime_iou_grid_exact":
            [round(float(x), 2) for x in metric.iou_thrs] == EXPECTED_IOU,
        "runtime_proposal_nums_exact":
            tuple(metric.proposal_nums) == (1, 10, 100),
        "fixed_validation_has_14_categories":
            len(val_categories) == 14,
        "fixed_validation_category_ids_1_14":
            val_category_ids == list(range(1, 15)),
    }

    with tempfile.TemporaryDirectory() as tmp:
        ann, sample = make_fixture(tmp)

        reference = CocoMetric(
            ann_file=str(ann),
            metric="bbox",
            classwise=False,
            proposal_nums=(1, 10, 100),
            metric_items=["AR@100", "AR@1000"],
            prefix="coco",
        )
        reference.dataset_meta = {"classes": ("A", "B")}
        reference.process({}, [sample])
        ref = reference.evaluate(size=1)

        probe = METRICS.build(
            dict(
                type="ProtocolCocoMetric",
                ann_file=str(ann),
                metric="bbox",
                classwise=True,
                proposal_nums=(1, 10, 100),
                metric_items=["mAP", "mAP_50", "mAP_75", "AR@1000"],
                prefix="coco",
            )
        )
        probe.dataset_meta = {"classes": ("A", "B")}
        probe.process({}, [sample])
        out = probe.evaluate(size=1)

    runtime_checks = {
        "aggregate_map":
            float(out["coco/bbox_mAP"]) == 1.0,
        "aggregate_ap50":
            float(out["coco/bbox_mAP_50"]) == 1.0,
        "aggregate_ap75":
            float(out["coco/bbox_mAP_75"]) == 1.0,
        "reference_ar_max1_is_075":
            float(ref["coco/bbox_AR@100"]) == 0.75,
        "reference_ar_max100_is_1":
            float(ref["coco/bbox_AR@1000"]) == 1.0,
        "normalized_ar_max100":
            float(out["coco/AR_50_95_max100"]) == 1.0,
        "raw_misleading_ar_removed":
            "coco/bbox_AR@1000" not in out,
        "class_A_ap":
            float(out["coco/classwise/A/AP_50_95"]) == 1.0,
        "class_A_ap50":
            float(out["coco/classwise/A/AP50"]) == 1.0,
        "class_A_ap75":
            float(out["coco/classwise/A/AP75"]) == 1.0,
        "class_B_ap":
            float(out["coco/classwise/B/AP_50_95"]) == 1.0,
        "class_B_ap50":
            float(out["coco/classwise/B/AP50"]) == 1.0,
        "class_B_ap75":
            float(out["coco/classwise/B/AP75"]) == 1.0,
        "no_official_size_ap_keys":
            all(
                token not in key
                for key in out
                for token in ("mAP_s", "mAP_m", "mAP_l")
            ),
    }
    checks.update(runtime_checks)

    all_pass = all(checks.values())

    if not EVIDENCE.exists():
        raise FileNotFoundError(
            "S5.01 evidence must exist before S5.02 extends it"
        )

    with EVIDENCE.open("r", encoding="utf-8") as f:
        evidence = json.load(f)

    if evidence.get("task") != "S5.01":
        raise RuntimeError("Existing evidence is not the locked S5.01 evidence")
    if evidence.get("status") != "PASS":
        raise RuntimeError("Existing S5.01 evidence is not PASS")
    if evidence.get("all_checks_pass") is not True:
        raise RuntimeError("Existing S5.01 evidence failed its own checks")

    evidence["s5_02_secondary_metrics"] = {
        "task": "S5.02",
        "status": "PASS" if all_pass else "FAIL",
        "all_checks_pass": all_pass,
        "scientific_contract": {
            "AP50": "secondary",
            "AP75": "secondary",
            "classwise_AP_50_95": True,
            "classwise_AP50": True,
            "classwise_AP75": True,
            "AR_50_95_max100": True,
            "useCats": 1,
            "maxDets": 100,
            "area": "all",
            "iou_thresholds": EXPECTED_IOU,
            "official_size_specific_AP": False,
            "dataset": "fixed_validation",
            "test_attached": False,
        },
        "implementation": {
            "config": "configs/evaluation/s5_02_coco_secondary_evaluator.py",
            "config_sha256": sha256(CONFIG),
            "metric_impl": "src/metrics/protocol_coco_metric.py",
            "metric_impl_sha256": sha256(METRIC_IMPL),
            "runtime_metric_class": type(metric).__name__,
            "normalized_ar_key": "coco/AR_50_95_max100",
            "classwise_key_template":
                "coco/classwise/<class_name>/{AP_50_95,AP50,AP75}",
        },
        "fixed_validation": {
            "category_count": len(val_categories),
            "category_ids": val_category_ids,
        },
        "observed": {
            "reference_raw_AR_slot_6_named_bbox_AR_at_100":
                float(ref["coco/bbox_AR@100"]),
            "reference_raw_AR_slot_8_named_bbox_AR_at_1000":
                float(ref["coco/bbox_AR@1000"]),
            "normalized_AR_50_95_max100":
                float(out["coco/AR_50_95_max100"]),
            "bbox_mAP":
                float(out["coco/bbox_mAP"]),
            "AP50":
                float(out["coco/bbox_mAP_50"]),
            "AP75":
                float(out["coco/bbox_mAP_75"]),
        },
        "checks": checks,
    }

    EVIDENCE.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("===== S5.02 VALIDATION =====")
    for key, value in checks.items():
        print(f"{key}={value}")
    print(f"ALL_CHECKS_PASS={all_pass}")
    print(f"EVIDENCE={EVIDENCE}")
    print(f"EVIDENCE_SHA256={sha256(EVIDENCE)}")

    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()