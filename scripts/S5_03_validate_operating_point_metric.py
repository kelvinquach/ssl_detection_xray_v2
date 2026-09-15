"""S5.03 golden tests for deterministic operating-point Recall."""

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from src.metrics.operating_point_metric import OperatingPointMetric

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts" / "preflight" / "evaluation" / "operating_point_golden_test.json"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_fixture(path):
    cats = [{"id": i, "name": f"class_{i}"} for i in range(1, 15)]
    imgs = [{"id": i, "width": 100, "height": 100, "file_name": f"{i}.jpg"} for i in range(1, 10)]
    anns = []
    aid = 1

    def add(img, bbox, cat=1):
        nonlocal aid
        x, y, w, h = bbox
        anns.append({"id": aid, "image_id": img, "category_id": cat,
                     "bbox": [x, y, w, h], "area": float(w * h), "iscrowd": 0})
        aid += 1

    add(1, [0, 0, 10, 10])
    add(2, [0, 0, 10, 10]); add(2, [4, 0, 10, 10])
    add(3, [0, 0, 10, 10]); add(3, [4, 0, 10, 10])
    add(4, [0, 0, 10, 10]); add(4, [4, 0, 10, 10])
    add(5, [4, 0, 10, 10]); add(5, [0, 0, 10, 10])
    add(6, [0, 0, 10, 10]); add(6, [4, 0, 10, 10])
    add(7, [0, 0, 10, 10])
    add(8, [0, 0, 10, 10])
    # image 9 intentionally zero-GT

    path.write_text(json.dumps({"images": imgs, "annotations": anns, "categories": cats}), encoding="utf-8")


def boxes(items):
    return np.asarray(items, dtype=np.float64).reshape(-1, 4)


def run():
    checks = []

    def check(name, ok, observed):
        checks.append({"name": name, "pass": bool(ok), "observed": observed})

    with tempfile.TemporaryDirectory() as td:
        ann = Path(td) / "fixture.json"
        build_fixture(ann)
        metric = OperatingPointMetric(str(ann), tau_eval=0.50, iou_match_threshold=0.50,
                                      max_dets_per_image=100, prefix="operating")

        check("locks", metric.tau_eval == 0.5 and metric.iou_match_threshold == 0.5
              and metric.max_dets_per_image == 100 and metric.cat_ids == tuple(range(1, 15)),
              {"tau": metric.tau_eval, "iou": metric.iou_match_threshold,
               "max": metric.max_dets_per_image, "cats": list(metric.cat_ids)})

        r = metric.match_image(1, boxes([[0, 0, 10, 10]]), np.array([0.50]), np.array([0]))
        check("tau_inclusive", r == {"TP": 1, "FP": 0, "FN": 0, "GT": 1, "retained_detections": 1}, r)

        r = metric.match_image(1, boxes([[0, 0, 10, 10]]), np.array([0.4999]), np.array([0]))
        check("below_tau_filtered", r == {"TP": 0, "FP": 0, "FN": 1, "GT": 1, "retained_detections": 0}, r)

        r = metric.match_image(1, boxes([[0, 0, 10, 10]]), np.array([0.90]), np.array([1]))
        check("category_aware", r["TP"] == 0 and r["FP"] == 1 and r["FN"] == 1, r)

        r = metric.match_image(1, boxes([[0, 0, 10, 10], [0, 0, 10, 10]]),
                               np.array([0.90, 0.80]), np.array([0, 0]))
        check("one_to_one", r["TP"] == 1 and r["FP"] == 1 and r["FN"] == 0, r)

        r = metric.match_image(2, boxes([[0, 0, 10, 10], [2, 0, 12, 10]]),
                               np.array([0.60, 0.90]), np.array([0, 0]))
        check("score_descending", r["TP"] == 1 and r["FP"] == 1 and r["FN"] == 1, r)

        r = metric.match_image(3, boxes([[0, 0, 10, 10], [2, 0, 12, 10]]),
                               np.array([0.90, 0.90]), np.array([0, 0]))
        check("equal_score_stable_detector_order", r["TP"] == 2 and r["FP"] == 0 and r["FN"] == 0, r)

        r4 = metric.match_image(4, boxes([[2, 0, 12, 10], [0, 0, 10, 10]]),
                                np.array([0.90, 0.80]), np.array([0, 0]))
        r5 = metric.match_image(5, boxes([[2, 0, 12, 10], [0, 0, 10, 10]]),
                                np.array([0.90, 0.80]), np.array([0, 0]))
        check("equal_iou_earliest_annotation_order", r4["TP"] == 1 and r5["TP"] == 2,
              {"earlier_gt1": r4, "earlier_gt2": r5})

        r = metric.match_image(6, boxes([[4, 0, 14, 10], [0, 0, 10, 10]]),
                               np.array([0.90, 0.80]), np.array([0, 0]))
        check("maximum_iou_selection", r["TP"] == 2 and r["FP"] == 0 and r["FN"] == 0, r)

        r = metric.match_image(7, boxes([[0, 0, 5, 10]]), np.array([0.90]), np.array([0]))
        check("iou_inclusive_050", r["TP"] == 1 and r["FN"] == 0, r)

        r = metric.match_image(9, boxes([[0, 0, 10, 10]]), np.array([0.90]), np.array([0]))
        check("zero_gt_no_fn", r["TP"] == 0 and r["FP"] == 1 and r["FN"] == 0 and r["GT"] == 0, r)

        guard = False
        try:
            metric.match_image(1, np.zeros((101, 4)), np.ones(101), np.zeros(101, dtype=np.int64))
        except ValueError as exc:
            guard = "more than 100 detector-native predictions" in str(exc)
        check("max100_guard_rejects_101", guard, guard)

        agg = metric.compute_metrics([
            {"img_id": 1, "bboxes": boxes([[0, 0, 10, 10]]),
             "scores": np.array([0.90]), "labels": np.array([0])},
            {"img_id": 8, "bboxes": boxes([[0, 0, 10, 10]]),
             "scores": np.array([0.10]), "labels": np.array([0])},
        ])
        check("recall_aggregation", agg == {"Recall_tau_eval": 0.5}, agg)
        check("recall_only_output", set(agg) == {"Recall_tau_eval"}, sorted(agg))

        raw = [a["id"] for a in metric._coco_api.dataset["annotations"] if a["image_id"] == 5]
        api = [a["id"] for a in metric._ground_truth_for_image(5)]
        check("fixed_coco_annotation_order_preserved", raw == api, {"raw": raw, "api": api})

    all_pass = all(c["pass"] for c in checks)
    evidence = {
        "schema_version": "1.0.0",
        "task": "S5.03",
        "evidence_type": "operating_point_golden_test",
        "status": "PASS" if all_pass else "FAIL",
        "metric": "Recall_tau_eval",
        "tau_eval": 0.50,
        "iou_match_threshold": 0.50,
        "max_dets_per_image": 100,
        "after_detector_native_nms_max100": True,
        "evaluator_runs_second_nms": False,
        "evaluator_truncates_top100": False,
        "matching": "CATEGORY_AWARE_ONE_TO_ONE",
        "matching_algorithm": "DETERMINISTIC_SCORE_DESCENDING_GREEDY",
        "prediction_order": "DESCENDING_CONFIDENCE_STABLE_DETECTOR_ORDER",
        "gt_candidate_policy": "UNMATCHED_SAME_CLASS_ONLY",
        "gt_selection": "MAX_IOU",
        "equal_score_tie_break": "PRESERVE_DETECTOR_OUTPUT_ORDER",
        "equal_iou_tie_break": "EARLIEST_FIXED_COCO_ANNOTATION_ORDER",
        "fixed_validation_only": True,
        "test_access": False,
        "fixture": "SYNTHETIC_COCO_14_CLASS",
        "checks_passed": sum(c["pass"] for c in checks),
        "checks_total": len(checks),
        "checks": checks,
        "source_sha256": {
            "src/metrics/operating_point_metric.py": sha256(ROOT / "src/metrics/operating_point_metric.py"),
            "src/metrics/__init__.py": sha256(ROOT / "src/metrics/__init__.py"),
            "configs/evaluation/s5_03_operating_point_evaluator.py": sha256(ROOT / "configs/evaluation/s5_03_operating_point_evaluator.py"),
            "scripts/S5_03_validate_operating_point_metric.py": sha256(Path(__file__)),
        },
    }
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("===== S5.03 OPERATING-POINT GOLDEN TEST =====")
    for c in checks:
        print(f'{c["name"]}={c["pass"]}')
    print(f'CHECKS_PASSED={evidence["checks_passed"]}')
    print(f'CHECKS_TOTAL={evidence["checks_total"]}')
    print(f'STATUS={evidence["status"]}')
    print(f'EVIDENCE={EVIDENCE}')
    print(f'EVIDENCE_SHA256={sha256(EVIDENCE)}')
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(run())
