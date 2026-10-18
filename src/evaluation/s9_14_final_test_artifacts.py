"""S9.14 per-run final-test artifact materializer.

Consumes the exact one-pass detector outputs retained by
FinalTestPredictionRetentionHook plus the already-computed frozen evaluator
metrics. It does not run model inference, NMS, score filtering before COCO AP,
or an independent matching algorithm.
"""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

EXPECTED_TEST_IMAGES = 734
EXPECTED_NEGATIVE_IMAGES = 75
EXPECTED_MAX_DETS = 100


def _json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _metric(metrics: Mapping[str, Any], suffix: str) -> float:
    matches = [key for key in metrics if key == suffix or key.endswith("/" + suffix)]
    if len(matches) != 1:
        raise KeyError(f"Expected exactly one metric ending in {suffix!r}; got {matches!r}")
    value = float(metrics[matches[0]])
    if not math.isfinite(value):
        raise ValueError(f"Metric {matches[0]!r} is non-finite: {value!r}")
    return value


def _close(name: str, observed: float, expected: float) -> None:
    if not math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            f"S9.14 derivation mismatch for {name}: "
            f"observed={observed!r} expected={expected!r}"
        )


def _validate_native_records(records: Sequence[Mapping[str, Any]]) -> None:
    if len(records) != EXPECTED_TEST_IMAGES:
        raise RuntimeError(
            f"S9.14 retained prediction count mismatch: "
            f"records={len(records)} expected={EXPECTED_TEST_IMAGES}"
        )
    seen: set[int] = set()
    for record in records:
        img_id = int(record["img_id"])
        if img_id in seen:
            raise RuntimeError(f"Duplicate final-test image id: {img_id}")
        seen.add(img_id)
        bboxes, scores, labels = record["bboxes"], record["scores"], record["labels"]
        if not (len(bboxes) == len(scores) == len(labels)):
            raise RuntimeError(f"Prediction field length mismatch for image {img_id}")
        if len(bboxes) > EXPECTED_MAX_DETS:
            raise RuntimeError(
                f"Detector-native output exceeds max_per_img=100 for image {img_id}"
            )


def _write_detections(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    protocol_metric: Any,
) -> None:
    if tuple(int(x) for x in protocol_metric.proposal_nums) != (1, 10, 100):
        raise RuntimeError("ProtocolCocoMetric proposal_nums is not (1, 10, 100)")
    if protocol_metric.cat_ids is None:
        raise RuntimeError("ProtocolCocoMetric cat_ids is not initialized")
    serializable = [
        {
            "img_id": int(r["img_id"]),
            "bboxes": np.asarray(r["bboxes"], dtype=np.float64).reshape(-1, 4),
            "scores": np.asarray(r["scores"], dtype=np.float64),
            "labels": np.asarray(r["labels"], dtype=np.int64),
        }
        for r in records
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(path.parent)) as td:
        files = protocol_metric.results2json(serializable, str(Path(td) / "detections"))
        bbox_file = Path(files["bbox"])
        if not bbox_file.is_file():
            raise RuntimeError("MMDetection did not materialize bbox detections")
        shutil.copyfile(bbox_file, path)


def _derive_operating(records: Sequence[Mapping[str, Any]], operating_metric: Any) -> dict:
    matched = []
    for r in records:
        matched.append(
            operating_metric.match_image(
                int(r["img_id"]),
                np.asarray(r["bboxes"], dtype=np.float64).reshape(-1, 4),
                np.asarray(r["scores"], dtype=np.float64),
                np.asarray(r["labels"], dtype=np.int64),
            )
        )
    gt_total = sum(int(x["GT"]) for x in matched)
    tp_total = sum(int(x["TP"]) for x in matched)
    fn_total = sum(int(x["FN"]) for x in matched)
    fp_total = sum(int(x["FP"]) for x in matched)
    negative = [x for x in matched if int(x["GT"]) == 0]
    negative_image_count = len(negative)
    fp_total_negative = sum(int(x["FP"]) for x in negative)
    negative_images_with_fp = sum(int(x["FP"]) >= 1 for x in negative)
    if negative_image_count != EXPECTED_NEGATIVE_IMAGES:
        raise RuntimeError(
            f"Negative-image count mismatch: {negative_image_count} "
            f"!= {EXPECTED_NEGATIVE_IMAGES}"
        )
    return {
        "evaluated_image_count": len(records),
        "GT_total": gt_total,
        "TP_total": tp_total,
        "FN_total": fn_total,
        "FP_total": fp_total,
        "Recall_tau_eval": tp_total / (tp_total + fn_total),
        "FP_per_image": fp_total / len(records),
        "negative_image_count": negative_image_count,
        "FP_total_on_negative": fp_total_negative,
        "FP_per_negative": fp_total_negative / negative_image_count,
        "negative_image_FAR": negative_images_with_fp / negative_image_count,
    }


def _write_classwise(
    path: Path,
    evaluator_metrics: Mapping[str, Any],
    protocol_metric: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    img_ids = list(protocol_metric.img_ids)
    for class_index, cat_id in enumerate(protocol_metric.cat_ids):
        cat = protocol_metric._coco_api.loadCats(int(cat_id))[0]
        name = str(cat["name"])
        support = len(
            protocol_metric._coco_api.getAnnIds(
                imgIds=img_ids,
                catIds=[int(cat_id)],
            )
        )
        if support == 0:
            row = [int(cat_id), name, 0, "NA", "NA", "NA", "UNDEFINED"]
        else:
            row = [
                int(cat_id),
                name,
                support,
                _metric(evaluator_metrics, f"classwise/{name}/AP_50_95"),
                _metric(evaluator_metrics, f"classwise/{name}/AP50"),
                _metric(evaluator_metrics, f"classwise/{name}/AP75"),
                "DEFINED",
            ]
        rows.append(row)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["class_id", "class_name", "gt_support", "AP_50_95", "AP50", "AP75", "status"]
        )
        writer.writerows(rows)
    os.replace(tmp, path)


def materialize_final_test_artifacts(
    *,
    output_dir: str | Path,
    run_id: str,
    evaluation_model_role: str,
    checkpoint_role: str,
    checkpoint_sha256: str,
    evaluator_config_sha256: str,
    test_split_sha256: str,
    test_authorization_sha256: str,
    evaluation_timestamp_utc: str,
    native_records: Sequence[Mapping[str, Any]],
    evaluator_metrics: Mapping[str, Any],
    protocol_metric: Any,
    operating_metric: Any,
    eligible_model_inventory_match: bool,
    checkpoint_sha256_match: bool,
    test_split_sha256_match: bool,
    authorized: bool,
) -> dict[str, Any]:
    """Materialize the seven mandatory per-run S9.14 final-test artifacts."""
    external_guards = {
        "authorized": bool(authorized),
        "eligible_model_inventory_match": bool(eligible_model_inventory_match),
        "checkpoint_sha256_match": bool(checkpoint_sha256_match),
        "test_split_sha256_match": bool(test_split_sha256_match),
    }
    if not all(external_guards.values()):
        raise RuntimeError(
            f"S9.14 final-test artifact materialization blocked by guards: {external_guards!r}"
        )

    out = Path(output_dir)
    _validate_native_records(native_records)

    derived = _derive_operating(native_records, operating_metric)
    _close(
        "Recall_tau_eval",
        _metric(evaluator_metrics, "Recall_tau_eval"),
        derived["Recall_tau_eval"],
    )
    _close(
        "FP_per_image",
        _metric(evaluator_metrics, "FP_per_image"),
        derived["FP_per_image"],
    )
    _close(
        "FP_per_negative",
        _metric(evaluator_metrics, "FP_per_negative"),
        derived["FP_per_negative"],
    )
    _close(
        "negative_image_FAR",
        _metric(evaluator_metrics, "negative_image_FAR"),
        derived["negative_image_FAR"],
    )

    coco = {
        "evaluation_split": "test",
        "evaluation_model_role": evaluation_model_role,
        "bbox_mAP_50_95": _metric(evaluator_metrics, "bbox_mAP"),
        "AP50": _metric(evaluator_metrics, "bbox_mAP_50"),
        "AP75": _metric(evaluator_metrics, "bbox_mAP_75"),
        "AR_50_95_max100": _metric(evaluator_metrics, "AR_50_95_max100"),
        "useCats": 1,
        "maxDets": 100,
        "extra_fixed_score_threshold_before_AP": False,
    }
    _json_write(out / "coco_metrics.json", coco)

    operating = {
        "evaluation_split": "test",
        "tau_eval": 0.50,
        "iou_match_threshold": 0.50,
        "matching": "CATEGORY_AWARE_ONE_TO_ONE",
        "matching_algorithm": "DETERMINISTIC_SCORE_DESCENDING_GREEDY",
        "prediction_order": "DESCENDING_CONFIDENCE_STABLE_DETECTOR_ORDER",
        "gt_candidate_policy": "UNMATCHED_SAME_CLASS_ONLY",
        "gt_selection": "MAX_IOU",
        "equal_score_tie_break": "PRESERVE_DETECTOR_OUTPUT_ORDER",
        "equal_iou_tie_break": "EARLIEST_FIXED_COCO_ANNOTATION_ORDER",
        "after_detector_native_nms_max100": True,
        **{k: derived[k] for k in (
            "evaluated_image_count", "GT_total", "TP_total", "FN_total",
            "Recall_tau_eval", "FP_total", "FP_per_image",
        )},
    }
    _json_write(out / "operating_point_metrics.json", operating)

    negative = {
        "tau_eval": 0.50,
        "iou_match_threshold": 0.50,
        "matching": "CATEGORY_AWARE_ONE_TO_ONE",
        **{k: derived[k] for k in (
            "negative_image_count", "FP_total_on_negative",
            "FP_per_negative", "negative_image_FAR",
        )},
    }
    _json_write(out / "negative_metrics.json", negative)

    _write_classwise(out / "classwise_ap.csv", evaluator_metrics, protocol_metric)
    _write_detections(out / "detections.json", native_records, protocol_metric)

    manifest = {
        "run_id": run_id,
        "checkpoint_role": checkpoint_role,
        "checkpoint_sha256": checkpoint_sha256,
        "evaluator_config_sha256": evaluator_config_sha256,
        "test_split_sha256": test_split_sha256,
        "test_authorization_sha256": test_authorization_sha256,
        "evaluation_timestamp_utc": evaluation_timestamp_utc,
    }
    _json_write(out / "evaluation_manifest.json", manifest)

    persisted_manifest = json.loads(
        (out / "evaluation_manifest.json").read_text(encoding="utf-8")
    )
    manifest_valid = persisted_manifest == manifest

    guards = {
        **external_guards,
        "evaluation_manifest_valid": bool(manifest_valid),
    }
    if not all(guards.values()):
        raise RuntimeError(f"S9.14 final evaluation closure blocked by guards: {guards!r}")

    report = {
        "run_id": run_id,
        "final_evaluation_status": "CLOSED_VERIFIED",
        **guards,
        "eligible_for_official_analysis": True,
    }
    _json_write(out / "final_evaluation_report.json", report)
    return report
