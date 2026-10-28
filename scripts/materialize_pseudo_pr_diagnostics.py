from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


REPO = Path(r"D:\ssl_detection_xray_v2")

PRED_ROOT = (
    REPO
    / "_tmp_qpseudo_count_audit"
    / "predictions"
)

VAL_JSON = (
    REPO
    / "data"
    / "processed"
    / "coco"
    / "instances_val.json"
)

OUT_DIR = (
    REPO
    / "analysis"
    / "pseudo_pr_diagnostics"
)

OUT_CSV = (
    OUT_DIR
    / "pseudo_pr_seed_class_iou.csv"
)

EXPECTED_RUNS = 80
EXPECTED_CLASSES = 14
EXPECTED_IMAGES = 734
EXPECTED_ACCEPTANCE_THRESHOLD = 0.90
EXPECTED_COMPARATOR = "STRICT_GREATER"

IOU_THRESHOLDS = np.arange(0.50, 0.951, 0.05)

RUN_RE = re.compile(
    r"^(OFF_SSL_(R50|SWIN)_B(01|05|10|20)_S"
    r"(01|02|03|04|05|06|07|08|09|10))$"
)

ARCH_MAP = {
    "R50": "R50-FPN",
    "SWIN": "Swin-T-FPN",
}

BUDGET_MAP = {
    "01": "1%",
    "05": "5%",
    "10": "10%",
    "20": "20%",
}


def read_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = [float(x) for x in box]
    return [
        x1,
        y1,
        x2 - x1,
        y2 - y1,
    ]


def build_coco_results(payload):
    """
    Convert retained accepted predictions to COCO result format.
    """
    results = []

    for image in payload["images"]:
        image_id = int(image["image_id"])

        for det in image.get("detections", []):
            score = float(det["score"])

            if not score > EXPECTED_ACCEPTANCE_THRESHOLD:
                raise RuntimeError(
                    f"Retained detection has score <= "
                    f"{EXPECTED_ACCEPTANCE_THRESHOLD}: {score}"
                )

            results.append(
                {
                    "image_id": image_id,
                    "category_id": int(det["category_id"]),
                    "bbox": xyxy_to_xywh(det["bbox_xyxy"]),
                    "score": score,
                }
            )

    return results


def evaluate_one_run(
    coco_gt: COCO,
    coco_results: list[dict],
):
    """
    Run COCOeval once and extract TP/FP/FN for:
      - each class
      - each IoU threshold
      - maxDets=100
      - area=all

    Special case:
      pycocotools.COCO.loadRes([]) raises IndexError for an empty
      detection list. For a run with zero accepted detections, the
      COCO-equivalent diagnostic result is deterministic:
        TP = 0
        FP = 0
        FN = number of non-ignored GT boxes for that class
        precision = undefined
        recall = 0
    """

    cat_ids = sorted(coco_gt.getCatIds())

    # ----------------------------------------------------------
    # Empty prediction set:
    # avoid pycocotools COCO.loadRes([]) IndexError.
    # ----------------------------------------------------------
    if len(coco_results) == 0:
        rows = []

        for cat_id in cat_ids:
            category = coco_gt.loadCats([cat_id])[0]
            class_name = str(category["name"])

            ann_ids = coco_gt.getAnnIds(
                imgIds=sorted(coco_gt.getImgIds()),
                catIds=[cat_id],
                iscrowd=None,
            )
            anns = coco_gt.loadAnns(ann_ids)

            # For this fixed validation file, normal bbox GTs are
            # non-ignored. Keep explicit filtering for robustness.
            valid_anns = [
                ann
                for ann in anns
                if int(ann.get("ignore", 0)) == 0
            ]

            gt_count = len(valid_anns)

            for iou_thr in IOU_THRESHOLDS:
                rows.append(
                    {
                        "category_id": int(cat_id),
                        "class_name": class_name,
                        "iou_threshold": round(
                            float(iou_thr),
                            2,
                        ),
                        "tp": 0,
                        "fp": 0,
                        "fn": int(gt_count),
                        "gt_count": int(gt_count),
                        "precision": "",
                        "recall": (
                            0.0
                            if gt_count > 0
                            else ""
                        ),
                        "prediction_count": 0,
                    }
                )

        return rows

    # ----------------------------------------------------------
    # Non-empty prediction set:
    # use normal COCOeval matching.
    # ----------------------------------------------------------
    coco_dt = coco_gt.loadRes(coco_results)

    evaluator = COCOeval(
        coco_gt,
        coco_dt,
        iouType="bbox",
    )

    evaluator.params.useCats = 1
    evaluator.params.catIds = cat_ids
    evaluator.params.imgIds = sorted(coco_gt.getImgIds())
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.params.iouThrs = IOU_THRESHOLDS.copy()
    evaluator.params.areaRng = [[0**2, 1e5**2]]
    evaluator.params.areaRngLbl = ["all"]

    evaluator.evaluate()

    rows = []

    # evalImgs ordering in COCOeval:
    # category-major × area-range × image
    eval_imgs = evaluator.evalImgs

    if len(eval_imgs) != (
        len(cat_ids)
        * len(evaluator.params.areaRng)
        * len(evaluator.params.imgIds)
    ):
        raise RuntimeError(
            "Unexpected COCOeval evalImgs dimensions"
        )

    image_count = len(evaluator.params.imgIds)
    area_count = len(evaluator.params.areaRng)

    for cat_index, cat_id in enumerate(cat_ids):
        category = coco_gt.loadCats([cat_id])[0]
        class_name = str(category["name"])

        start = (
            cat_index
            * area_count
            * image_count
        )
        end = start + image_count

        class_eval_imgs = eval_imgs[start:end]

        for thr_index, iou_thr in enumerate(
            IOU_THRESHOLDS
        ):
            tp = 0
            fp = 0
            fn = 0
            gt_total_nonignored = 0

            for ev in class_eval_imgs:
                if ev is None:
                    continue

                dt_matches = np.asarray(
                    ev["dtMatches"][thr_index]
                )
                gt_matches = np.asarray(
                    ev["gtMatches"][thr_index]
                )

                dt_ignore = np.asarray(
                    ev["dtIgnore"][thr_index],
                    dtype=bool,
                )

                gt_ignore = np.asarray(
                    ev["gtIgnore"],
                    dtype=bool,
                )

                valid_dt = ~dt_ignore
                valid_gt = ~gt_ignore

                tp += int(
                    np.logical_and(
                        dt_matches > 0,
                        valid_dt,
                    ).sum()
                )

                fp += int(
                    np.logical_and(
                        dt_matches == 0,
                        valid_dt,
                    ).sum()
                )

                fn += int(
                    np.logical_and(
                        gt_matches == 0,
                        valid_gt,
                    ).sum()
                )

                gt_total_nonignored += int(
                    valid_gt.sum()
                )

            if tp + fn != gt_total_nonignored:
                raise RuntimeError(
                    f"GT accounting mismatch for "
                    f"class={class_name}, "
                    f"IoU={iou_thr:.2f}: "
                    f"TP+FN={tp + fn}, "
                    f"GT={gt_total_nonignored}"
                )

            precision_den = tp + fp
            recall_den = tp + fn

            precision = (
                tp / precision_den
                if precision_den > 0
                else np.nan
            )

            recall = (
                tp / recall_den
                if recall_den > 0
                else np.nan
            )

            rows.append(
                {
                    "category_id": int(cat_id),
                    "class_name": class_name,
                    "iou_threshold": round(
                        float(iou_thr),
                        2,
                    ),
                    "tp": int(tp),
                    "fp": int(fp),
                    "fn": int(fn),
                    "gt_count": int(
                        gt_total_nonignored
                    ),
                    "precision": (
                        float(precision)
                        if not np.isnan(precision)
                        else ""
                    ),
                    "recall": (
                        float(recall)
                        if not np.isnan(recall)
                        else ""
                    ),
                    "prediction_count": int(
                        tp + fp
                    ),
                }
            )

    return rows

    evaluator = COCOeval(
        coco_gt,
        coco_dt,
        iouType="bbox",
    )

    evaluator.params.useCats = 1
    evaluator.params.catIds = sorted(coco_gt.getCatIds())
    evaluator.params.imgIds = sorted(coco_gt.getImgIds())
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.params.iouThrs = IOU_THRESHOLDS.copy()
    evaluator.params.areaRng = [[0**2, 1e5**2]]
    evaluator.params.areaRngLbl = ["all"]

    evaluator.evaluate()

    cat_ids = evaluator.params.catIds
    rows = []

    # evalImgs ordering in COCOeval:
    # category-major × area-range × image
    eval_imgs = evaluator.evalImgs

    if len(eval_imgs) != (
        len(cat_ids)
        * len(evaluator.params.areaRng)
        * len(evaluator.params.imgIds)
    ):
        raise RuntimeError(
            "Unexpected COCOeval evalImgs dimensions"
        )

    image_count = len(evaluator.params.imgIds)
    area_count = len(evaluator.params.areaRng)

    for cat_index, cat_id in enumerate(cat_ids):
        category = coco_gt.loadCats([cat_id])[0]
        class_name = str(category["name"])

        # With one area range, block start is:
        # cat_index * area_count * image_count
        start = cat_index * area_count * image_count
        end = start + image_count

        class_eval_imgs = eval_imgs[start:end]

        for thr_index, iou_thr in enumerate(IOU_THRESHOLDS):
            tp = 0
            fp = 0
            fn = 0
            gt_total_nonignored = 0

            for ev in class_eval_imgs:
                if ev is None:
                    continue

                dt_matches = np.asarray(
                    ev["dtMatches"][thr_index]
                )
                gt_matches = np.asarray(
                    ev["gtMatches"][thr_index]
                )

                dt_ignore = np.asarray(
                    ev["dtIgnore"][thr_index],
                    dtype=bool,
                )

                gt_ignore = np.asarray(
                    ev["gtIgnore"],
                    dtype=bool,
                )

                # COCO matching semantics:
                # dtMatches > 0 = matched GT id
                # dtIgnore = ignored detection
                # gtMatches > 0 = matched DT id
                # gtIgnore = ignored GT

                valid_dt = ~dt_ignore
                valid_gt = ~gt_ignore

                tp += int(
                    np.logical_and(
                        dt_matches > 0,
                        valid_dt,
                    ).sum()
                )

                fp += int(
                    np.logical_and(
                        dt_matches == 0,
                        valid_dt,
                    ).sum()
                )

                matched_valid_gt = np.logical_and(
                    gt_matches > 0,
                    valid_gt,
                )

                unmatched_valid_gt = np.logical_and(
                    gt_matches == 0,
                    valid_gt,
                )

                fn += int(
                    unmatched_valid_gt.sum()
                )

                gt_total_nonignored += int(
                    valid_gt.sum()
                )

            if tp + fn != gt_total_nonignored:
                raise RuntimeError(
                    f"GT accounting mismatch for "
                    f"class={class_name}, IoU={iou_thr:.2f}: "
                    f"TP+FN={tp+fn}, GT={gt_total_nonignored}"
                )

            precision_den = tp + fp
            recall_den = tp + fn

            precision = (
                tp / precision_den
                if precision_den > 0
                else np.nan
            )

            recall = (
                tp / recall_den
                if recall_den > 0
                else np.nan
            )

            rows.append(
                {
                    "category_id": int(cat_id),
                    "class_name": class_name,
                    "iou_threshold": round(
                        float(iou_thr),
                        2,
                    ),
                    "tp": int(tp),
                    "fp": int(fp),
                    "fn": int(fn),
                    "gt_count": int(
                        gt_total_nonignored
                    ),
                    "precision": (
                        float(precision)
                        if not np.isnan(precision)
                        else ""
                    ),
                    "recall": (
                        float(recall)
                        if not np.isnan(recall)
                        else ""
                    ),
                    "prediction_count": int(
                        tp + fp
                    ),
                }
            )

    return rows


def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not VAL_JSON.is_file():
        raise FileNotFoundError(
            f"Fixed validation GT missing: {VAL_JSON}"
        )

    pred_files = sorted(
        PRED_ROOT.rglob(
            "val_last_teacher_predictions.json"
        )
    )

    if len(pred_files) != EXPECTED_RUNS:
        raise RuntimeError(
            f"Expected {EXPECTED_RUNS} prediction files, "
            f"observed={len(pred_files)}"
        )

    coco_gt = COCO(str(VAL_JSON))

    if len(coco_gt.getImgIds()) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGES} validation images, "
            f"observed={len(coco_gt.getImgIds())}"
        )

    cat_ids = sorted(coco_gt.getCatIds())

    if cat_ids != list(
        range(1, EXPECTED_CLASSES + 1)
    ):
        raise RuntimeError(
            f"Expected category IDs 1..14, "
            f"observed={cat_ids}"
        )

    all_rows = []
    run_ids_seen = set()

    for index, path in enumerate(
        pred_files,
        start=1,
    ):
        attempt_dir = path.parent.parent
        run_dir = attempt_dir.parent

        run_id = run_dir.name
        attempt_id = attempt_dir.name

        match = RUN_RE.match(run_id)

        if not match:
            raise RuntimeError(
                f"Unexpected run_id: {run_id}"
            )

        if run_id in run_ids_seen:
            raise RuntimeError(
                f"Duplicate run_id: {run_id}"
            )

        run_ids_seen.add(run_id)

        arch_code = match.group(2)
        budget_code = match.group(3)
        seed_code = match.group(4)

        architecture = ARCH_MAP[
            arch_code
        ]
        budget = BUDGET_MAP[
            budget_code
        ]
        seed_index = int(seed_code)

        payload = read_json(path)

        threshold = float(
            payload.get(
                "classification_threshold"
            )
        )

        if abs(
            threshold
            - EXPECTED_ACCEPTANCE_THRESHOLD
        ) > 1e-12:
            raise RuntimeError(
                f"{run_id}: threshold mismatch "
                f"{threshold}"
            )

        comparator = payload.get(
            "acceptance_comparator"
        )

        if comparator != EXPECTED_COMPARATOR:
            raise RuntimeError(
                f"{run_id}: comparator mismatch "
                f"{comparator}"
            )

        images = payload.get("images")

        if not isinstance(
            images,
            list,
        ):
            raise RuntimeError(
                f"{run_id}: images is not list"
            )

        if len(images) != EXPECTED_IMAGES:
            raise RuntimeError(
                f"{run_id}: expected "
                f"{EXPECTED_IMAGES} images, "
                f"observed={len(images)}"
            )

        pred_image_ids = [
            int(x["image_id"])
            for x in images
        ]

        gt_image_ids = sorted(
            coco_gt.getImgIds()
        )

        if pred_image_ids != gt_image_ids:
            raise RuntimeError(
                f"{run_id}: prediction image IDs/order "
                "do not exactly match fixed validation"
            )

        coco_results = build_coco_results(
            payload
        )

        eval_rows = evaluate_one_run(
            coco_gt,
            coco_results,
        )

        expected_per_run_rows = (
            EXPECTED_CLASSES
            * len(IOU_THRESHOLDS)
        )

        if len(eval_rows) != expected_per_run_rows:
            raise RuntimeError(
                f"{run_id}: expected "
                f"{expected_per_run_rows} eval rows, "
                f"observed={len(eval_rows)}"
            )

        for row in eval_rows:
            row.update(
                {
                    "run_id": run_id,
                    "architecture": architecture,
                    "budget": budget,
                    "seed_index": seed_index,
                    "attempt_id": attempt_id,
                    "classification_threshold":
                        EXPECTED_ACCEPTANCE_THRESHOLD,
                    "acceptance_comparator":
                        EXPECTED_COMPARATOR,
                    "max_dets_per_image": 100,
                    "area_range": "all",
                    "matching_semantics":
                        "COCOeval_bbox_useCats1",
                }
            )

            all_rows.append(row)

        print(
            f"[{index:02d}/{EXPECTED_RUNS}] "
            f"{run_id} "
            f"predictions={len(coco_results)}"
        )

    expected_total_rows = (
        EXPECTED_RUNS
        * EXPECTED_CLASSES
        * len(IOU_THRESHOLDS)
    )

    if len(all_rows) != expected_total_rows:
        raise RuntimeError(
            f"Expected {expected_total_rows} rows, "
            f"observed={len(all_rows)}"
        )

    if len(run_ids_seen) != EXPECTED_RUNS:
        raise RuntimeError(
            f"Expected {EXPECTED_RUNS} unique runs, "
            f"observed={len(run_ids_seen)}"
        )

    fieldnames = [
        "run_id",
        "architecture",
        "budget",
        "seed_index",
        "attempt_id",
        "category_id",
        "class_name",
        "iou_threshold",
        "tp",
        "fp",
        "fn",
        "gt_count",
        "prediction_count",
        "precision",
        "recall",
        "classification_threshold",
        "acceptance_comparator",
        "max_dets_per_image",
        "area_range",
        "matching_semantics",
    ]

    with OUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        writer.writerows(
            sorted(
                all_rows,
                key=lambda x: (
                    x["architecture"],
                    x["budget"],
                    x["seed_index"],
                    x["category_id"],
                    x["iou_threshold"],
                ),
            )
        )

    # ----------------------------------------------------------
    # Final audit
    # ----------------------------------------------------------

    iou_values = sorted(
        {
            row["iou_threshold"]
            for row in all_rows
        }
    )

    if len(iou_values) != 10:
        raise RuntimeError(
            f"Expected 10 IoU thresholds, "
            f"observed={iou_values}"
        )

    # At IoU=0.50, every run × class prediction_count
    # should equal accepted pseudo count for that class.
    iou50_rows = [
        row
        for row in all_rows
        if abs(
            row["iou_threshold"] - 0.50
        ) < 1e-12
    ]

    if len(iou50_rows) != (
        EXPECTED_RUNS
        * EXPECTED_CLASSES
    ):
        raise RuntimeError(
            "IoU=0.50 row-count mismatch"
        )

    zero_prediction_rows = sum(
        row["prediction_count"] == 0
        for row in iou50_rows
    )

    undefined_precision_rows = sum(
        row["precision"] == ""
        for row in iou50_rows
    )

    if (
        zero_prediction_rows
        != undefined_precision_rows
    ):
        raise RuntimeError(
            "Undefined precision does not exactly "
            "match zero-prediction cells"
        )

    print("STATUS=PASS")
    print(
        f"OFFICIAL_SSL_RUNS="
        f"{len(run_ids_seen)}"
    )
    print(
        f"IOU_THRESHOLDS="
        f"{','.join(f'{x:.2f}' for x in iou_values)}"
    )
    print(
        f"OUTPUT_ROWS="
        f"{len(all_rows)}"
    )
    print(
        f"EXPECTED_OUTPUT_ROWS="
        f"{expected_total_rows}"
    )
    print(
        f"IOU50_ROWS="
        f"{len(iou50_rows)}"
    )
    print(
        f"IOU50_ZERO_PREDICTION_ROWS="
        f"{zero_prediction_rows}"
    )
    print(
        f"IOU50_UNDEFINED_PRECISION_ROWS="
        f"{undefined_precision_rows}"
    )
    print(
        f"OUTPUT={OUT_CSV}"
    )


if __name__ == "__main__":
    main()
