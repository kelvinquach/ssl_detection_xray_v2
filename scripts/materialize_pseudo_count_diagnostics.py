from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from statistics import median

REPO = Path(r"D:\ssl_detection_xray_v2")
PRED_ROOT = REPO / "_tmp_qpseudo_count_audit" / "predictions"
OUT_ROOT = REPO / "analysis" / "pseudo_count_diagnostics"

EXPECTED_RUNS = 80
EXPECTED_CLASSES = 14
EXPECTED_IMAGES_PER_RUN = 734
EXPECTED_THRESHOLD = 0.90
EXPECTED_COMPARATOR = "STRICT_GREATER"

CLASSES = {
    1: "Aortic enlargement",
    2: "Atelectasis",
    3: "Calcification",
    4: "Cardiomegaly",
    5: "Consolidation",
    6: "ILD",
    7: "Infiltration",
    8: "Lung Opacity",
    9: "Nodule/Mass",
    10: "Other lesion",
    11: "Pleural effusion",
    12: "Pleural thickening",
    13: "Pneumothorax",
    14: "Pulmonary fibrosis",
}

RUN_RE = re.compile(
    r"^(OFF_SSL_(R50|SWIN)_B(01|05|10|20)_S"
    r"(01|02|03|04|05|06|07|08|09|10))$"
)

BUDGET_MAP = {
    "01": "1%",
    "05": "5%",
    "10": "10%",
    "20": "20%",
}

ARCH_MAP = {
    "R50": "R50-FPN",
    "SWIN": "Swin-T-FPN",
}


def read_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(
        PRED_ROOT.rglob("val_last_teacher_predictions.json")
    )

    if len(pred_files) != EXPECTED_RUNS:
        raise RuntimeError(
            f"Expected {EXPECTED_RUNS} prediction files, "
            f"observed={len(pred_files)}"
        )

    class_rows = []
    run_rows = []
    observed_run_ids = set()

    for path in pred_files:
        # Expected path:
        # .../OFF_SSL_.../attempt_NNN/pseudo/val_last_teacher_predictions.json
        pseudo_dir = path.parent
        attempt_dir = pseudo_dir.parent
        run_dir = attempt_dir.parent

        run_id = run_dir.name
        attempt_id = attempt_dir.name

        m = RUN_RE.match(run_id)
        if not m:
            raise RuntimeError(f"Unexpected run_id: {run_id}")

        if run_id in observed_run_ids:
            raise RuntimeError(f"Duplicate canonical run_id: {run_id}")
        observed_run_ids.add(run_id)

        arch_code = m.group(2)
        budget_code = m.group(3)
        seed_code = m.group(4)

        architecture = ARCH_MAP[arch_code]
        budget = BUDGET_MAP[budget_code]
        seed_index = int(seed_code)

        payload = read_json(path)

        if float(payload.get("classification_threshold")) != EXPECTED_THRESHOLD:
            raise RuntimeError(
                f"{run_id}: threshold mismatch: "
                f"{payload.get('classification_threshold')}"
            )

        if payload.get("acceptance_comparator") != EXPECTED_COMPARATOR:
            raise RuntimeError(
                f"{run_id}: comparator mismatch: "
                f"{payload.get('acceptance_comparator')}"
            )

        if payload.get("pseudo_set") != "accepted_rcnn_classification":
            raise RuntimeError(
                f"{run_id}: unexpected pseudo_set="
                f"{payload.get('pseudo_set')}"
            )

        if payload.get("second_nms") is not False:
            raise RuntimeError(f"{run_id}: second_nms must be False")

        if payload.get("topk_truncation") is not False:
            raise RuntimeError(f"{run_id}: topk_truncation must be False")

        images = payload.get("images")
        if not isinstance(images, list):
            raise RuntimeError(f"{run_id}: images is not a list")

        if len(images) != EXPECTED_IMAGES_PER_RUN:
            raise RuntimeError(
                f"{run_id}: expected {EXPECTED_IMAGES_PER_RUN} images, "
                f"observed={len(images)}"
            )

        class_counts = Counter()
        accepted_scores = []
        accepted_total = 0
        detector_native_total = 0
        images_with_accepted = 0

        for image in images:
            detections = image.get("detections", [])
            accepted_count = int(image.get("accepted_count", -1))
            native_count = int(image.get("detector_native_count", -1))

            if accepted_count != len(detections):
                raise RuntimeError(
                    f"{run_id}, image_id={image.get('image_id')}: "
                    f"accepted_count={accepted_count}, "
                    f"detections={len(detections)}"
                )

            if native_count < accepted_count:
                raise RuntimeError(
                    f"{run_id}, image_id={image.get('image_id')}: "
                    "detector_native_count < accepted_count"
                )

            accepted_total += accepted_count
            detector_native_total += native_count

            if accepted_count > 0:
                images_with_accepted += 1

            for det in detections:
                category_id = int(det["category_id"])
                score = float(det["score"])

                if category_id not in CLASSES:
                    raise RuntimeError(
                        f"{run_id}: invalid category_id={category_id}"
                    )

                # Frozen Q_pseudo semantics: STRICT_GREATER 0.90.
                if not score > EXPECTED_THRESHOLD:
                    raise RuntimeError(
                        f"{run_id}: retained score does not satisfy "
                        f"> {EXPECTED_THRESHOLD}: {score}"
                    )

                class_counts[category_id] += 1
                accepted_scores.append(score)

        if sum(class_counts.values()) != accepted_total:
            raise RuntimeError(
                f"{run_id}: class-count sum != accepted_total"
            )

        # Always emit all 14 classes, including zero-count classes.
        for category_id in range(1, EXPECTED_CLASSES + 1):
            class_rows.append(
                {
                    "run_id": run_id,
                    "architecture": architecture,
                    "budget": budget,
                    "seed_index": seed_index,
                    "attempt_id": attempt_id,
                    "category_id": category_id,
                    "class_name": CLASSES[category_id],
                    "classification_threshold": EXPECTED_THRESHOLD,
                    "acceptance_comparator": EXPECTED_COMPARATOR,
                    "accepted_pseudo_count": class_counts[category_id],
                }
            )

        run_rows.append(
            {
                "run_id": run_id,
                "architecture": architecture,
                "budget": budget,
                "seed_index": seed_index,
                "attempt_id": attempt_id,
                "classification_threshold": EXPECTED_THRESHOLD,
                "acceptance_comparator": EXPECTED_COMPARATOR,
                "validation_image_count": len(images),
                "accepted_pseudo_total": accepted_total,
                "images_with_accepted_pseudo": images_with_accepted,
                "detector_native_total": detector_native_total,
                "accepted_score_min": (
                    min(accepted_scores) if accepted_scores else ""
                ),
                "accepted_score_median": (
                    median(accepted_scores) if accepted_scores else ""
                ),
                "accepted_score_max": (
                    max(accepted_scores) if accepted_scores else ""
                ),
            }
        )

    if len(observed_run_ids) != EXPECTED_RUNS:
        raise RuntimeError(
            f"Unique run count mismatch: {len(observed_run_ids)}"
        )

    if len(class_rows) != EXPECTED_RUNS * EXPECTED_CLASSES:
        raise RuntimeError(
            f"Expected {EXPECTED_RUNS * EXPECTED_CLASSES} class rows, "
            f"observed={len(class_rows)}"
        )

    class_csv = OUT_ROOT / "pseudo_count_seed_class.csv"
    run_csv = OUT_ROOT / "pseudo_count_run_summary.csv"

    with class_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "architecture",
                "budget",
                "seed_index",
                "attempt_id",
                "category_id",
                "class_name",
                "classification_threshold",
                "acceptance_comparator",
                "accepted_pseudo_count",
            ],
        )
        writer.writeheader()
        writer.writerows(
            sorted(
                class_rows,
                key=lambda x: (
                    x["architecture"],
                    x["budget"],
                    x["seed_index"],
                    x["category_id"],
                ),
            )
        )

    with run_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "architecture",
                "budget",
                "seed_index",
                "attempt_id",
                "classification_threshold",
                "acceptance_comparator",
                "validation_image_count",
                "accepted_pseudo_total",
                "images_with_accepted_pseudo",
                "detector_native_total",
                "accepted_score_min",
                "accepted_score_median",
                "accepted_score_max",
            ],
        )
        writer.writeheader()
        writer.writerows(
            sorted(
                run_rows,
                key=lambda x: (
                    x["architecture"],
                    x["budget"],
                    x["seed_index"],
                ),
            )
        )

    zero_class_rows = sum(
        row["accepted_pseudo_count"] == 0
        for row in class_rows
    )

    print("STATUS=PASS")
    print(f"OFFICIAL_SSL_RUNS={len(observed_run_ids)}")
    print(f"RUN_SUMMARY_ROWS={len(run_rows)}")
    print(f"SEED_CLASS_ROWS={len(class_rows)}")
    print(f"ZERO_COUNT_CLASS_ROWS={zero_class_rows}")
    print(f"CLASS_CSV={class_csv}")
    print(f"RUN_CSV={run_csv}")


if __name__ == "__main__":
    main()