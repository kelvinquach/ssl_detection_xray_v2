#!/usr/bin/env python3

import argparse
import csv
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

CONFIG_ORDER = {
    "C0": 0,
    "C1": 1,
    "C2": 2,
    "C3": 3,
    "A0": 4,
    "A1": 5,
    "A2": 6,
    "F0": 7,
    "F1": 8,
}

METRICS = [
    ("validation_bbox_mAP_50_95", "bbox_mAP_50_95"),
    ("validation_AP50", "AP50"),
    ("validation_AP75", "AP75"),
]

OUTPUT_FIELDS = [
    "variant_id",
    "ablation_run_id",
    "main_run_id",
    "family",
    "level",
    "seed_index",
    "training_seed",
    "metric_name",
    "variant_value",
    "main_same_seed_value",
    "paired_difference",
    "variant_selected_update",
    "main_selected_update",
    "variant_validation_checkpoint_sha256",
    "main_validation_checkpoint_sha256",
    "variant_validation_best_path",
    "main_validation_best_path",
    "pair_valid",
    "formal_hypothesis_test",
    "representation_authority",
]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def norm(value):
    if value is None:
        return ""
    return str(value).strip().strip('"')


def parse_bool(value):
    value = norm(value).lower()
    require(
        value in {"true", "false"},
        f"INVALID_BOOLEAN:{value!r}",
    )
    return value == "true"


def parse_decimal(value, label):
    text = norm(value)
    try:
        return Decimal(text)
    except InvalidOperation as exc:
        raise RuntimeError(
            f"{label}:NON_NUMERIC:{text!r}"
        ) from exc


def decimal_text(value):
    return format(value, "f")


def require_sha(value, label):
    value = norm(value)
    require(
        SHA256_RE.fullmatch(value) is not None,
        f"{label}:INVALID_SHA256:{value!r}",
    )
    return value


def load_csv(path):
    path = Path(path)
    require(path.is_file(), f"MISSING_FILE:{path}")
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def load_json(path):
    import json

    path = Path(path)
    require(path.is_file(), f"MISSING_FILE:{path}")
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(handle)

    require(
        isinstance(payload, dict),
        f"JSON_NOT_OBJECT:{path}",
    )
    return payload


def selected_validation_row(best_path, history_path, label):
    best = load_json(best_path)

    require(
        norm(best.get("selection_model")) == "EMA_TEACHER",
        f"{label}:SELECTION_MODEL_NOT_EMA_TEACHER",
    )
    require(
        norm(best.get("selection_metric"))
        == "bbox_mAP_50_95",
        f"{label}:SELECTION_METRIC_INVALID",
    )

    selected_update = norm(best.get("selected_update"))
    require(
        selected_update != "",
        f"{label}:SELECTED_UPDATE_MISSING",
    )

    history = load_csv(history_path)

    selected = [
        row
        for row in history
        if norm(row.get("optimizer_update"))
        == selected_update
    ]

    require(
        len(selected) == 1,
        f"{label}:SELECTED_HISTORY_ROWS={len(selected)}",
    )

    row = selected[0]

    require(
        norm(row.get("model_role")) == "EMA_TEACHER",
        f"{label}:HISTORY_MODEL_ROLE_INVALID",
    )

    bbox_map = parse_decimal(
        row.get("bbox_mAP_50_95"),
        f"{label}:bbox_mAP_50_95",
    )
    ap50 = parse_decimal(
        row.get("AP50"),
        f"{label}:AP50",
    )
    ap75 = parse_decimal(
        row.get("AP75"),
        f"{label}:AP75",
    )

    selected_metric = parse_decimal(
        best.get("selected_metric_value"),
        f"{label}:selected_metric_value",
    )

    require(
        bbox_map == selected_metric,
        f"{label}:BEST_HISTORY_MAP_MISMATCH:"
        f"{bbox_map}!={selected_metric}",
    )

    checkpoint_sha = require_sha(
        best.get("checkpoint_sha256"),
        f"{label}:CHECKPOINT",
    )

    return {
        "selected_update": selected_update,
        "checkpoint_sha256": checkpoint_sha,
        "bbox_mAP_50_95": bbox_map,
        "AP50": ap50,
        "AP75": ap75,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo",
        required=True,
    )
    parser.add_argument(
        "--pair-inventory",
        required=True,
    )
    parser.add_argument(
        "--run-inventory",
        required=True,
    )
    parser.add_argument(
        "--main-map",
        required=True,
    )
    parser.add_argument(
        "--variant-root",
        required=True,
    )
    parser.add_argument(
        "--main-root",
        required=True,
    )
    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    variant_root = Path(args.variant_root).resolve()
    main_root = Path(args.main_root).resolve()

    pairs = load_csv(args.pair_inventory)
    run_inventory = load_csv(args.run_inventory)
    main_map = load_csv(args.main_map)

    require(
        len(pairs) == 90,
        f"PAIR_ROWS={len(pairs)}",
    )
    require(
        len(run_inventory) == 90,
        f"RUN_INVENTORY_ROWS={len(run_inventory)}",
    )
    require(
        len(main_map) == 10,
        f"MAIN_MAP_ROWS={len(main_map)}",
    )

    require(
        len({
            norm(row["ablation_run_id"])
            for row in pairs
        }) == 90,
        "PAIR_ABLATION_RUN_IDS_NOT_UNIQUE",
    )

    run_by_id = {}

    for row in run_inventory:
        run_id = norm(row["ablation_run_id"])

        require(
            run_id not in run_by_id,
            f"DUPLICATE_RUN_INVENTORY:{run_id}",
        )

        config_id = norm(
            row["ablation_config_id"]
        ).upper()

        require(
            config_id in CONFIG_ORDER,
            f"{run_id}:INVALID_VARIANT_ID:{config_id}",
        )

        run_by_id[run_id] = row

    main_by_seed = {}

    for row in main_map:
        seed_index = int(norm(row["seed_index"]))
        training_seed = int(
            norm(row["training_seed"])
        )
        run_id = norm(row["main_run_id"])

        key = (
            seed_index,
            training_seed,
            run_id,
        )

        require(
            key not in main_by_seed,
            f"DUPLICATE_MAIN_REFERENCE:{key}",
        )

        require(
            1 <= seed_index <= 10,
            f"MAIN_SEED_INDEX_INVALID:{seed_index}",
        )

        require_sha(
            row["main_checkpoint_sha256"],
            f"{run_id}:MAIN_MAP_CHECKPOINT",
        )

        main_by_seed[key] = row

    output_rows = []

    for pair in pairs:
        ablation_run_id = norm(
            pair["ablation_run_id"]
        )
        main_run_id = norm(
            pair["main_run_id"]
        )
        family = norm(pair["family"])
        level = norm(pair["level"])
        seed_index = int(
            norm(pair["seed_index"])
        )
        training_seed = int(
            norm(pair["training_seed"])
        )

        require(
            parse_bool(pair["pair_valid"]),
            f"{ablation_run_id}:PAIR_NOT_VALID",
        )
        require(
            parse_bool(
                pair[
                    "config_diff_only_target_factor"
                ]
            ),
            f"{ablation_run_id}:OFAT_PAIR_INVALID",
        )

        require(
            ablation_run_id in run_by_id,
            f"{ablation_run_id}:RUN_INVENTORY_MISSING",
        )

        design = run_by_id[ablation_run_id]

        variant_id = norm(
            design["ablation_config_id"]
        ).upper()

        require(
            norm(design["family"]) == family,
            f"{ablation_run_id}:FAMILY_MISMATCH",
        )
        require(
            norm(design["level"]) == level,
            f"{ablation_run_id}:LEVEL_MISMATCH",
        )
        require(
            int(norm(design["seed_index"]))
            == seed_index,
            f"{ablation_run_id}:SEED_INDEX_MISMATCH",
        )
        require(
            int(norm(design["training_seed"]))
            == training_seed,
            f"{ablation_run_id}:TRAINING_SEED_MISMATCH",
        )

        main_key = (
            seed_index,
            training_seed,
            main_run_id,
        )

        require(
            main_key in main_by_seed,
            f"{ablation_run_id}:MAIN_REFERENCE_MISSING",
        )

        main_reference = main_by_seed[main_key]

        # -----------------------------------------------------------
        # Variant: use canonical attempt from logical_run_status.json.
        # -----------------------------------------------------------
        variant_status_path = (
            variant_root
            / ablation_run_id
            / "logical_run_status.json"
        )

        variant_status = load_json(
            variant_status_path
        )

        require(
            norm(
                variant_status.get(
                    "training_status"
                )
            ) == "CLOSED_VERIFIED",
            f"{ablation_run_id}:STATUS_INVALID",
        )
        require(
            variant_status.get(
                "training_completed"
            ) is True,
            f"{ablation_run_id}:NOT_COMPLETED",
        )
        require(
            variant_status.get(
                "training_valid"
            ) is True,
            f"{ablation_run_id}:NOT_VALID",
        )

        variant_attempt = norm(
            variant_status.get("attempt_used")
        )

        require(
            re.fullmatch(
                r"attempt_\d{3}",
                variant_attempt,
            )
            is not None,
            f"{ablation_run_id}:ATTEMPT_INVALID",
        )

        variant_dir = (
            variant_root
            / ablation_run_id
            / variant_attempt
        )

        variant_best_local = (
            variant_dir
            / "metrics"
            / "validation"
            / "validation_best_teacher.json"
        )

        variant_history_local = (
            variant_dir
            / "metrics"
            / "validation"
            / "validation_history.csv"
        )

        variant_metrics = selected_validation_row(
            variant_best_local,
            variant_history_local,
            ablation_run_id,
        )

        # -----------------------------------------------------------
        # Main: frozen official R50-B10 same-seed reference.
        # Main was not retrained for ablation.
        # -----------------------------------------------------------
        main_attempt = "attempt_001"

        main_dir = (
            main_root
            / main_run_id
            / main_attempt
        )

        main_best_local = (
            main_dir
            / "metrics"
            / "validation"
            / "validation_best_teacher.json"
        )

        main_history_local = (
            main_dir
            / "metrics"
            / "validation"
            / "validation_history.csv"
        )

        main_metrics = selected_validation_row(
            main_best_local,
            main_history_local,
            main_run_id,
        )

        expected_main_sha = require_sha(
            main_reference[
                "main_checkpoint_sha256"
            ],
            f"{main_run_id}:MAIN_MAP_CHECKPOINT",
        )

        require(
            main_metrics["checkpoint_sha256"]
            == expected_main_sha,
            f"{ablation_run_id}:"
            "MAIN_CHECKPOINT_SHA_MISMATCH",
        )

        canonical_variant_best_path = (
            "artifacts/runs/ablation/"
            f"{ablation_run_id}/"
            f"{variant_attempt}/"
            "metrics/validation/"
            "validation_best_teacher.json"
        )

        canonical_main_best_path = norm(
            main_reference[
                "main_validation_metrics_path"
            ]
        )

        require(
            canonical_main_best_path
            == (
                "artifacts/runs/official/"
                f"{main_run_id}/"
                "attempt_001/"
                "metrics/validation/"
                "validation_best_teacher.json"
            ),
            f"{main_run_id}:"
            "MAIN_VALIDATION_PATH_UNEXPECTED",
        )

        for metric_name, history_field in METRICS:
            variant_value = variant_metrics[
                history_field
            ]
            main_value = main_metrics[
                history_field
            ]

            paired_difference = (
                variant_value - main_value
            )

            output_rows.append({
                "variant_id":
                    variant_id,
                "ablation_run_id":
                    ablation_run_id,
                "main_run_id":
                    main_run_id,
                "family":
                    family,
                "level":
                    level,
                "seed_index":
                    str(seed_index),
                "training_seed":
                    str(training_seed),
                "metric_name":
                    metric_name,
                "variant_value":
                    decimal_text(variant_value),
                "main_same_seed_value":
                    decimal_text(main_value),
                "paired_difference":
                    decimal_text(
                        paired_difference
                    ),
                "variant_selected_update":
                    variant_metrics[
                        "selected_update"
                    ],
                "main_selected_update":
                    main_metrics[
                        "selected_update"
                    ],
                "variant_validation_checkpoint_sha256":
                    variant_metrics[
                        "checkpoint_sha256"
                    ],
                "main_validation_checkpoint_sha256":
                    main_metrics[
                        "checkpoint_sha256"
                    ],
                "variant_validation_best_path":
                    canonical_variant_best_path,
                "main_validation_best_path":
                    canonical_main_best_path,
                "pair_valid":
                    "true",
                "formal_hypothesis_test":
                    "false",
                "representation_authority":
                    "S10_SEED_LEVEL_PAIRED_DETECTION_EFFECT",
            })

    require(
        len(output_rows) == 270,
        f"OUTPUT_ROWS={len(output_rows)}",
    )

    counts_by_metric = {
        metric_name: sum(
            row["metric_name"] == metric_name
            for row in output_rows
        )
        for metric_name, _ in METRICS
    }

    require(
        all(
            count == 90
            for count in counts_by_metric.values()
        ),
        f"METRIC_COUNTS_INVALID:{counts_by_metric}",
    )

    require(
        len({
            (
                row["variant_id"],
                row["seed_index"],
                row["training_seed"],
                row["metric_name"],
            )
            for row in output_rows
        }) == 270,
        "OUTPUT_KEY_NOT_UNIQUE",
    )

    require(
        {
            row["variant_id"]
            for row in output_rows
        } == set(CONFIG_ORDER),
        "VARIANT_SET_INVALID",
    )

    output_rows.sort(
        key=lambda row: (
            CONFIG_ORDER[row["variant_id"]],
            int(row["seed_index"]),
            [
                name
                for name, _ in METRICS
            ].index(row["metric_name"]),
        )
    )

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_FIELDS,
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(
        "=== S10.1.08 ABLATION EFFECTS ==="
    )
    print(
        f"OUTPUT_ROWS={len(output_rows)}"
    )
    print(
        f"COLUMNS={len(OUTPUT_FIELDS)}"
    )
    print("PAIRS=90")
    print("METRICS_PER_PAIR=3")
    print(
        "MAP_ROWS="
        f"{counts_by_metric['validation_bbox_mAP_50_95']}"
    )
    print(
        "AP50_ROWS="
        f"{counts_by_metric['validation_AP50']}"
    )
    print(
        "AP75_ROWS="
        f"{counts_by_metric['validation_AP75']}"
    )
    print(
        "PAIRED_DIFFERENCE="
        "VARIANT_MINUS_MAIN_SAME_SEED"
    )
    print(
        "FORMAL_HYPOTHESIS_TEST=FALSE"
    )
    print(
        "QPSEUDO_INCLUDED=FALSE"
    )
    print(
        "FINAL_TEST_USED=FALSE"
    )
    print(
        "MAIN_RETRAINED=FALSE"
    )
    print(
        "S10_1_08_STATUS=PASS"
    )


if __name__ == "__main__":
    main()