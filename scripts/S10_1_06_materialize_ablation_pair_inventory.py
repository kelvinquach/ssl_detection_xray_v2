#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED_ROWS = 90

OUTPUT_FIELDS = [
    "ablation_run_id",
    "main_run_id",
    "family",
    "level",
    "seed_index",
    "training_seed",
    "config_diff_only_target_factor",
    "pair_valid",
]

C_LEVELS = {
    "C0": 0.50,
    "C1": 0.60,
    "C2": 0.70,
    "C3": 0.80,
}

A_LEVELS = {
    "A0": (False, False),
    "A1": (True, False),
    "A2": (False, True),
}

F_LEVELS = {"F0", "F1"}

FROZEN_FIELDS = [
    "scientific_source_sha256",
    "implementation_contract_sha256",
    "artifact_contract_sha256",
    "dataset_manifest_sha256",
    "train_split_sha256",
    "val_split_sha256",
    "test_split_sha256",
    "labeled_manifest_sha256",
    "unlabeled_manifest_sha256",
    "preprocessing_config_sha256",
    "labeled_pipeline_sha256",
    "optimizer_recipe_sha256",
    "evaluator_config_sha256",
    "training_protocol_sha256",
    "checkpoint_policy_sha256",
    "ssl_protocol_sha256",
    "seed_protocol_sha256",
    "main_frozen_executable_config_path",
    "main_frozen_executable_config_sha256",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "total_optimizer_updates",
    "validation_interval_updates",
    "effective_labeled_batch",
    "effective_unlabeled_batch",
    "amp",
    "gradient_clipping",
    "checkpoint_selection_metric",
]


def require(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def norm(x):
    if x is None:
        return ""
    return str(x).strip().strip('"')


def same_float(a, b, tol=1e-12):
    try:
        return math.isclose(
            float(a),
            float(b),
            rel_tol=0.0,
            abs_tol=tol,
        )
    except Exception:
        return False


def load_csv(path):
    path = Path(path)
    require(path.is_file(), f"MISSING_FILE: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_json(path):
    path = Path(path)
    require(path.is_file(), f"MISSING_FILE: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    require(isinstance(payload, dict), f"JSON_NOT_OBJECT: {path}")
    return payload


def validate_ofat(run_id, config_id, family, eff):
    require(
        isinstance(eff, dict),
        f"EFFECTIVE_SCIENTIFIC_FIELDS_INVALID: {run_id}",
    )
    require(
        norm(eff.get("ablation_family")).upper() == family,
        f"EFFECTIVE_FAMILY_MISMATCH: {run_id}",
    )
    require(
        eff.get("ofat_effective_semantics_valid") is True,
        f"OFAT_EFFECTIVE_SEMANTICS_INVALID: {run_id}",
    )
    require(
        eff.get("validation_only") is True,
        f"EFFECTIVE_VALIDATION_ONLY_INVALID: {run_id}",
    )
    require(
        eff.get("final_test_eligible") is False,
        f"EFFECTIVE_FINAL_TEST_ELIGIBLE_INVALID: {run_id}",
    )
    require(
        same_float(eff.get("initial_pseudo_score"), 0.50),
        f"INITIAL_PSEUDO_SCORE_CHANGED: {run_id}",
    )
    require(
        same_float(eff.get("rpn_pseudo_threshold"), 0.90),
        f"RPN_PSEUDO_THRESHOLD_CHANGED: {run_id}",
    )

    if family == "CONFIDENCE_THRESHOLD":
        require(
            config_id in C_LEVELS,
            f"INVALID_CONFIDENCE_CONFIG: {run_id}",
        )
        require(
            same_float(
                eff.get("rcnn_classification_threshold"),
                C_LEVELS[config_id],
            ),
            f"CONFIDENCE_TARGET_MISMATCH: {run_id}",
        )
        require(
            norm(eff.get("augmentation_variant")).upper() == "A3",
            f"CONFIDENCE_CHANGED_AUGMENTATION: {run_id}",
        )
        require(
            norm(eff.get("regression_filter_variant")).upper() == "F2",
            f"CONFIDENCE_CHANGED_REGRESSION: {run_id}",
        )

    elif family == "STRONG_PHOTOMETRIC_AUGMENTATION":
        require(
            config_id in A_LEVELS,
            f"INVALID_AUGMENTATION_CONFIG: {run_id}",
        )
        require(
            same_float(
                eff.get("rcnn_classification_threshold"),
                0.90,
            ),
            f"AUGMENTATION_CHANGED_CONFIDENCE: {run_id}",
        )
        require(
            norm(eff.get("augmentation_variant")).upper() == config_id,
            f"AUGMENTATION_TARGET_MISMATCH: {run_id}",
        )

        expected_brightness, expected_contrast = A_LEVELS[config_id]

        require(
            eff.get("brightness_enabled") is expected_brightness,
            f"AUGMENTATION_BRIGHTNESS_MISMATCH: {run_id}",
        )
        require(
            eff.get("contrast_enabled") is expected_contrast,
            f"AUGMENTATION_CONTRAST_MISMATCH: {run_id}",
        )
        require(
            norm(eff.get("regression_filter_variant")).upper() == "F2",
            f"AUGMENTATION_CHANGED_REGRESSION: {run_id}",
        )

    elif family == "REGRESSION_PSEUDO_LABEL_FILTERING":
        require(
            config_id in F_LEVELS,
            f"INVALID_REGRESSION_CONFIG: {run_id}",
        )
        require(
            same_float(
                eff.get("rcnn_classification_threshold"),
                0.90,
            ),
            f"REGRESSION_CHANGED_CONFIDENCE: {run_id}",
        )
        require(
            norm(eff.get("augmentation_variant")).upper() == "A3",
            f"REGRESSION_CHANGED_AUGMENTATION: {run_id}",
        )
        require(
            norm(eff.get("regression_filter_variant")).upper()
            == config_id,
            f"REGRESSION_TARGET_MISMATCH: {run_id}",
        )

        if config_id == "F0":
            require(
                norm(eff.get("regression_filter_rule"))
                == "NO_ADDITIONAL_REGRESSION_RELIABILITY_FILTER",
                f"F0_RULE_MISMATCH: {run_id}",
            )
            require(
                eff.get("regression_score_threshold") is None,
                f"F0_SCORE_THRESHOLD_INVALID: {run_id}",
            )
            require(
                eff.get("regression_uncertainty_threshold") is None,
                f"F0_UNCERTAINTY_THRESHOLD_INVALID: {run_id}",
            )

        elif config_id == "F1":
            require(
                norm(eff.get("regression_filter_rule"))
                == "REGRESSION_SCORE_GT_THRESHOLD",
                f"F1_RULE_MISMATCH: {run_id}",
            )
            require(
                same_float(
                    eff.get("regression_score_threshold"),
                    0.90,
                ),
                f"F1_SCORE_THRESHOLD_MISMATCH: {run_id}",
            )
            require(
                eff.get("regression_uncertainty_threshold") is None,
                f"F1_UNCERTAINTY_THRESHOLD_INVALID: {run_id}",
            )

    else:
        raise RuntimeError(
            f"UNKNOWN_ABLATION_FAMILY: {run_id}: {family}"
        )

    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation-run-inventory", required=True)
    ap.add_argument("--main-reference-map", required=True)
    ap.add_argument("--ablation-root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    inventory = load_csv(args.ablation_run_inventory)
    main_rows = load_csv(args.main_reference_map)
    root = Path(args.ablation_root)

    require(
        len(inventory) == EXPECTED_ROWS,
        f"ABLATION_INVENTORY_ROWS: expected=90 observed={len(inventory)}",
    )
    require(
        len(main_rows) == 10,
        f"MAIN_REFERENCE_ROWS: expected=10 observed={len(main_rows)}",
    )

    main_by_seed = {}
    for r in main_rows:
        seed_index = int(norm(r["seed_index"]))
        require(
            seed_index not in main_by_seed,
            f"DUPLICATE_MAIN_SEED_INDEX: {seed_index}",
        )
        main_by_seed[seed_index] = {
            "training_seed": int(norm(r["training_seed"])),
            "main_run_id": norm(r["main_run_id"]),
        }

    frozen_values = defaultdict(set)
    attempt_counts = Counter()
    family_counts = Counter()
    config_counts = Counter()
    output_rows = []

    for row in inventory:
        run_id = norm(row["ablation_run_id"])
        config_id = norm(row["ablation_config_id"]).upper()
        family = norm(row["family"]).upper()
        level = norm(row["level"]).upper()
        seed_index = int(norm(row["seed_index"]))
        training_seed = int(norm(row["training_seed"]))
        main_run_id = norm(row["main_run_id"])

        require(
            re.fullmatch(r"ABL_R50_B10_.+_S\d{2}", run_id) is not None,
            f"INVALID_ABLATION_RUN_ID: {run_id}",
        )
        require(
            norm(row["condition_role"]) == "ABLATION",
            f"CONDITION_ROLE_MISMATCH: {run_id}",
        )
        require(
            norm(row["method"]) == "SSL",
            f"METHOD_MISMATCH: {run_id}",
        )
        require(
            norm(row["architecture"]) == "R50-FPN",
            f"ARCHITECTURE_MISMATCH: {run_id}",
        )
        require(
            norm(row["budget"]) == "10pct",
            f"BUDGET_MISMATCH: {run_id}",
        )
        require(
            norm(row["is_main"]).lower() == "false",
            f"NON_MAIN_FLAG_INVALID: {run_id}",
        )
        require(
            norm(row["validation_only"]).lower() == "true",
            f"INVENTORY_VALIDATION_ONLY_INVALID: {run_id}",
        )
        require(
            norm(row["final_test_eligible"]).lower() == "false",
            f"INVENTORY_FINAL_TEST_ELIGIBLE_INVALID: {run_id}",
        )
        require(
            norm(row["test_access_authorized"]).lower() == "false",
            f"INVENTORY_TEST_ACCESS_INVALID: {run_id}",
        )

        require(
            seed_index in main_by_seed,
            f"MAIN_REFERENCE_SEED_MISSING: {run_id}",
        )
        ref = main_by_seed[seed_index]

        require(
            ref["training_seed"] == training_seed,
            f"MAIN_REFERENCE_TRAINING_SEED_MISMATCH: {run_id}",
        )
        require(
            ref["main_run_id"] == main_run_id,
            f"MAIN_REFERENCE_RUN_ID_MISMATCH: {run_id}",
        )
        require(
            main_run_id == f"OFF_SSL_R50_B10_S{seed_index:02d}",
            f"SAME_SEED_MAIN_BINDING_INVALID: {run_id}",
        )

        status_path = root / run_id / "logical_run_status.json"
        status = load_json(status_path)

        require(
            status.get("training_status") == "CLOSED_VERIFIED",
            f"LOGICAL_STATUS_NOT_CLOSED_VERIFIED: {run_id}",
        )
        require(
            status.get("training_completed") is True,
            f"TRAINING_NOT_COMPLETED: {run_id}",
        )
        require(
            status.get("training_valid") is True,
            f"TRAINING_NOT_VALID: {run_id}",
        )
        require(
            status.get("validation_only") is True,
            f"STATUS_VALIDATION_ONLY_INVALID: {run_id}",
        )
        require(
            status.get("eligible_for_final_test") is False,
            f"STATUS_FINAL_TEST_ELIGIBLE_INVALID: {run_id}",
        )
        require(
            status.get("test_access_authorized") is False,
            f"STATUS_TEST_ACCESS_INVALID: {run_id}",
        )
        require(
            status.get("final_test_accessed") is False,
            f"STATUS_FINAL_TEST_ACCESSED: {run_id}",
        )

        attempt = norm(status.get("attempt_used"))
        require(
            re.fullmatch(r"attempt_\d{3}", attempt) is not None,
            f"INVALID_CANONICAL_ATTEMPT: {run_id}",
        )
        attempt_counts[attempt] += 1

        manifest = load_json(
            root / run_id / attempt / "run_manifest.json"
        )

        expected_pairs = {
            "run_id": run_id,
            "run_type": "ABLATION",
            "condition_role": "ABLATION",
            "method": "SSL",
            "architecture": "R50-FPN",
            "budget": "10pct",
            "ablation_config_id": config_id,
            "ablation_family": family,
            "ablation_level": level,
            "scientific_change": norm(row["scientific_change"]),
            "main_run_id": main_run_id,
        }

        for key, expected in expected_pairs.items():
            observed = norm(manifest.get(key))
            if key in {
                "ablation_config_id",
                "ablation_family",
                "ablation_level",
            }:
                observed = observed.upper()
            require(
                observed == expected,
                f"RUN_MANIFEST_IDENTITY_MISMATCH: "
                f"{run_id}: {key}",
            )

        require(
            int(manifest.get("seed_index")) == seed_index,
            f"RUN_MANIFEST_SEED_INDEX_MISMATCH: {run_id}",
        )
        require(
            int(manifest.get("training_seed")) == training_seed,
            f"RUN_MANIFEST_TRAINING_SEED_MISMATCH: {run_id}",
        )

        require(
            manifest.get("validation_only") is True,
            f"MANIFEST_VALIDATION_ONLY_INVALID: {run_id}",
        )
        require(
            manifest.get("final_test_eligible") is False,
            f"MANIFEST_FINAL_TEST_ELIGIBLE_INVALID: {run_id}",
        )
        require(
            manifest.get("test_access_authorized") is False,
            f"MANIFEST_TEST_ACCESS_INVALID: {run_id}",
        )
        require(
            manifest.get("final_test_accessed") is False,
            f"MANIFEST_FINAL_TEST_ACCESSED: {run_id}",
        )
        require(
            norm(manifest.get("test_usage"))
            == "PROHIBITED_NON_MAIN_ABLATION",
            f"MANIFEST_TEST_USAGE_INVALID: {run_id}",
        )
        require(
            norm(manifest.get("final_evaluation_status"))
            == "NOT_ELIGIBLE",
            f"MANIFEST_FINAL_EVALUATION_STATUS_INVALID: {run_id}",
        )

        config_diff_only_target_factor = validate_ofat(
            run_id,
            config_id,
            family,
            manifest.get("effective_scientific_fields"),
        )

        for field in FROZEN_FIELDS:
            value = manifest.get(field)
            require(
                value is not None and norm(value) != "",
                f"FROZEN_FIELD_MISSING: {run_id}: {field}",
            )
            frozen_values[field].add(
                json.dumps(value, sort_keys=True)
            )

        family_counts[family] += 1
        config_counts[config_id] += 1

        pair_valid = (
            config_diff_only_target_factor
            and ref["training_seed"] == training_seed
            and ref["main_run_id"] == main_run_id
        )

        require(
            pair_valid,
            f"PAIR_INVALID: {run_id}",
        )

        output_rows.append({
            "ablation_run_id": run_id,
            "main_run_id": main_run_id,
            "family": family,
            "level": norm(row["level"]),
            "seed_index": str(seed_index),
            "training_seed": str(training_seed),
            "config_diff_only_target_factor": "true",
            "pair_valid": "true",
        })

    require(
        len(output_rows) == EXPECTED_ROWS,
        f"OUTPUT_ROWS_MISMATCH: {len(output_rows)}",
    )

    require(
        len(set(r["ablation_run_id"] for r in output_rows))
        == EXPECTED_ROWS,
        "ABLATION_RUN_IDS_NOT_UNIQUE",
    )

    require(
        attempt_counts == Counter({
            "attempt_001": 89,
            "attempt_002": 1,
        }),
        f"CANONICAL_ATTEMPT_COUNTS_INVALID: {dict(attempt_counts)}",
    )

    require(
        family_counts == Counter({
            "CONFIDENCE_THRESHOLD": 40,
            "STRONG_PHOTOMETRIC_AUGMENTATION": 30,
            "REGRESSION_PSEUDO_LABEL_FILTERING": 20,
        }),
        f"FAMILY_COUNTS_INVALID: {dict(family_counts)}",
    )

    expected_configs = (
        set(C_LEVELS)
        | set(A_LEVELS)
        | set(F_LEVELS)
    )

    require(
        set(config_counts) == expected_configs,
        f"CONFIG_SET_INVALID: {sorted(config_counts)}",
    )

    require(
        all(config_counts[c] == 10 for c in expected_configs),
        f"CONFIG_COUNTS_INVALID: {dict(config_counts)}",
    )

    for field in FROZEN_FIELDS:
        require(
            len(frozen_values[field]) == 1,
            f"FROZEN_FIELD_VARIATION: "
            f"{field}: unique={len(frozen_values[field])}",
        )

    output_rows.sort(
        key=lambda r: (
            r["family"],
            r["level"],
            int(r["seed_index"]),
            r["ablation_run_id"],
        )
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=OUTPUT_FIELDS,
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print("=== S10.1.06 ABLATION PAIR INVENTORY ===")
    print(f"OUTPUT_ROWS={len(output_rows)}")
    print(
        "UNIQUE_ABLATION_RUN_IDS="
        f"{len(set(r['ablation_run_id'] for r in output_rows))}"
    )
    print(
        "PAIR_VALID_TRUE="
        f"{sum(r['pair_valid'] == 'true' for r in output_rows)}"
    )
    print(
        "CONFIG_DIFF_ONLY_TARGET_FACTOR_TRUE="
        f"{sum(r['config_diff_only_target_factor'] == 'true' for r in output_rows)}"
    )
    print(f"COLUMNS={len(OUTPUT_FIELDS)}")
    print("S10_1_06_STATUS=PASS")


if __name__ == "__main__":
    main()