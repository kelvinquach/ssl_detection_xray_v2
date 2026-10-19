#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


EXPECTED_LOW_LABEL_ROWS = 160
EXPECTED_PAIRS = 80

OUTPUT_FIELDS = [
    "pair_id",
    "architecture",
    "budget",
    "seed_index",
    "training_seed",
    "sup_run_id",
    "ssl_run_id",
    "labeled_manifest_sha256_match",
    "train_split_sha256_match",
    "val_split_sha256_match",
    "test_split_sha256_match",
    "training_seed_match",
    "architecture_match",
    "detector_family_match",
    "preprocessing_config_sha256_match",
    "labeled_pipeline_sha256_match",
    "optimizer_recipe_sha256_match",
    "update_budget_match",
    "effective_labeled_batch_match",
    "evaluator_config_sha256_match",
    "checkpoint_rule_match",
    "observed_labeled_images_consumed_match",
    "observed_optimizer_updates_match",
    "labeled_supervision_exposure_match",
    "pair_valid",
]


def norm(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().strip('"')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(rows, f"CSV is empty: {path}")
    return rows


def read_json(path: Path) -> dict:
    require(path.is_file(), f"JSON not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def training_summary(
    training_root: Path,
    row: dict[str, str],
) -> dict:
    run_id = norm(row.get("s9_12.run_id"))
    attempt = norm(row.get("s9_12.attempt_used"))

    require(run_id, "Missing s9_12.run_id")
    require(attempt, f"{run_id}: missing s9_12.attempt_used")

    base = training_root / run_id / attempt
    hits = list(base.rglob("training_summary.json"))

    require(
        len(hits) == 1,
        (
            f"{run_id}/{attempt}: expected exactly one "
            f"training_summary.json, found {len(hits)}"
        ),
    )

    return read_json(hits[0])


def make_pair_id(
    architecture: str,
    budget: str,
    seed_index: str,
    training_seed: str,
) -> str:
    return (
        f"PAIR_{architecture}_{budget}_"
        f"S{int(seed_index):02d}_{training_seed}"
    )


def same(
    sup: dict[str, str],
    ssl: dict[str, str],
    field: str,
) -> bool:
    a = norm(sup.get(field))
    b = norm(ssl.get(field))
    return bool(a) and a == b


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize canonical S10.1 official_pair_inventory.csv "
            "for the 80 paired low-label SUP-SSL runs."
        )
    )
    parser.add_argument(
        "--eligible-model-inventory",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--training-root",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--pairing-report",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )
    args = parser.parse_args()

    rows = read_csv(args.eligible_model_inventory)
    pair_report = read_json(args.pairing_report)

    require(
        pair_report.get("status") == "PASS",
        "S7.24 pairing report status is not PASS",
    )
    require(
        pair_report.get("overall_pass") is True,
        "S7.24 pairing report overall_pass != true",
    )
    require(
        int(pair_report.get("pair_count", -1)) == 8,
        "S7.24 pairing report pair_count != 8",
    )

    templates = {
        (norm(p.get("architecture")), norm(p.get("budget"))): p
        for p in pair_report["pairs"]
    }
    require(len(templates) == 8, "Expected 8 pairing templates")

    low = [
        row
        for row in rows
        if norm(row.get("official_inventory.condition_role"))
        == "MAIN_LOW_LABEL"
        and norm(row.get("official_inventory.method"))
        in {"SUP", "SSL"}
    ]

    require(
        len(low) == EXPECTED_LOW_LABEL_ROWS,
        (
            f"Expected {EXPECTED_LOW_LABEL_ROWS} low-label rows, "
            f"found {len(low)}"
        ),
    )

    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = (
        defaultdict(list)
    )

    for row in low:
        key = (
            norm(row.get("official_inventory.architecture")),
            norm(row.get("official_inventory.budget")),
            norm(row.get("official_inventory.seed_index")),
            norm(row.get("official_inventory.training_seed")),
        )
        groups[key].append(row)

    require(
        len(groups) == EXPECTED_PAIRS,
        f"Expected {EXPECTED_PAIRS} pair keys, found {len(groups)}",
    )

    output: list[dict[str, str]] = []
    seen_pair_ids: set[str] = set()

    for key in sorted(
        groups,
        key=lambda x: (x[0], x[1], int(x[2]), int(x[3])),
    ):
        architecture, budget, seed_index, training_seed = key
        members = groups[key]

        sup_rows = [
            row for row in members
            if norm(row.get("official_inventory.method")) == "SUP"
        ]
        ssl_rows = [
            row for row in members
            if norm(row.get("official_inventory.method")) == "SSL"
        ]

        require(
            len(sup_rows) == 1 and len(ssl_rows) == 1,
            (
                f"{architecture}/{budget}/S{seed_index}: "
                f"SUP={len(sup_rows)} SSL={len(ssl_rows)}"
            ),
        )

        sup = sup_rows[0]
        ssl = ssl_rows[0]

        sup_run_id = norm(sup.get("s9_12.run_id"))
        ssl_run_id = norm(ssl.get("s9_12.run_id"))

        require(
            norm(sup.get("official_inventory.condition_role"))
            == "MAIN_LOW_LABEL",
            f"{sup_run_id}: invalid condition_role",
        )
        require(
            norm(ssl.get("official_inventory.condition_role"))
            == "MAIN_LOW_LABEL",
            f"{ssl_run_id}: invalid condition_role",
        )

        template = templates.get((architecture, budget))
        require(
            template is not None,
            f"No S7.24 template for {architecture}/{budget}",
        )

        required_match = template.get("required_match", {})

        ts_sup = training_summary(args.training_root, sup)
        ts_ssl = training_summary(args.training_root, ssl)

        flags: dict[str, bool] = {}

        flags["labeled_manifest_sha256_match"] = same(
            sup,
            ssl,
            "pre_run_assertions.observed_identity."
            "labeled_manifest_sha256",
        )
        flags["train_split_sha256_match"] = same(
            sup,
            ssl,
            "pre_run_assertions.observed_identity.train_split_sha256",
        )
        flags["val_split_sha256_match"] = same(
            sup,
            ssl,
            "pre_run_assertions.observed_identity.val_split_sha256",
        )
        flags["test_split_sha256_match"] = same(
            sup,
            ssl,
            "pre_run_assertions.observed_identity.test_split_sha256",
        )

        flags["training_seed_match"] = (
            same(
                sup,
                ssl,
                "pre_run_assertions.observed_identity.training_seed",
            )
            and norm(sup.get("official_inventory.training_seed"))
            == training_seed
            and norm(ssl.get("official_inventory.training_seed"))
            == training_seed
        )

        flags["architecture_match"] = (
            same(
                sup,
                ssl,
                "pre_run_assertions.observed_identity.architecture",
            )
            and norm(sup.get("official_inventory.architecture"))
            == architecture
            and norm(ssl.get("official_inventory.architecture"))
            == architecture
        )

        flags["detector_family_match"] = (
            required_match.get("detector_family") is True
        )

        flags["preprocessing_config_sha256_match"] = same(
            sup,
            ssl,
            "pre_run_assertions.observed_identity."
            "preprocessing_config_sha256",
        )
        flags["labeled_pipeline_sha256_match"] = same(
            sup,
            ssl,
            "pre_run_assertions.observed_identity."
            "labeled_pipeline_sha256",
        )
        flags["optimizer_recipe_sha256_match"] = same(
            sup,
            ssl,
            "pre_run_assertions.observed_identity."
            "optimizer_recipe_sha256",
        )
        flags["update_budget_match"] = same(
            sup,
            ssl,
            "pre_run_assertions.observed_identity."
            "total_optimizer_updates",
        )
        flags["effective_labeled_batch_match"] = same(
            sup,
            ssl,
            "pre_run_assertions.observed_identity."
            "effective_labeled_batch",
        )
        flags["evaluator_config_sha256_match"] = same(
            sup,
            ssl,
            "pre_run_assertions.observed_identity."
            "evaluator_config_sha256",
        )

        flags["checkpoint_rule_match"] = (
            required_match.get(
                "checkpoint_selection_metric_family"
            ) is True
            and same(
                sup,
                ssl,
                "checkpoint_index.checkpoint_policy",
            )
            and norm(
                sup.get("checkpoint_index.checkpoint_policy")
            ) == "LOCKED"
            and norm(
                ssl.get("checkpoint_index.checkpoint_policy")
            ) == "LOCKED"
            and same(
                sup,
                ssl,
                "pre_run_assertions.observed_identity."
                "checkpoint_policy_sha256",
            )
            and same(
                sup,
                ssl,
                "pre_run_assertions.observed_identity."
                "checkpoint_selection_metric",
            )
            and same(
                sup,
                ssl,
                "validation_best.selection_metric",
            )
            and norm(
                sup.get("validation_best.rule")
            ) == "ARGMAX_VALIDATION_BBOX_MAP_50_95"
            and norm(
                ssl.get("validation_best.rule")
            ) == "ARGMAX_VALIDATION_TEACHER_BBOX_MAP_50_95"
            and norm(
                sup.get("validation_best.selection_metric")
            ) == "bbox_mAP_50_95"
            and norm(
                ssl.get("validation_best.selection_metric")
            ) == "bbox_mAP_50_95"
            and norm(
                ssl.get("validation_best.selection_model")
            ) == "EMA_TEACHER"
            and bool(
                norm(sup.get("source_sha256.checkpoint_index"))
            )
            and bool(
                norm(ssl.get("source_sha256.checkpoint_index"))
            )
            and norm(
                sup.get("source_sha256.checkpoint_index")
            ) == norm(
                sup.get(
                    "retained_artifacts.checkpoints/"
                    "checkpoint_index.json.actual_sha256"
                )
            )
            and norm(
                ssl.get("source_sha256.checkpoint_index")
            ) == norm(
                ssl.get(
                    "retained_artifacts.checkpoints/"
                    "checkpoint_index.json.actual_sha256"
                )
            )
            and norm(
                sup.get(
                    "retained_artifacts.checkpoints/"
                    "checkpoint_index.json.storage_state"
                )
            ) == "LOCAL_VERIFIED"
            and norm(
                ssl.get(
                    "retained_artifacts.checkpoints/"
                    "checkpoint_index.json.storage_state"
                )
            ) == "LOCAL_VERIFIED"
        )

        flags["observed_labeled_images_consumed_match"] = (
            ts_sup["labeled_images_consumed_total"]
            == ts_ssl["labeled_images_consumed_total"]
        )
        flags["observed_optimizer_updates_match"] = (
            ts_sup["observed_optimizer_updates"]
            == ts_ssl["observed_optimizer_updates"]
        )

        flags["labeled_supervision_exposure_match"] = (
            flags["observed_labeled_images_consumed_match"]
            and flags["observed_optimizer_updates_match"]
            and flags["effective_labeled_batch_match"]
            and ts_sup["labeled_exposure_matches_update_count"] is True
            and ts_ssl["labeled_exposure_matches_update_count"] is True
        )

        pair_valid = all(flags.values())

        require(
            pair_valid,
            (
                f"Invalid pair {sup_run_id} / {ssl_run_id}: "
                + ",".join(
                    name for name, value in flags.items()
                    if not value
                )
            ),
        )

        pair_id = make_pair_id(
            architecture,
            budget,
            seed_index,
            training_seed,
        )
        require(
            pair_id not in seen_pair_ids,
            f"Duplicate pair_id: {pair_id}",
        )
        seen_pair_ids.add(pair_id)

        result = {
            "pair_id": pair_id,
            "architecture": architecture,
            "budget": budget,
            "seed_index": seed_index,
            "training_seed": training_seed,
            "sup_run_id": sup_run_id,
            "ssl_run_id": ssl_run_id,
        }

        for field, value in flags.items():
            result[field] = bool_text(value)

        result["pair_valid"] = bool_text(pair_valid)
        output.append(result)

    require(
        len(output) == EXPECTED_PAIRS,
        f"Expected {EXPECTED_PAIRS} output pairs, found {len(output)}",
    )
    require(
        len(seen_pair_ids) == EXPECTED_PAIRS,
        "Unique pair_id count != 80",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_FIELDS,
        )
        writer.writeheader()
        writer.writerows(output)

    print("=== S10.1.02 OFFICIAL PAIR INVENTORY ===")
    print(f"LOW_LABEL_INPUT_ROWS={len(low)}")
    print(f"PAIR_ROWS={len(output)}")
    print(f"UNIQUE_PAIR_IDS={len(seen_pair_ids)}")
    print(
        "PAIR_VALID_TRUE="
        f"{sum(r['pair_valid'] == 'true' for r in output)}"
    )
    print(
        "SUP_REFERENCE_ROWS="
        f"{sum(norm(r.get('official_inventory.condition_role')) == 'SUP_REFERENCE' for r in low)}"
    )
    print("S10_1_02_STATUS=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
