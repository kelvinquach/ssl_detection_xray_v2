#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import scipy
from scipy import stats


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

FROZEN_SEED_SUMMARY_SHA256 = (
    "2a404ac0f6097cad16816b550fcd1c9ca240fab5535854dc0fd5e4433e49ff3f"
)
FROZEN_RQ3_PAIRWISE_SHA256 = (
    "18c0c84fd396193e6019c9a55f051e6ab68ea9cb08e132017d92ddd017be8381"
)
FROZEN_SCIPY_VERSION = "1.15.3"

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

OUTPUT_FIELDS = [
    "variant_id",
    "ablation_run_id",
    "main_run_id",
    "family",
    "level",
    "seed_index",
    "training_seed",
    "q_pseudo_variant",
    "q_pseudo_main_same_seed",
    "delta_q_pseudo",
    "n_seeds",
    "variant_mean",
    "variant_sample_SD_ddof1",
    "main_mean",
    "paired_mean_difference",
    "paired_difference_sample_SD_ddof1",
    "paired_difference_two_sided_95CI_low",
    "paired_difference_two_sided_95CI_high",
    "ci_level",
    "variant_checkpoint_sha256",
    "main_checkpoint_sha256",
    "evaluator_config_sha256",
    "pseudo_acceptance_config_sha256",
    "validation_split_sha256",
    "variant_qpseudo_path",
    "main_qpseudo_path",
    "checkpoint_role",
    "dataset",
    "test_used",
    "hidden_u_gt_used",
    "formal_hypothesis_test",
    "statistics_semantics",
    "representation_authority",
]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def norm(value):
    if value is None:
        return ""
    return str(value).strip().strip('"')


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_sha(value, label):
    value = norm(value).lower()
    require(
        SHA256_RE.fullmatch(value) is not None,
        f"{label}:INVALID_SHA256:{value!r}",
    )
    return value


def finite_float(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{label}:NON_NUMERIC:{value!r}"
        ) from exc

    require(
        math.isfinite(result),
        f"{label}:NON_FINITE:{result!r}",
    )
    return result


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


def load_qpseudo(path, label):
    payload = load_json(path)

    require(
        norm(payload.get("checkpoint_role"))
        == "LAST_EMA_TEACHER",
        f"{label}:CHECKPOINT_ROLE_INVALID",
    )
    require(
        norm(payload.get("dataset"))
        == "fixed_validation",
        f"{label}:DATASET_INVALID",
    )
    require(
        payload.get("test_used") is False,
        f"{label}:TEST_USED_NOT_FALSE",
    )
    require(
        payload.get("hidden_u_gt_used") is False,
        f"{label}:HIDDEN_U_GT_USED_NOT_FALSE",
    )
    require(
        int(payload.get("validation_image_count"))
        == 734,
        f"{label}:VALIDATION_IMAGE_COUNT_INVALID",
    )

    return {
        "value": finite_float(
            payload.get("PL_mAP_50_95"),
            f"{label}:PL_mAP_50_95",
        ),
        "checkpoint_sha256": require_sha(
            payload.get("checkpoint_sha256"),
            f"{label}:CHECKPOINT",
        ),
        "evaluator_config_sha256": require_sha(
            payload.get("evaluator_config_sha256"),
            f"{label}:EVALUATOR",
        ),
        "pseudo_acceptance_config_sha256": require_sha(
            payload.get("pseudo_acceptance_config_sha256"),
            f"{label}:PSEUDO_ACCEPTANCE",
        ),
        "validation_split_sha256": require_sha(
            payload.get("validation_split_sha256"),
            f"{label}:VALIDATION_SPLIT",
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--repo", required=True)
    parser.add_argument("--pair-inventory", required=True)
    parser.add_argument("--run-inventory", required=True)
    parser.add_argument("--main-map", required=True)
    parser.add_argument("--variant-root", required=True)
    parser.add_argument("--main-root", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    variant_root = Path(args.variant_root).resolve()
    main_root = Path(args.main_root).resolve()

    sys.path.insert(0, str(repo))

    from src.statistics.seed_summary import (
        OFFICIAL_SEED_COUNT,
        OFFICIAL_TRAINING_SEEDS,
        SAMPLE_SD_DDOF,
        summarize_seed_runs,
    )
    from src.statistics.rq3_pairwise import (
        RQ3_PAIRWISE_ALPHA,
        RQ3_PAIRWISE_CI_LEVEL,
        RQ3_PAIRWISE_DF,
    )

    require(
        sha256_file(repo / "src/statistics/seed_summary.py")
        == FROZEN_SEED_SUMMARY_SHA256,
        "FROZEN_SEED_SUMMARY_SHA256_MISMATCH",
    )
    require(
        sha256_file(repo / "src/statistics/rq3_pairwise.py")
        == FROZEN_RQ3_PAIRWISE_SHA256,
        "FROZEN_RQ3_PAIRWISE_SHA256_MISMATCH",
    )
    require(
        scipy.__version__ == FROZEN_SCIPY_VERSION,
        f"SCIPY_VERSION_MISMATCH:"
        f"{scipy.__version__}!={FROZEN_SCIPY_VERSION}",
    )

    require(
        SAMPLE_SD_DDOF == 1,
        "SAMPLE_SD_DDOF_NOT_1",
    )
    require(
        OFFICIAL_SEED_COUNT == 10,
        "OFFICIAL_SEED_COUNT_NOT_10",
    )
    require(
        RQ3_PAIRWISE_ALPHA == 0.05,
        "CI_ALPHA_NOT_0_05",
    )
    require(
        RQ3_PAIRWISE_CI_LEVEL == 0.95,
        "CI_LEVEL_NOT_0_95",
    )
    require(
        RQ3_PAIRWISE_DF == 9,
        "CI_DF_NOT_9",
    )

    pairs = load_csv(args.pair_inventory)
    run_inventory = load_csv(args.run_inventory)
    main_map = load_csv(args.main_map)

    require(len(pairs) == 90, f"PAIR_ROWS={len(pairs)}")
    require(
        len(run_inventory) == 90,
        f"RUN_INVENTORY_ROWS={len(run_inventory)}",
    )
    require(
        len(main_map) == 10,
        f"MAIN_MAP_ROWS={len(main_map)}",
    )

    design_by_run = {}

    for row in run_inventory:
        run_id = norm(row["ablation_run_id"])
        variant_id = norm(
            row["ablation_config_id"]
        ).upper()

        require(
            variant_id in CONFIG_ORDER,
            f"{run_id}:INVALID_VARIANT_ID:{variant_id}",
        )
        require(
            run_id not in design_by_run,
            f"DUPLICATE_RUN_INVENTORY:{run_id}",
        )

        design_by_run[run_id] = row

    main_by_key = {}

    for row in main_map:
        key = (
            int(norm(row["seed_index"])),
            int(norm(row["training_seed"])),
            norm(row["main_run_id"]),
        )

        require(
            key not in main_by_key,
            f"DUPLICATE_MAIN_REFERENCE:{key}",
        )
        main_by_key[key] = row

    seed_rows_by_variant = {
        variant_id: {}
        for variant_id in CONFIG_ORDER
    }

    for pair in pairs:
        ablation_run_id = norm(
            pair["ablation_run_id"]
        )
        main_run_id = norm(pair["main_run_id"])
        seed_index = int(norm(pair["seed_index"]))
        training_seed = int(
            norm(pair["training_seed"])
        )

        require(
            norm(pair["pair_valid"]).lower()
            == "true",
            f"{ablation_run_id}:PAIR_NOT_VALID",
        )

        require(
            ablation_run_id in design_by_run,
            f"{ablation_run_id}:DESIGN_MISSING",
        )

        design = design_by_run[ablation_run_id]

        variant_id = norm(
            design["ablation_config_id"]
        ).upper()
        family = norm(design["family"])
        level = norm(design["level"])

        require(
            family == norm(pair["family"]),
            f"{ablation_run_id}:FAMILY_MISMATCH",
        )
        require(
            level == norm(pair["level"]),
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
            main_key in main_by_key,
            f"{ablation_run_id}:MAIN_REFERENCE_MISSING",
        )

        status_path = (
            variant_root
            / ablation_run_id
            / "logical_run_status.json"
        )
        status = load_json(status_path)

        require(
            norm(status.get("training_status"))
            == "CLOSED_VERIFIED",
            f"{ablation_run_id}:STATUS_INVALID",
        )
        require(
            status.get("training_completed") is True,
            f"{ablation_run_id}:NOT_COMPLETED",
        )
        require(
            status.get("training_valid") is True,
            f"{ablation_run_id}:NOT_VALID",
        )

        variant_attempt = norm(
            status.get("attempt_used")
        )
        require(
            re.fullmatch(
                r"attempt_\d{3}",
                variant_attempt,
            )
            is not None,
            f"{ablation_run_id}:ATTEMPT_INVALID",
        )

        variant_q_path = (
            variant_root
            / ablation_run_id
            / variant_attempt
            / "pseudo"
            / "qpseudo_validation.json"
        )

        main_q_path = (
            main_root
            / main_run_id
            / "attempt_001"
            / "pseudo"
            / "qpseudo_validation.json"
        )

        variant_q = load_qpseudo(
            variant_q_path,
            ablation_run_id,
        )
        main_q = load_qpseudo(
            main_q_path,
            main_run_id,
        )

        require(
            variant_q["evaluator_config_sha256"]
            == main_q["evaluator_config_sha256"],
            f"{ablation_run_id}:EVALUATOR_MISMATCH",
        )
        require(
            variant_q["pseudo_acceptance_config_sha256"]
            == main_q["pseudo_acceptance_config_sha256"],
            f"{ablation_run_id}:PSEUDO_ACCEPTANCE_MISMATCH",
        )
        require(
            variant_q["validation_split_sha256"]
            == main_q["validation_split_sha256"],
            f"{ablation_run_id}:VALIDATION_SPLIT_MISMATCH",
        )

        require(
            training_seed not in seed_rows_by_variant[
                variant_id
            ],
            f"{variant_id}:DUPLICATE_SEED:{training_seed}",
        )

        seed_rows_by_variant[variant_id][
            training_seed
        ] = {
            "variant_id": variant_id,
            "ablation_run_id": ablation_run_id,
            "main_run_id": main_run_id,
            "family": family,
            "level": level,
            "seed_index": seed_index,
            "training_seed": training_seed,
            "q_pseudo_variant": variant_q["value"],
            "q_pseudo_main_same_seed": main_q["value"],
            "delta_q_pseudo":
                variant_q["value"]
                - main_q["value"],
            "variant_checkpoint_sha256":
                variant_q["checkpoint_sha256"],
            "main_checkpoint_sha256":
                main_q["checkpoint_sha256"],
            "evaluator_config_sha256":
                variant_q["evaluator_config_sha256"],
            "pseudo_acceptance_config_sha256":
                variant_q[
                    "pseudo_acceptance_config_sha256"
                ],
            "validation_split_sha256":
                variant_q["validation_split_sha256"],
            "variant_qpseudo_path":
                "artifacts/runs/ablation/"
                f"{ablation_run_id}/"
                f"{variant_attempt}/"
                "pseudo/qpseudo_validation.json",
            "main_qpseudo_path":
                "artifacts/runs/official/"
                f"{main_run_id}/"
                "attempt_001/"
                "pseudo/qpseudo_validation.json",
        }

    output_rows = []

    for variant_id in sorted(
        CONFIG_ORDER,
        key=CONFIG_ORDER.get,
    ):
        seed_rows = seed_rows_by_variant[
            variant_id
        ]

        require(
            tuple(seed_rows.keys())
            == OFFICIAL_TRAINING_SEEDS,
            f"{variant_id}:SEED_ORDER_OR_SET_INVALID:"
            f"{tuple(seed_rows.keys())}",
        )

        variant_values = {
            seed: seed_rows[seed][
                "q_pseudo_variant"
            ]
            for seed in OFFICIAL_TRAINING_SEEDS
        }
        main_values = {
            seed: seed_rows[seed][
                "q_pseudo_main_same_seed"
            ]
            for seed in OFFICIAL_TRAINING_SEEDS
        }
        paired_differences = {
            seed: seed_rows[seed][
                "delta_q_pseudo"
            ]
            for seed in OFFICIAL_TRAINING_SEEDS
        }

        variant_summary = summarize_seed_runs(
            variant_values
        )
        main_summary = summarize_seed_runs(
            main_values
        )
        delta_summary = summarize_seed_runs(
            paired_differences
        )

        n = int(delta_summary["n"])
        df = n - 1

        require(
            n == OFFICIAL_SEED_COUNT,
            f"{variant_id}:N_INVALID:{n}",
        )
        require(
            df == RQ3_PAIRWISE_DF,
            f"{variant_id}:DF_INVALID:{df}",
        )

        effect = float(delta_summary["mean"])
        paired_sd = float(
            delta_summary["sample_sd"]
        )

        standard_error = paired_sd / math.sqrt(n)

        critical_value = float(
            stats.t.ppf(
                1.0
                - RQ3_PAIRWISE_ALPHA / 2.0,
                df=df,
            )
        )

        ci_low = (
            effect
            - critical_value * standard_error
        )
        ci_high = (
            effect
            + critical_value * standard_error
        )

        require(
            all(
                math.isfinite(x)
                for x in (
                    effect,
                    paired_sd,
                    critical_value,
                    ci_low,
                    ci_high,
                )
            ),
            f"{variant_id}:NON_FINITE_SUMMARY",
        )

        for seed in OFFICIAL_TRAINING_SEEDS:
            row = seed_rows[seed]

            output_rows.append({
                "variant_id":
                    row["variant_id"],
                "ablation_run_id":
                    row["ablation_run_id"],
                "main_run_id":
                    row["main_run_id"],
                "family":
                    row["family"],
                "level":
                    row["level"],
                "seed_index":
                    str(row["seed_index"]),
                "training_seed":
                    str(row["training_seed"]),
                "q_pseudo_variant":
                    repr(row["q_pseudo_variant"]),
                "q_pseudo_main_same_seed":
                    repr(
                        row[
                            "q_pseudo_main_same_seed"
                        ]
                    ),
                "delta_q_pseudo":
                    repr(row["delta_q_pseudo"]),
                "n_seeds":
                    str(n),
                "variant_mean":
                    repr(
                        float(
                            variant_summary["mean"]
                        )
                    ),
                "variant_sample_SD_ddof1":
                    repr(
                        float(
                            variant_summary[
                                "sample_sd"
                            ]
                        )
                    ),
                "main_mean":
                    repr(
                        float(
                            main_summary["mean"]
                        )
                    ),
                "paired_mean_difference":
                    repr(effect),
                "paired_difference_sample_SD_ddof1":
                    repr(paired_sd),
                "paired_difference_two_sided_95CI_low":
                    repr(ci_low),
                "paired_difference_two_sided_95CI_high":
                    repr(ci_high),
                "ci_level":
                    repr(
                        RQ3_PAIRWISE_CI_LEVEL
                    ),
                "variant_checkpoint_sha256":
                    row[
                        "variant_checkpoint_sha256"
                    ],
                "main_checkpoint_sha256":
                    row[
                        "main_checkpoint_sha256"
                    ],
                "evaluator_config_sha256":
                    row[
                        "evaluator_config_sha256"
                    ],
                "pseudo_acceptance_config_sha256":
                    row[
                        "pseudo_acceptance_config_sha256"
                    ],
                "validation_split_sha256":
                    row[
                        "validation_split_sha256"
                    ],
                "variant_qpseudo_path":
                    row[
                        "variant_qpseudo_path"
                    ],
                "main_qpseudo_path":
                    row[
                        "main_qpseudo_path"
                    ],
                "checkpoint_role":
                    "LAST_EMA_TEACHER",
                "dataset":
                    "fixed_validation",
                "test_used":
                    "false",
                "hidden_u_gt_used":
                    "false",
                "formal_hypothesis_test":
                    "false",
                "statistics_semantics":
                    "FROZEN_S6_SEED_SUMMARY_PLUS_STUDENT_T_CI_NO_TEST",
                "representation_authority":
                    "S10_ABLATION_QPSEUDO_ESTIMATION_ONLY",
            })

    require(
        len(output_rows) == 90,
        f"OUTPUT_ROWS={len(output_rows)}",
    )
    require(
        len({
            (
                row["variant_id"],
                row["training_seed"],
            )
            for row in output_rows
        }) == 90,
        "OUTPUT_KEY_NOT_UNIQUE",
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
        "=== S10.1.09 ABLATION QPSEUDO EFFECTS ==="
    )
    print(f"OUTPUT_ROWS={len(output_rows)}")
    print(f"COLUMNS={len(OUTPUT_FIELDS)}")
    print("VARIANTS=9")
    print("SEEDS_PER_VARIANT=10")
    print("QPSEUDO_METRIC=PL_mAP_50_95")
    print("CHECKPOINT_ROLE=LAST_EMA_TEACHER")
    print("DATASET=fixed_validation")
    print("DELTA=VARIANT_MINUS_MAIN_SAME_SEED")
    print("SAMPLE_SD_DDOF=1")
    print("CI_LEVEL=0.95")
    print("FORMAL_HYPOTHESIS_TEST=FALSE")
    print("P_VALUE_COMPUTED=FALSE")
    print("FINAL_TEST_USED=FALSE")
    print("HIDDEN_U_GT_USED=FALSE")
    print(
        "STATISTICS_CORE="
        "FROZEN_S6_REUSE"
    )
    print("S10_1_09_STATUS=PASS")


if __name__ == "__main__":
    main()