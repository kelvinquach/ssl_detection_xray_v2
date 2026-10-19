from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DETECTION_SOURCE = (
    ROOT / "artifacts/results/ablation/ablation_effects.csv"
)

QPSEUDO_SOURCE = (
    ROOT / "analysis/ablation/ablation_qpseudo_effects.csv"
)

OUTPUT = (
    ROOT / "analysis/ablation/ablation_seed_level_values.csv"
)

EXPECTED_DETECTION_SHA256 = (
    "23ca736c91ab27cb461c63c97e896cdb3017da8a7a114f1cb9693ced41d281f6"
)

EXPECTED_QPSEUDO_SHA256 = (
    "d67f2a595fa6f9d8949ac7022c3958818b2c0798e1f1c612aa04f7433f2877b8"
)

EXPECTED_VARIANTS = {
    "C0", "C1", "C2", "C3",
    "A0", "A1", "A2",
    "F0", "F1",
}

EXPECTED_METRICS = {
    "validation_bbox_mAP_50_95",
    "validation_AP50",
    "validation_AP75",
    "Q_pseudo",
}

FIELDNAMES = [
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
    "variant_checkpoint_sha256",
    "main_checkpoint_sha256",
    "variant_evidence_path",
    "main_evidence_path",
    "evaluator_config_sha256",
    "validation_split_sha256",
    "checkpoint_role",
    "dataset",
    "test_used",
    "hidden_u_gt_used",
    "pair_valid",
    "formal_hypothesis_test",
    "statistics_semantics",
    "representation_authority",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def norm(value) -> str:
    return str(value if value is not None else "").strip()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    require(DETECTION_SOURCE.is_file(), "DETECTION_SOURCE_MISSING")
    require(QPSEUDO_SOURCE.is_file(), "QPSEUDO_SOURCE_MISSING")

    detection_sha = sha256(DETECTION_SOURCE)
    qpseudo_sha = sha256(QPSEUDO_SOURCE)

    require(
        detection_sha == EXPECTED_DETECTION_SHA256,
        f"DETECTION_SOURCE_SHA_MISMATCH:{detection_sha}",
    )
    require(
        qpseudo_sha == EXPECTED_QPSEUDO_SHA256,
        f"QPSEUDO_SOURCE_SHA_MISMATCH:{qpseudo_sha}",
    )

    detection = load_csv(DETECTION_SOURCE)
    qpseudo = load_csv(QPSEUDO_SOURCE)

    require(len(detection) == 270, f"DETECTION_ROWS:{len(detection)}")
    require(len(qpseudo) == 90, f"QPSEUDO_ROWS:{len(qpseudo)}")

    rows: list[dict[str, str]] = []

    # ------------------------------------------------------------
    # Detection metrics: preserve existing paired seed-level values.
    # No statistics are recomputed here.
    # ------------------------------------------------------------
    for r in detection:
        require(norm(r["pair_valid"]).lower() == "true", "DETECTION_PAIR_INVALID")
        require(
            norm(r["formal_hypothesis_test"]).lower() == "false",
            "DETECTION_FORMAL_TEST_INVALID",
        )

        rows.append({
            "variant_id": norm(r["variant_id"]),
            "ablation_run_id": norm(r["ablation_run_id"]),
            "main_run_id": norm(r["main_run_id"]),
            "family": norm(r["family"]),
            "level": norm(r["level"]),
            "seed_index": norm(r["seed_index"]),
            "training_seed": norm(r["training_seed"]),
            "metric_name": norm(r["metric_name"]),
            "variant_value": norm(r["variant_value"]),
            "main_same_seed_value": norm(r["main_same_seed_value"]),
            "paired_difference": norm(r["paired_difference"]),
            "variant_checkpoint_sha256":
                norm(r["variant_validation_checkpoint_sha256"]),
            "main_checkpoint_sha256":
                norm(r["main_validation_checkpoint_sha256"]),
            "variant_evidence_path":
                norm(r["variant_validation_best_path"]),
            "main_evidence_path":
                norm(r["main_validation_best_path"]),
            "evaluator_config_sha256": "",
            "validation_split_sha256": "",
            "checkpoint_role": "BEST_EMA_TEACHER",
            "dataset": "fixed_validation",
            "test_used": "false",
            "hidden_u_gt_used": "false",
            "pair_valid": "true",
            "formal_hypothesis_test": "false",
            "statistics_semantics":
                "SOURCE_PRESERVING_SEED_LEVEL_REPRESENTATION",
            "representation_authority":
                "S10_ABLATION_SEED_LEVEL_VALUES",
        })

    # ------------------------------------------------------------
    # Q_pseudo: preserve existing paired seed-level values.
    # Q_pseudo is mandatory secondary mechanistic metric.
    # No statistics are recomputed here.
    # ------------------------------------------------------------
    for r in qpseudo:
        require(
            norm(r["formal_hypothesis_test"]).lower() == "false",
            "QPSEUDO_FORMAL_TEST_INVALID",
        )
        require(
            norm(r["test_used"]).lower() == "false",
            "QPSEUDO_TEST_USED",
        )
        require(
            norm(r["hidden_u_gt_used"]).lower() == "false",
            "QPSEUDO_HIDDEN_U_GT_USED",
        )

        rows.append({
            "variant_id": norm(r["variant_id"]),
            "ablation_run_id": norm(r["ablation_run_id"]),
            "main_run_id": norm(r["main_run_id"]),
            "family": norm(r["family"]),
            "level": norm(r["level"]),
            "seed_index": norm(r["seed_index"]),
            "training_seed": norm(r["training_seed"]),
            "metric_name": "Q_pseudo",
            "variant_value": norm(r["q_pseudo_variant"]),
            "main_same_seed_value":
                norm(r["q_pseudo_main_same_seed"]),
            "paired_difference": norm(r["delta_q_pseudo"]),
            "variant_checkpoint_sha256":
                norm(r["variant_checkpoint_sha256"]),
            "main_checkpoint_sha256":
                norm(r["main_checkpoint_sha256"]),
            "variant_evidence_path":
                norm(r["variant_qpseudo_path"]),
            "main_evidence_path":
                norm(r["main_qpseudo_path"]),
            "evaluator_config_sha256":
                norm(r["evaluator_config_sha256"]),
            "validation_split_sha256":
                norm(r["validation_split_sha256"]),
            "checkpoint_role": norm(r["checkpoint_role"]),
            "dataset": norm(r["dataset"]),
            "test_used": norm(r["test_used"]).lower(),
            "hidden_u_gt_used":
                norm(r["hidden_u_gt_used"]).lower(),
            "pair_valid": "true",
            "formal_hypothesis_test": "false",
            "statistics_semantics":
                norm(r["statistics_semantics"]),
            "representation_authority":
                "S10_ABLATION_SEED_LEVEL_VALUES",
        })

    # ------------------------------------------------------------
    # Canonical cardinality / identity checks
    # ------------------------------------------------------------
    require(len(rows) == 360, f"OUTPUT_ROWS:{len(rows)}")

    variants = {r["variant_id"] for r in rows}
    metrics = {r["metric_name"] for r in rows}
    seeds = {int(r["training_seed"]) for r in rows}

    require(
        variants == EXPECTED_VARIANTS,
        f"VARIANT_SET:{sorted(variants)}",
    )
    require(
        metrics == EXPECTED_METRICS,
        f"METRIC_SET:{sorted(metrics)}",
    )
    require(len(seeds) == 10, f"UNIQUE_SEEDS:{len(seeds)}")

    keys = {
        (
            r["variant_id"],
            r["training_seed"],
            r["metric_name"],
        )
        for r in rows
    }

    require(
        len(keys) == 360,
        f"NON_UNIQUE_VARIANT_SEED_METRIC_KEYS:{len(keys)}",
    )

    counts: dict[tuple[str, str], int] = {}

    for r in rows:
        key = (r["variant_id"], r["metric_name"])
        counts[key] = counts.get(key, 0) + 1

    require(
        len(counts) == 36,
        f"VARIANT_METRIC_GROUP_COUNT:{len(counts)}",
    )
    require(
        all(v == 10 for v in counts.values()),
        f"VARIANT_METRIC_SEED_COUNTS:{counts}",
    )

    require(
        all(r["pair_valid"] == "true" for r in rows),
        "PAIR_VALID_FAILURE",
    )
    require(
        all(
            r["formal_hypothesis_test"] == "false"
            for r in rows
        ),
        "FORMAL_HYPOTHESIS_TEST_FAILURE",
    )
    require(
        all(r["test_used"] == "false" for r in rows),
        "TEST_USED_FAILURE",
    )
    require(
        all(r["hidden_u_gt_used"] == "false" for r in rows),
        "HIDDEN_U_GT_USED_FAILURE",
    )

    rows.sort(
        key=lambda r: (
            r["family"],
            r["variant_id"],
            r["metric_name"],
            int(r["seed_index"]),
        )
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=FIELDNAMES,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=== S10.1 ABLATION SEED LEVEL PRODUCER ===")
    print("STATUS=PASS")
    print("DETECTION_SOURCE_SHA256=" + detection_sha)
    print("QPSEUDO_SOURCE_SHA256=" + qpseudo_sha)
    print("ROWS=" + str(len(rows)))
    print("UNIQUE_VARIANTS=" + str(len(variants)))
    print("UNIQUE_SEEDS=" + str(len(seeds)))
    print("UNIQUE_METRICS=" + str(len(metrics)))
    print(
        "METRICS=" +
        ",".join(sorted(metrics))
    )
    print("VARIANT_METRIC_GROUPS=" + str(len(counts)))
    print("ROWS_PER_VARIANT_METRIC=10")
    print("FORMAL_HYPOTHESIS_TEST=FALSE")
    print("TEST_USED=FALSE")
    print("HIDDEN_U_GT_USED=FALSE")
    print("STATISTICS_REIMPLEMENTED=FALSE")
    print(
        "OUTPUT=" +
        str(OUTPUT.relative_to(ROOT))
    )
    print("OUTPUT_SHA256=" + sha256(OUTPUT))


if __name__ == "__main__":
    main()