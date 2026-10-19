from __future__ import annotations

import csv
import hashlib
import math
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import scipy


ROOT = Path(__file__).resolve().parents[1]

VARIANT_SUMMARY = (
    ROOT / "analysis/ablation/ablation_variant_summary.csv"
)

DETECTION_EFFECTS = (
    ROOT / "artifacts/results/ablation/ablation_effects.csv"
)

QPSEUDO_EFFECTS = (
    ROOT / "analysis/ablation/ablation_qpseudo_effects.csv"
)

OUTPUT_CONFIDENCE = (
    ROOT / "analysis/ablation/confidence_summary.csv"
)

OUTPUT_AUGMENTATION = (
    ROOT / "analysis/ablation/augmentation_summary.csv"
)

OUTPUT_REGRESSION = (
    ROOT / "analysis/ablation/regression_filter_summary.csv"
)

EXPECTED_VARIANT_SUMMARY_SHA256 = (
    "838a828a6e151688125763e0ddcae39a91f833a9cdc5158eb0f6ce0a0ad4b547"
)

EXPECTED_DETECTION_EFFECTS_SHA256 = (
    "23ca736c91ab27cb461c63c97e896cdb3017da8a7a114f1cb9693ced41d281f6"
)

EXPECTED_QPSEUDO_EFFECTS_SHA256 = (
    "d67f2a595fa6f9d8949ac7022c3958818b2c0798e1f1c612aa04f7433f2877b8"
)

FROZEN_SEED_SUMMARY_SHA256 = (
    "2a404ac0f6097cad16816b550fcd1c9ca240fab5535854dc0fd5e4433e49ff3f"
)

FROZEN_SCIPY_VERSION = "1.15.3"

METRICS = (
    "validation_bbox_mAP_50_95",
    "validation_AP50",
    "validation_AP75",
    "Q_pseudo",
)

METRIC_ORDER = {
    "validation_bbox_mAP_50_95": 0,
    "validation_AP50": 1,
    "validation_AP75": 2,
    "Q_pseudo": 3,
}

FAMILIES = {
    "CONFIDENCE_THRESHOLD": {
        "non_main": ("C0", "C1", "C2", "C3"),
        "main_alias": "C4",
        "main_level": "0.90",
        "output": OUTPUT_CONFIDENCE,
    },
    "STRONG_PHOTOMETRIC_AUGMENTATION": {
        "non_main": ("A0", "A1", "A2"),
        "main_alias": "A3",
        "main_level": "brightness+contrast",
        "output": OUTPUT_AUGMENTATION,
    },
    "REGRESSION_PSEUDO_LABEL_FILTERING": {
        "non_main": ("F0", "F1"),
        "main_alias": "F2",
        "main_level": "initial_score_gt_0.50_and_reg_uncertainty_lt_0.02",
        "output": OUTPUT_REGRESSION,
    },
}

VARIANT_ORDER = {
    "C0": 0, "C1": 1, "C2": 2, "C3": 3, "C4": 4,
    "A0": 0, "A1": 1, "A2": 2, "A3": 3,
    "F0": 0, "F1": 1, "F2": 2,
}

FIELDNAMES = [
    "variant_id",
    "family",
    "level",
    "is_main_reference",
    "scientific_alias",
    "metric_name",
    "n_seeds",
    "variant_mean",
    "variant_sample_SD_ddof1",
    "main_mean",
    "paired_mean_difference",
    "paired_difference_sample_SD_ddof1",
    "paired_difference_two_sided_95CI_low",
    "paired_difference_two_sided_95CI_high",
    "ci_level",
    "formal_hypothesis_test",
    "p_value_computed",
    "multiplicity_family_added",
    "replication_unit",
    "statistics_semantics",
    "source_artifact",
    "source_artifact_sha256",
    "representation_authority",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def norm(value) -> str:
    return str(value if value is not None else "").strip()


def finite_float(value, label: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{label}:NON_NUMERIC:{value!r}"
        ) from exc

    require(
        math.isfinite(x),
        f"{label}:NON_FINITE:{x!r}",
    )
    return x


def load_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"MISSING_FILE:{path}")
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def write_csv(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
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


def main() -> None:
    variant_sha = sha256(VARIANT_SUMMARY)
    detection_sha = sha256(DETECTION_EFFECTS)
    qpseudo_sha = sha256(QPSEUDO_EFFECTS)

    require(
        variant_sha == EXPECTED_VARIANT_SUMMARY_SHA256,
        f"VARIANT_SUMMARY_SHA_MISMATCH:{variant_sha}",
    )
    require(
        detection_sha == EXPECTED_DETECTION_EFFECTS_SHA256,
        f"DETECTION_EFFECTS_SHA_MISMATCH:{detection_sha}",
    )
    require(
        qpseudo_sha == EXPECTED_QPSEUDO_EFFECTS_SHA256,
        f"QPSEUDO_EFFECTS_SHA_MISMATCH:{qpseudo_sha}",
    )

    sys.path.insert(0, str(ROOT))

    from src.statistics.seed_summary import (
        OFFICIAL_SEED_COUNT,
        OFFICIAL_TRAINING_SEEDS,
        REPLICATION_UNIT,
        SAMPLE_SD_DDOF,
        summarize_seed_runs,
    )

    require(
        sha256(ROOT / "src/statistics/seed_summary.py")
        == FROZEN_SEED_SUMMARY_SHA256,
        "FROZEN_SEED_SUMMARY_SHA256_MISMATCH",
    )

    require(
        scipy.__version__ == FROZEN_SCIPY_VERSION,
        f"SCIPY_VERSION_MISMATCH:"
        f"{scipy.__version__}!={FROZEN_SCIPY_VERSION}",
    )

    require(
        OFFICIAL_SEED_COUNT == 10,
        "OFFICIAL_SEED_COUNT_NOT_10",
    )
    require(
        SAMPLE_SD_DDOF == 1,
        "SAMPLE_SD_DDOF_NOT_1",
    )

    variant_rows = load_csv(VARIANT_SUMMARY)
    detection_rows = load_csv(DETECTION_EFFECTS)
    qpseudo_rows = load_csv(QPSEUDO_EFFECTS)

    require(
        len(variant_rows) == 36,
        f"VARIANT_SUMMARY_ROWS:{len(variant_rows)}",
    )
    require(
        len(detection_rows) == 270,
        f"DETECTION_ROWS:{len(detection_rows)}",
    )
    require(
        len(qpseudo_rows) == 90,
        f"QPSEUDO_ROWS:{len(qpseudo_rows)}",
    )

    # --------------------------------------------------------
    # Recover canonical Main same-seed values from the already
    # audited paired artifacts. Every value is repeated across
    # the 9 non-Main variants and must be identical.
    # --------------------------------------------------------
    main_detection = defaultdict(list)

    for r in detection_rows:
        key = (
            int(r["training_seed"]),
            norm(r["metric_name"]),
        )
        main_detection[key].append(
            norm(r["main_same_seed_value"])
        )

    require(
        len(main_detection) == 30,
        f"MAIN_DETECTION_GROUPS:{len(main_detection)}",
    )

    main_metric_values: dict[
        str,
        OrderedDict[int, float]
    ] = {}

    for metric in (
        "validation_bbox_mAP_50_95",
        "validation_AP50",
        "validation_AP75",
    ):
        seed_values = OrderedDict()

        for seed in OFFICIAL_TRAINING_SEEDS:
            vals = main_detection[(seed, metric)]

            require(
                len(vals) == 9,
                f"{metric}:{seed}:REPETITIONS:{len(vals)}",
            )
            require(
                len(set(vals)) == 1,
                f"{metric}:{seed}:MAIN_VALUE_NOT_UNIQUE",
            )

            seed_values[seed] = finite_float(
                vals[0],
                f"{metric}:{seed}:MAIN_VALUE",
            )

        main_metric_values[metric] = seed_values

    main_qpseudo = defaultdict(list)

    for r in qpseudo_rows:
        seed = int(r["training_seed"])
        main_qpseudo[seed].append(
            norm(r["q_pseudo_main_same_seed"])
        )

    require(
        len(main_qpseudo) == 10,
        f"MAIN_QPSEUDO_GROUPS:{len(main_qpseudo)}",
    )

    q_seed_values = OrderedDict()

    for seed in OFFICIAL_TRAINING_SEEDS:
        vals = main_qpseudo[seed]

        require(
            len(vals) == 9,
            f"Q_PSEUDO:{seed}:REPETITIONS:{len(vals)}",
        )
        require(
            len(set(vals)) == 1,
            f"Q_PSEUDO:{seed}:MAIN_VALUE_NOT_UNIQUE",
        )

        q_seed_values[seed] = finite_float(
            vals[0],
            f"Q_PSEUDO:{seed}:MAIN_VALUE",
        )

    main_metric_values["Q_pseudo"] = q_seed_values

    # --------------------------------------------------------
    # Main summaries: same canonical Main values are reused for
    # C4 / A3 / F2. Main vs Main paired difference is exactly 0.
    # --------------------------------------------------------
    main_summaries = {}

    for metric in METRICS:
        summary = summarize_seed_runs(
            main_metric_values[metric]
        )

        require(
            int(summary["n"]) == 10,
            f"{metric}:MAIN_N_INVALID",
        )

        main_summaries[metric] = summary

    # --------------------------------------------------------
    # Index existing 36 non-Main summaries.
    # --------------------------------------------------------
    variant_index = {}

    for r in variant_rows:
        key = (
            norm(r["family"]),
            norm(r["variant_id"]),
            norm(r["metric_name"]),
        )

        require(
            key not in variant_index,
            f"DUPLICATE_VARIANT_SUMMARY:{key}",
        )

        require(
            norm(r["formal_hypothesis_test"]).lower()
            == "false",
            f"{key}:FORMAL_TEST_INVALID",
        )
        require(
            norm(r["p_value_computed"]).lower()
            == "false",
            f"{key}:P_VALUE_INVALID",
        )
        require(
            norm(r["multiplicity_family_added"]).lower()
            == "false",
            f"{key}:MULTIPLICITY_INVALID",
        )

        variant_index[key] = r

    # --------------------------------------------------------
    # Materialize each family:
    # existing non-Main rows + scientific Main alias.
    # --------------------------------------------------------
    output_counts = {}

    for family, spec in FAMILIES.items():
        out_rows = []

        for variant_id in spec["non_main"]:
            for metric in METRICS:
                key = (
                    family,
                    variant_id,
                    metric,
                )

                require(
                    key in variant_index,
                    f"MISSING_VARIANT_SUMMARY:{key}",
                )

                src = variant_index[key]

                out_rows.append({
                    "variant_id":
                        variant_id,
                    "family":
                        family,
                    "level":
                        norm(src["level"]),
                    "is_main_reference":
                        "false",
                    "scientific_alias":
                        "",
                    "metric_name":
                        metric,
                    "n_seeds":
                        norm(src["n_seeds"]),
                    "variant_mean":
                        norm(src["variant_mean"]),
                    "variant_sample_SD_ddof1":
                        norm(src["variant_sample_SD_ddof1"]),
                    "main_mean":
                        norm(src["main_mean"]),
                    "paired_mean_difference":
                        norm(src["paired_mean_difference"]),
                    "paired_difference_sample_SD_ddof1":
                        norm(src[
                            "paired_difference_sample_SD_ddof1"
                        ]),
                    "paired_difference_two_sided_95CI_low":
                        norm(src[
                            "paired_difference_two_sided_95CI_low"
                        ]),
                    "paired_difference_two_sided_95CI_high":
                        norm(src[
                            "paired_difference_two_sided_95CI_high"
                        ]),
                    "ci_level":
                        norm(src["ci_level"]),
                    "formal_hypothesis_test":
                        "false",
                    "p_value_computed":
                        "false",
                    "multiplicity_family_added":
                        "false",
                    "replication_unit":
                        norm(src["replication_unit"]),
                    "statistics_semantics":
                        norm(src["statistics_semantics"]),
                    "source_artifact":
                        "analysis/ablation/ablation_variant_summary.csv",
                    "source_artifact_sha256":
                        variant_sha,
                    "representation_authority":
                        "S10_ABLATION_FAMILY_SUMMARY",
                })

        main_alias = spec["main_alias"]

        for metric in METRICS:
            summary = main_summaries[metric]

            mean = float(summary["mean"])
            sample_sd = float(summary["sample_sd"])

            out_rows.append({
                "variant_id":
                    main_alias,
                "family":
                    family,
                "level":
                    spec["main_level"],
                "is_main_reference":
                    "true",
                "scientific_alias":
                    "C4|A3|F2",
                "metric_name":
                    metric,
                "n_seeds":
                    "10",
                "variant_mean":
                    repr(mean),
                "variant_sample_SD_ddof1":
                    repr(sample_sd),
                "main_mean":
                    repr(mean),
                "paired_mean_difference":
                    repr(0.0),
                "paired_difference_sample_SD_ddof1":
                    repr(0.0),
                "paired_difference_two_sided_95CI_low":
                    repr(0.0),
                "paired_difference_two_sided_95CI_high":
                    repr(0.0),
                "ci_level":
                    repr(0.95),
                "formal_hypothesis_test":
                    "false",
                "p_value_computed":
                    "false",
                "multiplicity_family_added":
                    "false",
                "replication_unit":
                    REPLICATION_UNIT,
                "statistics_semantics":
                    "FROZEN_S6_SEED_SUMMARY_MAIN_REFERENCE_NO_TEST",
                "source_artifact":
                    (
                        "artifacts/results/ablation/ablation_effects.csv"
                        if metric != "Q_pseudo"
                        else
                        "analysis/ablation/ablation_qpseudo_effects.csv"
                    ),
                "source_artifact_sha256":
                    (
                        detection_sha
                        if metric != "Q_pseudo"
                        else qpseudo_sha
                    ),
                "representation_authority":
                    "S10_ABLATION_FAMILY_MAIN_REFERENCE",
            })

        out_rows.sort(
            key=lambda r: (
                VARIANT_ORDER[r["variant_id"]],
                METRIC_ORDER[r["metric_name"]],
            )
        )

        expected_rows = (
            len(spec["non_main"]) + 1
        ) * len(METRICS)

        require(
            len(out_rows) == expected_rows,
            f"{family}:OUTPUT_ROWS:{len(out_rows)}"
            f"!={expected_rows}",
        )

        expected_variants = (
            set(spec["non_main"])
            | {main_alias}
        )

        require(
            {r["variant_id"] for r in out_rows}
            == expected_variants,
            f"{family}:VARIANT_SET_INVALID",
        )

        require(
            {r["metric_name"] for r in out_rows}
            == set(METRICS),
            f"{family}:METRIC_SET_INVALID",
        )

        require(
            sum(
                r["is_main_reference"] == "true"
                for r in out_rows
            ) == 4,
            f"{family}:MAIN_REFERENCE_ROW_COUNT_INVALID",
        )

        require(
            all(
                r["formal_hypothesis_test"] == "false"
                for r in out_rows
            ),
            f"{family}:FORMAL_TEST_PRESENT",
        )

        require(
            all(
                r["p_value_computed"] == "false"
                for r in out_rows
            ),
            f"{family}:P_VALUE_PRESENT",
        )

        require(
            all(
                r["multiplicity_family_added"] == "false"
                for r in out_rows
            ),
            f"{family}:MULTIPLICITY_PRESENT",
        )

        output = spec["output"]
        write_csv(output, out_rows)

        output_counts[family] = (
            len(out_rows),
            output,
            sha256(output),
        )

    require(
        output_counts["CONFIDENCE_THRESHOLD"][0] == 20,
        "CONFIDENCE_ROWS_NOT_20",
    )
    require(
        output_counts[
            "STRONG_PHOTOMETRIC_AUGMENTATION"
        ][0] == 16,
        "AUGMENTATION_ROWS_NOT_16",
    )
    require(
        output_counts[
            "REGRESSION_PSEUDO_LABEL_FILTERING"
        ][0] == 12,
        "REGRESSION_ROWS_NOT_12",
    )

    print(
        "=== S10.1 ABLATION FAMILY SUMMARY PRODUCER ==="
    )
    print("STATUS=PASS")
    print("VARIANT_SUMMARY_SHA256=" + variant_sha)
    print("DETECTION_EFFECTS_SHA256=" + detection_sha)
    print("QPSEUDO_EFFECTS_SHA256=" + qpseudo_sha)
    print(
        "FROZEN_SEED_SUMMARY_SHA256="
        + FROZEN_SEED_SUMMARY_SHA256
    )
    print("SCIPY_VERSION=" + scipy.__version__)
    print("MAIN_ALIAS_SOURCE_CANONICAL=PASS")
    print("MAIN_RETRAINED=FALSE")
    print("MAIN_ALIASES=C4|A3|F2")
    print("CONFIDENCE_ROWS=20")
    print("AUGMENTATION_ROWS=16")
    print("REGRESSION_ROWS=12")
    print("TOTAL_ROWS=48")
    print("METRICS_PER_CONFIG=4")
    print("N_SEEDS=10")
    print("SAMPLE_SD_DDOF=1")
    print("FORMAL_HYPOTHESIS_TEST=FALSE")
    print("P_VALUE_COMPUTED=FALSE")
    print("MULTIPLICITY_FAMILY_ADDED=FALSE")
    print("STATISTICS_CORE=FROZEN_S6_REUSE")
    print("STATISTICS_REIMPLEMENTED=FALSE")

    for family in (
        "CONFIDENCE_THRESHOLD",
        "STRONG_PHOTOMETRIC_AUGMENTATION",
        "REGRESSION_PSEUDO_LABEL_FILTERING",
    ):
        count, output, digest = output_counts[family]
        print(
            family
            + "_OUTPUT="
            + str(output.relative_to(ROOT))
        )
        print(
            family
            + "_SHA256="
            + digest
        )


if __name__ == "__main__":
    main()