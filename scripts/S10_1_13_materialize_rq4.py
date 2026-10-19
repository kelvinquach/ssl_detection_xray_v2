#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0.0"
RQ = "RQ4"
STATISTICS_SEMANTICS = "FROZEN_S6_REUSE"
REPRESENTATION_AUTHORITY = "S10_IMPLEMENTATION_SCHEMA_MINIMAL_FROM_RQ4_CONTRACT"

EXPECTED_PYTHON = "3.10.13"

RAW_SHA256 = {
    "rq4_analysis_dataset.raw.csv": "fa2e8ab0bd8905995eec91b1212d0127708a4cfbe24e28e90ed6a8d745c6f847",
    "rq4_cr2_sensitivity.raw.json": "88de43e2ab56212c55c54e45a52828fbb71caa7cd2dde28ba6631c27a75e0da0",
    "rq4_diagnostics.raw.json": "54518d61630c7be74d3f05cdf2023cc248fce693851299be0747356419ec2308",
    "rq4_primary_mixed_model.raw.json": "bb6f92511f938ebd419f5a3138363ebfc8c4a4439b56b1869fa95469f9e271fd",
    "rq4_runtime_environment.raw.json": "e519676fe9d33f976e48031e08c62256aff97a1244fca78482e16393f51d2600",
    "rq4_seed_influence.raw.csv": "d6aaabf1e25c895eb4bf5da6fc0e4a2ca60a1aba65bfdccde355bb9351ef10f5",
    "S10_1_13_compute_rq4.R": "551b371cf8f53e4430640cb4f520b42b2d28842634029b4a943e863075128bfe",
}

SOURCE_ARTIFACT = "artifacts/results/paired_effects.csv"
SOURCE_ARTIFACT_SHA256 = "bad5a179711443636a13bd6501a1d90a7e8c5dadfb16b9c09df74bf502b727bb"

FROZEN_STATISTICS_MANIFEST = "artifacts/frozen/frozen_statistics_manifest.json"
FROZEN_STATISTICS_MANIFEST_SHA256 = "8f4370a9a39c990a26df58adbe64a252edb26c7115632e7caac5a30d01f93582"

STATISTICAL_ENVIRONMENT_REFERENCE = (
    "artifacts/preflight/environment/statistics_runtime_image_manifest.json"
)
STATISTICAL_ENVIRONMENT_REFERENCE_SHA256 = (
    "10a92f95543100f88950cf439d52609e1accefa4cccb2d46aff5f3287e9a4197"
)

STATISTICAL_RUNTIME_PROVENANCE_BRIDGE = (
    "artifacts/preflight/statistics/s6_14_statistical_runtime_provenance_bridge.json"
)
STATISTICAL_RUNTIME_PROVENANCE_BRIDGE_SHA256 = (
    "b01c8df833021e1e686526dc98efd15f4c5b29a4d70fa4d1ae5522ebbc5452ba"
)

FROZEN_STATISTICS_IMAGE = "quachthainguyen/sslxray-statistics:s6-r4.5.3-20260423"
FROZEN_STATISTICS_IMMUTABLE_REFERENCE = (
    "quachthainguyen/sslxray-statistics@"
    "sha256:bb99a07178ba662bd2e84c5e03421007668b7e35856c83cb6ee816cffab15d37"
)
FROZEN_STATISTICS_OCI_INDEX_DIGEST = (
    "sha256:bb99a07178ba662bd2e84c5e03421007668b7e35856c83cb6ee816cffab15d37"
)

RQ4_MIXED_MODEL_SOURCE_SHA256 = (
    "6c92eaf928c203cd68917b1a1ebc7cd3fd0b6977e541e9e3df7962a3c08ff1d4"
)
RQ4_CR2_SOURCE_SHA256 = (
    "9e9a21faf87c851ed9d74151d4b6c2860d70f2b143fb32ad4662dd2b0c049668"
)

EXPECTED_SEEDS = [
    (1, 204886845),
    (2, 1480646854),
    (3, 1798418854),
    (4, 2045683682),
    (5, 1814859839),
    (6, 1603952859),
    (7, 1878351743),
    (8, 875651179),
    (9, 477581743),
    (10, 869675675),
]

EXPECTED_CELLS = [
    "R50__1pct",
    "R50__5pct",
    "R50__10pct",
    "R50__20pct",
    "Swin-T__1pct",
    "Swin-T__5pct",
    "Swin-T__10pct",
    "Swin-T__20pct",
]

ANALYSIS_DATASET_FIELDS = [
    "architecture",
    "budget",
    "architecture_budget_cell",
    "seed_index",
    "training_seed",
    "Q_pseudo",
    "Q_WC",
    "X_per_1_pseudo_AP_point",
    "delta_mAP_test",
]

SEED_INFLUENCE_RAW_FIELDS = [
    "omitted_seed_index",
    "omitted_training_seed",
    "retained_seed_count",
    "row_count",
    "effect_estimate",
    "delta_from_full",
    "effect_direction",
    "convergence_status",
    "singularity_status",
    "max_abs_within_cell_center_sum",
]

SEED_INFLUENCE_CANONICAL_FIELDS = SEED_INFLUENCE_RAW_FIELDS + [
    "generation_script_sha256",
    "statistical_environment_reference",
    "statistical_environment_reference_sha256",
]

EXPECTED_PRIMARY = {
    "formula": "Y ~ X + architecture_budget_cell + (1 | training_seed)",
    "estimation": "REML",
    "fixed_effect_inference": "Kenward-Roger",
    "beta_Q_alternative": "greater",
    "effect_estimate": 0.0034226305921420197,
    "standard_error": 0.0031155203929546719,
    "df": 66.27190138425928,
    "t_statistic": 1.0985742862995973,
    "ci_low": -0.0027972302518094521,
    "ci_high": 0.0096424914360934907,
    "one_sided_p_value": 0.13796448268749345,
    "convergence_status": "CONVERGED",
    "singularity_status": "NON_SINGULAR",
    "n_rows": 80,
    "n_training_seeds": 10,
    "n_architecture_budget_cells": 8,
}

EXPECTED_CR2 = {
    "analysis": "RQ4_CR2_SATTERTHWAITE_SENSITIVITY",
    "sensitivity_only": True,
    "replaces_primary_kr": False,
    "formula": "Y ~ X_per_1_pseudo_AP_point + architecture_budget_cell",
    "covariance_type": "CR2",
    "cluster_variable": "training_seed",
    "cluster_count": 10,
    "architecture_budget_cell_count": 8,
    "row_count": 80,
    "df_method": "Satterthwaite",
    "effect_estimate": 0.0035839160839160854,
    "standard_error_cr2": 0.0021795707836599759,
    "t_statistic": 1.6443219512687295,
    "df_satterthwaite": 6.0092305775938701,
    "p_one_sided_greater": 0.075568754592777967,
    "ci_lower": -0.001747316321369089,
    "ci_upper": 0.0089151484892012606,
}

EXPECTED_RUNTIME = {
    "R": "4.5.3",
    "lme4": "2.0.1",
    "lmerTest": "3.2.1",
    "pbkrtest": "0.5.5",
    "jsonlite": "2.0.0",
    "clubSandwich": "0.6.2",
}


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        fail(f"Expected JSON object: {path}")
    return obj


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    return fields, rows


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        obj,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)


def assert_close(name: str, observed: Any, expected: float, tol: float = 1e-14) -> None:
    try:
        value = float(observed)
    except (TypeError, ValueError) as e:
        fail(f"{name}: not numeric: {observed!r}: {e}")
    if not math.isfinite(value):
        fail(f"{name}: non-finite: {value}")
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=tol):
        fail(f"{name}: observed={value!r}, expected={expected!r}")


def verify_file(path: Path, expected_sha: str) -> None:
    if not path.is_file():
        fail(f"Missing required file: {path}")
    observed = sha256_file(path)
    if observed != expected_sha:
        fail(f"SHA256 mismatch for {path}: {observed} != {expected_sha}")


def verify_raw_inputs(raw_dir: Path) -> None:
    for name, expected_sha in RAW_SHA256.items():
        verify_file(raw_dir / name, expected_sha)


def verify_repo_provenance(repo_root: Path) -> None:
    verify_file(
        repo_root / SOURCE_ARTIFACT,
        SOURCE_ARTIFACT_SHA256,
    )
    verify_file(
        repo_root / FROZEN_STATISTICS_MANIFEST,
        FROZEN_STATISTICS_MANIFEST_SHA256,
    )
    verify_file(
        repo_root / STATISTICAL_ENVIRONMENT_REFERENCE,
        STATISTICAL_ENVIRONMENT_REFERENCE_SHA256,
    )
    verify_file(
        repo_root / STATISTICAL_RUNTIME_PROVENANCE_BRIDGE,
        STATISTICAL_RUNTIME_PROVENANCE_BRIDGE_SHA256,
    )


def validate_runtime_environment(obj: dict[str, Any]) -> None:
    if str(obj.get("R")) != EXPECTED_RUNTIME["R"]:
        fail(f"Observed R version mismatch: {obj.get('R')!r}")
    packages = obj.get("packages")
    if not isinstance(packages, dict):
        fail("runtime_environment.packages is not an object")
    for name in ["lme4", "lmerTest", "pbkrtest", "jsonlite", "clubSandwich"]:
        if str(packages.get(name)) != EXPECTED_RUNTIME[name]:
            fail(
                f"Observed package version mismatch for {name}: "
                f"{packages.get(name)!r} != {EXPECTED_RUNTIME[name]!r}"
            )


def validate_analysis_dataset(
    fields: list[str],
    rows: list[dict[str, str]],
) -> None:
    if fields != ANALYSIS_DATASET_FIELDS:
        fail(f"Analysis dataset header mismatch: {fields}")
    if len(rows) != 80:
        fail(f"Analysis dataset must have 80 rows; got {len(rows)}")

    cell_counts = Counter()
    seed_counts = Counter()
    seed_map: dict[int, int] = {}
    cell_center_sums: defaultdict[str, float] = defaultdict(float)

    for row in rows:
        cell = row["architecture_budget_cell"]
        if cell not in EXPECTED_CELLS:
            fail(f"Unexpected architecture_budget_cell: {cell!r}")
        idx = int(row["seed_index"])
        seed = int(row["training_seed"])
        expected_seed = dict(EXPECTED_SEEDS).get(idx)
        if seed != expected_seed:
            fail(f"Seed mapping mismatch: seed_index={idx}, training_seed={seed}")
        seed_map[idx] = seed
        cell_counts[cell] += 1
        seed_counts[seed] += 1

        q = float(row["Q_pseudo"])
        qwc = float(row["Q_WC"])
        x = float(row["X_per_1_pseudo_AP_point"])
        y = float(row["delta_mAP_test"])
        for name, value in [("Q_pseudo", q), ("Q_WC", qwc), ("X", x), ("Y", y)]:
            if not math.isfinite(value):
                fail(f"Non-finite {name} in analysis dataset")
        if not math.isclose(x, qwc / 0.01, rel_tol=0.0, abs_tol=1e-14):
            fail(f"X scaling mismatch in row: {row}")
        cell_center_sums[cell] += qwc

    if list(sorted(seed_map.items())) != EXPECTED_SEEDS:
        fail("Ordered seed mapping mismatch")
    if any(cell_counts[c] != 10 for c in EXPECTED_CELLS):
        fail(f"Cell counts mismatch: {dict(cell_counts)}")
    if any(seed_counts[s] != 8 for _, s in EXPECTED_SEEDS):
        fail(f"Seed counts mismatch: {dict(seed_counts)}")
    max_center_error = max(abs(v) for v in cell_center_sums.values())
    if max_center_error > 1e-12:
        fail(f"Within-cell centering error too large: {max_center_error}")


def validate_primary(obj: dict[str, Any]) -> None:
    for key in [
        "formula",
        "estimation",
        "fixed_effect_inference",
        "beta_Q_alternative",
        "convergence_status",
        "singularity_status",
    ]:
        if obj.get(key) != EXPECTED_PRIMARY[key]:
            fail(f"Primary {key} mismatch: {obj.get(key)!r}")

    for key in ["n_rows", "n_training_seeds", "n_architecture_budget_cells"]:
        if int(obj.get(key)) != EXPECTED_PRIMARY[key]:
            fail(f"Primary {key} mismatch: {obj.get(key)!r}")

    for key in [
        "effect_estimate",
        "standard_error",
        "df",
        "t_statistic",
        "one_sided_p_value",
    ]:
        assert_close(f"primary.{key}", obj.get(key), EXPECTED_PRIMARY[key])

    ci = obj.get("two_sided_95_ci")
    if not isinstance(ci, dict):
        fail("Primary two_sided_95_ci missing/not object")
    assert_close("primary.ci.low", ci.get("low"), EXPECTED_PRIMARY["ci_low"])
    assert_close("primary.ci.high", ci.get("high"), EXPECTED_PRIMARY["ci_high"])

    messages = obj.get("convergence_messages")
    if messages != []:
        fail(f"Expected no primary convergence messages; got {messages!r}")


def validate_diagnostics(
    obj: dict[str, Any],
    seed_rows: list[dict[str, str]],
) -> None:
    if obj.get("diagnostics_not_used_for_model_selection") is not True:
        fail("Diagnostics model-selection guard is not TRUE")
    if obj.get("seed_influence_not_used_for_primary_seed_exclusion") is not True:
        fail("Seed-exclusion guard is not TRUE")

    rvf = obj.get("residual_vs_fitted")
    qq = obj.get("qq")
    influence = obj.get("seed_influence")
    if not isinstance(rvf, dict) or int(rvf.get("point_count", -1)) != 80:
        fail("Residual-vs-fitted point_count mismatch")
    if not isinstance(qq, dict) or int(qq.get("point_count", -1)) != 80:
        fail("Q-Q point_count mismatch")
    if not isinstance(influence, dict):
        fail("seed_influence diagnostics missing")

    locks = {
        "analysis_count": 10,
        "retained_seed_count_per_analysis": 9,
        "row_count_per_analysis": 72,
        "converged_count": 9,
        "singular_count": 1,
        "positive_effect_count": 10,
        "negative_effect_count": 0,
        "zero_effect_count": 0,
    }
    for key, expected in locks.items():
        if int(influence.get(key, -999)) != expected:
            fail(f"Diagnostics seed_influence.{key} mismatch: {influence.get(key)!r}")

    bool_locks = {
        "loso_p_value_computed": False,
        "loso_ci_computed": False,
        "primary_model_unchanged": True,
        "primary_seed_exclusion_allowed": False,
        "predictor_recentered_within_architecture_budget_cell_for_each_refit": True,
    }
    for key, expected in bool_locks.items():
        if influence.get(key) is not expected:
            fail(f"Diagnostics seed_influence.{key} mismatch: {influence.get(key)!r}")

    assert_close(
        "diagnostics.seed_influence.min_effect_estimate",
        influence.get("min_effect_estimate"),
        0.0020730090569801977,
    )
    assert_close(
        "diagnostics.seed_influence.max_effect_estimate",
        influence.get("max_effect_estimate"),
        0.004474710476938551,
    )
    assert_close(
        "diagnostics.seed_influence.max_abs_delta_from_full",
        influence.get("max_abs_delta_from_full"),
        0.001349621535161822,
    )

    if len(seed_rows) != 10:
        fail(f"Seed influence CSV must have 10 rows; got {len(seed_rows)}")
    for expected_idx, expected_seed in EXPECTED_SEEDS:
        row = seed_rows[expected_idx - 1]
        if int(row["omitted_seed_index"]) != expected_idx:
            fail(f"Seed influence row order mismatch at index {expected_idx}")
        if int(row["omitted_training_seed"]) != expected_seed:
            fail(f"Seed influence omitted seed mismatch at index {expected_idx}")
        if int(row["retained_seed_count"]) != 9 or int(row["row_count"]) != 72:
            fail(f"Seed influence retained counts mismatch at index {expected_idx}")
        if row["effect_direction"] != "POSITIVE":
            fail(f"Expected positive effect direction at index {expected_idx}")

    last = seed_rows[-1]
    if last["convergence_status"] != "NOT_CONVERGED":
        fail("Expected omitted seed 10 refit to be NOT_CONVERGED")
    if last["singularity_status"] != "SINGULAR":
        fail("Expected omitted seed 10 refit to be SINGULAR")

    if sum(r["convergence_status"] == "CONVERGED" for r in seed_rows) != 9:
        fail("Seed influence converged-count mismatch")
    if sum(r["singularity_status"] == "SINGULAR" for r in seed_rows) != 1:
        fail("Seed influence singular-count mismatch")


def validate_cr2(obj: dict[str, Any]) -> None:
    for key in [
        "analysis",
        "sensitivity_only",
        "replaces_primary_kr",
        "formula",
        "covariance_type",
        "cluster_variable",
        "df_method",
    ]:
        if obj.get(key) != EXPECTED_CR2[key]:
            fail(f"CR2 {key} mismatch: {obj.get(key)!r}")

    for key in ["cluster_count", "architecture_budget_cell_count", "row_count"]:
        if int(obj.get(key)) != EXPECTED_CR2[key]:
            fail(f"CR2 {key} mismatch: {obj.get(key)!r}")

    hypothesis = obj.get("hypothesis")
    if not isinstance(hypothesis, dict):
        fail("CR2 hypothesis missing/not object")
    if hypothesis.get("alternative") != "beta_Q > 0":
        fail(f"CR2 alternative mismatch: {hypothesis.get('alternative')!r}")
    if hypothesis.get("p_value_sidedness") != "one-sided greater":
        fail(f"CR2 p-value sidedness mismatch: {hypothesis.get('p_value_sidedness')!r}")

    effect = obj.get("effect")
    if not isinstance(effect, dict):
        fail("CR2 effect missing/not object")
    for key in [
        "effect_estimate",
        "standard_error_cr2",
        "t_statistic",
        "df_satterthwaite",
        "p_one_sided_greater",
        "ci_lower",
        "ci_upper",
    ]:
        source_key = "estimate" if key == "effect_estimate" else key
        assert_close(f"cr2.effect.{source_key}", effect.get(source_key), EXPECTED_CR2[key])


def provenance_block(
    producer_sha256: str,
    runtime_environment: dict[str, Any],
) -> dict[str, Any]:
    return {
        "generation_script": "scripts/S10_1_13_materialize_rq4.py",
        "generation_script_sha256": producer_sha256,
        "source_artifact": SOURCE_ARTIFACT,
        "source_sha256": SOURCE_ARTIFACT_SHA256,
        "raw_evidence_sha256": {
            key: RAW_SHA256[key]
            for key in sorted(RAW_SHA256)
        },
        "frozen_statistics_sources": {
            "src/statistics/rq4_mixed_model.R": RQ4_MIXED_MODEL_SOURCE_SHA256,
            "src/statistics/rq4_cr2_sensitivity.R": RQ4_CR2_SOURCE_SHA256,
            "scripts/S10_1_13_compute_rq4.R": RAW_SHA256["S10_1_13_compute_rq4.R"],
            FROZEN_STATISTICS_MANIFEST: FROZEN_STATISTICS_MANIFEST_SHA256,
        },
        "statistical_environment_reference": STATISTICAL_ENVIRONMENT_REFERENCE,
        "statistical_environment_reference_sha256": (
            STATISTICAL_ENVIRONMENT_REFERENCE_SHA256
        ),
        "statistical_runtime_provenance_bridge": (
            STATISTICAL_RUNTIME_PROVENANCE_BRIDGE
        ),
        "statistical_runtime_provenance_bridge_sha256": (
            STATISTICAL_RUNTIME_PROVENANCE_BRIDGE_SHA256
        ),
        "frozen_statistics_image": FROZEN_STATISTICS_IMAGE,
        "frozen_statistics_immutable_reference": (
            FROZEN_STATISTICS_IMMUTABLE_REFERENCE
        ),
        "frozen_statistics_oci_index_digest": (
            FROZEN_STATISTICS_OCI_INDEX_DIGEST
        ),
        "observed_statistical_runtime": runtime_environment,
        "statistics_semantics": STATISTICS_SEMANTICS,
        "representation_authority": REPRESENTATION_AUTHORITY,
    }


def make_payload(
    *,
    analysis_role: str,
    scientific_contract: dict[str, Any],
    result: dict[str, Any],
    producer_sha256: str,
    runtime_environment: dict[str, Any],
) -> dict[str, Any]:
    p = {
        "schema_version": SCHEMA_VERSION,
        "rq": RQ,
        "analysis_role": analysis_role,
        "scientific_contract": scientific_contract,
        "result": result,
    }
    p.update(provenance_block(producer_sha256, runtime_environment))
    return p


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    if platform.python_version() != EXPECTED_PYTHON:
        fail(
            f"Canonical producer requires Python {EXPECTED_PYTHON}; "
            f"observed {platform.python_version()}"
        )

    repo_root = Path(args.repo_root).resolve()
    raw_dir = Path(args.raw_dir).resolve()
    output_root = Path(args.output_root).resolve()

    if output_root.exists():
        fail(f"Output root already exists: {output_root}")

    producer_path = Path(__file__).resolve()
    producer_sha256 = sha256_file(producer_path)

    verify_raw_inputs(raw_dir)
    verify_repo_provenance(repo_root)

    dataset_fields, dataset_rows = read_csv(
        raw_dir / "rq4_analysis_dataset.raw.csv"
    )
    seed_fields, seed_rows = read_csv(
        raw_dir / "rq4_seed_influence.raw.csv"
    )
    primary = load_json(raw_dir / "rq4_primary_mixed_model.raw.json")
    diagnostics = load_json(raw_dir / "rq4_diagnostics.raw.json")
    cr2 = load_json(raw_dir / "rq4_cr2_sensitivity.raw.json")
    runtime_environment = load_json(
        raw_dir / "rq4_runtime_environment.raw.json"
    )

    validate_runtime_environment(runtime_environment)
    validate_analysis_dataset(dataset_fields, dataset_rows)
    if seed_fields != SEED_INFLUENCE_RAW_FIELDS:
        fail(f"Seed influence header mismatch: {seed_fields}")
    validate_primary(primary)
    validate_diagnostics(diagnostics, seed_rows)
    validate_cr2(cr2)

    rq4_dir = output_root / "analysis" / "rq4"
    robust_dir = output_root / "analysis" / "robustness"

    write_csv(
        rq4_dir / "rq4_analysis_dataset.csv",
        ANALYSIS_DATASET_FIELDS,
        dataset_rows,
    )

    primary_payload = make_payload(
        analysis_role="PRIMARY",
        scientific_contract={
            "question": (
                "Within architecture-budget cells, is higher fixed-validation "
                "EMA-Teacher LAST-checkpoint Q_pseudo positively associated "
                "with downstream paired test delta mAP?"
            ),
            "association_only_not_causal": True,
            "formula": "Y ~ X + architecture_budget_cell + (1 | training_seed)",
            "estimation": "REML",
            "fixed_effect_inference": "Kenward-Roger",
            "beta_Q_alternative": "greater",
            "effect_unit": (
                "downstream AP-point change per +1 pseudo AP point "
                "within architecture-budget cell"
            ),
        },
        result=primary,
        producer_sha256=producer_sha256,
        runtime_environment=runtime_environment,
    )
    write_json(
        rq4_dir / "rq4_primary_mixed_model.json",
        primary_payload,
    )

    diagnostics_payload = make_payload(
        analysis_role="DIAGNOSTIC_ROBUSTNESS",
        scientific_contract={
            "diagnostics": [
                "convergence",
                "singularity",
                "residual-versus-fitted pattern",
                "Q-Q plot",
                "influence of each training seed",
            ],
            "diagnostics_not_used_for_model_selection": True,
            "seed_influence_not_used_for_primary_seed_exclusion": True,
            "seed_influence_refit_count": 10,
            "seed_influence_semantics": (
                "Temporarily omit one training seed, recenter Q_pseudo within "
                "architecture-budget cell on the retained 9 seeds, and refit "
                "the same locked REML mixed model; inspect direction and "
                "magnitude only; do not compute LOSO p-values or LOSO CIs."
            ),
        },
        result=diagnostics,
        producer_sha256=producer_sha256,
        runtime_environment=runtime_environment,
    )
    write_json(
        rq4_dir / "rq4_diagnostics.json",
        diagnostics_payload,
    )

    cr2_payload = make_payload(
        analysis_role="SENSITIVITY",
        scientific_contract={
            "sensitivity_only": True,
            "replaces_primary_kr": False,
            "formula": "Y ~ X_per_1_pseudo_AP_point + architecture_budget_cell",
            "covariance_type": "CR2",
            "cluster_variable": "training_seed",
            "df_method": "Satterthwaite",
            "beta_Q_alternative": "greater",
            "report_divergence_from_primary_not_favorable_model_selection": True,
        },
        result=cr2,
        producer_sha256=producer_sha256,
        runtime_environment=runtime_environment,
    )
    write_json(
        rq4_dir / "rq4_cr2_sensitivity.json",
        cr2_payload,
    )
    write_json(
        robust_dir / "rq4_cr2_sensitivity.json",
        cr2_payload,
    )

    seed_canonical_rows: list[dict[str, Any]] = []
    for row in seed_rows:
        out = dict(row)
        out["generation_script_sha256"] = producer_sha256
        out["statistical_environment_reference"] = (
            STATISTICAL_ENVIRONMENT_REFERENCE
        )
        out["statistical_environment_reference_sha256"] = (
            STATISTICAL_ENVIRONMENT_REFERENCE_SHA256
        )
        seed_canonical_rows.append(out)

    write_csv(
        robust_dir / "rq4_seed_influence.csv",
        SEED_INFLUENCE_CANONICAL_FIELDS,
        seed_canonical_rows,
    )

    outputs = [
        rq4_dir / "rq4_analysis_dataset.csv",
        rq4_dir / "rq4_primary_mixed_model.json",
        rq4_dir / "rq4_diagnostics.json",
        rq4_dir / "rq4_cr2_sensitivity.json",
        robust_dir / "rq4_cr2_sensitivity.json",
        robust_dir / "rq4_seed_influence.csv",
    ]

    print("=== S10.1.13 RQ4 CANONICAL MATERIALIZATION ===")
    print(f"PYTHON_VERSION={platform.python_version()}")
    print(f"GENERATION_SCRIPT_SHA256={producer_sha256}")
    print("ANALYSIS_DATASET_ROWS=80")
    print(f"PRIMARY_EFFECT_ESTIMATE={primary['effect_estimate']}")
    print(f"PRIMARY_ONE_SIDED_P={primary['one_sided_p_value']}")
    print(f"PRIMARY_CONVERGENCE={primary['convergence_status']}")
    print(f"PRIMARY_SINGULARITY={primary['singularity_status']}")
    print(f"SEED_INFLUENCE_ANALYSES={len(seed_rows)}")
    print(
        "SEED_INFLUENCE_NOT_CONVERGED_COUNT="
        + str(sum(r["convergence_status"] != "CONVERGED" for r in seed_rows))
    )
    print(
        "SEED_INFLUENCE_SINGULAR_COUNT="
        + str(sum(r["singularity_status"] == "SINGULAR" for r in seed_rows))
    )
    print(
        "SEED_INFLUENCE_POSITIVE_COUNT="
        + str(sum(r["effect_direction"] == "POSITIVE" for r in seed_rows))
    )
    print(f"CR2_EFFECT_ESTIMATE={cr2['effect']['estimate']}")
    print(f"CR2_ONE_SIDED_P={cr2['effect']['p_one_sided_greater']}")
    print("=== OUTPUT SHA256 ===")
    for path in outputs:
        print(f"{sha256_file(path)}  {path.relative_to(output_root).as_posix()}")
    print("S10_1_13_RQ4_CANONICAL_MATERIALIZATION=PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"S10_1_13_RQ4_CANONICAL_MATERIALIZATION=FAIL: {exc}", file=sys.stderr)
        raise
