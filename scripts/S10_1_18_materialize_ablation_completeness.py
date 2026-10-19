from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONFIG_INVENTORY = ROOT / "artifacts/results/ablation/ablation_config_inventory.csv"
RUN_INVENTORY = ROOT / "artifacts/results/ablation/ablation_run_inventory.csv"
MAIN_MAP = ROOT / "artifacts/results/ablation/ablation_main_reference_map.csv"
PAIR_INVENTORY = ROOT / "artifacts/results/ablation/ablation_pair_inventory.csv"
ABLATION_EFFECTS = ROOT / "artifacts/results/ablation/ablation_effects.csv"
QPSEUDO_EFFECTS = ROOT / "analysis/ablation/ablation_qpseudo_effects.csv"

ABLATION_RUN_ROOT = ROOT / "artifacts/runs/ablation"

OUTPUT = (
    ROOT
    / "artifacts/results/ablation/ablation_completeness_report.json"
)

EXPECTED_SHA256 = {
    "ablation_config_inventory.csv":
        "e50cad2b583bde74b65c8460329ae1395ecdf5bc2ecfe9d74cee103e75e9e8f7",
    "ablation_run_inventory.csv":
        "0b354e24c4c918e94488f43b1ba271e2db08eb6a9ef080753e0680f4b63f362a",
    "ablation_main_reference_map.csv":
        "df7002a8ae83d6e55b229a289449c628cda6cb5027031aadb4e626c341ed8621",
    "ablation_pair_inventory.csv":
        "04a289c03cd91dbb9b96b0cfe39c301c925b32863cbb72859c008a0604cf124e",
    "ablation_effects.csv":
        "23ca736c91ab27cb461c63c97e896cdb3017da8a7a114f1cb9693ced41d281f6",
    "ablation_qpseudo_effects.csv":
        "d67f2a595fa6f9d8949ac7022c3958818b2c0798e1f1c612aa04f7433f2877b8",
}

FROZEN_SEEDS = (
    204886845,
    1480646854,
    1798418854,
    2045683682,
    1814859839,
    1603952859,
    1878351743,
    875651179,
    477581743,
    869675675,
)

EXPECTED_NON_MAIN_CONFIGS = {
    "C0", "C1", "C2", "C3",
    "A0", "A1", "A2",
    "F0", "F1",
}

EXPECTED_METRICS = {
    "validation_bbox_mAP_50_95",
    "validation_AP50",
    "validation_AP75",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def norm(v) -> str:
    return str(v if v is not None else "").strip()


def parse_bool(v) -> bool:
    x = norm(v).lower()
    if x == "true":
        return True
    if x == "false":
        return False
    raise RuntimeError(f"INVALID_BOOL:{v!r}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    sources = {
        "ablation_config_inventory.csv": CONFIG_INVENTORY,
        "ablation_run_inventory.csv": RUN_INVENTORY,
        "ablation_main_reference_map.csv": MAIN_MAP,
        "ablation_pair_inventory.csv": PAIR_INVENTORY,
        "ablation_effects.csv": ABLATION_EFFECTS,
        "ablation_qpseudo_effects.csv": QPSEUDO_EFFECTS,
    }

    observed_sha = {}

    for name, path in sources.items():
        require(path.is_file(), f"MISSING_SOURCE:{path}")
        observed_sha[name] = sha256(path)
        require(
            observed_sha[name] == EXPECTED_SHA256[name],
            f"SOURCE_SHA_MISMATCH:{name}:"
            f"{observed_sha[name]}!={EXPECTED_SHA256[name]}",
        )

    configs = read_csv(CONFIG_INVENTORY)
    runs = read_csv(RUN_INVENTORY)
    mains = read_csv(MAIN_MAP)
    pairs = read_csv(PAIR_INVENTORY)
    effects = read_csv(ABLATION_EFFECTS)
    qpseudo = read_csv(QPSEUDO_EFFECTS)

    # ------------------------------------------------------------
    # 1. Configuration identity
    # ------------------------------------------------------------
    require(len(configs) == 10, f"CONFIG_ROWS:{len(configs)}")

    config_ids = [norm(r["ablation_config_id"]).upper() for r in configs]
    require(len(set(config_ids)) == 10, "CONFIG_IDS_NOT_UNIQUE")

    main_configs = [
        r for r in configs
        if parse_bool(r["is_main"])
    ]
    non_main_configs = [
        r for r in configs
        if not parse_bool(r["is_main"])
    ]

    require(len(main_configs) == 1, f"MAIN_CONFIG_COUNT:{len(main_configs)}")
    require(
        len(non_main_configs) == 9,
        f"NON_MAIN_CONFIG_COUNT:{len(non_main_configs)}",
    )

    observed_non_main_ids = {
        norm(r["ablation_config_id"]).upper()
        for r in non_main_configs
    }
    require(
        observed_non_main_ids == EXPECTED_NON_MAIN_CONFIGS,
        f"NON_MAIN_CONFIG_SET:{sorted(observed_non_main_ids)}",
    )

    main_config_id = norm(
        main_configs[0]["ablation_config_id"]
    ).upper()
    require(main_config_id == "MAIN", f"MAIN_CONFIG_ID:{main_config_id}")

    require(
        parse_bool(main_configs[0]["validation_only"]) is False,
        "MAIN_VALIDATION_ONLY_INVALID",
    )
    require(
        parse_bool(main_configs[0]["final_test_eligible"]) is True,
        "MAIN_FINAL_TEST_ELIGIBILITY_INVALID",
    )

    for r in non_main_configs:
        require(
            parse_bool(r["validation_only"]) is True,
            f"NON_MAIN_VALIDATION_ONLY_INVALID:{r['ablation_config_id']}",
        )
        require(
            parse_bool(r["final_test_eligible"]) is False,
            f"NON_MAIN_FINAL_TEST_ELIGIBILITY_INVALID:{r['ablation_config_id']}",
        )

    # ------------------------------------------------------------
    # 2. Planned non-Main design identity
    # ------------------------------------------------------------
    require(len(runs) == 90, f"RUN_INVENTORY_ROWS:{len(runs)}")

    run_ids = [norm(r["ablation_run_id"]) for r in runs]
    require(len(set(run_ids)) == 90, "RUN_IDS_NOT_UNIQUE")

    run_config_ids = {
        norm(r["ablation_config_id"]).upper()
        for r in runs
    }
    require(
        run_config_ids == EXPECTED_NON_MAIN_CONFIGS,
        f"RUN_CONFIG_SET:{sorted(run_config_ids)}",
    )

    expected_seed_set = set(FROZEN_SEEDS)
    by_config: dict[str, list[dict[str, str]]] = defaultdict(list)

    for r in runs:
        config_id = norm(r["ablation_config_id"]).upper()
        by_config[config_id].append(r)

        require(norm(r["condition_role"]) == "ABLATION", "ROLE_INVALID")
        require(norm(r["method"]) == "SSL", "METHOD_INVALID")
        require(norm(r["architecture"]) == "R50-FPN", "ARCH_INVALID")
        require(norm(r["budget"]) == "10pct", "BUDGET_INVALID")
        require(parse_bool(r["is_main"]) is False, "RUN_IS_MAIN_INVALID")
        require(parse_bool(r["expected"]) is True, "RUN_EXPECTED_INVALID")
        require(parse_bool(r["validation_only"]) is True, "RUN_VALIDATION_ONLY_INVALID")
        require(parse_bool(r["final_test_eligible"]) is False, "RUN_FINAL_ELIGIBILITY_INVALID")
        require(parse_bool(r["test_access_authorized"]) is False, "RUN_TEST_AUTH_INVALID")

    for config_id, rows in by_config.items():
        require(
            len(rows) == 10,
            f"CONFIG_ROW_COUNT:{config_id}:{len(rows)}",
        )
        seeds = {int(r["training_seed"]) for r in rows}
        require(
            seeds == expected_seed_set,
            f"CONFIG_SEED_SET:{config_id}:{sorted(seeds)}",
        )

    # ------------------------------------------------------------
    # 3. Frozen Main reuse
    # ------------------------------------------------------------
    require(len(mains) == 10, f"MAIN_MAP_ROWS:{len(mains)}")

    main_run_ids = {norm(r["main_run_id"]) for r in mains}
    main_seeds = {int(r["training_seed"]) for r in mains}

    require(len(main_run_ids) == 10, "MAIN_RUN_IDS_NOT_UNIQUE")
    require(main_seeds == expected_seed_set, "MAIN_SEED_SET_INVALID")

    for r in mains:
        seed_index = int(r["seed_index"])
        require(
            norm(r["main_run_id"])
            == f"OFF_SSL_R50_B10_S{seed_index:02d}",
            f"MAIN_RUN_ID_BINDING_INVALID:{r['main_run_id']}",
        )

    # ------------------------------------------------------------
    # 4. Same-seed pairing
    # ------------------------------------------------------------
    require(len(pairs) == 90, f"PAIR_ROWS:{len(pairs)}")

    pair_variant_runs = {norm(r["ablation_run_id"]) for r in pairs}
    require(
        pair_variant_runs == set(run_ids),
        "PAIR_VARIANT_RUN_SET_INVALID",
    )

    pair_main_runs = {norm(r["main_run_id"]) for r in pairs}
    require(
        pair_main_runs == main_run_ids,
        "PAIR_MAIN_RUN_SET_INVALID",
    )

    for r in pairs:
        require(parse_bool(r["pair_valid"]) is True, "PAIR_INVALID")
        require(
            parse_bool(r["config_diff_only_target_factor"]) is True,
            "PAIR_NOT_OFAT",
        )

        seed_index = int(r["seed_index"])
        require(
            norm(r["main_run_id"])
            == f"OFF_SSL_R50_B10_S{seed_index:02d}",
            f"PAIR_MAIN_BINDING_INVALID:{r['ablation_run_id']}",
        )

    # ------------------------------------------------------------
    # 5. Canonical executed non-Main closure
    # ------------------------------------------------------------
    status_files = sorted(
        ABLATION_RUN_ROOT.glob("*/logical_run_status.json")
    )
    require(
        len(status_files) == 90,
        f"STATUS_FILE_COUNT:{len(status_files)}",
    )

    closed_run_ids = set()
    attempt_counts = Counter()

    for path in status_files:
        d = read_json(path)

        run_id = norm(d.get("run_id"))
        require(run_id in set(run_ids), f"UNEXPECTED_STATUS_RUN:{run_id}")

        require(
            norm(d.get("training_status")) == "CLOSED_VERIFIED",
            f"STATUS_NOT_CLOSED:{run_id}",
        )
        require(
            d.get("training_completed") is True,
            f"TRAINING_NOT_COMPLETED:{run_id}",
        )
        require(
            d.get("training_valid") is True,
            f"TRAINING_NOT_VALID:{run_id}",
        )
        require(
            d.get("validation_only") is True,
            f"STATUS_VALIDATION_ONLY_INVALID:{run_id}",
        )
        require(
            d.get("eligible_for_final_test") is False,
            f"STATUS_FINAL_ELIGIBILITY_INVALID:{run_id}",
        )
        require(
            d.get("test_access_authorized") is False,
            f"STATUS_TEST_AUTH_INVALID:{run_id}",
        )
        require(
            d.get("final_test_accessed") is False,
            f"STATUS_FINAL_TEST_ACCESSED:{run_id}",
        )

        attempt = norm(d.get("attempt_used"))
        require(
            re.fullmatch(r"attempt_\d{3}", attempt) is not None,
            f"ATTEMPT_INVALID:{run_id}:{attempt}",
        )
        attempt_counts[attempt] += 1

        completion_rel = norm(d.get("training_completion_report_path"))
        manifest_rel = norm(d.get("artifact_manifest_path"))

        require(completion_rel, f"COMPLETION_PATH_MISSING:{run_id}")
        require(manifest_rel, f"MANIFEST_PATH_MISSING:{run_id}")

        require(
            (path.parent / completion_rel).is_file(),
            f"COMPLETION_FILE_MISSING:{run_id}:{completion_rel}",
        )
        require(
            (path.parent / manifest_rel).is_file(),
            f"MANIFEST_FILE_MISSING:{run_id}:{manifest_rel}",
        )

        closed_run_ids.add(run_id)

    require(
        closed_run_ids == set(run_ids),
        "CLOSED_RUN_SET_MISMATCH",
    )

    require(
        attempt_counts
        == Counter({"attempt_001": 89, "attempt_002": 1}),
        f"ATTEMPT_COUNTS_INVALID:{dict(attempt_counts)}",
    )

    # ------------------------------------------------------------
    # 6. Detection-effect completeness
    # ------------------------------------------------------------
    require(len(effects) == 270, f"EFFECT_ROWS:{len(effects)}")

    effect_run_ids = {norm(r["ablation_run_id"]) for r in effects}
    require(
        effect_run_ids == set(run_ids),
        "EFFECT_RUN_SET_INVALID",
    )

    effect_metrics = {norm(r["metric_name"]) for r in effects}
    require(
        effect_metrics == EXPECTED_METRICS,
        f"EFFECT_METRIC_SET:{sorted(effect_metrics)}",
    )

    effect_keys = {
        (
            norm(r["ablation_run_id"]),
            norm(r["metric_name"]),
        )
        for r in effects
    }
    require(
        len(effect_keys) == 270,
        f"EFFECT_KEYS_NOT_UNIQUE:{len(effect_keys)}",
    )

    require(
        all(parse_bool(r["pair_valid"]) for r in effects),
        "EFFECT_PAIR_VALID_FAILURE",
    )
    require(
        all(
            parse_bool(r["formal_hypothesis_test"]) is False
            for r in effects
        ),
        "EFFECT_FORMAL_HYPOTHESIS_INVALID",
    )

    # ------------------------------------------------------------
    # 7. Q_pseudo mechanistic-effect completeness
    # ------------------------------------------------------------
    require(len(qpseudo) == 90, f"QPSEUDO_ROWS:{len(qpseudo)}")

    qp_run_ids = {
        norm(r["ablation_run_id"])
        for r in qpseudo
    }
    require(
        qp_run_ids == set(run_ids),
        "QPSEUDO_RUN_SET_INVALID",
    )

    # ------------------------------------------------------------
    # 8. Canonical 100-observation completeness
    # ------------------------------------------------------------
    non_main_training_count = len(closed_run_ids)
    main_reused_observation_count = len(main_run_ids)
    total_config_seed_observations = (
        non_main_training_count
        + main_reused_observation_count
    )

    require(non_main_training_count == 90, "NON_MAIN_TRAININGS_NOT_90")
    require(main_reused_observation_count == 10, "MAIN_REUSE_NOT_10")
    require(total_config_seed_observations == 100, "TOTAL_OBSERVATIONS_NOT_100")

    checks = {
        "unique_configs_10": len(set(config_ids)) == 10,
        "main_configs_1": len(main_configs) == 1,
        "non_main_configs_9": len(non_main_configs) == 9,
        "non_main_configs_exact": observed_non_main_ids == EXPECTED_NON_MAIN_CONFIGS,
        "non_main_10_seeds_per_config": all(
            len(rows) == 10
            and {int(r["training_seed"]) for r in rows} == expected_seed_set
            for rows in by_config.values()
        ),
        "non_main_additional_trainings_90": non_main_training_count == 90,
        "main_reused_observations_10": main_reused_observation_count == 10,
        "configuration_seed_observations_100": total_config_seed_observations == 100,
        "same_seed_pairs_90": len(pairs) == 90,
        "pair_valid_90": all(parse_bool(r["pair_valid"]) for r in pairs),
        "ofat_pairing_90": all(
            parse_bool(r["config_diff_only_target_factor"])
            for r in pairs
        ),
        "closed_verified_90": len(closed_run_ids) == 90,
        "validation_only_90": True,
        "final_test_eligible_false_90": True,
        "test_access_authorized_false_90": True,
        "final_test_accessed_false_90": True,
        "canonical_attempt_distribution": (
            attempt_counts
            == Counter({"attempt_001": 89, "attempt_002": 1})
        ),
        "detection_effect_rows_270": len(effects) == 270,
        "detection_metrics_exact_3": effect_metrics == EXPECTED_METRICS,
        "qpseudo_effect_rows_90": len(qpseudo) == 90,
        "formal_hypothesis_test_absent": all(
            parse_bool(r["formal_hypothesis_test"]) is False
            for r in effects
        ),
    }

    failures = [
        key for key, value in checks.items()
        if value is not True
    ]

    payload = {
        "schema_version": "1.0",
        "stage": "S10.1",
        "task": "ABLATION_COMPLETENESS",
        "status": "PASS" if not failures else "FAIL",
        "scientific_role": "SECONDARY_MECHANISTIC_ESTIMATION_FOCUSED",
        "architecture": "R50-FPN",
        "budget": "10pct",
        "training_seed_count": 10,
        "unique_config_count": 10,
        "main_config_count": 1,
        "non_main_config_count": 9,
        "non_main_additional_training_count": 90,
        "main_reused_observation_count": 10,
        "configuration_seed_observation_count": 100,
        "pair_count": 90,
        "closed_verified_count": 90,
        "attempt_counts": dict(sorted(attempt_counts.items())),
        "detection_effect_row_count": 270,
        "detection_metrics": sorted(effect_metrics),
        "qpseudo_effect_row_count": 90,
        "validation_only_non_main": True,
        "final_test_eligible_non_main": False,
        "test_access_authorized_non_main": False,
        "final_test_accessed_non_main": False,
        "main_retrained_for_ablation": False,
        "formal_hypothesis_test": False,
        "multiplicity_family_added": False,
        "checks": checks,
        "failure_count": len(failures),
        "failures": failures,
        "source_sha256": observed_sha,
        "execution_status_authority":
            "artifacts/runs/ablation/<run_id>/logical_run_status.json",
        "run_inventory_status_fields_role":
            "STATIC_PRE_EXECUTION_REPRESENTATION_NOT_EXECUTION_AUTHORITY",
        "statistics_reimplemented": False,
    }

    require(
        payload["status"] == "PASS",
        f"ABLATION_COMPLETENESS_FAILED:{failures}",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print("=== S10.1 ABLATION COMPLETENESS MATERIALIZATION ===")
    print("STATUS=" + payload["status"])
    print("UNIQUE_CONFIGS=" + str(payload["unique_config_count"]))
    print("NON_MAIN_CONFIGS=" + str(payload["non_main_config_count"]))
    print("MAIN_CONFIGS=" + str(payload["main_config_count"]))
    print("NON_MAIN_ADDITIONAL_TRAININGS=" + str(payload["non_main_additional_training_count"]))
    print("MAIN_REUSED_OBSERVATIONS=" + str(payload["main_reused_observation_count"]))
    print("CONFIGURATION_SEED_OBSERVATIONS=" + str(payload["configuration_seed_observation_count"]))
    print("PAIR_COUNT=" + str(payload["pair_count"]))
    print("CLOSED_VERIFIED=" + str(payload["closed_verified_count"]))
    print("ATTEMPT_001=" + str(payload["attempt_counts"].get("attempt_001", 0)))
    print("ATTEMPT_002=" + str(payload["attempt_counts"].get("attempt_002", 0)))
    print("VALIDATION_ONLY_NON_MAIN=" + str(payload["validation_only_non_main"]).upper())
    print("FINAL_TEST_ELIGIBLE_NON_MAIN=" + str(payload["final_test_eligible_non_main"]).upper())
    print("TEST_ACCESS_AUTHORIZED_NON_MAIN=" + str(payload["test_access_authorized_non_main"]).upper())
    print("FINAL_TEST_ACCESSED_NON_MAIN=" + str(payload["final_test_accessed_non_main"]).upper())
    print("DETECTION_EFFECT_ROWS=" + str(payload["detection_effect_row_count"]))
    print("QPSEUDO_EFFECT_ROWS=" + str(payload["qpseudo_effect_row_count"]))
    print("FAILURE_COUNT=" + str(payload["failure_count"]))
    print("OUTPUT=" + str(OUTPUT.relative_to(ROOT)))
    print("OUTPUT_SHA256=" + sha256(OUTPUT))
    print("STATISTICS_REIMPLEMENTED=FALSE")


if __name__ == "__main__":
    main()