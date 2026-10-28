#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

NON_MAIN_ORDER = [
    "C0", "C1", "C2", "C3",
    "A0", "A1", "A2",
    "F0", "F1",
]

OUTPUT_FIELDS = [
    "ablation_config_id",
    "family",
    "level",
    "is_main",
    "scientific_source_sha256",
    "resolved_config_sha256",
    "validation_only",
    "final_test_eligible",
    "executable_config_path",
    "executable_config_sha256",
    "scientific_aliases",
    "resolved_config_member_count",
    "resolved_config_sha256_semantics",
    "resolved_config_member_map",
    "representation_authority",
]


def require(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def norm(x):
    if x is None:
        return ""
    return str(x).strip().strip('"')


def parse_bool(x):
    v = norm(x).lower()
    require(v in {"true", "false"}, f"INVALID_BOOL:{x!r}")
    return v == "true"


def require_sha(x, label):
    x = norm(x)
    require(SHA256_RE.fullmatch(x) is not None, f"{label}_INVALID_SHA256:{x}")
    return x


def load_csv(path):
    path = Path(path)
    require(path.is_file(), f"MISSING_FILE:{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_json(path):
    path = Path(path)
    require(path.is_file(), f"MISSING_FILE:{path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    require(isinstance(payload, dict), f"JSON_NOT_OBJECT:{path}")
    return payload


def resolved_set_digest(members):
    canonical = json.dumps(
        members,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def member_map_text(members):
    return ";".join(
        f"S{x['seed_index']:02d}:{x['training_seed']}:{x['resolved_config_sha256']}"
        for x in members
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ablation-run-inventory", required=True)
    ap.add_argument("--eligible-model-inventory", required=True)
    ap.add_argument("--ablation-root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    run_inventory = load_csv(args.ablation_run_inventory)
    eligible = load_csv(args.eligible_model_inventory)
    ablation_root = Path(args.ablation_root)

    require(len(run_inventory) == 90, f"ABLATION_RUN_ROWS={len(run_inventory)}")
    require(len(eligible) == 180, f"ELIGIBLE_ROWS={len(eligible)}")

    grouped = defaultdict(list)
    attempt_counts = Counter()

    # ---------------------------------------------------------------
    # 9 non-Main scientific configurations.
    # Canonical attempts are selected ONLY from logical_run_status.json.
    # ---------------------------------------------------------------
    for row in run_inventory:
        run_id = norm(row["ablation_run_id"])
        config_id = norm(row["ablation_config_id"]).upper()

        require(config_id in NON_MAIN_ORDER, f"UNKNOWN_NON_MAIN_CONFIG:{config_id}")
        require(norm(row["condition_role"]) == "ABLATION", f"ROLE_INVALID:{run_id}")
        require(norm(row["method"]) == "SSL", f"METHOD_INVALID:{run_id}")
        require(norm(row["architecture"]) == "R50-FPN", f"ARCH_INVALID:{run_id}")
        require(norm(row["budget"]) == "10pct", f"BUDGET_INVALID:{run_id}")
        require(not parse_bool(row["is_main"]), f"IS_MAIN_INVALID:{run_id}")
        require(parse_bool(row["validation_only"]), f"VALIDATION_ONLY_INVALID:{run_id}")
        require(
            not parse_bool(row["final_test_eligible"]),
            f"FINAL_TEST_ELIGIBLE_INVALID:{run_id}",
        )

        status = load_json(
            ablation_root / run_id / "logical_run_status.json"
        )

        require(
            status.get("training_status") == "CLOSED_VERIFIED",
            f"STATUS_INVALID:{run_id}",
        )
        require(status.get("training_completed") is True, f"NOT_COMPLETED:{run_id}")
        require(status.get("training_valid") is True, f"NOT_VALID:{run_id}")

        attempt = norm(status.get("attempt_used"))
        require(
            re.fullmatch(r"attempt_\d{3}", attempt) is not None,
            f"ATTEMPT_INVALID:{run_id}",
        )
        attempt_counts[attempt] += 1

        manifest = load_json(
            ablation_root / run_id / attempt / "run_manifest.json"
        )

        require(
            norm(manifest.get("ablation_config_id")).upper() == config_id,
            f"CONFIG_ID_MISMATCH:{run_id}",
        )

        seed_index = int(manifest["seed_index"])
        training_seed = int(manifest["training_seed"])

        grouped[config_id].append({
            "seed_index": seed_index,
            "training_seed": training_seed,
            "family": norm(manifest["ablation_family"]),
            "level": norm(manifest["ablation_level"]),
            "scientific_source_sha256":
                require_sha(
                    manifest["scientific_source_sha256"],
                    f"{run_id}:SCIENTIFIC_SOURCE",
                ),
            "resolved_config_sha256":
                require_sha(
                    manifest["resolved_config_sha256"],
                    f"{run_id}:RESOLVED_CONFIG",
                ),
            "executable_config_path":
                norm(manifest["executable_config_path"]),
            "executable_config_sha256":
                require_sha(
                    manifest["executable_config_sha256"],
                    f"{run_id}:EXECUTABLE_CONFIG",
                ),
            "validation_only": manifest["validation_only"],
            "final_test_eligible": manifest["final_test_eligible"],
        })

    require(set(grouped) == set(NON_MAIN_ORDER), "NON_MAIN_CONFIG_SET_INVALID")
    require(
        attempt_counts == Counter({"attempt_001": 89, "attempt_002": 1}),
        f"CANONICAL_ATTEMPT_COUNTS_INVALID:{dict(attempt_counts)}",
    )

    output_rows = []

    for config_id in NON_MAIN_ORDER:
        rows = grouped[config_id]
        require(len(rows) == 10, f"{config_id}:ROWS={len(rows)}")

        rows.sort(key=lambda x: x["seed_index"])

        require(
            [x["seed_index"] for x in rows] == list(range(1, 11)),
            f"{config_id}:SEED_INDEX_SET_INVALID",
        )
        require(
            len({x["training_seed"] for x in rows}) == 10,
            f"{config_id}:TRAINING_SEEDS_NOT_UNIQUE",
        )

        def unique(field):
            vals = {x[field] for x in rows}
            require(
                len(vals) == 1,
                f"{config_id}:{field}:UNIQUE={len(vals)}",
            )
            return next(iter(vals))

        family = unique("family")
        level = unique("level")
        scientific_source_sha256 = unique("scientific_source_sha256")
        executable_config_path = unique("executable_config_path")
        executable_config_sha256 = unique("executable_config_sha256")

        require(unique("validation_only") is True, f"{config_id}:VALIDATION_ONLY")
        require(
            unique("final_test_eligible") is False,
            f"{config_id}:FINAL_TEST_ELIGIBLE",
        )

        members = [
            {
                "seed_index": x["seed_index"],
                "training_seed": x["training_seed"],
                "resolved_config_sha256": x["resolved_config_sha256"],
            }
            for x in rows
        ]

        require(
            len({x["resolved_config_sha256"] for x in members}) == 10,
            f"{config_id}:RESOLVED_SHA_EXPECTED_10_UNIQUE",
        )

        output_rows.append({
            "ablation_config_id": config_id,
            "family": family,
            "level": level,
            "is_main": "false",
            "scientific_source_sha256": scientific_source_sha256,
            "resolved_config_sha256": resolved_set_digest(members),
            "validation_only": "true",
            "final_test_eligible": "false",
            "executable_config_path": executable_config_path,
            "executable_config_sha256": executable_config_sha256,
            "scientific_aliases": "",
            "resolved_config_member_count": "10",
            "resolved_config_sha256_semantics":
                "SHA256_CANONICAL_JSON_OF_ORDERED_10_SEED_RESOLVED_CONFIG_IDENTITIES",
            "resolved_config_member_map": member_map_text(members),
            "representation_authority":
                "S10_PROVENANCE_ONLY_CONFIG_LEVEL_MAPPING",
        })

    # ---------------------------------------------------------------
    # Frozen Main configuration.
    # Main is NOT retrained for ablation.
    # C4 = A3 = F2 = one frozen Main scientific configuration.
    # ---------------------------------------------------------------
    run_field = "s9_12.run_id"

    main_rows = [
        r for r in eligible
        if re.fullmatch(
            r"OFF_SSL_R50_B10_S(?:0[1-9]|10)",
            norm(r.get(run_field)),
        )
    ]

    require(len(main_rows) == 10, f"MAIN_ROWS={len(main_rows)}")

    main_rows.sort(
        key=lambda r: int(norm(r["official_inventory.seed_index"]))
    )

    require(
        [int(norm(r["official_inventory.seed_index"])) for r in main_rows]
        == list(range(1, 11)),
        "MAIN_SEED_INDEX_SET_INVALID",
    )

    main_members = []

    main_science = set()
    main_exec_paths = set()
    main_exec_shas = set()

    for r in main_rows:
        run_id = norm(r[run_field])
        seed_index = int(norm(r["official_inventory.seed_index"]))
        training_seed = int(norm(r["official_inventory.training_seed"]))

        require(
            norm(r["s9_12.attempt_used"]) == "attempt_001",
            f"MAIN_ATTEMPT_INVALID:{run_id}",
        )
        require(
            norm(r["official_inventory.architecture"]) == "R50-FPN",
            f"MAIN_ARCH_INVALID:{run_id}",
        )
        require(
            norm(r["official_inventory.budget"]) == "10pct",
            f"MAIN_BUDGET_INVALID:{run_id}",
        )
        require(
            norm(r["official_inventory.condition_role"]) == "MAIN_LOW_LABEL",
            f"MAIN_ROLE_INVALID:{run_id}",
        )
        require(
            norm(r["official_inventory.method"]) == "SSL",
            f"MAIN_METHOD_INVALID:{run_id}",
        )

        require(
            parse_bool(r["logical_run_status.eligible_for_final_test"]),
            f"MAIN_NOT_ELIGIBLE:{run_id}",
        )
        require(
            norm(r["logical_run_status.training_status"]) == "CLOSED_VERIFIED",
            f"MAIN_STATUS_INVALID:{run_id}",
        )
        require(
            parse_bool(r["logical_run_status.training_valid"]),
            f"MAIN_TRAINING_INVALID:{run_id}",
        )

        main_science.add(
            require_sha(
                r["pre_run_assertions.expected_identity.scientific_source_sha256"],
                f"{run_id}:MAIN_SCIENTIFIC_SOURCE",
            )
        )

        main_exec_paths.add(
            norm(
                r[
                    "pre_run_assertions.expected_identity.executable_config_path"
                ]
            )
        )

        main_exec_shas.add(
            require_sha(
                r[
                    "pre_run_assertions.expected_identity.executable_config_sha256"
                ],
                f"{run_id}:MAIN_EXECUTABLE_CONFIG",
            )
        )

        resolved_sha = require_sha(
            r["retained_artifacts.resolved_config.yaml.actual_sha256"],
            f"{run_id}:MAIN_RESOLVED_CONFIG",
        )

        main_members.append({
            "seed_index": seed_index,
            "training_seed": training_seed,
            "resolved_config_sha256": resolved_sha,
        })

    require(len(main_science) == 1, "MAIN_SCIENTIFIC_SOURCE_NOT_UNIQUE")
    require(len(main_exec_paths) == 1, "MAIN_EXECUTABLE_PATH_NOT_UNIQUE")
    require(len(main_exec_shas) == 1, "MAIN_EXECUTABLE_SHA_NOT_UNIQUE")
    require(
        len({x["resolved_config_sha256"] for x in main_members}) == 10,
        "MAIN_RESOLVED_SHA_EXPECTED_10_UNIQUE",
    )

    main_exec_path = next(iter(main_exec_paths))
    main_exec_sha = next(iter(main_exec_shas))

    expected_main_path = (
        "configs/ssl/"
        "s5_06_soft_teacher_r50_fpn_scientific_eval_10pct.py"
    )

    require(
        main_exec_path == expected_main_path,
        f"MAIN_EXECUTABLE_PATH_UNEXPECTED:{main_exec_path}",
    )

    local_main_path = repo / main_exec_path
    require(local_main_path.is_file(), "LOCAL_MAIN_CONFIG_MISSING")

    local_main_sha = hashlib.sha256(local_main_path.read_bytes()).hexdigest()
    require(
        local_main_sha == main_exec_sha,
        "LOCAL_MAIN_CONFIG_SHA_MISMATCH",
    )

    output_rows.append({
        # S10 provenance-only row key.
        # This is NOT a new S9.11 runtime ablation config ID.
        "ablation_config_id": "MAIN",
        "family": "CONFIDENCE|AUGMENTATION|REGRESSION",
        "level": "C4|A3|F2",
        "is_main": "true",
        "scientific_source_sha256": next(iter(main_science)),
        "resolved_config_sha256": resolved_set_digest(main_members),
        "validation_only": "false",
        "final_test_eligible": "true",
        "executable_config_path": main_exec_path,
        "executable_config_sha256": main_exec_sha,
        "scientific_aliases": "C4|A3|F2",
        "resolved_config_member_count": "10",
        "resolved_config_sha256_semantics":
            "SHA256_CANONICAL_JSON_OF_ORDERED_10_SEED_RESOLVED_CONFIG_IDENTITIES",
        "resolved_config_member_map": member_map_text(main_members),
        "representation_authority":
            "S10_PROVENANCE_ONLY_MAIN_REFERENCE_MAPPING",
    })

    require(len(output_rows) == 10, f"OUTPUT_ROWS={len(output_rows)}")
    require(
        sum(r["is_main"] == "true" for r in output_rows) == 1,
        "MAIN_ROW_COUNT_INVALID",
    )
    require(
        len({r["ablation_config_id"] for r in output_rows}) == 10,
        "CONFIG_IDS_NOT_UNIQUE",
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    print("=== S10.1.07 ABLATION CONFIG INVENTORY ===")
    print(f"OUTPUT_ROWS={len(output_rows)}")
    print(f"COLUMNS={len(OUTPUT_FIELDS)}")
    print(
        "NON_MAIN_ROWS="
        f"{sum(r['is_main'] == 'false' for r in output_rows)}"
    )
    print(
        "MAIN_ROWS="
        f"{sum(r['is_main'] == 'true' for r in output_rows)}"
    )
    print(
        "RESOLVED_MEMBER_COUNT_10="
        f"{sum(r['resolved_config_member_count'] == '10' for r in output_rows)}"
    )
    print("MAIN_RETRAINED=FALSE")
    print("MAIN_ALIASES=C4|A3|F2")
    print("S10_1_07_STATUS=PASS")


if __name__ == "__main__":
    main()