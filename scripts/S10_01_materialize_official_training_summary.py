#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXPECTED_ROWS = 180
EXPECTED_SSL = 80
EXPECTED_SUP = 100

OUTPUT_FIELDS = [
    "run_id",
    "condition_role",
    "method",
    "architecture",
    "budget",
    "seed_index",
    "training_seed",
    "best_checkpoint_sha256",
    "best_validation_mAP_50_95",
    "last_checkpoint_sha256",
    "Q_pseudo_if_SSL",
    "training_status",
    "eligible_for_final_test",
]


def norm(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().strip('"')


def as_bool(value: object) -> bool:
    return norm(value).lower() == "true"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_inventory(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"Input inventory not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(rows, f"Input inventory is empty: {path}")
    return rows


def canonical_last_sha(row: dict[str, str], method: str) -> str:
    if method == "SSL":
        return norm(
            row.get(
                "retained_artifacts.checkpoints/"
                "last_ema_teacher.pth.actual_sha256"
            )
        )
    if method == "SUP":
        return norm(
            row.get(
                "retained_artifacts.checkpoints/"
                "last_model.pth.actual_sha256"
            )
        )
    raise RuntimeError(f"Unsupported method: {method}")


def validate_gate(row: dict[str, str], run_id: str) -> None:
    require(
        as_bool(row.get("pre_run_assertions.all_checks_pass")),
        f"{run_id}: pre_run_assertions.all_checks_pass != true",
    )
    require(
        norm(row.get("training_completion_report.training_status"))
        == "CLOSED_VERIFIED",
        f"{run_id}: training status is not CLOSED_VERIFIED",
    )
    require(
        as_bool(row.get("training_completion_report.eligible_for_final_test")),
        f"{run_id}: eligible_for_final_test != true",
    )
    require(
        norm(row.get("training_completion_report.expected_optimizer_updates"))
        == norm(row.get("training_completion_report.observed_optimizer_updates")),
        f"{run_id}: optimizer update count mismatch",
    )
    require(
        as_bool(row.get("training_completion_report.best_checkpoint_present")),
        f"{run_id}: BEST checkpoint missing",
    )
    require(
        as_bool(row.get("training_completion_report.last_checkpoint_present")),
        f"{run_id}: LAST checkpoint missing",
    )
    require(
        as_bool(row.get("training_completion_report.artifact_manifest_valid")),
        f"{run_id}: artifact manifest invalid",
    )
    require(
        as_bool(row.get("artifact_manifest.mandatory_artifacts_complete")),
        f"{run_id}: mandatory artifacts incomplete",
    )

    deviation = norm(
        row.get("training_completion_report.scientific_protocol_deviation")
    ).lower()
    require(
        deviation in {"", "false", "none"},
        f"{run_id}: unresolved scientific protocol deviation: {deviation}",
    )

    require(
        not as_bool(row.get("pre_run_assertions.final_test_accessed")),
        f"{run_id}: final test accessed during training",
    )
    require(
        not as_bool(
            row.get("pre_run_assertions.observed_identity.test_accessed")
        ),
        f"{run_id}: observed test_accessed=true during training",
    )
    require(
        as_bool(row.get("retry_deviation.same_seed_confirmed")),
        f"{run_id}: retry same-seed confirmation failed",
    )
    require(
        not as_bool(row.get("retry_deviation.scientific_protocol_changed")),
        f"{run_id}: retry changed scientific protocol",
    )
    require(
        norm(row.get("backup_sync_manifest.verification_status")) == "PASS",
        f"{run_id}: durable backup verification not PASS",
    )


def materialize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    require(
        len(rows) == EXPECTED_ROWS,
        f"Expected {EXPECTED_ROWS} canonical rows, found {len(rows)}",
    )

    output: list[dict[str, str]] = []
    seen_run_ids: set[str] = set()

    for row in rows:
        run_id = norm(row.get("s9_12.run_id"))
        require(run_id, "Missing s9_12.run_id")
        require(run_id not in seen_run_ids, f"Duplicate run_id: {run_id}")
        seen_run_ids.add(run_id)

        method = norm(row.get("official_inventory.method"))
        require(method in {"SUP", "SSL"}, f"{run_id}: invalid method={method}")

        validate_gate(row, run_id)

        best_sha = norm(row.get("selected_checkpoint.sha256"))
        best_val = norm(row.get("validation_best.selected_metric_value"))
        last_sha = canonical_last_sha(row, method)

        require(best_sha, f"{run_id}: missing BEST checkpoint SHA256")
        require(best_val, f"{run_id}: missing BEST validation mAP")
        require(last_sha, f"{run_id}: missing LAST checkpoint SHA256")

        if method == "SSL":
            qpseudo = norm(row.get("qpseudo_validation.PL_mAP_50_95"))
            require(qpseudo != "", f"{run_id}: missing SSL Q_pseudo")
            require(
                not as_bool(row.get("qpseudo_validation.test_used")),
                f"{run_id}: Q_pseudo used test data",
            )
            require(
                not as_bool(row.get("qpseudo_validation.hidden_u_gt_used")),
                f"{run_id}: Q_pseudo used hidden unlabeled GT",
            )
        else:
            qpseudo = ""
            require(
                norm(row.get("qpseudo_validation.PL_mAP_50_95")) == "",
                f"{run_id}: SUP unexpectedly contains Q_pseudo",
            )

        output.append(
            {
                "run_id": run_id,
                "condition_role": norm(
                    row.get("official_inventory.condition_role")
                ),
                "method": method,
                "architecture": norm(
                    row.get("official_inventory.architecture")
                ),
                "budget": norm(row.get("official_inventory.budget")),
                "seed_index": norm(row.get("official_inventory.seed_index")),
                "training_seed": norm(
                    row.get("official_inventory.training_seed")
                ),
                "best_checkpoint_sha256": best_sha,
                "best_validation_mAP_50_95": best_val,
                "last_checkpoint_sha256": last_sha,
                "Q_pseudo_if_SSL": qpseudo,
                "training_status": norm(
                    row.get("training_completion_report.training_status")
                ),
                "eligible_for_final_test": norm(
                    row.get(
                        "training_completion_report.eligible_for_final_test"
                    )
                ).lower(),
            }
        )

    require(len(seen_run_ids) == EXPECTED_ROWS, "Unique run count != 180")

    ssl_count = sum(r["method"] == "SSL" for r in output)
    sup_count = sum(r["method"] == "SUP" for r in output)

    require(
        ssl_count == EXPECTED_SSL,
        f"Expected {EXPECTED_SSL} SSL rows, found {ssl_count}",
    )
    require(
        sup_count == EXPECTED_SUP,
        f"Expected {EXPECTED_SUP} SUP rows, found {sup_count}",
    )

    role_counts: dict[tuple[str, str], int] = {}
    for row in output:
        key = (row["method"], row["condition_role"])
        role_counts[key] = role_counts.get(key, 0) + 1

    require(
        role_counts.get(("SSL", "MAIN_LOW_LABEL"), 0) == 80,
        "Expected 80 SSL MAIN_LOW_LABEL rows",
    )
    require(
        role_counts.get(("SUP", "MAIN_LOW_LABEL"), 0) == 80,
        "Expected 80 SUP MAIN_LOW_LABEL rows",
    )
    require(
        role_counts.get(("SUP", "SUP_REFERENCE"), 0) == 20,
        "Expected 20 SUP SUP_REFERENCE rows",
    )

    return output


def write_output(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize canonical S10 official_training_summary.csv "
            "from S9.12 eligible_model_inventory.csv."
        )
    )
    parser.add_argument("--eligible-model-inventory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = read_inventory(args.eligible_model_inventory)
    output = materialize(rows)
    write_output(args.output, output)

    print("=== S10.01 OFFICIAL TRAINING SUMMARY ===")
    print(f"INPUT_ROWS={len(rows)}")
    print(f"OUTPUT_ROWS={len(output)}")
    print(f"UNIQUE_RUN_IDS={len({r['run_id'] for r in output})}")
    print(f"SSL_ROWS={sum(r['method'] == 'SSL' for r in output)}")
    print(f"SUP_ROWS={sum(r['method'] == 'SUP' for r in output)}")
    print(
        "SSL_MAIN_LOW_LABEL="
        f"{sum(r['method'] == 'SSL' and r['condition_role'] == 'MAIN_LOW_LABEL' for r in output)}"
    )
    print(
        "SUP_MAIN_LOW_LABEL="
        f"{sum(r['method'] == 'SUP' and r['condition_role'] == 'MAIN_LOW_LABEL' for r in output)}"
    )
    print(
        "SUP_REFERENCE="
        f"{sum(r['method'] == 'SUP' and r['condition_role'] == 'SUP_REFERENCE' for r in output)}"
    )
    print("S10_01_STATUS=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
