#!/usr/bin/env python3
"""S9.12.A — materialize the eligible Main/reference model inventory.

Final-test preparation only. This script MUST NOT read final-test ground truth,
run final-test evaluation, or authorize final-test access.

Policy:
- population comes only from the canonical 180-row official run inventory;
- runtime eligibility comes from each logical_run_status.json;
- SUP selects BEST_CHECKPOINT; SSL selects BEST_EMA_TEACHER_CHECKPOINT;
- every manifest-listed retained artifact is accounted for;
- local retained artifacts must match manifest size + SHA-256;
- latest_resume.pth may be absent locally only when exact durable-backup
  provenance (path, SHA-256, size and PASS verification) is present;
- structured JSON evidence used for eligibility/provenance is flattened
  losslessly into the CSV; lists/scalars are canonical JSON;
- non-JSON/binary/log artifacts are represented by exact provenance metadata
  rather than embedding their payload bytes in the CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

OUTPUT_REL = Path("artifacts/final_test/eligible_model_inventory.csv")
OFFICIAL_INVENTORY_REL = Path("artifacts/results/official_run_inventory.csv")
OFFICIAL_RUNS_REL = Path("artifacts/runs/official")

COMMON_JSON_FILES = {
    "checkpoint_index": "checkpoints/checkpoint_index.json",
    "training_completion_report": "training_completion_report.json",
    "pre_run_assertions": "pre_run_assertions.json",
    "run_manifest": "run_manifest.json",
    "data_provenance": "data_provenance.json",
    "seed_runtime": "seed_runtime.json",
    "retry_deviation": "retry_deviation.json",
    "backup_sync_manifest": "backup_sync_manifest.json",
    "environment_reference": "environment_reference.json",
    "artifact_manifest": "artifact_manifest.json",
    "training_summary": "metrics/validation/training_summary.json",
}

SSL_JSON_FILES = {
    "qpseudo_validation": "pseudo/qpseudo_validation.json",
}

EXPECTED_GROUPS = {
    ("MAIN_LOW_LABEL", "SUP", "1pct"): 20,
    ("MAIN_LOW_LABEL", "SUP", "5pct"): 20,
    ("MAIN_LOW_LABEL", "SUP", "10pct"): 20,
    ("MAIN_LOW_LABEL", "SUP", "20pct"): 20,
    ("SUP_REFERENCE", "SUP", "100pct"): 20,
    ("MAIN_LOW_LABEL", "SSL", "1pct"): 20,
    ("MAIN_LOW_LABEL", "SSL", "5pct"): 20,
    ("MAIN_LOW_LABEL", "SSL", "10pct"): 20,
    ("MAIN_LOW_LABEL", "SSL", "20pct"): 20,
}


class S912InventoryError(RuntimeError):
    """Raised when S9.12.A cannot safely materialize the inventory."""


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise S912InventoryError(f"MISSING_JSON:{path}")
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(obj, dict):
        raise S912InventoryError(f"JSON_ROOT_NOT_OBJECT:{path}")
    return obj


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def flatten(prefix: str, value: Any, out: dict[str, str]) -> None:
    """Losslessly flatten dictionaries; preserve lists/scalars as canonical JSON."""
    if isinstance(value, dict):
        if not value:
            out[prefix] = canonical_json(value)
            return
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            flatten(child, value[key], out)
        return
    out[prefix] = canonical_json(value)


def atomic_write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent), text=True
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with tmp.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=fieldnames,
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def artifact_column_prefix(rel: str) -> str:
    # Preserve the exact relative path in the header namespace.
    return f"retained_artifacts.{rel}"


def verify_manifest_artifacts(
    *,
    run_id: str,
    attempt_dir: Path,
    artifact_manifest: dict[str, Any],
    backup_sync_manifest: dict[str, Any],
    row: dict[str, str],
) -> tuple[int, int]:
    artifacts = artifact_manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise S912InventoryError(f"ARTIFACT_MANIFEST_ARTIFACTS_NOT_LIST:{run_id}")

    backup_files = backup_sync_manifest.get("files", [])
    if not isinstance(backup_files, list):
        raise S912InventoryError(f"BACKUP_FILES_NOT_LIST:{run_id}")

    local_verified = 0
    remote_resume_verified = 0

    for item in artifacts:
        if not isinstance(item, dict):
            raise S912InventoryError(f"ARTIFACT_MANIFEST_ENTRY_NOT_OBJECT:{run_id}")

        rel = str(item.get("path", "")).strip()
        if not rel:
            raise S912InventoryError(f"ARTIFACT_MANIFEST_EMPTY_PATH:{run_id}")

        role = item.get("role")
        required = item.get("required")
        expected_sha = str(item.get("sha256", "")).lower()
        expected_size = item.get("size_bytes")
        if required is not True:
            raise S912InventoryError(f"MANIFEST_ARTIFACT_NOT_REQUIRED:{run_id}:{rel}")
        if len(expected_sha) != 64:
            raise S912InventoryError(f"MANIFEST_ARTIFACT_SHA_INVALID:{run_id}:{rel}")
        if not isinstance(expected_size, int) or expected_size < 0:
            raise S912InventoryError(f"MANIFEST_ARTIFACT_SIZE_INVALID:{run_id}:{rel}")

        p = attempt_dir / rel
        prefix = artifact_column_prefix(rel)
        row[f"{prefix}.path"] = canonical_json(rel)
        row[f"{prefix}.role"] = canonical_json(role)
        row[f"{prefix}.required"] = canonical_json(required)
        row[f"{prefix}.manifest_sha256"] = canonical_json(expected_sha)
        row[f"{prefix}.manifest_size_bytes"] = canonical_json(expected_size)

        if p.is_file():
            actual_size = p.stat().st_size
            actual_sha = file_sha256(p)
            if actual_size != expected_size:
                raise S912InventoryError(
                    f"ARTIFACT_SIZE_MISMATCH:{run_id}:{rel}:{expected_size}:{actual_size}"
                )
            if actual_sha != expected_sha:
                raise S912InventoryError(
                    f"ARTIFACT_SHA256_MISMATCH:{run_id}:{rel}:{expected_sha}:{actual_sha}"
                )

            row[f"{prefix}.storage_state"] = canonical_json("LOCAL_VERIFIED")
            row[f"{prefix}.actual_sha256"] = canonical_json(actual_sha)
            row[f"{prefix}.actual_size_bytes"] = canonical_json(actual_size)
            row[f"{prefix}.local_path"] = canonical_json(str(p))
            local_verified += 1
            continue

        if rel != "checkpoints/latest_resume.pth":
            raise S912InventoryError(f"REQUIRED_ARTIFACT_LOCAL_MISSING:{run_id}:{rel}")

        file_matches = [
            x
            for x in backup_files
            if isinstance(x, dict)
            and x.get("path") == rel
            and str(x.get("sha256", "")).lower() == expected_sha
            and x.get("size_bytes") == expected_size
            and x.get("verification_status") == "PASS"
            and str(x.get("destination", "")).strip()
        ]

        durable = backup_sync_manifest.get("LATEST_VERIFIED_DURABLE_RECOVERY_POINT")
        durable_ok = (
            isinstance(durable, dict)
            and durable.get("path") == rel
            and str(durable.get("sha256", "")).lower() == expected_sha
            and durable.get("size_bytes") == expected_size
        )

        if len(file_matches) != 1 or not durable_ok:
            raise S912InventoryError(
                f"LATEST_RESUME_DURABLE_BACKUP_NOT_EXACT:{run_id}:"
                f"matches={len(file_matches)}:durable_ok={durable_ok}"
            )

        backup_record = file_matches[0]
        row[f"{prefix}.storage_state"] = canonical_json("REMOTE_DURABLE_BACKUP_ONLY")
        row[f"{prefix}.backup_sha256"] = canonical_json(expected_sha)
        row[f"{prefix}.backup_size_bytes"] = canonical_json(expected_size)
        row[f"{prefix}.backup_destination"] = canonical_json(
            str(backup_record.get("destination"))
        )
        row[f"{prefix}.backup_verification_status"] = canonical_json("PASS")
        row[f"{prefix}.durable_recovery_point"] = canonical_json(durable)
        remote_resume_verified += 1

    return local_verified, remote_resume_verified


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and build rows but do not write eligible_model_inventory.csv.",
    )
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    inventory_path = root / OFFICIAL_INVENTORY_REL
    runs_root = root / OFFICIAL_RUNS_REL
    output_path = root / OUTPUT_REL

    if not inventory_path.is_file():
        raise S912InventoryError(f"OFFICIAL_INVENTORY_MISSING:{inventory_path}")

    with inventory_path.open(encoding="utf-8-sig", newline="") as fh:
        official_rows = list(csv.DictReader(fh))

    if len(official_rows) != 180:
        raise S912InventoryError(f"OFFICIAL_INVENTORY_ROW_COUNT:{len(official_rows)}")

    inventory_sha = file_sha256(inventory_path)
    output_rows: list[dict[str, str]] = []
    seen_run_ids: set[str] = set()
    seen_checkpoint_hashes: set[str] = set()
    group_counts: dict[tuple[str, str, str], int] = {}
    local_manifest_artifacts_verified = 0
    durable_resume_backups_verified = 0

    for ordinal, inv in enumerate(official_rows, start=1):
        run_id = str(inv.get("run_id", "")).strip()
        method = str(inv.get("method", "")).strip()
        condition_role = str(inv.get("condition_role", "")).strip()
        budget = str(inv.get("budget", "")).strip()

        if not run_id:
            raise S912InventoryError(f"EMPTY_RUN_ID_AT_ROW:{ordinal}")
        if run_id in seen_run_ids:
            raise S912InventoryError(f"DUPLICATE_RUN_ID:{run_id}")

        key = (condition_role, method, budget)
        if key not in EXPECTED_GROUPS:
            raise S912InventoryError(
                f"INELIGIBLE_POPULATION_MEMBER:{run_id}:{condition_role}:{method}:{budget}"
            )

        run_root = runs_root / run_id
        logical_status_path = run_root / "logical_run_status.json"
        logical_status = load_json(logical_status_path)

        eligibility_checks = {
            "run_id_match": logical_status.get("run_id") == run_id,
            "training_status_closed_verified":
                logical_status.get("training_status") == "CLOSED_VERIFIED",
            "training_completed": logical_status.get("training_completed") is True,
            "training_valid": logical_status.get("training_valid") is True,
            "eligible_for_final_test":
                logical_status.get("eligible_for_final_test") is True,
            "final_evaluation_not_authorized":
                logical_status.get("final_evaluation_status") == "NOT_AUTHORIZED",
        }
        if not all(eligibility_checks.values()):
            raise S912InventoryError(
                f"RUNTIME_ELIGIBILITY_FAILED:{run_id}:{canonical_json(eligibility_checks)}"
            )

        attempt_id = str(logical_status.get("attempt_used", "")).strip()
        if not attempt_id.startswith("attempt_"):
            raise S912InventoryError(f"INVALID_ATTEMPT_USED:{run_id}:{attempt_id}")
        attempt_dir = run_root / attempt_id

        if method == "SUP":
            checkpoint_role = "BEST_CHECKPOINT"
            checkpoint_rel = "checkpoints/best_model.pth"
            validation_rel = "metrics/validation/validation_best.json"
        elif method == "SSL":
            checkpoint_role = "BEST_EMA_TEACHER_CHECKPOINT"
            checkpoint_rel = "checkpoints/best_ema_teacher.pth"
            validation_rel = "metrics/validation/validation_best_teacher.json"
        else:
            raise S912InventoryError(f"UNEXPECTED_METHOD:{run_id}:{method}")

        checkpoint_path = attempt_dir / checkpoint_rel
        validation_path = attempt_dir / validation_rel
        if not checkpoint_path.is_file():
            raise S912InventoryError(f"SELECTED_CHECKPOINT_MISSING:{run_id}:{checkpoint_rel}")

        source_paths: dict[str, Path] = {
            "logical_run_status": logical_status_path,
            "validation_best": validation_path,
        }
        for name, rel in COMMON_JSON_FILES.items():
            source_paths[name] = attempt_dir / rel
        if method == "SSL":
            for name, rel in SSL_JSON_FILES.items():
                source_paths[name] = attempt_dir / rel

        blocks: dict[str, dict[str, Any]] = {
            "official_inventory": dict(inv),
            "logical_run_status": logical_status,
        }
        for name, path in source_paths.items():
            if name == "logical_run_status":
                continue
            blocks[name] = load_json(path)

        artifact_manifest = blocks["artifact_manifest"]
        backup_sync_manifest = blocks["backup_sync_manifest"]

        entries = {
            str(item.get("path", "")): item
            for item in artifact_manifest.get("artifacts", [])
            if isinstance(item, dict)
        }
        checkpoint_entry = entries.get(checkpoint_rel)
        if not isinstance(checkpoint_entry, dict):
            raise S912InventoryError(
                f"CHECKPOINT_NOT_IN_ARTIFACT_MANIFEST:{run_id}:{checkpoint_rel}"
            )

        actual_checkpoint_sha = file_sha256(checkpoint_path)
        manifest_checkpoint_sha = str(checkpoint_entry.get("sha256", "")).lower()
        if actual_checkpoint_sha != manifest_checkpoint_sha:
            raise S912InventoryError(f"CHECKPOINT_SHA256_MISMATCH:{run_id}")
        if checkpoint_entry.get("role") != checkpoint_role:
            raise S912InventoryError(
                f"CHECKPOINT_ROLE_MISMATCH:{run_id}:"
                f"{checkpoint_entry.get('role')}:{checkpoint_role}"
            )

        completion = blocks["training_completion_report"]
        if completion.get("training_status") != "CLOSED_VERIFIED":
            raise S912InventoryError(f"COMPLETION_STATUS_INVALID:{run_id}")
        if completion.get("eligible_for_final_test") is not True:
            raise S912InventoryError(f"COMPLETION_ELIGIBILITY_INVALID:{run_id}")
        if completion.get("scientific_protocol_deviation") is not False:
            raise S912InventoryError(f"SCIENTIFIC_PROTOCOL_DEVIATION:{run_id}")

        retry = blocks["retry_deviation"]
        if retry.get("training_seed") != logical_status.get("training_seed"):
            raise S912InventoryError(f"RETRY_SEED_MISMATCH:{run_id}")
        if retry.get("same_seed_confirmed") is not True:
            raise S912InventoryError(f"SAME_SEED_NOT_CONFIRMED:{run_id}")
        if retry.get("technical_failure") is not False:
            raise S912InventoryError(f"SUCCESSFUL_ATTEMPT_TECHNICAL_FAILURE:{run_id}")
        if retry.get("retry_required") is not False:
            raise S912InventoryError(f"SUCCESSFUL_ATTEMPT_RETRY_REQUIRED:{run_id}")

        if backup_sync_manifest.get("verification_status") != "PASS":
            raise S912InventoryError(f"BACKUP_VERIFICATION_NOT_PASS:{run_id}")
        if backup_sync_manifest.get("BACKUP_WORKER_STATUS") != "HEALTHY":
            raise S912InventoryError(f"BACKUP_WORKER_NOT_HEALTHY:{run_id}")

        row: dict[str, str] = {
            "s9_12.inventory_ordinal": canonical_json(ordinal),
            "s9_12.run_id": canonical_json(run_id),
            "s9_12.attempt_used": canonical_json(attempt_id),
        }

        blocks["selected_checkpoint"] = {
            "role": checkpoint_role,
            "relative_path": checkpoint_rel,
            "runtime_path": str(checkpoint_path),
            "sha256": actual_checkpoint_sha,
            "size_bytes": checkpoint_path.stat().st_size,
        }

        structured_order = [
            "official_inventory",
            "logical_run_status",
            "selected_checkpoint",
            "validation_best",
            "checkpoint_index",
            "training_completion_report",
            "pre_run_assertions",
            "run_manifest",
            "data_provenance",
            "seed_runtime",
            "retry_deviation",
            "backup_sync_manifest",
            "environment_reference",
            "artifact_manifest",
            "training_summary",
        ]
        if method == "SSL":
            structured_order.append("qpseudo_validation")

        for block_name in structured_order:
            flatten(block_name, blocks[block_name], row)

        row["source_paths.official_inventory"] = canonical_json(
            str(OFFICIAL_INVENTORY_REL)
        )
        row["source_sha256.official_inventory"] = canonical_json(inventory_sha)

        for source_name, source_path in sorted(source_paths.items()):
            try:
                rel_source = source_path.relative_to(root)
                source_value = str(rel_source)
            except ValueError:
                source_value = str(source_path)
            row[f"source_paths.{source_name}"] = canonical_json(source_value)
            row[f"source_sha256.{source_name}"] = canonical_json(
                file_sha256(source_path)
            )

        lv, rv = verify_manifest_artifacts(
            run_id=run_id,
            attempt_dir=attempt_dir,
            artifact_manifest=artifact_manifest,
            backup_sync_manifest=backup_sync_manifest,
            row=row,
        )
        local_manifest_artifacts_verified += lv
        durable_resume_backups_verified += rv

        output_rows.append(row)
        seen_run_ids.add(run_id)

        if actual_checkpoint_sha in seen_checkpoint_hashes:
            raise S912InventoryError(
                f"DUPLICATE_SELECTED_CHECKPOINT_SHA256:{run_id}:{actual_checkpoint_sha}"
            )
        seen_checkpoint_hashes.add(actual_checkpoint_sha)
        group_counts[key] = group_counts.get(key, 0) + 1

    if group_counts != EXPECTED_GROUPS:
        raise S912InventoryError(
            "ELIGIBLE_GROUP_COUNTS_MISMATCH:"
            + canonical_json({"|".join(k): v for k, v in sorted(group_counts.items())})
        )
    if len(output_rows) != 180:
        raise S912InventoryError(f"ELIGIBLE_MODEL_COUNT:{len(output_rows)}")
    if len(seen_run_ids) != 180:
        raise S912InventoryError(f"UNIQUE_RUN_ID_COUNT:{len(seen_run_ids)}")
    if len(seen_checkpoint_hashes) != 180:
        raise S912InventoryError(
            f"UNIQUE_CHECKPOINT_SHA256_COUNT:{len(seen_checkpoint_hashes)}"
        )
    if durable_resume_backups_verified != 180:
        raise S912InventoryError(
            f"DURABLE_RESUME_BACKUP_COUNT:{durable_resume_backups_verified}"
        )

    preferred_prefixes = [
        "s9_12.",
        "official_inventory.",
        "logical_run_status.",
        "selected_checkpoint.",
        "validation_best.",
        "checkpoint_index.",
        "training_completion_report.",
        "pre_run_assertions.",
        "run_manifest.",
        "data_provenance.",
        "seed_runtime.",
        "retry_deviation.",
        "backup_sync_manifest.",
        "environment_reference.",
        "artifact_manifest.",
        "training_summary.",
        "qpseudo_validation.",
        "retained_artifacts.",
        "source_paths.",
        "source_sha256.",
    ]

    all_fields = set().union(*(row.keys() for row in output_rows))
    fieldnames: list[str] = []
    for prefix in preferred_prefixes:
        fieldnames.extend(
            sorted(
                f
                for f in all_fields
                if f.startswith(prefix) and f not in fieldnames
            )
        )
    fieldnames.extend(sorted(all_fields - set(fieldnames)))

    if not args.dry_run:
        atomic_write_csv(output_path, fieldnames, output_rows)

    print("ELIGIBLE_MODEL_COUNT=180")
    print("UNIQUE_RUN_ID_COUNT=180")
    print("UNIQUE_CHECKPOINT_SHA256_COUNT=180")
    print("COLUMN_COUNT="+str(len(fieldnames)))
    print("LOCAL_MANIFEST_ARTIFACTS_VERIFIED="+str(local_manifest_artifacts_verified))
    print("DURABLE_RESUME_BACKUPS_VERIFIED="+str(durable_resume_backups_verified))
    print(
        "GROUP_COUNTS="
        + canonical_json({"|".join(k): v for k, v in sorted(group_counts.items())})
    )
    print("OUTPUT_MODE="+("DRY_RUN" if args.dry_run else "MATERIALIZED"))
    if not args.dry_run:
        print("OUTPUT="+str(output_path))
        print("OUTPUT_SHA256="+file_sha256(output_path))
    print("FINAL_TEST_ACCESSED=False")
    print("S9_12_ELIGIBLE_MODEL_INVENTORY=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
