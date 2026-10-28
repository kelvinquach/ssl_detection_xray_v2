#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path

EXPECTED_ROWS = 10

OUTPUT_FIELDS = [
    "seed_index",
    "training_seed",
    "main_run_id",
    "main_checkpoint_sha256",
    "main_validation_metrics_path",
    "main_qpseudo_path",
]


def norm(x):
    if x is None:
        return ""
    return str(x).strip().strip('"')


def require(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def load_csv(path):
    path = Path(path)
    require(path.is_file(), f"MISSING_FILE: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_json(path):
    path = Path(path)
    require(path.is_file(), f"MISSING_FILE: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eligible-model-inventory", required=True)
    ap.add_argument("--training-summary", required=True)
    ap.add_argument("--official-root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    eligible = load_csv(args.eligible_model_inventory)
    training = load_csv(args.training_summary)
    official_root = Path(args.official_root)

    train_by_run = {}
    for r in training:
        run_id = norm(r.get("run_id"))
        require(run_id, "EMPTY_TRAINING_SUMMARY_RUN_ID")
        require(
            run_id not in train_by_run,
            f"DUPLICATE_TRAINING_SUMMARY_RUN_ID: {run_id}",
        )
        train_by_run[run_id] = r

    mains = [
        r for r in eligible
        if re.fullmatch(
            r"OFF_SSL_R50_B10_S\d{2}",
            norm(r.get("s9_12.run_id")),
        )
    ]

    require(
        len(mains) == EXPECTED_ROWS,
        f"MAIN_REFERENCE_COUNT: expected={EXPECTED_ROWS} observed={len(mains)}",
    )

    output_rows = []

    for inv in mains:
        run_id = norm(inv.get("s9_12.run_id"))
        attempt = norm(inv.get("s9_12.attempt_used"))

        require(
            norm(inv.get("official_inventory.method")) == "SSL",
            f"METHOD_MISMATCH: {run_id}",
        )
        require(
            norm(inv.get("official_inventory.condition_role")) == "MAIN_LOW_LABEL",
            f"CONDITION_ROLE_MISMATCH: {run_id}",
        )
        require(
            norm(inv.get("official_inventory.architecture")) == "R50-FPN",
            f"ARCHITECTURE_MISMATCH: {run_id}",
        )
        require(
            norm(inv.get("official_inventory.budget")) == "10pct",
            f"BUDGET_MISMATCH: {run_id}",
        )
        require(
            re.fullmatch(r"attempt_\d{3}", attempt) is not None,
            f"INVALID_ATTEMPT: {run_id}: {attempt}",
        )

        require(
            run_id in train_by_run,
            f"MISSING_TRAINING_SUMMARY: {run_id}",
        )
        tr = train_by_run[run_id]

        require(
            norm(tr.get("method")) == "SSL",
            f"TRAINING_METHOD_MISMATCH: {run_id}",
        )
        require(
            norm(tr.get("condition_role")) == "MAIN_LOW_LABEL",
            f"TRAINING_ROLE_MISMATCH: {run_id}",
        )
        require(
            norm(tr.get("architecture")) == "R50-FPN",
            f"TRAINING_ARCHITECTURE_MISMATCH: {run_id}",
        )
        require(
            norm(tr.get("budget")) == "10pct",
            f"TRAINING_BUDGET_MISMATCH: {run_id}",
        )

        seed_index = norm(tr.get("seed_index"))
        training_seed = norm(tr.get("training_seed"))
        best_sha = norm(tr.get("best_checkpoint_sha256"))

        require(seed_index, f"EMPTY_SEED_INDEX: {run_id}")
        require(training_seed, f"EMPTY_TRAINING_SEED: {run_id}")
        require(best_sha, f"EMPTY_BEST_CHECKPOINT_SHA256: {run_id}")

        base = official_root / run_id / attempt

        validation_rel = (
            f"artifacts/runs/official/{run_id}/{attempt}/"
            "metrics/validation/validation_best_teacher.json"
        )
        qpseudo_rel = (
            f"artifacts/runs/official/{run_id}/{attempt}/"
            "pseudo/qpseudo_validation.json"
        )

        validation_path = (
            base / "metrics" / "validation" / "validation_best_teacher.json"
        )
        qpseudo_path = (
            base / "pseudo" / "qpseudo_validation.json"
        )
        checkpoint_index_path = (
            base / "checkpoints" / "checkpoint_index.json"
        )

        vb = load_json(validation_path)
        qp = load_json(qpseudo_path)
        ci = load_json(checkpoint_index_path)

        require(
            norm(vb.get("checkpoint_sha256")) == best_sha,
            f"BEST_CHECKPOINT_SHA256_MISMATCH: {run_id}",
        )
        require(
            norm(vb.get("selection_metric")) == "bbox_mAP_50_95",
            f"VALIDATION_SELECTION_METRIC_MISMATCH: {run_id}",
        )
        require(
            norm(vb.get("selection_model")) == "EMA_TEACHER",
            f"VALIDATION_SELECTION_MODEL_MISMATCH: {run_id}",
        )
        require(
            norm(vb.get("rule"))
            == "ARGMAX_VALIDATION_TEACHER_BBOX_MAP_50_95",
            f"VALIDATION_RULE_MISMATCH: {run_id}",
        )

        require(
            norm(ci.get("checkpoint_policy")) == "LOCKED",
            f"CHECKPOINT_POLICY_NOT_LOCKED: {run_id}",
        )

        require(
            qp.get("PL_mAP_50_95") is not None,
            f"QPSEUDO_MISSING_PL_MAP: {run_id}",
        )
        require(
            qp.get("test_used") is False,
            f"QPSEUDO_TEST_USED: {run_id}",
        )
        require(
            qp.get("hidden_u_gt_used") is False,
            f"QPSEUDO_HIDDEN_U_GT_USED: {run_id}",
        )

        output_rows.append({
            "seed_index": seed_index,
            "training_seed": training_seed,
            "main_run_id": run_id,
            "main_checkpoint_sha256": best_sha,
            "main_validation_metrics_path": validation_rel,
            "main_qpseudo_path": qpseudo_rel,
        })

    require(
        len(output_rows) == EXPECTED_ROWS,
        f"OUTPUT_ROW_COUNT: {len(output_rows)}",
    )
    require(
        len(set(r["main_run_id"] for r in output_rows)) == EXPECTED_ROWS,
        "MAIN_RUN_IDS_NOT_UNIQUE",
    )
    require(
        len(set(r["seed_index"] for r in output_rows)) == EXPECTED_ROWS,
        "SEED_INDEX_NOT_UNIQUE",
    )
    require(
        len(set(r["training_seed"] for r in output_rows)) == EXPECTED_ROWS,
        "TRAINING_SEEDS_NOT_UNIQUE",
    )

    output_rows.sort(key=lambda r: int(r["seed_index"]))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        w.writeheader()
        w.writerows(output_rows)

    print("=== S10.1.05 ABLATION MAIN REFERENCE MAP ===")
    print(f"OUTPUT_ROWS={len(output_rows)}")
    print(f"UNIQUE_MAIN_RUN_IDS={len(set(r['main_run_id'] for r in output_rows))}")
    print(f"UNIQUE_SEED_INDEX={len(set(r['seed_index'] for r in output_rows))}")
    print(f"UNIQUE_TRAINING_SEEDS={len(set(r['training_seed'] for r in output_rows))}")
    print(f"COLUMNS={len(OUTPUT_FIELDS)}")
    print("S10_1_05_STATUS=PASS")


if __name__ == "__main__":
    main()