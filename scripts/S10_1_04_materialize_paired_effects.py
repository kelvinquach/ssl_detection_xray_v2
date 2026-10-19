#!/usr/bin/env python3
import argparse
import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path

EXPECTED_PAIRS = 80

OUTPUT_FIELDS = [
    "pair_id",
    "architecture",
    "budget",
    "seed_index",
    "training_seed",

    "mAP_test_SUP",
    "mAP_test_SSL",
    "delta_mAP_test",

    "FP_per_negative_test_SUP",
    "FP_per_negative_test_SSL",
    "delta_FP_per_negative_test",

    "AP_test_Atelectasis_SUP",
    "AP_test_Atelectasis_SSL",
    "delta_AP_test_Atelectasis",

    "AP_test_Pneumothorax_SUP",
    "AP_test_Pneumothorax_SSL",
    "delta_AP_test_Pneumothorax",

    "Q_pseudo_validation_SSL",
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


def numeric(x, label):
    s = norm(x)
    require(s not in {"", "NA"}, f"MISSING_NUMERIC_VALUE: {label}")
    try:
        Decimal(s)
    except InvalidOperation:
        raise RuntimeError(f"INVALID_NUMERIC_VALUE: {label}: {s!r}")
    return s


def delta_required(ssl_value, sup_value, label):
    ssl_s = numeric(ssl_value, f"{label}:SSL")
    sup_s = numeric(sup_value, f"{label}:SUP")
    return format(Decimal(ssl_s) - Decimal(sup_s), "f")


def delta_optional(ssl_value, sup_value, label):
    ssl_s = norm(ssl_value)
    sup_s = norm(sup_value)

    if ssl_s in {"", "NA"} or sup_s in {"", "NA"}:
        return "NA"

    try:
        return format(Decimal(ssl_s) - Decimal(sup_s), "f")
    except InvalidOperation:
        raise RuntimeError(
            f"INVALID_OPTIONAL_NUMERIC_VALUE: {label}: "
            f"SUP={sup_s!r} SSL={ssl_s!r}"
        )


def same(a, b, field, pair_id):
    av = norm(a.get(field))
    bv = norm(b.get(field))
    require(
        av == bv,
        f"PAIR_IDENTITY_MISMATCH: {pair_id}: {field}: "
        f"SUP={av!r} SSL={bv!r}",
    )
    return av


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-inventory", required=True)
    ap.add_argument("--final-test-results", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    pairs = load_csv(args.pair_inventory)
    final_rows = load_csv(args.final_test_results)

    require(
        len(pairs) == EXPECTED_PAIRS,
        f"PAIR_INVENTORY_ROW_COUNT: expected={EXPECTED_PAIRS} observed={len(pairs)}",
    )

    valid_pairs = [p for p in pairs if norm(p.get("pair_valid")).lower() == "true"]
    require(
        len(valid_pairs) == EXPECTED_PAIRS,
        f"VALID_PAIR_COUNT: expected={EXPECTED_PAIRS} observed={len(valid_pairs)}",
    )

    pair_ids = [norm(p.get("pair_id")) for p in valid_pairs]
    require(
        len(set(pair_ids)) == EXPECTED_PAIRS,
        "PAIR_IDS_NOT_UNIQUE",
    )

    final_by_run = {}
    for r in final_rows:
        run_id = norm(r.get("run_id"))
        require(run_id, "EMPTY_FINAL_TEST_RUN_ID")
        require(
            run_id not in final_by_run,
            f"DUPLICATE_FINAL_TEST_RUN_ID: {run_id}",
        )
        final_by_run[run_id] = r

    output_rows = []

    for p in valid_pairs:
        pair_id = norm(p.get("pair_id"))
        sup_run_id = norm(p.get("sup_run_id"))
        ssl_run_id = norm(p.get("ssl_run_id"))

        require(
            sup_run_id in final_by_run,
            f"MISSING_SUP_FINAL_TEST_ROW: {pair_id}: {sup_run_id}",
        )
        require(
            ssl_run_id in final_by_run,
            f"MISSING_SSL_FINAL_TEST_ROW: {pair_id}: {ssl_run_id}",
        )

        sup = final_by_run[sup_run_id]
        ssl = final_by_run[ssl_run_id]

        require(
            norm(sup.get("method")) == "SUP",
            f"SUP_METHOD_MISMATCH: {pair_id}",
        )
        require(
            norm(ssl.get("method")) == "SSL",
            f"SSL_METHOD_MISMATCH: {pair_id}",
        )
        require(
            norm(sup.get("condition_role")) == "MAIN_LOW_LABEL",
            f"SUP_NOT_MAIN_LOW_LABEL: {pair_id}",
        )
        require(
            norm(ssl.get("condition_role")) == "MAIN_LOW_LABEL",
            f"SSL_NOT_MAIN_LOW_LABEL: {pair_id}",
        )

        for row, expected_run_id, side in (
            (sup, sup_run_id, "SUP"),
            (ssl, ssl_run_id, "SSL"),
        ):
            require(
                norm(row.get("run_id")) == expected_run_id,
                f"{side}_RUN_ID_MISMATCH: {pair_id}",
            )
            require(
                norm(row.get("final_evaluation_status")) == "CLOSED_VERIFIED",
                f"{side}_FINAL_EVALUATION_NOT_CLOSED_VERIFIED: {pair_id}",
            )

        architecture = same(sup, ssl, "architecture", pair_id)
        budget = same(sup, ssl, "budget", pair_id)
        seed_index = same(sup, ssl, "seed_index", pair_id)
        training_seed = same(sup, ssl, "training_seed", pair_id)

        require(
            architecture == norm(p.get("architecture")),
            f"PAIR_ARCHITECTURE_MISMATCH: {pair_id}",
        )
        require(
            budget == norm(p.get("budget")),
            f"PAIR_BUDGET_MISMATCH: {pair_id}",
        )
        require(
            seed_index == norm(p.get("seed_index")),
            f"PAIR_SEED_INDEX_MISMATCH: {pair_id}",
        )
        require(
            training_seed == norm(p.get("training_seed")),
            f"PAIR_TRAINING_SEED_MISMATCH: {pair_id}",
        )

        sup_map = numeric(
            sup.get("test_mAP_50_95"),
            f"{pair_id}:mAP:SUP",
        )
        ssl_map = numeric(
            ssl.get("test_mAP_50_95"),
            f"{pair_id}:mAP:SSL",
        )

        sup_fp = numeric(
            sup.get("FP_per_negative"),
            f"{pair_id}:FP_per_negative:SUP",
        )
        ssl_fp = numeric(
            ssl.get("FP_per_negative"),
            f"{pair_id}:FP_per_negative:SSL",
        )

        sup_atel = norm(sup.get("AP_Atelectasis"))
        ssl_atel = norm(ssl.get("AP_Atelectasis"))
        sup_pneu = norm(sup.get("AP_Pneumothorax"))
        ssl_pneu = norm(ssl.get("AP_Pneumothorax"))

        q_pseudo = numeric(
            ssl.get("Q_pseudo"),
            f"{pair_id}:Q_pseudo_validation_SSL",
        )
        require(
            norm(ssl.get("Q_pseudo_status")) == "AVAILABLE",
            f"SSL_QPSEUDO_NOT_AVAILABLE: {pair_id}",
        )

        row = {
            "pair_id": pair_id,
            "architecture": architecture,
            "budget": budget,
            "seed_index": seed_index,
            "training_seed": training_seed,

            "mAP_test_SUP": sup_map,
            "mAP_test_SSL": ssl_map,
            "delta_mAP_test": delta_required(
                ssl_map, sup_map, f"{pair_id}:mAP"
            ),

            "FP_per_negative_test_SUP": sup_fp,
            "FP_per_negative_test_SSL": ssl_fp,
            "delta_FP_per_negative_test": delta_required(
                ssl_fp, sup_fp, f"{pair_id}:FP_per_negative"
            ),

            "AP_test_Atelectasis_SUP": sup_atel if sup_atel else "NA",
            "AP_test_Atelectasis_SSL": ssl_atel if ssl_atel else "NA",
            "delta_AP_test_Atelectasis": delta_optional(
                ssl_atel, sup_atel, f"{pair_id}:Atelectasis"
            ),

            "AP_test_Pneumothorax_SUP": sup_pneu if sup_pneu else "NA",
            "AP_test_Pneumothorax_SSL": ssl_pneu if ssl_pneu else "NA",
            "delta_AP_test_Pneumothorax": delta_optional(
                ssl_pneu, sup_pneu, f"{pair_id}:Pneumothorax"
            ),

            "Q_pseudo_validation_SSL": q_pseudo,
        }

        output_rows.append(row)

    require(
        len(output_rows) == EXPECTED_PAIRS,
        f"OUTPUT_ROW_COUNT: expected={EXPECTED_PAIRS} observed={len(output_rows)}",
    )

    require(
        len(set(r["pair_id"] for r in output_rows)) == EXPECTED_PAIRS,
        "OUTPUT_PAIR_IDS_NOT_UNIQUE",
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        w.writeheader()
        w.writerows(output_rows)

    print("=== S10.1.04 PAIRED EFFECTS ===")
    print(f"PAIR_INPUT_ROWS={len(pairs)}")
    print(f"VALID_PAIR_ROWS={len(valid_pairs)}")
    print(f"FINAL_TEST_INPUT_ROWS={len(final_rows)}")
    print(f"OUTPUT_ROWS={len(output_rows)}")
    print(f"UNIQUE_PAIR_IDS={len(set(r['pair_id'] for r in output_rows))}")
    print(f"COLUMNS={len(OUTPUT_FIELDS)}")
    print("DELTA_DIRECTION=SSL_MINUS_SUP")
    print("S10_1_04_STATUS=PASS")


if __name__ == "__main__":
    main()