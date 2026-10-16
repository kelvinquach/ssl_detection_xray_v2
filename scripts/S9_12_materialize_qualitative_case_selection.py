#!/usr/bin/env python3
import argparse, csv, hashlib, json
from pathlib import Path

COLS = [
    "selection_rule_id","rule_version","split","candidate_definition",
    "selection_count","ordering_algorithm","ordering_salt","source_split_sha256",
    "materialization_gate","prediction_independent","selection_rank","image_id",
    "image_sha256","materialization_status",
]
ORDERING = 'ascending SHA256(ordering_salt + "|" + selection_rule_id + "|" + image_id)'
SALT = "SSOD_THESIS_QUAL_V1"

def sha256_file(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_json(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))

def blank_rule(rule_id, candidate_definition, count):
    return {
        "selection_rule_id": rule_id,
        "rule_version": "",
        "split": "final_test",
        "candidate_definition": candidate_definition,
        "selection_count": str(count),
        "ordering_algorithm": ORDERING,
        "ordering_salt": SALT,
        "source_split_sha256": "",
        "materialization_gate": "FINAL_TEST_AUTHORIZED",
        "prediction_independent": "TRUE",
        "selection_rank": "",
        "image_id": "",
        "image_sha256": "",
        "materialization_status": "",
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--output", default="thesis_artifacts/qualitative/qualitative_case_selection_manifest.csv")
    ap.add_argument("--materialize", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    inv = root/"artifacts/results/official_run_inventory.csv"
    valp = root/"data/coco/instances_val.json"

    if not inv.is_file():
        raise SystemExit("BLOCKED: official_run_inventory.csv missing")
    if not valp.is_file():
        raise SystemExit("BLOCKED: instances_val.json missing")

    with inv.open(encoding="utf-8-sig", newline="") as f:
        official = list(csv.DictReader(f))

    ssl = [r for r in official if r.get("method","").strip() == "SSL"]
    if len(ssl) != 80:
        raise SystemExit(f"BLOCKED: expected 80 official SSL runs, got {len(ssl)}")

    closed = 0
    for r in ssl:
        rid = r["run_id"].strip()
        sp = root/"artifacts/runs/official"/rid/"logical_run_status.json"
        if not sp.is_file():
            raise SystemExit(f"BLOCKED: missing logical status for {rid}")
        s = load_json(sp)
        if s.get("training_status") == "CLOSED_VERIFIED":
            closed += 1
    if closed != 80:
        raise SystemExit(f"BLOCKED: SSL_RUN_CLOSED_VERIFIED gate is {closed}/80")

    val = load_json(valp)
    images = val.get("images", [])
    ids = [x.get("id") for x in images]
    if len(ids) != 734 or len(set(ids)) != 734:
        raise SystemExit(f"BLOCKED: fixed validation membership invalid ({len(ids)}/{len(set(ids))})")

    by_id = {x.get("id"): x for x in images}
    ranked = []
    rule_id = "VAL_QPSEUDO_HASH_V1"
    for image_id in ids:
        token = f"{SALT}|{rule_id}|{image_id}".encode("utf-8")
        ranked.append((hashlib.sha256(token).hexdigest(), image_id))
    ranked.sort(key=lambda x: x[0])
    selected = ranked[:6]

    rows = [
        blank_rule("FT_ABNORMAL_HASH_V1", "images with >= 1 GT bbox", 6),
        blank_rule("FT_NO_FINDING_HASH_V1", "zero-GT images", 3),
    ]

    val_sha = sha256_file(valp)
    roots = [root/"data/images", root/"data/mmdet_images"]

    for rank, (_, image_id) in enumerate(selected, 1):
        fn = str(by_id[image_id].get("file_name","")).strip()
        matches = {}
        for d in roots:
            p = d/fn
            if p.is_file():
                matches[str(p.resolve())] = p
        if len(matches) != 1:
            raise SystemExit(f"BLOCKED: source image resolution for image_id={image_id} matched {len(matches)} files")
        p = next(iter(matches.values()))
        rows.append({
            "selection_rule_id": rule_id,
            "rule_version": "",
            "split": "fixed_validation",
            "candidate_definition": "all images in the fixed validation split",
            "selection_count": "6",
            "ordering_algorithm": ORDERING,
            "ordering_salt": SALT,
            "source_split_sha256": val_sha,
            "materialization_gate": "SSL_RUN_CLOSED_VERIFIED",
            "prediction_independent": "TRUE",
            "selection_rank": str(rank),
            "image_id": str(image_id),
            "image_sha256": sha256_file(p),
            "materialization_status": "",
        })

    if len(rows) != 8:
        raise SystemExit(f"BLOCKED: unexpected row count {len(rows)}")

    print("ROW_COUNT=8")
    print("FINAL_TEST_RULE_ROWS=2")
    print("VAL_QPSEUDO_SELECTED_ROWS=6")
    print("SSL_RUN_CLOSED_VERIFIED=80")
    print("VALIDATION_SPLIT_SHA256="+val_sha)
    print("FINAL_TEST_MEMBERSHIP_ACCESSED=False")
    print("FINAL_TEST_ACCESSED=False")

    if args.materialize:
        out = root/args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        print("OUTPUT="+str(out.relative_to(root)))
        print("OUTPUT_SHA256="+sha256_file(out))
        print("QUAL_SELECTION_MANIFEST_MATERIALIZED=PASS")
    else:
        print("OUTPUT_MODE=DRY_RUN_NO_WRITE")

if __name__ == "__main__":
    main()
