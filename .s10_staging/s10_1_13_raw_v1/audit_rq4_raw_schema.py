#!/usr/bin/env python3
import csv
import json
from pathlib import Path

print("=== S10.1.13 RQ4 RAW SCHEMA AUDIT ===")

json_files = [
    "rq4_primary_mixed_model.raw.json",
    "rq4_diagnostics.raw.json",
    "rq4_cr2_sensitivity.raw.json",
    "rq4_runtime_environment.raw.json",
]

for name in json_files:
    obj = json.loads(Path(name).read_text(encoding="utf-8"))
    print(f"--- {name} ---")
    print("TOP_KEYS=" + ",".join(obj.keys()))
    for k, v in obj.items():
        if isinstance(v, dict):
            print(f"{k}_KEYS=" + ",".join(v.keys()))
        elif isinstance(v, list):
            print(f"{k}_LIST_LEN={len(v)}")

csv_files = [
    "rq4_analysis_dataset.raw.csv",
    "rq4_seed_influence.raw.csv",
]

for name in csv_files:
    with open(name, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    print(f"--- {name} ---")
    print("HEADER=" + ",".join(rows[0].keys()))
    print("ROWS=" + str(len(rows)))
    print("FIRST_ROW=" + json.dumps(rows[0], ensure_ascii=False, sort_keys=True))
    print("LAST_ROW=" + json.dumps(rows[-1], ensure_ascii=False, sort_keys=True))

print("S10_1_13_RQ4_RAW_SCHEMA_AUDIT=PASS")
