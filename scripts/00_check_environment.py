#!/usr/bin/env python3
"""Phase 0 environment check.

Sets the global seed, collects an environment report, writes a seed-state
manifest, and dumps `pip freeze`. Designed to never crash on a partially
installed environment: if mmdet/mmcv/mmengine fail to import, the report
records `import_ok: false` for those packages and the script still exits 0
as long as the basic checks completed.

Usage:
    python scripts/00_check_environment.py \
        --seed 2026 \
        --output reports/phase0_environment_check.json \
        --seed-manifest data/manifests/seed_state_manifest.json \
        --freeze-output reports/phase0_pip_freeze.txt

This script does NOT touch any dataset.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Make `src` importable when run from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.env import collect_environment  # noqa: E402
from src.utils.seed import (  # noqa: E402
    get_rng_state_summary,
    save_seed_manifest,
    set_global_seed,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 0 environment & reproducibility check."
    )
    parser.add_argument("--seed", type=int, default=2026, help="Global seed.")
    parser.add_argument(
        "--output",
        type=str,
        default="reports/phase0_environment_check.json",
        help="Path to environment report JSON.",
    )
    parser.add_argument(
        "--seed-manifest",
        type=str,
        default="data/manifests/seed_state_manifest.json",
        help="Path to seed-state manifest JSON.",
    )
    parser.add_argument(
        "--freeze-output",
        type=str,
        default="reports/phase0_pip_freeze.txt",
        help="Path to pip freeze output.",
    )
    return parser.parse_args(argv)


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_pip_freeze(path: str | Path) -> bool:
    """Run `pip freeze` and write to path. Returns True on success."""
    out = ensure_parent(path)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=False,
        )
        out.write_text(result.stdout, encoding="utf-8")
        return result.returncode == 0
    except Exception as exc:  # pragma: no cover
        out.write_text(f"# pip freeze failed: {exc!r}\n", encoding="utf-8")
        return False


def main(argv=None) -> int:
    args = parse_args(argv)

    # 1) Set the global seed (captures determinism config).
    deterministic_config = set_global_seed(args.seed, deterministic=True)

    # 2) Collect the environment report.
    report = collect_environment()
    report["seed"] = args.seed
    report["deterministic_config"] = deterministic_config

    # 3) Write environment report JSON.
    out_report = ensure_parent(args.output)
    out_report.write_text(
        json.dumps(report, indent=2, sort_keys=False), encoding="utf-8"
    )

    # 4) Write seed-state manifest.
    rng_summary = get_rng_state_summary()
    save_seed_manifest(
        args.seed_manifest,
        seed=args.seed,
        deterministic_config=deterministic_config,
        rng_state_summary=rng_summary,
    )

    # 5) Write pip freeze.
    freeze_ok = write_pip_freeze(args.freeze_output)

    # 6) Human-readable console summary.
    imports = report["imports"]
    framework_ok = report["summary"]["framework_import_ok"]
    core_ok = report["summary"]["core_import_ok"]

    print("=" * 60)
    print("Phase 0 environment check")
    print("=" * 60)
    print(f"Python      : {report['platform']['python_version_info']}")
    print(f"Platform    : {report['platform']['platform']}")
    print(f"Seed        : {args.seed}")
    print(f"CUDA avail  : {report['cuda']['cuda_available']}")
    if report["cuda"]["devices"]:
        for dev in report["cuda"]["devices"]:
            print(f"  GPU[{dev.get('index')}]   : {dev.get('name')}")
    print("-" * 60)
    print("Imports:")
    for name, info in imports.items():
        status = "OK " if info["import_ok"] else "FAIL"
        ver = info["version"] or "-"
        print(f"  [{status}] {name:<13} {ver}")
    print("-" * 60)
    print(f"Core import ok      : {core_ok}")
    print(f"Framework import ok : {framework_ok} (mmengine/mmcv/mmdet)")
    print(f"pip freeze written  : {freeze_ok}")
    print(f"Report      -> {out_report}")
    print(f"Manifest    -> {args.seed_manifest}")
    print(f"pip freeze  -> {args.freeze_output}")
    print("=" * 60)

    if not framework_ok:
        print(
            "NOTE: MMDetection stack not fully importable "
            "(import_ok: false recorded in report). This is expected "
            "before installing mmdet/mmcv/mmengine.",
            file=sys.stderr,
        )

    # Exit 0 if the basic checks completed (report + manifest written).
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
