"""Phase 0 smoke tests for reproducibility & environment utilities.

These do not require torch/mmdet to be installed; they verify the utilities
degrade gracefully and produce serializable output.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.seed import (  # noqa: E402
    get_rng_state_summary,
    save_seed_manifest,
    set_global_seed,
)
from src.utils.env import collect_environment, probe_import  # noqa: E402


def test_set_global_seed_returns_serializable_config():
    cfg = set_global_seed(2026, deterministic=True)
    assert cfg["seed"] == 2026
    assert cfg["deterministic"] is True
    json.dumps(cfg)  # must be serializable


def test_rng_summary_is_serializable():
    set_global_seed(2026)
    summary = get_rng_state_summary()
    json.dumps(summary)
    assert "python_random" in summary


def test_save_seed_manifest(tmp_path):
    cfg = set_global_seed(123, deterministic=True)
    out = tmp_path / "seed.json"
    save_seed_manifest(out, seed=123, deterministic_config=cfg)
    data = json.loads(out.read_text())
    assert data["seed"] == 123
    assert data["manifest_type"] == "seed_state_manifest"


def test_probe_import_missing_is_graceful():
    res = probe_import("definitely_not_a_real_module_xyz")
    assert res["import_ok"] is False
    assert res["error"] is not None


def test_collect_environment_serializable():
    report = collect_environment()
    json.dumps(report)
    assert "imports" in report
    assert "cuda" in report
    assert report["summary"]["primary_framework"] == "mmdetection"
