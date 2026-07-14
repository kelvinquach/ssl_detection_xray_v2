"""Regression tests for the Phase 2D defensive guardrails.

Covers the two maintenance-patch concerns:
  1. The protocol YAML must genuinely drive the script and must not be able to
     drift silently away from Phase 2B / the canonical tables.
  2. The final coco_master.json must be atomically replaced ONLY after every hard
     check has passed; a failed run must leave any previous output untouched.

Standard-library unittest only. No new dependencies. No VinBigData data needed.

Run:
    python -m unittest tests.test_phase2D_guardrails -v
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "02D_build_coco_master.py"


def _load_module():
    """Import the Phase 2D script by path (its name is not a valid identifier)."""
    spec = importlib.util.spec_from_file_location("phase2d", SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"cannot load {SCRIPT_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase2d"] = mod
    spec.loader.exec_module(mod)
    return mod


P2D = _load_module()


# A minimal but structurally complete protocol document.
VALID_PROTOCOL = {
    "phase": "phase2D_coco_master_conversion",
    "expected_counts": {
        "images": 4894,
        "annotations": 36096,
        "categories": 14,
        "abnormal_images": 4394,
        "no_finding_images": 500,
        "no_finding_annotations": 0,
    },
    "tolerance": {
        "area_rel_tol": 1e-9,
        "area_abs_tol": 1e-6,
        "boundary_abs_tol": 1e-6,
        "coordinate_abs_tol": 1e-9,
    },
    "bbox_policy": {
        "source_format": "xyxy_original_image",
        "target_format": "coco_xywh_absolute",
    },
    "category_policy": {
        "supercategory": "chest_abnormality",
        "forbidden_category_names": ["no finding", "background", "normal"],
    },
    "path_policy": {"image_root_env_var": "VINBIGDATA_DICOM_ROOT"},
}

# Phase 2B fixture consistent with the valid protocol.
PHASE2B_FIXTURE = {
    "canonical_image_rows": 4894,
    "canonical_bbox_rows": 36096,
    "canonical_class_count": 14,
    "abnormal_images": 4394,
    "no_finding_images": 500,
    "dod_pass_candidate": True,
}


def write_yaml(path: Path, doc: dict) -> Path:
    """Serialize a protocol dict to YAML (PyYAML is a script dependency anyway)."""
    import yaml

    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestYamlStrictLoading(unittest.TestCase):
    """Test 1 — the strict loader must accept valid YAML and reject everything else."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_valid_yaml_loads(self) -> None:
        p = write_yaml(self.dir / "ok.yaml", VALID_PROTOCOL)
        cfg = P2D.load_protocol_strict(str(p))
        self.assertEqual(cfg.expect_images, 4894)
        self.assertEqual(cfg.expect_annotations, 36096)
        self.assertEqual(cfg.expect_categories, 14)
        self.assertEqual(cfg.expect_abnormal_images, 4394)
        self.assertEqual(cfg.expect_no_finding_images, 500)
        self.assertEqual(cfg.bbox_source_format, "xyxy_original_image")
        self.assertEqual(cfg.supercategory, "chest_abnormality")
        self.assertEqual(cfg.path_root_variable, "VINBIGDATA_DICOM_ROOT")
        self.assertIn("no finding", cfg.forbidden_category_names)

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(P2D.ProtocolError):
            P2D.load_protocol_strict(str(self.dir / "does_not_exist.yaml"))

    def test_missing_required_section_raises(self) -> None:
        doc = copy.deepcopy(VALID_PROTOCOL)
        del doc["tolerance"]
        p = write_yaml(self.dir / "no_tol.yaml", doc)
        with self.assertRaises(P2D.ProtocolError) as ctx:
            P2D.load_protocol_strict(str(p))
        self.assertIn("tolerance", str(ctx.exception))

    def test_missing_required_key_raises(self) -> None:
        doc = copy.deepcopy(VALID_PROTOCOL)
        del doc["expected_counts"]["annotations"]
        p = write_yaml(self.dir / "no_ann.yaml", doc)
        with self.assertRaises(P2D.ProtocolError) as ctx:
            P2D.load_protocol_strict(str(p))
        self.assertIn("annotations", str(ctx.exception))

    def test_malformed_yaml_raises(self) -> None:
        p = self.dir / "bad.yaml"
        p.write_text("expected_counts: [unclosed\n  : : :\n", encoding="utf-8")
        with self.assertRaises(P2D.ProtocolError):
            P2D.load_protocol_strict(str(p))

    def test_non_mapping_yaml_raises(self) -> None:
        p = self.dir / "list.yaml"
        p.write_text("- a\n- b\n", encoding="utf-8")
        with self.assertRaises(P2D.ProtocolError) as ctx:
            P2D.load_protocol_strict(str(p))
        self.assertIn("mapping", str(ctx.exception))

    def test_negative_tolerance_raises(self) -> None:
        doc = copy.deepcopy(VALID_PROTOCOL)
        doc["tolerance"]["area_abs_tol"] = -1e-6
        p = write_yaml(self.dir / "neg_tol.yaml", doc)
        with self.assertRaises(P2D.ProtocolError) as ctx:
            P2D.load_protocol_strict(str(p))
        self.assertIn("area_abs_tol", str(ctx.exception))

    def test_negative_count_raises(self) -> None:
        doc = copy.deepcopy(VALID_PROTOCOL)
        doc["expected_counts"]["images"] = -1
        p = write_yaml(self.dir / "neg_count.yaml", doc)
        with self.assertRaises(P2D.ProtocolError):
            P2D.load_protocol_strict(str(p))

    def test_non_integer_count_raises(self) -> None:
        doc = copy.deepcopy(VALID_PROTOCOL)
        doc["expected_counts"]["images"] = "4894"  # string, not int
        p = write_yaml(self.dir / "str_count.yaml", doc)
        with self.assertRaises(P2D.ProtocolError) as ctx:
            P2D.load_protocol_strict(str(p))
        self.assertIn("integer", str(ctx.exception))

    def test_float_count_raises(self) -> None:
        doc = copy.deepcopy(VALID_PROTOCOL)
        doc["expected_counts"]["annotations"] = 36096.5
        p = write_yaml(self.dir / "float_count.yaml", doc)
        with self.assertRaises(P2D.ProtocolError):
            P2D.load_protocol_strict(str(p))

    def test_non_finite_tolerance_raises(self) -> None:
        p = self.dir / "inf_tol.yaml"
        doc = copy.deepcopy(VALID_PROTOCOL)
        write_yaml(p, doc)
        text = p.read_text(encoding="utf-8").replace("area_rel_tol: 1.0e-09", "area_rel_tol: .inf")
        p.write_text(text, encoding="utf-8")
        with self.assertRaises(P2D.ProtocolError) as ctx:
            P2D.load_protocol_strict(str(p))
        self.assertIn("finite", str(ctx.exception))

    def test_internally_inconsistent_counts_raise(self) -> None:
        doc = copy.deepcopy(VALID_PROTOCOL)
        doc["expected_counts"]["abnormal_images"] = 4000  # 4000 + 500 != 4894
        p = write_yaml(self.dir / "inconsistent.yaml", doc)
        with self.assertRaises(P2D.ProtocolError) as ctx:
            P2D.load_protocol_strict(str(p))
        self.assertIn("internally inconsistent", str(ctx.exception))

    def test_missing_forbidden_names_raises(self) -> None:
        doc = copy.deepcopy(VALID_PROTOCOL)
        del doc["category_policy"]["forbidden_category_names"]
        p = write_yaml(self.dir / "no_forbidden.yaml", doc)
        with self.assertRaises(P2D.ProtocolError):
            P2D.load_protocol_strict(str(p))

    def test_no_silent_fallback_to_empty_dict(self) -> None:
        """A missing YAML must RAISE, never return {}."""
        try:
            result = P2D.load_protocol_strict(str(self.dir / "absent.yaml"))
        except P2D.ProtocolError:
            return  # correct
        self.fail(f"expected ProtocolError, silently returned {result!r}")


class TestProtocolDrift(unittest.TestCase):
    """Test 2 — YAML that disagrees with Phase 2B must hard-fail with PROTOCOL_DRIFT."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_no_drift_when_consistent(self) -> None:
        p = write_yaml(self.dir / "ok.yaml", VALID_PROTOCOL)
        cfg = P2D.load_protocol_strict(str(p))
        drift = P2D.crosscheck_protocol(
            cfg, PHASE2B_FIXTURE,
            actual_image_rows=4894, actual_bbox_rows=36096, actual_class_rows=14,
            actual_abnormal_images=4394, actual_no_finding_images=500,
        )
        self.assertEqual(drift, [])

    def test_yaml_images_drift_against_phase2b(self) -> None:
        doc = copy.deepcopy(VALID_PROTOCOL)
        doc["expected_counts"]["images"] = 4895
        doc["expected_counts"]["abnormal_images"] = 4395  # keep YAML self-consistent
        p = write_yaml(self.dir / "drift.yaml", doc)
        cfg = P2D.load_protocol_strict(str(p))

        drift = P2D.crosscheck_protocol(
            cfg, PHASE2B_FIXTURE,
            actual_image_rows=4894, actual_bbox_rows=36096, actual_class_rows=14,
            actual_abnormal_images=4394, actual_no_finding_images=500,
        )
        self.assertTrue(drift, "expected drift to be detected")
        joined = "\n".join(drift)
        self.assertIn("PROTOCOL_DRIFT", joined)
        self.assertIn("expected_images=4895", joined)
        self.assertIn("canonical_image_rows=4894", joined)

    def test_yaml_annotations_drift_against_actual_table(self) -> None:
        p = write_yaml(self.dir / "ok.yaml", VALID_PROTOCOL)
        cfg = P2D.load_protocol_strict(str(p))
        # YAML and Phase 2B agree, but the actual bbox table has a different size.
        drift = P2D.crosscheck_protocol(
            cfg, PHASE2B_FIXTURE,
            actual_image_rows=4894, actual_bbox_rows=36095, actual_class_rows=14,
            actual_abnormal_images=4394, actual_no_finding_images=500,
        )
        self.assertTrue(drift)
        self.assertIn("PROTOCOL_DRIFT", "\n".join(drift))

    def test_drift_blocks_the_run_end_to_end(self) -> None:
        """A drifting YAML must abort main() before any output is produced."""
        doc = copy.deepcopy(VALID_PROTOCOL)
        doc["expected_counts"]["images"] = 4895
        doc["expected_counts"]["abnormal_images"] = 4395
        yaml_path = write_yaml(self.dir / "drift.yaml", doc)

        p2b = self.dir / "p2b.json"
        p2b.write_text(json.dumps(PHASE2B_FIXTURE), encoding="utf-8")

        img = self.dir / "img.csv"
        img.write_text(
            "canonical_image_id,image_id,relative_dicom_path,image_width,image_height,"
            "is_negative,bbox_count\n0,a,train/a.dicom,100,100,False,0\n",
            encoding="utf-8",
        )
        bb = self.dir / "bb.csv"
        bb.write_text(
            "canonical_ann_id,image_id,canonical_class_id,x_min,y_min,x_max,y_max\n",
            encoding="utf-8",
        )
        cm = self.dir / "cm.csv"
        cm.write_text("canonical_class_id,class_name\n0,Cardiomegaly\n", encoding="utf-8")

        final = self.dir / "coco_master.json"
        rc = P2D.main([
            "--protocol-yaml", str(yaml_path),
            "--phase2b-validation-json", str(p2b),
            "--canonical-image-table", str(img),
            "--canonical-bbox-table", str(bb),
            "--canonical-class-mapping", str(cm),
            "--coco-master", str(final),
            "--validation-json", str(self.dir / "v.json"),
            "--report-md", str(self.dir / "r.md"),
            "--image-counts-csv", str(self.dir / "1.csv"),
            "--category-summary-csv", str(self.dir / "2.csv"),
            "--invalid-annotations-csv", str(self.dir / "3.csv"),
            "--no-finding-audit-csv", str(self.dir / "4.csv"),
        ])
        self.assertNotEqual(rc, 0, "drifting protocol must not exit 0")
        self.assertFalse(final.exists(), "no COCO output may be produced on drift")


class TestOutputPreservation(unittest.TestCase):
    """Test 3 — a failed run must leave a pre-existing final output untouched."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.final = self.dir / "coco_master.json"
        self.final.write_text(
            json.dumps({"marker": "PREVIOUS_VALID_OUTPUT", "images": [], "annotations": [],
                        "categories": []}),
            encoding="utf-8",
        )
        self.before_sha = sha256(self.final)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _make_temp(self) -> str:
        fd, name = tempfile.mkstemp(dir=str(self.dir), prefix=".tmp_", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"marker": "NEW_BUT_INVALID"}, fh)
        return name

    def test_failed_checks_preserve_previous_output(self) -> None:
        tmp = self._make_temp()
        out = P2D.promote_atomic(tmp, self.final, all_pre_promotion_checks_pass=False)

        self.assertFalse(out["final_output_replaced"])
        self.assertTrue(out["previous_valid_output_preserved_on_failure"])
        self.assertEqual(sha256(self.final), self.before_sha, "final output was modified")
        self.assertEqual(
            json.loads(self.final.read_text())["marker"], "PREVIOUS_VALID_OUTPUT"
        )
        self.assertFalse(Path(tmp).exists(), "temporary file must be cleaned up")

    def test_missing_temp_preserves_previous_output(self) -> None:
        out = P2D.promote_atomic(
            str(self.dir / "nonexistent.tmp"), self.final,
            all_pre_promotion_checks_pass=True,
        )
        self.assertFalse(out["final_output_replaced"])
        self.assertEqual(sha256(self.final), self.before_sha)


class TestAtomicPromotion(unittest.TestCase):
    """Test 4 — a fully passing run promotes the temp file atomically."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.final = self.dir / "coco_master.json"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_successful_promotion(self) -> None:
        payload = {
            "images": [{"id": 1, "file_name": "train/a.dicom", "width": 10, "height": 10}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5],
                 "area": 25, "iscrowd": 0}
            ],
            "categories": [{"id": 1, "name": "Cardiomegaly"}],
        }
        fd, tmp = tempfile.mkstemp(dir=str(self.dir), prefix=".tmp_", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, allow_nan=False)
        temp_sha = sha256(Path(tmp))

        out = P2D.promote_atomic(tmp, self.final, all_pre_promotion_checks_pass=True)

        self.assertTrue(out["final_output_replaced"])
        self.assertTrue(out["final_reparse_pass"])
        self.assertTrue(self.final.exists())
        self.assertEqual(sha256(self.final), temp_sha, "final hash must equal temp hash")
        self.assertEqual(out["final_output_sha256"], temp_sha)
        self.assertFalse(Path(tmp).exists(), "temporary file must no longer exist")

    def test_promotion_overwrites_previous(self) -> None:
        self.final.write_text(json.dumps({"marker": "OLD"}), encoding="utf-8")
        fd, tmp = tempfile.mkstemp(dir=str(self.dir), prefix=".tmp_", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"marker": "NEW", "images": [], "annotations": [], "categories": []}, fh)

        out = P2D.promote_atomic(tmp, self.final, all_pre_promotion_checks_pass=True)
        self.assertTrue(out["final_output_replaced"])
        self.assertEqual(json.loads(self.final.read_text())["marker"], "NEW")


if __name__ == "__main__":
    unittest.main(verbosity=2)
