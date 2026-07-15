"""Guardrail tests for Phase 2D.1A - Image Representation Protocol Decision.

Standard-library unittest only. No new dependencies beyond PyYAML (already a
script dependency). No DICOM / image data needed - this phase is decision-only.

Run:
    python -m unittest discover -s tests -p "test_phase2D1A_protocol_guardrails.py" -v
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "02D1A_image_representation_protocol.py"

YAML_PATH = REPO_ROOT / "configs" / "protocol" / "phase2D1_jpg_representation.yaml"
JSON_PATH = REPO_ROOT / "reports" / "phase2D1_image_representation_decision.json"
MD_PATH = REPO_ROOT / "reports" / "phase2D1_image_representation_decision.md"

PHASE2D_EVIDENCE = REPO_ROOT / "reports" / "phase2D_coco_master_validation.json"

# Banned modules that this decision-only phase must never import.
BANNED_MODULES = {"pydicom", "cv2", "PIL", "skimage", "SimpleITK", "gdcm", "imageio"}
# Banned attribute/callable usages (pixel access / image decode).
BANNED_ATTRS = {"pixel_array", "dcmread", "read_file", "imread", "imdecode"}


def _load_module():
    """Import the Phase 2D.1A script by path (its name is not an identifier)."""
    spec = importlib.util.spec_from_file_location("phase2d1a", SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"cannot load {SCRIPT_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase2d1a"] = mod
    spec.loader.exec_module(mod)
    return mod


MOD = _load_module()


def load_yaml_strict():
    import yaml

    with YAML_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_json():
    with JSON_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
class TestOutputsExist(unittest.TestCase):
    def test_yaml_exists_and_strict_loads(self):
        self.assertTrue(YAML_PATH.is_file(), f"missing {YAML_PATH}")
        doc = load_yaml_strict()
        self.assertIsInstance(doc, dict)

    def test_json_exists_and_parses(self):
        self.assertTrue(JSON_PATH.is_file(), f"missing {JSON_PATH}")
        self.assertIsInstance(load_json(), dict)

    def test_md_exists(self):
        self.assertTrue(MD_PATH.is_file(), f"missing {MD_PATH}")
        self.assertTrue(MD_PATH.read_text(encoding="utf-8").strip())


class TestRequiredSectionsAndKeys(unittest.TestCase):
    def setUp(self):
        self.doc = load_yaml_strict()

    def test_required_sections_present(self):
        for key in MOD.REQUIRED_POLICY_ITEMS:
            self.assertIn(key, self.doc, f"missing policy section: {key}")

    def test_metadata_keys(self):
        meta = self.doc["protocol_metadata"]
        self.assertEqual(meta["phase_id"], "2D.1A")
        self.assertEqual(meta["protocol_version"], "1.0.0")
        self.assertEqual(meta["status"], "decision_locked_pilot_pending")
        self.assertEqual(meta["seed"], 2026)
        self.assertEqual(meta["gpt_review_status"], "pending")

    def test_jpeg_encoding_keys(self):
        j = self.doc["jpeg_encoding"]
        self.assertIn("quality_candidates", j)
        self.assertIn("final_quality", j)
        self.assertIn("final_quality_status", j)


class TestLockedCounts(unittest.TestCase):
    def test_spec_locked_counts_match_module(self):
        spec = MOD.build_protocol_spec()
        self.assertEqual(spec["locked_input_counts"], MOD.LOCKED_COUNTS)
        self.assertEqual(
            MOD.LOCKED_COUNTS,
            {
                "images": 4894,
                "abnormal_images": 4394,
                "no_finding_images": 500,
                "annotations": 36096,
                "categories": 14,
                "no_finding_annotations": 0,
            },
        )

    def test_locked_counts_match_phase2D_evidence(self):
        if not PHASE2D_EVIDENCE.is_file():
            self.skipTest("phase2D evidence not present in this checkout")
        counts = json.loads(PHASE2D_EVIDENCE.read_text(encoding="utf-8"))["counts"]
        for key, expected in MOD.LOCKED_COUNTS.items():
            self.assertEqual(counts.get(key), expected, f"{key} mismatch vs Phase 2D")

    def test_json_report_locked_counts(self):
        rep = load_json()
        self.assertEqual(rep["locked_counts"], MOD.LOCKED_COUNTS)


class TestQualityAndThresholds(unittest.TestCase):
    def setUp(self):
        self.doc = load_yaml_strict()
        self.rep = load_json()

    def test_quality_candidates_equal_95_100(self):
        self.assertEqual(self.doc["jpeg_encoding"]["quality_candidates"], [95, 100])

    def test_final_quality_is_null(self):
        self.assertIsNone(self.doc["jpeg_encoding"]["final_quality"])

    def test_final_quality_status_pending_pilot(self):
        self.assertEqual(
            self.doc["jpeg_encoding"]["final_quality_status"], "pending_phase2D1B_pilot"
        )

    def test_report_flags_quality_pending(self):
        self.assertTrue(self.rep["final_jpeg_quality_is_pending"])

    def test_no_numeric_fidelity_thresholds_locked(self):
        for v in self.doc["thresholds_not_locked"].values():
            # note string is allowed; all numeric thresholds must be null
            self.assertNotIsInstance(v, (int, float))


class TestForbiddenTransforms(unittest.TestCase):
    def setUp(self):
        self.doc = load_yaml_strict()

    def test_direct_per_image_min_max_forbidden(self):
        self.assertEqual(
            self.doc["voi_windowing_policy"]["direct_observed_per_image_min_max"],
            "forbidden",
        )

    def test_percentile_clipping_forbidden(self):
        self.assertEqual(
            self.doc["voi_windowing_policy"]["automatic_percentile_clipping"],
            "forbidden",
        )

    def test_no_geometry_transforms(self):
        g = self.doc["geometry_bbox_policy"]
        for k in ("resize", "crop", "rotation", "flip", "transpose"):
            self.assertFalse(g[k], f"{k} must be false")
        self.assertTrue(g["preserve_width_and_height"])

    def test_bbox_scaling_not_validated(self):
        self.assertFalse(self.doc["geometry_bbox_policy"]["bbox_scaling_validated"])


class TestReadinessAndForbiddenFlags(unittest.TestCase):
    def setUp(self):
        self.doc = load_yaml_strict()
        self.rep = load_json()

    def test_all_readiness_flags_false(self):
        for k, v in self.doc["readiness_flags"].items():
            self.assertFalse(v, f"readiness flag {k} must be false")
        for k, v in self.rep["readiness_flags"].items():
            self.assertFalse(v, f"readiness flag {k} must be false in JSON")

    def test_all_forbidden_actions_false(self):
        for k, v in self.doc["forbidden_actions"].items():
            self.assertFalse(v, f"forbidden action {k} must be false")
        for k, v in self.rep["forbidden_actions"].items():
            self.assertFalse(v, f"forbidden action {k} must be false in JSON")


class TestPhaseStatusAndDoD(unittest.TestCase):
    def setUp(self):
        self.rep = load_json()

    def test_phase_status_not_pass(self):
        self.assertNotEqual(self.rep["phase_status"], "PASS")
        self.assertEqual(self.rep["phase_status"], "OPEN_REVIEW_REQUIRED")

    def test_gpt_review_pending(self):
        self.assertEqual(self.rep["gpt_review_status"], "pending")

    def test_dod_pass_candidate_true(self):
        self.assertTrue(self.rep["dod_pass_candidate"])
        self.assertEqual(self.rep["hard_errors"], [])

    def test_policy_coverage_20_of_20(self):
        self.assertEqual(self.rep["required_policy_items_total"], 20)
        self.assertEqual(self.rep["required_policy_items_documented"], 20)


class TestCrossOutputConsistency(unittest.TestCase):
    def test_single_source_of_truth_and_zero_drift(self):
        spec = MOD.build_protocol_spec()
        yaml_text = MOD.render_yaml_text(spec)
        report = MOD.build_report(spec, {"per_source": {}, "hard_errors": [], "warnings": []})
        json_text = MOD.render_json_text(report)
        md_text = MOD.render_markdown(spec, report)
        drift = MOD.compute_drift(spec, yaml_text, json_text, md_text)
        self.assertEqual(drift["cross_output_drift_count"], 0)
        self.assertTrue(drift["yaml_reload_matches"])
        self.assertTrue(drift["json_reload_matches"])
        self.assertTrue(drift["markdown_embeds_fingerprint"])

    def test_written_json_reports_zero_drift(self):
        rep = load_json()
        self.assertEqual(rep["cross_output_consistency"]["cross_output_drift_count"], 0)

    def test_written_yaml_matches_spec_fingerprint(self):
        spec = MOD.build_protocol_spec()
        doc = load_yaml_strict()
        self.assertEqual(MOD.protocol_sha256(doc), MOD.protocol_sha256(spec))


class TestNoBannedImportsOrUsage(unittest.TestCase):
    """The Phase 2D.1A script must not import or use any imaging library."""

    def setUp(self):
        self.source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.tree = ast.parse(self.source)

    def test_no_banned_imports(self):
        imported = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    imported.add(a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported.add(node.module.split(".")[0])
        offenders = imported & BANNED_MODULES
        self.assertEqual(offenders, set(), f"banned imports present: {offenders}")

    def test_no_pixel_or_decode_attribute_usage(self):
        # Only inspect *code* attribute accesses / calls, never string literals,
        # so that documenting field names inside PROTOCOL_SPEC is not flagged.
        used_attrs = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Attribute):
                used_attrs.add(node.attr)
            elif isinstance(node, ast.Name):
                used_attrs.add(node.id)
        offenders = used_attrs & BANNED_ATTRS
        self.assertEqual(offenders, set(), f"banned usage present: {offenders}")


class TestAtomicOutputPreservation(unittest.TestCase):
    """A simulated validation failure must not replace prior outputs and must
    clean up all temporary files."""

    def test_preserves_previous_and_cleans_temps_on_failure(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            target = base / "sub" / "out.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('{"previous": true}', encoding="utf-8")
            before = target.read_text(encoding="utf-8")

            ok, errors = MOD.atomic_write_all(
                {target: '{"new": true}'}, simulate_validation_failure=True
            )
            self.assertFalse(ok)
            self.assertTrue(errors)
            # Previous output untouched.
            self.assertEqual(target.read_text(encoding="utf-8"), before)
            # No temporary files left behind.
            leftovers = [p for p in target.parent.iterdir() if p.name != target.name]
            self.assertEqual(leftovers, [], f"temporary files not cleaned: {leftovers}")

    def test_successful_promotion_replaces_and_cleans(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            target = base / "out.json"
            ok, errors = MOD.atomic_write_all({target: '{"new": true}'})
            self.assertTrue(ok, f"unexpected errors: {errors}")
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"new": True})
            leftovers = [p for p in base.iterdir() if p.name != target.name]
            self.assertEqual(leftovers, [], f"temporary files not cleaned: {leftovers}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
