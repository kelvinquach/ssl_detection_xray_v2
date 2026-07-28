#!/usr/bin/env python3
"""Phase 2D.1B-Pilot guardrail + synthetic-conformance tests.

Design principles (Section 31 / 33 of the spec):
* Tests use synthetic fixtures and temporary directories only. They never
  depend on the 4,894 real DICOM files, never decode a real DICOM, and never
  run full conversion.
* Synthetic expected outputs are defined INDEPENDENTLY (hand-computed arrays,
  explicit locked formulas re-implemented locally, or fixed constants). A test
  never computes its expected value by calling the same production helper it is
  testing. Self-referential tests are forbidden.
* AST/source guardrails inspect real actions (imports, forbidden write paths,
  force=True, geometric transforms), not merely the appearance of a banned word
  inside a docstring or string literal.

Run (Windows CMD):
    python -m unittest discover -s tests -p "test_phase2D1B_pilot_guardrails.py" -v
"""
from __future__ import annotations

import ast
import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import dicom_jpg_protocol as P  # noqa: E402

SCRIPT_PATH = REPO_ROOT / "scripts" / "02D1B_pilot_dicom_to_jpg.py"
UTILS_PATH = REPO_ROOT / "src" / "utils" / "dicom_jpg_protocol.py"
PROTOCOL_YAML = REPO_ROOT / "configs" / "protocol" / "phase2D1_jpg_representation.yaml"


def load_script_module():
    """Import the digit-leading orchestrator via importlib (no pydicom needed).

    The module is registered in ``sys.modules`` before execution so that
    ``@dataclass`` on ``Args`` can resolve its (string) annotations under
    ``from __future__ import annotations``.
    """
    spec = importlib.util.spec_from_file_location("pilot_orchestrator", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Independent re-implementations used ONLY to derive expected values.          #
# These deliberately do NOT call any function in P.                            #
# --------------------------------------------------------------------------- #
def independent_linear_fraction(x, c, w):
    cc = c - 0.5
    ww = w - 1.0
    frac = (x - cc) / ww + 0.5
    return min(1.0, max(0.0, frac))


def independent_rint_uint8(frac):
    # round half to even, like numpy.rint
    import decimal
    val = frac * 255.0
    return int(decimal.Decimal(val).quantize(0, rounding=decimal.ROUND_HALF_EVEN))


# =========================================================================== #
class TestProtocolPreflight(unittest.TestCase):
    def setUp(self):
        import yaml
        with open(PROTOCOL_YAML, encoding="utf-8") as fh:
            self.protocol = yaml.safe_load(fh)

    def test_nested_version_path(self):
        self.assertEqual(P.resolved_field_path("protocol_version"),
                         "protocol_metadata.protocol_version")
        self.assertEqual(P.resolve_field(self.protocol, "protocol_version"), "1.0.0")

    def test_missing_nested_path_is_schema_mismatch(self):
        broken = {"protocol_metadata": {}}  # protocol_version absent
        with self.assertRaises(P.ProtocolSchemaMismatch):
            P.resolve_field(broken, "protocol_version")

    def test_similar_field_elsewhere_not_used(self):
        # A top-level protocol_version must NOT be accepted; only the nested path.
        sneaky = {"protocol_version": "1.0.0", "protocol_metadata": {}}
        with self.assertRaises(P.ProtocolSchemaMismatch):
            P.resolve_field(sneaky, "protocol_version")

    def test_fingerprint_matches_locked(self):
        ev = P.validate_protocol(self.protocol)
        self.assertTrue(ev["protocol_sha256_match"])
        self.assertEqual(ev["protocol_sha256"], P.EXPECTED_PROTOCOL_SHA256)

    def test_quality_candidates_and_null_final_quality(self):
        self.assertEqual(list(P.resolve_field(self.protocol, "quality_candidates")),
                         [95, 100])
        self.assertIsNone(P.resolve_field(self.protocol, "final_quality"))

    def test_drift_when_version_changed(self):
        mutated = {**self.protocol}
        mutated["protocol_metadata"] = {**self.protocol["protocol_metadata"],
                                        "protocol_version": "1.0.1"}
        with self.assertRaises(P.ProtocolDriftError):
            P.validate_protocol(mutated)

    def test_drift_when_fingerprint_changed(self):
        mutated = {**self.protocol, "extra_unexpected_key": True}
        with self.assertRaises(P.ProtocolDriftError):
            P.validate_protocol(mutated)


# =========================================================================== #
class TestDicomRootResolution(unittest.TestCase):
    def test_env_only(self):
        r = P.resolve_dicom_root(None, str(REPO_ROOT))
        self.assertEqual(r.source, "env")

    def test_cli_only(self):
        r = P.resolve_dicom_root(str(REPO_ROOT), None)
        self.assertEqual(r.source, "cli")

    def test_cli_env_equal(self):
        r = P.resolve_dicom_root(str(REPO_ROOT), str(REPO_ROOT))
        self.assertEqual(r.source, "cli_and_env_equal")

    def test_cli_env_conflict_hard_fail(self):
        other = REPO_ROOT / "src"
        with self.assertRaises(P.UnsupportedInputError):
            P.resolve_dicom_root(str(REPO_ROOT), str(other))

    def test_neither_hard_fail(self):
        with self.assertRaises(P.UnsupportedInputError):
            P.resolve_dicom_root(None, None)

    def test_path_traversal_rejected(self):
        with self.assertRaises(P.UnsupportedInputError):
            P.safe_resolve_under_root(REPO_ROOT, "../secret.dicom")

    def test_absolute_rejected(self):
        with self.assertRaises(P.UnsupportedInputError):
            P.safe_resolve_under_root(REPO_ROOT, "/etc/passwd")

    def test_relative_join_ok(self):
        p = P.safe_resolve_under_root(REPO_ROOT, "train/abc.dicom")
        self.assertTrue(str(p).endswith(os.path.join("train", "abc.dicom")))


# =========================================================================== #
class TestTheoreticalRanges(unittest.TestCase):
    def test_unsigned(self):
        self.assertEqual(P.theoretical_stored_range(12, 0), (0, 4095))
        self.assertEqual(P.theoretical_stored_range(14, 0), (0, 16383))

    def test_signed(self):
        self.assertEqual(P.theoretical_stored_range(12, 1), (-2048, 2047))
        self.assertEqual(P.theoretical_stored_range(16, 1), (-32768, 32767))


# =========================================================================== #
class TestModalityBranch(unittest.TestCase):
    def test_modality_lut_priority(self):
        self.assertEqual(P.modality_branch_name(True, True, True), "modality_lut")

    def test_rescale_both_present(self):
        self.assertEqual(P.modality_branch_name(False, True, True), "rescale")

    def test_identity(self):
        self.assertEqual(P.modality_branch_name(False, False, False), "identity")

    def test_incomplete_rescale_hard_fail(self):
        with self.assertRaises(P.ModalityBranchError):
            P.modality_branch_name(False, True, False)

    def test_negative_slope_endpoint_sorting(self):
        stored = np.array([[0, 50], [100, 25]], dtype=np.int64)
        values, low, high = P.apply_rescale(stored, -1.0, 100.0, 0, 100)
        # Independent expected: v = -x + 100; endpoints 100 and 0 -> sorted (0,100)
        self.assertEqual((low, high), (0.0, 100.0))
        self.assertTrue(np.allclose(values, np.array([[100.0, 50.0], [0.0, 75.0]])))


# =========================================================================== #
class TestWindowing(unittest.TestCase):
    def test_linear_independent(self):
        vals = np.array([800.0, 1000.0, 1200.0])
        out = P.window_linear(vals, center=1000.0, width=400.0)
        exp = np.array([independent_linear_fraction(v, 1000.0, 400.0) for v in vals])
        self.assertTrue(np.allclose(out, exp))
        # edges clip to 0 / 1
        self.assertAlmostEqual(out[0], 0.0)
        self.assertAlmostEqual(out[2], 1.0)

    def test_linear_exact_independent(self):
        vals = np.array([900.0, 1000.0, 1100.0])
        out = P.window_linear_exact(vals, 1000.0, 200.0)
        exp = np.clip((vals - 1000.0) / 200.0 + 0.5, 0.0, 1.0)
        self.assertTrue(np.allclose(out, exp))

    def test_sigmoid_independent(self):
        vals = np.array([1000.0])
        out = P.window_sigmoid(vals, 1000.0, 200.0)
        # at center, sigmoid == 0.5 exactly
        self.assertAlmostEqual(out[0], 0.5)

    def test_invalid_width_blocks(self):
        with self.assertRaises(P.ProtocolGapError):
            P.window_linear(np.array([1.0]), 10.0, 0.0)

    def test_fallback_degenerate_hard_fail(self):
        with self.assertRaises(P.DegenerateRangeError):
            P.fallback_modality_fraction(np.array([1.0]), 5.0, 5.0)


# =========================================================================== #
class TestPresentationPolarity(unittest.TestCase):
    def test_supported_table(self):
        cases = {
            ("MONOCHROME1", None): 1,
            ("MONOCHROME2", None): 0,
            ("MONOCHROME1", "INVERSE"): 1,
            ("MONOCHROME2", "IDENTITY"): 0,
        }
        for (pi, shape), expected_inv in cases.items():
            dec = P.presentation_polarity_decision(pi, shape, False)
            self.assertFalse(dec.metadata_conflict, (pi, shape))
            self.assertFalse(dec.protocol_gap, (pi, shape))
            self.assertEqual(dec.inversion_count, expected_inv, (pi, shape))

    def test_conflict_mono1_identity(self):
        dec = P.presentation_polarity_decision("MONOCHROME1", "IDENTITY", False)
        self.assertTrue(dec.metadata_conflict)
        self.assertTrue(dec.protocol_gap)
        self.assertIsNone(dec.inversion_count)

    def test_conflict_mono2_inverse(self):
        dec = P.presentation_polarity_decision("MONOCHROME2", "INVERSE", False)
        self.assertTrue(dec.metadata_conflict)
        self.assertTrue(dec.protocol_gap)
        self.assertIsNone(dec.inversion_count)

    def test_presentation_lut_sequence_is_gap(self):
        dec = P.presentation_polarity_decision("MONOCHROME2", None, True)
        self.assertTrue(dec.protocol_gap)
        self.assertIsNone(dec.inversion_count)

    def test_apply_presentation_invert_once(self):
        frac = np.array([0.25, 0.75])
        self.assertTrue(np.allclose(P.apply_presentation(frac, 1),
                                    np.array([0.75, 0.25])))
        self.assertTrue(np.allclose(P.apply_presentation(frac, 0), frac))

    def test_apply_presentation_rejects_double(self):
        with self.assertRaises(ValueError):
            P.apply_presentation(np.array([0.1]), 2)


# =========================================================================== #
class TestPadding(unittest.TestCase):
    def test_single_value_mask(self):
        stored = np.array([[0, 5], [5, 9]])
        mask = P.build_padding_mask(stored, 5, None)
        self.assertTrue(np.array_equal(mask, np.array([[False, True], [True, False]])))

    def test_inclusive_range_mask(self):
        stored = np.array([[0, 3], [7, 11]])
        mask = P.build_padding_mask(stored, 2, 8)  # low=2 high=8 inclusive
        self.assertTrue(np.array_equal(mask, np.array([[False, True], [True, False]])))

    def test_range_limit_without_value_hard_fail(self):
        with self.assertRaises(P.PaddingMetadataError):
            P.build_padding_mask(np.array([1]), None, 8)

    def test_padding_reapplied_zero_after_inversion(self):
        # A MONOCHROME1-style invert must not leave padding white.
        frac = np.array([[0.0, 1.0]])
        mask = np.array([[True, False]])
        inverted = P.apply_presentation(frac, 1)  # -> [[1.0, 0.0]]
        out = P.fraction_to_uint8(inverted, mask)
        self.assertEqual(out[0, 0], 0)  # padding forced to 0, not 255


# =========================================================================== #
class TestUint8(unittest.TestCase):
    def test_rint_half_even(self):
        out = P.fraction_to_uint8(np.array([0.0, 0.5, 1.0]))
        # independent: 0 -> 0 ; 0.5*255=127.5 -> round half even -> 128 ; 1 -> 255
        self.assertEqual(list(out), [0, 128, 255])
        self.assertEqual(independent_rint_uint8(0.5), 128)

    def test_nan_inf_hard_fail(self):
        with self.assertRaises(P.NonFiniteError):
            P.fraction_to_uint8(np.array([np.nan]))
        with self.assertRaises(P.NonFiniteError):
            P.fraction_to_uint8(np.array([np.inf]))

    def test_dtype_and_shape(self):
        out = P.fraction_to_uint8(np.zeros((3, 4)))
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape, (3, 4))


# =========================================================================== #
class TestMetrics(unittest.TestCase):
    def test_identical_psnr_none(self):
        a = np.zeros((4, 4), dtype=np.uint8)
        m = P.whole_image_error_metrics(a, a)
        self.assertEqual(m["mae"], 0.0)
        self.assertIsNone(m["psnr_db"])
        self.assertTrue(m["psnr_is_infinite"])

    def test_known_values_independent(self):
        ref = np.zeros((2, 2), dtype=np.uint8)
        tgt = np.array([[0, 0], [0, 255]], dtype=np.uint8)
        m = P.whole_image_error_metrics(ref, tgt)
        self.assertAlmostEqual(m["mae"], 255.0 / 4.0)
        self.assertAlmostEqual(m["rmse"], math.sqrt((255.0 ** 2) / 4.0))
        self.assertIsNotNone(m["psnr_db"])
        self.assertEqual(m["max_absolute_error"], 255.0)

    def test_roi_extraction_coords(self):
        self.assertEqual(P.roi_extraction_coords(1.2, 3.8, 5.1, 9.9), (1, 3, 6, 10))

    def test_small_roi_win_size(self):
        self.assertIsNone(P.largest_odd_win_size(2, 10))
        self.assertEqual(P.largest_odd_win_size(4, 10), 3)
        self.assertEqual(P.largest_odd_win_size(7, 9), 7)

    def test_no_naninf_serialized(self):
        m = P.whole_image_error_metrics(np.zeros((2, 2), np.uint8),
                                        np.zeros((2, 2), np.uint8))
        import json
        json.dumps(m, allow_nan=False)  # must not raise


# =========================================================================== #
class TestTieBreakAndSelection(unittest.TestCase):
    def test_tie_break_deterministic(self):
        self.assertEqual(P.tie_break_rank("abc"), P.tie_break_rank("abc"))
        self.assertNotEqual(P.tie_break_rank("abc"), P.tie_break_rank("abd"))
        # never uses builtin hash(): stable across processes -> known prefix
        self.assertEqual(len(P.tie_break_rank("abc")), 64)

    def test_selection_deterministic_and_covers(self):
        mod = load_script_module()
        # synthetic: 5 features, images each covering some; 2 negatives.
        image_features = {
            "i1": {"class=0", "extremum=dim_min_w"},
            "i2": {"class=1", "scope=no_finding"},
            "i3": {"class=2"},
            "i4": {"scope=no_finding"},
            "i5": {"class=0", "class=1", "class=2"},
        }
        all_features = set().union(*image_features.values())
        negatives = {"i2", "i4"}
        r1 = mod.deterministic_selection(dict(image_features), ["i1"], set(negatives), set(all_features))
        r2 = mod.deterministic_selection(dict(image_features), ["i1"], set(negatives), set(all_features))
        ids1 = [x["image_id"] for x in r1]
        ids2 = [x["image_id"] for x in r2]
        self.assertEqual(ids1, ids2)  # reproducible order
        covered = set().union(*(image_features[i] for i in ids1))
        self.assertTrue(all_features.issubset(covered))


# =========================================================================== #
class TestValidationStatusInvariants(unittest.TestCase):
    def setUp(self):
        self.mod = load_script_module()
        self.status = self.mod.baseline_validation_status()

    def test_readiness_flags_all_false(self):
        for k in ("jpg_training_representation_ready",
                  "coco_jpg_training_annotation_ready",
                  "mmdetection_dataset_loading_ready",
                  "empty_image_retention_ready",
                  "dataset_training_ready", "training_authorized"):
            self.assertIs(self.status[k], False, k)

    def test_never_pass_and_null_quality(self):
        self.assertNotEqual(self.status["phase_status"], "PASS")
        self.assertEqual(self.status["phase_status"], "OPEN_REVIEW_REQUIRED")
        self.assertIsNone(self.status["final_jpeg_quality"])
        self.assertFalse(self.status["full_conversion_authorized"])

    def test_positioning_flags(self):
        self.assertFalse(self.status["novel_algorithm_claimed"])
        self.assertFalse(self.status["full_dicom_standard_conformance_claimed"])
        self.assertFalse(self.status["clinical_validation_claimed"])
        self.assertFalse(self.status["downstream_superiority_evaluated"])
        self.assertEqual(self.status["controlled_downstream_ablation_status"],
                         "pending_mentor_confirmation")
        self.assertIsNone(self.status["controlled_downstream_ablation_required"])
        self.assertFalse(self.status["patient_space_orientation_independently_validated"])
        self.assertEqual(self.status["master_representation_channel_count"], 1)

    def test_decision_template_pending(self):
        dt = self.mod.DECISION_TEMPLATE_JSON
        self.assertIsNone(dt["final_jpeg_quality"])
        self.assertIsNone(dt["selected_candidate"])
        self.assertFalse(dt["full_conversion_authorized"])
        self.assertEqual(dt["decision_status"], "pending_gpt_and_researcher_review")


# =========================================================================== #
class TestForbiddenArtifactSnapshot(unittest.TestCase):
    def test_snapshot_absent_by_default(self):
        mod = load_script_module()
        snap = mod.snapshot_forbidden_artifacts()
        # In a clean repo none of the forbidden full-conversion artifacts exist.
        self.assertFalse(snap["preexisting_forbidden_artifact"], snap)


# =========================================================================== #
class TestSourceGuardrails(unittest.TestCase):
    """AST/source guardrails - inspect real actions, not docstring words."""

    def setUp(self):
        self.script_src = SCRIPT_PATH.read_text(encoding="utf-8")
        self.utils_src = UTILS_PATH.read_text(encoding="utf-8")
        self.script_ast = ast.parse(self.script_src)
        self.utils_ast = ast.parse(self.utils_src)

    def _imported_names(self, tree):
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    names.add(a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    names.add(node.module.split(".")[0])
        return names

    def test_no_training_framework_imports(self):
        forbidden = {"mmdet", "mmcv", "mmengine", "torch", "torchvision", "detectron2"}
        self.assertEqual(self._imported_names(self.script_ast) & forbidden, set())
        self.assertEqual(self._imported_names(self.utils_ast) & forbidden, set())

    def test_no_force_true_decode(self):
        # force=True would allow decoding malformed/unsupported inputs silently.
        for node in ast.walk(self.script_ast):
            if isinstance(node, ast.keyword) and node.arg == "force":
                self.assertFalse(
                    isinstance(node.value, ast.Constant) and node.value.value is True,
                    "force=True is forbidden",
                )

    def test_no_geometric_transform_calls(self):
        banned_attrs = {"resize", "rotate", "transpose", "thumbnail", "flip"}
        for node in ast.walk(self.script_ast):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, banned_attrs,
                                 f"geometric transform call {node.func.attr} forbidden")

    def test_no_write_to_full_jpg_train(self):
        # The full-conversion output dir may only appear as a forbidden-path
        # constant, never as a save/write target.
        self.assertIn("images_jpg", self.script_src)
        self.assertNotIn('images_jpg/train"', self.script_src.replace(
            'REPO_ROOT / "data" / "processed" / "images_jpg" / "train"', ""))

    def test_coco_master_jpg_never_created(self):
        # coco_master_jpg.json may only appear as a forbidden-artifact path,
        # never as a write/save target.
        occ = [ln for ln in self.script_src.splitlines() if "coco_master_jpg" in ln]
        self.assertTrue(occ)  # referenced (as a forbidden path / scope note)
        for ln in occ:
            for tok in ("atomic_write_text", "write_csv", ".save(", "json.dump", "open("):
                self.assertNotIn(tok, ln, "coco_master_jpg must never be a write target")
        # coco_master.json (the real master) is only ever opened read-only.
        for ln in self.script_src.splitlines():
            if "open(COCO_MASTER" in ln:
                self.assertIn('"r"', ln)

    def test_no_percentile_used_for_mapping_bounds(self):
        # np.percentile may appear ONLY in metrics (p95/p99), never in the
        # transform_pixels mapping function.
        func = next((n for n in ast.walk(self.script_ast)
                     if isinstance(n, ast.FunctionDef) and n.name == "transform_pixels"), None)
        self.assertIsNotNone(func)
        for node in ast.walk(func):
            if isinstance(node, ast.Attribute):
                self.assertNotEqual(node.attr, "percentile",
                                    "percentile clipping in mapping is forbidden")

    def test_synthetic_expected_values_are_independent(self):
        # Our own expected-value helper must not call into P.
        this_src = Path(__file__).read_text(encoding="utf-8")
        tree = ast.parse(this_src)
        for fn_name in ("independent_linear_fraction", "independent_rint_uint8"):
            fn = next(n for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef) and n.name == fn_name)
            for node in ast.walk(fn):
                if isinstance(node, ast.Attribute):
                    self.assertNotEqual(getattr(node.value, "id", None), "P",
                                        "independent helper must not call production P.*")


# =========================================================================== #
class TestAccidentalFullConversionGuards(unittest.TestCase):
    def test_locked_counts_constant(self):
        self.assertEqual(P.LOCKED_INPUT_COUNTS["images"], 4894)
        self.assertEqual(P.MIN_PILOT_IMAGES, 64)
        self.assertEqual(P.MIN_PILOT_NO_FINDING, 16)
        self.assertEqual(P.MAX_PILOT_IMAGES, 256)

    def test_selection_scope_cap(self):
        mod = load_script_module()
        # 300 images each with a UNIQUE feature => cannot stop before cap.
        feats = {f"i{n}": {f"f{n}"} for n in range(300)}
        allf = set().union(*feats.values())
        with self.assertRaises(P.PilotScopeExplosionError):
            mod.deterministic_selection(dict(feats), [], set(), set(allf))


try:
    import skimage  # noqa: F401
    HAVE_SKIMAGE = True
except Exception:
    HAVE_SKIMAGE = False

try:
    from PIL import Image  # noqa: F401
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False


# =========================================================================== #
# Blocker 7: window classification blocks present-but-invalid windows           #
# =========================================================================== #
class TestWindowClassification(unittest.TestCase):
    def test_absent(self):
        d = P.classify_window("", "")
        self.assertEqual(d.state, "absent")

    def test_valid_single(self):
        d = P.classify_window("1000", "400")
        self.assertEqual(d.state, "valid")
        self.assertEqual((d.center, d.width), (1000.0, 400.0))

    def test_valid_multi_uses_index0(self):
        d = P.classify_window("1000;2000", "400;800")
        self.assertEqual(d.state, "valid")
        self.assertEqual(d.center, 1000.0)
        self.assertEqual(len(d.centers), 2)

    def test_cardinality_mismatch_invalid(self):
        d = P.classify_window("1000;2000", "400")
        self.assertEqual(d.state, "invalid")
        self.assertEqual(d.reason, "cardinality_mismatch")

    def test_non_numeric_invalid(self):
        d = P.classify_window("abc", "400")
        self.assertEqual(d.state, "invalid")
        self.assertEqual(d.reason, "non_numeric")

    def test_nonpositive_width_invalid(self):
        d = P.classify_window("1000", "0")
        self.assertEqual(d.state, "invalid")
        self.assertEqual(d.reason, "invalid_width")

    def test_require_valid_window_blocks_invalid(self):
        d = P.classify_window("1000;2000", "400")
        with self.assertRaises(P.ProtocolGapError):
            P.require_valid_window(d)

    def test_require_valid_window_returns_center_width(self):
        d = P.classify_window("1000", "400")
        self.assertEqual(P.require_valid_window(d), (1000.0, 400.0))


# =========================================================================== #
# Blocker 8: Modality LUT output bounds come from LUT DATA, not index range      #
# =========================================================================== #
class TestModalityLutBounds(unittest.TestCase):
    def test_bounds_from_lut_data(self):
        # LUT DATA values (output intensities) are 100..355; the input index
        # range (descriptor) would be 0..255 - must NOT be used.
        lut_data = list(range(100, 356))
        low, high = P.modality_lut_output_bounds(lut_data)
        self.assertEqual((low, high), (100.0, 355.0))
        self.assertNotEqual((low, high), (0.0, 255.0))

    def test_empty_lut_raises(self):
        with self.assertRaises(P.ModalityBranchError):
            P.modality_lut_output_bounds([])

    def test_degenerate_lut_raises(self):
        with self.assertRaises(P.DegenerateRangeError):
            P.modality_lut_output_bounds([50, 50, 50])


# =========================================================================== #
# Blocker 6: presentation gap must raise; never None -> inversion=0              #
# =========================================================================== #
class TestPresentationGapEnforcement(unittest.TestCase):
    def test_supported_returns_int(self):
        dec = P.presentation_polarity_decision("MONOCHROME1", None, False)
        self.assertEqual(P.require_inversion_count(dec), 1)

    def test_conflict_raises_not_zero(self):
        dec = P.presentation_polarity_decision("MONOCHROME2", "INVERSE", False)
        with self.assertRaises(P.ProtocolGapError):
            P.require_inversion_count(dec)

    def test_lut_sequence_raises(self):
        dec = P.presentation_polarity_decision("MONOCHROME2", None, True)
        with self.assertRaises(P.ProtocolGapError):
            P.require_inversion_count(dec)


# =========================================================================== #
# Blocker 5: JPEG2000 decoder backend enforcement                               #
# =========================================================================== #
class TestJpeg2000Backend(unittest.TestCase):
    def test_is_jpeg2000(self):
        self.assertTrue(P.is_jpeg2000("1.2.840.10008.1.2.4.90"))
        self.assertTrue(P.is_jpeg2000("1.2.840.10008.1.2.4.91"))
        self.assertFalse(P.is_jpeg2000("1.2.840.10008.1.2.1"))

    def test_unknown_backend_raises(self):
        with self.assertRaises(P.UnsupportedInputError):
            P.ensure_jpeg2000_backend("not_a_backend")

    def test_unavailable_backend_hard_fail(self):
        # 'gdcm' is not installed in this environment -> must hard fail (no
        # silent fallback). If it happens to be installed, ensure() passes.
        if P.jpeg2000_backend_available("gdcm"):
            P.ensure_jpeg2000_backend("gdcm")  # should not raise
        else:
            with self.assertRaises(P.UnsupportedInputError):
                P.ensure_jpeg2000_backend("gdcm")


# =========================================================================== #
# Blocker 3/4: SSIM (whole + ROI small-window)                                  #
# =========================================================================== #
@unittest.skipUnless(HAVE_SKIMAGE, "skimage not installed")
class TestSsim(unittest.TestCase):
    def test_whole_image_ssim_identical(self):
        a = np.zeros((16, 16), dtype=np.uint8)
        out = P.whole_image_ssim(a, a)
        self.assertAlmostEqual(out["ssim"], 1.0, places=6)
        self.assertEqual(out["data_range"], 255)

    def test_roi_ssim_small_window(self):
        roi = np.zeros((4, 10), dtype=np.uint8)
        out = P.roi_ssim(roi, roi)
        self.assertTrue(out["evaluable"])
        self.assertEqual(out["win_size"], 3)  # largest odd <= min side (4)->3

    def test_roi_ssim_too_small(self):
        roi = np.zeros((2, 10), dtype=np.uint8)
        out = P.roi_ssim(roi, roi)
        self.assertFalse(out["evaluable"])
        self.assertIsNone(out["ssim"])
        self.assertEqual(out["reason"], "roi_too_small_for_ssim")


# =========================================================================== #
# Blocker 4: ROI summaries (micro/image-macro/class-macro/pairwise/worst)        #
# =========================================================================== #
class TestRoiSummaries(unittest.TestCase):
    def setUp(self):
        # Two images, two classes, both qualities; ROI_MAE values chosen so the
        # macro means differ from the micro mean (dominant image/class).
        self.rows = [
            {"annotation_id": "1", "canonical_ann_id": "1", "image_id": "a",
             "canonical_class_id": 0, "jpeg_quality": 95, "ROI_MAE": 10.0, "ROI_SSIM": 0.9, "ROI_PSNR": 40.0},
            {"annotation_id": "2", "canonical_ann_id": "2", "image_id": "a",
             "canonical_class_id": 0, "jpeg_quality": 95, "ROI_MAE": 30.0, "ROI_SSIM": 0.8, "ROI_PSNR": 35.0},
            {"annotation_id": "3", "canonical_ann_id": "3", "image_id": "b",
             "canonical_class_id": 1, "jpeg_quality": 95, "ROI_MAE": 2.0, "ROI_SSIM": 0.99, "ROI_PSNR": 50.0},
            {"annotation_id": "1", "canonical_ann_id": "1", "image_id": "a",
             "canonical_class_id": 0, "jpeg_quality": 100, "ROI_MAE": 5.0, "ROI_SSIM": 0.95, "ROI_PSNR": 45.0},
            {"annotation_id": "2", "canonical_ann_id": "2", "image_id": "a",
             "canonical_class_id": 0, "jpeg_quality": 100, "ROI_MAE": 15.0, "ROI_SSIM": 0.9, "ROI_PSNR": 42.0},
            {"annotation_id": "3", "canonical_ann_id": "3", "image_id": "b",
             "canonical_class_id": 1, "jpeg_quality": 100, "ROI_MAE": 1.0, "ROI_SSIM": 0.995, "ROI_PSNR": 55.0},
        ]

    def test_micro_vs_macro_differ(self):
        s = P.summarize_roi_metrics(self.rows, "ROI_MAE")[95]
        # micro = mean(10,30,2)=14; image-macro = mean(mean(10,30)=20, mean(2)=2)=11
        self.assertAlmostEqual(s["annotation_micro_mean"], 14.0)
        self.assertAlmostEqual(s["image_macro_mean"], 11.0)
        # class-macro = mean(class0=mean(10,30)=20, class1=2)=11
        self.assertAlmostEqual(s["class_macro_mean"], 11.0)

    def test_worst_case(self):
        w = P.worst_roi_cases(self.rows, "ROI_MAE", top=1, largest_is_worst=True)
        self.assertEqual(w[95][0]["ROI_MAE"], 30.0)

    def test_pairwise_delta(self):
        pw = {r["canonical_ann_id"]: r for r in P.pairwise_q100_minus_q95(self.rows)}
        # ann 1: q100(5) - q95(10) = -5
        self.assertAlmostEqual(pw["1"]["ROI_MAE_q100_minus_q95"], -5.0)


# =========================================================================== #
# Blocker 10: final coverage validation                                         #
# =========================================================================== #
class TestCoverageValidation(unittest.TestCase):
    def _full(self):
        classes = {f"class={i}" for i in range(14)}
        extrema = {f"extremum=dim_min_w", "extremum=bbox_rel_min"}
        strata = {"PhotometricInterpretation=MONOCHROME2"}
        return classes | extrema | strata

    def test_complete_passes(self):
        feats = self._full()
        res = P.validate_full_coverage(feats, feats)
        self.assertTrue(res["fully_covered"])
        self.assertEqual(res["classes_covered"], 14)

    def test_missing_class_raises(self):
        feats = self._full()
        covered = feats - {"class=7"}
        with self.assertRaises(P.UnsupportedInputError):
            P.validate_full_coverage(covered, feats)

    def test_missing_extremum_raises(self):
        feats = self._full()
        covered = feats - {"extremum=bbox_rel_min"}
        with self.assertRaises(P.UnsupportedInputError):
            P.validate_full_coverage(covered, feats)


# =========================================================================== #
# Blockers 1/2/11/12/13: evidence + mapping + atomic promotion (synthetic)       #
# --------------------------------------------------------------------------- #
# Builds a fully synthetic ctx (NO real DICOM, NO pydicom, NO skimage) and       #
# exercises write_all_evidence + validate_staging + promote_atomic end-to-end.   #
# =========================================================================== #
@unittest.skipUnless(HAVE_PIL, "Pillow required to create dummy image files")
class TestEvidenceAndPromotion(unittest.TestCase):
    def _header(self, oid):
        return {
            "image_id": oid, "SOPClassUID": "1.2.840.10008.5.1.4.1.1.1",
            "Modality": "DX", "Rows": 8, "Columns": 8,
            "PhotometricInterpretation": "MONOCHROME2",
            "TransferSyntaxUID": "1.2.840.10008.1.2.1",
            "BitsAllocated": 16, "BitsStored": 12, "HighBit": 11,
            "PixelRepresentation": 0, "SamplesPerPixel": 1,
            "NumberOfFrames_effective": 1, "VOILUTFunction": "ABSENT",
            "WindowCenter_all": "1000", "WindowWidth_all": "400",
            "PresentationLUTShape": "ABSENT", "presentation_lut_sequence_present": False,
        }

    def _transform_record(self, oid):
        return {
            "modality_branch": "rescale", "voi_branch": "windowing",
            "presentation_inversion_applied": False, "presentation_inversion_count": 0,
            "padding_present": (oid == "a"), "padding_pixel_count": (1 if oid == "a" else 0),
            "uint8_zero_fraction": 0.1, "uint8_255_fraction": 0.2,
            "source_dicom_sha256": "d" * 64, "pre_jpeg_uint8_sha256": "p" * 64,
            "reference_png_byte_sha256": "r" * 64,
            "reference_png_decoded_pixel_sha256": "x" * 64,
            "reference_png_exact_pixel_match": True, "decoder_backend": "pydicom_native",
            "rows": 8, "columns": 8,
        }

    def _build_ctx(self, staging):
        oids = ["a", "b"]
        id_to_coco = {o: {"id": i + 1, "canonical_image_id": i, "width": 8, "height": 8,
                          "scope_label": "abnormal", "is_negative": False,
                          "file_name": f"train/{o}.dicom"}
                      for i, o in enumerate(oids)}
        records = [{"image_id": o, "selection_order": i + 1,
                    "selected_for_features": ["mandatory_extremum"],
                    "newly_covered_feature_count": 1, "tie_break_rank": P.tie_break_rank(o)}
                   for i, o in enumerate(oids)]
        image_features = {"a": {"class=0", "extremum=dim_min_w",
                                "PhotometricInterpretation=MONOCHROME2"},
                          "b": {"class=1"}}
        all_features = set().union(*image_features.values())
        roi_rows = []
        fidelity_rows = []
        fidelity_by_key = {}
        geometry_rows = []
        for o in oids:
            for q in (95, 100):
                fr = {"original_image_id": o, "coco_image_id": id_to_coco[o]["id"],
                      "jpeg_quality": q, "decoder_backend": "pydicom_native",
                      "mae": 3.0, "rmse": 4.0, "psnr_db": 36.0,
                      "psnr_is_infinite": False, "ssim": 0.98, "max_absolute_error": 9.0,
                      "p95_absolute_error": 5.0, "p99_absolute_error": 7.0,
                      "percentile_method": "linear", "jpg_file_size_bytes": 100,
                      "pre_jpeg_uint8_bytes": 64, "compression_ratio": 0.64,
                      "jpg_bytes_per_pixel": 1.56, "ssim_win_size_whole_image": None,
                      "skimage_version": "test",
                      "output_jpg_sha256": f"o{o}{q}".ljust(64, "0"),
                      "decoded_jpg_uint8_sha256": f"j{o}{q}".ljust(64, "0")}
                fidelity_rows.append(fr)
                fidelity_by_key[(o, q)] = fr
                geometry_rows.append({"original_image_id": o, "jpeg_quality": q,
                                      "bbox_scaling_required": False,
                                      "pre_jpeg_shape_unchanged": True,
                                      "decoded_jpg_shape_unchanged": True,
                                      "reference_png_exact_pixel_match": True,
                                      "jpg_mode_L": True,
                                      "decoded_jpg_dtype_uint8": True,
                                      "exif_orientation_absent_or_1": True})
                roi_rows.append({"annotation_id": f"{o}", "canonical_ann_id": f"{o}",
                                 "source_row_id": "0", "rad_id": "R1", "image_id": o,
                                 "coco_image_id": id_to_coco[o]["id"],
                                 "category_id": (1 if o == "a" else 2),
                                 "canonical_class_id": (0 if o == "a" else 1),
                                 "class_id_original": 0,
                                 "class_name": ("Aortic enlargement" if o == "a" else "Atelectasis"),
                                 "canonical_x_min": 0.0, "canonical_y_min": 0.0,
                                 "canonical_x_max": 4.0, "canonical_y_max": 4.0,
                                 "bbox_area": 16.0, "relative_bbox_area": (0.01 if o == "a" else 0.05),
                                 "extraction_x0": 0, "extraction_y0": 0,
                                 "extraction_x1": 4, "extraction_y1": 4,
                                 "roi_width": 4, "roi_height": 4,
                                 "jpeg_quality": q,
                                 "ROI_MAE": 3.0, "ROI_PSNR": 36.0, "ROI_PSNR_is_infinite": False,
                                 "ROI_SSIM": (None if o == "b" else 0.9),
                                 "ROI_SSIM_evaluable": (False if o == "b" else True),
                                 "ROI_SSIM_reason": ("roi_too_small_for_ssim" if o == "b" else "ok"),
                                 "ROI_SSIM_win_size": None,
                                 "ROI_maximum_absolute_error": 9.0})
        return {
            "staging": staging, "decoder_env": {"python_version": "test"},
            "selected_ids": oids, "negative_ids": set(),
            "decode_stats": {"unique_pixel_decoded_image_count": 2},
            "protocol_evidence": {"protocol_sha256": P.EXPECTED_PROTOCOL_SHA256},
            "coco_sha256": P.EXPECTED_COCO_MASTER_SHA256,
            "coverage_result": {"fully_covered": True, "covered_total": len(all_features),
                                "all_features_total": len(all_features),
                                "classes_covered": 14, "classes_expected": 14,
                                "extrema_covered": 1, "extrema_expected": 1},
            "forbidden": {"preexisting_forbidden_artifact": False, "entries": []},
            "header_rows": [{"image_id": f"img{i}"} for i in range(P.LOCKED_INPUT_COUNTS["images"])],
            "image_features": image_features, "all_features": all_features,
            "id_to_coco": id_to_coco,
            "ann_by_oid": {"a": [{"category_id": 1, "canonical_class_id": 0, "area": 4.0}],
                           "b": [{"category_id": 2, "canonical_class_id": 1, "area": 4.0}]},
            "records": records, "class_names": {0: "Aortic enlargement", 1: "Atelectasis"},
            "class_image_count": {0: 100, 1: 5},
            "fidelity_rows": fidelity_rows, "fidelity_by_key": fidelity_by_key,
            "roi_rows": roi_rows,
            "roi_summary_mae": P.summarize_roi_metrics(roi_rows, "ROI_MAE"),
            "roi_summary_ssim": P.summarize_roi_metrics(roi_rows, "ROI_SSIM"),
            "roi_worst": P.worst_roi_cases(roi_rows, "ROI_MAE"),
            "roi_pairwise": P.pairwise_q100_minus_q95(roi_rows),
            "geometry_rows": geometry_rows,
            "headers": {o: self._header(o) for o in oids},
            "transform_records": {o: self._transform_record(o) for o in oids},
            "warnings_rows": [{"image_id": "a", "severity": "warning",
                               "code": "high_bit_not_bitsstored_minus_1", "detail": ""}],
            "uint8_cache": {o: np.zeros((8, 8), dtype=np.uint8) for o in oids},
            "n_selected_annotations": 2,
            "mandatory": {"dim_min_w": "a", "dim_max_w": "b"},
        }

    def _make_dummy_images(self, staging, oids):
        from PIL import Image
        for sub, ext in (("q95", "jpg"), ("q100", "jpg"), ("reference_uint8", "png")):
            d = staging / "images_jpg_pilot" / sub / "train"
            d.mkdir(parents=True, exist_ok=True)
            for o in oids:
                arr = np.zeros((8, 8), dtype=np.uint8)
                fmt = "JPEG" if ext == "jpg" else "PNG"
                Image.fromarray(arr, mode="L").save(d / f"{o}.{ext}", format=fmt)

    def test_full_evidence_and_atomic_promotion(self):
        import tempfile
        mod = load_script_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            staging = root / "staging"
            staging.mkdir()
            self._make_dummy_images(staging, ["a", "b"])
            ctx = self._build_ctx(staging)
            mod.generate_visual_evidence(ctx)  # real matplotlib files + manifest
            mod.write_all_evidence(ctx, decoded_count=2)

            reports = staging / "reports"
            # Blocker 2: all mandatory artifacts exist.
            for name in ("phase2D1B_pilot_environment.json",
                         "phase2D1B_pilot_synthetic_conformance.json",
                         "phase2D1B_pilot_synthetic_conformance.md",
                         "phase2D1B_pilot_multi_window_audit.csv",
                         "phase2D1B_pilot_reference_renderer_concordance.csv",
                         "phase2D1B_pilot_reference_viewer_manifest.csv",
                         "phase2D1B_pilot_header_inventory.csv",
                         "phase2D1B_pilot_metadata_strata.csv",
                         "phase2D1B_pilot_selection.csv",
                         "phase2D1B_pilot_selection_coverage.csv",
                         "phase2D1B_pilot_fidelity_metrics.csv",
                         "phase2D1B_pilot_bbox_roi_metrics.csv",
                         "phase2D1B_pilot_quality_summary.csv",
                         "phase2D1B_pilot_quality_pairwise.csv",
                         "phase2D1B_pilot_geometry_validation.csv",
                         "phase2D1B_pilot_visual_audit_manifest.csv",
                         "phase2D1B_pilot_errors.csv",
                         "phase2D1B_pilot_validation.json",
                         "phase2D1B_pilot_decision_template.json"):
                self.assertTrue((reports / name).exists(), name)

            # Blocker 1: header inventory has all 4894 data rows.
            with open(reports / "phase2D1B_pilot_header_inventory.csv", encoding="utf-8") as fh:
                n = sum(1 for _ in fh) - 1
            self.assertEqual(n, P.LOCKED_INPUT_COUNTS["images"])

            # Blocker 13: errors CSV records the actual warning + ROI-not-evaluable.
            errtext = (reports / "phase2D1B_pilot_errors.csv").read_text(encoding="utf-8")
            self.assertIn("high_bit_not_bitsstored_minus_1", errtext)
            self.assertIn("roi_ssim_not_evaluable", errtext)

            # Blocker 11: mapping full schema incl joined output/decoded hashes.
            maptext = (staging / "image_mapping" /
                       "phase2D1B_pilot_dicom_to_jpg_mapping.csv").read_text(encoding="utf-8")
            header = maptext.splitlines()[0]
            for col in ("output_jpg_sha256", "decoded_jpg_uint8_sha256", "bits_stored",
                        "sop_class_uid", "reference_png_byte_sha256", "decoder_backend"):
                self.assertIn(col, header)

            # Blocker 5 (visual): files exist in all 4 plot dirs.
            plots = staging / "plots" / "phase2D1B_pilot"
            for sub in ("full_image", "difference_heatmaps", "contact_sheets"):
                self.assertTrue(any((plots / sub).glob("*.png")), sub)

            # validate_staging must pass (expanded hard-checks + visual coverage).
            mod.validate_staging(staging, 2, 2, ctx["visual_expectations"])

            # Blocker 12: TRUE atomic promotion into temp targets.
            mod.REPORTS_DIR = root / "out_reports"
            mod.MAPPING_DIR = root / "out_mapping"
            mod.PILOT_OUT_DIR = root / "out_pilot"
            mod.PLOTS_DIR = root / "out_plots"
            mod.promote_atomic(staging, overwrite=False)
            self.assertTrue((mod.REPORTS_DIR / "phase2D1B_pilot_validation.json").exists())
            self.assertTrue((mod.MAPPING_DIR /
                             "phase2D1B_pilot_dicom_to_jpg_mapping.csv").exists())
            self.assertTrue((mod.PILOT_OUT_DIR / "q95" / "train" / "a.jpg").exists())
            # No '.promote' temporaries left behind.
            leftovers = list(mod.REPORTS_DIR.glob("*.promote"))
            self.assertEqual(leftovers, [])

    def test_promotion_overwrite_guard(self):
        import tempfile
        mod = load_script_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            staging = root / "staging"
            (staging / "reports").mkdir(parents=True)
            (staging / "image_mapping").mkdir(parents=True)
            (staging / "reports" / "x.json").write_text("{}", encoding="utf-8")
            mod.REPORTS_DIR = root / "out_reports"
            mod.MAPPING_DIR = root / "out_mapping"
            mod.PILOT_OUT_DIR = root / "out_pilot"
            mod.PLOTS_DIR = root / "out_plots"
            mod.REPORTS_DIR.mkdir()
            (mod.REPORTS_DIR / "x.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(P.UnsupportedInputError):
                mod.promote_atomic(staging, overwrite=False)


# =========================================================================== #
# A1: decode plugin explicitly passed + backend recorded                        #
# =========================================================================== #
class TestDecodePlugin(unittest.TestCase):
    def test_resolve_plugin_uncompressed_native(self):
        self.assertIsNone(P.resolve_decoding_plugin("1.2.840.10008.1.2.1", "pylibjpeg"))

    def test_resolve_plugin_jpeg2000_unavailable_raises(self):
        # gdcm not installed here -> must hard fail (no silent fallback).
        if not P.jpeg2000_backend_available("gdcm"):
            with self.assertRaises(P.UnsupportedInputError):
                P.resolve_decoding_plugin("1.2.840.10008.1.2.4.90", "gdcm")

    def test_decode_pixels_passes_plugin_and_records_backend(self):
        mod = load_script_module()
        captured = {}

        def fake_pixel_array(ds, decoding_plugin=None):
            captured["plugin"] = decoding_plugin
            return np.zeros((4, 4), dtype=np.uint16)

        header = {"TransferSyntaxUID": "1.2.840.10008.1.2.4.90"}
        orig = P.jpeg2000_backend_available
        try:
            P.jpeg2000_backend_available = lambda name: True  # force available
            arr, used = mod.decode_pixels(object(), header, "pylibjpeg",
                                          pixel_array_fn=fake_pixel_array)
        finally:
            P.jpeg2000_backend_available = orig
        self.assertEqual(captured["plugin"], "pylibjpeg")   # plugin passed in
        self.assertEqual(used, "pylibjpeg:pylibjpeg")        # actual backend recorded
        self.assertEqual(arr.shape, (4, 4))

    def test_decode_pixels_native_uncompressed(self):
        mod = load_script_module()
        header = {"TransferSyntaxUID": "1.2.840.10008.1.2.1"}
        arr, used = mod.decode_pixels(object(), header, "pylibjpeg",
                                      native_getter=lambda ds: np.ones((2, 2)))
        self.assertEqual(used, "pydicom_native")
        self.assertTrue((arr == 1).all())


# =========================================================================== #
# A1: Pillow JPEG2000 real capability                                           #
# =========================================================================== #
class TestPillowJp2k(unittest.TestCase):
    def test_capability_is_bool(self):
        self.assertIsInstance(P.pillow_jpeg2000_capable(), bool)

    def test_backend_available_uses_capability(self):
        # 'pillow' availability must equal the real codec capability, not import.
        self.assertEqual(P.jpeg2000_backend_available("pillow"),
                         P.pillow_jpeg2000_capable())


# =========================================================================== #
# A3: strict JSON rejects NaN / Infinity / -Infinity                            #
# =========================================================================== #
class TestStrictJson(unittest.TestCase):
    def test_accepts_normal(self):
        self.assertEqual(P.strict_json_loads('{"a": 1.5}'), {"a": 1.5})

    def test_rejects_nan(self):
        with self.assertRaises(ValueError):
            P.strict_json_loads('{"a": NaN}')

    def test_rejects_infinity(self):
        with self.assertRaises(ValueError):
            P.strict_json_loads('{"a": Infinity}')

    def test_rejects_neg_infinity(self):
        with self.assertRaises(ValueError):
            P.strict_json_loads('{"a": -Infinity}')


# =========================================================================== #
# A2: promotion backup + rollback on injected phase-B failure                   #
# =========================================================================== #
class TestPromotionRollback(unittest.TestCase):
    def test_injected_failure_restores_prior_evidence(self):
        mod = load_script_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            staging = root / "staging"
            (staging / "reports").mkdir(parents=True)
            (staging / "image_mapping").mkdir(parents=True)
            (staging / "reports" / "a.txt").write_text("NEW_A", encoding="utf-8")
            (staging / "reports" / "b.txt").write_text("NEW_B", encoding="utf-8")
            mod.REPORTS_DIR = root / "out_reports"
            mod.MAPPING_DIR = root / "out_mapping"
            mod.PILOT_OUT_DIR = root / "out_pilot"
            mod.PLOTS_DIR = root / "out_plots"
            mod.REPORTS_DIR.mkdir()
            (mod.REPORTS_DIR / "a.txt").write_text("PRIOR_A", encoding="utf-8")
            (mod.REPORTS_DIR / "b.txt").write_text("PRIOR_B", encoding="utf-8")

            def fail_hook(idx):
                if idx == 1:
                    raise RuntimeError("injected phase-B failure")

            with self.assertRaises(RuntimeError):
                mod.promote_atomic(staging, overwrite=True, _fail_hook=fail_hook)

            # Prior valid evidence fully restored.
            self.assertEqual((mod.REPORTS_DIR / "a.txt").read_text(), "PRIOR_A")
            self.assertEqual((mod.REPORTS_DIR / "b.txt").read_text(), "PRIOR_B")
            # No leftover temporaries.
            self.assertEqual(list(mod.REPORTS_DIR.glob("*.promote")), [])
            self.assertEqual(list(mod.REPORTS_DIR.glob("*.backup")), [])


# =========================================================================== #
# A4: synthetic conformance artifact completeness                               #
# =========================================================================== #
class TestSyntheticConformanceArtifact(unittest.TestCase):
    def test_all_pass_and_cases_present(self):
        mod = load_script_module()
        with tempfile.TemporaryDirectory() as td:
            reports = Path(td)
            mod.write_synthetic_conformance(reports)
            data = json.loads((reports / "phase2D1B_pilot_synthetic_conformance.json")
                              .read_text(encoding="utf-8"))
            self.assertTrue(data["all_pass"])
            names = " ".join(c["case"] for c in data["cases"])
            for token in ("unsigned_stored_range", "signed_stored_range",
                          "identity_modality_branch", "positive_rescale",
                          "negative_rescale", "modality_lut_output_bounds",
                          "window_linear", "window_linear_exact", "window_sigmoid",
                          "voi_lut_normalize", "MONOCHROME1", "MONOCHROME2",
                          "seqTrue", "pixel_padding_value_mask",
                          "pixel_padding_range_mask", "multi_valued_window",
                          "numpy_rint_uint8"):
                self.assertIn(token, names, token)
            self.assertTrue((reports / "phase2D1B_pilot_synthetic_conformance.md").exists())


# =========================================================================== #
# A7: canonical<->COCO bbox cross-check + out-of-bounds extraction               #
# =========================================================================== #
class TestBboxCrosscheck(unittest.TestCase):
    def _canon(self):
        return {"canonical_ann_id": "7", "image_id": "img", "canonical_class_id": 3,
                "x_min": "10", "y_min": "20", "x_max": "30", "y_max": "50"}

    def _coco(self):
        return {"canonical_ann_id": 7, "original_image_id": "img", "canonical_class_id": 3,
                "category_id": 4, "bbox": [10.0, 20.0, 20.0, 30.0]}

    def test_match_passes(self):
        P.crosscheck_canonical_coco_bbox(self._canon(), self._coco(), {3: 4})

    def test_missing_coco_raises(self):
        with self.assertRaises(P.UnsupportedInputError):
            P.crosscheck_canonical_coco_bbox(self._canon(), None, {3: 4})

    def test_category_mapping_from_metadata_not_hardcoded(self):
        # If metadata maps canonical 3 -> category 99, a coco category_id of 4
        # must fail (we do NOT hard-code +1).
        with self.assertRaises(P.UnsupportedInputError):
            P.crosscheck_canonical_coco_bbox(self._canon(), self._coco(), {3: 99})

    def test_coordinate_mismatch_raises(self):
        bad = self._coco()
        bad["bbox"] = [10.0, 20.0, 99.0, 30.0]  # x_max would be 109 != 30
        with self.assertRaises(P.UnsupportedInputError):
            P.crosscheck_canonical_coco_bbox(self._canon(), bad, {3: 4})

    def test_out_of_bounds_extraction_raises(self):
        with self.assertRaises(P.UnsupportedInputError):
            P.assert_extraction_in_bounds(0, 0, 10, 5, width=8, height=8)  # x1=10 > 8

    def test_in_bounds_ok(self):
        P.assert_extraction_in_bounds(0, 0, 4, 4, width=8, height=8)


# =========================================================================== #
# A10: reference renderer status from evidence                                  #
# =========================================================================== #
class TestReferenceRendererStatus(unittest.TestCase):
    def test_dependency_unavailable(self):
        self.assertEqual(P.reference_renderer_status(False, False),
                         "NOT_RUN_DEPENDENCY_UNAVAILABLE")

    def test_uncontrolled_config(self):
        self.assertEqual(P.reference_renderer_status(True, False),
                         "NOT_COMPARABLE_CONFIGURATION_UNCONTROLLED")

    def test_controlled_pass_fail(self):
        self.assertEqual(P.reference_renderer_status(True, True, concordant=True), "PASS")
        self.assertEqual(P.reference_renderer_status(True, True, concordant=False), "FAIL")


# =========================================================================== #
# A6: small-lesion / rare-class / per-class distributions                       #
# =========================================================================== #
class TestSummaryRankings(unittest.TestCase):
    def _roi(self):
        return [
            {"canonical_ann_id": "1", "image_id": "a", "canonical_class_id": 0,
             "jpeg_quality": 95, "relative_bbox_area": 0.05, "ROI_MAE": 5.0},
            {"canonical_ann_id": "2", "image_id": "b", "canonical_class_id": 1,
             "jpeg_quality": 95, "relative_bbox_area": 0.001, "ROI_MAE": 2.0},
            {"canonical_ann_id": "3", "image_id": "c", "canonical_class_id": 0,
             "jpeg_quality": 95, "relative_bbox_area": 0.02, "ROI_MAE": 8.0},
        ]

    def test_small_lesion_ranking(self):
        sl = P.small_lesion_ranking(self._roi())
        self.assertEqual(sl["ranking_basis"], "relative_bbox_area_ascending")
        self.assertEqual(sl["smallest_overall"]["canonical_ann_id"], "2")
        # smallest per class 0 is ann '3' (0.02) vs ann '1' (0.05)
        self.assertEqual(sl["smallest_per_class"][0]["canonical_ann_id"], "3")

    def test_rare_class_ranking(self):
        rare = P.rare_class_ranking({0: 100, 1: 5, 2: 50}, [0, 1], top=2)
        self.assertEqual(rare["ranking_basis"], "canonical_class_image_count_ascending")
        self.assertEqual(rare["rare_classes"][0], 1)  # class 1 has fewest images

    def test_per_class_distribution(self):
        dist = P.per_class_distribution(self._roi(), "ROI_MAE")
        by_class = {(d["canonical_class_id"]): d for d in dist if d["jpeg_quality"] == 95}
        self.assertAlmostEqual(by_class[0]["mean"], 6.5)  # (5+8)/2
        self.assertEqual(by_class[1]["n"], 1)


# =========================================================================== #
# A8: blocked run must not overwrite prior valid evidence                        #
# =========================================================================== #
class TestBlockedPreservation(unittest.TestCase):
    def test_emit_blocked_preserves_prior(self):
        mod = load_script_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mod.REPORTS_DIR = root / "reports"
            mod.REPORTS_DIR.mkdir()
            prior = mod.REPORTS_DIR / "phase2D1B_pilot_validation.json"
            prior.write_text('{"phase_status": "OPEN_REVIEW_REQUIRED"}', encoding="utf-8")

            pre = {"evidence": {"protocol_sha256": "s" * 64}}
            xcheck = {"coco_sha256": "c" * 64}
            rc = mod.emit_blocked(None, pre, xcheck, {}, {"entries": []},
                                  reason="dim_mismatch", image_id="x")
            self.assertEqual(rc, 6)
            # Prior VALID validation.json untouched.
            self.assertEqual(prior.read_text(encoding="utf-8"),
                             '{"phase_status": "OPEN_REVIEW_REQUIRED"}')
            # Blocked report written to the SEPARATE failure directory.
            blocked = list((mod.REPORTS_DIR / "phase2D1B_pilot_blocked").glob("*.json"))
            self.assertTrue(blocked)


# =========================================================================== #
# Final #1: rollback gap - failure BETWEEN backup move and replace               #
# =========================================================================== #
class TestPromotionMidGapRollback(unittest.TestCase):
    def test_failure_between_backup_and_replace_restores_prior(self):
        mod = load_script_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            staging = root / "staging"
            (staging / "reports").mkdir(parents=True)
            (staging / "image_mapping").mkdir(parents=True)
            (staging / "reports" / "a.txt").write_text("NEW_A", encoding="utf-8")
            (staging / "reports" / "b.txt").write_text("NEW_B", encoding="utf-8")
            mod.REPORTS_DIR = root / "out_reports"
            mod.MAPPING_DIR = root / "out_mapping"
            mod.PILOT_OUT_DIR = root / "out_pilot"
            mod.PLOTS_DIR = root / "out_plots"
            mod.REPORTS_DIR.mkdir()
            prior_a = b"PRIOR_A_BYTES"
            prior_b = b"PRIOR_B_BYTES"
            (mod.REPORTS_DIR / "a.txt").write_bytes(prior_a)
            (mod.REPORTS_DIR / "b.txt").write_bytes(prior_b)

            # Fail EXACTLY in the gap on the 2nd item (after item 0 completed,
            # after b's backup move but before b's replacement).
            def mid_hook(idx, dest):
                if idx == 1:
                    raise RuntimeError("crash in the backup<->replace gap")

            with self.assertRaises(RuntimeError):
                mod.promote_atomic(staging, overwrite=True, _mid_hook=mid_hook)

            # Prior destinations byte-identical restored (both items).
            self.assertEqual((mod.REPORTS_DIR / "a.txt").read_bytes(), prior_a)
            self.assertEqual((mod.REPORTS_DIR / "b.txt").read_bytes(), prior_b)
            # No temporaries and no partial new output.
            self.assertEqual(list(mod.REPORTS_DIR.glob("*.promote")), [])
            self.assertEqual(list(mod.REPORTS_DIR.glob("*.backup")), [])
            for p in mod.REPORTS_DIR.glob("*.txt"):
                self.assertIn(p.read_bytes(), (prior_a, prior_b))


# =========================================================================== #
# Final #2: category_id from metadata mapping, never canonical+1                 #
# =========================================================================== #
class TestRoiCategorySource(unittest.TestCase):
    def _canon(self):
        return {"canonical_ann_id": "1", "canonical_class_id": 3, "class_id_original": 3,
                "class_name": "Cardiomegaly", "source_row_id": "0", "rad_id": "R1",
                "x_min": "0", "y_min": "0", "x_max": "4", "y_max": "4",
                "bbox_width": "4", "bbox_height": "4"}

    def test_nonidentity_category_used(self):
        mod = load_script_module()
        # Non-identity mapping: canonical class 3 -> category 17 (not 4).
        row = mod._roi_row(self._canon(), "img", {"id": 1}, 95, {3: "Cardiomegaly"},
                           16.0, 0.01, (0, 0, 4, 4), 1.0, 40.0, 0.9, True, "ok", 3,
                           category_id=17)
        self.assertEqual(row["category_id"], 17)
        self.assertNotEqual(row["category_id"], 4)  # would be canonical(3)+1

    def test_missing_category_raises(self):
        mod = load_script_module()
        with self.assertRaises(P.UnsupportedInputError):
            mod._roi_row(self._canon(), "img", {"id": 1}, 95, {}, 16.0, 0.01,
                         (0, 0, 4, 4), 1.0, 40.0, 0.9, True, "ok", 3)


# =========================================================================== #
# Final #3: strict bbox coordinate tolerance (1e-6, not 1.0)                     #
# =========================================================================== #
class TestBboxTolerance(unittest.TestCase):
    def _canon(self):
        return {"canonical_ann_id": "1", "image_id": "img", "canonical_class_id": 0,
                "x_min": "10.0", "y_min": "20.0", "x_max": "30.0", "y_max": "50.0"}

    def _coco(self, dx=0.0):
        return {"canonical_ann_id": 1, "original_image_id": "img", "canonical_class_id": 0,
                "category_id": 1, "bbox": [10.0, 20.0, 20.0 + dx, 30.0]}

    def test_default_tolerance_is_strict(self):
        self.assertEqual(P.BBOX_COORD_TOLERANCE, 1e-6)

    def test_within_tolerance_passes(self):
        P.crosscheck_canonical_coco_bbox(self._canon(), self._coco(dx=1e-7), {0: 1})

    def test_beyond_tolerance_blocks(self):
        with self.assertRaises(P.UnsupportedInputError):
            P.crosscheck_canonical_coco_bbox(self._canon(), self._coco(dx=1e-3), {0: 1})

    def test_one_pixel_slack_no_longer_allowed(self):
        # A whole-pixel discrepancy used to pass under tol=1.0; must now block.
        with self.assertRaises(P.UnsupportedInputError):
            P.crosscheck_canonical_coco_bbox(self._canon(), self._coco(dx=0.9), {0: 1})


# =========================================================================== #
# Final #4: visual completeness - no silent truncation                          #
# =========================================================================== #
class TestVisualCompleteness(unittest.TestCase):
    def test_exceeding_cap_hard_fails(self):
        mod = load_script_module()
        orig = mod.VISUAL_SUBSET_MAX
        try:
            mod.VISUAL_SUBSET_MAX = 1  # force required set to exceed cap
            ctx = {
                "selected_ids": ["a", "b", "c"],
                "negative_ids": set(),
                "mandatory": {"dim_min_w": "a", "dim_max_w": "b", "px_min": "c"},
                "roi_rows": [], "class_image_count": {}, "image_features": {
                    "a": set(), "b": set(), "c": set()},
                "fidelity_rows": [], "transform_records": {}, "warnings_rows": [],
            }
            with self.assertRaises(P.UnsupportedInputError):
                mod.select_visual_subset(ctx)
        finally:
            mod.VISUAL_SUBSET_MAX = orig

    def test_missing_required_annotation_crop_raises(self):
        # A required annotation-level reason with no matching crop must raise
        # (actually invokes the coverage checker).
        mod = load_script_module()
        man = [{"image_id": "a", "artifact_type": "full_image",
                "artifact_path": "full_image/a.png", "canonical_ann_id": "",
                "reason": "extremum:dim_min_w", "stratum_signature": "s1"}]
        req = {"expect_extremum": True, "annotation_requests": [
            {"canonical_ann_id": "7", "image_id": "a", "reason": "worst_q95_roi"}],
            "min_no_finding_unique": 0, "min_no_finding_distinct_strata": 0,
            "warning_image_ids": []}
        with self.assertRaises(P.UnsupportedInputError):
            mod.check_visual_coverage(man, req, Path("/tmp"), {"7": "a"})


# =========================================================================== #
# Visual #1: No Finding coverage - unique images + distinct strata              #
# =========================================================================== #
class TestNoFindingVisualCoverage(unittest.TestCase):
    def _req(self):
        return {"expect_extremum": False, "annotation_requests": [],
                "warning_image_ids": [], "min_no_finding_unique": 4,
                "min_no_finding_distinct_strata": 4}

    def _nf_rows(self, images_sigs):
        rows = []
        for oid, sig in images_sigs:
            # 2 artifact rows per image (full_image + difference_heatmap).
            for at in ("full_image", "difference_heatmap"):
                rows.append({"image_id": oid, "artifact_type": at, "artifact_path": "x",
                             "canonical_ann_id": "", "reason": "no_finding_strata_diverse",
                             "stratum_signature": sig})
        return rows

    def test_two_images_two_rows_not_counted_as_four(self):
        mod = load_script_module()
        # 2 unique images, 2 rows each = 4 rows, but only 2 unique images.
        man = self._nf_rows([("i1", "sigA"), ("i2", "sigB")])
        with self.assertRaises(P.UnsupportedInputError):
            mod.check_visual_coverage(man, self._req(), Path("/tmp"), {})

    def test_four_images_same_stratum_fails(self):
        mod = load_script_module()
        man = self._nf_rows([("i1", "S"), ("i2", "S"), ("i3", "S"), ("i4", "S")])
        with self.assertRaises(P.UnsupportedInputError):
            mod.check_visual_coverage(man, self._req(), Path("/tmp"), {})

    def test_four_unique_distinct_strata_passes(self):
        mod = load_script_module()
        man = self._nf_rows([("i1", "S1"), ("i2", "S2"), ("i3", "S3"), ("i4", "S4")])
        mod.check_visual_coverage(man, self._req(), Path("/tmp"), set())  # no raise

    def test_select_subset_insufficient_distinct_strata_hard_fails(self):
        mod = load_script_module()
        # 4 No Finding images but all share the SAME metadata stratum signature.
        ctx = {
            "selected_ids": ["n1", "n2", "n3", "n4"],
            "negative_ids": {"n1", "n2", "n3", "n4"},
            "mandatory": {}, "roi_rows": [], "fidelity_rows": [],
            "transform_records": {}, "warnings_rows": [], "class_image_count": {},
            "image_features": {o: {"PhotometricInterpretation=MONOCHROME2"}
                               for o in ("n1", "n2", "n3", "n4")},
        }
        with self.assertRaises(P.UnsupportedInputError):
            mod.select_visual_subset(ctx)


# =========================================================================== #
# Visual #2: annotation-specific bbox crop coverage                             #
# =========================================================================== #
class TestAnnotationCropCoverage(unittest.TestCase):
    def test_crop_matches_activating_annotation(self):
        mod = load_script_module()
        with tempfile.TemporaryDirectory() as td:
            plots = Path(td)
            (plots / "bbox_crops").mkdir(parents=True)
            (plots / "bbox_crops" / "imgX__42.png").write_bytes(b"\x89PNG")
            man = [{"image_id": "imgX", "artifact_type": "bbox_crop",
                    "artifact_path": "bbox_crops/imgX__42.png", "canonical_ann_id": "42",
                    "reason": "worst_q95_roi;rare_class5", "stratum_signature": ""}]
            req = {"expect_extremum": False, "min_no_finding_unique": 0,
                   "min_no_finding_distinct_strata": 0, "warning_image_ids": [],
                   "annotation_requests": [
                       {"canonical_ann_id": "42", "image_id": "imgX", "reason": "worst_q95_roi"},
                       {"canonical_ann_id": "42", "image_id": "imgX", "reason": "rare_class5"}]}
            mod.check_visual_coverage(man, req, plots, {"42": "imgX"})  # passes

    def test_correct_ann_but_wrong_image_raises(self):
        # Crop file/manifest has the right canonical_ann_id but the WRONG image.
        mod = load_script_module()
        with tempfile.TemporaryDirectory() as td:
            plots = Path(td)
            (plots / "bbox_crops").mkdir(parents=True)
            (plots / "bbox_crops" / "wrong__42.png").write_bytes(b"\x89PNG")
            man = [{"image_id": "wrongImg", "artifact_type": "bbox_crop",
                    "artifact_path": "bbox_crops/wrong__42.png", "canonical_ann_id": "42",
                    "reason": "worst_q95_roi", "stratum_signature": ""}]
            req = {"expect_extremum": False, "min_no_finding_unique": 0,
                   "min_no_finding_distinct_strata": 0, "warning_image_ids": [],
                   "annotation_requests": [
                       {"canonical_ann_id": "42", "image_id": "imgX", "reason": "worst_q95_roi"}]}
            # ROI evidence links ann 42 -> imgX, but manifest crop is on wrongImg.
            with self.assertRaises(P.UnsupportedInputError):
                mod.check_visual_coverage(man, req, plots, {"42": "imgX"})

    def test_crop_ann_not_in_roi_evidence_raises(self):
        mod = load_script_module()
        with tempfile.TemporaryDirectory() as td:
            plots = Path(td)
            (plots / "bbox_crops").mkdir(parents=True)
            (plots / "bbox_crops" / "imgX__42.png").write_bytes(b"\x89PNG")
            man = [{"image_id": "imgX", "artifact_type": "bbox_crop",
                    "artifact_path": "bbox_crops/imgX__42.png", "canonical_ann_id": "42",
                    "reason": "worst_q95_roi", "stratum_signature": ""}]
            req = {"expect_extremum": False, "min_no_finding_unique": 0,
                   "min_no_finding_distinct_strata": 0, "warning_image_ids": [],
                   "annotation_requests": [{"canonical_ann_id": "42", "image_id": "imgX",
                                            "reason": "worst_q95_roi"}]}
            with self.assertRaises(P.UnsupportedInputError):
                mod.check_visual_coverage(man, req, plots, {})  # ann not in ROI evidence

    def test_generate_creates_annotation_specific_crops(self):
        # Two annotations on the SAME image must yield two distinct crops keyed
        # by canonical_ann_id (not a single first_roi crop).
        mod = load_script_module()
        with tempfile.TemporaryDirectory() as td:
            staging = Path(td)
            q95 = staging / "images_jpg_pilot" / "q95" / "train"
            q100 = staging / "images_jpg_pilot" / "q100" / "train"
            for d in (q95, q100):
                d.mkdir(parents=True)
            from PIL import Image
            for d in (q95, q100):
                Image.fromarray(np.zeros((8, 8), np.uint8), mode="L").save(d / "img.jpg")
            roi = []
            for q in (95, 100):
                for aid, rel in (("10", 0.01), ("20", 0.5)):
                    roi.append({"canonical_ann_id": aid, "annotation_id": aid,
                                "image_id": "img", "canonical_class_id": (0 if aid == "10" else 1),
                                "jpeg_quality": q, "relative_bbox_area": rel, "ROI_MAE": (9.0 if aid == "20" else 1.0),
                                "extraction_x0": 0, "extraction_y0": 0,
                                "extraction_x1": 4, "extraction_y1": 4, "class_name": "C"})
            ctx = {
                "staging": staging, "selected_ids": ["img"], "negative_ids": set(),
                "mandatory": {"dim_min_w": "img"}, "roi_rows": roi,
                "fidelity_rows": [{"original_image_id": "img", "jpeg_quality": 95, "mae": 3.0}],
                "transform_records": {"img": {"modality_branch": "identity", "voi_branch": "windowing",
                                              "presentation_inversion_count": 0,
                                              "padding_present": False,
                                              "uint8_zero_fraction": 0.0, "uint8_255_fraction": 0.0}},
                "warnings_rows": [], "class_image_count": {0: 100, 1: 3},
                "image_features": {"img": {"PhotometricInterpretation=MONOCHROME2"}},
                "uint8_cache": {"img": np.zeros((8, 8), np.uint8)},
            }
            mod.generate_visual_evidence(ctx)
            crops = {p.name for p in (staging / "plots" / "phase2D1B_pilot" / "bbox_crops").glob("*.png")}
            # smallest overall/class0 -> ann 10 ; class1/rare/worst -> ann 20
            self.assertIn("img__10.png", crops)
            self.assertIn("img__20.png", crops)
            man_crops = [m for m in ctx["visual_manifest"] if m["artifact_type"] == "bbox_crop"]
            ann_ids = {m["canonical_ann_id"] for m in man_crops}
            self.assertEqual(ann_ids, {"10", "20"})


# =========================================================================== #
# Window #2: DICOM LINEAR WindowWidth == 1 threshold + function-aware validity   #
# =========================================================================== #
class TestWindowWidthByFunction(unittest.TestCase):
    def test_linear_width1_threshold(self):
        # width==1 is a valid LINEAR threshold at (center-0.5).
        out = P.window_linear(np.array([999.0, 1000.0, 1001.0]), 1000.0, 1.0)
        # threshold = 999.5 : x<=999.5 -> 0 ; x>999.5 -> 1
        self.assertTrue(np.array_equal(out, np.array([0.0, 1.0, 1.0])))

    def test_linear_width_half_blocks(self):
        with self.assertRaises(P.ProtocolGapError):
            P.window_linear(np.array([1.0]), 10.0, 0.5)

    def test_linear_width_greater_than_one_formula(self):
        out = P.window_linear(np.array([800.0, 1200.0]), 1000.0, 400.0)
        self.assertTrue(np.isclose(out[0], 0.0) and np.isclose(out[1], 1.0))

    def test_linear_exact_width_half_ok(self):
        out = P.window_linear_exact(np.array([10.0]), 10.0, 0.5)
        self.assertAlmostEqual(out[0], 0.5)  # center -> 0.5, no raise

    def test_sigmoid_width_half_ok(self):
        out = P.window_sigmoid(np.array([10.0]), 10.0, 0.5)
        self.assertAlmostEqual(out[0], 0.5)  # center -> 0.5, no raise

    def test_validate_width_by_function(self):
        # LINEAR: <1 blocks, ==1 ok.
        with self.assertRaises(P.ProtocolGapError):
            P.validate_window_width_for_function(0.5, "LINEAR")
        P.validate_window_width_for_function(1.0, "LINEAR")
        P.validate_window_width_for_function(1.0, None)  # None -> LINEAR
        # LINEAR_EXACT / SIGMOID: >0 ok, <=0 blocks.
        P.validate_window_width_for_function(0.5, "LINEAR_EXACT")
        P.validate_window_width_for_function(0.5, "SIGMOID")
        with self.assertRaises(P.ProtocolGapError):
            P.validate_window_width_for_function(0.0, "SIGMOID")

    def test_header_preflight_blocks_linear_width_half(self):
        mod = load_script_module()
        header = {
            "modality_lut_present": False, "rescale_slope_present": False,
            "rescale_intercept_present": False, "PhotometricInterpretation": "MONOCHROME2",
            "PresentationLUTShape": "ABSENT", "presentation_lut_sequence_present": False,
            "voi_lut_present": False, "VOILUTFunction": "ABSENT",
            "WindowCenter_all": "1000", "WindowWidth_all": "0.5",
        }
        with self.assertRaises(P.ProtocolGapError):
            mod.header_transform_preflight(header)

    def test_header_preflight_allows_linear_width_one(self):
        mod = load_script_module()
        header = {
            "modality_lut_present": False, "rescale_slope_present": False,
            "rescale_intercept_present": False, "PhotometricInterpretation": "MONOCHROME2",
            "PresentationLUTShape": "ABSENT", "presentation_lut_sequence_present": False,
            "voi_lut_present": False, "VOILUTFunction": "ABSENT",
            "WindowCenter_all": "1000", "WindowWidth_all": "1",
        }
        mod.header_transform_preflight(header)  # no raise


if __name__ == "__main__":
    unittest.main(verbosity=2)
