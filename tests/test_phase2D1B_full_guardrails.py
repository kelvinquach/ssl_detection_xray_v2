#!/usr/bin/env python3
"""Phase 2D.1B-Full guardrail tests (behavioral + fault-injection + AST).

Synthetic fixtures and temporary directories only. These tests never convert the
4,894 real DICOM, never touch canonical inputs, and never run the real full
conversion. They exercise the actual gate/validation/promotion code paths on a
tiny synthetic scope by redirecting module constants into a temp tree.

Run (Windows CMD):
    python -m pytest tests/test_phase2D1B_full_guardrails.py -v
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
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

FULL_PATH = REPO_ROOT / "scripts" / "02D1B_full_dicom_to_jpg.py"
PILOT_TEST_PATH = REPO_ROOT / "tests" / "test_phase2D1B_pilot_guardrails.py"

try:
    import yaml
    HAVE_YAML = True
except Exception:
    HAVE_YAML = False

try:
    from PIL import Image  # noqa: F401
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False

try:
    import skimage  # noqa: F401
    HAVE_SKIMAGE = True
except Exception:
    HAVE_SKIMAGE = False

try:
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    HAVE_PYDICOM = True
except Exception:
    HAVE_PYDICOM = False


def load_full():
    spec = importlib.util.spec_from_file_location("phase2D1B_full_mod", FULL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Synthetic-scope fixtures                                                     #
# --------------------------------------------------------------------------- #
SYN_COUNTS = {"images": 2, "abnormal_images": 1, "no_finding_images": 1,
              "annotations": 1, "categories": 1, "no_finding_annotations": 0}
UNCOMPRESSED = "1.2.840.10008.1.2.1"


def make_protocol_dict(counts):
    return {
        "protocol_metadata": {"protocol_version": "1.0.0"},
        "jpeg_encoding": {"quality_candidates": [95, 100], "final_quality": None,
                          "final_quality_status": "pending_phase2D1B_pilot"},
        "geometry_bbox_policy": {"resize": False, "crop": False, "rotation": False,
                                 "flip": False, "transpose": False,
                                 "bbox_scaling_expected": False},
        "voi_windowing_policy": {"direct_observed_per_image_min_max": "forbidden",
                                 "automatic_percentile_clipping": "forbidden"},
        "final_quality_decision_rule": {"final_quality_must_remain_null_in_this_phase": True},
        "locked_input_counts": dict(counts),
        "readiness_flags": {"jpg_training_representation_ready": False,
                            "dataset_training_ready": False,
                            "training_authorized": False},
        "forbidden_actions": {"full_conversion_run": False, "training_started": False},
        "output_channel_policy": {"jpg_storage": {"jpeg_mode": "L", "channels": 1}},
    }


def make_master():
    return {
        "info": {}, "licenses": [],
        "images": [
            {"id": 1, "original_image_id": "aaa", "file_name": "train/aaa.dicom",
             "width": 8, "height": 8, "is_negative": False, "canonical_image_id": 0,
             "scope_label": "abnormal"},
            {"id": 2, "original_image_id": "bbb", "file_name": "train/bbb.dicom",
             "width": 8, "height": 8, "is_negative": True, "canonical_image_id": 1,
             "scope_label": "no_finding"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 3, 3],
             "area": 9, "iscrowd": 0, "canonical_class_id": 0,
             "original_image_id": "aaa", "canonical_ann_id": 0},
        ],
        "categories": [
            {"id": 1, "name": "Aortic enlargement", "canonical_class_id": 0,
             "class_id_original": 0, "supercategory": "chest_abnormality"},
        ],
    }


def _make_real_dicom(path, rows=8, cols=8, transfer=None):
    ds = Dataset()
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.WindowCenter = 2048
    ds.WindowWidth = 4096
    arr = (np.arange(rows * cols).reshape(rows, cols) % 4096).astype("<u2")
    ds.PixelData = arr.tobytes()
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    fm = FileMetaDataset()
    # Encoding (little-endian / explicit VR) is defined by the Transfer Syntax
    # in file_meta; the deprecated Dataset.is_little_endian / is_implicit_VR
    # attributes are NOT set. ExplicitVRLittleEndian => little-endian, explicit.
    fm.TransferSyntaxUID = transfer or ExplicitVRLittleEndian
    fm.MediaStorageSOPClassUID = ds.SOPClassUID
    fm.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta = fm
    # Prefer the current API (pydicom >= 3.0): enforce_file_format=True writes a
    # proper File-Format DICOM (preamble + file_meta) and derives VR/endianness
    # from the Transfer Syntax, so no deprecation warnings fire. Fall back to the
    # legacy signature only on older pydicom (where those APIs are not
    # deprecated). Behaviour is identical: an ExplicitVRLittleEndian file.
    try:
        ds.save_as(str(path), enforce_file_format=True)
    except TypeError:
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.save_as(str(path), write_like_original=False)


class Scope:
    """Builds a synthetic scope in ``root`` and patches module/P constants."""

    def __init__(self, mod, root, real_dicom=False, transfer=UNCOMPRESSED):
        self.mod = mod
        self.root = Path(root)
        self.pilot = mod.load_pilot()
        self.counts = dict(SYN_COUNTS)
        self.master = make_master()
        self._saved = {}
        self._build(real_dicom, transfer)
        self._patch()

    def _build(self, real_dicom, transfer):
        r = self.root
        (r / "dicom" / "train").mkdir(parents=True)
        for oid in ("aaa", "bbb"):
            p = r / "dicom" / "train" / f"{oid}.dicom"
            if real_dicom:
                _make_real_dicom(p, transfer=transfer)
            else:
                p.write_bytes(b"DICM-DUMMY")  # existence-only for preflight gates
        self.dicom_root = r / "dicom"

        self.master_path = r / "coco_master.json"
        self.master_path.write_text(json.dumps(self.master), encoding="utf-8")
        self.coco_sha = self.pilot.file_sha256(self.master_path)

        self.decision_path = r / "decision.json"
        self.decision_path.write_text(json.dumps({
            "final_jpeg_quality": 95, "selected_candidate": 95,
            "full_conversion_authorized": True,
            "decision_status": "approved"}), encoding="utf-8")

        self.protocol_dict = make_protocol_dict(self.counts)
        self.protocol_path = r / "protocol.yaml"
        self.protocol_path.write_text(yaml.safe_dump(self.protocol_dict), encoding="utf-8")
        self.protocol_sha = P.protocol_sha256(self.protocol_dict)

        self.phase2a_path = r / "phase2a.csv"
        lines = ["image_id,image_width,image_height,TransferSyntaxUID"]
        for oid in ("aaa", "bbb"):
            lines.append(f"{oid},8,8,{transfer}")
        self.phase2a_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        self.phase2d_path = r / "phase2d.json"
        self._write_phase2d()

        self.canon_bbox = r / "canon_bbox.csv"
        self.canon_bbox.write_text("canonical_ann_id\n0\n", encoding="utf-8")
        self.canon_class = r / "canon_class.csv"
        self.canon_class.write_text("canonical_class_id,image_count\n0,1\n", encoding="utf-8")

    def _write_phase2d(self):
        counts = dict(self.counts)
        counts["canonical_bbox_rows"] = self.counts["annotations"]
        self.phase2d_path.write_text(json.dumps({
            "output_sha256": self.coco_sha, "counts": counts}), encoding="utf-8")

    def rewrite_master(self, master):
        self.master = master
        self.master_path.write_text(json.dumps(master), encoding="utf-8")
        self.coco_sha = self.pilot.file_sha256(self.master_path)
        self._write_phase2d()
        P.EXPECTED_COCO_MASTER_SHA256 = self.coco_sha

    def _save(self, obj, name):
        self._saved[(id(obj), name)] = (obj, name, getattr(obj, name))

    def _patch(self):
        mod, r = self.mod, self.root
        pairs = [
            # Patch REPO_ROOT to the temp root so it remains an ancestor of the
            # redirected FINAL_TRAIN / COCO_JPG_FINAL. Production keeps using
            # repo-relative paths (relative_to(REPO_ROOT)); this only keeps the
            # synthetic outputs under the (patched) repo root. Restored in
            # restore() like every other constant.
            (mod, "REPO_ROOT", r),
            (mod, "PROTOCOL_YAML", self.protocol_path),
            (mod, "DECISION_JSON", self.decision_path),
            (mod, "COCO_MASTER", self.master_path),
            (mod, "PHASE2A_META", self.phase2a_path),
            (mod, "PHASE2D_VALID", self.phase2d_path),
            (mod, "CANONICAL_BBOX", self.canon_bbox),
            (mod, "CANONICAL_CLASS", self.canon_class),
            (mod, "CANONICAL_INPUT_FILES", (self.protocol_path, self.decision_path,
                                            self.master_path, self.canon_bbox,
                                            self.canon_class, self.phase2a_path,
                                            self.phase2d_path)),
            (mod, "STAGING_ROOT", r / "staging"),
            (mod, "STAGING_TRAIN", r / "staging" / "train"),
            (mod, "FINAL_ROOT", r / "final"),
            (mod, "FINAL_TRAIN", r / "final" / "train"),
            (mod, "BACKUP_ROOT", r / "backup"),
            (mod, "FAILED_ROOT", r / "failed"),
            (mod, "COCO_JPG_FINAL", r / "final_coco_master_jpg.json"),
            (mod, "COCO_JPG_STAGING", r / "staging" / "coco_master_jpg.json"),
            (mod, "REPORTS_DIR", r / "reports"),
            (mod, "LOCKED_SCOPE", 2),
            (P, "EXPECTED_PROTOCOL_SHA256", self.protocol_sha),
            (P, "EXPECTED_COCO_MASTER_SHA256", self.coco_sha),
            (P, "LOCKED_INPUT_COUNTS", dict(self.counts)),
            (P, "NUM_ABNORMAL_CLASSES", 1),
        ]
        for obj, name, val in pairs:
            self._save(obj, name)
            setattr(obj, name, val)
        (r / "reports").mkdir(exist_ok=True)

    def restore(self):
        for obj, name, old in self._saved.values():
            setattr(obj, name, old)


# =========================================================================== #
# A. Decision / protocol / canonical gates (behavioral via preflight_inputs)    #
# =========================================================================== #
@unittest.skipUnless(HAVE_YAML, "PyYAML required")
class TestGatesA(unittest.TestCase):
    def setUp(self):
        self.mod = load_full()
        self.tmp = tempfile.TemporaryDirectory()
        self.scope = Scope(self.mod, self.tmp.name)

    def tearDown(self):
        self.scope.restore()
        self.tmp.cleanup()

    def _preflight(self):
        return self.mod.preflight_inputs(self.scope.pilot, self.scope.dicom_root)

    def test_happy_preflight_passes(self):
        out = self._preflight()
        self.assertEqual(out["counts"]["images"], 2)

    def test_quality_not_95_fails(self):
        self.scope.decision_path.write_text(json.dumps({
            "final_jpeg_quality": 100, "full_conversion_authorized": True}),
            encoding="utf-8")
        with self.assertRaises(P.Phase2D1BError):
            self._preflight()

    def test_authorization_false_fails(self):
        self.scope.decision_path.write_text(json.dumps({
            "final_jpeg_quality": 95, "full_conversion_authorized": False}),
            encoding="utf-8")
        with self.assertRaises(P.Phase2D1BError):
            self._preflight()

    def test_missing_decision_fails(self):
        self.scope.decision_path.unlink()
        with self.assertRaises(P.Phase2D1BError):
            self._preflight()

    def test_protocol_drift_fails(self):
        bad = copy.deepcopy(self.scope.protocol_dict)
        bad["protocol_metadata"]["protocol_version"] = "9.9.9"
        self.scope.protocol_path.write_text(yaml.safe_dump(bad), encoding="utf-8")
        with self.assertRaises(P.Phase2D1BError):
            self._preflight()

    def test_coco_hash_drift_fails(self):
        m = copy.deepcopy(self.scope.master)
        m["images"][0]["width"] = 9  # changes bytes -> sha drift
        self.scope.master_path.write_text(json.dumps(m), encoding="utf-8")
        with self.assertRaises(P.Phase2D1BError):
            self._preflight()

    def test_missing_dicom_fails(self):
        (self.scope.dicom_root / "train" / "bbb.dicom").unlink()
        with self.assertRaises(P.Phase2D1BError):
            self._preflight()

    def test_extra_dicom_fails(self):
        (self.scope.dicom_root / "train" / "zzz.dicom").write_bytes(b"X")
        with self.assertRaises(P.Phase2D1BError):
            self._preflight()

    def test_nested_dicom_fails(self):
        nested = self.scope.dicom_root / "train" / "sub"
        nested.mkdir()
        (nested / "aaa.dicom").write_bytes(b"X")
        with self.assertRaises(P.Phase2D1BError):
            self._preflight()

    def test_duplicate_image_id_fails(self):
        m = copy.deepcopy(self.scope.master)
        m["images"][1]["id"] = 1  # duplicate id
        self.scope.rewrite_master(m)
        with self.assertRaises(P.Phase2D1BError):
            self._preflight()

    def test_duplicate_file_name_fails(self):
        m = copy.deepcopy(self.scope.master)
        m["images"][1]["file_name"] = "train/aaa.dicom"  # duplicate file_name
        self.scope.rewrite_master(m)
        with self.assertRaises(P.Phase2D1BError):
            self._preflight()


# =========================================================================== #
# Environment: JPEG2000 required but backend unavailable -> hard fail            #
# =========================================================================== #
@unittest.skipUnless(HAVE_YAML, "PyYAML required")
class TestEnvPhase2A(unittest.TestCase):
    IDS = {"aaa", "bbb"}

    def setUp(self):
        self.mod = load_full()
        self.tmp = tempfile.TemporaryDirectory()
        self.scope = Scope(self.mod, self.tmp.name, transfer="1.2.840.10008.1.2.4.90")
        self._orig = P.jpeg2000_backend_available

    def tearDown(self):
        P.jpeg2000_backend_available = self._orig
        self.scope.restore()
        self.tmp.cleanup()

    def _env(self):
        return self.mod.preflight_environment("pylibjpeg", self.IDS)

    def test_jpeg2000_unavailable_backend_fails(self):
        P.jpeg2000_backend_available = lambda name: False
        with self.assertRaises(P.Phase2D1BError):
            self._env()

    def test_missing_transfer_syntax_column_fails(self):
        self.scope.phase2a_path.write_text(
            "image_id,image_width,image_height\naaa,8,8\nbbb,8,8\n", encoding="utf-8")
        with self.assertRaises(P.Phase2D1BError):
            self._env()

    def test_missing_transfer_syntax_value_fails(self):
        self.scope.phase2a_path.write_text(
            "image_id,image_width,image_height,TransferSyntaxUID\n"
            "aaa,8,8,\nbbb,8,8,1.2.840.10008.1.2.1\n", encoding="utf-8")
        with self.assertRaises(P.Phase2D1BError):
            self._env()

    def test_missing_image_id_fails(self):
        self.scope.phase2a_path.write_text(
            "image_id,image_width,image_height,TransferSyntaxUID\n"
            "aaa,8,8,1.2.840.10008.1.2.1\n", encoding="utf-8")  # bbb missing
        with self.assertRaises(P.Phase2D1BError):
            self._env()

    def test_extra_image_id_fails(self):
        self.scope.phase2a_path.write_text(
            "image_id,image_width,image_height,TransferSyntaxUID\n"
            "aaa,8,8,1.2.840.10008.1.2.1\nbbb,8,8,1.2.840.10008.1.2.1\n"
            "ccc,8,8,1.2.840.10008.1.2.1\n", encoding="utf-8")
        with self.assertRaises(P.Phase2D1BError):
            self._env()

    def test_duplicate_image_id_fails(self):
        # Duplicate must NOT be silently overwritten by a dict.
        self.scope.phase2a_path.write_text(
            "image_id,image_width,image_height,TransferSyntaxUID\n"
            "aaa,8,8,1.2.840.10008.1.2.1\naaa,8,8,1.2.840.10008.1.2.1\n"
            "bbb,8,8,1.2.840.10008.1.2.1\n", encoding="utf-8")
        with self.assertRaises(P.Phase2D1BError):
            self._env()

    def test_duplicate_conflicting_syntax_fails(self):
        self.scope.phase2a_path.write_text(
            "image_id,image_width,image_height,TransferSyntaxUID\n"
            "aaa,8,8,1.2.840.10008.1.2.1\naaa,8,8,1.2.840.10008.1.2.4.90\n"
            "bbb,8,8,1.2.840.10008.1.2.1\n", encoding="utf-8")
        with self.assertRaises(P.Phase2D1BError):
            self._env()

    def test_empty_image_id_fails(self):
        self.scope.phase2a_path.write_text(
            "image_id,image_width,image_height,TransferSyntaxUID\n"
            ",8,8,1.2.840.10008.1.2.1\nbbb,8,8,1.2.840.10008.1.2.1\n", encoding="utf-8")
        with self.assertRaises(P.Phase2D1BError):
            self._env()


# =========================================================================== #
# B. Source immutability (snapshot / verify hashes)                             #
# =========================================================================== #
class TestSourceImmutability(unittest.TestCase):
    def test_drift_detected_per_canonical_input_and_dicom(self):
        mod = load_full()
        pilot = mod.load_pilot()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # Distinct canonical inputs + two DICOM.
            canon = []
            for name in ("protocol.yaml", "decision.json", "coco.json",
                         "bbox.csv", "class.csv", "phase2a.csv", "phase2d.json"):
                p = root / name
                p.write_bytes(name.encode())
                canon.append(p)
            (root / "train").mkdir()
            d1 = root / "train" / "aaa.dicom"
            d2 = root / "train" / "bbb.dicom"
            d1.write_bytes(b"DICOM-A")
            d2.write_bytes(b"DICOM-B")
            resolved = {"aaa": d1, "bbb": d2}
            saved = mod.CANONICAL_INPUT_FILES
            mod.CANONICAL_INPUT_FILES = tuple(canon)
            try:
                snap = mod.snapshot_input_hashes(pilot, resolved)
                mod.verify_input_hashes(pilot, resolved, snap)  # clean

                # Tamper EACH canonical input in turn -> must be detected.
                for p in canon:
                    original = p.read_bytes()
                    p.write_bytes(original + b"X")
                    with self.assertRaises(P.Phase2D1BError):
                        mod.verify_input_hashes(pilot, resolved, snap)
                    p.write_bytes(original)  # restore for next iteration
                    mod.verify_input_hashes(pilot, resolved, snap)  # clean again

                # Tamper each DICOM in turn.
                for d in (d1, d2):
                    original = d.read_bytes()
                    d.write_bytes(original + b"X")
                    with self.assertRaises(P.Phase2D1BError):
                        mod.verify_input_hashes(pilot, resolved, snap)
                    d.write_bytes(original)
            finally:
                mod.CANONICAL_INPUT_FILES = saved

    def test_same_basename_different_dirs_not_conflated(self):
        # Two canonical inputs share a basename but live in different dirs.
        mod = load_full()
        pilot = mod.load_pilot()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a").mkdir()
            (root / "b").mkdir()
            p1 = root / "a" / "meta.csv"
            p2 = root / "b" / "meta.csv"
            p1.write_bytes(b"ONE")
            p2.write_bytes(b"TWO")
            saved = mod.CANONICAL_INPUT_FILES
            mod.CANONICAL_INPUT_FILES = (p1, p2)
            try:
                snap = mod.snapshot_input_hashes(pilot, {})
                # Two distinct keys despite identical basename.
                self.assertEqual(len(snap), 2)
                mod.verify_input_hashes(pilot, {}, snap)  # clean
                # Tampering ONLY p2 must be detected (not masked by p1's key).
                p2.write_bytes(b"TWO-TAMPERED")
                with self.assertRaises(P.Phase2D1BError):
                    mod.verify_input_hashes(pilot, {}, snap)
            finally:
                mod.CANONICAL_INPUT_FILES = saved


# =========================================================================== #
# C/D. validate_full_outputs strengthened checks (real tiny JPEGs)              #
# =========================================================================== #
@unittest.skipUnless(HAVE_PIL, "Pillow required")
class TestValidateFullOutputs(unittest.TestCase):
    def setUp(self):
        self.mod = load_full()
        self.pilot = self.mod.load_pilot()
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.master = make_master()
        self.master_path = root / "coco_master.json"
        self.master_path.write_text(json.dumps(self.master), encoding="utf-8")
        self.coco_sha = self.pilot.file_sha256(self.master_path)
        self.staging_train = root / "staging" / "train"
        self.staging_train.mkdir(parents=True)
        self._saved = (self.mod.STAGING_TRAIN, self.mod.COCO_MASTER)
        self.mod.STAGING_TRAIN = self.staging_train
        self.mod.COCO_MASTER = self.master_path
        self.records = []
        for oid in ("aaa", "bbb"):
            out = self.staging_train / f"{oid}.jpg"
            Image.fromarray(np.zeros((8, 8), np.uint8), mode="L").save(
                out, format="JPEG", quality=95, optimize=False, progressive=False)
            self.records.append(self._record(oid, out))
        self.deriv = self.mod.build_coco_jpg_derivative(
            self.pilot, self.master, root / "coco_jpg.json")
        self.ctx = {"coco": self.master, "coco_sha256": self.coco_sha,
                    "meta": {"aaa": {"w": 8, "h": 8}, "bbb": {"w": 8, "h": 8}}}

    def tearDown(self):
        self.mod.STAGING_TRAIN, self.mod.COCO_MASTER = self._saved
        self.tmp.cleanup()

    def _record(self, oid, out):
        return {"image_id": oid, "source_relative_path": f"train/{oid}.dicom",
                "output_relative_path": f"train/{oid}.jpg", "source_dicom_sha256": "d" * 64,
                "pre_jpeg_uint8_sha256": "p" * 64,
                "output_jpeg_sha256": self.pilot.file_sha256(out),
                "decoded_jpeg_uint8_sha256": "j" * 64, "width": 8, "height": 8,
                "jpeg_quality": 95, "decoder_backend": "pydicom_native",
                "modality_branch": "identity", "voi_branch": "windowing",
                "pixel_padding_branch": "none", "presentation_branch": "identity",
                "presentation_inversion_count": 0, "padding_pixel_count": 0,
                "warnings": "", "status": "converted"}

    def test_happy_validation_passes(self):
        out = self.mod.validate_full_outputs(self.pilot, self.ctx, self.deriv, self.records)
        self.assertEqual(out["validated_images"], 2)

    def test_out_of_bounds_bbox_blocks(self):
        # Provide a VALID JPG set + records so execution reaches the bbox check.
        bad_master = copy.deepcopy(self.master)
        bad_master["annotations"][0]["bbox"] = [1, 1, 99, 99]
        bad_deriv = self.mod.build_coco_jpg_derivative(
            self.pilot, bad_master, Path(self.tmp.name) / "bad_jpg.json")
        ctx = dict(self.ctx)
        ctx["coco"] = bad_master
        # Match the master used for record/id checks.
        self.mod.COCO_MASTER = self.master_path
        with self.assertRaises(P.Phase2D1BError):
            self.mod.validate_full_outputs(self.pilot, ctx, bad_deriv, self.records)

    def test_mixed_quality_record_blocks(self):
        self.records[0]["jpeg_quality"] = 100
        with self.assertRaises(P.Phase2D1BError):
            self.mod.validate_full_outputs(self.pilot, self.ctx, self.deriv, self.records)

    def test_missing_jpg_blocks(self):
        (self.staging_train / "bbb.jpg").unlink()
        with self.assertRaises(P.Phase2D1BError):
            self.mod.validate_full_outputs(self.pilot, self.ctx, self.deriv, self.records)

    def test_hash_mismatch_blocks(self):
        # Tamper a JPEG after its record hash was captured.
        with open(self.staging_train / "aaa.jpg", "ab") as fh:
            fh.write(b"x")
        with self.assertRaises(P.Phase2D1BError):
            self.mod.validate_full_outputs(self.pilot, self.ctx, self.deriv, self.records)

    def test_unsafe_output_path_blocks(self):
        self.records[0]["output_relative_path"] = "../escape.jpg"
        with self.assertRaises(P.Phase2D1BError):
            self.mod.validate_full_outputs(self.pilot, self.ctx, self.deriv, self.records)

    def test_derivative_changed_annotation_blocks(self):
        bad = copy.deepcopy(self.deriv)
        bad["annotations"][0]["bbox"] = [0, 0, 2, 2]
        with self.assertRaises(P.Phase2D1BError):
            self.mod.validate_full_outputs(self.pilot, self.ctx, bad, self.records)

    def test_stray_nested_dir_blocks(self):
        (self.staging_train / "sub").mkdir()
        with self.assertRaises(P.Phase2D1BError):
            self.mod.validate_full_outputs(self.pilot, self.ctx, self.deriv, self.records)

    def test_source_hash_drift_blocks(self):
        ctx = dict(self.ctx)
        ctx["coco_sha256"] = "0" * 64  # pretend master changed
        with self.assertRaises(P.Phase2D1BError):
            self.mod.validate_full_outputs(self.pilot, ctx, self.deriv, self.records)


# =========================================================================== #
# E. Promotion transaction + fault injection (dummy files; no image libs)       #
# =========================================================================== #
class TestPromotionTransaction(unittest.TestCase):
    def setUp(self):
        self.mod = load_full()
        self.tmp = tempfile.TemporaryDirectory()
        r = Path(self.tmp.name)
        self._saved = {n: getattr(self.mod, n) for n in
                       ("STAGING_TRAIN", "FINAL_TRAIN", "FINAL_ROOT", "BACKUP_ROOT",
                        "COCO_JPG_FINAL", "COCO_JPG_STAGING", "REPO_ROOT")}
        self.mod.STAGING_ROOT = r / "staging"
        self.mod.STAGING_TRAIN = r / "staging" / "train"
        self.mod.FINAL_ROOT = r / "final"
        self.mod.FINAL_TRAIN = r / "final" / "train"
        self.mod.BACKUP_ROOT = r / "backup"
        self.mod.COCO_JPG_FINAL = r / "final_coco.json"
        self.mod.COCO_JPG_STAGING = r / "staging" / "coco.json"
        self.mod.REPO_ROOT = r
        # Fresh staged outputs.
        self.mod.STAGING_TRAIN.mkdir(parents=True)
        (self.mod.STAGING_TRAIN / "aaa.jpg").write_bytes(b"NEW-A")
        (self.mod.STAGING_TRAIN / "bbb.jpg").write_bytes(b"NEW-B")
        self.mod.COCO_JPG_STAGING.write_bytes(b"NEW-COCO")

    def tearDown(self):
        for n, v in self._saved.items():
            setattr(self.mod, n, v)
        self.tmp.cleanup()

    def _seed_prior_final(self):
        self.mod.FINAL_TRAIN.mkdir(parents=True)
        (self.mod.FINAL_TRAIN / "aaa.jpg").write_bytes(b"PRIOR-A")
        self.mod.COCO_JPG_FINAL.write_bytes(b"PRIOR-COCO")

    def test_happy_promote(self):
        promoted = self.mod.promote_outputs(overwrite=False)
        self.assertEqual((self.mod.FINAL_TRAIN / "aaa.jpg").read_bytes(), b"NEW-A")
        self.assertEqual(self.mod.COCO_JPG_FINAL.read_bytes(), b"NEW-COCO")
        self.assertIn("final_train", promoted)

    def _assert_prior_restored(self):
        self.assertEqual((self.mod.FINAL_TRAIN / "aaa.jpg").read_bytes(), b"PRIOR-A")
        self.assertEqual(self.mod.COCO_JPG_FINAL.read_bytes(), b"PRIOR-COCO")
        # New outputs still in staging; no backup/temp leftovers in final.
        self.assertTrue((self.mod.STAGING_TRAIN / "aaa.jpg").exists()
                        or (self.mod.FINAL_TRAIN / "aaa.jpg").read_bytes() == b"PRIOR-A")

    def test_fault_after_backup_images_rolls_back(self):
        self._seed_prior_final()

        def hook(step):
            if step == "after_backup_images":
                raise RuntimeError("inject")
        with self.assertRaises(P.Phase2D1BError):
            self.mod.promote_outputs(overwrite=True, _fault_hook=hook)
        self._assert_prior_restored()

    def test_fault_after_backup_coco_rolls_back(self):
        self._seed_prior_final()

        def hook(step):
            if step == "after_backup_coco":
                raise RuntimeError("inject")
        with self.assertRaises(P.Phase2D1BError):
            self.mod.promote_outputs(overwrite=True, _fault_hook=hook)
        self._assert_prior_restored()

    def test_fault_after_promote_images_rolls_back(self):
        self._seed_prior_final()

        def hook(step):
            if step == "after_promote_images":
                raise RuntimeError("inject")
        with self.assertRaises(P.Phase2D1BError):
            self.mod.promote_outputs(overwrite=True, _fault_hook=hook)
        self._assert_prior_restored()

    def test_fault_after_promote_coco_rolls_back(self):
        self._seed_prior_final()

        def hook(step):
            if step == "after_promote_coco":
                raise RuntimeError("inject")
        with self.assertRaises(P.Phase2D1BError):
            self.mod.promote_outputs(overwrite=True, _fault_hook=hook)
        self._assert_prior_restored()
        # New COCO restored to staging (available for a retry).
        self.assertTrue(self.mod.COCO_JPG_STAGING.exists())

    def test_restoration_failure_reported(self):
        self._seed_prior_final()
        real_replace = os.replace
        state = {"phase": "forward"}

        def flaky_replace(src, dst):
            # Fail during the ROLLBACK (first replace after the injected fault).
            if state["phase"] == "rollback":
                raise OSError("rollback replace failed")
            return real_replace(src, dst)

        def hook(step):
            if step == "after_promote_coco":
                state["phase"] = "rollback"
                raise RuntimeError("inject")
        saved = self.mod.os.replace
        self.mod.os.replace = flaky_replace
        try:
            with self.assertRaises(P.Phase2D1BError) as cm:
                self.mod.promote_outputs(overwrite=True, _fault_hook=hook)
            self.assertIn("restoration_failed", str(cm.exception))
        finally:
            self.mod.os.replace = saved

    # --- item 4: fail directly AT each forward os.replace (not via after-hook) ---
    def _run_forward_fail(self, target_index):
        self._seed_prior_final()
        real = os.replace
        state = {"n": 0}

        def flaky(src, dst):
            state["n"] += 1
            if state["n"] == target_index:
                raise OSError(f"forced fail at forward replace #{target_index}")
            return real(src, dst)

        saved = self.mod.os.replace
        self.mod.os.replace = flaky
        try:
            with self.assertRaises(P.Phase2D1BError):
                self.mod.promote_outputs(overwrite=True)
        finally:
            self.mod.os.replace = saved

    def _assert_clean_rollback(self):
        # Prior outputs intact.
        self.assertEqual((self.mod.FINAL_TRAIN / "aaa.jpg").read_bytes(), b"PRIOR-A")
        self.assertEqual(self.mod.COCO_JPG_FINAL.read_bytes(), b"PRIOR-COCO")
        # Staged outputs still present (rolled back, available for retry).
        self.assertEqual((self.mod.STAGING_TRAIN / "aaa.jpg").read_bytes(), b"NEW-A")
        self.assertTrue(self.mod.COCO_JPG_STAGING.exists())
        # No leftover backups.
        if self.mod.BACKUP_ROOT.exists():
            self.assertEqual(list(self.mod.BACKUP_ROOT.iterdir()), [])

    def test_forward_fail_prior_image_backup(self):
        self._run_forward_fail(1)   # FINAL_TRAIN -> backup_images
        self._assert_clean_rollback()

    def test_forward_fail_prior_coco_backup(self):
        self._run_forward_fail(2)   # COCO_JPG_FINAL -> backup_coco
        self._assert_clean_rollback()

    def test_forward_fail_new_image_promotion(self):
        self._run_forward_fail(3)   # STAGING_TRAIN -> FINAL_TRAIN
        self._assert_clean_rollback()

    def test_forward_fail_new_coco_promotion(self):
        self._run_forward_fail(4)   # COCO_JPG_STAGING -> COCO_JPG_FINAL
        self._assert_clean_rollback()

    # --- item 1: post-commit (promotion record) failure rolls back dataset ---
    def test_post_commit_failure_rolls_back(self):
        self._seed_prior_final()
        record_written = {"count": 0}

        def post_commit(promoted):
            record_written["count"] += 1
            raise RuntimeError("promotion-record write failed")

        rolled_back = {"count": 0}

        def post_commit_rollback():
            rolled_back["count"] += 1

        with self.assertRaises(P.Phase2D1BError):
            self.mod.promote_outputs(overwrite=True, _post_commit=post_commit,
                                     _post_commit_rollback=post_commit_rollback)
        self.assertEqual(record_written["count"], 1)
        self.assertEqual(rolled_back["count"], 1)
        # Dataset rolled back to prior; success not claimed.
        self.assertEqual((self.mod.FINAL_TRAIN / "aaa.jpg").read_bytes(), b"PRIOR-A")
        self.assertEqual(self.mod.COCO_JPG_FINAL.read_bytes(), b"PRIOR-COCO")

    # --- item 8: backup cleanup failure is surfaced, not silenced ---
    def test_backup_cleanup_failure_reported(self):
        self._seed_prior_final()
        import shutil as _shutil
        saved = self.mod.shutil.rmtree

        def boom(path, *a, **k):
            raise OSError("cannot remove backup")

        self.mod.shutil.rmtree = boom
        try:
            promoted = self.mod.promote_outputs(overwrite=True)
        finally:
            self.mod.shutil.rmtree = saved
        # Promotion still succeeded (dataset committed) but warning surfaced.
        self.assertEqual((self.mod.FINAL_TRAIN / "aaa.jpg").read_bytes(), b"NEW-A")
        self.assertTrue(promoted["backup_cleanup_warnings"])

    # --- item 2: dataset rollback failure must STILL run record cleanup ---
    def test_dataset_rollback_failure_still_runs_record_cleanup(self):
        self._seed_prior_final()
        state = {"phase": "forward", "pcr": 0}
        real = os.replace

        def flaky(src, dst):
            if state["phase"] == "rollback":
                raise OSError("journal rollback replace failed")
            return real(src, dst)

        def post_commit(promoted):
            state["phase"] = "rollback"
            raise RuntimeError("post-commit fail")

        def post_commit_rollback():
            state["pcr"] += 1  # record cleanup must still be attempted

        saved = self.mod.os.replace
        self.mod.os.replace = flaky
        try:
            with self.assertRaises(P.Phase2D1BError) as cm:
                self.mod.promote_outputs(overwrite=True, _post_commit=post_commit,
                                         _post_commit_rollback=post_commit_rollback)
            self.assertIn("restoration_failed", str(cm.exception))
            self.assertEqual(state["pcr"], 1)  # ran despite journal failure
        finally:
            self.mod.os.replace = saved

    # --- item 2: both rollbacks failing are aggregated ---
    def test_both_rollbacks_fail_aggregated(self):
        self._seed_prior_final()
        state = {"phase": "forward"}
        real = os.replace

        def flaky(src, dst):
            if state["phase"] == "rollback":
                raise OSError("journal rollback failed")
            return real(src, dst)

        def post_commit(promoted):
            state["phase"] = "rollback"
            raise RuntimeError("post-commit fail")

        def post_commit_rollback():
            raise OSError("record rollback failed")

        saved = self.mod.os.replace
        self.mod.os.replace = flaky
        try:
            with self.assertRaises(P.Phase2D1BError) as cm:
                self.mod.promote_outputs(overwrite=True, _post_commit=post_commit,
                                         _post_commit_rollback=post_commit_rollback)
            msg = str(cm.exception)
            self.assertIn("restoration_failed", msg)
            self.assertIn("journal:", msg)
            self.assertIn("post_commit_rollback:", msg)
        finally:
            self.mod.os.replace = saved

    # --- item 3: forward failure (steps 1-4) with a PRIOR promotion record must
    # leave that record byte-for-byte intact, using the REAL transaction. ---
    def _forward_fail_with_real_txn(self, target_index):
        self._seed_prior_final()
        r = Path(self.tmp.name)
        record = r / "phase2D1B_full_promotion.json"
        backup = r / "phase2D1B_full_promotion.json.prior"
        record.write_bytes(b"PRIOR-RECORD")
        pilot = self.mod.load_pilot()
        txn = self.mod.PromotionRecordTransaction(pilot, {}, record, backup)
        real = os.replace
        state = {"n": 0}

        def flaky(src, dst):
            state["n"] += 1
            if state["n"] == target_index:
                raise OSError(f"forced fail at forward replace #{target_index}")
            return real(src, dst)

        saved = self.mod.os.replace
        self.mod.os.replace = flaky
        try:
            with self.assertRaises(P.Phase2D1BError):
                self.mod.promote_outputs(overwrite=True, _post_commit=txn.finalize,
                                         _post_commit_rollback=txn.rollback)
        finally:
            self.mod.os.replace = saved
        # finalize never ran (failure was in dataset steps 1-4) -> record intact.
        self.assertFalse(txn.started)
        self.assertEqual(record.read_bytes(), b"PRIOR-RECORD")

    def test_forward_fail_preserves_prior_record_step1(self):
        self._forward_fail_with_real_txn(1)

    def test_forward_fail_preserves_prior_record_step2(self):
        self._forward_fail_with_real_txn(2)

    def test_forward_fail_preserves_prior_record_step3(self):
        self._forward_fail_with_real_txn(3)

    def test_forward_fail_preserves_prior_record_step4(self):
        self._forward_fail_with_real_txn(4)


# =========================================================================== #
# E. run_full: validation/report failure must NOT promote (synthetic run)       #
# =========================================================================== #
@unittest.skipUnless(HAVE_YAML and HAVE_PYDICOM and HAVE_PIL and HAVE_SKIMAGE,
                     "pydicom + Pillow + skimage + PyYAML required")
class TestRunFullNoPromoteOnFailure(unittest.TestCase):
    def setUp(self):
        self.mod = load_full()
        self.tmp = tempfile.TemporaryDirectory()
        self.scope = Scope(self.mod, self.tmp.name, real_dicom=True)
        os.environ["VINBIGDATA_DICOM_ROOT"] = str(self.scope.dicom_root)

    def tearDown(self):
        os.environ.pop("VINBIGDATA_DICOM_ROOT", None)
        self.scope.restore()
        self.tmp.cleanup()

    def _args(self, overwrite=False):
        return self.mod.Args(preflight_only=False, execute_full=True,
                             acknowledge_full_scope=2, jpeg_quality=95,
                             dicom_root=None, jpeg2000_decoder="pylibjpeg",
                             overwrite=overwrite)

    def test_validation_failure_no_promote(self):
        orig = self.mod.validate_full_outputs
        self.mod.validate_full_outputs = lambda *a, **k: (_ for _ in ()).throw(
            self.mod.FullError("validation boom"))
        try:
            with self.assertRaises(P.Phase2D1BError):
                self.mod.run_full(self._args())
        finally:
            self.mod.validate_full_outputs = orig
        self.assertFalse(self.mod.FINAL_TRAIN.exists()
                         and any(self.mod.FINAL_TRAIN.iterdir()))
        self.assertFalse(self.mod.COCO_JPG_FINAL.exists())

    def test_report_write_failure_no_promote(self):
        orig = self.mod.write_full_evidence
        self.mod.write_full_evidence = lambda *a, **k: (_ for _ in ()).throw(
            self.mod.FullError("report boom"))
        try:
            with self.assertRaises(P.Phase2D1BError):
                self.mod.run_full(self._args())
        finally:
            self.mod.write_full_evidence = orig
        self.assertFalse(self.mod.FINAL_TRAIN.exists()
                         and any(self.mod.FINAL_TRAIN.iterdir()))
        self.assertFalse(self.mod.COCO_JPG_FINAL.exists())

    def test_full_run_success_flags(self):
        # Snapshot EVERY source DICOM + the COCO master before the run.
        dicom_paths = {oid: self.scope.dicom_root / "train" / f"{oid}.dicom"
                       for oid in ("aaa", "bbb")}
        before = {oid: self.scope.pilot.file_sha256(p) for oid, p in dicom_paths.items()}
        master_before = self.scope.pilot.file_sha256(self.scope.master_path)

        rc = self.mod.run_full(self._args())
        self.assertEqual(rc, 0)
        self.assertEqual(len(list(self.mod.FINAL_TRAIN.glob("*.jpg"))), 2)
        self.assertTrue(self.mod.COCO_JPG_FINAL.exists())
        val = json.loads((self.mod.REPORTS_DIR /
                          "phase2D1B_full_validation.json").read_text(encoding="utf-8"))
        self.assertIs(val["full_conversion_completed"], False)
        self.assertIs(val["full_execution_finished"], True)
        self.assertIs(val["full_validation_candidate_pass"], True)
        self.assertIs(val["dataset_training_ready"], False)
        self.assertIs(val["training_authorized"], False)
        self.assertEqual(val["phase_status"], "OPEN")
        # Every source DICOM AND the COCO master are byte-identical after the run.
        for oid, p in dicom_paths.items():
            self.assertEqual(self.scope.pilot.file_sha256(p), before[oid], oid)
        self.assertEqual(self.scope.pilot.file_sha256(self.scope.master_path),
                         master_before)
        # Post-promotion record still does NOT claim completion.
        prom = json.loads((self.mod.REPORTS_DIR /
                           "phase2D1B_full_promotion.json").read_text(encoding="utf-8"))
        self.assertIs(prom["full_conversion_completed"], False)

    def test_promotion_record_failure_no_final(self):
        # A promotion-record write failure must roll back the freshly promoted
        # images + COCO (no prior final here) so nothing is left committed.
        orig = self.mod.write_promotion_record
        self.mod.write_promotion_record = lambda *a, **k: (_ for _ in ()).throw(
            self.mod.FullError("promotion record boom"))
        try:
            with self.assertRaises(P.Phase2D1BError):
                self.mod.run_full(self._args())
        finally:
            self.mod.write_promotion_record = orig
        self.assertFalse(self.mod.FINAL_TRAIN.exists()
                         and any(self.mod.FINAL_TRAIN.iterdir()))
        self.assertFalse(self.mod.COCO_JPG_FINAL.exists())
        self.assertFalse((self.mod.REPORTS_DIR /
                          "phase2D1B_full_promotion.json").exists())

    def test_prior_promotion_record_restored_byte_for_byte(self):
        # A prior promotion record must be preserved and restored on rollback.
        prior_bytes = b'{"prior":"record","full_conversion_completed":false}\n'
        record = self.mod.REPORTS_DIR / "phase2D1B_full_promotion.json"
        record.write_bytes(prior_bytes)
        orig = self.mod.write_promotion_record
        self.mod.write_promotion_record = lambda *a, **k: (_ for _ in ()).throw(
            self.mod.FullError("record boom"))
        try:
            with self.assertRaises(P.Phase2D1BError):
                self.mod.run_full(self._args())
        finally:
            self.mod.write_promotion_record = orig
        # Prior record restored byte-for-byte; no dataset committed.
        self.assertEqual(record.read_bytes(), prior_bytes)
        self.assertFalse((self.mod.REPORTS_DIR /
                          "phase2D1B_full_promotion.json.prior").exists())
        self.assertFalse(self.mod.FINAL_TRAIN.exists()
                         and any(self.mod.FINAL_TRAIN.iterdir()))
        self.assertFalse(self.mod.COCO_JPG_FINAL.exists())

    def test_cleanup_warning_in_structured_evidence(self):
        # Seed a prior final so promotion creates backups, then fail cleanup.
        self.mod.FINAL_TRAIN.mkdir(parents=True, exist_ok=True)
        (self.mod.FINAL_TRAIN / "old.jpg").write_bytes(b"OLD")
        self.mod.COCO_JPG_FINAL.write_bytes(b"OLD-COCO")
        saved = self.mod.shutil.rmtree
        self.mod.shutil.rmtree = lambda *a, **k: (_ for _ in ()).throw(
            OSError("cannot remove backup"))
        try:
            rc = self.mod.run_full(self._args(overwrite=True))
        finally:
            self.mod.shutil.rmtree = saved
        self.assertEqual(rc, 0)
        audit = json.loads((self.mod.REPORTS_DIR /
                            "phase2D1B_full_cleanup_audit.json").read_text(encoding="utf-8"))
        self.assertTrue(audit["backup_cleanup_warnings"])
        self.assertIs(audit["backup_cleanup_clean"], False)
        self.assertIs(audit["full_conversion_completed"], False)

    def test_stale_prior_backup_hard_fails_with_current_record(self):
        (self.mod.REPORTS_DIR / "phase2D1B_full_promotion.json").write_bytes(b"CUR")
        stale = self.mod.REPORTS_DIR / "phase2D1B_full_promotion.json.prior"
        stale.write_bytes(b"STALE-RECOVERY")
        with self.assertRaises(P.Phase2D1BError) as cm:
            self.mod.run_full(self._args())
        self.assertIn("stale_prior_promotion_record_backup_present", str(cm.exception))
        self.assertEqual(stale.read_bytes(), b"STALE-RECOVERY")  # not overwritten
        self.assertFalse(self.mod.FINAL_TRAIN.exists()
                         and any(self.mod.FINAL_TRAIN.iterdir()))
        self.assertFalse(self.mod.COCO_JPG_FINAL.exists())

    def test_stale_prior_backup_hard_fails_without_current_record(self):
        stale = self.mod.REPORTS_DIR / "phase2D1B_full_promotion.json.prior"
        stale.write_bytes(b"STALE-RECOVERY")
        with self.assertRaises(P.Phase2D1BError) as cm:
            self.mod.run_full(self._args())
        self.assertIn("stale_prior_promotion_record_backup_present", str(cm.exception))
        self.assertEqual(stale.read_bytes(), b"STALE-RECOVERY")

    def test_cleanup_audit_write_failure_reports_committed(self):
        orig = self.mod.write_cleanup_audit
        self.mod.write_cleanup_audit = lambda *a, **k: (_ for _ in ()).throw(
            self.mod.FullError("cleanup audit boom"))
        try:
            with self.assertRaises(P.Phase2D1BError) as cm:
                self.mod.run_full(self._args())
        finally:
            self.mod.write_cleanup_audit = orig
        self.assertIn("promotion_committed_but_cleanup_audit_write_failed",
                      str(cm.exception))
        # Dataset IS committed and NOT rolled back.
        self.assertEqual(len(list(self.mod.FINAL_TRAIN.glob("*.jpg"))), 2)
        self.assertTrue(self.mod.COCO_JPG_FINAL.exists())

    def test_record_backup_cleanup_failure_in_audit_clean_false(self):
        # A prior-record backup that cannot be removed must appear in the audit
        # with backup_cleanup_clean=false (not only logged).
        saved = self.mod.PromotionRecordTransaction.commit
        self.mod.PromotionRecordTransaction.commit = \
            lambda self: "promotion_record_backup_cleanup_failed:injected"
        try:
            rc = self.mod.run_full(self._args())
        finally:
            self.mod.PromotionRecordTransaction.commit = saved
        self.assertEqual(rc, 0)
        audit = json.loads((self.mod.REPORTS_DIR /
                            "phase2D1B_full_cleanup_audit.json").read_text(encoding="utf-8"))
        self.assertIn("promotion_record_backup_cleanup_failed:injected",
                      audit["backup_cleanup_warnings"])
        self.assertIs(audit["backup_cleanup_clean"], False)

    def test_failure_before_post_commit_preserves_prior_record(self):
        # A dataset-promotion failure (before _post_commit) must leave a prior
        # promotion record byte-for-byte intact via the REAL transaction.
        record = self.mod.REPORTS_DIR / "phase2D1B_full_promotion.json"
        record.write_bytes(b"PRIOR-RECORD-BYTES")
        real = os.replace

        def flaky(src, dst):
            if Path(dst) == self.mod.FINAL_TRAIN:  # image promotion step
                raise OSError("image promotion failed")
            return real(src, dst)

        saved = self.mod.os.replace
        self.mod.os.replace = flaky
        try:
            with self.assertRaises(P.Phase2D1BError):
                self.mod.run_full(self._args())
        finally:
            self.mod.os.replace = saved
        # Prior record untouched; no dataset committed.
        self.assertEqual(record.read_bytes(), b"PRIOR-RECORD-BYTES")
        self.assertFalse(self.mod.FINAL_TRAIN.exists()
                         and any(self.mod.FINAL_TRAIN.iterdir()))
        self.assertFalse(self.mod.COCO_JPG_FINAL.exists())

    def test_writer_creates_record_then_raises_no_prior(self):
        # Writer creates/replaces record_path and THEN raises, with no prior
        # record: rollback must remove the new/partial record and leave no .prior;
        # images + COCO must be rolled back.
        record = self.mod.REPORTS_DIR / "phase2D1B_full_promotion.json"
        prior_backup = self.mod.REPORTS_DIR / "phase2D1B_full_promotion.json.prior"

        def writer_then_raise(pilot, ctx, promoted):
            record.write_bytes(b"NEW-PARTIAL")
            raise self.mod.FullError("writer wrote then raised")

        orig = self.mod.write_promotion_record
        self.mod.write_promotion_record = writer_then_raise
        try:
            with self.assertRaises(P.Phase2D1BError):
                self.mod.run_full(self._args())
        finally:
            self.mod.write_promotion_record = orig
        self.assertFalse(record.exists())        # new/partial record removed
        self.assertFalse(prior_backup.exists())  # no .prior left
        self.assertFalse(self.mod.FINAL_TRAIN.exists()
                         and any(self.mod.FINAL_TRAIN.iterdir()))
        self.assertFalse(self.mod.COCO_JPG_FINAL.exists())

    def test_writer_creates_record_then_raises_with_prior(self):
        # Same, but a prior record exists: it must be restored byte-for-byte and
        # no .prior recovery artefact should remain.
        record = self.mod.REPORTS_DIR / "phase2D1B_full_promotion.json"
        prior_backup = self.mod.REPORTS_DIR / "phase2D1B_full_promotion.json.prior"
        record.write_bytes(b"PRIOR-RECORD-BYTES")

        def writer_then_raise(pilot, ctx, promoted):
            record.write_bytes(b"NEW-PARTIAL")
            raise self.mod.FullError("writer wrote then raised")

        orig = self.mod.write_promotion_record
        self.mod.write_promotion_record = writer_then_raise
        try:
            with self.assertRaises(P.Phase2D1BError):
                self.mod.run_full(self._args())
        finally:
            self.mod.write_promotion_record = orig
        self.assertEqual(record.read_bytes(), b"PRIOR-RECORD-BYTES")  # restored
        self.assertFalse(prior_backup.exists())  # no .prior left
        self.assertFalse(self.mod.FINAL_TRAIN.exists()
                         and any(self.mod.FINAL_TRAIN.iterdir()))
        self.assertFalse(self.mod.COCO_JPG_FINAL.exists())


# =========================================================================== #
# Item 5: mapping CSV/JSONL validated before promotion (reads real artefacts)    #
# =========================================================================== #
class TestMappingValidation(unittest.TestCase):
    PROTO = "s" * 64
    DEC = "x" * 64

    def _records(self):
        recs = []
        for oid in ("aaa", "bbb"):
            recs.append({
                "image_id": oid, "source_relative_path": f"train/{oid}.dicom",
                "output_relative_path": f"train/{oid}.jpg",
                "source_dicom_sha256": ("a" if oid == "aaa" else "b") * 64,
                "pre_jpeg_uint8_sha256": "p" * 64,
                "output_jpeg_sha256": ("o" if oid == "aaa" else "q") * 64,
                "decoded_jpeg_uint8_sha256": "j" * 64, "width": 8, "height": 8,
                "jpeg_quality": 95, "decoder_backend": "pydicom_native",
                "modality_branch": "identity", "voi_branch": "windowing",
                "pixel_padding_branch": "none", "presentation_branch": "identity",
                "presentation_inversion_count": 0, "padding_pixel_count": 0,
                "warnings": "", "status": "converted"})
        return recs

    def _build(self, td, recs=None):
        mod = load_full()
        pilot = mod.load_pilot()
        recs = recs or self._records()
        mod.build_full_mapping(pilot, recs, self.PROTO, self.DEC, Path(td))
        ctx = {"protocol_evidence": {"protocol_sha256": self.PROTO},
               "decision_sha256": self.DEC}
        return mod, pilot, ctx, recs

    def test_valid_mapping_passes(self):
        with tempfile.TemporaryDirectory() as td:
            mod, pilot, ctx, recs = self._build(td)
            out = mod.validate_mapping_artifacts(pilot, ctx, recs, Path(td))
            self.assertEqual(out["mapping_rows"], 2)

    def test_protocol_hash_drift_fails(self):
        with tempfile.TemporaryDirectory() as td:
            mod, pilot, ctx, recs = self._build(td)
            ctx["protocol_evidence"]["protocol_sha256"] = "z" * 64
            with self.assertRaises(P.Phase2D1BError):
                mod.validate_mapping_artifacts(pilot, ctx, recs, Path(td))

    def test_decision_hash_drift_fails(self):
        with tempfile.TemporaryDirectory() as td:
            mod, pilot, ctx, recs = self._build(td)
            ctx["decision_sha256"] = "z" * 64
            with self.assertRaises(P.Phase2D1BError):
                mod.validate_mapping_artifacts(pilot, ctx, recs, Path(td))

    def test_output_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as td:
            mod, pilot, ctx, recs = self._build(td)
            bad = copy.deepcopy(recs)
            bad[0]["output_jpeg_sha256"] = "0" * 64  # differs from written CSV
            with self.assertRaises(P.Phase2D1BError):
                mod.validate_mapping_artifacts(pilot, ctx, bad, Path(td))

    def test_id_set_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as td:
            mod, pilot, ctx, recs = self._build(td)
            extra = copy.deepcopy(recs) + [dict(recs[0], image_id="ccc")]
            with self.assertRaises(P.Phase2D1BError):
                mod.validate_mapping_artifacts(pilot, ctx, extra, Path(td))

    def test_csv_jsonl_inconsistent_fails(self):
        with tempfile.TemporaryDirectory() as td:
            mod, pilot, ctx, recs = self._build(td)
            jsonl = Path(td) / "phase2D1B_full_mapping.jsonl"
            lines = jsonl.read_text(encoding="utf-8").splitlines()
            obj = json.loads(lines[0])
            obj["image_id"] = "TAMPERED"
            lines[0] = json.dumps(obj)
            jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(P.Phase2D1BError):
                mod.validate_mapping_artifacts(pilot, ctx, recs, Path(td))


# =========================================================================== #
# Item 3: convert_one_image real fault injection (encode / decode failures)      #
# =========================================================================== #
@unittest.skipUnless(HAVE_PYDICOM and HAVE_PIL, "pydicom + Pillow required")
class TestConvertFaultInjection(unittest.TestCase):
    def setUp(self):
        self.mod = load_full()
        self.pilot = self.mod.load_pilot()
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.dpath = root / "aaa.dicom"
        _make_real_dicom(self.dpath)
        self.staging_train = root / "staging" / "train"
        self.staging_train.mkdir(parents=True)
        self.coco_image = {"id": 1, "original_image_id": "aaa",
                           "file_name": "train/aaa.dicom", "width": 8, "height": 8}
        self.meta = {"w": 8, "h": 8}

    def tearDown(self):
        self.tmp.cleanup()

    def _convert(self):
        return self.mod.convert_one_image(
            self.pilot, "aaa", self.dpath, self.coco_image, self.meta,
            self.staging_train, "pylibjpeg")

    def _assert_clean_staging(self):
        self.assertEqual(list(self.staging_train.glob("*.part")), [])
        self.assertEqual(list(self.staging_train.glob("*.jpg")), [])

    def test_encode_failure_propagates_and_cleans(self):
        from PIL import Image
        saved = Image.Image.save

        def boom(self, *a, **k):
            raise OSError("encode failed")

        Image.Image.save = boom
        try:
            with self.assertRaises(Exception):
                self._convert()
        finally:
            Image.Image.save = saved
        self._assert_clean_staging()

    def test_decode_validation_failure_propagates_and_cleans(self):
        from PIL import Image
        saved_open = Image.open

        def boom_open(*a, **k):
            raise OSError("decode failed")

        Image.open = boom_open
        try:
            with self.assertRaises(Exception):
                self._convert()
        finally:
            Image.open = saved_open
        # Final-in-staging JPG must be removed on post-encode failure.
        self._assert_clean_staging()


# =========================================================================== #
# Item 1/2: PromotionRecordTransaction rollback + stale-backup guard (REAL code) #
# =========================================================================== #
class TestPromotionRecordTransaction(unittest.TestCase):
    def _txn(self, td, unlink=None, replace=None, seed_record=None, seed_backup=None):
        mod = load_full()
        pilot = mod.load_pilot()
        rec = Path(td) / "phase2D1B_full_promotion.json"
        bak = Path(td) / "phase2D1B_full_promotion.json.prior"
        if seed_record is not None:
            rec.write_bytes(seed_record)
        if seed_backup is not None:
            bak.write_bytes(seed_backup)
        t = mod.PromotionRecordTransaction(pilot, {}, rec, bak,
                                           _unlink=unlink, _replace=replace)
        return mod, t, rec, bak

    def test_unlink_fail_restore_still_attempted(self):
        with tempfile.TemporaryDirectory() as td:
            def bad_unlink(p):
                raise OSError("unlink failed")
            mod, t, rec, bak = self._txn(td, unlink=bad_unlink,
                                         seed_record=b"NEW", seed_backup=b"PRIOR")
            t.started = True
            t.backed_up = True
            with self.assertRaises(P.Phase2D1BError) as cm:
                t.rollback()
            self.assertIn("record_unlink", str(cm.exception))
            # Restore was still attempted (real replace) -> prior byte-for-byte.
            self.assertEqual(rec.read_bytes(), b"PRIOR")

    def test_unlink_and_restore_both_fail_aggregated(self):
        with tempfile.TemporaryDirectory() as td:
            def bad_unlink(p):
                raise OSError("unlink failed")

            def bad_replace(a, b):
                raise OSError("replace failed")
            mod, t, rec, bak = self._txn(td, unlink=bad_unlink, replace=bad_replace,
                                         seed_record=b"NEW", seed_backup=b"PRIOR")
            t.started = True
            t.backed_up = True
            with self.assertRaises(P.Phase2D1BError) as cm:
                t.rollback()
            msg = str(cm.exception)
            self.assertIn("record_unlink", msg)
            self.assertIn("prior_restore", msg)

    def test_rollback_before_finalize_is_noop(self):
        # finalize never ran -> a valid current record must NOT be deleted.
        with tempfile.TemporaryDirectory() as td:
            mod, t, rec, bak = self._txn(td, seed_record=b"CURRENT-VALID")
            self.assertFalse(t.started)
            t.rollback()  # no-op, no raise
            self.assertEqual(rec.read_bytes(), b"CURRENT-VALID")  # untouched

    def test_backup_replace_failure_does_not_delete_current_record(self):
        # If the prior-record backup replace fails inside finalize, the current
        # record must remain intact (rollback must not unlink it).
        with tempfile.TemporaryDirectory() as td:
            def bad_replace(a, b):
                raise OSError("backup replace failed")
            mod, t, rec, bak = self._txn(td, replace=bad_replace,
                                         seed_record=b"CURRENT-VALID")
            with self.assertRaises(Exception):
                t.finalize({"x": 1})  # backup replace raises
            self.assertTrue(t.started)
            self.assertFalse(t.backed_up)
            self.assertFalse(t.wrote_new)
            t.rollback()  # must be a no-op for the current record
            self.assertEqual(rec.read_bytes(), b"CURRENT-VALID")  # not deleted

    def test_commit_backup_cleanup_failure_returns_warning(self):
        with tempfile.TemporaryDirectory() as td:
            def bad_unlink(p):
                raise OSError("cannot remove backup")
            mod, t, rec, bak = self._txn(td, unlink=bad_unlink, seed_backup=b"PRIOR")
            t.backed_up = True
            warning = t.commit()
            self.assertIsNotNone(warning)
            self.assertIn("promotion_record_backup_cleanup_failed", warning)

    def test_stale_backup_guard_with_current_record(self):
        with tempfile.TemporaryDirectory() as td:
            mod, t, rec, bak = self._txn(td, seed_record=b"CUR", seed_backup=b"STALE")
            with self.assertRaises(P.Phase2D1BError):
                t.check_no_stale_backup()
            self.assertEqual(bak.read_bytes(), b"STALE")  # not overwritten

    def test_stale_backup_guard_without_current_record(self):
        with tempfile.TemporaryDirectory() as td:
            mod, t, rec, bak = self._txn(td, seed_backup=b"STALE")  # no current record
            with self.assertRaises(P.Phase2D1BError):
                t.check_no_stale_backup()


# =========================================================================== #
# F. CLI mode / quality / acknowledgement gates                                 #
# =========================================================================== #
class TestFullCliGates(unittest.TestCase):
    def test_no_mode_shows_help_nonzero(self):
        self.assertEqual(load_full().main([]), 2)

    def test_both_modes_nonzero(self):
        self.assertEqual(load_full().main(
            ["--preflight-only", "--execute-full",
             "--acknowledge-full-scope", "4894", "--jpeg-quality", "95"]), 2)

    def test_wrong_quality_hard_fail(self):
        self.assertEqual(load_full().main(
            ["--execute-full", "--acknowledge-full-scope", "4894",
             "--jpeg-quality", "100"]), 1)

    def test_wrong_scope_hard_fail(self):
        self.assertEqual(load_full().main(
            ["--execute-full", "--acknowledge-full-scope", "999",
             "--jpeg-quality", "95"]), 1)

    def test_locked_constants(self):
        mod = load_full()
        self.assertEqual(mod.LOCKED_QUALITY, 95)
        self.assertEqual(mod.LOCKED_SCOPE, 4894)


# =========================================================================== #
# Mapping schema / status invariants                                            #
# =========================================================================== #
class TestFullMappingAndStatus(unittest.TestCase):
    def test_mapping_fields(self):
        mod = load_full()
        for col in ("image_id", "output_jpeg_sha256", "source_dicom_sha256",
                    "protocol_sha256", "decision_sha256", "jpeg_quality",
                    "decoder_backend", "status"):
            self.assertIn(col, mod.MAPPING_FIELDS)

    def test_base_status(self):
        mod = load_full()
        s = mod._base_full_status({})
        self.assertEqual(s["phase_status"], "OPEN")
        self.assertIs(s["full_conversion_completed"], False)
        self.assertIs(s["full_execution_finished"], False)
        self.assertIs(s["full_validation_candidate_pass"], False)
        self.assertIs(s["dataset_training_ready"], False)
        self.assertIs(s["training_authorized"], False)


# =========================================================================== #
# G. Prohibited-action AST/source guardrails (supplementary to behavioral)      #
# =========================================================================== #
class TestFullSourceGuardrails(unittest.TestCase):
    def setUp(self):
        self.src = FULL_PATH.read_text(encoding="utf-8")
        self.tree = ast.parse(self.src)

    def _imports(self):
        names = set()
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    names.add(a.name.split(".")[0])
            elif isinstance(n, ast.ImportFrom):
                if n.module:
                    names.add(n.module.split(".")[0])
        return names

    def test_no_training_framework_imports(self):
        self.assertEqual(self._imports() &
                         {"mmdet", "mmcv", "mmengine", "torch", "torchvision", "detectron2"},
                         set())

    def test_no_geometry_transform_calls(self):
        banned = {"resize", "rotate", "transpose", "thumbnail", "flip"}
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                self.assertNotIn(n.func.attr, banned)

    def test_no_force_true(self):
        for n in ast.walk(self.tree):
            if isinstance(n, ast.keyword) and n.arg == "force":
                self.assertFalse(isinstance(n.value, ast.Constant)
                                 and n.value.value is True)

    def test_only_quality_95(self):
        self.assertNotIn("quality=100", self.src)
        self.assertIn("quality=LOCKED_QUALITY", self.src)

    def test_never_readiness_or_completed_true(self):
        for bad in ('dataset_training_ready": True', 'training_authorized": True',
                    'full_conversion_completed": True',
                    "dataset_training_ready = True", "training_authorized = True"):
            self.assertNotIn(bad, self.src)

    def test_no_split_ap_pseudo(self):
        low = self.src.lower()
        for bad in ("train_test_split", "labeled_unlabeled", "compute_map",
                    "average_precision", "pseudo_label"):
            self.assertNotIn(bad, low)

    def test_no_superiority_claim(self):
        low = self.src.lower()
        self.assertNotIn("q95 better than q100", low)
        self.assertNotIn("quality 95 better", low)
        self.assertIn("downstream_detector_superiority_claimed", self.src)

    def test_master_source_read_only(self):
        for ln in self.src.splitlines():
            if "COCO_MASTER" in ln and "open(" in ln:
                self.assertIn('"r"', ln)
        self.assertNotIn('.dicom", "w"', self.src)
        self.assertNotIn('.dicom", "a"', self.src)


# =========================================================================== #
# Pilot guardrail tests must not be weakened (markers only; NOT a suite gate)    #
# =========================================================================== #
class TestPilotTestsNotWeakened(unittest.TestCase):
    def test_pilot_markers_present(self):
        # The official pilot regression gate is running the pilot test file
        # itself (done by the user after GPT review); this only asserts the file
        # was not emptied or stripped of its key guardrails.
        src = PILOT_TEST_PATH.read_text(encoding="utf-8")
        for marker in ("no_finding_strata_diverse", "check_visual_coverage",
                       "TestPresentationGapEnforcement", "def test_"):
            self.assertIn(marker, src, marker)


if __name__ == "__main__":
    unittest.main(verbosity=2)
