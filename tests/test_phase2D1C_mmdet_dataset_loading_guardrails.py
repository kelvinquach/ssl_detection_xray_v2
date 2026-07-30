#!/usr/bin/env python3
"""Phase 2D.1C guardrail tests (pure helpers + AST safety + CLI).

These tests exercise the pure, framework-free parts of the validation script:
COCO structural analysis, count validation, SHA-256, category->label mapping,
bbox/label validation, empty-bbox shape checks, deterministic empty-sample
selection, report schema, atomic report writing, errors-CSV-header-when-empty,
and AST guards proving there is no training / no source mutation.

They do NOT load MMDetection and do NOT decode the 7.1 GB of JPGs, so they run
anywhere. The real integration validation (the CocoDataset build, pipeline and
dataloader) must still be executed on the full dataset in the mmdet330 env; that
run is intentionally not mocked.

Run::

    /content/miniconda/envs/mmdet330/bin/python -m pytest \
      tests/test_phase2D1C_mmdet_dataset_loading_guardrails.py -q -rs
"""
from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "02D1C_validate_mmdet_dataset_loading.py"


def load_script():
    """Import the validation script as a module (no side effects on import)."""
    spec = importlib.util.spec_from_file_location("phase2d1c_mod", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = load_script()


# --------------------------------------------------------------------------- #
# Synthetic COCO fixtures (tiny; mirror the real schema shape).                #
# --------------------------------------------------------------------------- #
def make_coco(num_images=6, num_empty=2, num_cats=14):
    """Build a tiny valid COCO dict with a controllable number of empty images."""
    categories = [
        {"id": i + 1, "name": f"class_{i}", "canonical_class_id": i}
        for i in range(num_cats)
    ]
    images = [
        {"id": i + 1, "file_name": f"train/img_{i + 1}.jpg",
         "width": 100 + i, "height": 120 + i}
        for i in range(num_images)
    ]
    annotations = []
    ann_id = 1
    nonempty = num_images - num_empty
    for img in images[:nonempty]:
        annotations.append({
            "id": ann_id, "image_id": img["id"], "category_id": 1,
            "bbox": [10.0, 10.0, 20.0, 30.0], "area": 600.0, "iscrowd": 0,
        })
        ann_id += 1
    return {"images": images, "annotations": annotations,
            "categories": categories}


# --------------------------------------------------------------------------- #
# MMEngine serialized-data regression guard.                                   #
# --------------------------------------------------------------------------- #
def test_dataset_image_ids_uses_public_api_when_data_list_is_cleared():
    """IDs remain accessible when MMEngine serializes and clears data_list."""

    class SerializedDatasetStub:
        def __init__(self):
            # Mirrors MMEngine after serialize_data=True initialization:
            # the ordinary data_list is cleared, but public indexed access works.
            self.data_list = []
            self._records = [
                {"img_id": 101},
                {"image_id": 205},
                {"img_id": 309},
            ]
            self.requested_indices = []

        def __len__(self):
            return len(self._records)

        def get_data_info(self, index):
            self.requested_indices.append(index)
            return self._records[index]

    dataset = SerializedDatasetStub()

    assert len(dataset) == 3
    assert dataset.data_list == []

    observed = M.dataset_image_ids_in_order(dataset)

    assert observed == [101, 205, 309]
    assert dataset.requested_indices == [0, 1, 2]


# --------------------------------------------------------------------------- #
# CLI.                                                                         #
# --------------------------------------------------------------------------- #
def test_cli_help_runs():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert "--filter" not in result.stdout  # no accidental filter flag
    for flag in ("--repo-root", "--ann-file", "--data-root", "--batch-size",
                 "--num-workers", "--seed", "--expected-images",
                 "--expected-annotations", "--expected-categories",
                 "--expected-empty-images", "--strict"):
        assert flag in result.stdout, flag


def test_arg_parser_defaults():
    parser = M.build_arg_parser()
    args = parser.parse_args([])
    assert args.expected_images == 4894
    assert args.expected_annotations == 36096
    assert args.expected_categories == 14
    assert args.expected_empty_images == 500
    assert args.seed == 42
    assert args.strict is False


# --------------------------------------------------------------------------- #
# Path resolution.                                                             #
# --------------------------------------------------------------------------- #
def test_resolve_path_relative_and_absolute(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    rel = M.resolve_path(repo, "data/x.json")
    assert rel == (repo / "data/x.json").resolve()
    absolute = M.resolve_path(repo, str(tmp_path / "abs.json"))
    assert absolute == (tmp_path / "abs.json")


# --------------------------------------------------------------------------- #
# SHA-256.                                                                     #
# --------------------------------------------------------------------------- #
def test_sha256_of_file(tmp_path):
    import hashlib
    payload = b"hello phase 2d1c"
    file = tmp_path / "f.bin"
    file.write_bytes(payload)
    assert M.sha256_of_file(file) == hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------- #
# COCO structural analysis.                                                    #
# --------------------------------------------------------------------------- #
def test_structure_counts_and_zero_gt():
    coco = make_coco(num_images=6, num_empty=2)
    s = M.analyze_coco_structure(coco)
    assert s.num_images == 6
    assert s.num_annotations == 4
    assert s.num_categories == 14
    assert len(s.zero_gt_image_ids) == 2
    assert set(s.zero_gt_image_ids) == {5, 6}
    assert len(s.nonempty_image_ids) == 4


def test_structure_detects_duplicate_image_ids():
    coco = make_coco()
    coco["images"][1]["id"] = coco["images"][0]["id"]
    s = M.analyze_coco_structure(coco)
    assert s.duplicate_image_ids


def test_structure_detects_duplicate_annotation_ids():
    coco = make_coco()
    coco["annotations"][1]["id"] = coco["annotations"][0]["id"]
    s = M.analyze_coco_structure(coco)
    assert s.duplicate_annotation_ids


def test_structure_detects_invalid_image_ref():
    coco = make_coco()
    coco["annotations"][0]["image_id"] = 9999
    s = M.analyze_coco_structure(coco)
    assert s.invalid_annotation_image_refs == 1


def test_structure_detects_invalid_category_ref():
    coco = make_coco()
    coco["annotations"][0]["category_id"] = 9999
    s = M.analyze_coco_structure(coco)
    assert s.invalid_annotation_category_refs == 1


def test_structure_rejects_malformed_top_level():
    with pytest.raises(M.Phase2D1CError):
        M.analyze_coco_structure({"images": [], "annotations": []})


# --------------------------------------------------------------------------- #
# Category -> label mapping (must not confuse category_id with label index).   #
# --------------------------------------------------------------------------- #
def test_cat_id_to_label_is_positional():
    coco = make_coco(num_cats=14)
    mapping = M.build_cat_id_to_label(coco)
    # category ids 1..14 map to contiguous labels 0..13
    assert mapping[1] == 0
    assert mapping[14] == 13
    assert min(mapping.values()) == 0
    assert max(mapping.values()) == 13


def test_metainfo_classes_ordered_by_cat_id():
    coco = make_coco(num_cats=3)
    # shuffle category order in the JSON; ordering must still be by id
    coco["categories"] = list(reversed(coco["categories"]))
    classes = M.build_metainfo_classes(coco)
    assert classes == ("class_0", "class_1", "class_2")


# --------------------------------------------------------------------------- #
# JPG resolution / missing detection.                                          #
# --------------------------------------------------------------------------- #
def test_resolve_image_files_detects_missing(tmp_path):
    coco = make_coco(num_images=3, num_empty=0)
    root = tmp_path / "images_jpg"
    (root / "train").mkdir(parents=True)
    # create only 2 of 3 referenced files
    (root / "train" / "img_1.jpg").write_bytes(b"x")
    (root / "train" / "img_2.jpg").write_bytes(b"x")
    info = M.resolve_image_files(coco, root)
    assert info["num_referenced"] == 3
    assert info["num_missing"] == 1
    assert info["num_unique_resolved"] == 3


# --------------------------------------------------------------------------- #
# BBox / label validation.                                                     #
# --------------------------------------------------------------------------- #
def test_validate_bboxes_valid_case():
    res = M.validate_bboxes_and_labels(
        [[10, 10, 30, 40]], [0], img_width=100, img_height=100, num_classes=14)
    assert res["valid"] is True
    assert res["n_boxes"] == 1 and res["n_labels"] == 1


def test_validate_bboxes_rejects_x2_le_x1():
    res = M.validate_bboxes_and_labels(
        [[30, 10, 20, 40]], [0], img_width=100, img_height=100, num_classes=14)
    assert res["x2_gt_x1"] is False
    assert res["valid"] is False


def test_validate_bboxes_rejects_out_of_bounds():
    res = M.validate_bboxes_and_labels(
        [[10, 10, 500, 40]], [0], img_width=100, img_height=100, num_classes=14)
    assert res["in_bounds"] is False
    assert res["valid"] is False


def test_validate_bboxes_rejects_label_out_of_range():
    res = M.validate_bboxes_and_labels(
        [[10, 10, 30, 40]], [14], img_width=100, img_height=100, num_classes=14)
    assert res["labels_in_range"] is False
    assert res["valid"] is False


def test_validate_bboxes_rejects_count_mismatch():
    res = M.validate_bboxes_and_labels(
        [[10, 10, 30, 40]], [0, 1], img_width=100, img_height=100,
        num_classes=14)
    assert res["count_match"] is False
    assert res["valid"] is False


def test_empty_gt_shape_ok():
    import numpy as np
    res = M.check_empty_gt_shapes(np.zeros((0, 4)), np.zeros((0,)))
    assert res["bboxes_shape"] == [0, 4]
    assert res["labels_shape"] == [0]
    assert res["semantic_ok"] is True


def test_empty_gt_valid_via_validator():
    import numpy as np
    res = M.validate_bboxes_and_labels(
        np.zeros((0, 4)), np.zeros((0,)), 100, 100, 14)
    assert res["valid"] is True
    assert res["n_boxes"] == 0 and res["n_labels"] == 0


# --------------------------------------------------------------------------- #
# Deterministic empty-sample selection.                                        #
# --------------------------------------------------------------------------- #
def test_select_deterministic_indices_finds_empty():
    ordered = [1, 2, 3, 4, 5, 6]
    empties = [5, 6]
    chosen = M.select_deterministic_indices(ordered, empties, count=1)
    assert chosen == [4]  # position of image_id 5
    chosen2 = M.select_deterministic_indices(ordered, empties, count=5)
    assert chosen2 == [4, 5]


# --------------------------------------------------------------------------- #
# Report accumulator + status logic.                                          #
# --------------------------------------------------------------------------- #
def test_report_critical_failure_blocks():
    r = M.Report()
    r.add_check("a", True)
    r.add_check("b", False, critical=True)
    assert r.overall_pass is False
    assert r.blocking_failures


def test_report_noncritical_failure_does_not_block():
    r = M.Report()
    r.add_check("a", True)
    r.add_check("b", False, critical=False)
    assert r.overall_pass is True


# --------------------------------------------------------------------------- #
# Report writing (atomic) + errors CSV header when empty.                      #
# --------------------------------------------------------------------------- #
def test_write_json_atomic_roundtrip(tmp_path):
    path = tmp_path / "sub" / "r.json"
    M.write_json_atomic(path, {"phase": "2D.1C", "n": 1})
    assert json.loads(path.read_text())["phase"] == "2D.1C"


def test_errors_csv_header_only_when_no_errors(tmp_path):
    path = tmp_path / "errors.csv"
    M.write_csv_atomic(path, ["check", "detail"], [])
    lines = path.read_text().strip().splitlines()
    assert lines == ["check,detail"]


def test_markdown_render_has_status():
    report_obj = {
        "phase": "2D.1C", "generated_at_utc": "now",
        "overall_status": "FAIL", "dataset_loading_validated": False,
        "empty_image_retention_validated": False,
        "dataset_training_ready": False, "training_authorized": False,
        "environment_versions": {}, "input_paths": {}, "expected_counts": {},
        "observed_raw_counts": {}, "retention_dataset_results": {},
        "controlled_filtering_comparison": {}, "pipeline_validation_summary": {},
        "dataloader_validation_summary": {}, "checks": [], "errors": [],
        "warnings": [],
    }
    md = M.render_markdown(report_obj)
    assert "Phase 2D.1C" in md
    assert "training_authorized: False" in md
    assert "does not imply" in md


# --------------------------------------------------------------------------- #
# Failure exit code (integration skipped -> FAIL, non-zero exit).             #
# --------------------------------------------------------------------------- #
def test_missing_inputs_produce_failure_exit(tmp_path):
    """With no COCO/images, the run must FAIL and exit non-zero, and still
    write all four evidence files (report json/md, image audit, errors)."""
    repo = tmp_path / "repo"
    (repo / "reports").mkdir(parents=True)
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH),
         "--repo-root", str(repo),
         "--ann-file", "data/missing_coco.json",
         "--data-root", "data/missing_images"],
        capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "VALIDATION: FAIL" in result.stdout
    assert "TRAINING AUTHORIZED: FALSE" in result.stdout
    for name in ("phase2D1C_mmdet_dataset_loading_report.json",
                 "phase2D1C_mmdet_dataset_loading_report.md",
                 "phase2D1C_mmdet_dataset_image_audit.csv",
                 "phase2D1C_mmdet_dataset_errors.csv"):
        assert (repo / "reports" / name).is_file(), name
    data = json.loads(
        (repo / "reports" / "phase2D1C_mmdet_dataset_loading_report.json")
        .read_text())
    assert data["training_authorized"] is False
    assert data["dataset_training_ready"] is False
    assert data["overall_status"] == "FAIL"


def test_report_schema_keys_present(tmp_path):
    repo = tmp_path / "repo2"
    (repo / "reports").mkdir(parents=True)
    subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--repo-root", str(repo),
         "--ann-file", "none.json", "--data-root", "none"],
        capture_output=True, text=True, check=False)
    data = json.loads(
        (repo / "reports" / "phase2D1C_mmdet_dataset_loading_report.json")
        .read_text())
    for key in ("phase", "generated_at_utc", "environment_versions",
                "coco_sha256", "expected_counts", "observed_raw_counts",
                "mmdet_dataset_configuration", "retention_dataset_results",
                "controlled_filtering_comparison",
                "empty_gt_image_id_comparison", "pipeline_validation_summary",
                "bbox_label_validation_summary", "dataloader_validation_summary",
                "sample_evidence", "errors", "warnings", "checks",
                "overall_status", "dataset_loading_validated",
                "empty_image_retention_validated", "dataset_training_ready",
                "training_authorized"):
        assert key in data, key


# --------------------------------------------------------------------------- #
# AST safety: no training / no weights / no source mutation.                    #
# --------------------------------------------------------------------------- #
def _script_source():
    return SCRIPT_PATH.read_text(encoding="utf-8")


def _script_code_only():
    """Return the script source with comments and string literals removed,
    so token guards match real code, not prose in docstrings/comments."""
    import io
    import tokenize
    pieces = []
    with open(SCRIPT_PATH, "rb") as handle:
        for tok in tokenize.tokenize(handle.readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING,
                            tokenize.ENCODING):
                continue
            pieces.append(tok.string)
    return " ".join(pieces).lower()


def test_no_forbidden_training_tokens():
    code = _script_code_only()
    for token in ("train_detector", "init_detector", "load_checkpoint",
                  "runner.train", "backward", "optimizer", "load_from",
                  "pretrained"):
        assert token not in code, f"forbidden training token in code: {token}"


def test_filter_empty_gt_false_is_used_for_retention():
    src = _script_source()
    assert "filter_empty_gt=False" in src
    assert "filter_empty_gt=True" in src


def test_training_authorized_is_hardcoded_false():
    src = _script_source()
    assert '"training_authorized": False' in src


def test_no_writes_into_source_data_tree():
    """The script must not open the images_jpg tree or the COCO master for
    writing. Assert there is no write/append open on source data paths."""
    tree = ast.parse(_script_source())
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "open":
            mode = ""
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                mode = str(node.args[1].value)
            for kw in node.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    mode = str(kw.value.value)
            if any(ch in mode for ch in ("w", "a", "x", "+")):
                offenders.append(ast.dump(node))
    # All writes go through atomic helpers to report paths, never a raw
    # write-open on a source-data literal.
    assert offenders == [], f"unexpected write-open calls: {offenders}"


def test_no_pip_or_conda_calls():
    src = _script_source().lower()
    for token in ("pip install", "conda install", "subprocess", "os.system"):
        assert token not in src, token


# --------------------------------------------------------------------------- #
# Integration marker (executed only when MMDetection is importable).           #
# --------------------------------------------------------------------------- #
def _mmdet_available():
    try:
        import mmdet  # noqa: F401
        import mmengine  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _mmdet_available(),
                    reason="MMDetection not installed in this environment")
def test_build_validation_pipeline_types():
    pipeline = M.build_validation_pipeline()
    types = [t["type"] for t in pipeline]
    assert types == ["LoadImageFromFile", "LoadAnnotations", "PackDetInputs"]
    assert not any(t in types for t in
                   ("RandomFlip", "Resize", "RandomCrop", "PhotoMetricDistortion"))
