#!/usr/bin/env python3
r"""Phase 2D.1B-Full - Full Controlled-Scope DICOM-to-JPG Conversion & Validation.

This orchestrator converts the LOCKED 4,894-image controlled scope from DICOM to
single-channel grayscale JPEG at quality 95 (the pilot-locked final quality),
builds a path-only COCO derivative, validates every invariant, and promotes the
result transaction-like. It REUSES the frozen Phase 2D.1B pilot V6 transformation
core intact (imported, never modified) and the pure protocol helpers in
``src/utils/dicom_jpg_protocol.py``.

HARD SCOPE (enforced in code + tests):
    * Reuses the FROZEN pilot V6 transform core; does not modify it.
    * Never modifies source DICOM, coco_master.json, canonical annotations, the
      protocol YAML, or the historical decision evidence.
    * No geometry transform (resize/crop/rotate/flip/transpose); no bbox scaling.
    * No split / labeled-unlabeled subset / training / inference / pseudo-label /
      AP-mAP. Never sets dataset_training_ready or training_authorized true.
    * Quality is exactly 95; any other quality hard-fails.
    * Full execution requires explicit opt-in flags; preflight never auto-runs it.
    * Conversion writes only to staging; final is populated only by an
      all-or-nothing promotion after full validation passes.

The machine-readable quality/authorization source is the decision JSON
``reports/phase2D1B_pilot_decision_template.json`` (an official decision artefact
despite its file name); the Markdown template is never used as the decision
source.

Usage (Windows CMD):
    python scripts\02D1B_full_dicom_to_jpg.py --preflight-only
    python scripts\02D1B_full_dicom_to_jpg.py ^
        --execute-full --acknowledge-full-scope 4894 --jpeg-quality 95
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import os
import platform
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import dicom_jpg_protocol as P  # noqa: E402

LOG = logging.getLogger("phase2D1B_full")

# ------------------------------------------------------------------------- #
# Canonical inputs / locked evidence (paths only; values read at runtime)    #
# ------------------------------------------------------------------------- #
PILOT_SCRIPT = REPO_ROOT / "scripts" / "02D1B_pilot_dicom_to_jpg.py"
PROTOCOL_YAML = REPO_ROOT / "configs" / "protocol" / "phase2D1_jpg_representation.yaml"
DECISION_JSON = REPO_ROOT / "reports" / "phase2D1B_pilot_decision_template.json"
COCO_MASTER = REPO_ROOT / "data" / "processed" / "coco" / "coco_master.json"
CANONICAL_BBOX = REPO_ROOT / "data" / "processed" / "canonical" / "canonical_bbox_table.csv"
CANONICAL_CLASS = REPO_ROOT / "data" / "processed" / "canonical" / "canonical_class_mapping.csv"
PHASE2A_META = REPO_ROOT / "reports" / "phase2A_image_metadata.csv"
PHASE2D_VALID = REPO_ROOT / "reports" / "phase2D_coco_master_validation.json"

# ------------------------------------------------------------------------- #
# Output layout (staging / final / backup / failed)                          #
# ------------------------------------------------------------------------- #
FINAL_ROOT = REPO_ROOT / "data" / "processed" / "images_jpg"
FINAL_TRAIN = FINAL_ROOT / "train"
STAGING_ROOT = REPO_ROOT / "data" / "processed" / "images_jpg_staging"
STAGING_TRAIN = STAGING_ROOT / "train"
BACKUP_ROOT = REPO_ROOT / "data" / "processed" / "images_jpg_backup"
FAILED_ROOT = REPO_ROOT / "data" / "processed" / "images_jpg_failed"
COCO_JPG_FINAL = REPO_ROOT / "data" / "processed" / "coco" / "coco_master_jpg.json"
COCO_JPG_STAGING = STAGING_ROOT / "coco_master_jpg.json"
REPORTS_DIR = REPO_ROOT / "reports"

LOCKED_QUALITY = 95
LOCKED_SCOPE = P.LOCKED_INPUT_COUNTS["images"]  # 4894, from the locked protocol

# Rough per-image JPEG size fallback (bytes) if pilot fidelity evidence absent.
FALLBACK_JPG_BYTES = 1_500_000


class FullError(P.Phase2D1BError):
    """Any hard failure specific to the full conversion (blocks promotion)."""


# ------------------------------------------------------------------------- #
# Reuse the FROZEN pilot V6 core (import; never modify it)                     #
# ------------------------------------------------------------------------- #
def load_pilot():
    """Import the digit-leading frozen pilot module to reuse its V6 core."""
    spec = importlib.util.spec_from_file_location("phase2D1B_pilot_v6", PILOT_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # required so its @dataclass resolves annotations
    spec.loader.exec_module(mod)
    return mod


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return P.strict_json_loads(fh.read())


# Canonical inputs that must be byte-identical before and after a full run.
CANONICAL_INPUT_FILES = (PROTOCOL_YAML, DECISION_JSON, COCO_MASTER, CANONICAL_BBOX,
                         CANONICAL_CLASS, PHASE2A_META, PHASE2D_VALID)


def _input_key(path: Path) -> str:
    """Stable, path-distinguishing snapshot key (repo-relative when possible).

    Uses the full resolved path, NOT just the basename, so two inputs with the
    same file name in different directories are never conflated.
    """
    resolved = path.resolve()
    try:
        return "file:" + str(resolved.relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return "file:" + str(resolved)


def snapshot_input_hashes(pilot, resolved: Dict[str, Path]) -> Dict[str, str]:
    """SHA-256 snapshot of every source DICOM + all canonical evidence inputs."""
    snap: Dict[str, str] = {}
    for path in CANONICAL_INPUT_FILES:
        snap[_input_key(path)] = pilot.file_sha256(path)
    for oid, path in resolved.items():
        snap[f"dicom:{oid}"] = pilot.file_sha256(path)
    return snap


def verify_input_hashes(pilot, resolved: Dict[str, Path],
                        snap: Dict[str, str]) -> Dict[str, Any]:
    """Re-verify every snapshotted input is unchanged; hard fail on any drift."""
    changed: List[str] = []
    for path in CANONICAL_INPUT_FILES:
        if pilot.file_sha256(path) != snap.get(_input_key(path)):
            changed.append(_input_key(path))
    for oid, path in resolved.items():
        if pilot.file_sha256(path) != snap.get(f"dicom:{oid}"):
            changed.append(f"dicom:{oid}")
    if changed:
        raise FullError(f"input hash drift detected during run: {changed[:10]}")
    return {"verified_files": len(CANONICAL_INPUT_FILES),
            "verified_dicom": len(resolved)}


# ========================================================================= #
# Stage 1: preflight_inputs                                                   #
# ========================================================================= #
def preflight_inputs(pilot, dicom_root: Path) -> Dict[str, Any]:
    """Validate decision/protocol/canonical gates. Hard fail before any decode."""
    import yaml

    for required in (PROTOCOL_YAML, DECISION_JSON, COCO_MASTER, PHASE2D_VALID,
                     CANONICAL_BBOX, CANONICAL_CLASS, PHASE2A_META):
        if not required.exists():
            raise FullError(f"missing required canonical artefact: {required}")

    # Protocol version + fingerprint (YAML is frozen; not modified here).
    with open(PROTOCOL_YAML, "r", encoding="utf-8") as fh:
        protocol = yaml.safe_load(fh)
    protocol_evidence = P.validate_protocol(protocol)

    # Decision artefact (JSON is authoritative; Markdown is NOT a decision source).
    decision = read_json(DECISION_JSON)
    quality = decision.get("final_jpeg_quality")
    if quality != LOCKED_QUALITY:
        raise FullError(f"decision final_jpeg_quality={quality!r} != {LOCKED_QUALITY}")
    if decision.get("full_conversion_authorized") is not True:
        raise FullError("decision full_conversion_authorized is not true")
    decision_sha = pilot.file_sha256(DECISION_JSON)

    # Expected invariants are READ from canonical evidence, then checked to match
    # the locked state (never blindly hard-coded as the only source of truth).
    phase2d = read_json(PHASE2D_VALID)
    expected_sha = phase2d.get("output_sha256")
    expected_counts = phase2d.get("counts", {})
    if expected_sha != P.EXPECTED_COCO_MASTER_SHA256:
        raise FullError("phase2D evidence sha disagrees with locked COCO sha")
    for k, v in P.LOCKED_INPUT_COUNTS.items():
        if int(expected_counts.get(k, -1)) != int(v):
            raise FullError(f"phase2D evidence count drift: {k}={expected_counts.get(k)}")

    # COCO master hash must match the locked evidence.
    coco_sha = pilot.file_sha256(COCO_MASTER)
    if coco_sha != expected_sha:
        raise FullError(f"coco_master drift: {coco_sha} != {expected_sha}")

    coco = read_json(COCO_MASTER)
    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    counts = {
        "images": len(images),
        "annotations": len(annotations),
        "categories": len(categories),
        "abnormal_images": sum(1 for im in images if not im.get("is_negative")),
        "no_finding_images": sum(1 for im in images if im.get("is_negative")),
    }
    for k in ("images", "annotations", "categories", "abnormal_images",
              "no_finding_images"):
        if counts[k] != int(expected_counts.get(k)):
            raise FullError(f"live COCO count mismatch: {k}={counts[k]}")

    # No Finding is not a detection category; carries no annotations.
    neg_ids = {im["id"] for im in images if im.get("is_negative")}
    if sum(1 for a in annotations if a["image_id"] in neg_ids) != 0:
        raise FullError("No Finding images must have zero annotations")

    # Identifier / filename invariants (unique, relative, safe, .dicom).
    if len({im["id"] for im in images}) != len(images):
        raise FullError("duplicate COCO image id")
    if len({im["original_image_id"] for im in images}) != len(images):
        raise FullError("duplicate original_image_id")
    file_names = [im["file_name"] for im in images]
    if len(set(file_names)) != len(file_names):
        raise FullError("duplicate COCO file_name")
    for fn in file_names:
        pure = PurePosixPath(fn)
        if pure.is_absolute() or Path(fn).is_absolute() or ".." in pure.parts:
            raise FullError(f"unsafe file_name: {fn}")
        if not fn.startswith("train/") or not fn.endswith(".dicom"):
            raise FullError(f"unexpected file_name: {fn}")

    # Category / canonical mapping validated by metadata (never category=canon+1).
    cat_ids = sorted(c["id"] for c in categories)
    if cat_ids != list(range(1, P.NUM_ABNORMAL_CLASSES + 1)):
        raise FullError(f"category ids not contiguous 1..14: {cat_ids}")
    canon_ids = sorted(c["canonical_class_id"] for c in categories)
    if canon_ids != list(range(0, P.NUM_ABNORMAL_CLASSES)):
        raise FullError(f"canonical ids not contiguous 0..13: {canon_ids}")
    for c in categories:
        if c["name"].strip().lower() in ("no finding", "background"):
            raise FullError(f"forbidden category present: {c['name']}")

    # Exact-inventory DICOM resolution: every scope path exists; no missing,
    # no duplicates; extra .dicom under the train root is a hard fail.
    resolved: Dict[str, Path] = {}
    for im in images:
        oid = im["original_image_id"]
        path = P.safe_resolve_under_root(dicom_root, im["file_name"])
        if not path.exists():
            raise FullError(f"missing DICOM for scope: {im['file_name']}")
        if path.stem != oid:
            raise FullError(f"DICOM stem mismatch: {path.stem} != {oid}")
        resolved[oid] = path
    if len(set(resolved.values())) != len(resolved):
        raise FullError("duplicate resolved DICOM path")
    # Exact-inventory policy: recursively scan the train root so nested DICOM are
    # detected. Both MISSING (checked above per image) and EXTRA (including any
    # nested/unexpected .dicom) are hard failures.
    train_root = (dicom_root / "train")
    scope_resolved = {p.resolve() for p in resolved.values()}
    if train_root.exists():
        on_disk = {p.resolve() for p in train_root.rglob("*.dicom")}
        extra = on_disk - scope_resolved
        if extra:
            sample = sorted(str(p) for p in extra)[:5]
            raise FullError(
                f"exact-inventory violation: {len(extra)} extra/nested DICOM under "
                f"train root not in controlled scope (e.g. {sample})")
        nested = {p for p in on_disk if p.parent.resolve() != train_root.resolve()}
        if nested:
            raise FullError(
                f"exact-inventory violation: {len(nested)} nested DICOM detected "
                "below train root")

    LOG.info("preflight_inputs PASS: quality=95 authorized=true counts=%s", counts)
    return {
        "protocol_evidence": protocol_evidence, "decision": decision,
        "decision_sha256": decision_sha, "coco": coco, "coco_sha256": coco_sha,
        "counts": counts, "resolved": resolved,
        "class_names": {c["canonical_class_id"]: c["name"] for c in categories},
        "cat_by_canonical": {c["canonical_class_id"]: c["id"] for c in categories},
    }


# ========================================================================= #
# Stage 2: preflight_environment                                             #
# ========================================================================= #
PHASE2A_TRANSFER_SYNTAX_COLUMN = "TransferSyntaxUID"
PHASE2A_IMAGE_ID_COLUMN = "image_id"


def preflight_environment(jpeg2000_decoder: str, scope_ids: set) -> Dict[str, Any]:
    import importlib

    env: Dict[str, Any] = {"python_version": sys.version,
                           "platform": platform.platform(), "utc": utc_now(),
                           "jpeg2000_decoder_requested": jpeg2000_decoder}
    missing = []
    for name in ("pydicom", "numpy", "PIL", "skimage", "yaml"):
        try:
            mod = importlib.import_module(name)
            env[f"{name}_version"] = getattr(mod, "__version__", None)
        except Exception as exc:
            env[f"{name}_version"] = None
            env[f"{name}_error"] = repr(exc)
            missing.append(name)
    if missing:
        raise FullError(f"missing required dependencies: {missing}")

    # Real Pillow JPEG capability (encode/decode).
    try:
        from PIL import features
        env["pillow_jpeg"] = bool(features.check("jpg"))
    except Exception as exc:
        env["pillow_jpeg"] = False
        env["pillow_jpeg_error"] = repr(exc)
    if not env["pillow_jpeg"]:
        raise FullError("Pillow JPEG codec unavailable")

    # If the controlled scope contains ANY JPEG2000 transfer syntax (read from
    # Phase 2A metadata, no decode), the explicit backend MUST be available now.
    # No silent fallback is permitted. Phase 2A is validated strictly: the exact
    # transfer-syntax column must exist, every controlled-scope image must be
    # present with a non-empty transfer syntax, and the Phase 2A image_id set
    # must equal the controlled scope. An empty syntax set never silent-passes.
    env["jpeg2000_backend_available"] = P.jpeg2000_backend_available(jpeg2000_decoder)
    if not PHASE2A_META.exists():
        raise FullError("Phase 2A metadata missing for transfer-syntax preflight")
    with open(PHASE2A_META, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        if PHASE2A_TRANSFER_SYNTAX_COLUMN not in fieldnames:
            raise FullError(
                f"Phase 2A missing column {PHASE2A_TRANSFER_SYNTAX_COLUMN!r}")
        if PHASE2A_IMAGE_ID_COLUMN not in fieldnames:
            raise FullError(f"Phase 2A missing column {PHASE2A_IMAGE_ID_COLUMN!r}")
        ts_by_id: Dict[str, str] = {}
        for r in reader:
            iid = (r.get(PHASE2A_IMAGE_ID_COLUMN) or "").strip()
            ts = (r.get(PHASE2A_TRANSFER_SYNTAX_COLUMN) or "").strip()
            if iid == "":
                raise FullError("Phase 2A contains an empty image_id")
            if iid in ts_by_id:
                # A duplicate image_id must never be silently overwritten.
                if ts_by_id[iid] != ts:
                    raise FullError(
                        f"duplicate Phase 2A image_id {iid!r} with conflicting "
                        f"transfer syntax ({ts_by_id[iid]!r} vs {ts!r})")
                raise FullError(f"duplicate Phase 2A image_id {iid!r}")
            ts_by_id[iid] = ts
    phase2a_ids = set(ts_by_id.keys())
    if phase2a_ids != set(scope_ids):
        missing = sorted(set(scope_ids) - phase2a_ids)[:5]
        extra = sorted(phase2a_ids - set(scope_ids))[:5]
        raise FullError(
            f"Phase 2A image_id set != controlled scope (missing={missing} "
            f"extra={extra})")
    missing_ts = sorted(iid for iid in scope_ids if not ts_by_id.get(iid))
    if missing_ts:
        raise FullError(
            f"Phase 2A missing transfer syntax for {len(missing_ts)} scope images "
            f"(e.g. {missing_ts[:5]})")
    scope_syntaxes = {ts_by_id[iid] for iid in scope_ids}
    if not scope_syntaxes:
        raise FullError("empty transfer-syntax set for controlled scope")
    scope_needs_jpeg2000 = any(P.is_jpeg2000(ts) for ts in scope_syntaxes)
    env["scope_needs_jpeg2000"] = scope_needs_jpeg2000
    env["scope_transfer_syntaxes"] = sorted(scope_syntaxes)
    if scope_needs_jpeg2000:
        # Raises P.UnsupportedInputError (Phase2D1BError) if unavailable.
        P.ensure_jpeg2000_backend(jpeg2000_decoder)

    # Staging and final must be on the same filesystem for atomic rename.
    STAGING_ROOT.parent.mkdir(parents=True, exist_ok=True)
    FINAL_ROOT.parent.mkdir(parents=True, exist_ok=True)
    dev_staging = os.stat(STAGING_ROOT.parent).st_dev
    dev_final = os.stat(FINAL_ROOT.parent).st_dev
    env["same_filesystem"] = (dev_staging == dev_final)
    if not env["same_filesystem"]:
        raise FullError("staging and final are on different filesystems "
                        "(cannot assume atomic rename)")

    # Disk space: estimate needed from pilot q95 sizes when available.
    per_image = _estimate_jpg_bytes()
    needed = per_image * LOCKED_SCOPE
    # staging + final + backup + safety margin.
    total_needed = int(needed * 3 * 1.2)
    free = shutil.disk_usage(str(FINAL_ROOT.parent)).free
    env["estimated_per_image_bytes"] = per_image
    env["estimated_total_needed_bytes"] = total_needed
    env["disk_free_bytes"] = free
    if free < total_needed:
        raise FullError(f"insufficient disk: free={free} < needed~={total_needed}")

    LOG.info("preflight_environment PASS")
    return env


def _estimate_jpg_bytes() -> int:
    pilot_fid = REPORTS_DIR / "phase2D1B_pilot_fidelity_metrics.csv"
    if pilot_fid.exists():
        sizes = []
        with open(pilot_fid, "r", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if str(r.get("jpeg_quality")) == "95":
                    try:
                        sizes.append(int(float(r["jpg_file_size_bytes"])))
                    except (KeyError, ValueError):
                        pass
        if sizes:
            return max(1, int(sum(sizes) / len(sizes)))
    return FALLBACK_JPG_BYTES


# ========================================================================= #
# Stage 3: preflight_output_safety                                           #
# ========================================================================= #
def preflight_output_safety(overwrite: bool) -> Dict[str, Any]:
    info: Dict[str, Any] = {"overwrite": overwrite}
    # Default: never overwrite a populated final directory.
    if FINAL_TRAIN.exists() and any(FINAL_TRAIN.iterdir()):
        if not overwrite:
            raise FullError("final image directory exists and is non-empty "
                            "(refusing to overwrite)")
    if COCO_JPG_FINAL.exists() and not overwrite:
        raise FullError("coco_master_jpg.json already exists (refusing to overwrite)")
    # Staging must be clean before a run.
    if STAGING_ROOT.exists() and any(STAGING_ROOT.rglob("*")):
        raise FullError("staging directory is not clean; refusing to run")
    info["final_populated"] = FINAL_TRAIN.exists() and any(FINAL_TRAIN.iterdir()) \
        if FINAL_TRAIN.exists() else False
    LOG.info("preflight_output_safety PASS")
    return info


# ========================================================================= #
# Stage 4: convert_one_image (reuses frozen V6 core)                          #
# ========================================================================= #
def convert_one_image(pilot, oid: str, path: Path, coco_image: Dict[str, Any],
                      meta_dim: Dict[str, int], staging_train: Path,
                      jpeg2000_decoder: str) -> Dict[str, Any]:
    import numpy as np
    import pydicom
    from PIL import Image

    header = pilot.read_header(path)
    # Structural + transform preflight reuse the frozen validators (block gaps).
    pilot.validate_header_structural(header, coco_image["width"], coco_image["height"],
                                     meta_dim["w"], meta_dim["h"])
    pilot.header_transform_preflight(header)

    if P.is_jpeg2000(header["TransferSyntaxUID"]):
        P.ensure_jpeg2000_backend(jpeg2000_decoder)  # hard fail; no silent fallback

    ds = pydicom.dcmread(str(path), force=False)
    stored, backend = pilot.decode_pixels(ds, header, jpeg2000_decoder)
    if stored.ndim != 2 or stored.shape != (header["Rows"], header["Columns"]):
        raise FullError(f"decoded array shape invalid for {oid}")

    tr = pilot.transform_pixels(header, stored, ds)  # FROZEN V6 transform core
    uint8 = tr["uint8"]
    if uint8.shape != (header["Rows"], header["Columns"]):
        raise FullError(f"geometry changed during transform for {oid}")

    source_sha = pilot.file_sha256(path)
    pre_sha = P.pre_jpeg_sha256(uint8)

    # Encode a single q95 JPEG to a temp file, then atomic-replace within staging.
    # The .part temp is always cleaned up (try/finally) even if encode, decode or
    # replace fails; a failed conversion never leaves a partial or temp artefact.
    out_rel = f"train/{oid}.jpg"
    out_path = staging_train / f"{oid}.jpg"
    fd, tmp = tempfile.mkstemp(dir=str(staging_train), suffix=".part")
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        Image.fromarray(uint8, mode="L").save(tmp_path, format="JPEG",
                                              quality=LOCKED_QUALITY, optimize=False,
                                              progressive=False)
        os.replace(tmp_path, out_path)  # atomic within staging
        with Image.open(out_path) as jp:
            if jp.format != "JPEG" or jp.mode != "L" \
                    or jp.size != (header["Columns"], header["Rows"]):
                raise FullError(f"JPEG validation failed for {oid}")
            exif = jp.getexif()
            if exif and exif.get(0x0112) not in (None, 1):
                raise FullError(f"EXIF orientation present for {oid}")
            decoded = np.array(jp)
        out_sha = pilot.file_sha256(out_path)
    except Exception:
        # Remove any partially written final-in-staging file on failure.
        if out_path.exists():
            try:
                out_path.unlink()
            except OSError:
                pass
        raise
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

    return {
        "image_id": oid,
        "source_relative_path": coco_image["file_name"],
        "output_relative_path": out_rel,
        "source_dicom_sha256": source_sha,
        "pre_jpeg_uint8_sha256": pre_sha,
        "output_jpeg_sha256": out_sha,
        "decoded_jpeg_uint8_sha256": P.pre_jpeg_sha256(decoded),
        "width": header["Columns"], "height": header["Rows"],
        "jpeg_quality": LOCKED_QUALITY,
        "decoder_backend": backend,
        "modality_branch": tr["modality_branch"],
        "voi_branch": tr["voi_branch"],
        "presentation_branch": ("inverted" if tr["presentation_inversion_count"]
                                else "identity"),
        "presentation_inversion_count": tr["presentation_inversion_count"],
        "pixel_padding_branch": ("padded" if tr["padding_present"] else "none"),
        "padding_pixel_count": tr["padding_pixel_count"],
        "warnings": "",
        "status": "converted",
    }


# ========================================================================= #
# Stage 5: build_full_mapping                                                #
# ========================================================================= #
MAPPING_FIELDS = [
    "image_id", "source_relative_path", "output_relative_path",
    "source_dicom_sha256", "pre_jpeg_uint8_sha256", "output_jpeg_sha256",
    "decoded_jpeg_uint8_sha256", "width", "height", "jpeg_quality",
    "protocol_version", "protocol_sha256", "decision_sha256", "decoder_backend",
    "modality_branch", "voi_branch", "pixel_padding_branch", "presentation_branch",
    "presentation_inversion_count", "padding_pixel_count", "warnings", "status",
]


def build_full_mapping(pilot, records: List[Dict[str, Any]], protocol_sha: str,
                       decision_sha: str, out_dir: Path) -> Path:
    rows = []
    for r in records:
        row = dict(r)
        row["protocol_version"] = P.EXPECTED_PROTOCOL_VERSION
        row["protocol_sha256"] = protocol_sha
        row["decision_sha256"] = decision_sha
        rows.append(row)
    csv_path = out_dir / "phase2D1B_full_mapping.csv"
    pilot.write_csv(csv_path, MAPPING_FIELDS, rows)
    # Structured JSONL alongside the CSV.
    jsonl_path = out_dir / "phase2D1B_full_mapping.jsonl"
    fd, tmp = tempfile.mkstemp(dir=str(out_dir), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps({k: row.get(k) for k in MAPPING_FIELDS},
                                ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, jsonl_path)
    return csv_path


def validate_mapping_artifacts(pilot, ctx: Dict[str, Any], records: List[Dict[str, Any]],
                               out_dir: Path) -> Dict[str, Any]:
    """Read the written mapping CSV + JSONL and validate them before promotion.

    Checks the exact image-ID set, protocol version/hash, decision hash, and the
    per-image output/source hashes against the in-memory records, and that the
    CSV and JSONL are mutually consistent. Reads the real artefacts (not just the
    field list).
    """
    csv_path = out_dir / "phase2D1B_full_mapping.csv"
    jsonl_path = out_dir / "phase2D1B_full_mapping.jsonl"
    if not csv_path.exists() or not jsonl_path.exists():
        raise FullError("mapping artefacts missing")

    with open(csv_path, "r", encoding="utf-8") as fh:
        csv_rows = list(csv.DictReader(fh))
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        jsonl_rows = [P.strict_json_loads(line) for line in fh if line.strip()]

    expected_ids = {r["image_id"] for r in records}
    csv_ids = [r["image_id"] for r in csv_rows]
    if set(csv_ids) != expected_ids or len(csv_ids) != len(expected_ids):
        raise FullError("mapping CSV image-ID set mismatch")
    if [r.get("image_id") for r in jsonl_rows] != csv_ids:
        raise FullError("mapping CSV and JSONL are inconsistent")

    protocol_sha = ctx["protocol_evidence"]["protocol_sha256"]
    decision_sha = ctx["decision_sha256"]
    rec_by_id = {r["image_id"]: r for r in records}
    for crow, jrow in zip(csv_rows, jsonl_rows):
        oid = crow["image_id"]
        if crow != {k: str(_csv_cellish(jrow.get(k))) for k in crow}:
            # Compare CSV row against JSONL row field-by-field (string form).
            for k in crow:
                if str(crow[k]) != str(_csv_cellish(jrow.get(k))):
                    raise FullError(f"mapping CSV/JSONL field mismatch: {oid}.{k}")
        if crow.get("protocol_version") != P.EXPECTED_PROTOCOL_VERSION:
            raise FullError(f"mapping protocol_version wrong for {oid}")
        if crow.get("protocol_sha256") != protocol_sha:
            raise FullError(f"mapping protocol_sha256 wrong for {oid}")
        if crow.get("decision_sha256") != decision_sha:
            raise FullError(f"mapping decision_sha256 wrong for {oid}")
        rec = rec_by_id[oid]
        if crow.get("output_jpeg_sha256") != rec["output_jpeg_sha256"]:
            raise FullError(f"mapping output hash mismatch for {oid}")
        if crow.get("source_dicom_sha256") != rec["source_dicom_sha256"]:
            raise FullError(f"mapping source hash mismatch for {oid}")
        if str(crow.get("jpeg_quality")) != str(LOCKED_QUALITY):
            raise FullError(f"mapping quality != 95 for {oid}")
    return {"mapping_rows": len(csv_rows)}


def _csv_cellish(value: Any) -> Any:
    """Match how write_csv renders values (None->'', lists->';'-joined)."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(x) for x in value)
    return value


# ========================================================================= #
# Stage 6: build_coco_jpg_derivative                                         #
# ========================================================================= #
def build_coco_jpg_derivative(pilot, coco: Dict[str, Any], staging_path: Path) -> Dict[str, Any]:
    """Deep-copy the master in memory; change ONLY images[].file_name -> .jpg.

    Never touches the master on disk. IDs, dimensions, annotations, categories,
    bbox, area and iscrowd are untouched. Output file names are relative + unique.
    """
    import copy

    derivative = copy.deepcopy(coco)
    seen = set()
    for im in derivative["images"]:
        oid = im["original_image_id"]
        new_name = f"train/{oid}.jpg"
        if new_name in seen:
            raise FullError(f"duplicate derivative file_name: {new_name}")
        seen.add(new_name)
        if PurePosixPath(new_name).is_absolute() or ".." in PurePosixPath(new_name).parts:
            raise FullError(f"unsafe derivative file_name: {new_name}")
        im["file_name"] = new_name
    # Serialize to staging only (promoted after full validation).
    staging_path.parent.mkdir(parents=True, exist_ok=True)
    pilot.atomic_write_text(staging_path,
                            json.dumps(derivative, ensure_ascii=False, allow_nan=False,
                                       indent=2) + "\n")
    return derivative


# ========================================================================= #
# Stage 7: validate_full_outputs                                             #
# ========================================================================= #
def validate_full_outputs(pilot, ctx: Dict[str, Any], derivative: Dict[str, Any],
                          records: List[Dict[str, Any]]) -> Dict[str, Any]:
    import numpy as np
    from PIL import Image

    coco = ctx["coco"]
    images = coco["images"]
    staging_train = STAGING_TRAIN
    meta = ctx["meta"]
    expected_ids = {im["original_image_id"] for im in images}
    id_to_coco = {im["original_image_id"]: im for im in images}

    # --- record IDs unique and an exact set over the controlled scope ---
    rec_ids = [r["image_id"] for r in records]
    if len(rec_ids) != len(set(rec_ids)):
        raise FullError("duplicate record image_id")
    if set(rec_ids) != expected_ids:
        raise FullError("record image_id set != controlled scope image_id set")

    # --- staging directory has ONLY the expected flat .jpg files ---
    entries = list(staging_train.iterdir())
    for e in entries:
        if e.is_dir():
            raise FullError(f"unexpected nested output directory: {e.name}")
        if e.suffix in (".part", ".tmp"):
            raise FullError("leftover temp/partial file in staging")
        if e.suffix != ".jpg":
            raise FullError(f"unexpected non-JPEG file in staging: {e.name}")
    jpgs = {p.stem for p in staging_train.glob("*.jpg")}
    if jpgs != expected_ids:
        raise FullError("staged JPG set does not match image_id set one-to-one")
    if len({p.name for p in staging_train.glob("*.jpg")}) != len(jpgs):
        raise FullError("duplicate output JPEG file names")

    # --- per record: output path expected+safe+unique; hash/dims/quality/status ---
    seen_paths = set()
    for r in records:
        oid = r["image_id"]
        im = id_to_coco[oid]
        out_rel = r["output_relative_path"]
        if out_rel != f"train/{oid}.jpg":
            raise FullError(f"unexpected output path for {oid}: {out_rel}")
        pure = PurePosixPath(out_rel)
        if pure.is_absolute() or ".." in pure.parts:
            raise FullError(f"unsafe output path: {out_rel}")
        if out_rel in seen_paths:
            raise FullError(f"duplicate output path: {out_rel}")
        seen_paths.add(out_rel)
        if int(r["jpeg_quality"]) != LOCKED_QUALITY:
            raise FullError(f"record quality != 95 for {oid}")
        if r.get("status") != "converted":
            raise FullError(f"record status not converted for {oid}")
        out_path = staging_train / f"{oid}.jpg"
        # Actual JPEG byte hash must match the recorded hash.
        if pilot.file_sha256(out_path) != r["output_jpeg_sha256"]:
            raise FullError(f"output JPEG hash mismatch vs record for {oid}")
        with Image.open(out_path) as jp:
            if jp.format != "JPEG" or jp.mode != "L":
                raise FullError(f"JPEG invalid for {oid}")
            w, h = jp.size
        if (w, h) != (im["width"], im["height"]) or (w, h) != (meta[oid]["w"], meta[oid]["h"]):
            raise FullError(f"dimension mismatch (geometry change) for {oid}")
        if (int(r["width"]), int(r["height"])) != (w, h):
            raise FullError(f"record dimensions mismatch actual JPEG for {oid}")

    # --- derivative is EXACT structural equality except images[].file_name ---
    if set(derivative.keys()) != set(coco.keys()):
        raise FullError("derivative top-level keys changed")
    if len(derivative["images"]) != len(images):
        raise FullError("derivative image count mismatch")
    for dim, mim in zip(derivative["images"], images):
        if set(dim.keys()) != set(mim.keys()):
            raise FullError("derivative image keys changed")
        for key in mim:
            if key == "file_name":
                if dim["file_name"] != f"train/{mim['original_image_id']}.jpg":
                    raise FullError("derivative file_name incorrect")
                continue
            if dim.get(key) != mim.get(key):
                raise FullError(f"derivative changed field {key} (must be invariant)")
    if json.dumps(derivative["annotations"], sort_keys=True) != \
            json.dumps(coco["annotations"], sort_keys=True):
        raise FullError("derivative annotations changed (must be invariant)")
    if json.dumps(derivative["categories"], sort_keys=True) != \
            json.dumps(coco["categories"], sort_keys=True):
        raise FullError("derivative categories changed (must be invariant)")
    for extra_key in set(coco.keys()) - {"images", "annotations", "categories"}:
        if json.dumps(derivative.get(extra_key), sort_keys=True) != \
                json.dumps(coco.get(extra_key), sort_keys=True):
            raise FullError(f"derivative changed top-level key {extra_key}")

    # --- No Finding semantics unchanged; bbox within bounds ---
    neg = {im["id"] for im in derivative["images"] if im.get("is_negative")}
    if sum(1 for a in derivative["annotations"] if a["image_id"] in neg) != 0:
        raise FullError("No Finding gained annotations in derivative")
    dims = {im["id"]: (im["width"], im["height"]) for im in derivative["images"]}
    for a in derivative["annotations"]:
        x, y, bw, bh = a["bbox"]
        W, H = dims[a["image_id"]]
        if x < 0 or y < 0 or x + bw > W or y + bh > H:
            raise FullError(f"bbox out of bounds in derivative ann {a['id']}")

    # --- source + canonical inputs unchanged after conversion ---
    if pilot.file_sha256(COCO_MASTER) != ctx["coco_sha256"]:
        raise FullError("coco_master.json changed during full run")
    if ctx.get("input_snapshot") is not None and ctx.get("resolved") is not None:
        verify_input_hashes(pilot, ctx["resolved"], ctx["input_snapshot"])

    LOG.info("validate_full_outputs PASS (candidate)")
    return {"validated_images": len(jpgs), "one_to_one": True}


# ========================================================================= #
# Stage 8: promote_outputs (transaction-like, Windows-safe)                   #
# ========================================================================= #
def promote_outputs(overwrite: bool, _fault_hook=None,
                    _post_commit=None, _post_commit_rollback=None) -> Dict[str, Any]:
    """Promote validated staging -> final as ONE transaction (images + COCO).

    Only called AFTER conversion + validation + evidence writes fully PASS. Both
    the final image directory and coco_master_jpg.json are promoted together. A
    journal of inverse operations is kept so that a failure at ANY critical
    ``os.replace`` (including between the image and COCO moves) fully rolls back:
    every completed move is undone and any prior valid final image directory and
    prior COCO derivative are restored. Staging/final are on the same filesystem
    (verified in preflight); cross-filesystem rename is never assumed atomic.

    If a rollback step itself fails, we raise a distinct ``restoration_failed``
    error and never claim success. ``_fault_hook(step)`` is a test seam that may
    raise at a named step: 'after_backup_images', 'after_backup_coco',
    'after_promote_images', 'after_promote_coco'.
    """
    if FINAL_TRAIN.exists() and any(FINAL_TRAIN.iterdir()) and not overwrite:
        raise FullError("final directory populated; refusing to overwrite")

    FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    ts = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
    backup_images = BACKUP_ROOT / f"train_{ts}"
    backup_coco = BACKUP_ROOT / f"coco_master_jpg_{ts}.json"

    journal: List[Any] = []   # inverse ops, applied in reverse on failure

    def _fault(step: str) -> None:
        if _fault_hook is not None:
            _fault_hook(step)

    try:
        # 1) move any prior valid final image dir aside.
        if FINAL_TRAIN.exists():
            os.replace(FINAL_TRAIN, backup_images)
            journal.append(("restore_images", backup_images, FINAL_TRAIN))
            _fault("after_backup_images")
        # 2) move any prior COCO derivative aside.
        if COCO_JPG_FINAL.exists():
            os.replace(COCO_JPG_FINAL, backup_coco)
            journal.append(("restore_coco", backup_coco, COCO_JPG_FINAL))
            _fault("after_backup_coco")
        # 3) promote new images.
        os.replace(STAGING_TRAIN, FINAL_TRAIN)
        journal.append(("undo_promote_images", FINAL_TRAIN, STAGING_TRAIN))
        _fault("after_promote_images")
        # 4) promote new COCO derivative.
        os.replace(COCO_JPG_STAGING, COCO_JPG_FINAL)
        journal.append(("undo_promote_coco", COCO_JPG_FINAL, COCO_JPG_STAGING))
        _fault("after_promote_coco")
        # 5) post-commit finalizer (e.g. promotion record) is part of the SAME
        # transaction: if it fails, the image + COCO promotion is rolled back so
        # a committed dataset never lacks its post-promotion record.
        promoted = {"final_train": str(FINAL_TRAIN.relative_to(REPO_ROOT)),
                    "coco_jpg": str(COCO_JPG_FINAL.relative_to(REPO_ROOT))}
        if _post_commit is not None:
            _post_commit(promoted)
    except Exception as exc:
        # Always attempt BOTH the dataset (journal) rollback AND the post-commit
        # rollback, even if one of them fails; aggregate every restoration error.
        restoration_errors: List[str] = []
        for kind, src, dst in reversed(journal):
            try:
                os.replace(src, dst)  # move new out / restore prior
            except Exception as e:  # collect, keep restoring the rest
                restoration_errors.append(f"journal:{kind}:{e!r}")
        if _post_commit_rollback is not None:
            try:
                _post_commit_rollback()
            except Exception as e:
                restoration_errors.append(f"post_commit_rollback:{e!r}")
        if restoration_errors:
            raise FullError(
                f"restoration_failed after promotion error {exc!r}: "
                f"{restoration_errors} (manual recovery required; success NOT claimed)")
        raise FullError(f"promotion_failed; prior valid output restored ({exc!r})")

    # Success: drop backups, surfacing (not silencing) any cleanup failure.
    cleanup_warnings: List[str] = []
    if backup_images.exists():
        try:
            shutil.rmtree(backup_images)
        except Exception as exc:  # do not silent-ignore
            cleanup_warnings.append(f"backup_images_cleanup_failed:{exc!r}")
            LOG.warning("backup image cleanup failed: %r", exc)
    if backup_coco.exists():
        try:
            backup_coco.unlink()
        except Exception as exc:
            cleanup_warnings.append(f"backup_coco_cleanup_failed:{exc!r}")
            LOG.warning("backup COCO cleanup failed: %r", exc)
    LOG.info("promote_outputs PASS%s",
             " (with backup-cleanup warnings)" if cleanup_warnings else "")
    return {"final_train": str(FINAL_TRAIN.relative_to(REPO_ROOT)),
            "coco_jpg": str(COCO_JPG_FINAL.relative_to(REPO_ROOT)),
            "backup_cleanup_warnings": cleanup_warnings}


# ========================================================================= #
# Stage 9: write_reports                                                     #
# ========================================================================= #
def _base_full_status(extra: Dict[str, Any]) -> Dict[str, Any]:
    # full_conversion_completed stays False until GPT review; the script only
    # ever reports execution/validation-candidate progress, never completion.
    status = {
        "phase_id": "2D.1B-Full",
        "phase_status": "OPEN",
        "final_jpeg_quality": LOCKED_QUALITY,
        "full_execution_finished": False,
        "full_validation_candidate_pass": False,
        "full_conversion_completed": False,
        "dataset_training_ready": False,
        "training_authorized": False,
        "downstream_detector_superiority_claimed": False,
        "utc": utc_now(),
    }
    status.update(extra)
    return status


def write_preflight_report(pilot, evidence: Dict[str, Any]) -> None:
    status = _base_full_status({"stage": "preflight", "preflight_pass": True,
                                **evidence})
    pilot.atomic_write_text(REPORTS_DIR / "phase2D1B_full_preflight.json",
                            pilot.strict_json_dumps(status) + "\n")
    pilot.atomic_write_text(
        REPORTS_DIR / "phase2D1B_full_preflight.md",
        "# Phase 2D.1B-Full - Preflight\n\n"
        "Preflight validated inputs, environment and output safety. Full "
        "conversion is NOT started by preflight and requires explicit opt-in.\n\n"
        "- phase_status: OPEN\n- final_jpeg_quality: 95\n"
        "- full_conversion_completed: false\n"
        "- dataset_training_ready: false\n- training_authorized: false\n",
    )


# The evidence files that MUST exist before any promotion.
REQUIRED_EVIDENCE = (
    "phase2D1B_full_mapping.csv",
    "phase2D1B_full_validation.json",
    "phase2D1B_full_validation.md",
    "phase2D1B_full_metadata_audit.csv",
    "phase2D1B_full_bbox_audit.csv",
    "phase2D1B_full_no_finding_audit.csv",
    "phase2D1B_full_errors.csv",
)


def require_evidence_present(out_dir: Path) -> None:
    """Gate: refuse to promote unless all candidate evidence is on disk."""
    missing = [name for name in REQUIRED_EVIDENCE if not (out_dir / name).exists()]
    if missing:
        raise FullError(f"evidence incomplete before promotion: {missing}")


def write_full_evidence(pilot, ctx: Dict[str, Any], records: List[Dict[str, Any]],
                        validation: Dict[str, Any]) -> None:
    """Write ALL candidate evidence BEFORE promotion.

    A report-write failure here happens before the dataset is promoted, so it can
    never leave a new final dataset lacking evidence. full_conversion_completed
    stays false; only execution/validation-candidate progress is recorded.
    """
    status = _base_full_status({
        "stage": "full_execution_candidate",
        "full_execution_finished": True,
        "full_validation_candidate_pass": True,
        "full_conversion_completed": False,
        "validated_images": validation.get("validated_images"),
        "protocol_sha256": ctx["protocol_evidence"]["protocol_sha256"],
        "coco_master_sha256": ctx["coco_sha256"],
        "decision_sha256": ctx["decision_sha256"],
        "input_hash_verification": ctx.get("input_hash_verification"),
    })
    pilot.atomic_write_text(REPORTS_DIR / "phase2D1B_full_validation.json",
                            pilot.strict_json_dumps(status) + "\n")
    pilot.atomic_write_text(
        REPORTS_DIR / "phase2D1B_full_validation.md",
        "# Phase 2D.1B-Full - Validation (candidate)\n\n"
        "Full controlled-scope conversion executed to JPEG quality 95 and passed "
        "all candidate invariants. This is NOT a completion or PASS claim and does "
        "NOT assert downstream detector superiority, clinical equivalence, or "
        "dataset training readiness; those remain out of scope and unauthorized. "
        "Completion is decided only after GPT review.\n\n"
        "- phase_status: OPEN (awaiting GPT review)\n"
        "- full_execution_finished: true\n"
        "- full_validation_candidate_pass: true\n"
        "- full_conversion_completed: false\n"
        "- dataset_training_ready: false\n- training_authorized: false\n",
    )

    # Metadata / bbox / No Finding audits.
    audit_rows = [{
        "image_id": r["image_id"], "modality_branch": r["modality_branch"],
        "voi_branch": r["voi_branch"], "presentation_branch": r["presentation_branch"],
        "pixel_padding_branch": r["pixel_padding_branch"],
        "decoder_backend": r["decoder_backend"], "warnings": r["warnings"],
    } for r in records]
    pilot.write_csv(REPORTS_DIR / "phase2D1B_full_metadata_audit.csv",
                    ["image_id", "modality_branch", "voi_branch",
                     "presentation_branch", "pixel_padding_branch",
                     "decoder_backend", "warnings"], audit_rows)

    coco = ctx["coco"]
    id_to_dim = {im["id"]: (im["width"], im["height"]) for im in coco["images"]}
    bbox_rows = []
    for a in coco["annotations"]:
        x, y, bw, bh = a["bbox"]
        W, H = id_to_dim[a["image_id"]]
        bbox_rows.append({
            "annotation_id": a["id"], "image_id": a["image_id"],
            "category_id": a["category_id"], "canonical_class_id": a.get("canonical_class_id"),
            "x": x, "y": y, "width": bw, "height": bh, "area": a["area"],
            "in_bounds": (x >= 0 and y >= 0 and x + bw <= W and y + bh <= H),
        })
    pilot.write_csv(REPORTS_DIR / "phase2D1B_full_bbox_audit.csv",
                    ["annotation_id", "image_id", "category_id", "canonical_class_id",
                     "x", "y", "width", "height", "area", "in_bounds"], bbox_rows)

    nf_rows = [{"image_id": im["id"], "original_image_id": im["original_image_id"],
                "is_negative": im.get("is_negative", False),
                "annotation_count": 0}
               for im in coco["images"] if im.get("is_negative")]
    pilot.write_csv(REPORTS_DIR / "phase2D1B_full_no_finding_audit.csv",
                    ["image_id", "original_image_id", "is_negative", "annotation_count"],
                    nf_rows)

    # Errors CSV (empty on a clean run; real errors would appear here).
    pilot.write_csv(REPORTS_DIR / "phase2D1B_full_errors.csv",
                    ["image_id", "severity", "code", "detail"], ctx.get("errors", []))


def write_promotion_record(pilot, ctx: Dict[str, Any], promoted: Dict[str, Any]) -> None:
    """Small post-promotion record. Still NOT a completion/PASS claim.

    full_conversion_completed remains false and phase_status remains OPEN; only
    GPT review may decide completion.
    """
    status = _base_full_status({
        "stage": "post_promotion",
        "full_execution_finished": True,
        "full_validation_candidate_pass": True,
        "full_conversion_completed": False,
        "promoted": promoted,
    })
    pilot.atomic_write_text(REPORTS_DIR / "phase2D1B_full_promotion.json",
                            pilot.strict_json_dumps(status) + "\n")


def write_cleanup_audit(pilot, promoted: Dict[str, Any],
                        cleanup_warnings: List[str]) -> None:
    """Persist backup-cleanup warnings into a structured audit artefact.

    Cleanup happens after a successful promotion, so these warnings inherently
    post-date the promotion record; recording them in a durable artefact (not
    only the log) keeps them auditable. Still NOT a completion/PASS claim.
    """
    status = _base_full_status({
        "stage": "post_promotion_cleanup",
        "full_execution_finished": True,
        "full_validation_candidate_pass": True,
        "full_conversion_completed": False,
        "promoted": promoted,
        "backup_cleanup_warnings": cleanup_warnings,
        "backup_cleanup_clean": len(cleanup_warnings) == 0,
    })
    pilot.atomic_write_text(REPORTS_DIR / "phase2D1B_full_cleanup_audit.json",
                            pilot.strict_json_dumps(status) + "\n")


class PromotionRecordTransaction:
    """Transactional handling of the post-promotion record.

    A prior promotion record is atomically backed up (to ``<record>.prior``)
    before a new one is written, and restored byte-for-byte on rollback. The
    rollback is fully best-effort internally: it ALWAYS attempts to clean up the
    new/partial record AND ALWAYS attempts to restore the prior record even if
    the cleanup failed, aggregating errors from both operations. A stale
    ``.prior`` recovery artefact is never overwritten - it hard-fails first.

    ``_unlink`` / ``_replace`` are injectable for tests (default to the real
    filesystem ops); this is the exact rollback used by ``run_full``.
    """

    def __init__(self, pilot, ctx: Dict[str, Any], record_path: Path,
                 backup_path: Path, _unlink=None, _replace=None):
        self.pilot = pilot
        self.ctx = ctx
        self.record_path = record_path
        self.backup_path = backup_path
        self.had_prior = record_path.exists()
        # Explicit transaction state so rollback never deletes a valid prior
        # record when finalize did not run or did not yet touch the record.
        self.started = False       # finalize entered
        self.backed_up = False     # prior record moved to <record>.prior
        self.wrote_new = False     # a new-record WRITE was started (set BEFORE
        #                            the writer, so a partial/created-then-raised
        #                            record is still cleaned up on rollback)
        self._unlink = _unlink or (lambda p: p.unlink())
        self._replace = _replace or os.replace

    def check_no_stale_backup(self) -> None:
        """Hard-fail if a stale recovery backup exists (with or without a current
        promotion record). Never overwrite a residual recovery artefact."""
        if self.backup_path.exists():
            raise FullError(
                f"stale_prior_promotion_record_backup_present: {self.backup_path} "
                "already exists; manual recovery required before running "
                "(refusing to overwrite a residual recovery artefact)")

    def finalize(self, promoted: Dict[str, Any]) -> None:
        """Part of the promotion transaction: back up prior record, write new."""
        self.started = True
        if self.record_path.exists():
            self._replace(self.record_path, self.backup_path)  # atomic backup
            self.backed_up = True
        # Mark the write as started BEFORE invoking the writer: if the writer
        # creates/replaces record_path and then raises, rollback must still
        # remove that new/partial record.
        self.wrote_new = True
        write_promotion_record(self.pilot, self.ctx, promoted)

    def rollback(self) -> None:
        """Best-effort rollback driven by explicit transaction state.

        * If finalize never started -> no-op (the current/prior record is left
          untouched; it was not modified by this transaction).
        * record_path is cleaned up ONLY if this transaction actually backed up
          the prior record or began writing a new one - never based solely on
          record_path.exists().
        * If a <record>.prior backup was created, ALWAYS attempt to restore it.
        Errors from both operations are aggregated.
        """
        if not self.started:
            return  # finalize did not run; do not touch the current record
        errors: List[str] = []
        if (self.backed_up or self.wrote_new) and self.record_path.exists():
            try:
                self._unlink(self.record_path)  # remove new/partial record only
            except Exception as e:  # collect, still attempt restore below
                errors.append(f"record_unlink:{e!r}")
        if self.backed_up and self.backup_path.exists():
            try:
                self._replace(self.backup_path, self.record_path)
            except Exception as e:
                errors.append(f"prior_restore:{e!r}")
        if errors:
            raise FullError(f"finalize_rollback_errors:{errors}")

    def commit(self) -> Optional[str]:
        """On success drop the prior-record backup, surfacing (not hiding) any
        cleanup failure as a warning string (also recorded in the audit)."""
        if self.backed_up and self.backup_path.exists():
            try:
                self._unlink(self.backup_path)
            except Exception as exc:
                LOG.warning("promotion-record backup cleanup failed: %r", exc)
                return f"promotion_record_backup_cleanup_failed:{exc!r}"
        return None


# ========================================================================= #
# Runners                                                                    #
# ========================================================================= #
def run_preflight(args: "Args") -> int:
    pilot = load_pilot()
    resolution = P.resolve_dicom_root(args.dicom_root, os.environ.get(P.ENV_VAR_NAME))
    inputs = preflight_inputs(pilot, resolution.root)
    env = preflight_environment(args.jpeg2000_decoder, set(inputs["resolved"].keys()))
    safety = preflight_output_safety(args.overwrite)
    write_preflight_report(pilot, {"counts": inputs["counts"],
                                   "dicom_root_source": resolution.source,
                                   "environment": env, "output_safety": safety})
    LOG.info("PREFLIGHT PASS (no conversion performed)")
    return 0


def run_full(args: "Args") -> int:
    pilot = load_pilot()
    resolution = P.resolve_dicom_root(args.dicom_root, os.environ.get(P.ENV_VAR_NAME))
    inputs = preflight_inputs(pilot, resolution.root)
    preflight_environment(args.jpeg2000_decoder, set(inputs["resolved"].keys()))
    preflight_output_safety(args.overwrite)

    # phase2A dimensions for cross-check.
    meta: Dict[str, Dict[str, int]] = {}
    with open(PHASE2A_META, "r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            meta[r["image_id"]] = {"w": int(float(r["image_width"])),
                                   "h": int(float(r["image_height"]))}

    STAGING_TRAIN.mkdir(parents=True, exist_ok=True)
    coco = inputs["coco"]
    id_to_coco = {im["original_image_id"]: im for im in coco["images"]}
    resolved = inputs["resolved"]

    # (4) SHA-256 snapshot of every source DICOM + canonical inputs BEFORE decode.
    input_snapshot = snapshot_input_hashes(pilot, resolved)

    # Conversion writes only to staging.
    records: List[Dict[str, Any]] = []
    for oid, path in resolved.items():
        rec = convert_one_image(pilot, oid, path, id_to_coco[oid], meta[oid],
                                STAGING_TRAIN, args.jpeg2000_decoder)
        records.append(rec)

    # Sources must be byte-identical after conversion.
    verified = verify_input_hashes(pilot, resolved, input_snapshot)

    ctx = {"coco": coco, "coco_sha256": inputs["coco_sha256"],
           "protocol_evidence": inputs["protocol_evidence"],
           "decision_sha256": inputs["decision_sha256"], "meta": meta, "errors": [],
           "resolved": resolved, "input_snapshot": input_snapshot,
           "input_hash_verification": verified}

    derivative = build_coco_jpg_derivative(pilot, coco, COCO_JPG_STAGING)
    validation = validate_full_outputs(pilot, ctx, derivative, records)

    # (2) Commit order: write ALL candidate evidence BEFORE any promotion, then
    # gate on its presence, so a report-write failure never leaves a new final
    # dataset lacking evidence.
    build_full_mapping(pilot, records, inputs["protocol_evidence"]["protocol_sha256"],
                       inputs["decision_sha256"], REPORTS_DIR)
    validate_mapping_artifacts(pilot, ctx, records, REPORTS_DIR)
    write_full_evidence(pilot, ctx, records, validation)
    require_evidence_present(REPORTS_DIR)

    # Final source-immutability check immediately before promotion.
    verify_input_hashes(pilot, resolved, input_snapshot)

    # The post-promotion record is written INSIDE the promotion transaction so a
    # record-write failure rolls back the freshly committed images + COCO. A
    # PRIOR record is atomically backed up and restored byte-for-byte on
    # rollback; a stale recovery backup is never overwritten.
    promotion_record_path = REPORTS_DIR / "phase2D1B_full_promotion.json"
    promotion_record_backup = REPORTS_DIR / "phase2D1B_full_promotion.json.prior"
    txn = PromotionRecordTransaction(pilot, ctx, promotion_record_path,
                                     promotion_record_backup)
    # (2) Stale recovery-artefact guard BEFORE promotion (applies whether or not
    # a current promotion record exists).
    txn.check_no_stale_backup()

    promoted = promote_outputs(args.overwrite, _post_commit=txn.finalize,
                               _post_commit_rollback=txn.rollback)
    # ---- Dataset + COCO are now COMMITTED. Nothing below rolls back the
    # dataset; failures here are reported as post-commit conditions. ----

    # Drop the prior-record backup and capture any cleanup failure so it can be
    # recorded in the structured audit (not only logged). A prior-record backup
    # that cannot be removed keeps backup_cleanup_clean=false.
    record_backup_warning = txn.commit()
    cleanup_warnings = list(promoted.get("backup_cleanup_warnings", []))
    if record_backup_warning:
        cleanup_warnings.append(record_backup_warning)

    # Persist backup-cleanup warnings into a STRUCTURED audit artefact. An
    # audit-write failure after commit is a distinct post-commit condition and
    # must NOT be described as a rollback.
    try:
        write_cleanup_audit(pilot, promoted, cleanup_warnings)
    except Exception as exc:
        raise FullError(
            f"promotion_committed_but_cleanup_audit_write_failed: {exc!r} "
            "(dataset + COCO derivative ARE COMMITTED and were NOT rolled back; "
            "a manual cleanup-audit is required)")

    LOG.info("FULL CONVERSION EXECUTED (quality 95). phase_status remains OPEN; "
             "full_conversion_completed stays false pending GPT review; no "
             "training authorized.")
    return 0


# ========================================================================= #
# CLI                                                                        #
# ========================================================================= #
@dataclass
class Args:
    preflight_only: bool
    execute_full: bool
    acknowledge_full_scope: Optional[int]
    jpeg_quality: Optional[int]
    dicom_root: Optional[str]
    jpeg2000_decoder: str
    overwrite: bool


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Phase 2D.1B-Full controlled-scope DICOM->JPG conversion.",
        add_help=True)
    ap.add_argument("--preflight-only", action="store_true",
                    help="Run input/environment/output-safety preflight and exit.")
    ap.add_argument("--execute-full", action="store_true",
                    help="Run the full conversion (requires explicit opt-in flags).")
    ap.add_argument("--acknowledge-full-scope", type=int, default=None,
                    help="Must equal the controlled scope (4894) for --execute-full.")
    ap.add_argument("--jpeg-quality", type=int, default=None,
                    help="Must be exactly 95 for --execute-full.")
    ap.add_argument("--dicom-root", default=None,
                    help="Override DICOM root (else VINBIGDATA_DICOM_ROOT).")
    ap.add_argument("--jpeg2000-decoder", default="pylibjpeg",
                    choices=["pylibjpeg", "gdcm", "pillow"],
                    help="Explicit JPEG2000 backend (no silent fallback).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Allow overwriting an existing final output (off by default).")
    return ap


def parse_args(argv: Optional[Sequence[str]] = None) -> Tuple[Optional[Args], argparse.ArgumentParser]:
    ap = build_parser()
    ns = ap.parse_args(argv)
    return Args(ns.preflight_only, ns.execute_full, ns.acknowledge_full_scope,
                ns.jpeg_quality, ns.dicom_root, ns.jpeg2000_decoder, ns.overwrite), ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args, ap = parse_args(argv)

    # Exactly one mode; no default runs the full conversion.
    if args.preflight_only == args.execute_full:
        # Neither or both -> show help and exit non-zero.
        ap.print_help(sys.stderr)
        return 2

    try:
        if args.preflight_only:
            return run_preflight(args)
        # --execute-full requires explicit, exact opt-in.
        if args.acknowledge_full_scope != LOCKED_SCOPE:
            raise FullError(
                f"--acknowledge-full-scope must equal {LOCKED_SCOPE}; got "
                f"{args.acknowledge_full_scope}")
        if args.jpeg_quality != LOCKED_QUALITY:
            raise FullError(
                f"--jpeg-quality must be exactly {LOCKED_QUALITY}; got {args.jpeg_quality}")
        return run_full(args)
    except P.Phase2D1BError as exc:
        LOG.error("hard fail (%s): %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
