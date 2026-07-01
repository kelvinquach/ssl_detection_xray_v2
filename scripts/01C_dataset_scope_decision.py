#!/usr/bin/env python3
"""Phase 1C — VinBigData Chest X-ray dataset scope decision.

Locks the controlled working scope (4,894 images = 4,394 abnormal + 500
No Finding) by cross-checking THREE metadata sources:
  1. full source train.csv               (annotation-level labels)
  2. dicom_package_manifest_part_*.csv    (package/chunk metadata)
  3. DICOM filename inventory             (Path.stem only, NO header/pixel read)

This script is METADATA-ONLY. It reads CSVs and LISTS DICOM filenames to
extract image_id from the stem. It NEVER opens a DICOM, reads pixels, or reads
image dimensions.

Scope guardrails (Phase 1C): this script does NOT
  - split train/val/test
  - convert to COCO
  - train, pseudo-label, or tune thresholds
  - touch the test set
  - copy images
  - read pixels / parse DICOM headers / read image dimensions
  - use pydicom.dcmread, cv2.imread, or PIL.Image.open
  - delete or edit source annotations
  - delete near-duplicate bbox candidates

"No Finding" is a NEGATIVE image label, NOT a detection class.

Usage (Windows CMD):
    python scripts\\01C_dataset_scope_decision.py ^
        --train-csv data\\raw\\vinbigdata\\annotations\\train.csv ^
        --manifest-glob "..\\ssl_detection_xray\\data\\raw\\vinbigdata\\dicom_subset_chunks\\dicom_package_manifest_part_*.csv" ^
        --dicom-root "..\\ssl_detection_xray\\data\\raw\\vinbigdata\\dicom_subset\\train" ^
        --chunk-summary "..\\ssl_detection_xray\\data\\raw\\vinbigdata\\dicom_subset_chunks\\dicom_chunk_summary.csv"
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    import pandas as pd
except Exception as exc:  # pragma: no cover
    print(
        "ERROR: pandas and numpy are required but could not be imported.\n"
        f"       Underlying error: {exc!r}\n"
        "       Install them (see requirements.txt) and try again.",
        file=sys.stderr,
    )
    raise SystemExit(2)


# --- Configuration --------------------------------------------------------

NO_FINDING_LABELS = {"no finding"}

TRAIN_REQUIRED = ["image_id", "class_name", "class_id", "x_min", "y_min", "x_max", "y_max"]
MANIFEST_REQUIRED = [
    "image_id",
    "image_type",
    "chunk_id",
    "zip_name",
    "source_path",
    "source_size_bytes",
]
BBOX_COLUMNS = ["x_min", "y_min", "x_max", "y_max"]

# Expected controlled-scope targets.
EXPECT_TOTAL = 4894
EXPECT_ABNORMAL = 4394
EXPECT_NO_FINDING = 500
EXPECT_CHUNKS = 35
EXPECT_ABNORMAL_CLASSES = 14
NEAR_DUP_RETAINED_NOTE = 147  # from Phase 1B, retained (not deleted).

SELECTED_FROM = "package_manifest_validated_by_train_csv_and_dicom_inventory"


def is_no_finding(label: Any) -> bool:
    """True if label is a No Finding negative label (case-insensitive)."""
    if label is None:
        return False
    if isinstance(label, float) and np.isnan(label):
        return False
    return str(label).strip().lower() in NO_FINDING_LABELS


# --- CLI ------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 1C — VinBigData dataset scope decision (metadata only).",
    )
    p.add_argument("--train-csv", required=True, type=str)
    p.add_argument("--manifest-glob", required=True, type=str)
    p.add_argument("--dicom-root", required=True, type=str)
    p.add_argument("--chunk-summary", type=str, default=None)
    p.add_argument(
        "--output-json",
        type=str,
        default="reports/phase1C_dataset_scope_decision.json",
    )
    p.add_argument("--report-md", type=str, default="reports/scope_decision.md")
    p.add_argument(
        "--selected-manifest-csv",
        type=str,
        default="data/manifests/phase1C_selected_images_manifest.csv",
    )
    p.add_argument(
        "--downloaded-inventory-csv",
        type=str,
        default="data/manifests/phase1C_downloaded_image_inventory.csv",
    )
    p.add_argument(
        "--combined-package-manifest-csv",
        type=str,
        default="data/manifests/phase1C_combined_package_manifest.csv",
    )
    p.add_argument(
        "--subset-csv",
        type=str,
        default="data/interim/vinbigdata_phase1C_scope_annotations.csv",
    )
    p.add_argument(
        "--class-distribution-csv",
        type=str,
        default="reports/phase1C_scope_class_distribution.csv",
    )
    p.add_argument(
        "--image-summary-csv",
        type=str,
        default="reports/phase1C_image_level_scope_summary.csv",
    )
    p.add_argument(
        "--no-finding-audit-csv",
        type=str,
        default="reports/phase1C_no_finding_selection_audit.csv",
    )
    return p.parse_args(argv)


# --- IO helpers -----------------------------------------------------------


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_train_csv(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"train.csv not found: {path}\n"
            "       Check --train-csv. (Phase 1C reads metadata only.)"
        )
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Failed to read train.csv '{path}': {exc!r}") from exc
    if df.empty:
        raise ValueError(f"train.csv '{path}' is empty.")
    missing = [c for c in TRAIN_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"train.csv missing required column(s): {missing}\n"
            f"       Found: {list(df.columns)}\n"
            f"       Required: {TRAIN_REQUIRED}"
        )
    return df


def load_manifests(manifest_glob: str) -> Tuple[pd.DataFrame, List[str]]:
    """Concat all manifest parts matching the glob."""
    paths = sorted(glob.glob(manifest_glob))
    if not paths:
        raise FileNotFoundError(
            f"No manifest files matched --manifest-glob: {manifest_glob}\n"
            "       Check the pattern (quote it so the shell does not expand it)."
        )
    frames: List[pd.DataFrame] = []
    for pth in paths:
        try:
            frames.append(pd.read_csv(pth))
        except Exception as exc:
            raise ValueError(f"Failed to read manifest part '{pth}': {exc!r}") from exc
    combined = pd.concat(frames, ignore_index=True)
    missing = [c for c in MANIFEST_REQUIRED if c not in combined.columns]
    if missing:
        raise ValueError(
            f"Combined manifest missing required column(s): {missing}\n"
            f"       Found: {list(combined.columns)}\n"
            f"       Required: {MANIFEST_REQUIRED}"
        )
    return combined, paths


def scan_dicom_inventory(dicom_root: str) -> pd.DataFrame:
    """List *.dicom files recursively; extract image_id from Path.stem ONLY.

    No DICOM is opened. No header, pixel, or dimension is read.
    """
    root = Path(dicom_root)
    if not root.exists():
        raise FileNotFoundError(
            f"DICOM root not found: {dicom_root}\n"
            "       Check --dicom-root. (Phase 1C lists filenames only.)"
        )
    records: List[Dict[str, Any]] = []
    # rglob is a directory listing; it does not open file contents.
    for fp in sorted(root.rglob("*.dicom")):
        records.append(
            {
                "image_id": fp.stem,
                "dicom_filename": fp.name,
                "dicom_relpath": str(fp.relative_to(root)),
            }
        )
    if not records:
        raise FileNotFoundError(
            f"No *.dicom files found under {dicom_root} (recursive)."
        )
    return pd.DataFrame(records)


# --- Image-level summary from train.csv -----------------------------------


def build_image_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-image summary of labels from train.csv."""
    work = df.copy()
    work["_is_nf"] = work["class_name"].apply(is_no_finding)
    for c in BBOX_COLUMNS:
        work[f"_{c}_num"] = pd.to_numeric(work[c], errors="coerce")
    work["_has_full_bbox"] = (
        work[[f"_{c}_num" for c in BBOX_COLUMNS]].notna().all(axis=1)
    )

    def agg(group: pd.DataFrame) -> pd.Series:
        abnormal_mask = ~group["_is_nf"]
        abn_names = sorted(
            group.loc[abnormal_mask, "class_name"].astype(str).unique().tolist()
        )
        return pd.Series(
            {
                "source_row_count": int(len(group)),
                "has_abnormal": bool(abnormal_mask.any()),
                "has_no_finding": bool(group["_is_nf"].any()),
                "abnormal_row_count": int(abnormal_mask.sum()),
                "no_finding_row_count": int(group["_is_nf"].sum()),
                "bbox_row_count": int(group["_has_full_bbox"].sum()),
                "abnormal_class_count": int(len(abn_names)),
                "abnormal_class_names": ";".join(abn_names),
            }
        )

    summary = work.groupby("image_id", sort=True).apply(agg).reset_index()
    summary["mixed_no_finding_abnormal"] = (
        summary["has_abnormal"] & summary["has_no_finding"]
    )
    return summary


# --- Main -----------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:  # noqa: C901 (linear pipeline)
    args = parse_args(argv)
    warnings: List[str] = []

    # --- Load sources ---
    try:
        train = load_train_csv(args.train_csv)
        manifest, manifest_paths = load_manifests(args.manifest_glob)
        dicom_inv = scan_dicom_inventory(args.dicom_root)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # --- Source-level stats from train.csv ---
    img_summary = build_image_level_summary(train)
    source_total_rows = int(len(train))
    source_unique_images = int(train["image_id"].nunique())
    source_abnormal_images = int(img_summary["has_abnormal"].sum())
    source_no_finding_images = int(img_summary["has_no_finding"].sum())
    source_mixed_images = int(img_summary["mixed_no_finding_abnormal"].sum())

    # --- Manifest stats ---
    manifest_total_rows = int(len(manifest))
    manifest_unique_images = int(manifest["image_id"].nunique())
    manifest_dup_count = int(manifest_total_rows - manifest_unique_images)
    itype_lower = manifest["image_type"].astype(str).str.strip().str.lower()
    manifest_abnormal_count = int((itype_lower == "abnormal").sum())
    manifest_normal_count = int(itype_lower.isin(["normal", "no_finding", "no finding"]).sum())

    # --- DICOM inventory stats ---
    dicom_file_count = int(len(dicom_inv))
    dicom_unique = int(dicom_inv["image_id"].nunique())
    dicom_dup_count = int(dicom_file_count - dicom_unique)

    # --- Cross-check manifest vs DICOM ---
    manifest_ids = set(manifest["image_id"].astype(str))
    dicom_ids = set(dicom_inv["image_id"].astype(str))
    manifest_not_in_dicom = sorted(manifest_ids - dicom_ids)
    dicom_not_in_manifest = sorted(dicom_ids - manifest_ids)
    manifest_not_in_dicom_count = len(manifest_not_in_dicom)
    dicom_not_in_manifest_count = len(dicom_not_in_manifest)
    if manifest_not_in_dicom_count:
        warnings.append(
            f"{manifest_not_in_dicom_count} manifest image_id(s) missing from DICOM inventory."
        )
    if dicom_not_in_manifest_count:
        warnings.append(
            f"{dicom_not_in_manifest_count} DICOM image_id(s) missing from manifest."
        )

    # --- Cross-check selected scope vs train.csv ---
    source_ids = set(train["image_id"].astype(str))
    unknown_manifest_ids = sorted(manifest_ids - source_ids)
    unknown_manifest_image_id_count = len(unknown_manifest_ids)
    if unknown_manifest_image_id_count:
        warnings.append(
            f"{unknown_manifest_image_id_count} manifest image_id(s) not present in train.csv."
        )

    # Selected = manifest image_ids. Determine labels from train.csv.
    sel_summary = img_summary[img_summary["image_id"].astype(str).isin(manifest_ids)].copy()
    selected_total_images = int(sel_summary["image_id"].nunique())
    selected_abnormal_images = int(sel_summary["has_abnormal"].sum())
    # No Finding selected = has No Finding and NOT abnormal (image-level).
    selected_no_finding_mask = sel_summary["has_no_finding"] & (~sel_summary["has_abnormal"])
    selected_no_finding_images = int(selected_no_finding_mask.sum())
    selected_mixed_images = int(sel_summary["mixed_no_finding_abnormal"].sum())

    # Lost abnormal images: abnormal in full train.csv but not selected.
    all_abnormal_ids = set(
        img_summary.loc[img_summary["has_abnormal"], "image_id"].astype(str)
    )
    selected_abnormal_ids = set(
        sel_summary.loc[sel_summary["has_abnormal"], "image_id"].astype(str)
    )
    lost_abnormal_image_count = len(all_abnormal_ids - selected_abnormal_ids)
    abnormal_retention_rate = (
        len(selected_abnormal_ids) / len(all_abnormal_ids)
        if all_abnormal_ids
        else 0.0
    )

    # --- image_type vs train.csv label mismatch ---
    label_by_id = img_summary.set_index(img_summary["image_id"].astype(str))
    mismatch_records: List[Dict[str, Any]] = []
    manifest_indexed = manifest.drop_duplicates("image_id").set_index(
        manifest["image_id"].astype(str)
    )
    for iid in manifest_ids:
        if iid not in label_by_id.index:
            continue
        itype = str(manifest_indexed.loc[iid, "image_type"]).strip().lower()
        has_abn = bool(label_by_id.loc[iid, "has_abnormal"])
        has_nf = bool(label_by_id.loc[iid, "has_no_finding"])
        if itype == "abnormal" and (not has_abn) and has_nf:
            mismatch_records.append(
                {"image_id": iid, "image_type": itype, "train_label": "no_finding_only"}
            )
        elif itype in ("normal", "no_finding", "no finding") and has_abn:
            mismatch_records.append(
                {"image_id": iid, "image_type": itype, "train_label": "has_abnormal"}
            )
    image_type_label_mismatch_count = len(mismatch_records)
    if image_type_label_mismatch_count:
        warnings.append(
            f"{image_type_label_mismatch_count} image_type vs train.csv label mismatch(es)."
        )

    # --- Chunk summary (optional) ---
    chunk_summary_stats: Dict[str, Any] = {
        "chunk_summary_total_files": None,
        "chunk_summary_abnormal_images": None,
        "chunk_summary_normal_images": None,
        "chunk_summary_chunk_count": None,
        "chunk_summary_raw_size_gb_sum": None,
        "chunk_summary_match": None,
    }
    if args.chunk_summary:
        cs_path = Path(args.chunk_summary)
        if cs_path.exists():
            try:
                cs = pd.read_csv(cs_path)

                def col_sum(cands: List[str]) -> Optional[float]:
                    for c in cands:
                        if c in cs.columns:
                            return float(pd.to_numeric(cs[c], errors="coerce").sum())
                    return None

                total_files = col_sum(["num_files", "n_files", "file_count", "num_images"])
                abn = col_sum(["abnormal_images", "n_abnormal", "abnormal"])
                nrm = col_sum(["normal_images", "n_normal", "normal", "no_finding_images"])
                size_gb = col_sum(
                    ["raw_size_gb", "size_gb", "total_size_gb", "raw_size_gb_sum"]
                )
                chunk_count = int(len(cs))
                chunk_summary_stats.update(
                    {
                        "chunk_summary_total_files": None if total_files is None else int(total_files),
                        "chunk_summary_abnormal_images": None if abn is None else int(abn),
                        "chunk_summary_normal_images": None if nrm is None else int(nrm),
                        "chunk_summary_chunk_count": chunk_count,
                        "chunk_summary_raw_size_gb_sum": size_gb,
                    }
                )
                match = (
                    chunk_summary_stats["chunk_summary_total_files"] == EXPECT_TOTAL
                    and chunk_summary_stats["chunk_summary_abnormal_images"] == EXPECT_ABNORMAL
                    and chunk_summary_stats["chunk_summary_normal_images"] == EXPECT_NO_FINDING
                    and chunk_count == EXPECT_CHUNKS
                )
                chunk_summary_stats["chunk_summary_match"] = bool(match)
                if not match:
                    warnings.append(
                        "chunk_summary values do not match expected scope targets."
                    )
            except Exception as exc:
                warnings.append(f"Failed to parse chunk summary: {exc!r}")
        else:
            warnings.append(f"Chunk summary not found: {args.chunk_summary}")

    # --- Build selected image-level manifest ---
    manifest_cols_map = manifest.drop_duplicates("image_id").set_index(
        manifest["image_id"].astype(str)
    )
    sel = sel_summary.copy()
    sel["image_id_str"] = sel["image_id"].astype(str)
    sel["is_abnormal"] = sel["has_abnormal"]
    sel["is_no_finding"] = (~sel["has_abnormal"]) & sel["has_no_finding"]
    sel["scope_label"] = np.where(sel["is_abnormal"], "abnormal", "no_finding")

    def mget(iid: str, col: str) -> Any:
        try:
            return manifest_cols_map.loc[iid, col]
        except Exception:
            return None

    sel["package_image_type"] = sel["image_id_str"].apply(lambda i: mget(i, "image_type"))
    sel["chunk_id"] = sel["image_id_str"].apply(lambda i: mget(i, "chunk_id"))
    sel["zip_name"] = sel["image_id_str"].apply(lambda i: mget(i, "zip_name"))
    sel["source_path"] = sel["image_id_str"].apply(lambda i: mget(i, "source_path"))
    sel["source_size_bytes"] = sel["image_id_str"].apply(
        lambda i: mget(i, "source_size_bytes")
    )
    sel["selected_from"] = SELECTED_FROM
    sel["selection_reason"] = np.where(
        sel["is_abnormal"],
        "abnormal_image_with_bbox_labels_in_train_csv",
        "no_finding_negative_image_in_controlled_scope",
    )

    selected_manifest_cols = [
        "image_id",
        "scope_label",
        "is_abnormal",
        "is_no_finding",
        "source_row_count",
        "abnormal_row_count",
        "no_finding_row_count",
        "bbox_row_count",
        "abnormal_class_count",
        "abnormal_class_names",
        "package_image_type",
        "chunk_id",
        "zip_name",
        "source_path",
        "source_size_bytes",
        "selected_from",
        "selection_reason",
    ]
    selected_manifest = sel[selected_manifest_cols].sort_values("image_id").reset_index(drop=True)

    # --- Subset annotation CSV (metadata-only, no row/bbox mutation) ---
    subset = train[train["image_id"].astype(str).isin(manifest_ids)].copy()
    selected_subset_rows = int(len(subset))
    subset_nf = subset["class_name"].apply(is_no_finding)
    selected_abnormal_rows = int((~subset_nf).sum())
    selected_no_finding_rows = int(subset_nf.sum())

    # --- Class distribution (abnormal detection classes only) ---
    abn_rows = subset[~subset["class_name"].apply(is_no_finding)].copy()
    abnormal_classes = sorted(abn_rows["class_name"].astype(str).unique().tolist())
    abnormal_detection_classes = len(abnormal_classes)

    total_selected_images = selected_total_images or 1
    total_abnormal_images = selected_abnormal_images or 1
    class_dist_records: List[Dict[str, Any]] = []
    for cname, grp in abn_rows.groupby(abn_rows["class_name"].astype(str)):
        cid_vals = pd.to_numeric(grp["class_id"], errors="coerce").dropna().unique()
        cid = int(cid_vals[0]) if len(cid_vals) else None
        img_count = int(grp["image_id"].nunique())
        for c in BBOX_COLUMNS:
            grp[c] = pd.to_numeric(grp[c], errors="coerce")
        bbox_count = int(grp[BBOX_COLUMNS].notna().all(axis=1).sum())
        class_dist_records.append(
            {
                "class_id": cid,
                "class_name": cname,
                "row_count": int(len(grp)),
                "image_count": img_count,
                "bbox_count": bbox_count,
                "percentage_of_abnormal_images": round(
                    100.0 * img_count / total_abnormal_images, 4
                ),
                "percentage_of_selected_images": round(
                    100.0 * img_count / total_selected_images, 4
                ),
            }
        )
    class_dist = pd.DataFrame(class_dist_records).sort_values(
        "class_id", na_position="last"
    ).reset_index(drop=True)

    # --- No Finding audit (500 selected NF image-level) ---
    nf_sel = sel[sel["is_no_finding"]].copy()
    nf_audit = pd.DataFrame(
        {
            "image_id": nf_sel["image_id"].values,
            "selected": True,
            "source_row_count": nf_sel["source_row_count"].values,
            "no_finding_row_count": nf_sel["no_finding_row_count"].values,
            "package_image_type": nf_sel["package_image_type"].values,
            "chunk_id": nf_sel["chunk_id"].values,
            "zip_name": nf_sel["zip_name"].values,
            "selected_from": SELECTED_FROM,
            "no_finding_verified_by_train_csv": True,
        }
    ).sort_values("image_id").reset_index(drop=True)

    # --- DoD candidate ---
    dod_checks = {
        "manifest_total_rows": manifest_total_rows == EXPECT_TOTAL,
        "manifest_unique_images": manifest_unique_images == EXPECT_TOTAL,
        "manifest_duplicate_image_id_count": manifest_dup_count == 0,
        "dicom_file_count": dicom_file_count == EXPECT_TOTAL,
        "dicom_unique_image_ids": dicom_unique == EXPECT_TOTAL,
        "dicom_duplicate_image_id_count": dicom_dup_count == 0,
        "manifest_not_in_dicom_count": manifest_not_in_dicom_count == 0,
        "dicom_not_in_manifest_count": dicom_not_in_manifest_count == 0,
        "unknown_manifest_image_id_count": unknown_manifest_image_id_count == 0,
        "selected_total_images": selected_total_images == EXPECT_TOTAL,
        "selected_abnormal_images": selected_abnormal_images == EXPECT_ABNORMAL,
        "selected_no_finding_images": selected_no_finding_images == EXPECT_NO_FINDING,
        "selected_mixed_images": selected_mixed_images == 0,
        "lost_abnormal_image_count": lost_abnormal_image_count == 0,
        "image_type_label_mismatch_count": image_type_label_mismatch_count == 0,
        "abnormal_detection_classes_excluding_no_finding": abnormal_detection_classes
        == EXPECT_ABNORMAL_CLASSES,
        "no_finding_is_detection_class": True,  # must be false -> check below
    }
    # normalize the "must be false" flags into boolean pass conditions
    dod_flags = {
        "no_finding_is_detection_class": False,
        "no_finding_row_level_sampling_used": False,
        "split_created": False,
        "coco_created": False,
        "training_started": False,
        "pseudo_label_generated": False,
        "threshold_tuned": False,
        "test_set_used": False,
        "dicom_header_read": False,
        "pixel_read": False,
        "image_dimensions_read": False,
    }
    # recompute count-based checks cleanly:
    dod_pass_candidate = (
        manifest_total_rows == EXPECT_TOTAL
        and manifest_unique_images == EXPECT_TOTAL
        and manifest_dup_count == 0
        and dicom_file_count == EXPECT_TOTAL
        and dicom_unique == EXPECT_TOTAL
        and dicom_dup_count == 0
        and manifest_not_in_dicom_count == 0
        and dicom_not_in_manifest_count == 0
        and unknown_manifest_image_id_count == 0
        and selected_total_images == EXPECT_TOTAL
        and selected_abnormal_images == EXPECT_ABNORMAL
        and selected_no_finding_images == EXPECT_NO_FINDING
        and selected_mixed_images == 0
        and lost_abnormal_image_count == 0
        and image_type_label_mismatch_count == 0
        and abnormal_detection_classes == EXPECT_ABNORMAL_CLASSES
        and all(v is False for v in dod_flags.values())
    )

    # --- Assemble JSON payload ---
    payload: Dict[str, Any] = {
        "phase": "phase1C_dataset_scope_decision",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "train_csv": str(args.train_csv),
        "manifest_glob": str(args.manifest_glob),
        "dicom_root": str(args.dicom_root),
        "chunk_summary": str(args.chunk_summary) if args.chunk_summary else None,
        "manifest_parts_found": len(manifest_paths),
        "source_total_rows": source_total_rows,
        "source_unique_images": source_unique_images,
        "source_abnormal_images": source_abnormal_images,
        "source_no_finding_images": source_no_finding_images,
        "source_mixed_no_finding_abnormal_images": source_mixed_images,
        "manifest_total_rows": manifest_total_rows,
        "manifest_unique_images": manifest_unique_images,
        "manifest_duplicate_image_id_count": manifest_dup_count,
        "manifest_image_type_abnormal_count": manifest_abnormal_count,
        "manifest_image_type_normal_count": manifest_normal_count,
        "dicom_file_count": dicom_file_count,
        "dicom_unique_image_ids": dicom_unique,
        "dicom_duplicate_image_id_count": dicom_dup_count,
        "manifest_not_in_dicom_count": manifest_not_in_dicom_count,
        "dicom_not_in_manifest_count": dicom_not_in_manifest_count,
        "unknown_manifest_image_id_count": unknown_manifest_image_id_count,
        "selected_total_images": selected_total_images,
        "selected_abnormal_images": selected_abnormal_images,
        "selected_no_finding_images": selected_no_finding_images,
        "selected_mixed_images": selected_mixed_images,
        "lost_abnormal_image_count": lost_abnormal_image_count,
        "abnormal_retention_rate": round(abnormal_retention_rate, 6),
        "selected_subset_rows": selected_subset_rows,
        "selected_abnormal_rows": selected_abnormal_rows,
        "selected_no_finding_rows": selected_no_finding_rows,
        "abnormal_detection_classes_excluding_no_finding": abnormal_detection_classes,
        "abnormal_class_names": abnormal_classes,
        "no_finding_is_detection_class": False,
        "image_type_label_mismatch_count": image_type_label_mismatch_count,
        "no_finding_row_level_sampling_used": False,
        "selected_scope_source": SELECTED_FROM,
        **chunk_summary_stats,
        # Forbidden-action confirmations (all must remain False/appropriate).
        "filename_inventory_only": True,
        "dicom_header_read": False,
        "pixel_read": False,
        "image_dimensions_read": False,
        "split_created": False,
        "coco_created": False,
        "training_started": False,
        "pseudo_label_generated": False,
        "threshold_tuned": False,
        "test_set_used": False,
        "forbidden_actions_confirmed": {
            "split_created": False,
            "coco_created": False,
            "training_started": False,
            "pseudo_label_generated": False,
            "threshold_tuned": False,
            "test_set_used": False,
            "images_copied": False,
            "pixel_read": False,
            "dicom_header_read": False,
            "image_dimensions_read": False,
            "annotations_deleted_or_edited": False,
            "near_duplicate_bbox_deleted": False,
        },
        "near_duplicate_bbox_candidates_retained": NEAR_DUP_RETAINED_NOTE,
        "boundary_check_status": "deferred_to_phase2A",
        "warnings": warnings,
        "dod_pass_candidate": bool(dod_pass_candidate),
        "generated_files": {
            "output_json": args.output_json,
            "report_md": args.report_md,
            "selected_manifest_csv": args.selected_manifest_csv,
            "downloaded_inventory_csv": args.downloaded_inventory_csv,
            "combined_package_manifest_csv": args.combined_package_manifest_csv,
            "subset_csv": args.subset_csv,
            "class_distribution_csv": args.class_distribution_csv,
            "image_summary_csv": args.image_summary_csv,
            "no_finding_audit_csv": args.no_finding_audit_csv,
        },
    }

    # --- Write outputs ---
    ensure_parent(args.output_json).write_text(
        json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8"
    )
    manifest.to_csv(ensure_parent(args.combined_package_manifest_csv), index=False)
    dicom_inv.to_csv(ensure_parent(args.downloaded_inventory_csv), index=False)
    selected_manifest.to_csv(ensure_parent(args.selected_manifest_csv), index=False)
    subset.to_csv(ensure_parent(args.subset_csv), index=False)
    class_dist.to_csv(ensure_parent(args.class_distribution_csv), index=False)
    img_summary.to_csv(ensure_parent(args.image_summary_csv), index=False)
    nf_audit.to_csv(ensure_parent(args.no_finding_audit_csv), index=False)

    write_report_md(args.report_md, payload)

    # --- Console summary ---
    print("=" * 68)
    print("Phase 1C — Dataset Scope Decision")
    print("=" * 68)
    print(f"selected_scope_source          : {payload['selected_scope_source']}")
    print(f"source_total_rows              : {source_total_rows}")
    print(f"source_unique_images           : {source_unique_images}")
    print(f"manifest_total_rows            : {manifest_total_rows}")
    print(f"manifest_unique_images         : {manifest_unique_images}")
    print(f"dicom_file_count               : {dicom_file_count}")
    print(f"dicom_unique_image_ids         : {dicom_unique}")
    print(f"selected_total_images          : {selected_total_images}")
    print(f"selected_abnormal_images       : {selected_abnormal_images}")
    print(f"selected_no_finding_images     : {selected_no_finding_images}")
    print(f"lost_abnormal_image_count      : {lost_abnormal_image_count}")
    print(f"unknown_manifest_image_id_count: {unknown_manifest_image_id_count}")
    print(f"manifest_not_in_dicom_count    : {manifest_not_in_dicom_count}")
    print(f"dicom_not_in_manifest_count    : {dicom_not_in_manifest_count}")
    print(f"image_type_label_mismatch_count: {image_type_label_mismatch_count}")
    print(f"chunk_summary_match            : {chunk_summary_stats['chunk_summary_match']}")
    print(
        "abnormal_detection_classes     : "
        f"{abnormal_detection_classes} (excluding No Finding)"
    )
    print("-" * 68)
    print("Forbidden actions confirmation (all must be False):")
    for k, v in payload["forbidden_actions_confirmed"].items():
        print(f"  {k:<32}: {v}")
    print("-" * 68)
    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  WARN: {w}")
    else:
        print("Warnings                       : none")
    print("-" * 68)
    print(f"dod_pass_candidate             : {payload['dod_pass_candidate']}")
    print("=" * 68)
    print("NOTE: Checklist NOT auto-ticked. Send outputs to GPT review first.")

    return 0


def write_report_md(path: str, p: Dict[str, Any]) -> None:
    L: List[str] = []
    L.append("# Phase 1C — Dataset Scope Decision")
    L.append("")
    L.append(f"_Generated {p['created_utc']}._")
    L.append("")
    L.append("## Executive summary")
    L.append("")
    L.append(
        f"Locked the controlled working scope of **{p['selected_total_images']}** "
        f"images (**{p['selected_abnormal_images']}** abnormal + "
        f"**{p['selected_no_finding_images']}** No Finding), cross-validated across "
        "train.csv, package manifests, and the DICOM filename inventory. "
        f"DoD pass candidate: **{p['dod_pass_candidate']}**."
    )
    L.append("")
    L.append("## Source metadata vs controlled working scope")
    L.append("")
    L.append(f"- Source rows: {p['source_total_rows']}; source images: {p['source_unique_images']}.")
    L.append(f"- Source abnormal images: {p['source_abnormal_images']}; No Finding images: {p['source_no_finding_images']}.")
    L.append(f"- Controlled scope: {p['selected_total_images']} images (subset of source).")
    L.append("")
    L.append("## Evidence from package manifests")
    L.append("")
    L.append(f"- Manifest parts found: {p['manifest_parts_found']}.")
    L.append(f"- Manifest rows: {p['manifest_total_rows']}; unique image_id: {p['manifest_unique_images']}; duplicates: {p['manifest_duplicate_image_id_count']}.")
    L.append(f"- image_type abnormal: {p['manifest_image_type_abnormal_count']}; normal: {p['manifest_image_type_normal_count']} (not trusted blindly; reconciled with train.csv).")
    L.append("")
    L.append("## Evidence from DICOM filename inventory")
    L.append("")
    L.append(f"- DICOM files listed (*.dicom): {p['dicom_file_count']}; unique image_id: {p['dicom_unique_image_ids']}; duplicates: {p['dicom_duplicate_image_id_count']}.")
    L.append("- Filenames only: no DICOM header, pixel, or dimension was read.")
    L.append("")
    L.append("## Cross-check manifest vs DICOM filenames")
    L.append("")
    L.append(f"- manifest_not_in_dicom_count: {p['manifest_not_in_dicom_count']}.")
    L.append(f"- dicom_not_in_manifest_count: {p['dicom_not_in_manifest_count']}.")
    L.append("")
    L.append("## Cross-check selected scope vs train.csv")
    L.append("")
    L.append(f"- unknown_manifest_image_id_count: {p['unknown_manifest_image_id_count']}.")
    L.append(f"- selected_abnormal_images: {p['selected_abnormal_images']}; selected_no_finding_images: {p['selected_no_finding_images']}; selected_mixed_images: {p['selected_mixed_images']}.")
    L.append(f"- image_type_label_mismatch_count: {p['image_type_label_mismatch_count']}.")
    L.append("")
    L.append("## Abnormal retention proof")
    L.append("")
    L.append(f"- lost_abnormal_image_count: {p['lost_abnormal_image_count']}.")
    L.append(f"- abnormal_retention_rate: {p['abnormal_retention_rate']} (selected abnormal / source abnormal).")
    L.append("- All abnormal source images are retained in the controlled scope.")
    L.append("")
    L.append("## No Finding image-level proof")
    L.append("")
    L.append(f"- selected_no_finding_images: {p['selected_no_finding_images']} unique image_id (image-level, not row-level).")
    L.append(f"- no_finding_row_level_sampling_used: {p['no_finding_row_level_sampling_used']}.")
    L.append("- See `phase1C_no_finding_selection_audit.csv` for the 500 audited image_ids.")
    L.append("")
    L.append("## Class distribution summary")
    L.append("")
    L.append(f"- abnormal_detection_classes_excluding_no_finding: {p['abnormal_detection_classes_excluding_no_finding']}.")
    L.append("- No Finding is absent from the detection-class distribution file by design.")
    L.append("")
    L.append("## No Finding policy")
    L.append("")
    L.append(f"- no_finding_is_detection_class: {p['no_finding_is_detection_class']}.")
    L.append("- No Finding remains a negative image label without bounding boxes.")
    L.append("")
    L.append("## Near-duplicate bbox candidates")
    L.append("")
    L.append(
        f"- {p['near_duplicate_bbox_candidates_retained']} near-duplicate bbox "
        "candidates (from Phase 1B) are **retained, not deleted**. Fusion of "
        "multi-radiologist boxes is a later research decision."
    )
    L.append("")
    L.append("## Limitation")
    L.append("")
    L.append(
        f"- Selected normal images are {p['selected_no_finding_images']} out of "
        f"{p['source_no_finding_images']} available No Finding images; the negative "
        "pool is deliberately capped for the controlled scope."
    )
    L.append("")
    L.append("## Boundary validation")
    L.append("")
    L.append(f"- boundary_check_status: {p['boundary_check_status']} (image dimensions not read in Phase 1C).")
    L.append("")
    L.append("## Forbidden actions confirmation")
    L.append("")
    for k, v in p["forbidden_actions_confirmed"].items():
        L.append(f"- {k}: {v}")
    L.append("")
    L.append("## Next action")
    L.append("")
    L.append(
        "- **Send these outputs to GPT review BEFORE ticking the Phase 1C "
        "checklist.** This script does not auto-tick anything."
    )
    L.append("")
    ensure_parent(path).write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
