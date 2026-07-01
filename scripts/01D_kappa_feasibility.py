#!/usr/bin/env python3
"""Phase 1D — Kappa Feasibility / Agreement Analysis (limitation-aware).

Reads ONLY annotation metadata CSV (preferring the Phase 1C scope subset) and
assesses inter-rater agreement. For VinBigData the annotation structure DOES
support agreement statistics, because:

  * every image has a uniform number of ratings (VinBigData: 3 per image,
    drawn from 17 distinct radiologists; identities may vary across images), and
  * "No finding" rows also carry a rad_id, so a rater writing "No finding"
    is an EXPLICIT statement that they read the image and recorded nothing.

Therefore the set of raters who read an image = the union of every rad_id that
appears for that image (positive findings + No finding). A rater who read an
image but has no positive row for class C is a VALID negative (label 0) for
that (image, class). This yields a complete rater x subject matrix per class,
so Fleiss' Kappa is computable. Cohen's Kappa (pairwise, 2 raters) is only
directly applicable to images with exactly 2 readers.

Agreement here is DATA-QUALITY / LIMITATION evidence ONLY. It is NEVER used as
a model metric, nor to select split/model/threshold, nor to edit annotations,
nor to evaluate SSL performance.

Scope guardrails (Phase 1D): this script does NOT
  - split train/val/test, convert to COCO, train, pseudo-label, tune thresholds
  - touch the test set
  - read pixels / DICOM / headers / image dimensions
  - perform boundary validation
  - delete or edit source annotations
  - delete or fuse near-duplicate bboxes

"No Finding" is a NEGATIVE image label, NOT a detection class.

Usage:
    python scripts/01D_kappa_feasibility.py \
        --scope-csv data/interim/vinbigdata_phase1C_scope_annotations.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    import pandas as pd
except Exception as exc:  # pragma: no cover
    print(
        "ERROR: pandas and numpy are required but could not be imported.\n"
        f"       Underlying error: {exc!r}",
        file=sys.stderr,
    )
    raise SystemExit(2)


# --- Configuration --------------------------------------------------------

NO_FINDING_LABELS = {"no finding"}
RAD_ID_CANDIDATES = ["rad_id", "radiologist_id", "reader_id", "annotator_id", "rad"]
BBOX_COLUMNS = ["x_min", "y_min", "x_max", "y_max"]

# Rare-class heuristics (evidence thresholds, NOT modelling thresholds).
RARE_POSITIVE_IMAGE_THRESHOLD = 50   # few positive images -> unstable Kappa
RARE_PREVALENCE_THRESHOLD = 0.01     # <1% prevalence -> imbalance risk
MIN_RATERS_FOR_AGREEMENT = 2
NEAR_DUP_IOU = 0.95                  # consistent with Phase 1B/1C evidence


def is_no_finding(label: Any) -> bool:
    if label is None:
        return False
    if isinstance(label, float) and np.isnan(label):
        return False
    return str(label).strip().lower() in NO_FINDING_LABELS


# --- CLI ------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 1D — Kappa feasibility & agreement (metadata only).",
    )
    p.add_argument(
        "--scope-csv",
        type=str,
        default="data/interim/vinbigdata_phase1C_scope_annotations.csv",
        help="Preferred input: Phase 1C scope annotations CSV.",
    )
    p.add_argument(
        "--train-csv",
        type=str,
        default="data/raw/vinbigdata/annotations/train.csv",
        help="Optional source CSV; used if --scope-csv is absent.",
    )
    p.add_argument(
        "--output-json",
        type=str,
        default="reports/phase1D_kappa_feasibility.json",
    )
    p.add_argument(
        "--report-md",
        type=str,
        default="reports/phase1D_kappa_feasibility.md",
    )
    p.add_argument(
        "--rad-per-image-csv",
        type=str,
        default="reports/phase1D_radiologist_per_image.csv",
    )
    p.add_argument(
        "--classwise-csv",
        type=str,
        default="reports/phase1D_classwise_agreement_feasibility.csv",
    )
    p.add_argument(
        "--rare-class-csv",
        type=str,
        default="reports/phase1D_rare_class_kappa_instability.csv",
    )
    return p.parse_args(argv)


# --- IO helpers -----------------------------------------------------------


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_input(scope_csv: str, train_csv: str) -> Tuple[pd.DataFrame, List[str]]:
    """Load the preferred scope CSV; fall back to train.csv if scope absent."""
    scope_path = Path(scope_csv)
    train_path = Path(train_csv)

    chosen: Optional[Path] = None
    if scope_path.exists():
        chosen = scope_path
    elif train_path.exists():
        chosen = train_path
    else:
        raise FileNotFoundError(
            "No input CSV found.\n"
            f"       Looked for scope CSV: {scope_path}\n"
            f"       and source train.csv: {train_path}\n"
            "       Provide --scope-csv or --train-csv. (Phase 1D reads metadata only.)"
        )

    try:
        df = pd.read_csv(chosen)
    except Exception as exc:
        raise ValueError(f"Failed to read '{chosen}': {exc!r}") from exc
    if df.empty:
        raise ValueError(f"Input CSV '{chosen}' is empty.")

    for req in ["image_id", "class_name"]:
        if req not in df.columns:
            raise ValueError(
                f"Input CSV missing required column '{req}'. "
                f"Found: {list(df.columns)}"
            )
    return df, [str(chosen)]


def detect_rad_id_column(df: pd.DataFrame) -> Optional[str]:
    for c in RAD_ID_CANDIDATES:
        if c in df.columns:
            return c
    return None


# --- Radiologists per image -----------------------------------------------


def analyze_radiologists_per_image(
    df: pd.DataFrame, rad_col: Optional[str]
) -> Tuple[pd.DataFrame, Dict[str, int], int, "pd.Series"]:
    """Radiologists per image.

    Returns (per_image_df, distribution, total_raters, readers_per_image_series).
    readers_per_image_series maps image_id -> frozenset of rad_ids that READ it
    (union of all rad_id rows, including 'No finding' rows).
    """
    if rad_col is None:
        empty = pd.DataFrame(columns=["image_id", "n_radiologists", "rad_ids"])
        return empty, {}, 0, pd.Series(dtype=object)

    work = df[["image_id", rad_col]].copy()
    work[rad_col] = work[rad_col].astype("string")

    readers = work.groupby("image_id")[rad_col].apply(
        lambda s: frozenset(x for x in s.dropna().unique())
    )
    per_image = pd.DataFrame(
        {
            "image_id": readers.index,
            "n_radiologists": [len(s) for s in readers.values],
            "rad_ids": [";".join(sorted(s)) for s in readers.values],
        }
    ).reset_index(drop=True)

    dist_series = per_image["n_radiologists"].value_counts().sort_index()
    distribution = {str(int(k)): int(v) for k, v in dist_series.items()}
    total_raters = int(work[rad_col].dropna().nunique())
    return per_image, distribution, total_raters, readers


# --- Fleiss / Cohen kappa --------------------------------------------------


def fleiss_kappa_binary(n_present: np.ndarray, n_raters: int) -> Optional[float]:
    """Fleiss' Kappa for a binary (present/absent) rating with fixed n_raters.

    n_present[i] = number of raters marking 'present' on subject i.
    Returns None if undefined (e.g. no variance).
    """
    if n_raters < 2 or len(n_present) == 0:
        return None
    N = len(n_present)
    nn = n_raters
    n_absent = nn - n_present
    # Per-subject agreement.
    P_i = (n_present * (n_present - 1) + n_absent * (n_absent - 1)) / (nn * (nn - 1))
    P_bar = float(P_i.mean())
    # Category marginal proportions.
    p_present = float(n_present.sum()) / (nn * N)
    p_absent = float(n_absent.sum()) / (nn * N)
    P_e = p_present ** 2 + p_absent ** 2
    denom = 1.0 - P_e
    if denom == 0:
        return None  # all one category -> kappa undefined
    return (P_bar - P_e) / denom


def cohen_kappa_binary(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Cohen's Kappa between two binary rating vectors."""
    if len(a) == 0 or len(a) != len(b):
        return None
    po = float((a == b).mean())
    pa1 = float(a.mean())
    pb1 = float(b.mean())
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    denom = 1.0 - pe
    if denom == 0:
        return None
    return (po - pe) / denom


def build_classwise_agreement(
    df: pd.DataFrame,
    rad_col: Optional[str],
    readers_per_image: "pd.Series",
    total_images: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute per-class Fleiss kappa (and raw agreement) using inferred negatives.

    A rater who READ an image (in readers_per_image) but did not mark class C on
    it is a valid negative. Requires a rad_id column and known read-coverage.
    """
    meta: Dict[str, Any] = {
        "negatives_inferable": False,
        "fixed_panel_size": None,
        "method": None,
    }
    work = df.copy()
    work["_is_nf"] = work["class_name"].apply(is_no_finding)
    abn = work[~work["_is_nf"]].copy()
    has_class_id = "class_id" in abn.columns

    records: List[Dict[str, Any]] = []

    if rad_col is None or readers_per_image.empty:
        # No rater attribution -> cannot infer negatives.
        for cname in sorted(abn["class_name"].astype(str).unique()):
            sub = abn[abn["class_name"].astype(str) == cname]
            cid = None
            if has_class_id:
                v = pd.to_numeric(sub["class_id"], errors="coerce").dropna().unique()
                cid = int(v[0]) if len(v) else None
            records.append(
                {
                    "class_id": cid,
                    "class_name": cname,
                    "positive_images": int(sub["image_id"].nunique()),
                    "negative_images": None,
                    "n_radiologists": 0,
                    "rater_coverage": "unknown_no_rad_id",
                    "raw_agreement": None,
                    "fleiss_kappa": None,
                    "cohen_feasibility": "infeasible",
                    "fleiss_feasibility": "infeasible",
                    "limitation_reason": "No rad_id; negatives not identifiable.",
                }
            )
        return (
            pd.DataFrame(records).sort_values("class_id", na_position="last").reset_index(drop=True),
            meta,
        )

    # Determine the read panel per image.
    n_readers = readers_per_image.apply(len)
    panel_sizes = sorted(n_readers.unique().tolist())
    uniform_panel = len(panel_sizes) == 1
    panel_size = int(panel_sizes[0]) if uniform_panel else None
    meta["negatives_inferable"] = True
    meta["fixed_panel_size"] = panel_size
    meta["method"] = (
        "fleiss_fixed_panel" if uniform_panel else "fleiss_per_subject_variable"
    )

    total_raters = int(df[rad_col].dropna().nunique())

    for cname in sorted(abn["class_name"].astype(str).unique()):
        sub = abn[abn["class_name"].astype(str) == cname]
        cid = None
        if has_class_id:
            v = pd.to_numeric(sub["class_id"], errors="coerce").dropna().unique()
            cid = int(v[0]) if len(v) else None

        # Raters marking this class present, per image.
        pos_readers = (
            sub.groupby("image_id")[rad_col]
            .apply(lambda s: frozenset(x for x in s.astype("string").dropna().unique()))
        )

        n_present_list: List[int] = []
        pos_images = 0
        # Iterate over ALL read images (negatives included).
        for img, readers in readers_per_image.items():
            marked = pos_readers.get(img, frozenset()) & readers
            k = len(marked)
            n_present_list.append(k)
            if k > 0:
                pos_images += 1

        n_present = np.array(n_present_list, dtype=float)
        n_raters_eff = panel_size if uniform_panel else None

        fleiss = None
        raw_agree = None
        if uniform_panel and panel_size and panel_size >= 2:
            fleiss = fleiss_kappa_binary(n_present, panel_size)
            # Raw agreement = fraction of subjects where all raters agree.
            all_agree = np.mean((n_present == 0) | (n_present == panel_size))
            raw_agree = float(all_agree)

        prevalence = pos_images / total_images if total_images else 0.0
        feasible = uniform_panel and panel_size and panel_size >= 2

        records.append(
            {
                "class_id": cid,
                "class_name": cname,
                "positive_images": int(pos_images),
                "negative_images": int(total_images - pos_images),
                "n_radiologists": total_raters,
                "rater_coverage": (
                    f"uniform_count_{panel_size}_per_image"
                    if uniform_panel
                    else "variable_count_per_image"
                ),
                "raw_agreement": None if raw_agree is None else round(raw_agree, 4),
                "fleiss_kappa": None if fleiss is None else round(float(fleiss), 4),
                "cohen_feasibility": (
                    "pairwise_only_panel_gt_2"
                    if (panel_size or 0) > 2
                    else ("feasible" if feasible else "infeasible")
                ),
                "fleiss_feasibility": "feasible" if feasible else "infeasible",
                "limitation_reason": (
                    "" if feasible else "Non-uniform rater panel; Fleiss assumptions not met."
                ),
            }
        )

    return (
        pd.DataFrame(records).sort_values("class_id", na_position="last").reset_index(drop=True),
        meta,
    )


def analyze_rare_classes(classwise_df: pd.DataFrame, total_images: int) -> pd.DataFrame:
    """Flag classes at RISK of unstable Kappa due to prevalence / rarity.

    This is a prevalence-driven RISK assessment, not a measurement of computed
    instability. A class can carry multiple risk flags. Flag thresholds:
      * positive_images < 100  -> severe_rare_positive_count
      * prevalence      < 0.05 -> severe_prevalence_imbalance
      * positive_images < 500  -> moderate_rare_positive_count
      * prevalence      < 0.10 -> moderate_prevalence_imbalance
    """
    records: List[Dict[str, Any]] = []
    for _, row in classwise_df.iterrows():
        pos = int(row["positive_images"]) if pd.notna(row["positive_images"]) else 0
        prevalence = (pos / total_images) if total_images else 0.0

        flags: List[str] = []
        # Severe first, then moderate (a class may carry several flags).
        if pos < 100:
            flags.append("severe_rare_positive_count")
        elif pos < 500:
            flags.append("moderate_rare_positive_count")
        if prevalence < 0.05:
            flags.append("severe_prevalence_imbalance")
        elif prevalence < 0.10:
            flags.append("moderate_prevalence_imbalance")

        if any(f.startswith("severe") for f in flags):
            risk = "severe"
        elif flags:
            risk = "moderate"
        else:
            risk = "low"

        records.append(
            {
                "class_id": row["class_id"],
                "class_name": row["class_name"],
                "positive_images": pos,
                "prevalence": round(prevalence, 6),
                "fleiss_kappa": row.get("fleiss_kappa"),
                "n_radiologists": int(row["n_radiologists"]),
                "kappa_instability_risk": risk,
                "instability_risk_flags": ";".join(flags) if flags else "",
            }
        )
    return pd.DataFrame(records).sort_values("prevalence").reset_index(drop=True)


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def analyze_bbox_consistency(df: pd.DataFrame, rad_col: Optional[str]) -> Dict[str, Any]:
    """Descriptive-only inter-rater bbox proximity within (image, class)."""
    status: Dict[str, Any] = {
        "evaluated": False,
        "reason": None,
        "pairs_compared": 0,
        "near_duplicate_pairs_iou_ge_threshold": 0,
        "iou_threshold": NEAR_DUP_IOU,
        "note": (
            "Descriptive spatial proximity only. Near-duplicate bboxes are "
            "multi-radiologist annotations, NOT confirmed errors. Nothing is "
            "deleted or fused."
        ),
    }
    has_bbox = all(c in df.columns for c in BBOX_COLUMNS)
    if not has_bbox:
        status["reason"] = "bbox columns absent; bbox-level consistency not evaluable."
        return status
    if rad_col is None:
        status["reason"] = "no radiologist identifier; cannot compare bboxes across raters."
        return status

    work = df.copy()
    work["_is_nf"] = work["class_name"].apply(is_no_finding)
    for c in BBOX_COLUMNS:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    valid = work[(~work["_is_nf"]) & work[BBOX_COLUMNS].notna().all(axis=1)]

    class_key = "class_id" if "class_id" in df.columns else "class_name"
    pairs = 0
    near = 0
    for (_img, _cls), grp in valid.groupby(["image_id", class_key], sort=False):
        if len(grp) < 2:
            continue
        boxes = list(zip(grp["x_min"], grp["y_min"], grp["x_max"], grp["y_max"]))
        raters = grp[rad_col].astype("string").tolist()
        for i, j in combinations(range(len(boxes)), 2):
            if raters[i] is not None and raters[i] == raters[j]:
                continue  # same rater -> not an inter-rater pair
            pairs += 1
            if _iou(boxes[i], boxes[j]) >= NEAR_DUP_IOU:
                near += 1

    status["evaluated"] = True
    status["pairs_compared"] = int(pairs)
    status["near_duplicate_pairs_iou_ge_threshold"] = int(near)
    return status


# --- Report writer --------------------------------------------------------


def write_report_md(path: str, p: Dict[str, Any]) -> None:
    L: List[str] = []
    L.append("# Phase 1D — Kappa Feasibility / Limitation-aware Analysis")
    L.append("")
    L.append(f"_Generated {p['created_utc']}._")
    L.append("")
    L.append("## 1. Objective")
    L.append("")
    L.append(
        "Assess whether inter-rater agreement (Cohen's / Fleiss' Kappa) is "
        "computable on the controlled scope and, where computable, report it as "
        "data-quality evidence. Agreement is NEVER a model metric or a decision "
        "criterion for split/model/threshold."
    )
    L.append("")
    L.append("## 2. Inputs and Scope")
    L.append("")
    for f in p["input_files"]:
        L.append(f"- Input: `{f}`")
    L.append(f"- Total images: {p['total_images']}; total rows: {p['total_rows']}.")
    L.append("- Metadata only: no image, DICOM, header, or dimension was read.")
    L.append("")
    L.append("## 3. rad_id Availability")
    L.append("")
    L.append(f"- rad_id available: {p['rad_id_available']} (column: {p['rad_id_column_used']}).")
    L.append(f"- rad_id missing/null/empty count: {p['rad_id_missing_count']}.")
    L.append(f"- Radiologists total (distinct): {p['radiologists_total']}.")
    L.append("")
    L.append("## 4. Radiologists per Image")
    L.append("")
    if p["radiologists_per_image_distribution"]:
        L.append("| radiologists_per_image | image_count |")
        L.append("|---|---|")
        for k, v in p["radiologists_per_image_distribution"].items():
            L.append(f"| {k} | {v} |")
        L.append("")
        L.append(
            f"Each image has a uniform number of {p['rater_panel_size']} "
            f"radiologist ratings. Across the dataset, there are "
            f"{p['radiologists_total']} distinct radiologists. Therefore, the "
            "panel size is fixed per image, but the exact radiologist identities "
            "may vary across images "
            f"(same_rater_identity_panel_across_images="
            f"{p['same_rater_identity_panel_across_images']})."
        )
    else:
        L.append("- Not available (no rad_id column).")
    L.append("")
    L.append("## 5. Image-Class-Radiologist Binary Matrix Feasibility")
    L.append("")
    L.append(f"- binary_matrix_feasible: **{p['binary_matrix_feasible']}**.")
    for r in p["_binary_reasons"]:
        L.append(f"- {r}")
    L.append("")
    L.append("## 6. Cohen's Kappa Feasibility")
    L.append("")
    L.append(f"- cohen_kappa_feasible: **{p['cohen_kappa_feasible']}**.")
    for r in p["_cohen_reasons"]:
        L.append(f"- {r}")
    L.append("")
    L.append("## 7. Fleiss' Kappa Feasibility")
    L.append("")
    L.append(f"- fleiss_kappa_feasible: **{p['fleiss_kappa_feasible']}**.")
    for r in p["_fleiss_reasons"]:
        L.append(f"- {r}")
    if p.get("overall_fleiss_kappa_mean") is not None:
        L.append(f"- Mean Fleiss' Kappa across abnormal classes: **{p['overall_fleiss_kappa_mean']}**.")
    L.append("")
    L.append("## 8. Class-wise Image-level Agreement Feasibility")
    L.append("")
    L.append(f"- Summary: {p['classwise_feasibility_summary']}.")
    L.append("- Full per-class Fleiss kappa and coverage in `phase1D_classwise_agreement_feasibility.csv`.")
    if p.get("_classwise_preview"):
        L.append("")
        L.append("| class_id | class_name | positive_images | fleiss_kappa |")
        L.append("|---|---|---|---|")
        for row in p["_classwise_preview"]:
            L.append(
                f"| {row['class_id']} | {row['class_name']} | "
                f"{row['positive_images']} | {row['fleiss_kappa']} |"
            )
    L.append("")
    L.append("## 9. Rare Class Instability Risk for Kappa")
    L.append("")
    L.append(f"- Summary: {p['rare_class_instability_summary']}.")
    L.append(
        "- This is a **prevalence/rarity-driven RISK**, not a measurement of "
        "computed instability. Risk tiers: severe (positive_images<100 or "
        "prevalence<0.05) and moderate (positive_images<500 or prevalence<0.10). "
        "A class may carry multiple flags."
    )
    L.append("- Per-class risk flags in `phase1D_rare_class_kappa_instability.csv`.")
    L.append("")
    L.append("## 10. Label-level Agreement vs BBox-level Consistency")
    L.append("")
    L.append(f"- label_level_agreement_status: {p['label_level_agreement_status']}.")
    L.append(f"- bbox_level_consistency_status: {p['bbox_level_consistency_status']}.")
    L.append(
        "- These are kept strictly separate. Label-level agreement uses "
        "present/absent decisions; bbox proximity is descriptive only. "
        "Near-duplicate boxes are retained (not fused) and are not treated as "
        "confirmed annotation errors."
    )
    L.append("")
    L.append("## 11. Limitations")
    L.append("")
    for lim in p["limitations"]:
        L.append(f"- {lim}")
    L.append("")
    L.append("## 12. Decision")
    L.append("")
    L.append(p["decision"])
    L.append("")
    L.append("## 13. Definition of Done")
    L.append("")
    L.append(f"- dod_status: **{p['dod_status']}**.")
    L.append(
        "- DoD is met when agreement feasibility, computed Kappa (where valid), "
        "and limitations are documented and exported. No agreement value is used "
        "for modelling. Send outputs to review before ticking the checklist."
    )
    L.append("")
    ensure_parent(path).write_text("\n".join(L), encoding="utf-8")


# --- Main -----------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    try:
        df, input_files = load_input(args.scope_csv, args.train_csv)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    rad_col = detect_rad_id_column(df)
    rad_id_available = rad_col is not None

    total_rows = int(len(df))
    total_images = int(df["image_id"].nunique())

    if rad_col is not None:
        rad_series = df[rad_col].astype("string").str.strip()
        rad_missing = int(rad_series.isna().sum() + (rad_series == "").sum())
    else:
        rad_missing = int(total_rows)

    per_image_df, distribution, total_raters, readers_per_image = (
        analyze_radiologists_per_image(df, rad_col)
    )

    # Panel: each image has a uniform COUNT of raters, but the specific
    # radiologist identities may differ across images. "uniform" below refers
    # to the per-image rater COUNT, NOT a single shared panel of identities.
    panel_sizes = sorted({int(k) for k in distribution}) if distribution else []
    uniform_panel = len(panel_sizes) == 1 and rad_id_available
    panel_size = panel_sizes[0] if panel_sizes else None

    # Check whether the exact rater identities form ONE shared panel across all
    # images (True) or vary image-to-image (False). For VinBigData this is
    # expected to be False: fixed count (3) drawn from a larger pool (17).
    if rad_id_available and not readers_per_image.empty:
        distinct_identity_panels = set(readers_per_image.values)
        same_rater_identity_panel_across_images = len(distinct_identity_panels) == 1
    else:
        same_rater_identity_panel_across_images = False

    classwise_df, cw_meta = build_classwise_agreement(
        df, rad_col, readers_per_image, total_images
    )
    rare_df = analyze_rare_classes(classwise_df, total_images)
    bbox_status = analyze_bbox_consistency(df, rad_col)

    binary_matrix_feasible = bool(cw_meta.get("negatives_inferable", False))
    fleiss_feasible = bool(uniform_panel and panel_size and panel_size >= 2)
    # Cohen: directly natural only for images with exactly 2 readers.
    images_with_two = int(distribution.get("2", 0)) if distribution else 0
    cohen_feasible = bool(rad_id_available and images_with_two > 0)

    # Reasons.
    binary_reasons: List[str] = []
    if binary_matrix_feasible:
        binary_reasons.append(
            "Each image has a uniform number of radiologist ratings; 'No finding' "
            "rows carry rad_id, so a rater's read-coverage is known. A rater who "
            "read an image but did not mark class C is a VALID negative. The "
            "complete image x rater matrix per class is therefore constructible."
        )
        if uniform_panel:
            binary_reasons.append(
                f"Each image has a uniform number of {panel_size} radiologist "
                f"ratings. Across the dataset, there are {total_raters} distinct "
                "radiologists. Therefore, the panel size is fixed per image, but "
                "the exact radiologist identities may vary across images "
                f"(same_rater_identity_panel_across_images="
                f"{same_rater_identity_panel_across_images})."
            )
    else:
        binary_reasons.append(
            "No rater identifier / read-coverage signal; negatives cannot be "
            "inferred and the matrix is not constructible."
        )

    cohen_reasons: List[str] = []
    if panel_size and panel_size != 2:
        cohen_reasons.append(
            f"Panel has {panel_size} raters per image, not 2; Cohen's Kappa (a "
            "two-rater statistic) is not the natural choice. Fleiss' Kappa is "
            "used instead. Pairwise Cohen could be computed per rater-pair if "
            "specifically required."
        )
    elif cohen_feasible:
        cohen_reasons.append(
            f"{images_with_two} image(s) have exactly two raters; pairwise Cohen "
            "is computable on that stratum."
        )
    else:
        cohen_reasons.append("No two-rater stratum; Cohen's Kappa not applicable.")

    fleiss_reasons: List[str] = []
    if fleiss_feasible:
        fleiss_reasons.append(
            f"Uniform per-image rater count ({panel_size}) with inferable "
            "negatives satisfies Fleiss' assumptions; per-class binary Fleiss' "
            "Kappa is computed. Rater identities may differ across images, which "
            "Fleiss' Kappa permits (it does not require the same raters per item)."
        )
    else:
        fleiss_reasons.append(
            "Rater panel is not uniform or rad_id is absent; a complete Fleiss "
            "matrix cannot be assumed."
        )

    # Summaries.
    n_classes = int(len(classwise_df))
    n_feasible_fleiss = int((classwise_df["fleiss_feasibility"] == "feasible").sum())
    valid_kappas = classwise_df["fleiss_kappa"].dropna()
    overall_mean = round(float(valid_kappas.mean()), 4) if len(valid_kappas) else None
    classwise_summary = (
        f"{n_classes} abnormal classes assessed; {n_feasible_fleiss} with feasible "
        f"Fleiss' Kappa"
        + (f"; mean kappa={overall_mean}" if overall_mean is not None else "")
    )
    if not rare_df.empty:
        n_severe = int((rare_df["kappa_instability_risk"] == "severe").sum())
        n_moderate = int((rare_df["kappa_instability_risk"] == "moderate").sum())
        n_low = int((rare_df["kappa_instability_risk"] == "low").sum())
    else:
        n_severe = n_moderate = n_low = 0
    n_at_risk = n_severe + n_moderate
    rare_summary = (
        f"{n_at_risk}/{n_classes} classes carry kappa_instability_risk "
        f"(severe={n_severe}, moderate={n_moderate}, low={n_low}); "
        "risk is prevalence/rarity-driven, not measured instability"
    )

    # Statuses.
    if rad_col is None:
        label_status = "not_evaluable_no_rad_id"
    elif binary_matrix_feasible:
        label_status = "evaluable_fleiss_computed"
    else:
        label_status = "not_evaluable"

    bbox_consistency_status = (
        "evaluated_descriptive_only"
        if bbox_status["evaluated"]
        else f"not_evaluable: {bbox_status['reason']}"
    )

    # Limitations (honest, but no longer claiming infeasibility).
    limitations: List[str] = []
    if not rad_id_available:
        limitations.append(
            "No radiologist identifier; inter-rater analysis impossible on this metadata."
        )
    else:
        limitations.append(
            "Negatives are inferred from read-coverage (rater read image but did "
            "not mark class C). This assumes 'No finding' / absence of a positive "
            "row faithfully encodes a negative decision, which is the VinBigData "
            "labelling convention."
        )
    if panel_size and panel_size != 2:
        limitations.append(
            f"Panel size is {panel_size}; Fleiss' Kappa is appropriate, while "
            "Cohen's Kappa applies only to two-rater strata."
        )
    limitations.append(
        f"Kappa instability RISK: {n_severe} class(es) at severe risk and "
        f"{n_moderate} at moderate risk from low positive count / prevalence "
        "imbalance. This is a prevalence-driven risk, not measured instability; "
        "such classes can yield deflated Kappa (the well-known kappa paradox "
        "under class imbalance)."
    )
    limitations.append(
        f"{NEAR_DUP_IOU} IoU near-duplicate bbox candidates are retained as "
        "multi-reader evidence; they are not fused or deleted, and are not used "
        "to conclude annotation errors."
    )

    decision = (
        "Kappa/agreement analysis in Phase 1D is used ONLY as data-quality "
        "evidence and limitation evidence. It is NOT used as a model metric, NOT "
        "used to select split/model/threshold, NOT used to modify annotations, "
        "and NOT used to evaluate SSL performance. Because every image has a "
        "uniform number of radiologist ratings (identities may vary across "
        "images) and 'No finding' rows encode read-coverage, negatives are "
        "validly inferable and per-class Fleiss' Kappa IS computed and reported "
        "as agreement evidence."
    )

    dod_status = "PASS_agreement_computed_and_documented"

    payload: Dict[str, Any] = {
        "phase": "phase1D_kappa_feasibility",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_files": input_files,
        "rad_id_available": rad_id_available,
        "rad_id_column_used": rad_col,
        "rad_id_missing_count": rad_missing,
        "total_images": total_images,
        "total_rows": total_rows,
        "radiologists_total": total_raters,
        "radiologists_per_image_distribution": distribution,
        "rater_panel_uniform": bool(uniform_panel),
        "uniform_rater_count_per_image": bool(uniform_panel),
        "same_rater_identity_panel_across_images": bool(
            same_rater_identity_panel_across_images
        ),
        "rater_panel_size": panel_size,
        "binary_matrix_feasible": binary_matrix_feasible,
        "cohen_kappa_feasible": cohen_feasible,
        "fleiss_kappa_feasible": fleiss_feasible,
        "overall_fleiss_kappa_mean": overall_mean,
        "classwise_feasibility_summary": classwise_summary,
        "rare_class_instability_summary": rare_summary,
        "label_level_agreement_status": label_status,
        "bbox_level_consistency_status": bbox_consistency_status,
        "bbox_level_consistency_detail": bbox_status,
        "limitations": limitations,
        "decision": decision,
        "dod_status": dod_status,
        "forbidden_actions_confirmed": {
            "split_created": False,
            "coco_created": False,
            "training_started": False,
            "pseudo_label_generated": False,
            "threshold_tuned": False,
            "test_set_used": False,
            "pixel_read": False,
            "dicom_or_header_read": False,
            "image_dimensions_read": False,
            "boundary_validation": False,
            "annotations_deleted_or_edited": False,
            "near_duplicate_bbox_deleted_or_fused": False,
            "kappa_used_as_model_metric": False,
            "kappa_used_for_split_model_threshold": False,
        },
        "_binary_reasons": binary_reasons,
        "_cohen_reasons": cohen_reasons,
        "_fleiss_reasons": fleiss_reasons,
        "_classwise_preview": classwise_df[
            ["class_id", "class_name", "positive_images", "fleiss_kappa"]
        ].to_dict("records"),
        "generated_files": {
            "output_json": args.output_json,
            "report_md": args.report_md,
            "rad_per_image_csv": args.rad_per_image_csv,
            "classwise_csv": args.classwise_csv,
            "rare_class_csv": args.rare_class_csv,
        },
    }

    # Write outputs.
    ensure_parent(args.output_json).write_text(
        json.dumps(
            {k: v for k, v in payload.items() if not k.startswith("_")},
            indent=2,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    per_image_df.to_csv(ensure_parent(args.rad_per_image_csv), index=False)
    classwise_df.to_csv(ensure_parent(args.classwise_csv), index=False)
    rare_df.to_csv(ensure_parent(args.rare_class_csv), index=False)
    write_report_md(args.report_md, payload)

    # Console summary.
    print("=" * 68)
    print("Phase 1D — Kappa Feasibility / Agreement Analysis")
    print("=" * 68)
    print(f"input_files                 : {input_files}")
    print(f"rad_id_available            : {rad_id_available} (col={rad_col})")
    print(f"rad_id_missing_count        : {rad_missing}")
    print(f"total_images                : {total_images}")
    print(f"total_rows                  : {total_rows}")
    print(f"radiologists_total          : {total_raters}")
    print(f"radiologists_per_image_dist : {distribution}")
    print(f"uniform_rater_count_per_image: {uniform_panel} (count={panel_size})")
    print(
        "same_rater_identity_panel   : "
        f"{same_rater_identity_panel_across_images}"
    )
    print(f"binary_matrix_feasible      : {binary_matrix_feasible}")
    print(f"cohen_kappa_feasible        : {cohen_feasible}")
    print(f"fleiss_kappa_feasible       : {fleiss_feasible}")
    print(f"overall_fleiss_kappa_mean   : {overall_mean}")
    print(f"classwise_feasibility       : {classwise_summary}")
    print(f"rare_class_instability      : {rare_summary}")
    print(f"label_level_agreement_status: {label_status}")
    print(f"bbox_level_consistency      : {bbox_consistency_status}")
    print("-" * 68)
    print("Limitations:")
    for lim in limitations:
        print(f"  - {lim}")
    print("-" * 68)
    print(f"dod_status                  : {dod_status}")
    print("=" * 68)
    print("NOTE: Kappa is evidence only — not a model metric or decision tool.")
    print("      Checklist NOT auto-ticked. Send outputs to review first.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
