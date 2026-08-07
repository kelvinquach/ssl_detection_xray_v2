"""Phase 2E — Fixed Train/Validation/Test Split.

Phase 2E-C0-R3: materialize and independently validate the fixed split.

R3 deliberately does not contain a second split algorithm.  It loads the
already-PASSed C0-R2 implementation, recreates that exact in-memory candidate,
locks it against the R2 evidence (sizes, repair-log lengths, and image-ID
SHA-256 values), materializes all official artifacts in a staging tree, then
independently reads those bytes back before transactionally promoting them.

Default project inputs:
  scripts/02E_C0_R2_exact_constrained_candidate_split.py
  data/processed/coco/coco_master_jpg.json

Official outputs:
  data/processed/coco/instances_{train,val,test}.json
  data/manifests/fixed_split_manifest.csv
  data/manifests/split_lock_manifest.json
  data/manifests/leakage_check_report.json
  reports/split_negative_distribution.csv
  reports/phase2E_build_fixed_split_validation_report.json
  reports/phase2E_build_fixed_split_log.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any


SPLIT_ORDER = ("train", "val", "test")
EXPECTED_TOTAL_IMAGES = 4894
EXPECTED_TOTAL_ANNOTATIONS = 36096
EXPECTED_TOTAL_CATEGORIES = 14
EXPECTED_SIZES = {"train": 3426, "val": 734, "test": 734}
EXPECTED_NO_FINDING = {"train": 350, "val": 75, "test": 75}
EXPECTED_R2_SIZE_REPAIR_MOVES = 23
EXPECTED_R2_NO_FINDING_SWAPS = 3
EXPECTED_R2_IMAGE_ID_SHA256 = {
    "train": "628b9bb8ba25129a928abe994b101b4c4efd5588d389feb60da6de2a371fa11a",
    "val": "87c23ebed4d1e6965731fc0b31245859f49e777119813c6152efde3531ba58c6",
    "test": "1f7903e069e872bf2e5fe13bb4d0fa257dc4a1c2c8290a621d3f7286ada66b37",
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--r2-script",
        type=Path,
        default=Path("scripts/02E_C0_R2_exact_constrained_candidate_split.py"),
        help="Exact R2 implementation that produced the locked candidate.",
    )
    parser.add_argument(
        "--coco",
        type=Path,
        default=Path("data/processed/coco/coco_master_jpg.json"),
        help="Locked master COCO JSON.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root beneath which the nine official artifacts are written.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id_key(value: object) -> str:
    return str(value)


def sha256_image_ids(values: list[Any]) -> str:
    canonical = "\n".join(str(v) for v in sorted(values, key=stable_id_key))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        fail(f"Expected JSON object in {path}")
    return value


def load_r2(path: Path) -> ModuleType:
    if not path.is_file():
        fail(f"R2 script not found: {path}")
    spec = importlib.util.spec_from_file_location("phase2e_c0_r2_locked", path)
    if spec is None or spec.loader is None:
        fail(f"Cannot load R2 script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = ("load_coco", "make_candidate", "sha256_image_ids")
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        fail(f"R2 script is missing required callables: {missing}")
    return module


def validate_locked_master(coco: dict[str, Any]) -> None:
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])
    if (len(images), len(annotations), len(categories)) != (
        EXPECTED_TOTAL_IMAGES,
        EXPECTED_TOTAL_ANNOTATIONS,
        EXPECTED_TOTAL_CATEGORIES,
    ):
        fail(
            "Locked COCO counts changed: "
            f"images={len(images)}, annotations={len(annotations)}, "
            f"categories={len(categories)}"
        )
    image_ids = [item["id"] for item in images]
    ann_ids = [item["id"] for item in annotations]
    category_ids = [item["id"] for item in categories]
    if len(set(image_ids)) != len(image_ids):
        fail("Duplicate image IDs in master COCO")
    if len(set(ann_ids)) != len(ann_ids):
        fail("Duplicate annotation IDs in master COCO")
    if len(set(category_ids)) != len(category_ids):
        fail("Duplicate category IDs in master COCO")
    image_set, category_set = set(image_ids), set(category_ids)
    if any(a.get("image_id") not in image_set for a in annotations):
        fail("Master COCO has annotation -> image reference errors")
    if any(a.get("category_id") not in category_set for a in annotations):
        fail("Master COCO has annotation -> category reference errors")


def recreate_and_lock_r2_candidate(
    r2: ModuleType, coco_path: Path
) -> tuple[dict[str, list[Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    # R2's loader reads its module-level COCO_PATH. Override only the path;
    # every split/repair constant and algorithm remains the exact R2 code.
    r2.COCO_PATH = coco_path
    image_ids, labels, no_finding, names, _ = r2.load_coco()
    initial, exact_size, candidate, size_log, nf_log = r2.make_candidate(
        image_ids, labels, no_finding, names
    )

    if {name: len(candidate[name]) for name in SPLIT_ORDER} != EXPECTED_SIZES:
        fail("R2 candidate sizes do not match the locked evidence")
    if len(size_log) != EXPECTED_R2_SIZE_REPAIR_MOVES:
        fail("R2 SIZE_REPAIR_MOVE count does not match the locked evidence")
    if len(nf_log) != EXPECTED_R2_NO_FINDING_SWAPS:
        fail("R2 NO_FINDING_SWAP count does not match the locked evidence")

    # Also require R2 to be internally reproducible in this exact environment.
    _, _, repeated, repeated_size_log, repeated_nf_log = r2.make_candidate(
        image_ids, labels, no_finding, names
    )
    for name in SPLIT_ORDER:
        if not (candidate[name] == repeated[name]).all():
            fail(f"R2 candidate is not reproducible for split={name}")
    if size_log != repeated_size_log or nf_log != repeated_nf_log:
        fail("R2 repair logs are not reproducible")

    ids_by_split: dict[str, list[Any]] = {}
    for name in SPLIT_ORDER:
        ids = image_ids[candidate[name]].tolist()
        observed = sha256_image_ids(ids)
        if observed != EXPECTED_R2_IMAGE_ID_SHA256[name]:
            fail(
                f"R2 image-ID SHA-256 mismatch for {name}: "
                f"observed={observed}, expected={EXPECTED_R2_IMAGE_ID_SHA256[name]}"
            )
        ids_by_split[name] = ids

    all_sets = {name: set(ids_by_split[name]) for name in SPLIT_ORDER}
    if any(
        all_sets[left] & all_sets[right]
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        fail("R2 candidate has overlap")
    if len(set().union(*all_sets.values())) != EXPECTED_TOTAL_IMAGES:
        fail("R2 candidate does not cover the full master scope")
    return ids_by_split, size_log, nf_log


def subset_coco(master: dict[str, Any], split_ids: set[Any]) -> dict[str, Any]:
    # Preserve master order and IDs. Do not renumber images, annotations, or categories.
    result = {key: value for key, value in master.items() if key not in {"images", "annotations", "categories"}}
    result["images"] = [item for item in master["images"] if item["id"] in split_ids]
    result["annotations"] = [item for item in master["annotations"] if item["image_id"] in split_ids]
    result["categories"] = master["categories"]
    return result


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def write_manifest(
    path: Path, master: dict[str, Any], ids_by_split: dict[str, list[Any]]
) -> None:
    split_of = {
        image_id: split
        for split in SPLIT_ORDER
        for image_id in ids_by_split[split]
    }
    ann_count = Counter(a["image_id"] for a in master["annotations"])
    classes: dict[Any, set[Any]] = {}
    for annotation in master["annotations"]:
        classes.setdefault(annotation["image_id"], set()).add(annotation["category_id"])
    fields = (
        "image_id",
        "split",
        "file_name",
        "zero_gt",
        "annotation_count",
        "class_count",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for image in master["images"]:
            image_id = image["id"]
            writer.writerow(
                {
                    "image_id": image_id,
                    "split": split_of[image_id],
                    "file_name": image["file_name"],
                    "zero_gt": int(ann_count[image_id] == 0),
                    "annotation_count": ann_count[image_id],
                    "class_count": len(classes.get(image_id, set())),
                }
            )


def write_negative_distribution(
    path: Path, master: dict[str, Any], ids_by_split: dict[str, list[Any]]
) -> None:
    annotated_ids = {a["image_id"] for a in master["annotations"]}
    fields = (
        "split",
        "total_images",
        "negative_images",
        "abnormal_images",
        "negative_fraction",
        "negative_percent",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for split in SPLIT_ORDER:
            total = len(ids_by_split[split])
            negative = sum(image_id not in annotated_ids for image_id in ids_by_split[split])
            writer.writerow(
                {
                    "split": split,
                    "total_images": total,
                    "negative_images": negative,
                    "abnormal_images": total - negative,
                    "negative_fraction": f"{negative / total:.12f}",
                    "negative_percent": f"{100.0 * negative / total:.9f}",
                }
            )


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def independently_validate_staging(
    staging: Path,
    master: dict[str, Any],
    ids_by_split: dict[str, list[Any]],
) -> dict[str, Any]:
    master_image_ids = {item["id"] for item in master["images"]}
    master_ann_ids = {item["id"] for item in master["annotations"]}
    master_category_ids = [item["id"] for item in master["categories"]]
    observed_sets: dict[str, set[Any]] = {}
    observed_ann_sets: dict[str, set[Any]] = {}
    split_summary: dict[str, Any] = {}

    for name in SPLIT_ORDER:
        coco_file = staging / "data" / "processed" / "coco" / f"instances_{name}.json"
        data = load_json(coco_file)
        images = data.get("images", [])
        annotations = data.get("annotations", [])
        categories = data.get("categories", [])
        image_ids = [item["id"] for item in images]
        image_set = set(image_ids)
        ann_ids = [item["id"] for item in annotations]
        ann_set = set(ann_ids)
        if len(images) != EXPECTED_SIZES[name] or len(image_set) != len(images):
            fail(f"Independent validation: invalid image count/uniqueness in {name}")
        if image_set != set(ids_by_split[name]):
            fail(f"Independent validation: {name}.json membership differs from locked R2")
        if sha256_image_ids(image_ids) != EXPECTED_R2_IMAGE_ID_SHA256[name]:
            fail(f"Independent validation: image-ID hash differs in {name}")
        if len(ann_ids) != len(ann_set):
            fail(f"Independent validation: duplicate annotation IDs in {name}")
        if any(a.get("image_id") not in image_set for a in annotations):
            fail(f"Independent validation: annotation ownership error in {name}")
        if [item["id"] for item in categories] != master_category_ids:
            fail(f"Independent validation: category schema changed in {name}")
        counts_by_image = Counter(a["image_id"] for a in annotations)
        zero_gt = sum(counts_by_image[image_id] == 0 for image_id in image_ids)
        if zero_gt != EXPECTED_NO_FINDING[name]:
            fail(
                f"Independent validation: zero-GT mismatch in {name}: "
                f"{zero_gt} != {EXPECTED_NO_FINDING[name]}"
            )
        present_categories = {a["category_id"] for a in annotations}
        if present_categories != set(master_category_ids):
            fail(f"Independent validation: not all 14 categories are present in {name}")
        observed_sets[name] = image_set
        observed_ann_sets[name] = ann_set
        split_summary[name] = {
            "images": len(images),
            "annotations": len(annotations),
            "zero_gt_images": zero_gt,
            "categories_present": len(present_categories),
            "sha256_image_ids": sha256_image_ids(image_ids),
            "sha256_coco_json": sha256_file(coco_file),
        }

    if observed_sets["train"] & observed_sets["val"]:
        fail("Independent validation: train/val image overlap")
    if observed_sets["train"] & observed_sets["test"]:
        fail("Independent validation: train/test image overlap")
    if observed_sets["val"] & observed_sets["test"]:
        fail("Independent validation: val/test image overlap")
    if set().union(*observed_sets.values()) != master_image_ids:
        fail("Independent validation: split image union differs from master")
    if any(
        observed_ann_sets[left] & observed_ann_sets[right]
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        fail("Independent validation: annotation IDs overlap between splits")
    if set().union(*observed_ann_sets.values()) != master_ann_ids:
        fail("Independent validation: annotation union differs from master")
    if sum(v["annotations"] for v in split_summary.values()) != EXPECTED_TOTAL_ANNOTATIONS:
        fail("Independent validation: annotation total changed")

    manifest = read_manifest(staging / "data" / "manifests" / "fixed_split_manifest.csv")
    if len(manifest) != EXPECTED_TOTAL_IMAGES:
        fail("Independent validation: manifest row count changed")
    manifest_ids = [row["image_id"] for row in manifest]
    if len(set(manifest_ids)) != EXPECTED_TOTAL_IMAGES:
        fail("Independent validation: manifest image IDs are not unique")
    # CSV turns IDs into strings, so compare canonically.
    if set(manifest_ids) != {str(v) for v in master_image_ids}:
        fail("Independent validation: manifest image-ID union differs from master")
    manifest_split_counts = Counter(row["split"] for row in manifest)
    if dict(manifest_split_counts) != EXPECTED_SIZES:
        fail(f"Independent validation: manifest split sizes changed: {manifest_split_counts}")
    manifest_zero_gt = Counter(
        row["split"] for row in manifest if row["zero_gt"] == "1"
    )
    if dict(manifest_zero_gt) != EXPECTED_NO_FINDING:
        fail(f"Independent validation: manifest zero-GT allocation changed: {manifest_zero_gt}")

    negative_rows = read_manifest(staging / "reports" / "split_negative_distribution.csv")
    if [row.get("split") for row in negative_rows] != list(SPLIT_ORDER):
        fail("Independent validation: negative-distribution split order/schema changed")
    for row in negative_rows:
        split = row["split"]
        if int(row["total_images"]) != EXPECTED_SIZES[split]:
            fail(f"Independent validation: negative-distribution size mismatch in {split}")
        if int(row["negative_images"]) != EXPECTED_NO_FINDING[split]:
            fail(f"Independent validation: negative-distribution count mismatch in {split}")

    return {
        "gate": "PASS",
        "fixed_split_created": True,
        "fixed_split_validated": True,
        "split_summary": split_summary,
        "image_overlap": {"train_val": 0, "train_test": 0, "val_test": 0},
        "image_union": EXPECTED_TOTAL_IMAGES,
        "annotation_union": EXPECTED_TOTAL_ANNOTATIONS,
        "manifest_rows": EXPECTED_TOTAL_IMAGES,
    }


def official_relative_paths() -> list[Path]:
    return [
        Path("data/processed/coco/instances_train.json"),
        Path("data/processed/coco/instances_val.json"),
        Path("data/processed/coco/instances_test.json"),
        Path("data/manifests/fixed_split_manifest.csv"),
        Path("data/manifests/split_lock_manifest.json"),
        Path("data/manifests/leakage_check_report.json"),
        Path("reports/split_negative_distribution.csv"),
        Path("reports/phase2E_C0_R3_validation_report.json"),
        Path("reports/phase2E_C0_R3_materialization_log.json"),
    ]


def promote_with_rollback(staging: Path, project_root: Path, paths: list[Path]) -> None:
    existing = [project_root / relative for relative in paths if (project_root / relative).exists()]
    if existing:
        fail("Refusing to overwrite official artifact(s): " + ", ".join(map(str, existing)))
    promoted: list[Path] = []
    try:
        for relative in paths:
            source = staging / relative
            target = project_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
            promoted.append(target)
    except Exception:
        for target in reversed(promoted):
            if target.exists():
                target.unlink()
        raise


def main() -> None:
    args = parse_args()
    r2_path = args.r2_script.resolve()
    coco_path = args.coco.resolve()
    project_root = args.project_root.resolve()
    output_paths = official_relative_paths()

    print("=== PHASE 2E-C0-R3: MATERIALIZE & INDEPENDENTLY VALIDATE FIXED SPLIT ===")
    print("MODE= STAGE_VALIDATE_THEN_PROMOTE")
    print("R2_SCRIPT=", r2_path)
    print("COCO=", coco_path)
    print("PROJECT_ROOT=", project_root)
    print("PYTHON=", platform.python_version())
    print("SEED_POLICY= INHERIT_R2_PRE_SPECIFIED_LOCKED_NO_SEED_SEARCH")

    if not coco_path.is_file():
        fail(f"Master COCO not found: {coco_path}")
    if not project_root.is_dir():
        fail(f"Project root not found: {project_root}")
    existing = [project_root / relative for relative in output_paths if (project_root / relative).exists()]
    if existing:
        fail("Refusing to overwrite official artifact(s): " + ", ".join(map(str, existing)))

    master = load_json(coco_path)
    validate_locked_master(master)
    r2 = load_r2(r2_path)
    ids_by_split, size_log, nf_log = recreate_and_lock_r2_candidate(r2, coco_path)
    print("R2_CANDIDATE_LOCK= PASS")
    for name in SPLIT_ORDER:
        print(f"R2_{name.upper()}_SHA256_IMAGE_IDS=", EXPECTED_R2_IMAGE_ID_SHA256[name])

    staging = Path(tempfile.mkdtemp(prefix=".phase2E_R3.staging-", dir=str(project_root)))
    promoted = False
    try:
        split_sets = {name: set(ids_by_split[name]) for name in SPLIT_ORDER}
        for name in SPLIT_ORDER:
            path = staging / "data" / "processed" / "coco" / f"instances_{name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, subset_coco(master, split_sets[name]))
        manifest_path = staging / "data" / "manifests" / "fixed_split_manifest.csv"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        write_manifest(manifest_path, master, ids_by_split)
        reports_dir = staging / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        write_negative_distribution(
            reports_dir / "split_negative_distribution.csv", master, ids_by_split
        )

        validation = independently_validate_staging(staging, master, ids_by_split)
        coco_sha256 = {
            name: sha256_file(
                staging / "data" / "processed" / "coco" / f"instances_{name}.json"
            )
            for name in SPLIT_ORDER
        }
        lock_manifest = {
            "phase": "Phase 2E — Fixed Train/Validation/Test Split",
            "stage": "2E-C0-R3",
            "status": "LOCKED",
            "seed": 42,
            "seed_policy": "PRE_SPECIFIED_LOCKED_NO_SEED_SEARCH",
            "split_ratio": {"train": 0.70, "val": 0.15, "test": 0.15},
            "split_sizes": EXPECTED_SIZES,
            "no_finding_allocation": EXPECTED_NO_FINDING,
            "image_id_sha256": EXPECTED_R2_IMAGE_ID_SHA256,
            "coco_json_sha256": coco_sha256,
            "master_coco_sha256": sha256_file(coco_path),
            "r2_script_sha256": sha256_file(r2_path),
        }
        write_json(staging / "data" / "manifests" / "split_lock_manifest.json", lock_manifest)

        leakage_report = {
            "phase": "2E-C0-R3",
            "status": "PASS",
            "unit_of_split": "image_id",
            "image_overlap": validation["image_overlap"],
            "image_union": validation["image_union"],
            "expected_image_union": EXPECTED_TOTAL_IMAGES,
            "annotation_overlap": {"train_val": 0, "train_test": 0, "val_test": 0},
            "annotation_union": validation["annotation_union"],
            "expected_annotation_union": EXPECTED_TOTAL_ANNOTATIONS,
            "completeness_pass": True,
            "zero_overlap_pass": True,
        }
        write_json(staging / "data" / "manifests" / "leakage_check_report.json", leakage_report)

        report = {
            "phase": "Phase 2E — Fixed Train/Validation/Test Split",
            "stage": "2E-C0-R3",
            "phase": "2E-C0-R3",
            "status": "PASS",
            "mode": "stage_validate_then_promote",
            "inputs": {
                "master_coco": str(coco_path),
                "master_coco_sha256": sha256_file(coco_path),
                "r2_script": str(r2_path),
                "r2_script_sha256": sha256_file(r2_path),
            },
            "locked_r2_evidence": {
                "seed": 42,
                "seed_policy": "PRE_SPECIFIED_LOCKED_NO_SEED_SEARCH",
                "size_repair_moves": len(size_log),
                "no_finding_swaps": len(nf_log),
                "image_id_sha256": EXPECTED_R2_IMAGE_ID_SHA256,
            },
            "expected": {
                "split_sizes": EXPECTED_SIZES,
                "no_finding": EXPECTED_NO_FINDING,
                "total_images": EXPECTED_TOTAL_IMAGES,
                "total_annotations": EXPECTED_TOTAL_ANNOTATIONS,
                "total_categories": EXPECTED_TOTAL_CATEGORIES,
            },
            "independent_readback_validation": validation,
        }
        validation_report_path = reports_dir / "phase2E_C0_R3_validation_report.json"
        write_json(validation_report_path, report)

        materialization_log = {
            "phase": "2E-C0-R3",
            "status": "PASS",
            "mode": "stage_validate_then_transactional_promote",
            "files_written": len(output_paths),
            "official_artifacts": [str(path).replace("\\", "/") for path in output_paths],
            "r2_size_repair_moves": size_log,
            "r2_no_finding_swaps": nf_log,
            "overwrite_policy": "REFUSE_IF_ANY_OFFICIAL_ARTIFACT_EXISTS",
            "rollback_policy": "REMOVE_FILES_PROMOTED_BY_FAILED_RUN",
        }
        write_json(reports_dir / "phase2E_C0_R3_materialization_log.json", materialization_log)

        report_readback = load_json(validation_report_path)
        if report_readback.get("status") != "PASS":
            fail("Validation report readback did not preserve PASS status")
        for relative in output_paths:
            if not (staging / relative).is_file():
                fail(f"Missing staged official artifact: {relative}")
        promote_with_rollback(staging, project_root, output_paths)
        promoted = True
    finally:
        if not promoted and staging.exists():
            shutil.rmtree(staging)

    print("--- INDEPENDENT READBACK VALIDATION ---")
    for name in SPLIT_ORDER:
        summary = validation["split_summary"][name]
        print(
            f"{name.upper()}_COUNTS= images:{summary['images']} "
            f"annotations:{summary['annotations']} zero_gt:{summary['zero_gt_images']} "
            f"categories:{summary['categories_present']}"
        )
        print(f"{name.upper()}_SHA256_COCO_JSON=", summary["sha256_coco_json"])
    print("IMAGE_OVERLAP=", validation["image_overlap"])
    print("IMAGE_UNION=", validation["image_union"])
    print("ANNOTATION_UNION=", validation["annotation_union"])
    print("MANIFEST_ROWS=", validation["manifest_rows"])
    print("FIXED_SPLIT_GATE= PASS")
    print("FILES_WRITTEN= 9")
    print("FIXED_SPLIT_CREATED= True")
    print("FIXED_SPLIT_VALIDATED= True")
    for relative in output_paths:
        print("OUTPUT_FILE=", project_root / relative)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("FIXED_SPLIT_GATE= ERROR")
        print("FIXED_SPLIT_CREATED= False")
        print("FIXED_SPLIT_VALIDATED= False")
        print(f"ERROR_TYPE= {type(error).__name__}", file=sys.stderr)
        print(f"ERROR_MESSAGE= {error}", file=sys.stderr)
        sys.exit(1)
