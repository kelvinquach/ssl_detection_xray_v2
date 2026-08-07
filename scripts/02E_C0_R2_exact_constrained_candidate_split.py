"""Phase 2E-C0-R2 — exact-constrained multilabel candidate split.

Preserves the C0-R1 initial candidate and exact-size repair exactly, then adds
a deterministic exact No Finding swap repair. Creates and audits the final
candidate in memory only; it never writes split manifests or modifies data.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


COCO_PATH = Path("data/processed/coco/coco_master_jpg.json")
RANDOM_SEED = 42
SPLIT_ORDER = ("train", "val", "test")
EXPECTED_SPLIT_SIZES = {"train": 3426, "val": 734, "test": 734}
EXPECTED_NO_FINDING = {"train": 350, "val": 75, "test": 75}
MAX_PREVALENCE_DEVIATION_PP = 1.0
MAX_CARDINALITY_DEVIATION = 0.10
MIN_PNEUMOTHORAX_VAL_TEST = 10


def fail(message: str) -> None:
    raise RuntimeError(message)


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "NOT_INSTALLED"


def stable_id_key(value: object) -> str:
    """Return the exact deterministic image-ID key used by C0-R1."""
    return str(value)


def sha256_image_ids(values: list[Any]) -> str:
    canonical = "\n".join(str(v) for v in sorted(values, key=stable_id_key))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_coco() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, int]]:
    if not COCO_PATH.is_file():
        fail(f"COCO file not found: {COCO_PATH}")
    with COCO_PATH.open("r", encoding="utf-8") as stream:
        coco = json.load(stream)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])
    if len(images) != 4894 or len(annotations) != 36096 or len(categories) != 14:
        fail(
            "Locked input counts changed: "
            f"images={len(images)}, annotations={len(annotations)}, "
            f"categories={len(categories)}"
        )

    image_ids_list = [item["id"] for item in images]
    if len(set(image_ids_list)) != len(image_ids_list):
        fail("Duplicate COCO image IDs detected")
    category_ids = [item["id"] for item in categories]
    if len(set(category_ids)) != 14:
        fail("Duplicate COCO category IDs detected")

    image_ids = np.asarray(image_ids_list, dtype=object)
    image_index = {image_id: index for index, image_id in enumerate(image_ids_list)}
    category_index = {category_id: index for index, category_id in enumerate(category_ids)}
    labels_14 = np.zeros((len(images), 14), dtype=np.uint8)
    bad_image_references = 0
    bad_category_references = 0
    for annotation in annotations:
        img_idx = image_index.get(annotation.get("image_id"))
        cat_idx = category_index.get(annotation.get("category_id"))
        if img_idx is None:
            bad_image_references += 1
            continue
        if cat_idx is None:
            bad_category_references += 1
            continue
        labels_14[img_idx, cat_idx] = 1

    if bad_image_references or bad_category_references:
        fail("COCO contains invalid image/category references")
    no_finding = (labels_14.sum(axis=1) == 0).astype(np.uint8).reshape(-1, 1)
    if int(no_finding.sum()) != 500:
        fail(f"Expected 500 zero-GT images, found {int(no_finding.sum())}")
    names = [str(item["name"]) for item in categories]
    counts = {
        "images": len(images), "annotations": len(annotations),
        "categories": len(categories), "bad_images": bad_image_references,
        "bad_categories": bad_category_references,
    }
    return image_ids, labels_14, no_finding, names, counts


def make_initial_candidate(matrix: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    dummy_x = np.zeros((len(matrix), 1), dtype=np.uint8)
    outer = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=0.30, random_state=seed
    )
    train_idx, holdout_idx = next(outer.split(dummy_x, matrix))
    holdout_x = np.zeros((len(holdout_idx), 1), dtype=np.uint8)
    inner = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=0.50, random_state=seed
    )
    val_local, test_local = next(
        inner.split(holdout_x, matrix[holdout_idx])
    )
    return {
        "train": np.sort(train_idx.astype(np.int64)),
        "val": np.sort(holdout_idx[val_local]),
        "test": np.sort(holdout_idx[test_local]),
    }


def balance_objective(members: dict[str, set[int]], matrix: np.ndarray) -> tuple[float, float]:
    """Use the exact 15-indicator prevalence objective locked in C0-R1."""
    full_prev = matrix.mean(axis=0)
    deviations: list[np.ndarray] = []
    for name in SPLIT_ORDER:
        idx = np.asarray(sorted(members[name]), dtype=np.int64)
        deviations.append(np.abs(matrix[idx].mean(axis=0) - full_prev))
    matrix_dev = np.vstack(deviations)
    return float(matrix_dev.max()), float(np.square(matrix_dev).sum())


def exact_size_repair(
    initial: dict[str, np.ndarray], image_ids: np.ndarray, matrix: np.ndarray
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    members = {name: set(map(int, initial[name])) for name in SPLIT_ORDER}
    log: list[dict[str, Any]] = []
    while any(len(members[name]) != EXPECTED_SPLIT_SIZES[name] for name in SPLIT_ORDER):
        sources = [name for name in SPLIT_ORDER if len(members[name]) > EXPECTED_SPLIT_SIZES[name]]
        destinations = [name for name in SPLIT_ORDER if len(members[name]) < EXPECTED_SPLIT_SIZES[name]]
        if not sources or not destinations:
            fail("Exact-size repair reached an inconsistent state")
        best: tuple[Any, ...] | None = None
        for source in sources:
            for destination in destinations:
                for idx in sorted(members[source], key=lambda i: stable_id_key(image_ids[i])):
                    members[source].remove(idx)
                    members[destination].add(idx)
                    objective = balance_objective(members, matrix)
                    members[destination].remove(idx)
                    members[source].add(idx)
                    key = tuple(round(x, 15) for x in objective) + (
                        SPLIT_ORDER.index(source), SPLIT_ORDER.index(destination),
                        stable_id_key(image_ids[idx]),
                    )
                    if best is None or key < best[0]:
                        best = (key, source, destination, idx)
        if best is None:
            fail("No exact-size repair move found")
        _, source, destination, idx = best
        members[source].remove(idx)
        members[destination].add(idx)
        log.append({"iteration": len(log) + 1, "image_id": image_ids[idx], "source": source, "destination": destination})
    return {name: np.asarray(sorted(members[name]), dtype=np.int64) for name in SPLIT_ORDER}, log


def abnormal_objective(
    members: dict[str, set[int]], counts: dict[str, np.ndarray], total_counts: np.ndarray
) -> tuple[float, float, float]:
    total = sum(map(len, members.values()))
    full_prev = total_counts / float(total)
    full_card = float(total_counts.sum() / total)
    dev, card = [], []
    for name in SPLIT_ORDER:
        size = len(members[name])
        dev.append(np.abs(counts[name] / float(size) - full_prev))
        card.append(abs(float(counts[name].sum() / size) - full_card))
    stacked = np.vstack(dev)
    return float(stacked.max()), float(np.square(stacked).sum()), max(card)


def pneumothorax_column(category_names: list[str]) -> int:
    found = [i for i, name in enumerate(category_names) if name.strip().lower() == "pneumothorax"]
    if len(found) != 1:
        fail(f"Expected exactly one Pneumothorax category, found {len(found)}")
    return found[0]


def hard_constraints(counts: dict[str, np.ndarray], ptx_col: int) -> bool:
    return bool(
        all(np.all(counts[name] > 0) for name in SPLIT_ORDER)
        and counts["val"][ptx_col] >= MIN_PNEUMOTHORAX_VAL_TEST
        and counts["test"][ptx_col] >= MIN_PNEUMOTHORAX_VAL_TEST
    )


def exact_no_finding_swap_repair(
    candidate: dict[str, np.ndarray], image_ids: np.ndarray, labels: np.ndarray,
    no_finding: np.ndarray, category_names: list[str],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    members = {name: set(map(int, candidate[name])) for name in SPLIT_ORDER}
    counts = {name: labels[sorted(members[name])].sum(axis=0, dtype=np.int64) for name in SPLIT_ORDER}
    total_counts = labels.sum(axis=0, dtype=np.int64)
    nf = no_finding.reshape(-1)
    ptx_col = pneumothorax_column(category_names)
    log: list[dict[str, Any]] = []

    def nf_counts() -> dict[str, int]:
        return {name: int(nf[sorted(members[name])].sum()) for name in SPLIT_ORDER}

    while nf_counts() != EXPECTED_NO_FINDING:
        current = nf_counts()
        sources = [name for name in SPLIT_ORDER if current[name] > EXPECTED_NO_FINDING[name]]
        destinations = [name for name in SPLIT_ORDER if current[name] < EXPECTED_NO_FINDING[name]]
        best: tuple[Any, ...] | None = None
        for source in sources:
            nf_indices = sorted((i for i in members[source] if nf[i] == 1), key=lambda i: stable_id_key(image_ids[i]))
            for destination in destinations:
                abnormal_indices = sorted((i for i in members[destination] if nf[i] == 0), key=lambda i: stable_id_key(image_ids[i]))
                for nf_idx in nf_indices:
                    for abnormal_idx in abnormal_indices:
                        vector = labels[abnormal_idx].astype(np.int64)
                        counts[source] += vector
                        counts[destination] -= vector
                        if hard_constraints(counts, ptx_col):
                            objective = abnormal_objective(members, counts, total_counts)
                            key = tuple(round(x, 15) for x in objective) + (
                                SPLIT_ORDER.index(source), SPLIT_ORDER.index(destination),
                                stable_id_key(image_ids[nf_idx]), stable_id_key(image_ids[abnormal_idx]),
                            )
                            if best is None or key < best[0]:
                                best = (key, source, destination, nf_idx, abnormal_idx, objective)
                        counts[source] -= vector
                        counts[destination] += vector
        if best is None:
            fail("No eligible deterministic No Finding swap was found")
        _, source, destination, nf_idx, abnormal_idx, objective = best
        members[source].remove(nf_idx); members[destination].add(nf_idx)
        members[destination].remove(abnormal_idx); members[source].add(abnormal_idx)
        vector = labels[abnormal_idx].astype(np.int64)
        counts[source] += vector; counts[destination] -= vector
        log.append({
            "iteration": len(log) + 1, "no_finding_image_id": image_ids[nf_idx],
            "abnormal_image_id": image_ids[abnormal_idx], "no_finding_source": source,
            "no_finding_destination": destination, "abnormal_source": destination,
            "abnormal_destination": source, "objective": objective,
        })
    repaired = {name: np.asarray(sorted(members[name]), dtype=np.int64) for name in SPLIT_ORDER}
    if {name: len(repaired[name]) for name in SPLIT_ORDER} != EXPECTED_SPLIT_SIZES:
        fail("No Finding repair changed split sizes")
    return repaired, log


def make_candidate(image_ids: np.ndarray, labels: np.ndarray, no_finding: np.ndarray, names: list[str]):
    matrix = np.hstack([labels, no_finding])
    initial = make_initial_candidate(matrix, RANDOM_SEED)
    exact_size, size_log = exact_size_repair(initial, image_ids, matrix)
    final, nf_log = exact_no_finding_swap_repair(exact_size, image_ids, labels, no_finding, names)
    return initial, exact_size, final, size_log, nf_log


def candidates_equal(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> bool:
    return all(np.array_equal(left[name], right[name]) for name in SPLIT_ORDER)


def main() -> None:
    print("=== PHASE 2E-C0-R2: EXACT-CONSTRAINED MULTILABEL CANDIDATE SPLIT ===")
    print("MODE= READ_ONLY_IN_MEMORY")
    print("COCO=", COCO_PATH)
    print("RANDOM_SEED=", RANDOM_SEED)
    print("SEED_POLICY= PRE_SPECIFIED_LOCKED_NO_SEED_SEARCH")
    print("PYTHON=", platform.python_version(), "|", sys.version.replace("\n", " "))
    print("NUMPY=", np.__version__)
    print("SCIKIT_LEARN=", package_version("scikit-learn"))
    print("ITERATIVE_STRATIFICATION=", package_version("iterative-stratification"))

    image_ids, labels, no_finding, names, input_counts = load_coco()
    print("--- INPUT COUNTS ---")
    print("TOTAL_IMAGES=", input_counts["images"])
    print("TOTAL_ANNOTATIONS=", input_counts["annotations"])
    print("TOTAL_CATEGORIES=", input_counts["categories"])
    print("BAD_IMAGE_REFERENCES=", input_counts["bad_images"])
    print("BAD_CATEGORY_REFERENCES=", input_counts["bad_categories"])
    print("ZERO_GT_IMAGES=", int(no_finding.sum()))

    initial, exact_size, candidate, size_log, nf_log = make_candidate(image_ids, labels, no_finding, names)
    r_initial, r_exact, repeated, r_size_log, r_nf_log = make_candidate(image_ids, labels, no_finding, names)
    initial_sizes = {name: len(initial[name]) for name in SPLIT_ORDER}
    exact_sizes = {name: len(exact_size[name]) for name in SPLIT_ORDER}
    actual_sizes = {name: len(candidate[name]) for name in SPLIT_ORDER}

    print("--- INITIAL CANDIDATE ---"); print("INITIAL_SIZES=", initial_sizes)
    print("--- EXACT-SIZE REPAIR ---"); print("SIZE_REPAIR_MOVES=", len(size_log))
    print("TARGET_SIZES=", EXPECTED_SPLIT_SIZES); print("EXACT_SIZE_REPAIRED_SIZES=", exact_sizes)
    for item in size_log:
        print("SIZE_REPAIR_MOVE=", item["iteration"], "| IMAGE_ID=", item["image_id"], "| FROM=", item["source"], "| TO=", item["destination"])
    print("--- EXACT NO FINDING SWAP REPAIR ---"); print("NO_FINDING_SWAP_MOVES=", len(nf_log))
    for item in nf_log:
        print("NO_FINDING_SWAP=", item["iteration"], "| NO_FINDING_IMAGE_ID=", item["no_finding_image_id"], "| NO_FINDING_FROM=", item["no_finding_source"], "| NO_FINDING_TO=", item["no_finding_destination"], "| ABNORMAL_IMAGE_ID=", item["abnormal_image_id"], "| ABNORMAL_FROM=", item["abnormal_source"], "| ABNORMAL_TO=", item["abnormal_destination"])
    print("FINAL_SIZES=", actual_sizes)

    split_sets = {name: set(image_ids[candidate[name]].tolist()) for name in SPLIT_ORDER}
    overlaps = {
        "train_val": len(split_sets["train"] & split_sets["val"]),
        "train_test": len(split_sets["train"] & split_sets["test"]),
        "val_test": len(split_sets["val"] & split_sets["test"]),
    }
    union_ids = set().union(*split_sets.values())
    reproducible = (
        candidates_equal(initial, r_initial) and candidates_equal(exact_size, r_exact)
        and candidates_equal(candidate, repeated) and size_log == r_size_log and nf_log == r_nf_log
    )
    print("--- CANDIDATE STRUCTURE ---")
    print("TRAIN_VAL_OVERLAP=", overlaps["train_val"]); print("TRAIN_TEST_OVERLAP=", overlaps["train_test"]); print("VAL_TEST_OVERLAP=", overlaps["val_test"])
    print("UNION_SIZE=", len(union_ids)); print("INITIAL_REPRODUCIBLE=", candidates_equal(initial, r_initial))
    print("EXACT_SIZE_REPRODUCIBLE=", candidates_equal(exact_size, r_exact)); print("FINAL_REPAIRED_REPRODUCIBLE=", candidates_equal(candidate, repeated))
    print("SIZE_REPAIR_LOG_REPRODUCIBLE=", size_log == r_size_log); print("NO_FINDING_SWAP_LOG_REPRODUCIBLE=", nf_log == r_nf_log)

    full_prev = labels.mean(axis=0) * 100.0
    full_card = float(labels.sum(axis=1).mean())
    print("FULL_MEAN_LABEL_CARDINALITY=", round(full_card, 6))
    all_present = exact_nf = prev_pass = card_pass = True
    split_class_counts: dict[str, np.ndarray] = {}
    for name in SPLIT_ORDER:
        idx = candidate[name]
        counts = labels[idx].sum(axis=0, dtype=np.int64); split_class_counts[name] = counts
        prevalence = labels[idx].mean(axis=0) * 100.0
        deviation = np.abs(prevalence - full_prev)
        nf_count = int(no_finding[idx].sum())
        cardinality = float(labels[idx].sum(axis=1).mean())
        card_deviation = abs(cardinality - full_card)
        all_present = all_present and bool(np.all(counts > 0))
        exact_nf = exact_nf and nf_count == EXPECTED_NO_FINDING[name]
        prev_pass = prev_pass and bool(np.all(deviation <= MAX_PREVALENCE_DEVIATION_PP + 1e-12))
        card_pass = card_pass and card_deviation <= MAX_CARDINALITY_DEVIATION + 1e-12
        print(f"--- {name.upper()} ---"); print("SIZE=", len(idx)); print("NO_FINDING=", nf_count, "TARGET=", EXPECTED_NO_FINDING[name])
        print("MEAN_LABEL_CARDINALITY=", round(cardinality, 6)); print("CARDINALITY_ABS_DEVIATION=", round(card_deviation, 6))
        print("MAX_PREVALENCE_DEVIATION_PP=", round(float(deviation.max()), 6)); print("ALL_CLASSES_PRESENT=", bool(np.all(counts > 0)))
        print("SHA256_IMAGE_IDS=", sha256_image_ids(list(split_sets[name])))
        print("CLASS | COUNT | PREVALENCE_PERCENT | FULL_PERCENT | DEVIATION_PP")
        for class_name, count, split_p, full_p, dev in zip(names, counts, prevalence, full_prev, deviation):
            print(class_name, "|", int(count), "|", round(float(split_p), 6), "|", round(float(full_p), 6), "|", round(float(dev), 6))

    ptx_col = pneumothorax_column(names)
    ptx_counts = {name: int(split_class_counts[name][ptx_col]) for name in SPLIT_ORDER}
    ptx_pass = ptx_counts["val"] >= 10 and ptx_counts["test"] >= 10
    size_pass = actual_sizes == EXPECTED_SPLIT_SIZES
    zero_overlap = all(value == 0 for value in overlaps.values())
    completeness = len(union_ids) == len(image_ids) and union_ids == set(image_ids.tolist())
    print("--- HARD GUARDRAILS ---")
    print("EXACT_SIZE_PASS=", size_pass); print("ZERO_OVERLAP_PASS=", zero_overlap); print("COMPLETENESS_PASS=", completeness)
    print("REPRODUCIBILITY_PASS=", reproducible); print("ALL_CLASSES_PRESENT_PASS=", all_present)
    print("EXACT_NO_FINDING_ALLOCATION_PASS=", exact_nf); print("PREVALENCE_DEVIATION_LE_1PP_PASS=", prev_pass)
    print("CARDINALITY_DEVIATION_LE_0.10_PASS=", card_pass); print("PNEUMOTHORAX_COUNTS=", ptx_counts)
    print("PNEUMOTHORAX_VAL_TEST_GE_10_PASS=", ptx_pass)
    final_pass = all((size_pass, zero_overlap, completeness, reproducible, all_present, exact_nf, prev_pass, card_pass, ptx_pass))
    print("CANDIDATE_SPLIT_GATE=", "PASS" if final_pass else "FAIL")
    print("FILES_WRITTEN= 0"); print("FIXED_SPLIT_CREATED= False")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("CANDIDATE_SPLIT_GATE= ERROR")
        print("FILES_WRITTEN= 0")
        print("FIXED_SPLIT_CREATED= False")
        print(f"ERROR_TYPE= {type(error).__name__}", file=sys.stderr)
        print(f"ERROR_MESSAGE= {error}", file=sys.stderr)
        sys.exit(1)
