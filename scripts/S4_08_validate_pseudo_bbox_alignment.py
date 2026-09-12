#!/usr/bin/env python3
"""S4.08 — verify image/pseudo-box alignment for weak Teacher -> strong Student."""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config


ATOL = 1e-4
RTOL = 1e-6

CONFIGS = {
    "r50": "configs/ssl/s4_07_soft_teacher_r50_fpn_strong.py",
    "swin": "configs/ssl/s4_07_soft_teacher_swin_t_fpn_strong.py",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default="/workspace/ssod/project",
    )
    parser.add_argument(
        "--output",
        default=(
            "/workspace/ssod/project/artifacts/preflight/ssl/"
            "pseudo_bbox_alignment_report.json"
        ),
    )
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    output = Path(args.output).resolve()

    sys.path.insert(0, str(repo))

    from mmdet.models.detectors.semi_base import (
        SemiBaseDetector,
        bbox_project,
    )
    from mmdet.models.detectors.soft_teacher import SoftTeacher
    from mmdet.registry import DATASETS
    from mmdet.utils import register_all_modules

    register_all_modules()
    import src.transforms.grayscale_photometric  # noqa: F401

    checks: dict[str, bool] = {}

    # ------------------------------------------------------------
    # 1. Framework pseudo-box geometry path audit.
    # ------------------------------------------------------------
    semi_loss_src = inspect.getsource(SemiBaseDetector.loss)
    semi_get_src = inspect.getsource(SemiBaseDetector.get_pseudo_instances)
    semi_project_src = inspect.getsource(
        SemiBaseDetector.project_pseudo_instances
    )
    soft_get_src = inspect.getsource(SoftTeacher.get_pseudo_instances)

    checks["framework_teacher_branch_present"] = (
        "unsup_teacher" in semi_loss_src
    )
    checks["framework_student_branch_present"] = (
        "unsup_student" in semi_loss_src
    )
    checks["framework_teacher_projects_inverse_homography"] = (
        "homography_matrix).inverse()" in semi_get_src
        or "homography_matrix).inverse()" in soft_get_src
    )
    checks["framework_teacher_projects_to_ori_shape"] = (
        "ori_shape" in soft_get_src
    )
    checks["framework_student_projects_forward_homography"] = (
        "data_samples.homography_matrix" in semi_project_src
    )
    checks["framework_student_projects_to_img_shape"] = (
        "data_samples.img_shape" in semi_project_src
    )
    checks["softteacher_inherits_project_pseudo_instances"] = (
        SoftTeacher.project_pseudo_instances
        is SemiBaseDetector.project_pseudo_instances
    )

    # ------------------------------------------------------------
    # 2. Resolved config geometry contract.
    # ------------------------------------------------------------
    config_reports = {}

    for arch, relpath in CONFIGS.items():
        cfg = Config.fromfile(str(repo / relpath))

        weak = cfg.weak_pipeline
        strong = cfg.strong_pipeline

        weak_resize = next(
            x for x in weak if x["type"] == "Resize"
        )
        strong_resize = next(
            x for x in strong if x["type"] == "Resize"
        )

        weak_pack = next(
            x for x in weak if x["type"] == "PackDetInputs"
        )
        strong_pack = next(
            x for x in strong if x["type"] == "PackDetInputs"
        )

        arch_checks = {
            "weak_resize_scale_exact":
                tuple(weak_resize["scale"]) == (1333, 800),
            "strong_resize_scale_exact":
                tuple(strong_resize["scale"]) == (1333, 800),
            "weak_keep_ratio_true":
                weak_resize["keep_ratio"] is True,
            "strong_keep_ratio_true":
                strong_resize["keep_ratio"] is True,
            "weak_strong_resize_geometry_identical":
                tuple(weak_resize["scale"])
                == tuple(strong_resize["scale"])
                and weak_resize["keep_ratio"]
                == strong_resize["keep_ratio"],
            "weak_homography_retained":
                "homography_matrix"
                in tuple(weak_pack["meta_keys"]),
            "strong_homography_retained":
                "homography_matrix"
                in tuple(strong_pack["meta_keys"]),
        }

        for name, passed in arch_checks.items():
            checks[f"{arch}_{name}"] = bool(passed)

        config_reports[arch] = {
            "config": relpath,
            "checks": arch_checks,
        }

    # ------------------------------------------------------------
    # 3. Real-U runtime geometry.
    # ------------------------------------------------------------
    cfg = Config.fromfile(
        str(repo / CONFIGS["r50"])
    )

    dataset_cfg = runpy.run_path(
        str(repo / "configs/dataset/s2_coco_dataset.py")
    )
    u_cfg = copy.deepcopy(dataset_cfg["DATASETS"]["U_1pct"])
    u_cfg["pipeline"] = copy.deepcopy(cfg.unlabeled_pipeline)

    dataset = DATASETS.build(u_cfg)

    dataset_size = len(dataset)
    sample_count = min(32, dataset_size)
    indices = np.linspace(
        0,
        dataset_size - 1,
        num=sample_count,
        dtype=int,
    ).tolist()

    failures = []
    sample_reports = []

    global_max_abs = 0.0
    global_max_rel = 0.0
    worst_case = None

    max_padding_right = 0.0
    max_padding_bottom = 0.0
    samples_with_padding = 0
    distinct_shape_triples = set()

    for idx in indices:
        item = dataset[idx]

        inputs = item["inputs"]
        samples = item["data_samples"]

        weak_input = inputs["unsup_teacher"]
        strong_input = inputs["unsup_student"]

        weak = samples["unsup_teacher"]
        strong = samples["unsup_student"]

        H_weak = torch.tensor(
            np.asarray(weak.homography_matrix),
            dtype=torch.float32,
        )
        H_strong = torch.tensor(
            np.asarray(strong.homography_matrix),
            dtype=torch.float32,
        )

        ori_h, ori_w = tuple(weak.ori_shape)
        weak_h, weak_w = tuple(weak.img_shape)
        strong_h, strong_w = tuple(strong.img_shape)

        distinct_shape_triples.add(
            (
                tuple(weak.ori_shape),
                tuple(weak.img_shape),
                tuple(strong.img_shape),
            )
        )

        content_w_weak = min(
            float(weak_w),
            float(ori_w * H_weak[0, 0].item()),
        )
        content_h_weak = min(
            float(weak_h),
            float(ori_h * H_weak[1, 1].item()),
        )

        content_w_strong = min(
            float(strong_w),
            float(ori_w * H_strong[0, 0].item()),
        )
        content_h_strong = min(
            float(strong_h),
            float(ori_h * H_strong[1, 1].item()),
        )

        pad_right = max(
            0.0,
            float(weak_w) - content_w_weak,
        )
        pad_bottom = max(
            0.0,
            float(weak_h) - content_h_weak,
        )

        max_padding_right = max(
            max_padding_right,
            pad_right,
        )
        max_padding_bottom = max(
            max_padding_bottom,
            pad_bottom,
        )

        if pad_right > ATOL or pad_bottom > ATOL:
            samples_with_padding += 1

        cw = content_w_weak
        ch = content_h_weak

        # Synthetic pseudo boxes are intentionally restricted to valid
        # image content. Padding-only coordinates are not valid image
        # coordinates and are clipped when projected back to ori_shape.
        boxes_weak = torch.tensor([
            [
                0.00 * cw,
                0.00 * ch,
                1.00 * cw,
                1.00 * ch,
            ],
            [
                0.01 * cw,
                0.01 * ch,
                0.15 * cw,
                0.20 * ch,
            ],
            [
                0.05 * cw,
                0.10 * ch,
                0.40 * cw,
                0.50 * ch,
            ],
            [
                0.20 * cw,
                0.35 * ch,
                0.70 * cw,
                0.80 * ch,
            ],
            [
                0.55 * cw,
                0.15 * ch,
                0.95 * cw,
                0.60 * ch,
            ],
            [
                0.80 * cw,
                0.80 * ch,
                0.999 * cw,
                0.999 * ch,
            ],
        ], dtype=torch.float32)

        boxes_ori = bbox_project(
            boxes_weak.clone(),
            torch.linalg.inv(H_weak),
            tuple(weak.ori_shape),
        )

        boxes_strong = bbox_project(
            boxes_ori.clone(),
            H_strong,
            tuple(strong.img_shape),
        )

        abs_diff = (
            boxes_strong - boxes_weak
        ).abs()

        denom = torch.maximum(
            boxes_weak.abs(),
            torch.ones_like(boxes_weak),
        )
        rel_diff = abs_diff / denom

        sample_max_abs = float(
            abs_diff.max().item()
        )
        sample_max_rel = float(
            rel_diff.max().item()
        )

        if sample_max_abs > global_max_abs:
            global_max_abs = sample_max_abs
            pos = torch.nonzero(
                abs_diff == abs_diff.max(),
                as_tuple=False,
            )[0]

            bi = int(pos[0])
            ci = int(pos[1])

            worst_case = {
                "dataset_index": idx,
                "img_id": int(weak.img_id),
                "ori_shape": list(weak.ori_shape),
                "img_shape": list(weak.img_shape),
                "box_index": bi,
                "coord_index": ci,
                "weak_box": boxes_weak[bi].tolist(),
                "strong_box": boxes_strong[bi].tolist(),
                "abs_diff": sample_max_abs,
                "relative_diff": sample_max_rel,
            }

        global_max_rel = max(
            global_max_rel,
            sample_max_rel,
        )

        same_h = bool(np.allclose(
            np.asarray(weak.homography_matrix),
            np.asarray(strong.homography_matrix),
            atol=1e-7,
            rtol=0.0,
        ))

        roundtrip = bool(torch.allclose(
            boxes_weak,
            boxes_strong,
            atol=ATOL,
            rtol=RTOL,
        ))

        sample_checks = {
            "same_img_id":
                weak.img_id == strong.img_id,
            "same_img_path":
                weak.img_path == strong.img_path,
            "same_input_shape":
                tuple(weak_input.shape)
                == tuple(strong_input.shape),
            "same_img_shape":
                tuple(weak.img_shape)
                == tuple(strong.img_shape),
            "same_scale_factor":
                tuple(weak.scale_factor)
                == tuple(strong.scale_factor),
            "same_homography":
                same_h,
            "same_valid_content_extent":
                abs(
                    content_w_weak
                    - content_w_strong
                ) <= ATOL
                and abs(
                    content_h_weak
                    - content_h_strong
                ) <= ATOL,
            "pseudo_box_roundtrip_allclose":
                roundtrip,
            "hidden_u_gt_unused":
                len(weak.gt_instances.bboxes) == 0
                and len(strong.gt_instances.bboxes) == 0,
        }

        sample_report = {
            "dataset_index": idx,
            "img_id": int(weak.img_id),
            "ori_shape": list(weak.ori_shape),
            "weak_img_shape": list(weak.img_shape),
            "strong_img_shape": list(strong.img_shape),
            "valid_content_extent_wh": [
                content_w_weak,
                content_h_weak,
            ],
            "padding_right_bottom": [
                pad_right,
                pad_bottom,
            ],
            "max_abs_coord_diff": sample_max_abs,
            "max_relative_coord_diff": sample_max_rel,
            "checks": sample_checks,
        }

        sample_reports.append(sample_report)

        if not all(sample_checks.values()):
            failures.append(sample_report)

    checks["runtime_sample_count_32"] = (
        sample_count == 32
    )
    checks["runtime_all_samples_pass"] = (
        len(failures) == 0
    )
    checks["runtime_weak_strong_same_geometry"] = (
        all(
            r["checks"]["same_img_shape"]
            and r["checks"]["same_scale_factor"]
            and r["checks"]["same_homography"]
            for r in sample_reports
        )
    )
    checks["runtime_pseudo_box_alignment"] = (
        all(
            r["checks"][
                "pseudo_box_roundtrip_allclose"
            ]
            for r in sample_reports
        )
    )
    checks["runtime_hidden_u_gt_unused"] = (
        all(
            r["checks"]["hidden_u_gt_unused"]
            for r in sample_reports
        )
    )

    # Explicitly document the diagnostic pitfall discovered during S4.08:
    # padded coordinates are not valid image-content coordinates.
    padding_note = {
        "padding_is_part_of_tensor_shape": True,
        "padding_is_not_original_image_content": True,
        "bbox_project_clamps_to_ori_shape": True,
        "padding_only_coordinate_roundtrip_not_required": True,
        "reason": (
            "A box extending into padding is clipped when weak-view "
            "coordinates are projected back to ori_shape; therefore "
            "padding-only coordinates are not a valid invertibility test "
            "for image/pseudo-box alignment."
        ),
    }

    failed_checks = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    report = {
        "schema_version": "1.0",
        "stage": "S4",
        "task": "S4.08",
        "category": "Geometry",
        "action": "Verify image-pseudo-box alignment",
        "constraint": "No coordinate mismatch.",
        "expected_result": "PASS",
        "official_training_authorized": False,
        "final_test_authorized": False,
        "global_preflight_closed": False,
        "alignment_contract": {
            "source_branch": "unsup_teacher",
            "source_model": "EMA_TEACHER",
            "destination_branch": "unsup_student",
            "destination_model": "STUDENT",
            "framework_path": [
                "Teacher weak prediction coordinates",
                "bbox_project(H_weak^-1, ori_shape)",
                "original-image coordinates",
                "bbox_project(H_strong, strong_img_shape)",
                "Student strong coordinates",
            ],
            "numerical_tolerance": {
                "atol": ATOL,
                "rtol": RTOL,
                "comparison": "torch.allclose",
            },
        },
        "framework": {
            "semi_base_loss_source_sha256":
                hashlib.sha256(
                    semi_loss_src.encode("utf-8")
                ).hexdigest(),
            "semi_base_get_pseudo_instances_source_sha256":
                hashlib.sha256(
                    semi_get_src.encode("utf-8")
                ).hexdigest(),
            "semi_base_project_pseudo_instances_source_sha256":
                hashlib.sha256(
                    semi_project_src.encode("utf-8")
                ).hexdigest(),
            "soft_teacher_get_pseudo_instances_source_sha256":
                hashlib.sha256(
                    soft_get_src.encode("utf-8")
                ).hexdigest(),
        },
        "configs": config_reports,
        "runtime": {
            "dataset_key": "U_1pct",
            "dataset_size": dataset_size,
            "sample_count": sample_count,
            "indices": indices,
            "distinct_shape_triples":
                len(distinct_shape_triples),
            "samples_with_padding":
                samples_with_padding,
            "max_padding_right":
                max_padding_right,
            "max_padding_bottom":
                max_padding_bottom,
            "global_max_abs_coord_diff":
                global_max_abs,
            "global_max_relative_coord_diff":
                global_max_rel,
            "worst_case": worst_case,
            "failed_sample_count":
                len(failures),
            "failed_samples": failures,
            "samples": sample_reports,
        },
        "padding_diagnostic": padding_note,
        "checks": checks,
        "check_count": len(checks),
        "failed_checks": failed_checks,
        "all_checks_pass": not failed_checks,
        "no_coordinate_mismatch":
            len(failures) == 0,
    }

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print("EVIDENCE=", output)
    print("CHECK_COUNT=", len(checks))
    print("FAILED_CHECKS=", failed_checks)
    print("DATASET_SIZE=", dataset_size)
    print("SAMPLE_COUNT=", sample_count)
    print(
        "DISTINCT_SHAPE_TRIPLES=",
        len(distinct_shape_triples),
    )
    print(
        "SAMPLES_WITH_PADDING=",
        samples_with_padding,
    )
    print(
        "MAX_PADDING_RIGHT=",
        max_padding_right,
    )
    print(
        "MAX_PADDING_BOTTOM=",
        max_padding_bottom,
    )
    print(
        "GLOBAL_MAX_ABS_COORD_DIFF=",
        global_max_abs,
    )
    print(
        "GLOBAL_MAX_REL_COORD_DIFF=",
        global_max_rel,
    )
    print(
        "FAILED_SAMPLE_COUNT=",
        len(failures),
    )
    print(
        "HIDDEN_U_GT_USED=",
        not checks["runtime_hidden_u_gt_unused"],
    )
    print(
        "NO_COORDINATE_MISMATCH=",
        report["no_coordinate_mismatch"],
    )

    if failed_checks:
        print("S4_08_PSEUDO_BBOX_ALIGNMENT=FAIL")
        return 1

    print("S4_08_TRACKER_RESULT=PASS")
    print("S4_08_PSEUDO_BBOX_ALIGNMENT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())