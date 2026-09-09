#!/usr/bin/env python3
"""S2.01 — Minimal fixed-COCO loading validation. No training."""

from __future__ import annotations

import argparse
import os
import runpy
from pathlib import Path


EXPECTED = {
    "train": 3426,
    "val": 734,
    "test": 734,
    "L_1pct": 34,
    "L_5pct": 171,
    "L_10pct": 343,
    "L_20pct": 685,
    "U_1pct": 3392,
    "U_5pct": 3255,
    "U_10pct": 3083,
    "U_20pct": 2741,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/workspace/ssod/project")
    parser.add_argument("--ssod-root", default="/workspace/ssod")
    parser.add_argument("--prepare-compat-link", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo_root)
    ssod = Path(args.ssod_root)

    image_root = ssod / "data/images"
    compat_root = ssod / "data/mmdet_images"
    compat_link = compat_root / "train"

    if not image_root.is_dir():
        raise RuntimeError(f"missing canonical image root: {image_root}")

    if not compat_link.is_symlink():
        if compat_link.exists():
            raise RuntimeError(f"compatibility path is not symlink: {compat_link}")
        if not args.prepare_compat_link:
            raise RuntimeError(
                f"missing compatibility link: {compat_link}; "
                "use --prepare-compat-link"
            )
        compat_root.mkdir(parents=True, exist_ok=True)
        compat_link.symlink_to(image_root, target_is_directory=True)

    if compat_link.resolve() != image_root.resolve():
        raise RuntimeError("compatibility link target mismatch")

    os.environ["SSOD_ROOT"] = str(ssod)
    cfg = runpy.run_path(str(repo / "configs/dataset/s2_coco_dataset.py"))

    from mmengine.registry import init_default_scope
    from mmdet.registry import DATASETS

    init_default_scope("mmdet")

    dataset_cfgs = cfg["DATASETS"]

    if set(dataset_cfgs) != set(EXPECTED):
        raise RuntimeError("fixed dataset identities mismatch")

    for name, expected_len in EXPECTED.items():
        dataset = DATASETS.build(dataset_cfgs[name])
        dataset.full_init()

        if len(dataset) != expected_len:
            raise RuntimeError(
                f"{name}: expected {expected_len}, observed {len(dataset)}"
            )

        sample = dataset[0]
        if sample is None:
            raise RuntimeError(f"{name}: sample 0 failed to load")

        info = dataset.get_data_info(0)
        if not Path(info["img_path"]).is_file():
            raise RuntimeError(f"{name}: sample path does not resolve")

        print(f"PASS {name}: images={len(dataset)} sample_resolves=True")

    print("S2_01_COCO_DATASET_LOADING=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())