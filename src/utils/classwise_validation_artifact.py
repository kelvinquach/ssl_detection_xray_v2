"""Shared writer for BEST-checkpoint class-wise validation AP artifacts."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


FIXED_VALIDATION_ANN_FILE = Path(
    "/workspace/ssod/data/coco/instances_val.json"
)

CLASSWISE_VALIDATION_FIELDS = (
    "category_id",
    "class_name",
    "AP_50_95",
    "AP50",
    "AP75",
)


def _load_fixed_validation_categories(
    ann_file: Path,
) -> list[tuple[int, str]]:
    payload = json.loads(
        Path(ann_file).read_text(encoding="utf-8")
    )
    categories = sorted(
        payload.get("categories", []),
        key=lambda item: int(item["id"]),
    )
    category_ids = [
        int(item["id"])
        for item in categories
    ]

    if category_ids != list(range(1, 15)):
        raise ValueError(
            "Fixed-validation category IDs must be exactly 1..14."
        )

    rows: list[tuple[int, str]] = []
    for item in categories:
        class_name = str(item["name"]).strip()
        if not class_name:
            raise ValueError(
                "Fixed-validation class names must be non-empty."
            )
        rows.append((int(item["id"]), class_name))

    if len({name for _, name in rows}) != 14:
        raise ValueError(
            "Fixed-validation class names must be unique."
        )

    return rows


def _ap_cell(
    metrics: Mapping[str, Any],
    key: str,
) -> float | str:
    if key not in metrics:
        raise KeyError(
            f"Validation metrics are missing required class-wise key {key!r}."
        )

    value = metrics[key]
    if isinstance(value, bool):
        raise TypeError(
            f"Class-wise validation metric {key!r} must be numeric."
        )

    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Class-wise validation metric {key!r} must be numeric."
        ) from exc

    if math.isnan(numeric) or numeric < 0.0:
        return "NA"
    if math.isinf(numeric) or numeric > 1.0:
        raise ValueError(
            f"Class-wise validation metric {key!r} is outside [0, 1]."
        )

    return numeric


def write_classwise_validation_ap(
    path: Path,
    metrics: Mapping[str, Any],
    *,
    ann_file: Path = FIXED_VALIDATION_ANN_FILE,
) -> None:
    """Atomically write class-wise AP for the selected BEST validation event."""
    categories = _load_fixed_validation_categories(ann_file)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )

    try:
        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=CLASSWISE_VALIDATION_FIELDS,
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()

            for category_id, class_name in categories:
                prefix = f"coco/classwise/{class_name}/"
                writer.writerow(
                    {
                        "category_id": category_id,
                        "class_name": class_name,
                        "AP_50_95": _ap_cell(
                            metrics,
                            prefix + "AP_50_95",
                        ),
                        "AP50": _ap_cell(
                            metrics,
                            prefix + "AP50",
                        ),
                        "AP75": _ap_cell(
                            metrics,
                            prefix + "AP75",
                        ),
                    }
                )

            handle.flush()
            os.fsync(handle.fileno())

        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


__all__ = [
    "CLASSWISE_VALIDATION_FIELDS",
    "FIXED_VALIDATION_ANN_FILE",
    "write_classwise_validation_ap",
]
