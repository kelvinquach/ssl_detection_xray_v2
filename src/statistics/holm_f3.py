"""Holm step-down multiplicity adjustment for locked RQ3 family F3.

S6.10 scope:
- exactly six prespecified RQ3 pairwise budget contrasts;
- raw two-sided p-values from the S6.04 paired-difference tests;
- Holm step-down control of FWER at 0.05;
- individual effect confidence intervals are not Holm-adjusted.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

F3_FAMILY_ID = "F3"
F3_MEMBER_ORDER = (
    "5%-1%",
    "10%-1%",
    "20%-1%",
    "10%-5%",
    "20%-5%",
    "20%-10%",
)
F3_M = 6
F3_ALPHA = 0.05
F3_METHOD = "HOLM_STEP_DOWN"


class HolmF3Error(ValueError):
    """Raised when an F3 Holm input violates the locked scientific contract."""


def _validate_raw_p_values(raw_p_values: Mapping[str, Real]) -> dict[str, float]:
    if not isinstance(raw_p_values, Mapping):
        raise HolmF3Error("raw_p_values must be a mapping of F3 member to raw p-value")

    observed_members = set(raw_p_values.keys())
    expected_members = set(F3_MEMBER_ORDER)
    if observed_members != expected_members:
        missing = [member for member in F3_MEMBER_ORDER if member not in observed_members]
        extra = sorted(observed_members - expected_members)
        raise HolmF3Error(
            "F3 membership mismatch: "
            f"missing={missing or []}, extra={extra or []}; "
            f"expected exactly {list(F3_MEMBER_ORDER)}"
        )

    validated: dict[str, float] = {}
    for member in F3_MEMBER_ORDER:
        value = raw_p_values[member]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise HolmF3Error(f"{member} raw p-value must be a real numeric scalar")
        p_value = float(value)
        if not math.isfinite(p_value):
            raise HolmF3Error(f"{member} raw p-value must be finite")
        if p_value < 0.0 or p_value > 1.0:
            raise HolmF3Error(f"{member} raw p-value must lie in [0, 1]")
        validated[member] = p_value

    return validated


def holm_f3(raw_p_values: Mapping[str, Real]) -> dict[str, Any]:
    """Apply the locked Holm step-down procedure to family F3.

    Ties in raw p-values are resolved deterministically by F3_MEMBER_ORDER.
    """
    validated = _validate_raw_p_values(raw_p_values)
    canonical_index = {member: idx for idx, member in enumerate(F3_MEMBER_ORDER)}

    ordered = sorted(
        validated.items(),
        key=lambda item: (item[1], canonical_index[item[0]]),
    )

    ordered_results: list[dict[str, Any]] = []
    running_adjusted = 0.0
    step_down_active = True

    for rank, (member, raw_p_value) in enumerate(ordered, start=1):
        multiplier = F3_M - rank + 1
        threshold = F3_ALPHA / multiplier
        running_adjusted = max(
            running_adjusted,
            min(1.0, multiplier * raw_p_value),
        )
        reject = bool(step_down_active and raw_p_value <= threshold)
        if not reject:
            step_down_active = False

        ordered_results.append(
            {
                "rank": rank,
                "member": member,
                "raw_p_value": raw_p_value,
                "holm_multiplier": multiplier,
                "holm_threshold": threshold,
                "adjusted_p_value": running_adjusted,
                "reject_at_fwer_0_05": reject,
            }
        )

    by_member = {
        result["member"]: {
            key: value
            for key, value in result.items()
            if key != "member"
        }
        for result in ordered_results
    }

    return {
        "family": F3_FAMILY_ID,
        "family_size": F3_M,
        "method": F3_METHOD,
        "fwer": F3_ALPHA,
        "member_order": list(F3_MEMBER_ORDER),
        "family_membership_locked": True,
        "individual_ci_adjusted": False,
        "ordered_results": ordered_results,
        "by_member": by_member,
    }
