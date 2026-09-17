"""Locked Holm F2 multiplicity adjustment for S6.09.

Scientific contract:
- family members are exactly RQ3, RQ4, RQ6, RQ7;
- family size m = 4;
- procedure is Holm step-down;
- family-wise error rate alpha = 0.05;
- family membership cannot be changed or shrunk after observing results;
- individual confidence intervals are not adjusted here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

F2_FAMILY_ID = "F2"
F2_MEMBER_ORDER = ("RQ3", "RQ4", "RQ6", "RQ7")
F2_M = 4
F2_ALPHA = 0.05
F2_METHOD = "HOLM_STEP_DOWN"


class HolmF2Error(ValueError):
    """Raised when F2 input violates the locked multiplicity contract."""


def _coerce_p_value(value: object, member: str) -> float:
    if isinstance(value, bool):
        raise HolmF2Error(f"{member} p-value must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise HolmF2Error(f"{member} p-value must be numeric") from exc
    if not math.isfinite(result):
        raise HolmF2Error(f"{member} p-value must be finite")
    if result < 0.0 or result > 1.0:
        raise HolmF2Error(f"{member} p-value must be within [0, 1]")
    return result


def holm_f2(raw_p_values: Mapping[str, object]) -> dict[str, object]:
    """Apply locked Holm step-down adjustment to the exact F2 family."""
    if not isinstance(raw_p_values, Mapping):
        raise HolmF2Error("raw_p_values must be a mapping")

    observed_keys = tuple(raw_p_values.keys())
    if len(observed_keys) != F2_M or set(observed_keys) != set(F2_MEMBER_ORDER):
        raise HolmF2Error(
            "F2 requires exactly RQ3, RQ4, RQ6 and RQ7; family cannot be shrunk or expanded"
        )

    values = {
        member: _coerce_p_value(raw_p_values[member], member)
        for member in F2_MEMBER_ORDER
    }

    member_index = {member: index for index, member in enumerate(F2_MEMBER_ORDER)}
    ordered = sorted(
        F2_MEMBER_ORDER,
        key=lambda member: (values[member], member_index[member]),
    )

    ordered_rows: list[dict[str, object]] = []
    running_adjusted = 0.0
    still_rejecting = True

    for zero_index, member in enumerate(ordered):
        rank = zero_index + 1
        multiplier = F2_M - zero_index
        raw_p = values[member]
        threshold = F2_ALPHA / multiplier
        scaled_p = min(1.0, multiplier * raw_p)
        running_adjusted = max(running_adjusted, scaled_p)
        adjusted_p = min(1.0, running_adjusted)

        if still_rejecting and raw_p <= threshold:
            reject = True
        else:
            reject = False
            still_rejecting = False

        ordered_rows.append(
            {
                "rank": rank,
                "member": member,
                "raw_p_value": raw_p,
                "holm_multiplier": multiplier,
                "holm_threshold": threshold,
                "adjusted_p_value": adjusted_p,
                "reject_at_fwer_0_05": reject,
            }
        )

    by_member = {
        row["member"]: {
            "rank": row["rank"],
            "raw_p_value": row["raw_p_value"],
            "holm_multiplier": row["holm_multiplier"],
            "holm_threshold": row["holm_threshold"],
            "adjusted_p_value": row["adjusted_p_value"],
            "reject_at_fwer_0_05": row["reject_at_fwer_0_05"],
        }
        for row in ordered_rows
    }

    return {
        "family": F2_FAMILY_ID,
        "method": F2_METHOD,
        "family_size": F2_M,
        "fwer": F2_ALPHA,
        "member_order": list(F2_MEMBER_ORDER),
        "ordered_results": ordered_rows,
        "by_member": by_member,
        "family_membership_locked": True,
        "individual_ci_adjusted": False,
    }
