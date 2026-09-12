"""Operational LATEST_RESUME checkpoint handling for SSL.

S4.17 reuses the already-validated S3.13 operational checkpoint mechanism
without changing its persistence semantics. MMEngine checkpoint state contains
the complete SSL model state_dict, including distinct Student and Teacher
states, while ProtocolResumeCheckpointHook owns seed/counter/RNG protocol
state.

This wrapper exists only to keep SSL configuration semantics explicit.
"""

from mmdet.registry import HOOKS

from src.utils.sup_latest_resume_checkpoint_hook import (
    SupervisedLatestResumeCheckpointHook,
)


@HOOKS.register_module()
class SSLLatestResumeCheckpointHook(
    SupervisedLatestResumeCheckpointHook
):
    """Materialize and provenance-track operational SSL LATEST_RESUME."""


__all__ = ["SSLLatestResumeCheckpointHook"]