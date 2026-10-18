"""S9.14 final-test backup adapter.

Reuses the verified S9 per-attempt Google Drive transport/checksum/round-trip
verification while adding the permanent final-test evidence paths.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.s9_active_backup import ActiveBackupManager


FINAL_TEST_BACKUP_PATHS = (
    "evaluation/final_test/coco_metrics.json",
    "evaluation/final_test/operating_point_metrics.json",
    "evaluation/final_test/classwise_ap.csv",
    "evaluation/final_test/negative_metrics.json",
    "evaluation/final_test/detections.json",
    "evaluation/final_test/evaluation_manifest.json",
    "evaluation/final_test/final_evaluation_report.json",
)


class FinalTestActiveBackupManager(ActiveBackupManager):
    """ActiveBackupManager with S9.14 final-test artifact discovery."""

    def _existing_files(self) -> list[tuple[str, Path]]:
        discovered = {
            relative: source
            for relative, source in super()._existing_files()
        }
        for relative in FINAL_TEST_BACKUP_PATHS:
            source = self.attempt_dir / relative
            if source.is_file():
                discovered[relative] = source
        return [(key, discovered[key]) for key in sorted(discovered)]

    def _manifest(
        self,
        reason: str,
        verification_status: str,
    ) -> dict[str, Any]:
        payload = super()._manifest(reason, verification_status)
        payload.update(
            {
                "backup_scope": "S9_14_FINAL_TEST",
                "final_test_artifact_paths": list(FINAL_TEST_BACKUP_PATHS),
                "scientific_protocol_changed": False,
            }
        )
        return payload

    def start_final_test(self) -> None:
        """Start verified backup, then record explicit final-test phase entry."""
        super().start()
        self.sync("final_test_evaluation_start", raise_on_error=True)
