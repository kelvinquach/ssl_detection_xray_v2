"""Per-class confidence history for SSL with non-VLM refinement.

SSL + REF means:
Semi-Supervised Learning with non-VLM pseudo-label refinement.

Each detection class keeps a fixed-length history of Teacher confidence
scores from earlier successful optimizer updates.

Scores from the current training batch are staged first. They become part
of the committed history only after the optimizer update succeeds.
"""

from collections import deque
from math import isfinite
from typing import Mapping, Sequence


class ClassConfidenceHistory:
    """Maintain committed Teacher confidence scores for each class."""

    def __init__(
        self,
        number_of_classes: int = 14,
        maximum_scores_per_class: int = 256,
    ) -> None:
        if number_of_classes <= 0:
            raise ValueError(
                "number_of_classes must be positive"
            )

        if maximum_scores_per_class <= 0:
            raise ValueError(
                "maximum_scores_per_class must be positive"
            )

        self.number_of_classes = number_of_classes
        self.maximum_scores_per_class = (
            maximum_scores_per_class
        )

        self._committed_scores = {
            class_id: deque(
                maxlen=maximum_scores_per_class
            )
            for class_id in range(number_of_classes)
        }

        self._staged_scores = None

    def _validate_class_id(
        self,
        class_id: int,
    ) -> None:
        if (
            isinstance(class_id, bool)
            or not isinstance(class_id, int)
            or class_id < 0
            or class_id >= self.number_of_classes
        ):
            raise ValueError(
                "class_id is outside the configured class range"
            )

    @staticmethod
    def _validate_score(
        score: float,
    ) -> float:
        score = float(score)

        if not isfinite(score):
            raise ValueError(
                "confidence scores must be finite"
            )

        if score < 0.0 or score > 1.0:
            raise ValueError(
                "confidence scores must be between 0 and 1"
            )

        return score

    def get_committed_scores(
        self,
        class_id: int,
    ) -> tuple[float, ...]:
        """Return committed scores for one class in insertion order."""
        self._validate_class_id(
            class_id
        )

        return tuple(
            self._committed_scores[class_id]
        )

    def stage_current_batch(
        self,
        scores_by_class: Mapping[
            int,
            Sequence[float],
        ],
    ) -> None:
        """Stage current-batch scores without changing committed history."""
        if self._staged_scores is not None:
            raise RuntimeError(
                "current batch scores are already staged"
            )

        staged_scores = {
            class_id: []
            for class_id in range(
                self.number_of_classes
            )
        }

        for class_id, scores in scores_by_class.items():
            self._validate_class_id(
                class_id
            )

            staged_scores[class_id] = [
                self._validate_score(score)
                for score in scores
            ]

        self._staged_scores = staged_scores

    def commit_staged_scores(
        self,
    ) -> None:
        """Commit staged scores after a successful optimizer update."""
        if self._staged_scores is None:
            raise RuntimeError(
                "there are no staged scores to commit"
            )

        for class_id in range(
            self.number_of_classes
        ):
            self._committed_scores[
                class_id
            ].extend(
                self._staged_scores[class_id]
            )

        self._staged_scores = None

    def discard_staged_scores(
        self,
    ) -> None:
        """Discard staged scores when the optimizer update is skipped."""
        if self._staged_scores is None:
            raise RuntimeError(
                "there are no staged scores to discard"
            )

        self._staged_scores = None

    def committed_state(
        self,
    ) -> dict:
        """Return only committed state for checkpoint serialization."""
        return {
            "number_of_classes": self.number_of_classes,
            "maximum_scores_per_class": (
                self.maximum_scores_per_class
            ),
            "scores_by_class": {
                str(class_id): list(
                    self._committed_scores[
                        class_id
                    ]
                )
                for class_id in range(
                    self.number_of_classes
                )
            },
        }

    def restore_committed_state(
        self,
        state: dict,
    ) -> None:
        """Restore committed history from checkpoint state."""
        if not isinstance(state, dict):
            raise TypeError(
                "history state must be a dictionary"
            )

        if state.get(
            "number_of_classes"
        ) != self.number_of_classes:
            raise ValueError(
                "number_of_classes does not match"
            )

        if state.get(
            "maximum_scores_per_class"
        ) != self.maximum_scores_per_class:
            raise ValueError(
                "maximum_scores_per_class does not match"
            )

        scores_by_class = state.get(
            "scores_by_class"
        )

        if not isinstance(
            scores_by_class,
            dict,
        ):
            raise ValueError(
                "scores_by_class is missing or invalid"
            )

        restored_scores = {}

        for class_id in range(
            self.number_of_classes
        ):
            key = str(class_id)

            if key not in scores_by_class:
                raise ValueError(
                    f"missing history for class {class_id}"
                )

            restored_scores[class_id] = [
                self._validate_score(score)
                for score in scores_by_class[key]
            ]

            if (
                len(restored_scores[class_id])
                > self.maximum_scores_per_class
            ):
                raise ValueError(
                    "restored class history exceeds configured maximum"
                )

        for class_id in range(
            self.number_of_classes
        ):
            self._committed_scores[
                class_id
            ].clear()

            self._committed_scores[
                class_id
            ].extend(
                restored_scores[class_id]
            )

        self._staged_scores = None
