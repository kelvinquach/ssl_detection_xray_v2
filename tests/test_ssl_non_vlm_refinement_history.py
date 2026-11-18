import unittest

from src.utils.ssl_non_vlm_refinement_history import (
    ClassConfidenceHistory,
)


class TestClassConfidenceHistory(unittest.TestCase):

    def test_current_batch_is_not_visible_before_commit(self):
        history = ClassConfidenceHistory(
            number_of_classes=2,
            maximum_scores_per_class=256,
        )

        history.stage_current_batch(
            {
                0: [0.81, 0.82],
            }
        )

        self.assertEqual(
            history.get_committed_scores(0),
            (),
        )

    def test_successful_commit_adds_scores(self):
        history = ClassConfidenceHistory(
            number_of_classes=2,
            maximum_scores_per_class=256,
        )

        history.stage_current_batch(
            {
                0: [0.81, 0.82],
                1: [0.91],
            }
        )

        history.commit_staged_scores()

        self.assertEqual(
            history.get_committed_scores(0),
            (0.81, 0.82),
        )

        self.assertEqual(
            history.get_committed_scores(1),
            (0.91,),
        )

    def test_discard_does_not_change_history(self):
        history = ClassConfidenceHistory(
            number_of_classes=2,
            maximum_scores_per_class=256,
        )

        history.stage_current_batch(
            {
                0: [0.81],
            }
        )

        history.discard_staged_scores()

        self.assertEqual(
            history.get_committed_scores(0),
            (),
        )

    def test_history_keeps_only_most_recent_scores(self):
        history = ClassConfidenceHistory(
            number_of_classes=1,
            maximum_scores_per_class=3,
        )

        history.stage_current_batch(
            {
                0: [0.70, 0.80],
            }
        )
        history.commit_staged_scores()

        history.stage_current_batch(
            {
                0: [0.90, 1.00],
            }
        )
        history.commit_staged_scores()

        self.assertEqual(
            history.get_committed_scores(0),
            (0.80, 0.90, 1.00),
        )

    def test_classes_have_independent_histories(self):
        history = ClassConfidenceHistory(
            number_of_classes=2,
            maximum_scores_per_class=256,
        )

        history.stage_current_batch(
            {
                0: [0.71],
                1: [0.93],
            }
        )
        history.commit_staged_scores()

        self.assertEqual(
            history.get_committed_scores(0),
            (0.71,),
        )

        self.assertEqual(
            history.get_committed_scores(1),
            (0.93,),
        )

    def test_second_stage_before_commit_is_rejected(self):
        history = ClassConfidenceHistory(
            number_of_classes=1,
            maximum_scores_per_class=256,
        )

        history.stage_current_batch(
            {
                0: [0.80],
            }
        )

        with self.assertRaises(RuntimeError):
            history.stage_current_batch(
                {
                    0: [0.90],
                }
            )

    def test_invalid_score_is_rejected(self):
        history = ClassConfidenceHistory(
            number_of_classes=1,
            maximum_scores_per_class=256,
        )

        with self.assertRaises(ValueError):
            history.stage_current_batch(
                {
                    0: [1.01],
                }
            )

    def test_committed_state_does_not_include_staged_scores(self):
        history = ClassConfidenceHistory(
            number_of_classes=1,
            maximum_scores_per_class=256,
        )

        history.stage_current_batch(
            {
                0: [0.85],
            }
        )

        state = history.committed_state()

        self.assertEqual(
            state["scores_by_class"]["0"],
            [],
        )

    def test_committed_state_can_be_restored(self):
        original = ClassConfidenceHistory(
            number_of_classes=2,
            maximum_scores_per_class=256,
        )

        original.stage_current_batch(
            {
                0: [0.75, 0.80],
                1: [0.95],
            }
        )
        original.commit_staged_scores()

        state = original.committed_state()

        restored = ClassConfidenceHistory(
            number_of_classes=2,
            maximum_scores_per_class=256,
        )

        restored.restore_committed_state(
            state
        )

        self.assertEqual(
            restored.get_committed_scores(0),
            (0.75, 0.80),
        )

        self.assertEqual(
            restored.get_committed_scores(1),
            (0.95,),
        )


if __name__ == "__main__":
    unittest.main()
